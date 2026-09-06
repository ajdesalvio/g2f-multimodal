"""Shared axis table: GDD / AGDD time axes derived from daily weather.

This module owns the **single source of truth** for the alternative time
axes (``gdd``, ``agdd``) that both the per-sample VI/DL path and the
env-level weather-FPCA path consume.

Two layers live here:

- :func:`build_axis_table` / :class:`AxisTable` — the env-level axis
  table. Built once from the weather CSV (fold-independent — AGDD is
  deterministic from weather). Exposes two lookups:
    * :meth:`AxisTable.lookup` — map a sample's VI observation DAPs onto
      an alternative axis (``gdd`` / ``agdd``). VI / sample path.
    * :meth:`AxisTable.env_axis` — per-env time vector + aligned weather
      values on the requested axis. Weather-FPCA path.

- :class:`AxisSourceProcessor` (priority ``-1``) — the in-place processor
  that attaches ``sample["gdd"]`` / ``sample["agdd"]`` via ``lookup``.

**What we import from the R reference (and nothing
else:** (1) ``AGDD = cumsum(GDD)`` per Env sorted by DAP — ``GDD`` is the
precomputed CSV column, not recomputed by the R reference; (2) the env-level dedup
→ clean-axis procedure (average all weather cols incl. GDD over rows
sharing an ``(Env, AGDD)`` value, ``first()`` for don't-cares, then
re-``cumsum(GDD)`` for a strictly-unique monotonic axis); (3) the
DAP→AGDD remap. Normalization, ``n_components`` and domain clipping stay
ours.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
import torch

from .base import BaseProcessor, OrchestratorContext
from .registry import register_processor
from .weather import _interpolate_weather, load_weather_csv

logger = logging.getLogger(__name__)

# Axes the table can produce / resolve. ``dap`` is the native reader axis
# (always available, no weather lookup needed); ``gdd`` / ``agdd`` are
# derived here.
AXIS_NAMES: tuple[str, ...] = ("dap", "gdd", "agdd")
DERIVED_AXES: tuple[str, ...] = ("gdd", "agdd")

# Default rounding used to detect "equal AGDD" rows during dedup. AGDD is
# a float cumsum; consecutive zero-GDD days produce values that are equal
# up to float noise, so we group on a rounded key.
_DEFAULT_DEDUP_DECIMALS = 6


# ── GDD resolution ──────────────────────────────────────────────────


def _default_gdd_cfg() -> dict:
    """The canonical GDD config — ``source: column`` reproduces the R reference
    exactly (trust the CSV's precomputed daily ``GDD``)."""
    return {
        "source": "column",
        "column": "GDD",
        "t_base": 10.0,
        "t_cap": None,
        "tmax_col": "T2M_MAX",
        "tmin_col": "T2M_MIN",
    }


def _resolve_gdd(
    values: np.ndarray,
    var_names: list[str],
    gdd_cfg: dict,
) -> np.ndarray:
    """Daily GDD vector ``(n_days,)`` for one env's weather matrix.

    ``source: column`` (default) reads the precomputed CSV ``GDD``
    column verbatim — bit-identical to the R reference. ``source: recompute``
    derives it from ``(T2M_MAX, T2M_MIN)`` with a configurable base /
    cap: ``GDD = max(0, mean(min(Tmax, cap), min(Tmin, cap)) - t_base)``.
    The recompute path is our optional addition and is documented as
    approximate (the CSV's exact formula/cap is external).
    """
    cfg = {**_default_gdd_cfg(), **(gdd_cfg or {})}
    source = cfg["source"]
    idx = {n: i for i, n in enumerate(var_names)}

    if source == "column":
        col = cfg["column"]
        if col not in idx:
            raise ValueError(
                f"axis_source: GDD column {col!r} not in weather CSV "
                f"(available: {var_names})."
            )
        return values[:, idx[col]].astype(np.float64)

    if source == "recompute":
        for key in ("tmax_col", "tmin_col"):
            if cfg[key] not in idx:
                raise ValueError(
                    f"axis_source: GDD recompute needs column {cfg[key]!r} "
                    f"(available: {var_names})."
                )
        tmax = values[:, idx[cfg["tmax_col"]]].astype(np.float64)
        tmin = values[:, idx[cfg["tmin_col"]]].astype(np.float64)
        t_cap = cfg["t_cap"]
        if t_cap is not None:
            tmax = np.minimum(tmax, float(t_cap))
            tmin = np.minimum(tmin, float(t_cap))
        gdd = (tmax + tmin) / 2.0 - float(cfg["t_base"])
        return np.clip(gdd, 0.0, None)

    raise ValueError(
        f"axis_source: gdd.source must be 'column' or 'recompute', "
        f"got {source!r}."
    )


# ── Dedup (the R reference's env-level clean-axis procedure) ──────────────


def _nanmean_cols(block: np.ndarray) -> np.ndarray:
    """Column means ignoring NaN; all-NaN columns yield NaN.

    Mirrors R's ``mean(x, na.rm = TRUE)``, which returns ``NaN`` when
    every value is ``NA``. ``np.nanmean`` agrees but warns on an
    all-NaN slice, so the warning is suppressed rather than surfaced as
    log noise on a legitimate, reference-matching case.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(block, axis=0)


def _dedup_axis(
    daps: np.ndarray,
    gdd: np.ndarray,
    values: np.ndarray,
    round_decimals: int = _DEFAULT_DEDUP_DECIMALS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse rows sharing an AGDD value, then re-``cumsum`` (the R reference V3).

    Rows with equal AGDD (necessarily a contiguous run, since AGDD is a
    monotonic non-decreasing cumsum — equal AGDD means zero GDD between
    them) are averaged across **all** weather columns (incl. GDD), with
    the **first** row's DAP kept for don't-care columns. The cleaned
    table is then re-``cumsum(GDD)`` for a strictly-unique monotonic
    axis.

    Returns ``(dedup_dap, dedup_agdd, dedup_values)`` where the lengths
    are the number of distinct AGDD groups (``<= n_days``).
    """
    agdd = np.cumsum(gdd)
    keys = np.round(agdd, round_decimals)
    # Group boundaries: a new group starts wherever the rounded AGDD key
    # changes from the previous row. keys is monotonic non-decreasing.
    starts = np.concatenate(([0], np.flatnonzero(np.diff(keys)) + 1))
    ends = np.concatenate((starts[1:], [len(keys)]))

    dedup_dap = np.empty(len(starts), dtype=daps.dtype)
    dedup_gdd = np.empty(len(starts), dtype=np.float64)
    dedup_values = np.empty((len(starts), values.shape[1]), dtype=values.dtype)
    for g, (s, e) in enumerate(zip(starts, ends)):
        dedup_dap[g] = daps[s]                       # first() for don't-cares
        dedup_gdd[g] = gdd[s:e].mean()               # average GDD
        # nanmean, not mean: the reference collapses with
        # `mean(.x, na.rm = TRUE)`, so a NaN in one row of a group must
        # not poison the group's average. An all-NaN group yields NaN
        # (matching R), which fdapace then drops downstream.
        dedup_values[g] = _nanmean_cols(values[s:e])

    dedup_agdd = np.cumsum(dedup_gdd)
    return dedup_dap, dedup_agdd, dedup_values


# ── Env-level axis record ────────────────────────────────────────────


@dataclass
class EnvAxis:
    """Per-env daily axis arrays + the deduped (clean-AGDD) view.

    ``dap`` / ``gdd`` / ``agdd`` / ``weather`` are the full daily grid
    (sorted ascending by DAP). ``dedup_*`` are the collapsed
    strictly-unique-AGDD view used by the weather-FPCA path; ``None``
    when the table was built with ``dedup=False``.
    """

    dap: np.ndarray            # (n_days,) sorted DAPs
    gdd: np.ndarray            # (n_days,) daily GDD
    agdd: np.ndarray           # (n_days,) cumsum(gdd) on the full grid
    weather: np.ndarray        # (n_days, n_vars)
    dedup_dap: np.ndarray | None = None      # (m,)
    dedup_agdd: np.ndarray | None = None     # (m,)
    dedup_weather: np.ndarray | None = None  # (m, n_vars)

    def daily_axis(self, axis: str) -> np.ndarray:
        """Full daily coordinate vector for ``axis`` (``dap``/``gdd``/``agdd``)."""
        if axis == "dap":
            return self.dap.astype(np.float64)
        if axis == "gdd":
            return self.gdd
        if axis == "agdd":
            return self.agdd
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXIS_NAMES}")


class AxisTable:
    """Env-level axis table — the shared source of truth.

    Build via :func:`build_axis_table`. Two consumers:

    - :meth:`lookup` — VI / sample path: map a sample's observed DAPs
      onto a derived axis.
    - :meth:`env_axis` — weather path: per-env time vector + aligned
      weather values on the requested axis (deduped for ``agdd``).
    """

    def __init__(
        self,
        env_axes: dict[str, EnvAxis],
        var_names: list[str],
        deduped: bool,
    ):
        self._env_axes = env_axes
        self.var_names = list(var_names)
        self.deduped = deduped

    def __contains__(self, env: str) -> bool:
        return env in self._env_axes

    def envs(self) -> list[str]:
        return sorted(self._env_axes)

    # ── VI / sample path ────────────────────────────────────────────

    def lookup(
        self,
        env: str,
        dap: np.ndarray,
        axis: str,
        dedup_scale: bool = False,
    ) -> np.ndarray:
        """Map sample VI observation DAPs onto ``axis``; returns ``(T,)``.

        ``axis == "dap"`` returns the input DAPs unchanged.

        For derived axes there are two AGDD conventions:

        - **full daily grid** (``dedup_scale=False``, default): each DAP is
          resolved against the env's full daily axis (every DAP has a
          well-defined raw-``cumsum`` AGDD; zero-GDD days carry the flat
          value, no NAs). Exact integer-DAP matches take a fast path;
          off-grid DAPs are linearly interpolated.
        - **deduped clean-AGDD scale** (``dedup_scale=True``, ``axis=="agdd"``
          only): the R reference's ``dap.gdd`` — each DAP maps to the **re-cumsummed**
          AGDD of its deduped representative; a DAP that dedup collapsed away
          (a non-first day of a zero-GDD run) returns **NaN**, so the VI-FPCA
          bridge drops it exactly as the R reference's ``left_join(dap.gdd)`` NA does.
          Requires a deduped table.
        """
        dap = np.asarray(dap, dtype=np.float64)
        if axis == "dap":
            return dap.copy()
        if axis not in DERIVED_AXES:
            raise ValueError(
                f"unknown axis {axis!r}; expected one of {AXIS_NAMES}"
            )
        if env not in self._env_axes:
            raise KeyError(f"axis_source: env {env!r} not in axis table.")

        ea = self._env_axes[env]

        if dedup_scale and axis == "agdd":
            if not self.deduped or ea.dedup_dap is None:
                raise RuntimeError(
                    "AxisTable.lookup(dedup_scale=True) needs a deduped "
                    "table — build_axis_table(..., dedup=True)."
                )
            # The R-reference dap.gdd: representative DAP -> re-cumsummed AGDD; DAPs
            # collapsed in dedup are absent -> NaN (dropped downstream).
            d2a = {
                int(round(d)): a
                for d, a in zip(ea.dedup_dap, ea.dedup_agdd)
            }
            return np.array(
                [d2a.get(int(round(d)), np.nan) for d in dap],
                dtype=np.float64,
            )

        grid_dap = ea.dap.astype(np.float64)
        grid_val = ea.daily_axis(axis)
        dap_to_idx = {int(round(d)): i for i, d in enumerate(grid_dap)}

        out = np.empty(len(dap), dtype=np.float64)
        for i, d in enumerate(dap):
            di = int(round(d))
            if di in dap_to_idx:
                out[i] = grid_val[dap_to_idx[di]]
            else:
                # Reuse the weather interpolator: it treats the axis
                # vector as a single "weather var" column.
                out[i] = _interpolate_weather(
                    d, grid_dap, grid_val[:, None]
                )[0]
        return out

    # ── Weather path ────────────────────────────────────────────────

    def daily_axis(self, env: str, axis: str) -> np.ndarray:
        """Full daily coordinate vector for ``env`` on ``axis``.

        Returns the env's native daily grid (sorted ascending by DAP, no
        dedup) for ``dap`` / ``gdd`` / ``agdd`` — ``agdd`` is the raw
        ``cumsum(GDD)`` over every daily row (zero-GDD days carry the flat
        value, no NAs). This is the un-deduped counterpart to
        :meth:`env_axis`; the DL raw-weather peer reads it to attach
        ``weather_gdd`` / ``weather_agdd`` columns aligned to the daily
        ``weather_dap`` grid (no dedup for DL).
        """
        if env not in self._env_axes:
            raise KeyError(f"axis_source: env {env!r} not in axis table.")
        return self._env_axes[env].daily_axis(axis)

    def env_axis(self, env: str, axis: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-env time vector + aligned weather values on ``axis``.

        ``axis == "dap"`` returns the full daily grid (matches today's
        weather path). ``axis == "agdd"`` returns the **deduped**
        clean-AGDD grid (strictly-unique monotonic time axis, required
        for fdapace) and the averaged weather values.

        Returns ``(coord(m,), values(m, n_vars))``.
        """
        if env not in self._env_axes:
            raise KeyError(f"axis_source: env {env!r} not in axis table.")
        ea = self._env_axes[env]
        if axis == "dap":
            return ea.dap.astype(np.float64), ea.weather
        if axis == "gdd":
            return ea.gdd, ea.weather
        if axis == "agdd":
            if ea.dedup_agdd is None or ea.dedup_weather is None:
                raise RuntimeError(
                    "axis_source: env_axis(axis='agdd') needs a deduped "
                    "table; build_axis_table(..., dedup=True)."
                )
            return ea.dedup_agdd, ea.dedup_weather
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXIS_NAMES}")


def build_axis_table(
    csv_path: str,
    gdd_cfg: dict | None = None,
    envs: list[str] | None = None,
    dedup: bool = True,
    missing_values: str = "drop",
    dtype: type = np.float64,
) -> AxisTable:
    """Build the shared env-level :class:`AxisTable` from the weather CSV.

    Reuses :func:`load_weather_csv` (per-env DAP sort) for the daily
    grid, derives ``GDD`` per ``gdd_cfg``, computes
    ``AGDD = cumsum(GDD)``, and — when ``dedup`` — builds the
    collapsed clean-AGDD view (the R reference's clean-axis procedure).

    ``envs=None`` loads every env in the CSV (AGDD is fold-independent).

    ``missing_values`` is forwarded to :func:`load_weather_csv`; it
    defaults to ``"drop"`` (NaN cells preserved, the reference
    behaviour). Consumers needing a gap-free daily series must pass
    ``"ffill"`` explicitly. ``GDD`` itself has no NaNs in the shipped
    CSV, so the derived AGDD axis is identical either way — only the
    weather VALUES carried alongside it differ.

    ``dtype`` is likewise forwarded; it defaults to ``np.float64`` (full
    precision for fdapace). Tensor-building callers pass ``np.float32``.
    The derived GDD/AGDD axes are computed in float64 regardless, so the
    axis itself is unaffected by this choice.
    """
    weather_data, var_names = load_weather_csv(
        csv_path, envs=envs, weather_vars=None,
        missing_values=missing_values, dtype=dtype,
    )
    env_axes: dict[str, EnvAxis] = {}
    for env, (daps, values) in weather_data.items():
        # load_weather_csv already sorts each env by DAP ascending.
        gdd = _resolve_gdd(values, var_names, gdd_cfg or {})
        agdd = np.cumsum(gdd)
        if dedup:
            d_dap, d_agdd, d_vals = _dedup_axis(daps, gdd, values)
        else:
            d_dap = d_agdd = d_vals = None
        env_axes[env] = EnvAxis(
            dap=daps,
            gdd=gdd,
            agdd=agdd,
            weather=values,
            dedup_dap=d_dap,
            dedup_agdd=d_agdd,
            dedup_weather=d_vals,
        )
    return AxisTable(env_axes, var_names, deduped=dedup)


# ── AxisSourceProcessor (priority -1, in-place axis attach) ──────────


@register_processor
class AxisSourceProcessor(BaseProcessor):
    """Attach derived time axes (``gdd`` / ``agdd``) to each sample.

    Runs FIRST (priority ``-1`` — before ``metadata_features`` at 0 and
    every consumer) so the alternative axes are present on the sample
    dict before any FPCA / View consumer reads them. In-place: writes
    ``sample["gdd"]`` / ``sample["agdd"]`` as ``[T, 1]`` tensors aligned
    to the sample's ``dap`` grid via :meth:`AxisTable.lookup`, then
    returns ``None`` (no ``derived_features``, no batch fields, no view).

    Caching is intentionally light: the axis table is deterministic and
    cheap to rebuild from the CSV (``AGDD = cumsum(GDD)``), so
    :meth:`save_cache` is a no-op and :meth:`load_fitted` simply rebuilds
    the table from ``csv_path`` (always available at predict time). The
    ``cache_key`` still participates in the orchestrator's bookkeeping so
    distinct CSV / GDD / dedup configs are recorded.

    Parameters
    ----------
    csv_path : path to the weather CSV (source of ``GDD``).
    gdd : GDD resolution config (``source`` / ``column`` / ``t_base`` /
        ``t_cap`` / ``tmax_col`` / ``tmin_col``); see :func:`_resolve_gdd`.
    produce : which derived axes to attach (subset of ``{"gdd", "agdd"}``).
    dedup : build the deduped clean-AGDD view (needed for the weather
        ``env_axis`` path; harmless for the per-sample lookup).
    vi_dedup : place per-sample ``agdd`` on the deduped clean-AGDD scale
        (the reference ``dap.gdd`` convention); collapsed DAPs get NaN.
    """

    name: ClassVar[str] = "axis_source"
    # Unique and < every consumer: metadata_features owns 0, vi_fpca 1,
    # and priority is an int ClassVar (no 0.5) — so -1.
    priority: ClassVar[int] = -1
    # Writes axes onto the sample dict, not derived_features / batch
    # fields / views.
    produces_batch_fields: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        csv_path: str,
        gdd: dict | None = None,
        produce: list[str] | None = None,
        dedup: bool = True,
        vi_dedup: bool = False,
    ):
        self.csv_path = csv_path
        self.gdd_cfg = dict(gdd) if gdd else {}
        produce = list(produce) if produce is not None else list(DERIVED_AXES)
        bad = [a for a in produce if a not in DERIVED_AXES]
        if bad:
            raise ValueError(
                f"axis_source.produce must be a subset of {DERIVED_AXES}, "
                f"got unsupported {bad}. ('dap' is the native reader axis "
                f"and is always present.)"
            )
        self.produce = produce
        self.dedup = bool(dedup)
        # vi_dedup: place per-sample `agdd` on the R reference's deduped clean-AGDD
        # scale (dap.gdd) and emit NaN for collapsed DAPs, so the VI-FPCA
        # bridge drops those obs exactly as the reference left_join NA does (bit-exact
        # AGDD env-GxE/weather-GxE). Default False keeps the full-grid raw-cumsum lookup
        # (no NaN — safe for DL agdd coords / agdd-as-channel views).
        self.vi_dedup = bool(vi_dedup)
        if self.vi_dedup and not self.dedup:
            raise ValueError(
                "axis_source.vi_dedup=true requires dedup=true (the deduped "
                "clean-AGDD table is what supplies the dap.gdd scale)."
            )
        self._table: AxisTable | None = None

    @classmethod
    def from_config(cls, config: dict) -> "AxisSourceProcessor":
        """Hydra-config factory used by the orchestrator."""
        return cls(
            csv_path=config["csv_path"],
            gdd=config.get("gdd"),
            produce=config.get("produce"),
            dedup=config.get("dedup", True),
            vi_dedup=config.get("vi_dedup", False),
        )

    # ── Coverage ─────────────────────────────────────────────────────

    @classmethod
    def coverage_mask(
        cls,
        metadata_df: "pd.DataFrame",
        proc_config: dict,
    ) -> np.ndarray | None:
        """Drop samples whose ``Env.Year`` is absent from the weather CSV.

        Cheap single-column read of the CSV's ``Env`` index. Returns
        ``None`` when disabled so the orchestrator skips the filter.
        """
        if not proc_config.get("enabled", False):
            return None
        csv_envs = set(
            pd.read_csv(proc_config["csv_path"], usecols=["Env"])["Env"]
            .astype(str)
        )
        env_year = (
            metadata_df["Env"].astype(str)
            + "."
            + metadata_df["Year"].astype(str)
        )
        return env_year.isin(csv_envs).to_numpy()

    # ── Lifecycle ────────────────────────────────────────────────────

    def fit(self, ctx: OrchestratorContext) -> None:
        """Build the (fold-independent) axis table from the CSV."""
        self._build_table()

    def _build_table(self) -> None:
        if self._table is None:
            # Only the axis COORDS are read from this table, and GDD has
            # no NaNs, so both modes give an identical axis. Pinned to the
            # historical "ffill" so DL behaviour is provably unchanged.
            self._table = build_axis_table(
                self.csv_path, gdd_cfg=self.gdd_cfg, dedup=self.dedup,
                missing_values="ffill", dtype=np.float32,
            )

    @property
    def table(self) -> AxisTable:
        """The built :class:`AxisTable` (after ``fit`` / ``load_fitted``)."""
        if self._table is None:
            raise RuntimeError(
                "AxisSourceProcessor: axis table not built — call fit() or "
                "load_fitted() first."
            )
        return self._table

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """In-place: attach each produced axis to every sample as ``[T,1]``."""
        split_data = ctx.split_data(split)
        if not split_data:
            return None
        self._build_table()
        envs = ctx.envs_for_split(split)
        for sample, env in zip(split_data, envs):
            dap = sample["dap"].numpy().squeeze(-1)
            for axis in self.produce:
                # vi_dedup applies only to agdd (the R reference's dap.gdd scale +
                # NaN for collapsed DAPs); gdd / the raw path are unaffected.
                vals = self.table.lookup(
                    env, dap, axis,
                    dedup_scale=(self.vi_dedup and axis == "agdd"),
                ).astype(np.float32)
                sample[axis] = torch.from_numpy(vals).unsqueeze(-1)
        return None

    # ── Caching (light: rebuild from CSV) ───────────────────────────

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Delegate to the context-independent key."""
        return self.compute_cache_key()

    def compute_cache_key(self) -> str:
        """Content-addressed over CSV abspath+mtime + gdd/produce/dedup."""
        abspath = os.path.abspath(self.csv_path)
        mtime = int(os.path.getmtime(abspath))
        h = hashlib.sha256()
        h.update(b"axis_source_v1")
        h.update(abspath.encode())
        h.update(str(mtime).encode())
        h.update(b"|gdd=")
        h.update(json.dumps(self.gdd_cfg, sort_keys=True).encode())
        h.update(b"|produce=")
        h.update(json.dumps(sorted(self.produce)).encode())
        h.update(f"|dedup={self.dedup}".encode())
        h.update(f"|vi_dedup={self.vi_dedup}".encode())
        return h.hexdigest()

    def load_fitted(self, cache_dir: str, key: str) -> None:
        """Rebuild the axis table from the CSV (deterministic + cheap)."""
        self._build_table()
        logger.info("axis_source: axis table rebuilt from %s", self.csv_path)
