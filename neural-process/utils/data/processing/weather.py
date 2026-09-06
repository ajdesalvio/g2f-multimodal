"""Weather data processing for three integration modes.

Mode 1 — ``WeatherConcatProcessor``: look up weather at VI DAPs, widen the VI
tensor in-place.  Stateless (no fitting).

Mode 2 — ``WeatherFPCAProcessor``: run FPCA on daily weather curves via R
fdapace, producing per-environment FPC score vectors.  Reuses
``baselines/fpca_compute.R`` by reformatting weather into the same tall CSV
schema.

Mode 3 — ``RawWeatherProcessor``: attach raw daily weather curves per sample
for a separate set encoder.  Normalizes using train-environment statistics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
import warnings
from typing import ClassVar

import numpy as np
import pandas as pd
import torch

from .base import BaseProcessor, OrchestratorContext
from .registry import register_processor

logger = logging.getLogger(__name__)

# Columns in the weather CSV that are metadata (not weather variables).
METADATA_COLS = {"DAP", "Env", "YYYYMMDD"}

# Path to the shared R FPCA script.
_R_SCRIPT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        "baselines",
        "fpca_compute.R",
    )
)


# ── Shared CSV loading ───────────────────────────────────────────────────


# How ``load_weather_csv`` treats cells that are NaN in the CSV.
WEATHER_MISSING_MODES: tuple[str, ...] = ("drop", "ffill")


def load_weather_csv(
    csv_path: str,
    envs: list[str] | None = None,
    weather_vars: list[str] | None = None,
    missing_values: str = "drop",
    dtype: type = np.float64,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Load and parse the weather CSV.

    Parameters
    ----------
    csv_path : path to ``EnvRtype_Weather_Data_Cleaned_V2.csv``.
    envs : environments to keep.  ``None`` keeps all.
    weather_vars : weather variable column names to keep.  ``None`` keeps
        all non-metadata columns.
    missing_values : {"drop", "ffill"}
        How to treat cells that are NaN in the CSV.

        - ``"drop"`` (default, and the reference behaviour): leave the
          NaN in place.  fdapace removes NA (y, t) pairs per curve
          (``FPCA`` -> ``HandleNumericsAndNAN``; the projection path
          filters on ``is.finite(y)``), so a missing day becomes a gap
          in that curve — exactly what the R reference does with its
          ``values_fill = NA_real_`` wide matrix.
        - ``"ffill"``: forward-fill then backward-fill within each
          environment, yielding a dense grid with no NaN.  Required by
          consumers that need a gap-free daily series (the DL raw-weather
          encoder); NOT the reference, because it feeds fdapace
          fabricated observations.

        Only ``PTR`` and ``PAR_TEMP`` carry NaNs in the shipped CSV, so
        the two modes agree on every other variable.
    dtype : numpy dtype for the value array. Defaults to ``np.float64``,
        matching R's ``double`` — the FPCA path must not lose precision,
        because the values are handed to fdapace via a text CSV where a
        narrower dtype saves neither memory nor disk, only significant
        digits (float32 keeps ~7, float64 ~16; measured max relative
        error on PTR: 5.9e-08). Tensor-building consumers pass
        ``np.float32`` explicitly, which is what torch wants anyway.

    Returns
    -------
    weather_data : ``{env_name: (daps, values)}`` where ``daps`` is
        ``(n_days,)`` int and ``values`` is ``(n_days, n_vars)`` of
        ``dtype``.
    var_names : ordered list of weather variable column names used.
    """
    df = pd.read_csv(csv_path)

    # Filter environments
    if envs is not None:
        unknown = set(envs) - set(df["Env"].unique())
        if unknown:
            raise ValueError(
                f"Environments not found in weather CSV: {sorted(unknown)}"
            )
        df = df[df["Env"].isin(envs)]

    # Determine weather variable columns
    all_var_cols = [c for c in df.columns if c not in METADATA_COLS]
    if weather_vars is not None:
        unknown = set(weather_vars) - set(all_var_cols)
        if unknown:
            raise ValueError(
                f"Weather variables not found in CSV: {sorted(unknown)}"
            )
        var_names = [c for c in all_var_cols if c in set(weather_vars)]
    else:
        var_names = all_var_cols

    # NaN handling — see the ``missing_values`` docstring.
    if missing_values not in WEATHER_MISSING_MODES:
        raise ValueError(
            f"missing_values must be one of {WEATHER_MISSING_MODES}, "
            f"got {missing_values!r}."
        )
    df = df.sort_values(["Env", "DAP"])
    if missing_values == "ffill":
        df[var_names] = df.groupby("Env")[var_names].transform(
            lambda s: s.ffill().bfill()
        )
        remaining_nans = df[var_names].isnull().sum().sum()
        if remaining_nans > 0:
            warnings.warn(
                f"{remaining_nans} NaN values remain after forward/backward fill.",
                stacklevel=2,
            )

    # Build per-environment arrays
    weather_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for env_name, grp in df.groupby("Env"):
        grp = grp.sort_values("DAP")
        daps = grp["DAP"].values.astype(int)
        values = grp[var_names].values.astype(dtype)
        weather_data[str(env_name)] = (daps, values)

    return weather_data, var_names


# ── WeatherConcatProcessor (Mode 1) ───────────────────────────


@register_processor
class WeatherConcatProcessor(BaseProcessor):
    """Look up weather at VI DAPs and widen ``channels``.

    Stateless — no fitting step.  For each sample, indexes into the
    environment's daily weather at the sample's VI observation DAPs and
    concatenates the result to ``channels`` (in-place).

    Parameters
    ----------
    weather_csv_path : path to weather CSV.
    weather_vars : subset of weather variable names, or ``None`` for all.
    """

    # ── BaseProcessor metadata ──────────────────────────────────────
    name: ClassVar[str] = "weather_concat"
    # Priority 9: must run AFTER raw_vi (priority 8), which slices the
    # native VI columns before this processor appends its weather tail.
    priority: ClassVar[int] = 9
    # No new batch-field prefixes — widens the existing vi_y tensor.
    produces_batch_fields: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        weather_csv_path: str,
        weather_vars: list[str] | None = None,
    ):
        self.weather_csv_path = weather_csv_path
        self.weather_vars = list(weather_vars) if weather_vars is not None else None
        self._var_names: list[str] | None = None
        self._weather_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = (
            None
        )

    @classmethod
    def from_config(cls, config: dict) -> "WeatherConcatProcessor":
        """Hydra-config factory used by the orchestrator."""
        return cls(
            weather_csv_path=config["csv_path"],
            weather_vars=config.get("weather_vars"),
        )

    def load_weather(self, envs: list[str] | None = None) -> None:
        """Load and cache weather data from CSV."""
        if self._weather_data is None:
            self._weather_data, self._var_names = load_weather_csv(
                self.weather_csv_path,
                envs=envs,
                weather_vars=self.weather_vars,
                # Dense daily grid: this consumer feeds tensors, not
                # fdapace, so a NaN would propagate into the model. NOT
                # the FPCA reference behaviour — see load_weather_csv.
                missing_values="ffill",
                # float32 as before: these become torch tensors, and the
                # precision argument that applies to the FPCA path does
                # not apply here. Keeps this path bit-identical.
                dtype=np.float32,
            )

    def process(
        self,
        split_data: list[dict[str, torch.Tensor]],
        split_envs: list[str],
    ) -> None:
        """Widen ``channels`` in-place.

        Parameters
        ----------
        split_data : list of sample dicts (modified in-place).
        split_envs : per-sample environment names (same length as
            ``split_data``).
        """
        if not split_data:
            return

        self.load_weather(envs=None)

        interpolation_warnings: list[tuple[str, int]] = []

        for sample, env in zip(split_data, split_envs):
            if env not in self._weather_data:
                raise ValueError(
                    f"Environment {env!r} not found in weather data."
                )

            w_daps, w_values = self._weather_data[env]
            vi_daps = sample["dap"].numpy().squeeze(-1)

            # Build a DAP → row index mapping for fast lookup
            dap_to_idx = {int(d): i for i, d in enumerate(w_daps)}

            weather_rows = []
            for vi_dap in vi_daps:
                vi_dap_int = int(round(vi_dap))
                if vi_dap_int in dap_to_idx:
                    weather_rows.append(w_values[dap_to_idx[vi_dap_int]])
                else:
                    # Linear interpolation fallback
                    row = _interpolate_weather(
                        vi_dap, w_daps, w_values
                    )
                    weather_rows.append(row)
                    interpolation_warnings.append((env, vi_dap_int))

            weather_at_daps = np.stack(weather_rows).astype(np.float32)
            weather_tensor = torch.from_numpy(weather_at_daps)

            # Widen: (T_i, 37) → (T_i, 37 + W)
            sample["channels"] = torch.cat(
                [sample["channels"], weather_tensor], dim=-1
            )
            # Keep channel_names consistent with the widened tensor.
            if "channel_names" in sample:
                sample["channel_names"] = (
                    list(sample["channel_names"]) + list(self._var_names or [])
                )

        if interpolation_warnings:
            pairs = interpolation_warnings[:10]
            msg = ", ".join(f"({e}, DAP={d})" for e, d in pairs)
            extra = (
                f" (+{len(interpolation_warnings) - 10} more)"
                if len(interpolation_warnings) > 10
                else ""
            )
            warnings.warn(
                f"Weather interpolation used for {len(interpolation_warnings)} "
                f"(env, DAP) pairs: {msg}{extra}",
                stacklevel=2,
            )

    @property
    def n_weather_vars(self) -> int:
        """Number of weather variables used."""
        if self._var_names is None:
            raise RuntimeError(
                "Weather data not loaded yet. Call load_weather() or "
                "process() first."
            )
        return len(self._var_names)

    @property
    def var_names(self) -> list[str]:
        """Ordered list of weather variable names."""
        if self._var_names is None:
            raise RuntimeError(
                "Weather data not loaded yet. Call load_weather() or "
                "process() first."
            )
        return list(self._var_names)

    # ── BaseProcessor lifecycle ────────────────────────────────────

    def fit(self, ctx: OrchestratorContext) -> None:
        """Stateless — just preload the weather CSV so transform is fast."""
        self.load_weather(envs=None)

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """In-place: widen each sample's `channels` and
        return `None` (signal to the orchestrator to skip the attach).
        """
        split_data = ctx.split_data(split)
        if not split_data:
            return None
        envs = ctx.envs_for_split(split)
        self.process(split_data, envs)
        return None

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Identical to `compute_cache_key()` — context-independent
        because the processor is stateless w.r.t. dataset content
        (only the CSV mtime + var subset matter).
        """
        return self.compute_cache_key()

    # ── Cache: save / load fitted state ──────────────────────────

    def compute_cache_key(self) -> str:
        """Content-addressed cache key.

        Based on the weather CSV abspath + mtime and the configured
        ``weather_vars`` subset. Does not need per-sample state because
        :class:`WeatherConcatProcessor` is stateless.
        """
        abspath = os.path.abspath(self.weather_csv_path)
        mtime = int(os.path.getmtime(abspath))
        h = hashlib.sha256()
        h.update(b"weather_concat_v1")
        h.update(abspath.encode())
        h.update(str(mtime).encode())
        h.update(b"|vars=")
        h.update(json.dumps(self.weather_vars, sort_keys=True).encode())
        return h.hexdigest()

    def save_cache(self, cache_dir: str, cache_key: str) -> None:
        """Persist per-env weather arrays to
        ``cache_dir/weather_concat_<key[:16]>.npz``.

        Must be called after at least one :meth:`process` /
        :meth:`load_weather` so ``self._weather_data`` is populated.
        """
        if self._weather_data is None:
            raise RuntimeError(
                "Cannot save_cache before load_weather() / process()."
            )
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"weather_concat_{cache_key[:16]}.npz")

        envs = sorted(self._weather_data.keys())
        payload: dict[str, np.ndarray] = {
            "env_names": np.array(envs, dtype=object),
            "var_names": np.array(self._var_names, dtype=object),
        }
        for env in envs:
            daps, values = self._weather_data[env]
            payload[f"{env}__daps"] = daps
            payload[f"{env}__values"] = values

        tmp_path = path + f".tmp.{os.getpid()}.npz"
        np.savez(tmp_path, **payload)
        try:
            os.replace(tmp_path, path)
        except FileNotFoundError:
            pass
        logger.info("Weather-concat cache saved: %s", path)

    def load_fitted(self, cache_dir: str, cache_key: str) -> None:
        """Restore per-env weather arrays from cache.

        After this call, :meth:`process` skips the CSV read entirely —
        the in-memory ``_weather_data`` is populated directly from the
        ``.npz`` written by :meth:`save_cache`.
        """
        path = os.path.join(cache_dir, f"weather_concat_{cache_key[:16]}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Weather-concat cache not found: {path}"
            )
        data = np.load(path, allow_pickle=True)
        envs = [str(e) for e in data["env_names"]]
        self._var_names = [str(v) for v in data["var_names"]]
        self._weather_data = {
            env: (data[f"{env}__daps"], data[f"{env}__values"])
            for env in envs
        }
        logger.info("Weather-concat cache loaded: %s", path)


def _interpolate_weather(
    target_dap: float,
    daps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate weather values at a single DAP.

    Parameters
    ----------
    target_dap : the DAP to interpolate at.
    daps : sorted array of available DAPs, shape ``(n_days,)``.
    values : weather values, shape ``(n_days, n_vars)``.

    Returns
    -------
    Interpolated row, shape ``(n_vars,)``.
    """
    if target_dap <= daps[0]:
        return values[0].copy()
    if target_dap >= daps[-1]:
        return values[-1].copy()

    idx_right = np.searchsorted(daps, target_dap)
    idx_left = idx_right - 1
    frac = (target_dap - daps[idx_left]) / (daps[idx_right] - daps[idx_left])
    return (
        values[idx_left] * (1 - frac) + values[idx_right] * frac
    ).astype(np.float32)


# ── RawWeatherProcessor (Mode 3) ─────────────────────────────────────────


@register_processor
class RawWeatherProcessor(BaseProcessor):
    """Attach raw daily weather curves for a separate set encoder.

    Two independent scaling knobs control the daily DAP axis and the
    weather-value axis; each accepts:

    - ``"global_zscore"``: mean/std pooled over training-environment
      rows (per-column for values, scalar for DAPs), applied to every
      env. Analog of the dataset-level ``vi_x`` / ``vi_y`` z-score for
      the weather peer. Slug code: ``gz``.
    - ``"none"``: pass raw values through unchanged.

    ``within_env_zscore`` is rejected on both axes — daily rows belong
    to one env each, so per-env stats would be defined but the
    resulting curves would all be centered on their own env's mean,
    erasing the between-env variation that's the whole point of
    conditioning the model on weather.

    Parameters
    ----------
    weather_csv_path : path to weather CSV.
    weather_vars : subset of weather variable names, or ``None`` for all.
    dap_scaling : ``"global_zscore"`` (default) or ``"none"`` — controls
        the ``weather_dap`` axis.
    value_scaling : ``"global_zscore"`` (default) or ``"none"`` —
        controls the ``weather_values`` axis.
    axis / dedup / gdd : time-axis selection (``dap`` | ``gdd`` | ``agdd``)
        plus the AGDD-grid dedup flag and GDD resolution config.
    extra_axes / extra_axes_scaling : additional per-sample axis columns
        attached alongside the primary axis, with their scaling policy.
    """

    _ALLOWED_SCALINGS = ("none", "global_zscore")
    _ALLOWED_AXES = ("dap", "agdd")
    _ALLOWED_EXTRA_AXES = ("gdd", "agdd")

    # ── BaseProcessor metadata ──────────────────────────────────────
    name: ClassVar[str] = "raw_weather"
    priority: ClassVar[int] = 7
    # Attaches per-sample `weather_dap` / `weather_values`, which the collate
    # path assembles into the `"weather"` view (G2FBatch.views["weather"]).
    produces_views: ClassVar[tuple[str, ...]] = ("weather",)

    def __init__(
        self,
        weather_csv_path: str,
        weather_vars: list[str] | None = None,
        dap_scaling: str = "global_zscore",
        value_scaling: str = "global_zscore",
        axis: str = "dap",
        dedup: bool = True,
        gdd: dict | None = None,
        extra_axes: list[str] | None = None,
        extra_axes_scaling: str = "global_zscore",
    ):
        for nm, value in (("dap_scaling", dap_scaling),
                          ("value_scaling", value_scaling),
                          ("extra_axes_scaling", extra_axes_scaling)):
            if value not in self._ALLOWED_SCALINGS:
                raise ValueError(
                    f"{nm} must be one of {self._ALLOWED_SCALINGS} "
                    f"(within_env_zscore is rejected — would erase "
                    f"between-env variation), got {value!r}."
                )
        if axis not in self._ALLOWED_AXES:
            raise ValueError(
                f"raw_weather.axis must be one of {self._ALLOWED_AXES}, "
                f"got {axis!r}."
            )
        extra = list(extra_axes) if extra_axes else []
        bad_extra = [a for a in extra if a not in self._ALLOWED_EXTRA_AXES]
        if bad_extra:
            raise ValueError(
                f"raw_weather.extra_axes must be a subset of "
                f"{self._ALLOWED_EXTRA_AXES}, got unsupported {bad_extra}."
            )
        if len(set(extra)) != len(extra):
            raise ValueError(
                f"raw_weather.extra_axes has duplicates: {extra}."
            )
        if extra and axis != "dap":
            raise ValueError(
                f"raw_weather.extra_axes={extra} is only supported on the "
                f"native daily grid (axis='dap'); got axis={axis!r}. The "
                f"agdd grid is deduped and already carries AGDD as its "
                f"coordinate, so extra gdd/agdd channels are redundant there."
            )
        self.weather_csv_path = weather_csv_path
        self.weather_vars = list(weather_vars) if weather_vars is not None else None
        self.dap_scaling = dap_scaling
        self.value_scaling = value_scaling
        # Weather set-encoder time axis: 'dap' (native daily grid) or 'agdd'
        # (the deduped clean-AGDD grid from the shared axis table).
        # The weather curve's `weather_dap` coordinate carries AGDD when
        # axis='agdd' — mirroring weather_fpca's 2d substitution. `gdd`
        # configures AGDD construction (CSV GDD column = reference-exact);
        # `dedup` builds the unique monotonic AGDD grid.
        self.axis = axis
        self.dedup = bool(dedup)
        self.gdd_cfg = dict(gdd) if gdd else {}
        # Extra gdd/agdd time axes attached as separate per-sample columns on
        # the native daily weather grid (un-deduped raw cumsum), so the
        # config-driven weather View can take them as extra channels (or
        # coords) — the DL "weather encoder sees gdd/agdd" path. Normalized
        # with `extra_axes_scaling` from train-env stats only (no leakage).
        self.extra_axes = extra
        self.extra_axes_scaling = extra_axes_scaling
        self._var_names: list[str] | None = None
        self._weather_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = (
            None
        )
        # Per-env daily gdd/agdd vectors aligned to the native weather grid
        # (env -> {axis -> (n_days,)}). Populated by `_load_axis_daily` when
        # `extra_axes` is non-empty.
        self._axis_daily: dict[str, dict[str, np.ndarray]] | None = None
        self._norm_stats: dict[str, np.ndarray | float] | None = None

    @classmethod
    def from_config(cls, config: dict) -> "RawWeatherProcessor":
        """Hydra-config factory used by the orchestrator."""
        return cls(
            weather_csv_path=config["csv_path"],
            weather_vars=config.get("weather_vars"),
            dap_scaling=config["dap_scaling"],
            value_scaling=config["value_scaling"],
            axis=config.get("axis", "dap"),
            dedup=config.get("dedup", True),
            gdd=config.get("gdd"),
            extra_axes=config.get("extra_axes"),
            extra_axes_scaling=config.get("extra_axes_scaling", "global_zscore"),
        )

    def _load_weather(self, envs: list[str] | None = None) -> None:
        """Load weather data if not already loaded.

        On ``axis='agdd'`` the daily ``(DAP, values)`` per env is replaced by
        the deduped ``(AGDD, averaged-values)`` grid from the shared axis
        table, so the whole fit/transform/cache path below operates on the
        AGDD curve unchanged (only ``weather_dap`` now carries AGDD).
        """
        if self._weather_data is None:
            self._weather_data, self._var_names = load_weather_csv(
                self.weather_csv_path,
                envs=envs,
                weather_vars=self.weather_vars,
                # Dense daily grid: this consumer feeds tensors, not
                # fdapace, so a NaN would propagate into the model. NOT
                # the FPCA reference behaviour — see load_weather_csv.
                missing_values="ffill",
                # float32 as before: these become torch tensors, and the
                # precision argument that applies to the FPCA path does
                # not apply here. Keeps this path bit-identical.
                dtype=np.float32,
            )
            if self.axis == "agdd":
                self._weather_data = self._agdd_weather_data(
                    list(self._weather_data.keys())
                )
            if self.extra_axes:
                # axis=='dap' guaranteed by __init__ when extra_axes is set,
                # so `_weather_data` here is the native daily grid that the
                # daily gdd/agdd vectors align to row-for-row.
                self._load_axis_daily(list(self._weather_data.keys()))

    def _load_axis_daily(self, envs: list[str]) -> None:
        """Attach per-env daily gdd/agdd vectors on the native weather grid.

        Reuses the shared axis table (``dedup=False`` — DL needs the full
        daily grid, no collapse) to read each env's raw ``cumsum(GDD)`` axis
        aligned to the same daily DAPs ``load_weather_csv`` produced for
        ``weather_dap`` / ``weather_values``. Imported lazily to avoid the
        ``weather`` ↔ ``axis_source`` module cycle.
        """
        from .axis_source import build_axis_table

        table = build_axis_table(
            self.weather_csv_path,
            gdd_cfg=self.gdd_cfg,
            envs=envs,
            dedup=False,
            # Dense grid for the DL encoder — see load_weather_csv.
            missing_values="ffill",
            dtype=np.float32,
        )
        self._axis_daily = {
            env: {
                ax: table.daily_axis(env, ax).astype(np.float64)
                for ax in self.extra_axes
            }
            for env in envs
        }

    def _agdd_weather_data(
        self, envs: list[str],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per-env ``(agdd_coord, values)`` on the deduped clean-AGDD grid,
        with value columns sliced/reordered to this processor's
        ``weather_vars`` subset (``self._var_names``).

        ``build_axis_table`` loads every weather column (it needs ``GDD`` for
        the cumsum even if the subset excludes it), so we map the subset names
        onto the table's full column order. Imported lazily to avoid the
        ``weather`` ↔ ``axis_source`` module cycle.
        """
        from .axis_source import build_axis_table

        table = build_axis_table(
            self.weather_csv_path,
            gdd_cfg=self.gdd_cfg,
            envs=envs,
            dedup=self.dedup,
            # Dense grid for the DL encoder — see load_weather_csv.
            missing_values="ffill",
            dtype=np.float32,
        )
        col_idx = [table.var_names.index(n) for n in (self._var_names or [])]
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for env in envs:
            coord, values = table.env_axis(env, "agdd")
            out[env] = (
                coord.astype(np.float32),
                values[:, col_idx].astype(np.float32),
            )
        return out

    def fit_impl(self, train_envs: list[str]) -> None:
        """Compute normalization statistics from training environments.

        Stats are computed only for the axes whose scaling is
        ``"global_zscore"``; ``"none"`` axes contribute no entries.
        When both axes are ``"none"``, ``_norm_stats`` is set to an
        empty dict to signal "fit was called" so downstream
        ``transform()`` can distinguish that from "fit was skipped".

        Parameters
        ----------
        train_envs : environment names in the training split.
        """
        self._load_weather(envs=None)  # load all envs (need them for transform)

        # Validate train envs even when not normalizing — fail at fit
        # time rather than at the first transform.
        for env in train_envs:
            if env not in self._weather_data:
                raise ValueError(
                    f"Training environment {env!r} not found in weather data."
                )

        # Pool all training-environment rows for whichever axis needs stats.
        # An empty dict still signals "fit was called" (vs. skipped) to
        # transform(); when nothing needs z-scoring it simply stays empty.
        stats: dict[str, np.ndarray | float] = {}

        # Extra gdd/agdd axes: pool the per-env daily vectors over the
        # TRAINING envs only (leakage-safe — same pooling as dap/value below),
        # one scalar mean/std per axis applied to every env at transform time.
        if self.extra_axes and self.extra_axes_scaling == "global_zscore":
            if self._axis_daily is None:
                raise RuntimeError(
                    "extra_axes requested but daily axis vectors not loaded; "
                    "_load_weather() must run before fit_impl()."
                )
            for ax in self.extra_axes:
                pooled = np.concatenate(
                    [self._axis_daily[env][ax] for env in train_envs]
                )
                ax_mean = float(pooled.mean())
                # ddof=1 keeps z-scoring consistent across the data pipeline.
                ax_std = float(pooled.std(ddof=1))
                if not np.isfinite(ax_std) or ax_std == 0:
                    ax_std = 1.0
                stats[f"{ax}_mean"] = ax_mean
                stats[f"{ax}_std"] = ax_std

        if self.dap_scaling == "global_zscore":
            all_daps = np.concatenate(
                [
                    self._weather_data[env][0].astype(np.float64)
                    for env in train_envs
                ]
            )
            dap_mean = float(all_daps.mean())
            # ddof=1 keeps z-scoring consistent across the data pipeline.
            dap_std = float(all_daps.std(ddof=1))
            if not np.isfinite(dap_std) or dap_std == 0:
                dap_std = 1.0
            stats["dap_mean"] = dap_mean
            stats["dap_std"] = dap_std

        if self.value_scaling == "global_zscore":
            all_vals = np.concatenate(
                [
                    self._weather_data[env][1].astype(np.float64)
                    for env in train_envs
                ],
                axis=0,
            )
            val_mean = all_vals.mean(axis=0).astype(np.float32)
            # ddof=1 keeps z-scoring consistent across the data pipeline.
            val_std = all_vals.std(axis=0, ddof=1).astype(np.float32)
            # Avoid division by zero for constant columns or single-row
            # pools (n=1 → ddof=1 yields nan, clamped to 1.0).
            val_std[~np.isfinite(val_std) | (val_std == 0)] = 1.0
            stats["val_mean"] = val_mean
            stats["val_std"] = val_std

        self._norm_stats = stats

    def transform_impl(
        self,
        split_data: list[dict[str, torch.Tensor]],
        split_envs: list[str],
    ) -> None:
        """Attach normalized weather curves to sample dicts in-place.

        Adds ``weather_dap`` (shape ``(T_w, 1)``) and ``weather_values``
        (shape ``(T_w, W)``) to each sample dict.

        Parameters
        ----------
        split_data : list of sample dicts (modified in-place).
        split_envs : per-sample environment names.
        """
        if not split_data:
            return

        if self._norm_stats is None:
            raise RuntimeError("Must call fit_impl() before transform_impl().")

        apply_dap = self.dap_scaling != "none"
        apply_val = self.value_scaling != "none"
        if apply_dap:
            dap_mean = self._norm_stats["dap_mean"]
            dap_std = self._norm_stats["dap_std"]
        if apply_val:
            val_mean = self._norm_stats["val_mean"]
            val_std = self._norm_stats["val_std"]

        for sample, env in zip(split_data, split_envs):
            if env not in self._weather_data:
                raise ValueError(
                    f"Environment {env!r} not found in weather data."
                )

            daps, values = self._weather_data[env]

            if apply_dap:
                daps_proc = (
                    (daps.astype(np.float32) - dap_mean) / dap_std
                ).reshape(-1, 1)
            else:
                daps_proc = daps.astype(np.float32).reshape(-1, 1)

            if apply_val:
                vals_proc = (values - val_mean) / val_std
            else:
                vals_proc = values

            sample["weather_dap"] = torch.from_numpy(
                daps_proc.astype(np.float32)
            )
            sample["weather_values"] = torch.from_numpy(
                vals_proc.astype(np.float32)
            )
            # Names for the weather_values columns so the config-driven
            # weather View (coords:[weather_dap] channels:[weather.*]) can
            # resolve/slice them via assemble_view. Mirrors the VI
            # `channel_names` the reader attaches.
            sample["weather_channel_names"] = list(self._var_names or [])

            # Extra gdd/agdd time axes on the same daily grid (un-deduped),
            # z-scored with train-only stats, attached as `weather_gdd` /
            # `weather_agdd` [T_w, 1] so the weather View can take them as
            # extra channels (channels:[weather.*, weather_gdd, weather_agdd])
            # or coords. Aligned row-for-row with `weather_dap` (both daily).
            if self.extra_axes:
                if self._axis_daily is None:
                    raise RuntimeError(
                        "extra_axes requested but daily axis vectors not "
                        "loaded; _load_weather() must run before "
                        "transform_impl()."
                    )
                n_w = vals_proc.shape[0]
                for ax in self.extra_axes:
                    ax_daily = self._axis_daily[env][ax].astype(np.float64)
                    if ax_daily.shape[0] != n_w:
                        raise ValueError(
                            f"raw_weather.extra_axes: {ax!r} grid length "
                            f"{ax_daily.shape[0]} != weather grid {n_w} for "
                            f"env {env!r} — daily axis/weather grids diverged."
                        )
                    if self.extra_axes_scaling == "global_zscore":
                        ax_daily = (
                            ax_daily - self._norm_stats[f"{ax}_mean"]
                        ) / self._norm_stats[f"{ax}_std"]
                    sample[f"weather_{ax}"] = torch.from_numpy(
                        ax_daily.reshape(-1, 1).astype(np.float32)
                    )

    @property
    def norm_stats(self) -> dict[str, np.ndarray | float]:
        """Normalization statistics computed during ``fit_impl()``."""
        if self._norm_stats is None:
            raise RuntimeError(
                "Must call fit_impl() before accessing norm_stats."
            )
        return dict(self._norm_stats)

    @property
    def var_names(self) -> list[str]:
        """Ordered list of weather variable names."""
        if self._var_names is None:
            raise RuntimeError("Weather data not loaded yet.")
        return list(self._var_names)

    @property
    def n_weather_vars(self) -> int:
        """Number of weather variables (= weather-curve value width W).

        Equals the assembled ``weather`` view's channel width ``C``, which
        ``setup._inject_view_dims`` writes into ``params.weather_y_dim``. Mirrors ``WeatherConcatProcessor.n_weather_vars``.
        """
        return len(self.var_names)

    # ── BaseProcessor lifecycle ────────────────────────────────────

    def fit(self, ctx: OrchestratorContext) -> None:
        """Pull train envs from `ctx`, delegate to `fit_impl`."""
        train_envs = ctx.unique_envs_for_split("train")
        self.fit_impl(train_envs)

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """In-place: attach `weather_dap` / `weather_values` to each
        sample. Returns `None` since we mutated samples directly.
        """
        split_data = ctx.split_data(split)
        if not split_data:
            return None
        envs = ctx.envs_for_split(split)
        self.transform_impl(split_data, envs)
        return None

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Delegate to `compute_cache_key`, pulling train envs from ctx."""
        train_envs = ctx.unique_envs_for_split("train")
        return self.compute_cache_key(train_envs)

    # ── Cache: save / load fitted state ──────────────────────────

    def compute_cache_key(self, train_envs: list[str]) -> str:
        """Content-addressed cache key for the fitted normalization stats.

        Derived from the weather CSV abspath+mtime, the configured
        ``weather_vars`` subset, the per-axis scaling choices, and the
        sorted unique training-env list (norm stats depend on which
        envs were pooled for fit; under ``"none"`` axes the train_envs
        don't affect output but are kept in the key for shape symmetry
        across scaling combinations).
        """
        abspath = os.path.abspath(self.weather_csv_path)
        mtime = int(os.path.getmtime(abspath))
        h = hashlib.sha256()
        h.update(b"raw_weather_v3")
        h.update(abspath.encode())
        h.update(str(mtime).encode())
        h.update(b"|vars=")
        h.update(json.dumps(self.weather_vars, sort_keys=True).encode())
        h.update(b"|dap_scaling=")
        h.update(self.dap_scaling.encode())
        h.update(b"|value_scaling=")
        h.update(self.value_scaling.encode())
        h.update(b"|train_envs=")
        h.update("|".join(sorted(set(train_envs))).encode())
        # Always tag the axis name so every entry is self-describing on disk
        # (DAP included). dedup/gdd only shape the AGDD grid, so fold those in
        # only for the derived axis. The substituted AGDD arrays also alter the
        # persisted .npz content; the tag keeps DAP vs AGDD provenance legible.
        h.update(b"|axis=")
        h.update(self.axis.encode())
        if self.axis != "dap":
            h.update(f"|dedup={self.dedup}".encode())
            h.update(b"|gdd=")
            h.update(json.dumps(self.gdd_cfg, sort_keys=True).encode())
        # Extra gdd/agdd axes alter the persisted .npz (extra stats +
        # per-sample columns), so tag the key when present. Only tagged when
        # non-empty, so the default DAP key stays byte-identical to existing
        # caches. The gdd config feeds the cumsum even on the dap path.
        if self.extra_axes:
            h.update(b"|extra_axes=")
            h.update(json.dumps(sorted(self.extra_axes)).encode())
            h.update(b"|extra_axes_scaling=")
            h.update(self.extra_axes_scaling.encode())
            if self.axis == "dap":  # gdd cfg not already folded in above
                h.update(b"|extra_gdd=")
                h.update(json.dumps(self.gdd_cfg, sort_keys=True).encode())
        return h.hexdigest()

    def save_cache(self, cache_dir: str, cache_key: str) -> None:
        """Persist per-env weather arrays + norm stats to
        ``cache_dir/raw_weather_<key[:16]>.npz``.

        Must be called after :meth:`fit` so norm stats are populated.
        """
        if self._norm_stats is None or self._weather_data is None:
            raise RuntimeError(
                "Cannot save_cache before fit() / load_weather()."
            )
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"raw_weather_{cache_key[:16]}.npz")

        envs = sorted(self._weather_data.keys())
        payload: dict[str, np.ndarray] = {
            "env_names": np.array(envs, dtype=object),
            "var_names": np.array(self._var_names, dtype=object),
            "dap_scaling": np.array(self.dap_scaling),
            "value_scaling": np.array(self.value_scaling),
            "extra_axes": np.array(sorted(self.extra_axes), dtype=object),
            "extra_axes_scaling": np.array(self.extra_axes_scaling),
        }
        if self.dap_scaling == "global_zscore":
            payload["dap_mean"] = np.asarray(self._norm_stats["dap_mean"])
            payload["dap_std"] = np.asarray(self._norm_stats["dap_std"])
        if self.value_scaling == "global_zscore":
            payload["val_mean"] = np.asarray(self._norm_stats["val_mean"])
            payload["val_std"] = np.asarray(self._norm_stats["val_std"])
        if self.extra_axes and self.extra_axes_scaling == "global_zscore":
            for ax in self.extra_axes:
                payload[f"{ax}_mean"] = np.asarray(self._norm_stats[f"{ax}_mean"])
                payload[f"{ax}_std"] = np.asarray(self._norm_stats[f"{ax}_std"])
        for env in envs:
            daps, values = self._weather_data[env]
            payload[f"{env}__daps"] = daps
            payload[f"{env}__values"] = values

        tmp_path = path + f".tmp.{os.getpid()}.npz"
        np.savez(tmp_path, **payload)
        try:
            os.replace(tmp_path, path)
        except FileNotFoundError:
            pass
        logger.info("Raw-weather cache saved: %s", path)

    def load_fitted(self, cache_dir: str, cache_key: str) -> None:
        """Restore per-env weather arrays + norm stats from cache.

        After this call, :meth:`transform` skips both the CSV read and
        the fit step — in-memory state is populated directly from the
        ``.npz`` written by :meth:`save_cache`.
        """
        path = os.path.join(cache_dir, f"raw_weather_{cache_key[:16]}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Raw-weather cache not found: {path}"
            )
        data = np.load(path, allow_pickle=True)
        envs = [str(e) for e in data["env_names"]]
        self._var_names = [str(v) for v in data["var_names"]]
        self._weather_data = {
            env: (data[f"{env}__daps"], data[f"{env}__values"])
            for env in envs
        }

        cached_dap = str(data["dap_scaling"])
        cached_val = str(data["value_scaling"])
        if cached_dap != self.dap_scaling or cached_val != self.value_scaling:
            raise ValueError(
                f"Raw-weather scaling mismatch: cache has "
                f"dap_scaling={cached_dap!r}, value_scaling={cached_val!r}; "
                f"current has dap_scaling={self.dap_scaling!r}, "
                f"value_scaling={self.value_scaling!r}."
            )
        # extra_axes metadata absent on pre-existing caches → treat as []/none.
        cached_extra = (
            [str(a) for a in data["extra_axes"]]
            if "extra_axes" in data else []
        )
        cached_extra_scaling = (
            str(data["extra_axes_scaling"])
            if "extra_axes_scaling" in data else "global_zscore"
        )
        if (cached_extra != sorted(self.extra_axes)
                or (self.extra_axes
                    and cached_extra_scaling != self.extra_axes_scaling)):
            raise ValueError(
                f"Raw-weather extra_axes mismatch: cache has "
                f"extra_axes={cached_extra}, scaling={cached_extra_scaling!r}; "
                f"current has extra_axes={sorted(self.extra_axes)}, "
                f"scaling={self.extra_axes_scaling!r}."
            )

        stats: dict[str, np.ndarray | float] = {}
        if self.dap_scaling == "global_zscore":
            stats["dap_mean"] = float(data["dap_mean"])
            stats["dap_std"] = float(data["dap_std"])
        if self.value_scaling == "global_zscore":
            stats["val_mean"] = data["val_mean"]
            stats["val_std"] = data["val_std"]
        if self.extra_axes and self.extra_axes_scaling == "global_zscore":
            for ax in self.extra_axes:
                stats[f"{ax}_mean"] = float(data[f"{ax}_mean"])
                stats[f"{ax}_std"] = float(data[f"{ax}_std"])
        self._norm_stats = stats

        # The daily gdd/agdd vectors are deterministic from the CSV (not
        # persisted) — rebuild them so transform() can attach the columns
        # without a fit pass. The CSV is available at predict time (same
        # assumption axis_source makes).
        if self.extra_axes:
            self._load_axis_daily(sorted(self._weather_data.keys()))
        logger.info("Raw-weather cache loaded: %s", path)


# ── WeatherFPCAProcessor (Mode 2) ────────────────────────────────────────


def _env_content_hash(
    weather_data: dict[str, tuple[np.ndarray, np.ndarray]],
    env_list: list[str],
) -> str:
    """Content-addressed hash over weather data for listed environments."""
    h = hashlib.sha256()
    h.update(b"weather_fpca_v1")
    for env in sorted(env_list):
        daps, values = weather_data[env]
        h.update(env.encode())
        h.update(daps.tobytes())
        h.update(values.tobytes())
    return h.hexdigest()


@register_processor
class WeatherFPCAProcessor(BaseProcessor):
    """FPC scores from daily weather curves via R fdapace.

    Each weather variable gets its own independent FPCA (same approach as
    VI FPCA — 42 independent univariate FPCAs).  Output: per-environment
    feature vector of shape ``[n_weather_vars * K_w]``.

    ``fit_scope`` controls which environments are used to fit FPCA:

    - ``"all"``: all dataset environments (pools the held-out env into
      the score basis — transductive).
    - ``"train"`` (default): train environments only — val/test
      projected via CE/BLUP.

    ``scaling`` controls post-FPCA score normalization.  Env-level
    features (one row per env) only support pass-through and
    train-stats-only z-score; ``within_env_zscore`` is rejected at
    construction since one-row-per-env makes per-env stats degenerate.

    Parameters
    ----------
    n_components : number of FPC scores to keep per weather variable.
    weather_csv_path : path to weather CSV.
    fit_vars : variables FPCA is actually **computed** for (the *fit set*),
        or ``None`` (default) to fit every column in the CSV. This is a
        compute/cost knob, NOT a downstream selection — restricting it
        avoids fitting variables you don't need (e.g. the expensive
        AGDD axis, where the cost is per-variable). The cache stores
        exactly the fit set, so ``fit_vars`` is folded into the cache key
        (``None`` keeps the historical full-fit key byte-identical).
        Distinct from ``emit_vars``: by per-variable FPCA independence the
        scores of any variable are bit-identical whether it was fit alone
        or alongside others, so this never changes outputs — only which
        variables get computed and which cache entry is used.
    emit_vars : variables whose scores are sliced out and handed
        downstream (the *emit set*), or ``None`` for "every variable in
        the fit set". Pure **post-load slice** over the cached fit — it is
        NOT part of the cache key, so different emit subsets reuse the same
        fit. Must be a subset of the fit set (``emit_vars ⊆ fit_vars``);
        requesting an emit variable that was not fit is an error. Per-column
        FPCA independence + per-column scaler stats make slice-then-scale
        bit-identical to scale-on-subset.
    fit_scope : ``"all"`` or ``"train"``.
    scaling : ``"none"`` (default) or ``"global_zscore"``.
        ``"global_zscore"`` z-scores the per-env score matrix using
        per-column mean/std computed over the **train** envs only
        (regardless of ``fit_scope``), then applies those stats to every
        env. Keeping the normalization train-only avoids leaking
        held-out-env information even when ``fit_scope="all"`` pools all
        envs for the FPCA basis. Slug code: ``gz``. Stats are computed on
        the post-slice columns, so they shape-match the emitted features
        for any ``emit_vars`` choice.
    cache_dir : directory for caching.  ``None`` disables caching.
    """

    _ALLOWED_SCALINGS = ("none", "global_zscore")

    # ── BaseProcessor metadata ──────────────────────────────────────
    name: ClassVar[str] = "weather_fpca"
    priority: ClassVar[int] = 3

    @classmethod
    def from_config(cls, config: dict) -> "WeatherFPCAProcessor":
        """Hydra-config factory used by the orchestrator."""
        return cls(
            n_components=config.get("n_components", 4),
            weather_csv_path=config["csv_path"],
            fit_vars=config.get("fit_vars"),
            emit_vars=config.get("emit_vars"),
            fit_scope=config.get("fit_scope", "train"),
            scaling=config.get("scaling", "none"),
            axis=config.get("axis", "dap"),
            dedup=config.get("dedup", True),
            gdd=config.get("gdd"),
            missing_values=config.get("missing_values", "drop"),
            # cache_dir comes from the orchestrator at construction time;
            # callers go through FeatureProcessor which passes it. Bare
            # from_config(config) leaves it None (caching disabled).
            cache_dir=config.get("cache_dir"),
        )

    _ALLOWED_AXES = ("dap", "agdd")

    def __init__(
        self,
        n_components: int,
        weather_csv_path: str,
        fit_vars: list[str] | None = None,
        emit_vars: list[str] | None = None,
        fit_scope: str = "train",
        scaling: str = "none",
        axis: str = "dap",
        dedup: bool = True,
        gdd: dict | None = None,
        missing_values: str = "drop",
        cache_dir: str | None = None,
    ):
        if n_components < 1:
            raise ValueError(
                f"n_components must be >= 1, got {n_components}"
            )
        if fit_scope not in ("all", "train"):
            raise ValueError(
                f"fit_scope must be 'all' or 'train', got {fit_scope!r}"
            )
        if scaling not in self._ALLOWED_SCALINGS:
            raise ValueError(
                f"scaling must be one of {self._ALLOWED_SCALINGS} "
                f"(env-level features can't use within_env_zscore — "
                f"one row per env makes within-env std degenerate), "
                f"got {scaling!r}."
            )
        if axis not in self._ALLOWED_AXES:
            raise ValueError(
                f"weather_fpca.axis must be one of {self._ALLOWED_AXES}, "
                f"got {axis!r}."
            )
        # Default "drop" mirrors the R reference: NaN cells stay NaN and
        # fdapace drops them per curve. "ffill" restores the historical
        # fabricated-value behaviour if a caller explicitly wants it.
        if missing_values not in WEATHER_MISSING_MODES:
            raise ValueError(
                f"weather_fpca.missing_values must be one of "
                f"{WEATHER_MISSING_MODES}, got {missing_values!r}."
            )
        # fit set (compute scope) vs emit set (post-load slice) — kept
        # strictly separate. `fit_vars=None` → fit every CSV column (the
        # historical Design-B behaviour). `emit_vars=None` → emit every
        # variable in the fit set.
        if fit_vars is not None and len(fit_vars) == 0:
            raise ValueError(
                "fit_vars must be a non-empty list or None "
                "(None = fit every CSV column)."
            )
        self.n_components = n_components
        self.missing_values = missing_values
        self.weather_csv_path = weather_csv_path
        self.fit_vars = list(fit_vars) if fit_vars is not None else None
        self.emit_vars = list(emit_vars) if emit_vars is not None else None
        # Enforce emit ⊆ fit early when both are explicit, so a misconfig
        # fails at construction rather than after a multi-hour fit. When
        # fit_vars is None the fit set is "all CSV columns" (unknown until
        # the CSV is read), so the check is deferred to slice time
        # (_resolve_subset_indices).
        if self.fit_vars is not None and self.emit_vars is not None:
            not_fit = [v for v in self.emit_vars if v not in set(self.fit_vars)]
            if not_fit:
                raise ValueError(
                    f"emit_vars {not_fit} are not in the fit set "
                    f"fit_vars={sorted(self.fit_vars)} — the emit set must be "
                    f"a subset of the fit set (you can only slice out a "
                    f"variable that was fit)."
                )
        self.fit_scope = fit_scope
        self.scaling = scaling
        # Weather time axis: 'dap' (native daily grid) or 'agdd' (the
        # deduped clean-AGDD grid from the shared axis table). The
        # R FPCA call is identical for both — only the time coordinate
        # changes. `gdd` configures AGDD construction (defaults to the
        # CSV GDD column = reference-exact); `dedup` builds the unique
        # monotonic AGDD axis fdapace needs.
        self.axis = axis
        self.dedup = bool(dedup)
        self.gdd_cfg = dict(gdd) if gdd else {}
        self.cache_dir = cache_dir

        # Fitted state — `_var_names_full` is the column order R fit on
        # (= the fit set: every CSV column when fit_vars is None, else
        # fit_vars in CSV order). `emit_vars` is resolved against it as a
        # post-load slice.
        self._var_names_full: list[str] | None = None
        self._max_k: int | None = None
        self._last_cache_key: str | None = None
        # Train env list captured at fit time (or read from cache
        # metadata at load_fitted time). Used to refit the scaler in
        # memory when scaling="global_zscore" — the scaler is no longer
        # persisted to disk because it derives cheaply from the full
        # per-env score matrix and a known fit-env set.
        self._train_envs_for_scaler: list[str] | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None

    # ── Subset resolution (post-load slice over the cached full fit) ──

    @property
    def _n_weather_vars_full(self) -> int:
        if self._var_names_full is None:
            raise RuntimeError(
                "WeatherFPCAProcessor: full var list not yet known. "
                "Call fit_and_transform_all() or load_fitted() first."
            )
        return len(self._var_names_full)

    @property
    def var_names(self) -> list[str]:
        """Resolved emit subset (preserves the fit-set column order)."""
        if self._var_names_full is None:
            raise RuntimeError(
                "Weather data not loaded yet. Call fit_and_transform_all() "
                "or load_fitted() first."
            )
        if self.emit_vars is None:
            return list(self._var_names_full)
        requested = set(self.emit_vars)
        return [v for v in self._var_names_full if v in requested]

    @property
    def n_weather_vars(self) -> int:
        """Subset size — drives feature_dims and the slice shape."""
        return len(self.var_names)

    def _resolve_subset_indices(self) -> list[int]:
        """Indices into ``_var_names_full`` (the fit set) for the emit subset.

        This is also where the ``emit_vars ⊆ fit_vars`` invariant is
        enforced for the deferred case (``fit_vars=None`` at construction,
        so the fit set was unknown then). ``emit_vars=None`` emits every
        variable in the fit set.
        """
        if self._var_names_full is None:
            raise RuntimeError(
                "WeatherFPCAProcessor: full var list not yet known."
            )
        if self.emit_vars is None:
            return list(range(len(self._var_names_full)))
        idx_by_name = {n: i for i, n in enumerate(self._var_names_full)}
        not_fit = [v for v in self.emit_vars if v not in idx_by_name]
        if not_fit:
            raise ValueError(
                f"emit_vars {sorted(not_fit)} are not in the fit set "
                f"(variables FPCA was computed for): {self._var_names_full}. "
                f"The emit set must be a subset of the fit set; if you need "
                f"these variables, add them to fit_vars (or leave fit_vars "
                f"unset to fit every CSV column)."
            )
        return [idx_by_name[v] for v in self._var_names_full
                if v in set(self.emit_vars)]

    def fit_and_transform_all(
        self,
        train_envs: list[str],
        val_envs: list[str] | None = None,
        test_envs: list[str] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Fit FPCA and compute scores for all environments.

        Returns ``{"weather_fpc_scores": {env_name: vector(n_vars * K)}}``
        — a flat per-environment mapping covering every env in
        ``train_envs ∪ val_envs ∪ test_envs``, regardless of ``fit_scope``.

        The per-env return shape is deliberately unstructured w.r.t. splits:
        downstream consumers (FeatureProcessor) look up each sample's env
        directly, so there is no opportunity for ordering bugs in the
        sample → feature mapping.
        """
        # Load weather data for all mentioned envs
        all_envs = list(train_envs)
        if val_envs:
            all_envs.extend(val_envs)
        if test_envs:
            all_envs.extend(test_envs)
        all_envs = sorted(set(all_envs))

        # Load the fit set: every CSV column when fit_vars is None (Design
        # B — `emit_vars` is then a post-load slice and the full fit is
        # reused across emit subsets), or just `fit_vars` when restricted
        # (compute only what's needed — e.g. the expensive AGDD axis).
        weather_data, var_names = load_weather_csv(
            self.weather_csv_path,
            envs=all_envs,
            weather_vars=self.fit_vars,
            missing_values=self.missing_values,
            # Full precision to fdapace — see load_weather_csv's `dtype`.
            dtype=np.float64,
        )
        # AGDD axis: replace each env's (daily DAP grid, values) with the
        # deduped (clean-AGDD coord, averaged values) from the shared axis
        # table. The R FPCA call is unchanged — only Lt changes.
        # The substituted arrays flow into the cache key via
        # `_compute_cache_key`'s content hash, so DAP vs AGDD fits land in
        # distinct entries (belt-and-suspenders: axis/dedup are also added
        # to the key explicitly).
        if self.axis == "agdd":
            weather_data = self._agdd_weather_data(all_envs, var_names)
        self._var_names_full = var_names
        self._train_envs_for_scaler = sorted(set(train_envs))

        # Cache key — always over all envs.  For fit_scope="train" the val/
        # test scores still depend on which val/test envs were requested
        # (CE/BLUP projection), so reusing a train-only cache across runs
        # with different val/test sets would silently return stale scores.
        cache_key = self._compute_cache_key(weather_data, train_envs, all_envs)
        self._last_cache_key = cache_key

        # Try cache
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("Weather FPCA cache hit — skipping R subprocess.")
            env_scores_full = cached
        else:
            env_scores_full = self._run_fpca(
                weather_data, var_names, train_envs, val_envs, test_envs
            )
            if self.cache_dir is not None:
                self._save_cache(cache_key, env_scores_full, train_envs)

        # Fit the per-column scaler on the train envs every call. Cheap
        # (one stack + mean/std) and avoids on-disk scaler files that
        # would otherwise need a per-subset filename. Always train-only,
        # independent of fit_scope, to keep the normalization leakage-safe.
        if self.scaling == "global_zscore":
            self._fit_scaler(env_scores_full, train_envs)

        # Slice each env's full score vector to the requested var subset
        # and n_components, then apply the (already fitted) scaler.
        sliced = self._slice_env_scores(env_scores_full)
        return {"weather_fpc_scores": sliced}

    @property
    def feature_dims(self) -> dict[str, int]:
        """Output dimensions: ``{"weather_fpc_scores": n_vars * n_components}``.

        Reports the **subset** count, not the cached full count.
        """
        if self._var_names_full is None:
            raise RuntimeError(
                "Must call fit_and_transform_all() before feature_dims."
            )
        return {"weather_fpc_scores": self.n_weather_vars * self.n_components}

    @property
    def max_k(self) -> int | None:
        """Max components available from R (after fit)."""
        return self._max_k

    # ── BaseProcessor lifecycle ────────────────────────────────────

    def fit(self, ctx: OrchestratorContext) -> None:
        """Fit FPCA on the union of train/val/test envs (matches the
        existing single-R-subprocess optimisation), stash per-env feature
        dict for later split broadcasts.

        For ``fit_scope="all"`` the R fit pools all envs; for
        ``fit_scope="train"`` the held-out envs are projected via CE/BLUP
        in the same R call. Either way, the returned per-env mapping
        covers every env in the dataset.
        """
        train_envs = ctx.unique_envs_for_split("train")
        val_envs = ctx.unique_envs_for_split("val") or None
        test_envs = ctx.unique_envs_for_split("test") or None
        # fit_and_transform_all returns {"weather_fpc_scores": {env: vec}}
        self._fitted_env_features = self.fit_and_transform_all(
            train_envs, val_envs, test_envs,
        )

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """Broadcast per-env features to per-sample for the given split.

        Works for both fit-mode (uses `self._fitted_env_features` set
        during `fit()`) and predict-mode (delegates to `transform_envs`
        on the env subset for the split).
        """
        sample_envs = ctx.envs_for_split(split)
        if not sample_envs:
            return None

        if hasattr(self, "_fitted_env_features"):
            env_features = self._fitted_env_features
        else:
            # Predict-mode after load_fitted: derive per-env features for
            # exactly the envs this split needs.
            unique_envs = sorted(set(sample_envs))
            env_features = self.transform_envs(unique_envs)

        out: dict[str, np.ndarray] = {}
        for feat_name, env_dict in env_features.items():
            out[feat_name] = np.stack(
                [env_dict[e] for e in sample_envs]
            ).astype(np.float32)
        return out

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Return the cache key computed during `fit()`.

        `fit_and_transform_all` sets `self._last_cache_key` as a side
        effect, so post-fit callers always have a valid key without
        recomputation.
        """
        if self._last_cache_key is None:
            raise RuntimeError(
                "WeatherFPCAProcessor.cache_key called before fit()."
            )
        return self._last_cache_key

    def save_cache(self, cache_dir: str, key: str) -> None:
        """No-op: the underlying cache is written from inside
        `fit_and_transform_all` (which knows the per-env score matrix
        before the slicing/scaling step). Kept here so the
        orchestrator's uniform call sequence still works.
        """

    # ── Private: AGDD axis substitution ─────────────────────────

    def _agdd_weather_data(
        self, envs: list[str], var_names: list[str] | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per-env ``(agdd_coord, values)`` from the shared axis table.

        Builds a (deterministic) :class:`AxisTable` from the same CSV and
        returns the deduped clean-AGDD grid + averaged weather values per
        env — the weather AGDD path. Imported lazily to avoid the
        ``weather`` ↔ ``axis_source`` module cycle.

        ``var_names`` selects (and orders) the value columns to return,
        mirroring :meth:`RawWeatherProcessor._agdd_weather_data`. It must be
        a subset of the table's columns. ``None`` returns every column in
        the table's order (the historical full-fit behaviour). The table is
        ALWAYS built over every CSV column regardless — GDD is needed for the
        ``cumsum`` and the dedup averages all columns — so the kept columns'
        values are identical whether or not the others were selected (this
        is what makes a restricted ``fit_vars`` fit bit-identical to a full
        fit sliced to the same variables).
        """
        from .axis_source import build_axis_table

        table = build_axis_table(
            self.weather_csv_path,
            gdd_cfg=self.gdd_cfg,
            envs=envs,
            dedup=self.dedup,
            missing_values=self.missing_values,
            dtype=np.float64,
        )
        col_idx = (
            None if var_names is None
            else [table.var_names.index(n) for n in var_names]
        )
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for env in envs:
            coord, values = table.env_axis(env, "agdd")
            if col_idx is not None:
                values = values[:, col_idx]
            out[env] = (
                coord.astype(np.float64),
                # float64: this feeds fdapace, not a tensor.
                values.astype(np.float64),
            )
        return out

    # ── Private: cache key ─────────────────────────────────────

    def _compute_cache_key(
        self,
        weather_data: dict[str, tuple[np.ndarray, np.ndarray]],
        train_envs: list[str],
        all_envs: list[str],
    ) -> str:
        h = hashlib.sha256()
        # v4: Design B — the cache stores the per-env score matrix for the
        # FIT SET (every CSV column when fit_vars is None, else fit_vars).
        # `emit_vars` is a post-load slice and NOT part of the key, so emit
        # subsets reuse the same fit. `n_components` and `scaling` are also
        # post-load slices — same cache, different views.
        h.update(b"weather_fpca_v4")
        h.update(self.fit_scope.encode())
        # Tagged unconditionally (unlike `fit_vars` below): "drop" vs
        # "ffill" changes the CURVES fdapace sees, so every pre-existing
        # entry — all computed under the old unconditional ffill — must
        # miss. `_env_content_hash` already differs (NaN vs filled), so
        # this is belt-and-suspenders, but a silent hit here would
        # resurrect fabricated scores.
        h.update(b"|missing=")
        h.update(self.missing_values.encode())
        h.update(_env_content_hash(weather_data, all_envs).encode())
        if self.fit_scope == "train":
            h.update(b"|train=")
            h.update(",".join(sorted(train_envs)).encode())
        # Always tag the axis name so every entry is self-describing on disk
        # (DAP included). The substituted AGDD arrays already alter
        # `_env_content_hash` above; `dedup`/`gdd` only shape the AGDD grid, so
        # they're folded in only for the derived axis.
        h.update(b"|axis=")
        h.update(self.axis.encode())
        if self.axis != "dap":
            h.update(f"|dedup={self.dedup}".encode())
            h.update(b"|gdd=")
            h.update(json.dumps(self.gdd_cfg, sort_keys=True).encode())
        # Restricting the fit set changes what the cache stores, so it must
        # key a distinct entry. Folded in ONLY when restricted, so the
        # default full-fit (fit_vars=None) key stays byte-identical to the
        # historical Design-B key — existing caches remain valid. (The
        # subset already alters `_env_content_hash` above since fewer columns
        # are loaded; this explicit tag is belt-and-suspenders + legibility,
        # mirroring how `axis` is tagged.)
        if self.fit_vars is not None:
            h.update(b"|fit_vars=")
            h.update(",".join(sorted(self.fit_vars)).encode())
        return h.hexdigest()

    # ── Private: R execution ─────────────────────────────────────────────

    def _run_fpca(
        self,
        weather_data: dict[str, tuple[np.ndarray, np.ndarray]],
        var_names: list[str],
        train_envs: list[str],
        val_envs: list[str] | None,
        test_envs: list[str] | None,
    ) -> dict[str, np.ndarray]:
        """Reformat weather into tall CSV, call R, return per-env scores.

        Returns ``{env_name: vector(n_vars * max_k)}`` for every env in
        the input — labelled directly via the ``id_to_env`` mapping built
        alongside the tall DataFrame, with no row-order assumptions.
        """
        tmpdir = tempfile.mkdtemp(prefix="weather_fpca_")
        try:
            input_csv = os.path.join(tmpdir, "sparse_curves.csv")
            output_csv = os.path.join(tmpdir, "fpc_scores.csv")
            model_dir = os.path.join(tmpdir, "models")
            os.makedirs(model_dir)

            # Build tall DataFrame and the sample_id → env mapping
            tall_df, id_to_env = self._build_tall_df(
                weather_data, var_names, train_envs, val_envs, test_envs
            )
            tall_df.to_csv(input_csv, index=False)

            # Cap the requested K at N-1 for the N envs fdapace is fitting
            # on: Sparse FPCA on N curves can yield at most N-1 components,
            # and fdapace silently returns fewer when the smoothed
            # covariance is lower-rank (e.g. 10 components from 18 envs).
            # The request is only a ceiling on how many scores to read back
            # — fit time is dominated by the GCV covariance smoothing and is
            # independent of K (K=1 and K=17 measured identically). We also
            # cap at 20 since we only ever slice down to a handful.
            n_fit_envs = (
                len(set(train_envs) | set(val_envs or []) | set(test_envs or []))
                if self.fit_scope == "all"
                else len(set(train_envs))
            )
            r_n_components = max(1, min(20, n_fit_envs - 1))

            # nRegGrid: the reference sets 100 on the FULL 19-env weather
            # fit (its `fpca.R`, feeding CV2/CV1) and leaves it at
            # fdapace's default 51 for the LOEO fits (CV0/CV00) and for
            # every VI fit. "Full" here means no env is held out of the
            # basis — every env lands in R's "train" split.
            #
            # Not tagged in the cache key: `held_out` is exactly
            # `all_envs - train_envs`, and both already feed the key (via
            # `_env_content_hash` over all_envs and the train_envs tag),
            # so this value can't vary for a fixed key.
            held_out = (
                set(val_envs or []) | set(test_envs or [])
            ) - set(train_envs)
            n_reg_grid = 0 if held_out else 100

            self._call_rscript(
                input_csv, output_csv, model_dir, r_n_components,
                n_reg_grid=n_reg_grid,
            )

            from .vi_fpca import _read_fpc_scores_all

            raw_scores = _read_fpc_scores_all(
                output_csv, n_vis=self._n_weather_vars_full
            )

            # Flatten across R-splits into a per-env dict.  Each row of
            # each R-split's X is labelled by its sample_id (the position
            # in raw_scores[r_split][1]); id_to_env tells us which env it
            # is.  No ordering assumptions — we look up by id, not by row.
            env_scores: dict[str, np.ndarray] = {}
            max_k = 0
            for r_split, (X, ids, mk) in raw_scores.items():
                max_k = mk
                for row_idx, sample_id in enumerate(ids):
                    env = id_to_env[int(sample_id)]
                    env_scores[env] = X[row_idx]
            self._max_k = max_k if raw_scores else None

            return env_scores

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _build_tall_df(
        self,
        weather_data: dict[str, tuple[np.ndarray, np.ndarray]],
        var_names: list[str],
        train_envs: list[str],
        val_envs: list[str] | None,
        test_envs: list[str] | None,
    ) -> tuple[pd.DataFrame, dict[int, str]]:
        """Reformat weather into the tall CSV schema for fpca_compute.R.

        Columns: sample_id, vi_index, dap, value, yield, split

        Returns ``(tall_df, id_to_env)`` so callers can map R's sample_id
        column back to the originating environment without making any
        assumptions about row ordering inside R's output.
        """
        # Assign integer IDs to environments (sorted for determinism)
        if self.fit_scope == "all":
            # All envs treated as "train" for R
            all_envs = sorted(
                set(train_envs)
                | set(val_envs or [])
                | set(test_envs or [])
            )
            env_to_id = {e: i for i, e in enumerate(all_envs)}
            env_to_split = {e: "train" for e in all_envs}
        else:
            # train envs → "train", val → "val", test → "test"
            all_envs_set = set(train_envs)
            env_to_split = {e: "train" for e in train_envs}
            # Use setdefault so the highest-precedence split wins
            # (train > val > test). Under cv_spec masking is within-env, so
            # an env can appear in BOTH train and test; if its weather curve
            # is observed at fit time (any train rows) it must stay "train"
            # and belong in the basis fit. Only purely-held-out envs (in test
            # but not train) are projected as "test".
            if val_envs:
                all_envs_set.update(val_envs)
                for e in val_envs:
                    env_to_split.setdefault(e, "val")
            if test_envs:
                all_envs_set.update(test_envs)
                for e in test_envs:
                    env_to_split.setdefault(e, "test")
            all_envs = sorted(all_envs_set)
            env_to_id = {e: i for i, e in enumerate(all_envs)}

        parts = []
        n_vars = len(var_names)
        for env in all_envs:
            daps, values = weather_data[env]
            env_id = env_to_id[env]
            split_name = env_to_split[env]
            n_days = len(daps)

            for vi_idx in range(n_vars):
                parts.append(
                    pd.DataFrame(
                        {
                            "sample_id": env_id,
                            "vi_index": vi_idx,
                            # Generic FPCA time axis (DAP or AGDD per `axis`);
                            # R reads a column literally named `t`.
                            "t": daps,
                            "value": values[:, vi_idx],
                            "yield": 0.0,
                            "split": split_name,
                        }
                    )
                )

        id_to_env = {i: e for e, i in env_to_id.items()}
        return pd.concat(parts, ignore_index=True), id_to_env

    def _call_rscript(
        self,
        input_csv: str,
        output_csv: str,
        model_dir: str,
        n_components: int,
        n_reg_grid: int = 0,
    ) -> None:
        """Call R fpca_compute.R in train mode.

        ``n_reg_grid=0`` leaves ``nRegGrid`` unset so fdapace's default
        (51) applies. See :meth:`_run_fpca` for when 100 is passed.
        """
        r_script = os.path.normpath(os.path.abspath(_R_SCRIPT))
        cmd = [
            "Rscript",
            r_script,
            "--mode",
            "train",
            "--input_csv",
            input_csv,
            "--output_csv",
            output_csv,
            "--model_dir",
            model_dir,
            "--n_components",
            str(n_components),
        ]
        if n_reg_grid > 0:
            cmd += ["--n_reg_grid", str(n_reg_grid)]
        logger.info("Running weather FPCA: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout.rstrip())
        if result.stderr:
            logger.warning(result.stderr.rstrip())
        if result.returncode != 0:
            raise RuntimeError(
                f"Rscript failed (return code {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

    # ── Private: cache ─────────────────────────────────────────

    def _cache_subdir(self, cache_key: str) -> str:
        tag = cache_key[:16]
        return os.path.join(self.cache_dir, "weather_fpca", tag)

    def _load_cache(self, cache_key: str) -> dict[str, np.ndarray] | None:
        """Load cached per-env scores if valid.

        Returns ``{env_name: vector(n_vars_full * max_k)}`` or ``None``
        on miss. Populates ``self._var_names_full`` and
        ``self._train_envs_for_scaler`` from metadata so the scaler can
        be refit in memory without needing disk-resident scaler files.
        """
        if self.cache_dir is None:
            return None

        cache_path = self._cache_subdir(cache_key)
        fp_path = os.path.join(cache_path, "fingerprint")
        scores_path = os.path.join(cache_path, "fpc_scores.npz")
        meta_path = os.path.join(cache_path, "metadata.json")

        if not (
            os.path.isfile(fp_path)
            and os.path.isfile(scores_path)
            and os.path.isfile(meta_path)
        ):
            return None

        with open(fp_path) as f:
            cached_key = f.read().strip()
        if cached_key != cache_key:
            return None

        with open(meta_path) as f:
            meta = json.load(f)
        self._max_k = int(meta["max_k"])
        self._var_names_full = [str(v) for v in meta["var_names_full"]]
        self._train_envs_for_scaler = [str(e) for e in meta["train_envs"]]

        data = np.load(scores_path, allow_pickle=True)
        env_names = data["env_names"].tolist()
        scores = data["scores"]
        return {env: scores[i] for i, env in enumerate(env_names)}

    def _save_cache(
        self,
        cache_key: str,
        env_scores: dict[str, np.ndarray],
        train_envs: list[str],
    ) -> None:
        """Save per-env scores and metadata to cache directory.

        Writes atomically (uniquely-tagged staging dir + rename) so parallel
        SLURM jobs sharing the same cache key cannot corrupt each other.
        Persists ``var_names_full`` (column order R fit on) and
        ``train_envs`` (needed at predict-mode load to refit the
        scaler in memory under ``fit_scope="train"``).
        """
        cache_path = self._cache_subdir(cache_key)
        # PID alone is unique only WITHIN a node. Two SLURM jobs on
        # different nodes can share a PID, and when they also share a
        # cache key — as env and env-free variants of the same model do,
        # since their upstream FPCA settings are identical — they would
        # write into the SAME staging dir on shared scratch, interleave,
        # and rename a corrupt mixture into place. The uuid makes the
        # staging path globally unique; the PID is kept for legibility
        # when tracing a stray directory back to its job.
        staging_path = cache_path + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:12]}"
        os.makedirs(staging_path, exist_ok=True)

        env_names = sorted(env_scores.keys())
        scores = np.stack([env_scores[e] for e in env_names])
        np.savez(
            os.path.join(staging_path, "fpc_scores.npz"),
            scores=scores,
            env_names=np.array(env_names, dtype=object),
        )

        meta = {
            "max_k": self._max_k or 0,
            "var_names_full": list(self._var_names_full or []),
            "train_envs": sorted(set(train_envs)),
            "fit_scope": self.fit_scope,
            # The fit set this entry was computed for (None = every CSV
            # column). Validated on load_fitted to catch a stale/mismatched
            # cache; var_names_full already records the resolved columns.
            "fit_vars": sorted(self.fit_vars) if self.fit_vars is not None else None,
            "n_envs": len(env_names),
        }
        with open(os.path.join(staging_path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        with open(os.path.join(staging_path, "fingerprint"), "w") as f:
            f.write(cache_key)

        # Atomically move staging dir to final path. If a sibling
        # process already placed the final dir, the rename fails —
        # the cache is warm regardless of which process won.
        try:
            os.rename(staging_path, cache_path)
        except OSError:
            # Another process won the race — clean up our staging dir.
            shutil.rmtree(staging_path, ignore_errors=True)

        logger.info("Cached weather FPCA to %s", cache_path)

    # ── Predict mode: load fitted + transform new envs ──────────

    def load_fitted(self, cache_dir: str, cache_key: str) -> None:
        """Load cached per-env FPC scores for predict-mode lookup.

        After this call, :meth:`transform_envs` returns sliced scores
        for any env present in the cache without re-running R.

        The cache — written by :meth:`_save_cache` during train — stores
        the full ``(n_weather_vars * max_k,)`` vector per env name for
        *every* env that was present at fit time (both fit_scope="all"
        and fit_scope="train" cover all envs in the resulting dict).

        Raises
        ------
        FileNotFoundError
            If the cache subdir or ``fpc_scores.npz`` is missing.
        """
        cache_path = os.path.join(
            cache_dir, "weather_fpca", cache_key[:16]
        )
        scores_path = os.path.join(cache_path, "fpc_scores.npz")
        meta_path = os.path.join(cache_path, "metadata.json")
        if not os.path.isfile(scores_path):
            raise FileNotFoundError(
                f"Weather FPCA cache missing fpc_scores.npz at: {scores_path}"
            )
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Weather FPCA cache missing metadata.json at: {meta_path}"
            )

        with open(meta_path) as f:
            meta = json.load(f)
        self._max_k = int(meta["max_k"])
        self._var_names_full = [str(v) for v in meta["var_names_full"]]
        self._train_envs_for_scaler = [str(e) for e in meta["train_envs"]]
        if meta.get("fit_scope") != self.fit_scope:
            raise ValueError(
                f"Weather FPCA fit_scope mismatch: cache has "
                f"{meta.get('fit_scope')!r}, current has {self.fit_scope!r}"
            )
        # The fit set this cache was computed for. Pre-existing caches
        # predate the field → treat a missing key as the full fit (None).
        cached_fit_vars = meta.get("fit_vars", None)
        current_fit_vars = (
            sorted(self.fit_vars) if self.fit_vars is not None else None
        )
        if cached_fit_vars != current_fit_vars:
            raise ValueError(
                f"Weather FPCA fit_vars mismatch: cache was fit on "
                f"{cached_fit_vars!r}, current processor has "
                f"{current_fit_vars!r}. The cache key should have prevented "
                f"this — check your config."
            )

        data = np.load(scores_path, allow_pickle=True)
        env_names = [str(e) for e in data["env_names"]]
        scores = data["scores"]
        self._loaded_env_scores_full: dict[str, np.ndarray] = {
            env: scores[i] for i, env in enumerate(env_names)
        }
        self._last_cache_key = cache_key

        # Refit the scaler in memory from the cached full scores +
        # the training-time train env list. Cheap and shape-correct
        # for whatever `emit_vars` the caller is currently using —
        # no per-subset scaler files needed. Train-only stats regardless
        # of fit_scope.
        if self.scaling == "global_zscore":
            self._fit_scaler(
                self._loaded_env_scores_full,
                train_envs=self._train_envs_for_scaler,
            )

        logger.info("Weather FPCA fitted state loaded from %s", cache_path)

    def transform_envs(self, env_list: list[str]) -> dict[str, np.ndarray]:
        """Look up sliced FPC scores for a list of env names.

        Requires :meth:`load_fitted` (or :meth:`fit_and_transform_all`)
        first. Returns ``{"weather_fpc_scores": {env: vector(n_comp*n_vars)}}``
        — the same per-env mapping as :meth:`fit_and_transform_all`.

        Raises
        ------
        KeyError
            If any requested env is not in the loaded cache.
        """
        loaded = getattr(self, "_loaded_env_scores_full", None)
        if loaded is None:
            raise RuntimeError(
                "WeatherFPCAProcessor.transform_envs requires "
                "load_fitted() first."
            )
        missing = [e for e in env_list if e not in loaded]
        if missing:
            raise KeyError(
                f"Weather FPCA cache is missing envs {missing}. "
                f"Available envs: {sorted(loaded.keys())}"
            )
        subset = {env: loaded[env] for env in env_list}
        return {"weather_fpc_scores": self._slice_env_scores(subset)}

    # ── Private: result building ───────────────────────────────

    def _slice_full_to_subset_k(self, vec_full: np.ndarray) -> np.ndarray:
        """Slice ``(n_vars_full * max_k,)`` → ``(n_vars_subset * n_components,)``.

        Two sequential slices: select subset rows from the
        ``(n_vars_full, max_k)`` reshape, then take the first
        ``n_components`` columns. Zero-pad if ``n_components > max_k``
        for parity with :func:`_slice_fpc_scores`.
        """
        max_k = self._max_k
        n_full = self._n_weather_vars_full
        block = vec_full.reshape(n_full, max_k)
        subset_idx = self._resolve_subset_indices()
        block = block[subset_idx]                          # (n_subset, max_k)

        n_comp = self.n_components
        if n_comp == max_k:
            return block.reshape(-1)
        if n_comp < max_k:
            return block[:, :n_comp].reshape(-1)
        # n_comp > max_k — pad with zeros, mirroring _slice_fpc_scores.
        warnings.warn(
            f"Requested n_components={n_comp} but R produced only "
            f"{max_k} — extra components will be zero-padded.",
            stacklevel=3,
        )
        out = np.zeros((len(subset_idx), n_comp), dtype=block.dtype)
        out[:, :max_k] = block
        return out.reshape(-1)

    def _slice_env_scores(
        self,
        env_scores_full: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Slice each env's full score vector to the requested var subset
        and ``n_components`` per var, then apply the configured scaler.
        """
        if not env_scores_full:
            return {}

        sliced: dict[str, np.ndarray] = {
            env: self._slice_full_to_subset_k(vec)
            for env, vec in env_scores_full.items()
        }

        if self.scaling == "global_zscore":
            if self._scaler_mean is None or self._scaler_std is None:
                raise RuntimeError(
                    "scaling='global_zscore' but scaler stats not fitted "
                    "(call fit_and_transform_all or load_fitted first)."
                )
            for env in sliced:
                sliced[env] = (sliced[env] - self._scaler_mean) / self._scaler_std

        return sliced

    # ── Private: scaler ────────────────────────────────────────

    def _fit_scaler(
        self,
        env_scores_full: dict[str, np.ndarray],
        train_envs: list[str],
    ) -> None:
        """Fit per-column mean/std on the SUBSET-and-K-sliced scores of
        the TRAIN envs only.

        The z-score normalization statistics are always derived from the
        train split — never val/test — regardless of ``fit_scope``.
        ``fit_scope`` governs the FPCA *basis* fit (train-only vs all
        envs); the score normalization stays train-only either way so it
        cannot leak held-out-env information into downstream kernels.

        Stats are computed on the same shape that downstream consumers
        see, so the scaler is dimensionally aligned with the sliced
        output for whatever ``emit_vars`` is currently set.
        """
        fit_envs = sorted(set(train_envs))
        rows = [
            self._slice_full_to_subset_k(env_scores_full[env])
            for env in fit_envs if env in env_scores_full
        ]
        if not rows:
            raise RuntimeError(
                "WeatherFPCAProcessor._fit_scaler: no train-env scores "
                "available to fit the scaler."
            )
        X = np.stack(rows).astype(np.float64)
        mean = X.mean(axis=0)
        # ddof=1 matches R's sd() — the R reference's safe_scale_matrix → scale()
        # uses sd() under the hood.
        std = X.std(axis=0, ddof=1)
        # Match the R reference's safe_scale_matrix: clamp std=0 / non-finite to 1
        std[~np.isfinite(std) | (std == 0)] = 1.0
        self._scaler_mean = mean
        self._scaler_std = std
