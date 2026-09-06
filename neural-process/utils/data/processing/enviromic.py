"""Enviromic feature extraction from weather-based environment similarity.

Builds environment-to-environment kernels from aggregated weather data.

``fit_scope`` controls which environments are used to build the kernel:
- ``"train"`` (default): train environments only — the held-out env(s) get
  cross-kernel rows (their similarity to the train envs, computed at transform
  time). This is the deployment-honest / leakage-safe choice: for a held-out
  environment (a future year / new location — CV0, CV00, env_year_loo) you do
  NOT have that environment's weather at training time, so it must not enter the
  fit. "weather is public" does not rescue this — a *future* season's weather is
  simply not available when you train.
- ``"all"``: pool every environment (train + held-out) into the fit —
  transductive, matches the full-N R reference, but it uses the held-out env's
  weather (which a future-year deployment would not have), so it is NOT
  leakage-free for environment-holdout schemes. Set explicitly only for
  deliberate reference parity. (Only ``genomic`` legitimately defaults to
  ``"all"`` — a new line's genotype IS known at prediction time.)

Per-source ``scaling`` controls how the aggregated weather matrix is
normalized before kernel construction:
- ``"none"`` (default): pass-through.
- ``"global_zscore"``: per-column z-score using stats over the
  fitting-set envs (matches the R reference's ``safe_scale_matrix`` on weather
  scores). Slug code: ``gz``.

Env-level features (one row per env) cannot use ``within_env_zscore`` —
the within-env std is degenerate. ``__init__`` rejects it explicitly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import ClassVar

import numpy as np

from .base import BaseProcessor, OrchestratorContext
from .registry import register_processor
from .kernels import (
    center_kernel_cross,
    center_kernel_train,
    compute_kernel,
    eigen_decompose,
    project_eigen_cross,
    project_eigen_train,
    read_component_params,
    slice_components,
)
from .score_scalers import build_scaler

logger = logging.getLogger(__name__)


def aggregate_weather(
    weather_dict: dict[str, np.ndarray],
    aggregation: str = "mean",
) -> dict[str, np.ndarray]:
    """Aggregate daily weather curves into per-environment summary vectors.

    Parameters
    ----------
    weather_dict : mapping ``{env_name: array(n_days, n_vars)}``.
    aggregation : ``"mean"``, ``"std"``, or ``"mean_std"``.

    Returns
    -------
    agg_dict : mapping ``{env_name: array(n_agg_features,)}``.
    """
    # ddof=1 keeps the std summary consistent with the rest of the data
    # pipeline (score_scalers, weather processors, dataset z-score).
    result = {}
    for env, curves in weather_dict.items():
        if aggregation == "mean":
            result[env] = curves.mean(axis=0)
        elif aggregation == "std":
            result[env] = curves.std(axis=0, ddof=1)
        elif aggregation == "mean_std":
            result[env] = np.concatenate(
                [curves.mean(axis=0), curves.std(axis=0, ddof=1)]
            )
        else:
            raise ValueError(
                f"Unknown aggregation: {aggregation!r}. "
                f"Expected 'mean', 'std', or 'mean_std'."
            )
    return result


def _source_scaling(src: dict) -> str:
    return src.get("scaling", "none")


@register_processor
class EnviromicFeatureBuilder(BaseProcessor):
    """Enviromic feature extraction from weather-based environment similarity.

    Parameters
    ----------
    sources : list of dicts, each with keys:
        - ``name``: str — output key
        - ``type``: str — ``"raw_weather"`` (aggregated summary),
          ``"enviromic_relmat"`` (eigen-projected kernel), or
          ``"enviromic_relmat_rows"`` (raw centered kernel rows)
        - ``n_components``: int | None — hard-fix component count for
          eigen sources. Raises if greater than effective rank.
        - ``n_components_max``: int | None — soft ceiling for eigen
          sources. Mutually exclusive with ``n_components``.
        - ``aggregation``: str — ``"mean"``, ``"std"``, or ``"mean_std"``
          (for raw_weather and kernel sources)
        - ``kernel``: str — ``"linear"`` or ``"rbf"`` (for kernel sources)
        - ``gamma``: float | None — RBF bandwidth
        - ``scaling``: str — ``"none"`` (default) or ``"global_zscore"``.
          Sources with different scalings get separate kernel state;
          sources sharing a scaling reuse it.
    fit_scope : ``"all"`` or ``"train"``.
    """

    _EIGEN_TYPES = {"enviromic_relmat"}
    _ROW_TYPES = {"enviromic_relmat_rows"}
    _KERNEL_TYPES = _EIGEN_TYPES | _ROW_TYPES
    _ALLOWED_SCALINGS = ("none", "global_zscore")

    # ── BaseProcessor metadata ──────────────────────────────────────
    name: ClassVar[str] = "enviromic"
    priority: ClassVar[int] = 6
    has_eigen_sources: ClassVar[bool] = True
    eigen_source_types: ClassVar[frozenset[str]] = frozenset(
        {"enviromic_relmat"}
    )

    _ALLOWED_AXES = ("dap", "agdd")

    @classmethod
    def from_config(cls, config: dict) -> "EnviromicFeatureBuilder":
        """Hydra-config factory used by the orchestrator."""
        return cls(
            sources=config.get("sources", []),
            fit_scope=config.get("fit_scope", "train"),
            center_kernel=config.get("center_kernel", False),
            weather_vars=config.get("weather_vars"),
            axis=config.get("axis", "dap"),
            dedup=config.get("dedup", True),
            gdd=config.get("gdd"),
        )

    def __init__(
        self,
        sources: list[dict],
        fit_scope: str = "train",
        center_kernel: bool = False,
        weather_vars: list[str] | None = None,
        axis: str = "dap",
        dedup: bool = True,
        gdd: dict | None = None,
    ):
        if axis not in self._ALLOWED_AXES:
            raise ValueError(
                f"enviromic.axis must be one of {self._ALLOWED_AXES}, "
                f"got {axis!r}."
            )
        self.sources = sources
        self.fit_scope = fit_scope
        # When False, the env-similarity kernel is eigen-decomposed in
        # its raw form (no double-centering, no cross-kernel offsets).
        self.center_kernel = bool(center_kernel)
        # Subset of weather variables used to build the env-similarity
        # kernel. None = all non-metadata columns in the CSV.
        self.weather_vars = (
            list(weather_vars) if weather_vars is not None else None
        )
        # Time axis over which the daily weather is aggregated into the
        # env summary (mean/std). 'dap' = the native daily grid; 'agdd' =
        # the deduped clean-AGDD grid from the shared axis table,
        # which collapses dup-AGDD (zero-GDD) days before aggregating.
        # `dedup`/`gdd` configure the AGDD grid as elsewhere.
        self.axis = axis
        self.dedup = bool(dedup)
        self.gdd_cfg = dict(gdd) if gdd else {}
        self._validate_sources()

        # Scaling-INDEPENDENT state (raw aggregates over the fitting set).
        self._env_order: list[str] | None = None
        self._env_to_idx: dict[str, int] | None = None
        self._agg_features: dict[str, np.ndarray] | None = None

        # Per-scaling state — keys are scaling config names; each value
        # holds: scaler, K_centered, col_means, grand_mean, V, D.
        self._state: dict[str, dict] = {}

    def _validate_sources(self):
        valid_types = {"raw_weather"} | self._KERNEL_TYPES
        names = set()
        for src in self.sources:
            if src["type"] not in valid_types:
                raise ValueError(
                    f"Unknown enviromic source type: {src['type']!r}. "
                    f"Valid: {sorted(valid_types)}"
                )
            if src["name"] in names:
                raise ValueError(
                    f"Duplicate source name: {src['name']!r}"
                )
            names.add(src["name"])
            scaling = _source_scaling(src)
            if scaling not in self._ALLOWED_SCALINGS:
                raise ValueError(
                    f"scaling={scaling!r} on enviromic source "
                    f"{src['name']!r}: env-level features can only use "
                    f"{list(self._ALLOWED_SCALINGS)} (one row per env "
                    f"makes within-env std degenerate)."
                )

    def _unique_scalings(self) -> list[str]:
        return sorted({_source_scaling(s) for s in self.sources})

    def fit_impl(
        self,
        weather_dict: dict[str, np.ndarray],
        train_envs: list[str],
        all_envs: list[str] | None = None,
    ) -> None:
        """Fit enviromic features.

        Parameters
        ----------
        weather_dict : ``{env_name: array(n_days, n_vars)}`` for all
            environments in the dataset.
        train_envs : environment names in the training split.
        all_envs : all environment names in the dataset.  Required when
            ``fit_scope="all"``.
        """
        if self.fit_scope == "all":
            if all_envs is None:
                raise ValueError(
                    "all_envs is required when fit_scope='all'."
                )
            fit_envs = sorted(set(all_envs))
        else:
            fit_envs = sorted(set(train_envs))

        self._env_order = fit_envs
        self._env_to_idx = {e: i for i, e in enumerate(fit_envs)}

        # Aggregate weather for the fitting set (scaling-independent).
        aggregation = self._get_aggregation()
        fit_weather = {e: weather_dict[e] for e in fit_envs}
        self._agg_features = aggregate_weather(fit_weather, aggregation)

        # Per-scaling kernel state.
        for scaling in self._unique_scalings():
            X_raw = np.stack([self._agg_features[e] for e in fit_envs])
            scaler = build_scaler(scaling)
            scaler.fit(X_raw)
            X = scaler.transform(X_raw)

            state: dict = {"scaler": scaler}

            needs_kernel = any(
                s["type"] in self._KERNEL_TYPES
                and _source_scaling(s) == scaling
                for s in self.sources
            )
            if needs_kernel:
                kernel_type = self._get_kernel_type(scaling)
                gamma = self._get_gamma(scaling)
                K = compute_kernel(X, kernel=kernel_type, gamma=gamma)
                if self.center_kernel:
                    K_used, col_means, grand_mean = center_kernel_train(K)
                else:
                    K_used = K
                    col_means = np.zeros(K.shape[0], dtype=np.float64)
                    grand_mean = 0.0
                state["K_centered"] = K_used
                state["col_means"] = col_means
                state["grand_mean"] = grand_mean
                if any(
                    s["type"] in self._EIGEN_TYPES
                    and _source_scaling(s) == scaling
                    for s in self.sources
                ):
                    # Always filter numerical noise. R-reference parity for the
                    # transductive (BGLR/GBLUP) path flows through
                    # raw_kernel() — unfiltered, with BGLR's own tolD
                    # applied downstream — not through this eigenbasis.
                    # eps=0 under fit_scope='all' kept spurious near-zero
                    # modes that defeat the rank check and bloat cache
                    # and compute.
                    eig_eps = 1e-10
                    V, D = eigen_decompose(K_used, eps=eig_eps)
                    state["V"] = V
                    state["D"] = D
                else:
                    state["V"] = None
                    state["D"] = None
            else:
                state["K_centered"] = None
                state["col_means"] = None
                state["grand_mean"] = None
                state["V"] = None
                state["D"] = None

            self._state[scaling] = state

    def transform_impl(
        self,
        sample_envs: list[str],
        split_name: str,
        weather_dict: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Transform environment names into feature arrays.

        Parameters
        ----------
        sample_envs : per-sample environment list (may have duplicates).
        split_name : "train", "val", or "test".
        weather_dict : needed for non-fitting environments when
            ``fit_scope="train"`` and ``split_name != "train"``.
        """
        result = {}

        for src in self.sources:
            name = src["name"]
            stype = src["type"]
            scaling = _source_scaling(src)
            n_comp, n_comp_max = read_component_params(src)

            if stype == "raw_weather":
                result[name] = self._transform_raw(
                    sample_envs, scaling, weather_dict,
                )

            elif stype == "enviromic_relmat":
                result[name] = self._transform_eigen(
                    sample_envs, scaling,
                    n_comp, n_comp_max, weather_dict,
                )

            elif stype == "enviromic_relmat_rows":
                result[name] = self._transform_rows(
                    sample_envs, scaling, weather_dict,
                )

        return result

    # ── BaseProcessor lifecycle ────────────────────────────────────

    def _load_weather_values(
        self, csv_path: str, all_envs: list[str],
    ) -> dict[str, np.ndarray]:
        """Per-env weather value matrix ``{env: array(n_rows, n_vars)}`` for
        the configured axis.

        ``axis='dap'`` reads the native daily grid (today's behaviour).
        ``axis='agdd'`` collapses dup-AGDD days first: it returns the deduped
        clean-AGDD weather values from the shared axis table, sliced/reordered
        to ``weather_vars`` (the table loads every column for the GDD cumsum,
        so subset selection is by name). Only the values matter for the
        aggregate — the time coordinate is discarded either way.
        """
        from .weather import load_weather_csv

        if self.axis == "dap":
            wd, _ = load_weather_csv(
                csv_path, envs=all_envs, weather_vars=self.weather_vars,
                # Not part of the FPCA reference models (enviromic is
                # enabled in no fpca config); pinned to the historical
                # mode so this processor is untouched by the change.
                missing_values="ffill",
                dtype=np.float32,
            )
            return {env: vals for env, (_, vals) in wd.items()}

        from .axis_source import build_axis_table

        table = build_axis_table(
            csv_path, gdd_cfg=self.gdd_cfg, envs=all_envs, dedup=self.dedup,
            missing_values="ffill",
            dtype=np.float32,
        )
        if self.weather_vars is None:
            col_idx = list(range(len(table.var_names)))
        else:
            col_idx = [table.var_names.index(n) for n in self.weather_vars]
        out: dict[str, np.ndarray] = {}
        for env in all_envs:
            _, values = table.env_axis(env, "agdd")
            out[env] = values[:, col_idx].astype(np.float32)
        return out

    def fit(self, ctx: OrchestratorContext) -> None:
        """Load weather (all envs in the dataset), pull train envs +
        all envs from `ctx`, delegate to `fit_impl`. Stash the weather
        dict so `transform()` can reuse it without re-reading the CSV.
        """
        all_envs = ctx.all_envs()
        self._weather_values = self._load_weather_values(
            ctx.proc_config["csv_path"], all_envs,
        )
        train_envs = ctx.envs_for_split("train")
        self.fit_impl(self._weather_values, train_envs, all_envs)

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """Pull split's envs from `ctx`, delegate to `transform_impl`.
        Lazily loads the weather in predict mode (after
        `load_fitted`, where `fit` was skipped).
        """
        idxs = ctx.split_indices.get(split, [])
        if not idxs:
            return None
        if getattr(self, "_weather_values", None) is None:
            all_envs = ctx.all_envs()
            self._weather_values = self._load_weather_values(
                ctx.proc_config["csv_path"], all_envs,
            )
        envs = ctx.envs_for_split(split)
        return self.transform_impl(envs, split, self._weather_values)

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Recompute via `compute_cache_key` from cached weather dict."""
        train_envs = ctx.envs_for_split("train")
        all_envs = ctx.all_envs()
        return self.compute_cache_key(
            self._weather_values, train_envs, all_envs,
        )

    @property
    def feature_dims(self) -> dict[str, int]:
        """Output dimensions per source. Available after fit().

        Reflects the actually-used width per D1 (see
        ``GenomicFeatureBuilder.feature_dims`` for semantics).
        """
        dims = {}
        for src in self.sources:
            stype = src["type"]
            scaling = _source_scaling(src)
            state = self._state[scaling]
            n_comp, n_comp_max = read_component_params(src)

            if stype == "raw_weather":
                sample_agg = next(iter(self._agg_features.values()))
                dims[src["name"]] = len(sample_agg)
            elif stype == "enviromic_relmat":
                _, D_m = slice_components(
                    state["V"], state["D"],
                    n_components=n_comp,
                    n_components_max=n_comp_max,
                )
                dims[src["name"]] = len(D_m)
            elif stype == "enviromic_relmat_rows":
                dims[src["name"]] = len(self._env_order)
        return dims

    # ── Cache: save / load fitted state ──────────────────────────

    def compute_cache_key(
        self,
        weather_dict: dict[str, np.ndarray],
        train_envs: list[str],
        all_envs: list[str] | None = None,
    ) -> str:
        """Content-addressed cache key for the fitted state.

        Hashes: source config (name/type/aggregation/kernel/gamma/scaling
        — rank-slice fields excluded), center_kernel, fit_scope,
        weather_vars/axis/dedup/gdd, the fit env list, and weather
        curve content for the fitting set.
        """
        if self.fit_scope == "all":
            if all_envs is None:
                raise ValueError(
                    "all_envs is required when fit_scope='all'."
                )
            fit_envs = sorted(set(all_envs))
        else:
            fit_envs = sorted(set(train_envs))

        h = hashlib.sha256()
        # v5 applies only to fit_scope='all' entries: their v4 eigen
        # spectra were computed with eps=0 (numerical-noise modes kept)
        # and must not be served. Train-scope entries are bit-identical
        # under both versions and stay warm on v4.
        h.update(
            b"enviromic_v5" if self.fit_scope == "all" else b"enviromic_v4"
        )
        src_repr = json.dumps(
            [
                {
                    "name": s["name"],
                    "type": s["type"],
                    "aggregation": s.get("aggregation", "mean"),
                    "kernel": s.get("kernel", "linear"),
                    "gamma": s.get("gamma"),
                    "scaling": _source_scaling(s),
                }
                for s in self.sources
            ],
            sort_keys=True,
        ).encode()
        h.update(src_repr)
        h.update(f"|fit_scope={self.fit_scope}".encode())
        h.update(f"|center_kernel={self.center_kernel}".encode())
        h.update(b"|weather_vars=")
        h.update(json.dumps(self.weather_vars, sort_keys=True).encode())
        # Always tag the axis name so every entry is self-describing on disk
        # (DAP included). dedup/gdd only shape the AGDD grid, so they're folded
        # in only for the derived axis; the substituted AGDD curves already
        # alter the weather-content hash below.
        h.update(b"|axis=")
        h.update(self.axis.encode())
        if self.axis != "dap":
            h.update(f"|dedup={self.dedup}".encode())
            h.update(b"|gdd=")
            h.update(json.dumps(self.gdd_cfg, sort_keys=True).encode())
        h.update(b"|envs=")
        h.update("|".join(fit_envs).encode())
        h.update(b"|weather=")
        for env in fit_envs:
            curves = np.ascontiguousarray(weather_dict[env], dtype=np.float64)
            h.update(env.encode())
            h.update(b":")
            h.update(curves.tobytes())
            h.update(b";")
        return h.hexdigest()

    def save_cache(self, cache_dir: str, cache_key: str) -> None:
        """Persist fitted state to ``cache_dir/enviromic_<key[:16]>.npz``."""
        if self._env_order is None:
            raise RuntimeError("Cannot save_cache before fit().")

        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"enviromic_{cache_key[:16]}.npz")

        # Aligned arrays (env_order is the canonical row order).
        agg_matrix = np.stack(
            [self._agg_features[e] for e in self._env_order]
        )

        payload: dict[str, np.ndarray] = {
            "env_order": np.array(self._env_order, dtype=object),
            "agg_matrix": agg_matrix,
            "config_json": np.array(
                json.dumps(
                    {
                        "sources": self.sources,
                        "fit_scope": self.fit_scope,
                        "center_kernel": self.center_kernel,
                        "weather_vars": self.weather_vars,
                        "scalings": self._unique_scalings(),
                    }
                ),
                dtype=object,
            ),
        }
        for scaling, state in self._state.items():
            prefix = f"scaling[{scaling}]"
            for k, v in state["scaler"].state_dict().items():
                payload[f"{prefix}/scaler.{k}"] = v
            if state["K_centered"] is not None:
                payload[f"{prefix}/K_centered"] = state["K_centered"]
                payload[f"{prefix}/col_means"] = state["col_means"]
                payload[f"{prefix}/grand_mean"] = np.asarray(state["grand_mean"])
            if state["V"] is not None:
                payload[f"{prefix}/V"] = state["V"]
                payload[f"{prefix}/D"] = state["D"]

        tmp_path = path + f".tmp.{os.getpid()}.npz"
        np.savez(tmp_path, **payload)
        try:
            os.replace(tmp_path, path)
        except FileNotFoundError:
            pass
        logger.info("Enviromic cache saved: %s", path)

    def load_fitted(self, cache_dir: str, cache_key: str) -> None:
        """Restore fitted state from ``cache_dir/enviromic_<key[:16]>.npz``.

        Validates that the cached config (sources, fit_scope) matches the
        current instance; raises ``ValueError`` on mismatch.
        """
        path = os.path.join(cache_dir, f"enviromic_{cache_key[:16]}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Enviromic cache not found: {path}")

        data = np.load(path, allow_pickle=True)
        saved = json.loads(str(data["config_json"].item()))
        if (
            saved.get("sources") != self.sources
            or saved.get("fit_scope") != self.fit_scope
            or saved["center_kernel"] != self.center_kernel
            or saved["weather_vars"] != self.weather_vars
        ):
            raise ValueError(
                f"Enviromic config mismatch: cache has {saved}, "
                f"current has sources={self.sources}, "
                f"fit_scope={self.fit_scope}, "
                f"center_kernel={self.center_kernel}, "
                f"weather_vars={self.weather_vars}"
            )

        self._env_order = [str(e) for e in data["env_order"]]
        self._env_to_idx = {e: i for i, e in enumerate(self._env_order)}
        agg_matrix = data["agg_matrix"]
        self._agg_features = {
            e: agg_matrix[i] for i, e in enumerate(self._env_order)
        }

        self._state = {}
        for scaling in saved.get("scalings", ["none"]):
            prefix = f"scaling[{scaling}]"
            scaler = build_scaler(scaling)
            scaler_state = {
                k.split(".", 1)[1]: data[f"{prefix}/{k}"]
                for k in (
                    fname[len(prefix) + 1:]
                    for fname in data.files
                    if fname.startswith(f"{prefix}/scaler.")
                )
            }
            if scaler_state:
                scaler.load_state(scaler_state)

            state: dict = {"scaler": scaler}
            for k in ("K_centered", "col_means", "V", "D"):
                key = f"{prefix}/{k}"
                state[k] = data[key] if key in data.files else None
            gm_key = f"{prefix}/grand_mean"
            state["grand_mean"] = (
                float(data[gm_key]) if gm_key in data.files else None
            )
            self._state[scaling] = state

        logger.info("Enviromic cache loaded: %s", path)

    # ── Private helpers ──────────────────────────────────────────

    def _get_aggregation(self) -> str:
        aggs = {s.get("aggregation", "mean") for s in self.sources}
        if len(aggs) > 1:
            raise ValueError(
                f"All sources must use the same aggregation, got {aggs}"
            )
        return aggs.pop()

    def _get_kernel_type(self, scaling: str | None = None) -> str:
        kernels = {
            s.get("kernel", "linear")
            for s in self.sources
            if s["type"] in self._KERNEL_TYPES
            and (scaling is None or _source_scaling(s) == scaling)
        }
        if len(kernels) > 1:
            raise ValueError(
                f"All kernel sources sharing scaling={scaling!r} must use "
                f"the same kernel, got {kernels}"
            )
        return kernels.pop() if kernels else "linear"

    def _get_gamma(self, scaling: str | None = None) -> float | None:
        gammas = {
            s.get("gamma")
            for s in self.sources
            if s["type"] in self._KERNEL_TYPES
            and (scaling is None or _source_scaling(s) == scaling)
        }
        gammas.discard(None)
        if len(gammas) > 1:
            raise ValueError(
                f"All kernel sources sharing scaling={scaling!r} must use "
                f"the same gamma, got {gammas}"
            )
        return gammas.pop() if gammas else None

    def _scaled_fit_aggregates(self, scaling: str) -> np.ndarray:
        """Recompute scaled aggregates for the fitting set under ``scaling``.

        Cheap: the scaler is already fitted; this is just one transform call.
        """
        X_raw = np.stack(
            [self._agg_features[e] for e in self._env_order]
        )
        return self._state[scaling]["scaler"].transform(X_raw)

    def _scaled_aggregates(
        self, sample_envs: list[str], scaling: str,
        weather_dict: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        """Build per-sample scaled aggregates (handles non-fit envs)."""
        aggregation = self._get_aggregation()
        rows = []
        for env in sample_envs:
            if env in self._agg_features:
                rows.append(self._agg_features[env])
            elif weather_dict is not None and env in weather_dict:
                agg = aggregate_weather({env: weather_dict[env]}, aggregation)
                rows.append(agg[env])
            else:
                raise ValueError(
                    f"Environment {env!r} not in fitted envs or weather_dict."
                )
        X_raw = np.stack(rows)
        return self._state[scaling]["scaler"].transform(X_raw)

    def _is_in_fitting_set(self, sample_envs: list[str]) -> bool:
        return all(e in self._env_to_idx for e in sample_envs)

    def _transform_raw(
        self,
        sample_envs: list[str],
        scaling: str,
        weather_dict: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        return self._scaled_aggregates(sample_envs, scaling, weather_dict)

    def _transform_eigen(
        self,
        sample_envs: list[str],
        scaling: str,
        n_components: int | None,
        n_components_max: int | None,
        weather_dict: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        state = self._state[scaling]
        V_m, D_m = slice_components(
            state["V"], state["D"],
            n_components=n_components,
            n_components_max=n_components_max,
        )

        if self.fit_scope == "all" or self._is_in_fitting_set(sample_envs):
            Z = project_eigen_train(V_m, D_m)
            indices = [self._env_to_idx[e] for e in sample_envs]
            return Z[indices]
        else:
            X_new = self._scaled_aggregates(sample_envs, scaling, weather_dict)
            X_fit = self._scaled_fit_aggregates(scaling)
            kernel_type = self._get_kernel_type(scaling)
            gamma = self._get_gamma(scaling)
            K_cross = compute_kernel(
                X_new, X_fit, kernel=kernel_type, gamma=gamma
            )
            K_for_proj = (
                center_kernel_cross(
                    K_cross, state["col_means"], state["grand_mean"]
                )
                if self.center_kernel
                else K_cross
            )
            return project_eigen_cross(K_for_proj, V_m, D_m)

    def _transform_rows(
        self,
        sample_envs: list[str],
        scaling: str,
        weather_dict: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        state = self._state[scaling]
        if self.fit_scope == "all" or self._is_in_fitting_set(sample_envs):
            indices = [self._env_to_idx[e] for e in sample_envs]
            return state["K_centered"][indices]
        else:
            X_new = self._scaled_aggregates(sample_envs, scaling, weather_dict)
            X_fit = self._scaled_fit_aggregates(scaling)
            kernel_type = self._get_kernel_type(scaling)
            gamma = self._get_gamma(scaling)
            K_cross = compute_kernel(
                X_new, X_fit, kernel=kernel_type, gamma=gamma
            )
            return (
                center_kernel_cross(
                    K_cross, state["col_means"], state["grand_mean"]
                )
                if self.center_kernel
                else K_cross
            )
