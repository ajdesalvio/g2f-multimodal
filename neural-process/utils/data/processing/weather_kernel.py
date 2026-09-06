"""Weather relationship-matrix processor — env-level kernel from weather FPC scores.

Builds an environment-similarity kernel ``K_W`` from
``weather_fpc_scores`` (the per-env FPC score vectors emitted by
``WeatherFPCAProcessor``) and exposes it as a per-sample feature via
env-broadcast::

    weather_score_matrix : (n_envs, n_vars * K)
    weather_score_matrix <- safe_scale_matrix(weather_score_matrix)
    K_W <- tcrossprod(weather_score_matrix) / ncol(weather_score_matrix)
    KE_W <- Ze %*% K_W %*% t(Ze)             # project to obs level

The processor stops at the env-level eigen projection ``V √D`` (or the
raw kernel rows for the ``weather_relmat_rows`` source). The
``Ze K_W Ze^T`` projection to observation level happens implicitly when
the orchestrator broadcasts each sample's env-row into its
``derived_features`` slot — two samples in the same env share the same
feature vector by construction.

``fit_scope``
-------------
``fit_scope`` controls the **K_W eigenbasis** fit set only. The FPC-score
scaler (z-score of the weather scores) ALWAYS uses train-env statistics —
never val/test — under both scopes, so the normalization can never leak
held-out-env information into K_W.

- ``"train"`` (default): the K_W eigenbasis is fit using only train
  envs' (scaled) weather FPC scores; non-train envs (val/test) are
  cross-projected onto the train basis at transform time. This is the
  leakage-safe shape required by the inductive policy — at deployment
  time we don't have future-year weather, so test envs must not
  influence the kernel fit.
- ``"all"``: the K_W eigenbasis pools all envs at fit time
  (`tcrossprod(scores) / ncol` over the full 19-env score matrix —
  BGLR-transductive-safe). The z-score stats are still train-only, so
  this is intentionally NOT byte-identical to the R reference, which
  normalizes with all-env stats.

Source types
------------
- ``weather_relmat``: per-sample feature equals the eigen-projection
  ``V[env_index] * sqrt(D)`` (sliced to ``n_components`` /
  ``n_components_max`` per the D1 contract). For BGLR the implicit
  kernel between two samples ``i, j`` is
  ``Z_i Z_j^T = V_W D_W V_W^T [env_i, env_j] = K_W[env_i, env_j]``,
  which is exactly ``KE_W[i, j]``.
- ``weather_relmat_rows``: per-sample feature equals the row of
  ``K_W`` corresponding to the sample's env (length = number of fit
  envs).

Per-source ``scaling``
----------------------
- ``"none"`` (default): pass weather_fpc_scores through unchanged.
- ``"global_zscore"``: per-column z-score whose mean/std are computed
  over the **train-env** rows only (regardless of ``fit_scope``), then
  applied to every env. Slug code: ``gz``.

Env-level features (one row per env) cannot use ``within_env_zscore`` —
within-env std is degenerate. ``__init__`` rejects it explicitly.

Processor-level ``center_kernel``
---------------------------------
- ``False`` (default): feed the raw kernel into eigendecomposition
  (BGLR/R-parity choice; passes the uncentered ``K_W`` straight into
  RKHS terms — R applies no kernel centering).
- ``True``: textbook double-centered kernel-PCA recipe.

Depends on ``WeatherFPCAProcessor`` (priority 3). The orchestrator
enforces this via ``cls.requires``.
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
from .score_scalers import build_scaler, slug_code_for

logger = logging.getLogger(__name__)


_ALLOWED_FIT_SCOPES = ("all", "train")


def _source_scaling(src: dict) -> str:
    return src.get("scaling", "none")


@register_processor
class WeatherKernelProcessor(BaseProcessor):
    """Env-level weather kernel ``K_W`` from upstream weather FPC scores.

    Parameters
    ----------
    sources : list of dicts. Each source has:
        - ``name``: str — output key referenced by
          ``regressor.feature_keys`` and ``interaction.components``.
        - ``type``: ``"weather_relmat"`` (eigen V·√D, env-broadcast)
          or ``"weather_relmat_rows"`` (raw kernel rows, env-broadcast).
        - ``n_components`` / ``n_components_max``: D1 two-knob slice
          (post-load; eigen sources only).
        - ``scaling``: ``"none"`` (default) or ``"global_zscore"``.
    center_kernel : bool, default ``False``. When ``False``, eigen-decompose
        the raw kernel without double-centering — BGLR-parity recipe.
    fit_scope : ``"train"`` (default) or ``"all"``. See module docstring.
    """

    _EIGEN_TYPES = frozenset({"weather_relmat"})
    _ROW_TYPES = frozenset({"weather_relmat_rows"})
    _ALLOWED_SCALINGS = ("none", "global_zscore")

    # ── BaseProcessor metadata ──────────────────────────────────────
    name: ClassVar[str] = "weather_kernel"
    priority: ClassVar[int] = 4
    requires: ClassVar[tuple[str, ...]] = ("weather_fpca",)
    has_eigen_sources: ClassVar[bool] = True
    eigen_source_types: ClassVar[frozenset[str]] = frozenset(
        {"weather_relmat"}
    )

    @classmethod
    def from_config(cls, config: dict) -> "WeatherKernelProcessor":
        """Hydra-config factory used by the orchestrator."""
        return cls(
            sources=config.get("sources", []),
            center_kernel=config.get("center_kernel", False),
            fit_scope=config.get("fit_scope", "train"),
        )

    def __init__(
        self,
        sources: list[dict],
        center_kernel: bool = False,
        fit_scope: str = "train",
    ):
        if fit_scope not in _ALLOWED_FIT_SCOPES:
            raise ValueError(
                f"fit_scope={fit_scope!r}: must be one of "
                f"{list(_ALLOWED_FIT_SCOPES)}"
            )
        self.sources = sources
        self.center_kernel = bool(center_kernel)
        self.fit_scope = fit_scope
        self._validate_sources()

        # Full env-level inputs (all envs the dataset knows about).
        # Used at transform time to cross-project envs that weren't in
        # the fitting set (under fit_scope='train').
        self._all_env_order: list[str] | None = None
        self._all_env_score_matrix: np.ndarray | None = None
        self._all_env_to_idx: dict[str, int] | None = None

        # Fit-set env order (the rows the eigenbasis was learned from).
        # Under fit_scope='all' equals _all_env_order; under
        # fit_scope='train' equals sorted(set(train_envs)).
        self._fit_env_order: list[str] | None = None
        self._fit_env_to_idx: dict[str, int] | None = None

        # Per-scaling state. Each value: scaler + K_centered + col_means
        # + grand_mean + (V, D) when an eigen source uses this scaling.
        self._state: dict[str, dict] = {}

    def _validate_sources(self):
        valid_types = self._EIGEN_TYPES | self._ROW_TYPES
        names = set()
        for src in self.sources:
            if src["type"] not in valid_types:
                raise ValueError(
                    f"Unknown weather_kernel source type: "
                    f"{src['type']!r}. Valid: {sorted(valid_types)}"
                )
            if src["name"] in names:
                raise ValueError(
                    f"Duplicate source name: {src['name']!r}"
                )
            names.add(src["name"])
            scaling = _source_scaling(src)
            if scaling not in self._ALLOWED_SCALINGS:
                raise ValueError(
                    f"scaling={scaling!r} on weather_kernel source "
                    f"{src['name']!r}: env-level features can only use "
                    f"{list(self._ALLOWED_SCALINGS)} (one row per env "
                    f"makes within-env std degenerate)."
                )

    def _unique_scalings(self) -> list[str]:
        return sorted({_source_scaling(s) for s in self.sources})

    # ── Helpers: env-level score reduction ──────────────────────────

    @staticmethod
    def _per_env_scores(
        per_sample_scores: np.ndarray,
        per_sample_envs: list[str],
    ) -> tuple[list[str], np.ndarray]:
        """Reduce per-sample env-broadcast scores to one row per env.

        Every sample in the same env carries the same vector by
        construction (``WeatherFPCAProcessor`` emits per-env scores
        broadcast to obs level), so we take the first occurrence per
        env; the production path trusts this invariant.

        Returns ``(env_order, env_score_matrix)`` with rows in
        sorted-env-name order.
        """
        if len(per_sample_envs) != per_sample_scores.shape[0]:
            raise ValueError(
                f"weather_kernel: env labels length "
                f"({len(per_sample_envs)}) does not match score rows "
                f"({per_sample_scores.shape[0]})."
            )
        seen: dict[str, int] = {}
        for i, e in enumerate(per_sample_envs):
            if e not in seen:
                seen[e] = i
        env_order = sorted(seen.keys())
        rows = np.stack(
            [per_sample_scores[seen[e]] for e in env_order]
        )
        return env_order, rows

    # ── Fit / transform implementation ──────────────────────────────

    def fit_impl(
        self,
        env_order: list[str],
        env_score_matrix: np.ndarray,
        train_envs: list[str],
    ) -> None:
        """Fit per-scaling kernel state from a per-env score matrix.

        Parameters
        ----------
        env_order : list of env names — the canonical row order of
            ``env_score_matrix`` covering ALL envs in the dataset
            (train + val + test). Used at transform time to look up
            non-fit envs' raw scores for cross-projection.
        env_score_matrix : ``(n_envs, D)`` array of weather FPC scores,
            one row per env, in ``env_order`` order.
        train_envs : env names used to fit the FPC-score scaler (always)
            and the K_W eigenbasis (under ``fit_scope='train'``). The
            scaler's z-score mean/std are computed from these envs only
            under both fit scopes — never val/test — so the normalization
            cannot leak held-out-env information. Under
            ``fit_scope='train'`` these envs additionally define the
            eigenbasis fit set (non-train envs are cross-projected at
            transform time); under ``fit_scope='all'`` the eigenbasis
            pools every env in ``env_order`` but the scaler still uses
            train only.
        """
        env_order = list(env_order)
        env_score_matrix = np.asarray(env_score_matrix, dtype=np.float64)
        if len(env_order) != env_score_matrix.shape[0]:
            raise ValueError(
                f"env_order length ({len(env_order)}) does not match "
                f"env_score_matrix rows ({env_score_matrix.shape[0]})."
            )

        self._all_env_order = env_order
        self._all_env_score_matrix = env_score_matrix
        self._all_env_to_idx = {e: i for i, e in enumerate(env_order)}

        # train_envs always drives the score scaler (see below), so it
        # must be valid under every fit_scope — validate it up front.
        unknown = [e for e in train_envs if e not in self._all_env_to_idx]
        if unknown:
            raise ValueError(
                f"weather_kernel.fit_impl: train_envs contains envs "
                f"not present in env_order: {sorted(set(unknown))}"
            )
        train_fit_envs = sorted(set(train_envs))
        if not train_fit_envs:
            raise ValueError(
                "weather_kernel.fit_impl: train_envs is empty; cannot "
                "fit the FPC-score scaler."
            )

        # The K_W eigenbasis fit set follows fit_scope: all envs pooled
        # (transductive K_W, reference-parity) or train envs only (inductive,
        # held-out envs cross-projected at transform time).
        fit_envs = list(env_order) if self.fit_scope == "all" else train_fit_envs

        self._fit_env_order = fit_envs
        self._fit_env_to_idx = {e: i for i, e in enumerate(fit_envs)}

        fit_indices = np.array(
            [self._all_env_to_idx[e] for e in fit_envs], dtype=np.int64
        )
        fit_score_matrix = env_score_matrix[fit_indices]

        # The score scaler ALWAYS derives its z-score mean/std from train
        # envs only — never val/test — regardless of fit_scope. Setting the
        # normalization stats from held-out-env scores would leak test
        # information into K_W, so the scaler stays train-only even when the
        # eigenbasis pools all envs under fit_scope='all'.
        scaler_indices = np.array(
            [self._all_env_to_idx[e] for e in train_fit_envs], dtype=np.int64
        )
        scaler_fit_matrix = env_score_matrix[scaler_indices]

        for scaling in self._unique_scalings():
            scaler = build_scaler(scaling)
            scaler.fit(scaler_fit_matrix)
            X_fit_scaled = scaler.transform(fit_score_matrix)

            K = compute_kernel(X_fit_scaled, kernel="linear")
            if self.center_kernel:
                K_used, col_means, grand_mean = center_kernel_train(K)
            else:
                K_used = K
                col_means = np.zeros(K.shape[0], dtype=np.float64)
                grand_mean = 0.0

            state: dict = {
                "scaler": scaler,
                "K_centered": K_used,
                "col_means": col_means,
                "grand_mean": grand_mean,
                "V": None,
                "D": None,
            }
            if any(
                s["type"] in self._EIGEN_TYPES
                and _source_scaling(s) == scaling
                for s in self.sources
            ):
                # Always filter numerical noise. R-reference parity for the
                # transductive (BGLR/GBLUP) path flows through
                # raw_kernel() — unfiltered, with BGLR's own tolD applied
                # downstream — not through this eigenbasis. eps=0 under
                # fit_scope='all' kept spurious near-zero modes that
                # defeat the rank check and bloat cache and compute.
                eig_eps = 1e-10
                V, D = eigen_decompose(K_used, eps=eig_eps)
                state["V"] = V
                state["D"] = D
            self._state[scaling] = state

    def _scaled_row_for_env(
        self, env: str, scaling: str,
    ) -> np.ndarray:
        """Return the scaled FPC-score row for one env (may be a non-fit env)."""
        if env not in self._all_env_to_idx:
            raise KeyError(
                f"Env not seen at fit time: {env!r}. "
                f"weather_kernel saw envs: {sorted(self._all_env_to_idx)}."
            )
        idx = self._all_env_to_idx[env]
        raw = self._all_env_score_matrix[idx : idx + 1]
        return self._state[scaling]["scaler"].transform(raw)

    def _scaled_fit_matrix(self, scaling: str) -> np.ndarray:
        """Return scaled FPC scores for the fit-set envs (n_fit, D)."""
        fit_indices = np.array(
            [self._all_env_to_idx[e] for e in self._fit_env_order],
            dtype=np.int64,
        )
        fit_raw = self._all_env_score_matrix[fit_indices]
        return self._state[scaling]["scaler"].transform(fit_raw)

    def transform_impl(
        self,
        sample_envs: list[str],
    ) -> dict[str, np.ndarray]:
        """Build per-sample features by env-broadcasting fitted state.

        Fit-set envs are looked up directly in ``V·√D`` (eigen) or the
        K_W rows (rows). Non-fit envs (under ``fit_scope='train'``) are
        cross-projected via ``compute_kernel(scaled_test, scaled_fit,
        'linear')`` → optional centering → ``project_eigen_cross``.
        """
        if self._fit_env_to_idx is None:
            raise RuntimeError(
                "WeatherKernelProcessor.transform called before fit."
            )

        # Group sample-env positions by whether the env was in the fit set.
        fit_positions: list[int] = []
        cross_positions: list[int] = []
        cross_envs: list[str] = []
        for pos, env in enumerate(sample_envs):
            if env in self._fit_env_to_idx:
                fit_positions.append(pos)
            else:
                cross_positions.append(pos)
                cross_envs.append(env)

        n = len(sample_envs)
        fit_idx_in_fit_order = np.array(
            [self._fit_env_to_idx[sample_envs[p]] for p in fit_positions],
            dtype=np.int64,
        )

        # Per-source output assembly.
        result: dict[str, np.ndarray] = {}
        cached_cross: dict[str, dict] = {}

        for src in self.sources:
            name = src["name"]
            stype = src["type"]
            scaling = _source_scaling(src)
            n_comp, n_comp_max = read_component_params(src)
            state = self._state[scaling]

            if stype == "weather_relmat":
                V_m, D_m = slice_components(
                    state["V"], state["D"],
                    n_components=n_comp,
                    n_components_max=n_comp_max,
                )
                Z_fit = project_eigen_train(V_m, D_m)  # (n_fit, m)
                width = Z_fit.shape[1]
                out = np.empty((n, width), dtype=Z_fit.dtype)
                if fit_positions:
                    out[fit_positions] = Z_fit[fit_idx_in_fit_order]
                if cross_positions:
                    Z_cross = self._cross_project_eigen(
                        cross_envs, scaling, V_m, D_m, cached_cross,
                    )
                    out[cross_positions] = Z_cross
                result[name] = out

            elif stype == "weather_relmat_rows":
                K_fit = state["K_centered"]  # (n_fit, n_fit)
                width = K_fit.shape[1]
                out = np.empty((n, width), dtype=K_fit.dtype)
                if fit_positions:
                    out[fit_positions] = K_fit[fit_idx_in_fit_order]
                if cross_positions:
                    K_cross_centered = self._cross_project_rows(
                        cross_envs, scaling, cached_cross,
                    )
                    out[cross_positions] = K_cross_centered
                result[name] = out

        return result

    def _cross_project_eigen(
        self,
        cross_envs: list[str],
        scaling: str,
        V_m: np.ndarray,
        D_m: np.ndarray,
        cache: dict[str, dict],
    ) -> np.ndarray:
        """Project non-fit envs onto the train eigenbasis."""
        K_cross_centered = self._cross_project_rows(
            cross_envs, scaling, cache,
        )
        return project_eigen_cross(K_cross_centered, V_m, D_m)

    def _cross_project_rows(
        self,
        cross_envs: list[str],
        scaling: str,
        cache: dict[str, dict],
    ) -> np.ndarray:
        """Compute the centered K[cross, fit] rows for non-fit envs."""
        slot = cache.setdefault(scaling, {})
        cache_key = tuple(cross_envs)
        if cache_key in slot:
            return slot[cache_key]
        state = self._state[scaling]
        X_cross = np.stack(
            [self._scaled_row_for_env(e, scaling)[0] for e in cross_envs]
        )
        X_fit = self._scaled_fit_matrix(scaling)
        K_cross = compute_kernel(X_cross, X_fit, kernel="linear")
        if self.center_kernel:
            K_cross_centered = center_kernel_cross(
                K_cross, state["col_means"], state["grand_mean"],
            )
        else:
            K_cross_centered = K_cross
        slot[cache_key] = K_cross_centered
        return K_cross_centered

    def raw_kernel(
        self,
        ctx: OrchestratorContext,
        source_name: str,
        sample_indices: np.ndarray,
    ) -> np.ndarray:
        """Return obs-level ``Z_e @ K_env @ Z_e^T`` for the given rows.

        Requires ``fit_scope='all'`` so ``K_centered`` is the full
        env-level kernel over every env that appears in the dataset.
        Obs-level expansion is done by ``np.ix_`` on the env-index
        vector (cheaper than materializing ``Z_e``).
        """
        if self.fit_scope != "all":
            raise RuntimeError(
                f"WeatherKernelProcessor.raw_kernel requires "
                f"fit_scope='all'; got {self.fit_scope!r}."
            )
        src = next(s for s in self.sources if s["name"] == source_name)
        scaling = _source_scaling(src)
        K_env = self._state[scaling]["K_centered"]
        env_series = ctx.env_year()
        sample_envs = env_series.loc[sample_indices].tolist()
        for env in sample_envs:
            if env not in self._all_env_to_idx:
                raise KeyError(
                    f"WeatherKernelProcessor.raw_kernel: env {env!r} "
                    f"not in fitted kernel."
                )
        env_idx = np.array(
            [self._all_env_to_idx[e] for e in sample_envs], dtype=np.int64,
        )
        return K_env[np.ix_(env_idx, env_idx)]

    # ── BaseProcessor lifecycle ─────────────────────────────────────

    def fit(self, ctx: OrchestratorContext) -> None:
        """Pull ``weather_fpc_scores`` for every sample, reduce to one
        row per env, and delegate to ``fit_impl`` together with the
        train-split env list (used when ``fit_scope='train'``).
        """
        all_scores_chunks = []
        all_envs: list[str] = []
        for split, _ in ctx.iter_splits():
            all_scores_chunks.append(
                ctx.stack_derived_feature("weather_fpc_scores", split)
            )
            all_envs.extend(ctx.envs_for_split(split))
        if not all_scores_chunks:
            raise RuntimeError(
                "weather_kernel.fit: no splits with weather_fpc_scores. "
                "Ensure weather_fpca is enabled and runs first."
            )
        all_scores = np.concatenate(all_scores_chunks, axis=0)
        env_order, env_score_matrix = self._per_env_scores(
            all_scores, all_envs,
        )
        train_envs = list(ctx.envs_for_split("train"))
        self.fit_impl(env_order, env_score_matrix, train_envs)

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """Look up each sample's env, emit broadcast features."""
        if not ctx.split_data(split):
            return None
        sample_envs = ctx.envs_for_split(split)
        return self.transform_impl(sample_envs)

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Recompute via ``compute_cache_key`` from per-env scores."""
        all_scores_chunks = []
        all_envs: list[str] = []
        for split, _ in ctx.iter_splits():
            all_scores_chunks.append(
                ctx.stack_derived_feature("weather_fpc_scores", split)
            )
            all_envs.extend(ctx.envs_for_split(split))
        all_scores = np.concatenate(all_scores_chunks, axis=0)
        env_order, env_score_matrix = self._per_env_scores(
            all_scores, all_envs,
        )
        train_envs = list(ctx.envs_for_split("train"))
        return self.compute_cache_key(env_order, env_score_matrix, train_envs)

    @property
    def feature_dims(self) -> dict[str, int]:
        """Per-source output width. Available after ``fit``."""
        dims: dict[str, int] = {}
        for src in self.sources:
            stype = src["type"]
            scaling = _source_scaling(src)
            state = self._state[scaling]
            n_comp, n_comp_max = read_component_params(src)

            if stype == "weather_relmat":
                _, D_m = slice_components(
                    state["V"], state["D"],
                    n_components=n_comp,
                    n_components_max=n_comp_max,
                )
                dims[src["name"]] = len(D_m)
            elif stype == "weather_relmat_rows":
                dims[src["name"]] = len(self._fit_env_order)
        return dims

    # ── Cache: save / load fitted state ────────────────────────────

    def compute_cache_key(
        self,
        env_order: list[str],
        env_score_matrix: np.ndarray,
        train_envs: list[str],
    ) -> str:
        """Content-addressed cache key for the fitted state.

        Hashes: source config (name/type/scaling — D1 slice fields
        excluded), center_kernel, fit_scope, full env order, the
        per-env score matrix content, and the train-env subset whenever
        the result depends on it. The train-env subset matters under
        ``fit_scope='train'`` (it defines the eigenbasis fit set) and
        whenever any source uses a non-``none`` scaling (the FPC-score
        scaler is always train-only, so its stats — and thus K_W — are
        per-fold even under ``fit_scope='all'``). Omitting it there would
        let LOEO folds collide on a single cache entry.
        """
        h = hashlib.sha256()
        # v4 applies only to fit_scope='all' entries: their v3 eigen
        # spectra were computed with eps=0 (numerical-noise modes kept)
        # and must not be served. Train-scope entries are bit-identical
        # under both versions and stay warm on v3.
        h.update(
            b"weather_kernel_v4"
            if self.fit_scope == "all"
            else b"weather_kernel_v3"
        )
        src_repr = json.dumps(
            [
                {
                    "name": s["name"],
                    "type": s["type"],
                    "scaling": _source_scaling(s),
                }
                for s in self.sources
            ],
            sort_keys=True,
        ).encode()
        h.update(src_repr)
        h.update(f"|center_kernel={self.center_kernel}".encode())
        h.update(f"|fit_scope={self.fit_scope}".encode())
        h.update(b"|envs=")
        h.update("|".join(env_order).encode())
        h.update(b"|scores=")
        h.update(
            np.ascontiguousarray(
                env_score_matrix, dtype=np.float64,
            ).tobytes()
        )
        scaler_is_train_dependent = any(
            _source_scaling(s) != "none" for s in self.sources
        )
        if self.fit_scope == "train" or scaler_is_train_dependent:
            h.update(b"|train_envs=")
            h.update("|".join(sorted(set(train_envs))).encode())
        return h.hexdigest()

    def save_cache(self, cache_dir: str, cache_key: str) -> None:
        """Persist fitted state to ``cache_dir/weather_kernel_<key[:16]>.npz``."""
        if self._all_env_order is None or self._fit_env_order is None:
            raise RuntimeError("Cannot save_cache before fit().")

        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(
            cache_dir, f"weather_kernel_{cache_key[:16]}.npz"
        )

        payload: dict[str, np.ndarray] = {
            "all_env_order": np.array(self._all_env_order, dtype=object),
            "fit_env_order": np.array(self._fit_env_order, dtype=object),
            "all_env_score_matrix": np.asarray(
                self._all_env_score_matrix, dtype=np.float64,
            ),
            "config_json": np.array(
                json.dumps(
                    {
                        "sources": self.sources,
                        "center_kernel": self.center_kernel,
                        "fit_scope": self.fit_scope,
                        "scalings": self._unique_scalings(),
                    }
                ),
                dtype=object,
            ),
        }
        for scaling, state in self._state.items():
            prefix = f"scaling[{scaling}]"
            payload[f"{prefix}/K_centered"] = state["K_centered"]
            payload[f"{prefix}/col_means"] = state["col_means"]
            payload[f"{prefix}/grand_mean"] = np.asarray(
                state["grand_mean"]
            )
            if state["V"] is not None:
                payload[f"{prefix}/V"] = state["V"]
                payload[f"{prefix}/D"] = state["D"]
            for k, v in state["scaler"].state_dict().items():
                payload[f"{prefix}/scaler.{k}"] = v

        tmp_path = path + f".tmp.{os.getpid()}.npz"
        np.savez(tmp_path, **payload)
        try:
            os.replace(tmp_path, path)
        except FileNotFoundError:
            pass
        logger.info("weather_kernel cache saved: %s", path)

    def load_fitted(self, cache_dir: str, cache_key: str) -> None:
        """Restore fitted state from cache."""
        path = os.path.join(
            cache_dir, f"weather_kernel_{cache_key[:16]}.npz"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"weather_kernel cache not found: {path}"
            )

        data = np.load(path, allow_pickle=True)
        saved = json.loads(str(data["config_json"].item()))
        if (
            saved.get("sources") != self.sources
            or saved["center_kernel"] != self.center_kernel
            or saved.get("fit_scope") != self.fit_scope
        ):
            raise ValueError(
                f"weather_kernel config mismatch: cache has {saved}, "
                f"current has sources={self.sources}, "
                f"center_kernel={self.center_kernel}, "
                f"fit_scope={self.fit_scope}"
            )

        self._all_env_order = [str(e) for e in data["all_env_order"]]
        self._all_env_to_idx = {
            e: i for i, e in enumerate(self._all_env_order)
        }
        self._all_env_score_matrix = np.asarray(
            data["all_env_score_matrix"], dtype=np.float64,
        )
        self._fit_env_order = [str(e) for e in data["fit_env_order"]]
        self._fit_env_to_idx = {
            e: i for i, e in enumerate(self._fit_env_order)
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
            state = {
                "scaler": scaler,
                "K_centered": data[f"{prefix}/K_centered"],
                "col_means": data[f"{prefix}/col_means"],
                "grand_mean": float(data[f"{prefix}/grand_mean"]),
                "V": (
                    data[f"{prefix}/V"]
                    if f"{prefix}/V" in data.files else None
                ),
                "D": (
                    data[f"{prefix}/D"]
                    if f"{prefix}/D" in data.files else None
                ),
            }
            self._state[scaling] = state

        logger.info("weather_kernel cache loaded: %s", path)


def weather_kernel_source_slug_code(src: dict) -> str:
    """Convenience wrapper: slug code for a weather_kernel source's scaling."""
    return slug_code_for(_source_scaling(src))
