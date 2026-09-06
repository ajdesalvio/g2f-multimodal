"""Phenomic feature extraction from VI FPC score-derived relationship matrices.

Builds phenomic similarity kernels from VI FPC scores. Each source can
independently choose how to scale the input scores before kernel
construction:

- ``scaling: none`` (default) — pass-through, raw FPC scores.
- ``scaling: global_zscore`` — per-column z-score using train-only stats.
- ``scaling: within_env_zscore`` — per-env per-column z-score. Each
  env's mean/std are computed from all dataset samples in that env,
  including held-out rows (transductive). This matches the R reference's
  ``safe_scale_matrix`` per-env protocol byte-for-byte but uses
  held-out env data to scale held-out env features.

Two sources with different scalings get separate fitted state (their own
kernel, eigendecomposition, and centering offsets); two sources sharing
the same scaling reuse state and only differ in slicing (eigen vs rows,
n_components). Train-only fitting is preserved for ``none`` and
``global_zscore``; ``within_env_zscore`` is transductive by design.

``fit_scope`` controls which samples are used to build the phenomic
kernel:

- ``"train"`` (default): train-only kernel fit; val/test rows are
  produced via cross-kernel projection onto the train eigenbasis.
  Held-out env's VI scores never enter the eigendecomposition.
- ``"all"``: kernel fit on the union of train+val+test FPC scores.
  The held-out env's rows are part of the eigendecomposition, so
  the eigenbasis is informed by held-out data. Use only when this
  is intentional and the input features are guaranteed to be
  pre-yield observable.

Depends on VIFPCAProcessor output — must run after it in the execution
order.
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
from .score_scalers import (
    _SCALING_REGISTRY,
    build_scaler,
    requires_envs,
    slug_code_for,
)

logger = logging.getLogger(__name__)


def _source_scaling(src: dict) -> str:
    """Read the scaling field from a source dict, defaulting to 'none'."""
    return src.get("scaling", "none")


@register_processor
class PhenomicFeatureBuilder(BaseProcessor):
    """Phenomic feature extraction from VI FPC scores.

    Parameters
    ----------
    sources : list of dicts. Each source has:
        - ``name``: str — output key (e.g., "phenomic_relmat_we").
        - ``type``: ``"phenomic_relmat"`` (eigen-projected) or
          ``"phenomic_relmat_rows"`` (raw centered kernel rows).
        - ``n_components`` / ``n_components_max``: D1 two-knob contract,
          post-load slice.
        - ``scaling``: ``"none"`` (default), ``"global_zscore"``, or
          ``"within_env_zscore"`` — see module docstring.
    kernel : ``"linear"`` or ``"rbf"``.
    gamma : RBF bandwidth. Ignored for linear kernel.
    """

    _EIGEN_TYPES = {"phenomic_relmat"}
    _ROW_TYPES = {"phenomic_relmat_rows"}

    # ── BaseProcessor metadata ──────────────────────────────────────
    name: ClassVar[str] = "phenomic"
    priority: ClassVar[int] = 2
    requires: ClassVar[tuple[str, ...]] = ("vi_fpca",)
    has_eigen_sources: ClassVar[bool] = True
    eigen_source_types: ClassVar[frozenset[str]] = frozenset(
        {"phenomic_relmat"}
    )

    _ALLOWED_FIT_SCOPES = ("train", "all")

    @classmethod
    def from_config(cls, config: dict) -> "PhenomicFeatureBuilder":
        """Hydra-config factory used by the orchestrator."""
        return cls(
            sources=config.get("sources", []),
            kernel=config.get("kernel", "linear"),
            gamma=config.get("gamma"),
            center_kernel=config.get("center_kernel", False),
            fit_scope=config.get("fit_scope", "train"),
        )

    def __init__(
        self,
        sources: list[dict],
        kernel: str = "linear",
        gamma: float | None = None,
        center_kernel: bool = False,
        fit_scope: str = "train",
    ):
        self.sources = sources
        self.kernel = kernel
        self.gamma = gamma
        # When False, the kernel is eigen-decomposed in its raw form
        # (no double-centering, no cross-kernel centering offsets) —
        # K_P / K_W / GRMs are passed straight to RKHS terms.
        # True is the textbook kernel-PCA recipe (removes the constant
        # mode the regression intercept absorbs); the shipped default
        # is False (reference parity).
        self.center_kernel = bool(center_kernel)
        if fit_scope not in self._ALLOWED_FIT_SCOPES:
            raise ValueError(
                f"fit_scope={fit_scope!r}: must be one of "
                f"{list(self._ALLOWED_FIT_SCOPES)}."
            )
        self.fit_scope = fit_scope
        self._validate_sources()

        # Per-scaling state — keys are scaling config names; each value
        # is a dict with train_scores_scaled, K_centered, V, D, col_means,
        # grand_mean, and a fitted scaler instance.  When
        # center_kernel=False, ``K_centered`` holds the raw kernel and
        # col_means/grand_mean are zero-arrays (kept for cache schema
        # uniformity).
        self._state: dict[str, dict] = {}

    def _validate_sources(self):
        valid_types = self._EIGEN_TYPES | self._ROW_TYPES
        names = set()
        for src in self.sources:
            if src["type"] not in valid_types:
                raise ValueError(
                    f"Unknown phenomic source type: {src['type']!r}. "
                    f"Valid: {sorted(valid_types)}"
                )
            if src["name"] in names:
                raise ValueError(
                    f"Duplicate source name: {src['name']!r}"
                )
            names.add(src["name"])
            scaling = _source_scaling(src)
            if scaling not in _SCALING_REGISTRY:
                raise ValueError(
                    f"Unknown scaling {scaling!r} on source "
                    f"{src['name']!r}. Valid: {sorted(_SCALING_REGISTRY)}."
                )

    def _unique_scalings(self) -> list[str]:
        return sorted({_source_scaling(s) for s in self.sources})

    @property
    def needs_envs(self) -> bool:
        """True iff any configured source requires env labels."""
        return any(requires_envs(s) for s in self._unique_scalings())

    def _needs_union(self) -> bool:
        """True when fit_impl/cache_key require the union of all splits.

        ``fit_scope='all'`` always needs the union. ``within_env_zscore``
        needs it for per-env stats. Both conditions can hold together.
        """
        return self.fit_scope == "all" or self.needs_envs

    def fit_impl(
        self,
        train_fpc_scores: np.ndarray,
        train_envs: list[str] | None = None,
        all_fpc_scores: np.ndarray | None = None,
        all_envs: list[str] | None = None,
    ) -> None:
        """Fit per-scaling kernel state from FPC scores.

        Parameters
        ----------
        train_fpc_scores : ``(n_train, D)`` array of train-split FPC scores.
        train_envs : per-sample env labels for ``train_fpc_scores``.
            Required when any source uses ``within_env_zscore``.
        all_fpc_scores : ``(n_all, D)`` array stacking train+val+test
            FPC scores. Required when any source uses
            ``within_env_zscore`` (per-env stats come from this union)
            **or** when ``fit_scope='all'`` (kernel is built on this
            union; mirrors the R reference's full N×N KP).
        all_envs : per-sample env labels aligned with ``all_fpc_scores``.
            Required when any source uses ``within_env_zscore``.
        """
        if self.needs_envs:
            if train_envs is None or all_fpc_scores is None or all_envs is None:
                raise ValueError(
                    "PhenomicFeatureBuilder.fit requires train_envs, "
                    "all_fpc_scores, and all_envs when any source uses "
                    "within_env_zscore."
                )
        if self.fit_scope == "all" and all_fpc_scores is None:
            raise ValueError(
                "PhenomicFeatureBuilder.fit requires all_fpc_scores "
                "when fit_scope='all'."
            )

        if self.fit_scope == "all":
            fit_scores = all_fpc_scores
            fit_envs = all_envs
        else:
            fit_scores = train_fpc_scores
            fit_envs = train_envs

        for scaling in self._unique_scalings():
            scaler = build_scaler(scaling)
            if scaling == "none":
                scaler.fit(fit_scores)
                scaled_fit = fit_scores
            elif scaling == "global_zscore":
                # Stats from the fit set (train under 'train' scope, all
                # under 'all' scope) applied to every split at transform.
                scaler.fit(fit_scores)
                scaled_fit = scaler.transform(fit_scores)
            elif scaling == "within_env_zscore":
                # Per-env stats from the union of all dataset samples in
                # each env (transductive, identical under both scopes).
                # Same scaling then re-applied per-split at transform
                # time via env lookup.
                scaler.fit(all_fpc_scores, all_envs)
                scaled_fit = scaler.transform(fit_scores, fit_envs)
            else:
                raise AssertionError(f"unhandled scaling {scaling!r}")

            K_train = compute_kernel(
                scaled_fit,
                kernel=self.kernel,
                gamma=self.gamma,
            )
            if self.center_kernel:
                K_used, col_means, grand_mean = center_kernel_train(K_train)
            else:
                # Skip double-centering — eigen-decompose the raw kernel.
                # Uniform col_means/grand_mean kept as schema placeholders
                # so the cache layout stays identical across both modes.
                K_used = K_train
                col_means = np.zeros(K_train.shape[0], dtype=np.float64)
                grand_mean = 0.0

            # Field name retained as ``train_scores_scaled`` for cache
            # schema stability. Under ``fit_scope='all'`` it actually
            # holds the full union's scaled scores, which is the matrix
            # the cross-kernel projection in transform_impl is built
            # against.
            state = {
                "scaler": scaler,
                "train_scores_scaled": scaled_fit,
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
                # fit_scope='all' kept every strictly-positive mode,
                # i.e. ~N/2 spurious ~1e-17 eigenvalues that defeat the
                # rank check and bloat cache and compute.
                eig_eps = 1e-10
                V, D = eigen_decompose(K_used, eps=eig_eps)
                state["V"] = V
                state["D"] = D

            self._state[scaling] = state

    def transform_impl(
        self,
        fpc_scores: np.ndarray,
        split_name: str,
        sample_envs: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Transform FPC scores into phenomic features for one split.

        Parameters
        ----------
        fpc_scores : ``(n_samples, D)`` array of this split's FPC scores.
        split_name : ``"train"``, ``"val"``, or ``"test"`` (advisory; the
            in-fitting-set check now drives the index-vs-cross-kernel
            branch).
        sample_envs : per-sample env labels. Required when any source
            uses ``within_env_zscore``.
        """
        if self.needs_envs and sample_envs is None:
            raise ValueError(
                "PhenomicFeatureBuilder.transform requires sample_envs "
                "when any source uses within_env_zscore."
            )

        result = {}

        for src in self.sources:
            name = src["name"]
            stype = src["type"]
            scaling = _source_scaling(src)
            n_comp, n_comp_max = read_component_params(src)
            state = self._state[scaling]

            scaler = state["scaler"]
            if scaling == "none":
                scaled = np.asarray(fpc_scores, dtype=np.float64)
            elif scaling == "global_zscore":
                scaled = scaler.transform(fpc_scores)
            elif scaling == "within_env_zscore":
                scaled = scaler.transform(fpc_scores, sample_envs)
            else:
                raise AssertionError(f"unhandled scaling {scaling!r}")

            if stype == "phenomic_relmat":
                V_m, D_m = slice_components(
                    state["V"], state["D"],
                    n_components=n_comp,
                    n_components_max=n_comp_max,
                )
                # Train-set short-circuit: when scaled FPC scores match
                # the fitted train_scores_scaled bit-for-bit, we can
                # return the precomputed projection directly. Otherwise
                # build the cross-kernel against the fitted train set.
                if scaled.shape == state["train_scores_scaled"].shape and np.array_equal(
                    scaled, state["train_scores_scaled"]
                ):
                    result[name] = project_eigen_train(V_m, D_m)
                else:
                    K_cross = compute_kernel(
                        scaled,
                        state["train_scores_scaled"],
                        kernel=self.kernel,
                        gamma=self.gamma,
                    )
                    K_for_proj = (
                        center_kernel_cross(
                            K_cross, state["col_means"], state["grand_mean"]
                        )
                        if self.center_kernel
                        else K_cross
                    )
                    result[name] = project_eigen_cross(K_for_proj, V_m, D_m)

            elif stype == "phenomic_relmat_rows":
                if scaled.shape == state["train_scores_scaled"].shape and np.array_equal(
                    scaled, state["train_scores_scaled"]
                ):
                    result[name] = state["K_centered"].copy()
                else:
                    K_cross = compute_kernel(
                        scaled,
                        state["train_scores_scaled"],
                        kernel=self.kernel,
                        gamma=self.gamma,
                    )
                    result[name] = (
                        center_kernel_cross(
                            K_cross, state["col_means"], state["grand_mean"]
                        )
                        if self.center_kernel
                        else K_cross
                    )

        return result

    # ── BaseProcessor lifecycle ────────────────────────────────────

    def fit(self, ctx: OrchestratorContext) -> None:
        """Pull `vi_fpc_scores` from upstream (already attached to
        sample dicts by the vi_fpca processor) + env labels; delegate
        to `fit_impl`.
        """
        train_scores = ctx.stack_derived_feature(
            "vi_fpc_scores", "train",
        )
        train_envs = (
            ctx.envs_for_split("train") if self.needs_envs else None
        )
        if self._needs_union():
            all_scores_chunks = []
            all_envs: list[str] = []
            for split, _ in ctx.iter_splits():
                all_scores_chunks.append(
                    ctx.stack_derived_feature("vi_fpc_scores", split)
                )
                all_envs.extend(ctx.envs_for_split(split))
            all_scores = np.concatenate(all_scores_chunks, axis=0)
        else:
            all_scores = None
            all_envs = None
        self.fit_impl(
            train_scores,
            train_envs=train_envs,
            all_fpc_scores=all_scores,
            all_envs=all_envs,
        )

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """Stack the split's FPC scores + envs and delegate to
        `transform_impl`.
        """
        split_data = ctx.split_data(split)
        if not split_data:
            return None
        scores = ctx.stack_derived_feature("vi_fpc_scores", split)
        sample_envs = (
            ctx.envs_for_split(split) if self.needs_envs else None
        )
        return self.transform_impl(scores, split, sample_envs=sample_envs)

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Recompute via `compute_cache_key` on stacked train (+ all when
        ``fit_scope='all'`` or any source uses ``within_env_zscore``)
        scores from the orchestrator context.
        """
        train_scores = ctx.stack_derived_feature(
            "vi_fpc_scores", "train",
        )
        train_envs = (
            ctx.envs_for_split("train") if self.needs_envs else None
        )
        if self._needs_union():
            all_scores_chunks = []
            all_envs: list[str] = []
            for split, _ in ctx.iter_splits():
                all_scores_chunks.append(
                    ctx.stack_derived_feature("vi_fpc_scores", split)
                )
                all_envs.extend(ctx.envs_for_split(split))
            all_scores = np.concatenate(all_scores_chunks, axis=0)
        else:
            all_scores = None
            all_envs = None
        return self.compute_cache_key(
            train_scores,
            train_envs=train_envs,
            all_fpc_scores=all_scores,
            all_envs=all_envs,
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

            if stype == "phenomic_relmat":
                _, D_m = slice_components(
                    state["V"], state["D"],
                    n_components=n_comp,
                    n_components_max=n_comp_max,
                )
                dims[src["name"]] = len(D_m)
            elif stype == "phenomic_relmat_rows":
                dims[src["name"]] = state["train_scores_scaled"].shape[0]
        return dims

    def raw_kernel(
        self,
        ctx: OrchestratorContext,
        source_name: str,
        sample_indices: np.ndarray,
    ) -> np.ndarray:
        """Return obs-level raw K (no eigen filter) for the given rows.

        Requires ``fit_scope='all'`` so the kernel was built over the
        union of (train, val, test) FPC scores. ``K_centered`` in state
        is the obs-level kernel under the source's scaling (raw kernel
        when ``center_kernel=False``, the double-centered kernel
        otherwise — matching what is handed to BGLR in both modes).

        Parameters
        ----------
        sample_indices : metadata_df row indices defining the output
            row order. Shape ``(M,)``. Returned K has shape ``(M, M)``.
        """
        if self.fit_scope != "all":
            raise RuntimeError(
                f"PhenomicFeatureBuilder.raw_kernel requires "
                f"fit_scope='all'; got {self.fit_scope!r}."
            )
        src = next(s for s in self.sources if s["name"] == source_name)
        scaling = _source_scaling(src)
        state = self._state[scaling]
        K_full = state["K_centered"]
        all_md_idx: list[int] = []
        for split in ("train", "val", "test"):
            all_md_idx.extend(ctx.split_indices.get(split, []))
        md_to_k = {md_idx: k_row for k_row, md_idx in enumerate(all_md_idx)}
        k_rows = np.array(
            [md_to_k[int(i)] for i in sample_indices], dtype=np.int64,
        )
        return K_full[np.ix_(k_rows, k_rows)]

    # ── Cache: save / load fitted state ──────────────────────────

    def compute_cache_key(
        self,
        train_fpc_scores: np.ndarray,
        train_envs: list[str] | None = None,
        all_fpc_scores: np.ndarray | None = None,
        all_envs: list[str] | None = None,
    ) -> str:
        """Content-addressed cache key for the fitted state.

        Hashes: source config (name/type/scaling — D1 slice fields are
        excluded), kernel + gamma, fit_scope, train FPC scores. When
        ``fit_scope='all'`` or any source uses ``within_env_zscore``,
        also hashes the union FPC scores + the per-env composition
        (sorted unique envs and their sample counts) — both axes
        change the fit set or its per-env stats and must invalidate
        cache hits across folds.
        """
        h = hashlib.sha256()
        # v5 applies only to fit_scope='all' entries: their v4 eigen
        # spectra were computed with eps=0 (numerical-noise modes kept)
        # and must not be served. Train-scope entries are bit-identical
        # under both versions and stay warm on v4.
        h.update(
            b"phenomic_v5" if self.fit_scope == "all" else b"phenomic_v4"
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
        h.update(
            f"|kernel={self.kernel}|gamma={self.gamma}"
            f"|center_kernel={self.center_kernel}"
            f"|fit_scope={self.fit_scope}".encode()
        )
        h.update(b"|train_scores=")
        h.update(np.ascontiguousarray(train_fpc_scores, dtype=np.float64).tobytes())

        if self._needs_union():
            if all_fpc_scores is None:
                raise ValueError(
                    "compute_cache_key requires all_fpc_scores when "
                    "fit_scope='all' or any source uses within_env_zscore."
                )
            # Hash the union content + per-env composition.  This keeps
            # cache entries distinct across folds (different env
            # composition or different sample counts within an env both
            # change the per-env stats; ``fit_scope='all'`` additionally
            # makes the kernel itself depend on the union).
            h.update(b"|all_scores=")
            h.update(np.ascontiguousarray(all_fpc_scores, dtype=np.float64).tobytes())
            if all_envs is not None:
                envs_arr = np.asarray(all_envs)
                unique_envs = sorted(set(all_envs))
                counts = [int(np.sum(envs_arr == e)) for e in unique_envs]
                h.update(b"|envs=")
                h.update("|".join(f"{e}:{c}" for e, c in zip(unique_envs, counts)).encode())

        return h.hexdigest()

    def save_cache(self, cache_dir: str, cache_key: str) -> None:
        """Persist fitted state to ``cache_dir/phenomic_<key[:16]>.npz``."""
        if not self._state:
            raise RuntimeError("Cannot save_cache before fit().")

        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"phenomic_{cache_key[:16]}.npz")

        payload: dict[str, np.ndarray] = {
            "config_json": np.array(
                json.dumps(
                    {
                        "sources": self.sources,
                        "kernel": self.kernel,
                        "gamma": self.gamma,
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
            payload[f"{prefix}/train_scores_scaled"] = state["train_scores_scaled"]
            payload[f"{prefix}/col_means"] = state["col_means"]
            payload[f"{prefix}/grand_mean"] = np.asarray(state["grand_mean"])
            payload[f"{prefix}/K_centered"] = state["K_centered"]
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
        logger.info("Phenomic cache saved: %s", path)

    def load_fitted(self, cache_dir: str, cache_key: str) -> None:
        """Restore fitted state from ``cache_dir/phenomic_<key[:16]>.npz``.

        Validates that the cached config (sources + kernel + gamma)
        matches the current instance; raises ``ValueError`` on mismatch.
        """
        path = os.path.join(cache_dir, f"phenomic_{cache_key[:16]}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Phenomic cache not found: {path}")

        data = np.load(path, allow_pickle=True)
        saved = json.loads(str(data["config_json"].item()))
        if (
            saved.get("sources") != self.sources
            or saved.get("kernel") != self.kernel
            or saved.get("gamma") != self.gamma
            or saved["center_kernel"] != self.center_kernel
            or saved.get("fit_scope", "train") != self.fit_scope
        ):
            raise ValueError(
                f"Phenomic config mismatch: cache has {saved}, "
                f"current has sources={self.sources}, "
                f"kernel={self.kernel}, gamma={self.gamma}, "
                f"center_kernel={self.center_kernel}, "
                f"fit_scope={self.fit_scope}"
            )

        self._state = {}
        for scaling in saved.get("scalings", []):
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
                "train_scores_scaled": data[f"{prefix}/train_scores_scaled"],
                "col_means": data[f"{prefix}/col_means"],
                "grand_mean": float(data[f"{prefix}/grand_mean"]),
                "K_centered": data[f"{prefix}/K_centered"],
                "V": data[f"{prefix}/V"] if f"{prefix}/V" in data.files else None,
                "D": data[f"{prefix}/D"] if f"{prefix}/D" in data.files else None,
            }
            self._state[scaling] = state

        logger.info("Phenomic cache loaded: %s", path)


def phenomic_source_slug_code(src: dict) -> str:
    """Convenience wrapper: slug code for a phenomic source's scaling."""
    return slug_code_for(_source_scaling(src))
