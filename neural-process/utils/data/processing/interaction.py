"""Hadamard-kernel interactions over already-attached derived features.

Builds k-way interaction kernels by element-wise multiplication of
**raw** linear kernels over each component:

    K_int(i, j) = ∏_m  K_m(i, j)
                = ∏_m  ⟨φ_m(i), φ_m(j)⟩

For linear kernels this is identical to the linear kernel of the
**tensor-product feature**:

    φ_⊗(i) = ⊗_m φ_m(i)
    ⟨φ_⊗(i), φ_⊗(j)⟩  =  ∏_m  ⟨φ_m(i), φ_m(j)⟩

The per-component kernel is the **unnormalized** dot product
``X X^T`` — not ``X X^T / p`` as returned by
``kernels.compute_kernel(linear)``. This matches the recipe in the R reference's
relationship-matrix setup script, where
``KE_identity = Ze Ze'`` and ``KG_GE_A = KE_identity * KG_G_A`` carry
no per-component scalar divisor. ``compute_kernel``'s ``/p`` scaling
is the right convention for ``phenomic.py``'s ``KP = X X' / D``
(the R reference's ``tcrossprod(VI_all) / ncol(VI_all)``) but introduces a
spurious scalar in the Hadamard composition.

So one source covers any k-way interaction by listing K already-
attached derived-feature names. The R reference's within-env / weather-
modulated interaction kernels reduce to specific 2-component cases:

    KG_GE_A   = ⊙(genomic_add, env_onehot)
    KG_GE_D   = ⊙(genomic_dom, env_onehot)
    KP_PE     = ⊙(phenomic_relmat, env_onehot)
    KE_W      ≈ ⊙(weather_fpc_scores)              (k=1, main effect)
    KG_GE_AW  = ⊙(genomic_add, weather_fpc_scores)
    KG_GE_DW  = ⊙(genomic_dom, weather_fpc_scores)
    KP_PW     = ⊙(phenomic_relmat, weather_fpc_scores)

Three-way interactions (e.g. GxExW) are expressible as
``components: [genomic_add, env_onehot, weather_fpc_scores]`` — the
processor's complexity is invariant in k.

Construction strategy
---------------------
This v1 always builds the **kernel-matrix path**: compute each
component's pairwise kernel, take their Hadamard product, eigen-
decompose. Memory is O(n²) per intermediate kernel — fine for the G2F
LOO scale (n ≈ 10,109; each kernel is ~80 MB). The tensor-feature
path (per-sample outer product → SVD) can be added later as an
``auto`` heuristic for the case where ``∏_m d_m << n`` and all
components use linear kernels.

For now: linear component kernels only. The Hadamard product is PSD
(Schur product theorem) so the kernel is mathematically valid for
any kernel choice per component, but the tensor-product identity only
holds for linear kernels — keeping it linear for v1 keeps the math
clean and the auto-construction trivial.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import ClassVar

import numpy as np

from .base import BaseProcessor, OrchestratorContext
from .kernels import (
    center_kernel_cross,
    center_kernel_train,
    eigen_decompose,
    project_eigen_cross,
    project_eigen_train,
    read_component_params,
    slice_components,
)
from .registry import register_processor

logger = logging.getLogger(__name__)


# ── fit_scope policy ──
#
# Each interaction source declares `fit_scope: 'train' | 'all'`
# (default `'train'`). Under `'all'` the Hadamard kernel and its
# eigenbasis are fit over the full row set (train+val+test); under
# `'train'` they are fit on train rows only and held-out rows are
# cross-projected at transform time.
#
# Both modes are leakage-safe in the BGLR-transductive setting (the
# kernel never sees `y_test`). Project-wide rationale: FPCA bases stay
# train-only (basis-inductive), but the kernels built from those FPC
# scores may pool the full row set.

_ALLOWED_FIT_SCOPES: frozenset[str] = frozenset({"train", "all"})


@register_processor
class InteractionKernelBuilder(BaseProcessor):
    """Hadamard-of-kernels processor over already-attached features.

    Each source declares a list of upstream derived-feature names; the
    processor stacks each split's per-sample features for every
    component, builds a linear kernel per component, takes their
    Hadamard product, optionally double-centers, eigen-decomposes,
    and projects to ``n_components`` (or ``n_components_max``).

    Parameters
    ----------
    sources : list of dicts, each with:
        - ``name``: str — output key in `derived_features`.
        - ``type``: ``"interaction_relmat"`` (eigen-projected) or
          ``"interaction_relmat_rows"`` (raw centered kernel rows).
        - ``components``: list of str — names of upstream
          derived-feature keys to combine. Length ≥ 1; k=1 collapses
          to a main-effect kernel of one feature.
        - ``n_components`` / ``n_components_max``: D1 two-knob slice;
          mutually exclusive.
        - ``fit_scope``: ``"train"`` (default) or ``"all"``. When
          ``"all"``, the eigenbasis is fit on the union of every split
          (train+val+test); held-out rows live in the eigenbasis. When
          ``"train"``, the eigenbasis is fit on train rows only and
          held-out rows are cross-projected at transform time. Both
          modes are leakage-safe in the BGLR-transductive setting
          (the kernel never sees ``y_test``). ``"all"`` is rejected
          on ``interaction_relmat_rows`` because per-sample row width
          would silently grow from n_train to N.
    center_kernel : bool, default False. When False the Hadamard
        kernel is eigen-decomposed in raw form (no double-centering,
        no cross-kernel offsets) — BGLR-parity. Slug suffix: ``_uc``.
    """

    _EIGEN_TYPES = frozenset({"interaction_relmat"})
    _ROW_TYPES = frozenset({"interaction_relmat_rows"})

    # ── BaseProcessor metadata ──────────────────────────────────────
    name: ClassVar[str] = "interaction"
    # Runs after every processor that emits derived_features used as
    # components (vi_fpca, phenomic, weather_fpca, genomic, enviromic,
    # metadata_features) and after raw_weather / weather_concat (which
    # don't produce derived_features but mutate sample fields). The
    # interaction processor reads only derived_features, so its
    # priority needs only to be > every upstream provider's.
    # Priority 10: kept above raw_vi (8) / weather_concat (9) so it still
    # runs last in the pipeline.
    priority: ClassVar[int] = 10
    has_eigen_sources: ClassVar[bool] = True
    eigen_source_types: ClassVar[frozenset[str]] = frozenset(
        {"interaction_relmat"}
    )

    @classmethod
    def from_config(cls, config: dict) -> "InteractionKernelBuilder":
        """Hydra-config factory used by the orchestrator."""
        return cls(
            sources=config.get("sources", []),
            center_kernel=config.get("center_kernel", False),
        )

    def __init__(
        self,
        sources: list[dict],
        center_kernel: bool = False,
    ):
        self.sources = sources
        self.center_kernel = bool(center_kernel)
        self._validate_sources()
        # Per-source fitted state: fit_components (per-component
        # features sized for the source's fit_scope), col_means,
        # grand_mean, fit_scope, fit_split_offsets, V, D, and (for
        # row-type sources only) K_used.
        self._state: dict[str, dict] = {}

    def _validate_sources(self) -> None:
        valid_types = self._EIGEN_TYPES | self._ROW_TYPES
        names = set()
        for src in self.sources:
            for required in ("name", "type", "components"):
                if required not in src:
                    raise ValueError(
                        f"interaction source missing {required!r}: "
                        f"{src!r}."
                    )
            if src["type"] not in valid_types:
                raise ValueError(
                    f"Unknown interaction source type: "
                    f"{src['type']!r}. Valid: {sorted(valid_types)}."
                )
            if src["name"] in names:
                raise ValueError(
                    f"Duplicate source name: {src['name']!r}."
                )
            names.add(src["name"])
            comps = list(src["components"])
            if not comps:
                raise ValueError(
                    f"interaction source {src['name']!r} needs at "
                    f"least one component, got empty list."
                )
            if len(set(comps)) != len(comps):
                raise ValueError(
                    f"interaction source {src['name']!r} has "
                    f"duplicate components: {comps}."
                )
            # ── fit_scope (per-source) ────────────────────────────
            scope = src.setdefault("fit_scope", "train")
            if scope not in _ALLOWED_FIT_SCOPES:
                raise ValueError(
                    f"interaction source {src['name']!r}: "
                    f"fit_scope={scope!r} not allowed. "
                    f"Valid: {sorted(_ALLOWED_FIT_SCOPES)}."
                )
            if scope == "all" and src["type"] in self._ROW_TYPES:
                # Reject `interaction_relmat_rows` + `all` — the per-
                # sample row width grows from n_train to N, silently
                # changing downstream model architecture. Lift only
                # after the DL path is audited.
                raise ValueError(
                    f"interaction source {src['name']!r}: "
                    f"fit_scope='all' is not supported with "
                    f"type={src['type']!r}. Per-sample row width "
                    f"would silently grow from n_train to N. Use "
                    f"type='interaction_relmat' (eigen-projected) "
                    f"or keep fit_scope='train'."
                )

    def raw_kernel(
        self,
        ctx: OrchestratorContext,
        source_name: str,
        sample_indices: np.ndarray,
    ) -> np.ndarray:
        """Return the obs-level Hadamard of raw component kernels.

        Looks up each component's producing processor via
        ``ctx.feature_to_processor`` and calls its own ``raw_kernel``,
        then takes the element-wise product. The output K is in the
        row order defined by ``sample_indices``.
        """
        src = next(s for s in self.sources if s["name"] == source_name)
        registry = ctx.feature_to_processor
        K: np.ndarray | None = None
        for comp in src["components"]:
            producer = registry.get(comp)
            if producer is None:
                raise KeyError(
                    f"InteractionKernelBuilder.raw_kernel: source "
                    f"{source_name!r} component {comp!r} has no "
                    f"registered producer. The orchestrator builds "
                    f"`feature_to_processor` by scanning each "
                    f"processor's `sources` list — make sure the "
                    f"upstream processor is enabled and declares this "
                    f"name."
                )
            K_m = producer.raw_kernel(ctx, comp, sample_indices)
            K = K_m if K is None else K * K_m
        assert K is not None
        return K

    # ── BaseProcessor lifecycle ────────────────────────────────────

    def fit(self, ctx: OrchestratorContext) -> None:
        """Build per-source fit kernel + eigen-decomposition.

        For each source, stacks the fit-scope per-component features
        (n_fit × d_m for each m, where n_fit = n_train when
        fit_scope='train' and n_fit = N (train+val+test) when
        fit_scope='all'), computes the Hadamard product of linear
        component kernels, optionally double-centers, and eigen-
        decomposes if any source needs eigen projection.

        Under `fit_scope='all'` the per-source state additionally
        records `fit_split_offsets` so that `transform()` can slice
        `V[lo:hi] · √D` directly at projection time (D6) instead of
        recomputing a cross-kernel against rows that are already in the
        fit set.
        """
        for src in self.sources:
            scope = src["fit_scope"]
            fit_components, fit_split_offsets = self._stack_for_scope(
                ctx, src["components"], scope,
            )
            K_fit = self._hadamard_kernel(fit_components)
            if self.center_kernel:
                K_used, col_means, grand_mean = center_kernel_train(K_fit)
            else:
                K_used = K_fit
                col_means = np.zeros(K_fit.shape[0], dtype=np.float64)
                grand_mean = 0.0

            state: dict = {
                "fit_components": fit_components,
                "fit_scope": scope,
                "fit_split_offsets": fit_split_offsets,
                "col_means": col_means,
                "grand_mean": grand_mean,
                "V": None,
                "D": None,
            }
            if src["type"] in self._EIGEN_TYPES:
                # Always filter numerical noise. R-reference parity for the
                # transductive (BGLR/GBLUP) path flows through
                # raw_kernel() — unfiltered, with BGLR's own tolD applied
                # downstream — not through this eigenbasis. eps=0 under
                # fit_scope='all' kept ~N/2 spurious ~1e-17 modes that
                # defeat the rank check and bloat cache and compute.
                eig_eps = 1e-10
                V, D = eigen_decompose(K_used, eps=eig_eps)
                state["V"] = V
                state["D"] = D
                # Eigen path doesn't need K_used after this; drop it to
                # save ~820 MB per source under fit_scope='all'.
            else:
                # `interaction_relmat_rows` returns kernel rows at
                # transform — needs K_used. (Validator already rejects
                # rows + scope='all', so this is always n_train×n_train.)
                state["K_used"] = K_used
            self._state[src["name"]] = state
            logger.info(
                "interaction source %r: components=%s, scope=%s, "
                "K shape=%s, rank≈%d",
                src["name"], src["components"], scope, K_used.shape,
                len(state["D"]) if state["D"] is not None else -1,
            )

    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """Project the named split's per-component features onto each
        source's fitted eigenspace (eigen sources) or build kernel
        rows against the fitted set (rows sources).

        Under ``fit_scope='all'``, the requested split's rows are
        already in the fit set; this method short-circuits to a row-
        slice of ``V · √D`` instead of recomputing a cross-kernel
        (D6). Validators
        reject ``interaction_relmat_rows`` + ``fit_scope='all'``, so
        that combination never reaches this code path.
        """
        if not ctx.split_data(split):
            return None
        result: dict[str, np.ndarray] = {}
        for src in self.sources:
            state = self._state[src["name"]]
            n_comp, n_comp_max = read_component_params(src)

            # ── fit_scope='all': D6 row-slice short-circuit ──────
            if state["fit_scope"] == "all":
                offsets = state["fit_split_offsets"]
                if split not in offsets:
                    # Split was empty at fit time; nothing to project.
                    continue
                lo, hi = offsets[split]
                # Defensive size check: split sizes must be stable
                # between fit and transform under scope='all'.
                n_split_now = len(ctx.split_data(split))
                if hi - lo != n_split_now:
                    raise RuntimeError(
                        f"interaction source {src['name']!r}: "
                        f"split={split!r} had {hi - lo} samples at "
                        f"fit time but {n_split_now} now. Cache may "
                        f"be stale."
                    )
                if src["type"] in self._EIGEN_TYPES:
                    V_m, D_m = slice_components(
                        state["V"], state["D"],
                        n_components=n_comp,
                        n_components_max=n_comp_max,
                    )
                    result[src["name"]] = (
                        V_m[lo:hi] * np.sqrt(D_m)[np.newaxis, :]
                    )
                # rows + scope='all' is rejected by validator.
                continue

            # ── fit_scope='train': existing cross-kernel path ────
            split_components = self._stack_components(
                ctx, src["components"], split,
            )
            if src["type"] in self._EIGEN_TYPES:
                V_m, D_m = slice_components(
                    state["V"], state["D"],
                    n_components=n_comp,
                    n_components_max=n_comp_max,
                )
                # Train short-circuit: when split_components match the
                # fitted fit_components bit-for-bit, return the
                # precomputed projection.
                if self._is_fit_bit_match(
                    split_components, state["fit_components"],
                ):
                    result[src["name"]] = project_eigen_train(V_m, D_m)
                else:
                    K_cross = self._hadamard_cross_kernel(
                        split_components, state["fit_components"],
                    )
                    K_for_proj = (
                        center_kernel_cross(
                            K_cross, state["col_means"],
                            state["grand_mean"],
                        )
                        if self.center_kernel
                        else K_cross
                    )
                    result[src["name"]] = project_eigen_cross(
                        K_for_proj, V_m, D_m,
                    )
            else:  # interaction_relmat_rows
                if self._is_fit_bit_match(
                    split_components, state["fit_components"],
                ):
                    result[src["name"]] = state["K_used"].copy()
                else:
                    K_cross = self._hadamard_cross_kernel(
                        split_components, state["fit_components"],
                    )
                    result[src["name"]] = (
                        center_kernel_cross(
                            K_cross, state["col_means"],
                            state["grand_mean"],
                        )
                        if self.center_kernel
                        else K_cross
                    )
        return result

    @property
    def feature_dims(self) -> dict[str, int]:
        dims = {}
        for src in self.sources:
            state = self._state[src["name"]]
            n_comp, n_comp_max = read_component_params(src)
            if src["type"] in self._EIGEN_TYPES:
                _, D_m = slice_components(
                    state["V"], state["D"],
                    n_components=n_comp,
                    n_components_max=n_comp_max,
                )
                dims[src["name"]] = len(D_m)
            else:
                dims[src["name"]] = state["K_used"].shape[0]
        return dims

    # ── Cache: save / load fitted state ────────────────────────────

    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Hash source config + center_kernel + per-component fit-scope
        feature stacks. Hashing the actual feature bytes propagates any
        upstream cache change automatically. Per-source ``fit_scope``
        is folded into both the JSON header and the feature-bytes scope
        used for hashing.
        """
        h = hashlib.sha256()
        # v2: per-component kernel switched from /p-normalized
        # `compute_kernel(linear)` to raw `X X^T` for R-reference parity
        # (`KE_identity = Ze Ze'` etc.). Bumping the sentinel evicts
        # any v1 cache entries still on disk — their stored eigen
        # spectra would be off by per-component scalar prefactors.
        # v3 applies only when a source pools fit_scope='all': those v2
        # eigen spectra were computed with eps=0 (numerical-noise modes
        # kept) and must not be served. All-train entries are
        # bit-identical under both versions and stay warm on v2.
        h.update(
            b"interaction_v3"
            if any(s["fit_scope"] == "all" for s in self.sources)
            else b"interaction_v2"
        )
        src_repr = json.dumps(
            [
                {
                    "name": s["name"],
                    "type": s["type"],
                    "components": list(s["components"]),
                    "fit_scope": s["fit_scope"],
                }
                for s in self.sources
            ],
            sort_keys=True,
        ).encode()
        h.update(src_repr)
        h.update(f"|center_kernel={self.center_kernel}".encode())
        for src in self.sources:
            scope = src["fit_scope"]
            for comp in src["components"]:
                feat = self._stack_for_cache_hash(ctx, comp, scope)
                h.update(
                    f"|{src['name']}.{comp}@{scope}=".encode()
                )
                h.update(
                    np.ascontiguousarray(
                        feat, dtype=np.float64,
                    ).tobytes()
                )
        return h.hexdigest()

    @staticmethod
    def _stack_for_cache_hash(
        ctx: OrchestratorContext, comp: str, scope: str,
    ) -> np.ndarray:
        """Stack a component for cache-key hashing in canonical order.

        Mirrors the iteration order used by ``_stack_for_scope`` so that
        a fit-time scope change shows up in the hash via different bytes
        rather than via a different label alone.
        """
        if scope == "train":
            return ctx.stack_derived_feature(comp, "train")
        parts = []
        for split_name, _ in ctx.iter_splits():
            parts.append(ctx.stack_derived_feature(comp, split_name))
        return np.concatenate(parts, axis=0)

    def save_cache(self, cache_dir: str, key: str) -> None:
        """Persist per-source fitted state to
        ``cache_dir/interaction_<key[:16]>.npz``.

        ``K_used`` is persisted only for row-type sources (eigen sources
        don't need it after fit). ``fit_scope`` and ``fit_split_offsets``
        are always persisted so ``load_fitted`` can reconstruct the
        transform-time short-circuit.
        """
        if not self._state:
            raise RuntimeError("Cannot save_cache before fit().")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"interaction_{key[:16]}.npz")

        payload: dict[str, np.ndarray] = {
            "config_json": np.array(
                json.dumps({
                    "sources": self.sources,
                    "center_kernel": self.center_kernel,
                }),
                dtype=object,
            ),
        }
        for src_name, state in self._state.items():
            prefix = f"src[{src_name}]"
            if "K_used" in state:
                payload[f"{prefix}/K_used"] = state["K_used"]
            payload[f"{prefix}/col_means"] = state["col_means"]
            payload[f"{prefix}/grand_mean"] = np.asarray(
                state["grand_mean"]
            )
            payload[f"{prefix}/fit_scope"] = np.array(
                state["fit_scope"], dtype=object,
            )
            payload[f"{prefix}/fit_split_offsets"] = np.array(
                json.dumps(state["fit_split_offsets"]),
                dtype=object,
            )
            if state["V"] is not None:
                payload[f"{prefix}/V"] = state["V"]
                payload[f"{prefix}/D"] = state["D"]
            for comp_name, comp_feat in state["fit_components"].items():
                payload[f"{prefix}/comp[{comp_name}]"] = comp_feat
        tmp_path = path + f".tmp.{os.getpid()}.npz"
        np.savez(tmp_path, **payload)
        try:
            os.replace(tmp_path, path)
        except FileNotFoundError:
            pass
        logger.info("interaction cache saved: %s", path)

    def load_fitted(self, cache_dir: str, key: str) -> None:
        path = os.path.join(cache_dir, f"interaction_{key[:16]}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"interaction cache not found: {path}"
            )
        data = np.load(path, allow_pickle=True)
        saved = json.loads(str(data["config_json"].item()))
        if (
            saved.get("sources") != self.sources
            or saved["center_kernel"] != self.center_kernel
        ):
            raise ValueError(
                f"interaction config mismatch: cache has {saved}, "
                f"current has sources={self.sources}, "
                f"center_kernel={self.center_kernel}."
            )
        self._state = {}
        for src in self.sources:
            src_name = src["name"]
            prefix = f"src[{src_name}]"
            fit_components = {}
            for comp in src["components"]:
                fit_components[comp] = data[f"{prefix}/comp[{comp}]"]
            offsets_raw = data[f"{prefix}/fit_split_offsets"].item()
            offsets = {
                k: tuple(v) for k, v in json.loads(offsets_raw).items()
            }
            state: dict = {
                "fit_components": fit_components,
                "fit_scope": str(data[f"{prefix}/fit_scope"].item()),
                "fit_split_offsets": offsets,
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
            if f"{prefix}/K_used" in data.files:
                state["K_used"] = data[f"{prefix}/K_used"]
            self._state[src_name] = state
        logger.info("interaction cache loaded: %s", path)

    # ── Private helpers ────────────────────────────────────────────

    @staticmethod
    def _stack_components(
        ctx: OrchestratorContext,
        component_names: list[str],
        split: str,
    ) -> dict[str, np.ndarray]:
        """Stack each component's per-sample feature for one split.

        Returns ``{component_name: (n_samples, d_m) ndarray}``.
        """
        out: dict[str, np.ndarray] = {}
        for comp in component_names:
            try:
                out[comp] = ctx.stack_derived_feature(comp, split)
            except KeyError as exc:
                raise KeyError(
                    f"interaction source needs component {comp!r} but "
                    f"it isn't attached as a derived_feature on the "
                    f"{split} split. Make sure the upstream processor "
                    f"emitting it is enabled and runs at a lower "
                    f"priority. Original error: {exc}"
                ) from None
        return out

    @staticmethod
    def _stack_for_scope(
        ctx: OrchestratorContext,
        component_names: list[str],
        scope: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int]]]:
        """Stack components per the fit scope and report split offsets.

        Returns
        -------
        fit_components : ``{component_name: (n_fit, d_m) ndarray}``.
            ``n_fit = n_train`` for ``scope='train'``;
            ``n_fit = N`` (train+val+test) for ``scope='all'``.
        fit_split_offsets : ``{split: (lo, hi)}`` row-range map into the
            stacked arrays. Empty for ``scope='train'``; populated in
            canonical iter_splits order (train, val, test) for
            ``scope='all'``. Used by ``transform()`` to slice the eigen
            projection directly without a cross-kernel.
        """
        if scope == "train":
            comps = InteractionKernelBuilder._stack_components(
                ctx, component_names, "train",
            )
            return comps, {}

        # scope == "all": concat in canonical (train, val, test) order
        per_split: dict[str, dict[str, np.ndarray]] = {}
        for split_name, _ in ctx.iter_splits():
            per_split[split_name] = (
                InteractionKernelBuilder._stack_components(
                    ctx, component_names, split_name,
                )
            )

        # Build offsets and concatenate per component.
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for split_name in ("train", "val", "test"):
            if split_name not in per_split:
                continue
            n_split = next(iter(per_split[split_name].values())).shape[0]
            offsets[split_name] = (cursor, cursor + n_split)
            cursor += n_split

        fit_components: dict[str, np.ndarray] = {}
        for comp in component_names:
            parts = []
            for split_name in ("train", "val", "test"):
                if split_name in per_split:
                    parts.append(per_split[split_name][comp])
            fit_components[comp] = np.concatenate(parts, axis=0)
        return fit_components, offsets

    @staticmethod
    def _hadamard_kernel(
        components: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Build the Hadamard product of raw linear kernels over all
        components. Single component → one ``X X^T`` block.

        Uses unnormalized ``comp_feat @ comp_feat.T`` (no ``/p``) so
        that ``KE_identity = Ze Ze'`` and ``KG_GE_A = KE_identity *
        KG_G_A`` reproduce the R reference recipe exactly. See module
        docstring.
        """
        K = None
        for comp_feat in components.values():
            K_m = comp_feat @ comp_feat.T
            K = K_m if K is None else K * K_m
        assert K is not None  # validated at construction (≥1 component)
        return K

    @staticmethod
    def _hadamard_cross_kernel(
        new_components: dict[str, np.ndarray],
        train_components: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Cross-kernel: ⊙_m K_m(new_features_m, train_features_m).

        Both dicts must share the same component-name set (validated
        by `_stack_components` at fit + transform time). Same raw
        ``X Y^T`` convention as :meth:`_hadamard_kernel`.
        """
        K = None
        for comp_name, comp_new in new_components.items():
            comp_train = train_components[comp_name]
            K_m = comp_new @ comp_train.T
            K = K_m if K is None else K * K_m
        assert K is not None
        return K

    @staticmethod
    def _is_fit_bit_match(
        split_components: dict[str, np.ndarray],
        fit_components: dict[str, np.ndarray],
    ) -> bool:
        """True iff every component's split features bit-match the
        fitted features. Triggers the project_eigen_train short-circuit
        at transform time under fit_scope='train'.
        """
        for name, fit_feat in fit_components.items():
            split_feat = split_components.get(name)
            if split_feat is None:
                return False
            if split_feat.shape != fit_feat.shape:
                return False
            if not np.array_equal(split_feat, fit_feat):
                return False
        return True
