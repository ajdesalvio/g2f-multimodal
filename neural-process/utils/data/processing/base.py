"""Base contract for feature processors.

`OrchestratorContext` snapshots dataset-level state passed into every
processor lifecycle method, hiding dataset internals behind helpers so
processors can't reach into private attributes.

`BaseProcessor` declares the lifecycle every processor implements
(`fit`, `transform`, `cache_key`, optional `save_cache` / `load_fitted`,
`feature_dims`) plus class-level metadata (`name`, `priority`,
`requires`, `produces_batch_fields`, `has_eigen_sources`,
`eigen_source_types`, `coverage_mask`) that replaces the orchestrator's
hardcoded name-based lookups.

Together these let the orchestrator dispatch via a uniform virtual-method
call instead of an if/elif chain on processor names.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

if TYPE_CHECKING:
    import pandas as pd

    from utils.data.dataset import G2FDataset

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorContext:
    """Per-call snapshot of orchestrator-level state.

    The orchestrator builds one `OrchestratorContext` and reuses it across
    every enabled processor's lifecycle methods. Processors reach in via
    the typed helper methods rather than into `dataset` internals — this
    keeps the dataset-side API a stable contract surface.

    Parameters
    ----------
    dataset : G2FDataset
        The dataset under construction. Internal attributes (`_train`,
        `_val`, `_test`, `metadata_df`, `split_indices`) are read by the
        helpers below; processors should call those helpers rather than
        accessing the attributes directly.
    proc_config : dict
        This processor's config block — the value of
        `dataset.processing.<name>` from the composed Hydra cfg.
    cache_dir : str | None
        Root directory for persisted fitted state. `None` disables
        caching entirely.
    upstream : dict[str, BaseProcessor]
        Already-fitted upstream processors, keyed by `proc.name`. The
        orchestrator populates this from `cls.requires` before calling
        `fit` / `transform`. Empty when no requirements declared.
    feature_to_processor : dict[str, BaseProcessor]
        Source-name → producing-processor registry. The orchestrator
        scans every enabled processor's ``sources`` list and maps each
        ``source["name"]`` to its producer. Used by
        `InteractionKernelBuilder.raw_kernel` to look up component
        producers without hardcoding processor names. Empty when the
        orchestrator wasn't the constructor (e.g. a context built
        manually).
    """

    dataset: "G2FDataset"
    proc_config: dict
    cache_dir: str | None
    upstream: dict[str, "BaseProcessor"] = field(default_factory=dict)
    feature_to_processor: dict[str, "BaseProcessor"] = field(default_factory=dict)

    # ── Dataset accessors ───────────────────────────────────────────

    @property
    def metadata_df(self) -> "pd.DataFrame":
        return self.dataset.metadata_df

    @property
    def split_indices(self) -> dict[str, list[int]]:
        return self.dataset.split_indices

    # ── VI-FPCA fit-mask (D3) ───────────────────────────────────────

    @property
    def fit_mask(self) -> "np.ndarray | None":
        """Boolean FIT-set mask over ``metadata_df`` rows, or ``None``.

        Set by the role-based split path (``cv_2_1`` / ``cv_0_00``) to the
        ``FIT``-role rows so ``vi_fpca`` fits the basis on FIT only and
        projects the in-train-split ``OBSERVE`` rows. ``None`` (the
        default, and always for ``env_year_loo``) means the FIT set equals
        the train split — ``vi_fpca`` then behaves byte-identically to
        today. Weather/genomic ignore this entirely (D4): their fit set
        follows from the train-split environments.
        """
        return getattr(self.dataset, "fit_mask", None)

    def fit_flags_for_split(self, split: str) -> "np.ndarray | None":
        """The ``fit_mask`` restricted to one split, in that split's
        sample order (or ``None`` when there is no fit-mask / the split is
        empty). ``split_indices`` are ``metadata_df`` positions, so this
        indexes the positional ``fit_mask`` directly.
        """
        fit_mask = self.fit_mask
        if fit_mask is None:
            return None
        idxs = self.split_indices.get(split, [])
        if not idxs:
            return None
        return fit_mask[np.asarray(idxs, dtype=int)]

    def split_data(self, split: str) -> list[dict]:
        """Per-sample dict list for one split, `[]` when the split is empty."""
        return getattr(self.dataset, f"_{split}", []) or []

    def iter_splits(self):
        """Yield `(split_name, split_data)` for every non-empty split."""
        for name in ("train", "val", "test"):
            data = self.split_data(name)
            if data:
                yield name, data

    # ── Env / pedigree label helpers ────────────────────────────────

    def env_year(self) -> "pd.Series":
        """Canonical `Env.Year` identifier series indexed like `metadata_df`.

        Weather and enviromic CSVs (and `cv/folds.py`) key environments by
        the `Env.Year` string; dataset metadata stores them in separate
        columns, so this helper combines them once for orchestrator-wide
        consistency.
        """
        meta = self.metadata_df
        return meta["Env"].astype(str) + "." + meta["Year"].astype(str)

    def envs_for_split(self, split: str) -> list[str]:
        """Per-sample env labels for a split (length matches sample count)."""
        idxs = self.split_indices.get(split, [])
        if not idxs:
            return []
        return self.env_year().loc[idxs].tolist()

    def unique_envs_for_split(self, split: str) -> list[str]:
        """Distinct env labels appearing in a split."""
        idxs = self.split_indices.get(split, [])
        if not idxs:
            return []
        return self.env_year().loc[idxs].unique().tolist()

    def all_envs(self) -> list[str]:
        """Distinct env labels across the entire dataset."""
        return self.env_year().unique().tolist()

    def vi_names(self) -> list[str]:
        """Canonical VI ordering for ``channels[:, j]``.

        Captured at ingestion (alphabetic by VI name) and persisted in
        the dataset's read cache. Used by ``VIFPCAProcessor.vi_subset``
        to map names → tensor column indices for the post-load slice.
        """
        return list(getattr(self.dataset, "vi_names", []))

    def pedigrees_for_split(self, split: str) -> list[str]:
        """Per-sample pedigree labels for a split."""
        idxs = self.split_indices.get(split, [])
        if not idxs:
            return []
        return self.metadata_df.loc[idxs, "Pedigree"].tolist()

    def all_pedigrees(self) -> list[str]:
        """Per-sample pedigree labels concatenated across train+val+test in
        the canonical (train, val, test) order. Used by processors that
        fit on the union of all observed pedigrees (e.g. genomic with
        `fit_scope=all`).
        """
        all_idxs: list[int] = []
        for split in ("train", "val", "test"):
            all_idxs.extend(self.split_indices.get(split, []))
        if not all_idxs:
            return []
        return self.metadata_df.loc[all_idxs, "Pedigree"].tolist()

    # ── Derived-feature helpers ─────────────────────────────────────

    def stack_derived_feature(self, name: str, split: str) -> np.ndarray:
        """Stack a per-sample derived feature into an `(N, D)` matrix.

        Reads `sample["derived_features"][name]` from each sample in the
        named split. Tensors are converted to numpy. Raises `KeyError` if
        the feature wasn't attached for any sample (callers can check
        whether the upstream processor was actually run first).
        """
        rows = []
        for sample in self.split_data(split):
            vec = sample["derived_features"][name]
            if isinstance(vec, torch.Tensor):
                vec = vec.numpy()
            rows.append(vec)
        return np.stack(rows)


class BaseProcessor(ABC):
    """Lifecycle contract every feature processor implements.

    Subclasses declare class-level metadata (`name`, `priority`, optional
    `requires` / `produces_batch_fields` / `has_eigen_sources` /
    `eigen_source_types` / `coverage_mask`) and implement four lifecycle
    methods (`__init__`, `fit`, `transform`, `cache_key`). `save_cache` /
    `load_fitted` / `feature_dims` have default no-op implementations
    suitable for stateless processors; processors with persistent fitted
    state override them.

    The orchestrator instantiates each enabled processor with its config
    block (`cls(proc_config)`), then drives the lifecycle uniformly:

        for name in registry.names_in_priority_order(enabled):
            ctx = OrchestratorContext(dataset, proc.config, cache_dir,
                                       upstream=...)
            proc.fit(ctx)
            if cache_dir:
                key = proc.cache_key(ctx)
                proc.save_cache(cache_dir, key)
            for split, _ in ctx.iter_splits():
                features = proc.transform(ctx, split)
                if features is not None:
                    attach_to_samples(dataset, split, features)

    No processor-specific branches in the orchestrator — adding a new
    processor only requires writing the class and decorating it with
    `@register_processor`.
    """

    # ── Class-level metadata ────────────────────────────────────────

    name: ClassVar[str]
    """Stable identifier; also the config-block key under
    `dataset.processing.<name>` and the artifact-side cache prefix."""

    priority: ClassVar[int]
    """Execution order. Lower runs first. Must respect `requires` (a
    processor's priority must be higher than every name it requires).
    """

    requires: ClassVar[tuple[str, ...]] = ()
    """Names of upstream processors that must be enabled. The orchestrator
    populates `OrchestratorContext.upstream` with the matching fitted
    processor instances before calling `fit` / `transform`.
    """

    produces_batch_fields: ClassVar[tuple[str, ...]] = ()
    """Legacy: per-sample batch-field names this processor produces. Retained
    for the generic registry mechanism (`batch_field_owners`); set-encoder
    modalities now declare `produces_views` instead.
    """

    produces_views: ClassVar[tuple[str, ...]] = ()
    """Named set-views this processor makes available on `G2FBatch.views`
    (e.g. `("weather",)` for `raw_weather`). Used via `view_owners` by the
    dataset's view filter to check the backing processor is enabled for any
    non-"main" view. The VI view `"main"` is produced by the collate path,
    not a processor, so it is never declared here.
    """

    has_eigen_sources: ClassVar[bool] = False
    """True iff the processor declares eigen-projected sources subject to
    the D1 rank-consistency contract (see `rank_check.py`).
    """

    eigen_source_types: ClassVar[frozenset[str]] = frozenset()
    """Source-type strings eligible for `effective_ranks` reporting (e.g.
    `frozenset({"additive_grm", "dominance_grm", "raw"})` for genomic).
    Empty when `has_eigen_sources=False`.
    """

    # ── Construction (orchestrator-facing factory) ─────────────────

    @classmethod
    def from_config(cls, config: dict) -> "BaseProcessor":
        """Factory: instantiate the processor from a Hydra config dict.

        The orchestrator calls this on each enabled processor's class,
        passing the corresponding `dataset.processing.<name>` block.
        The default implementation forwards the dict to `__init__` —
        override when the constructor takes typed args (the standard
        pattern for existing processors with explicit `csv_path` /
        `n_components` / etc. signatures).
        """
        return cls(config)

    # ── Lifecycle (every concrete processor implements these) ──────

    @abstractmethod
    def fit(self, ctx: OrchestratorContext) -> None:
        """Learn fitted state from the train split.

        Processors with `fit_scope="all"` (e.g. genomic, weather_fpca,
        enviromic) may fit on the train ∪ val ∪ test union when the input
        modality is yield-independent (genotype, weather) — see each
        processor's docstring for the per-modality leakage policy.
        """

    @abstractmethod
    def transform(
        self, ctx: OrchestratorContext, split: str,
    ) -> dict[str, np.ndarray] | None:
        """Project the named split's data using fitted state.

        Returns
        -------
        dict[str, np.ndarray] | None
            Per-source feature arrays keyed by source name, with rows
            aligned to `ctx.split_data(split)` order. The orchestrator
            attaches them to each sample's `derived_features`.

            `None` signals an in-place processor (`axis_source`,
            `weather_concat`, `raw_weather`, `raw_vi`) that mutated
            sample dicts directly — the
            orchestrator skips the attach step.
        """

    @abstractmethod
    def cache_key(self, ctx: OrchestratorContext) -> str:
        """Content-addressed cache key for `fit` inputs.

        Excludes post-load slice params (`n_components` /
        `n_components_max`) so K-sweeps reuse the same cached fit.
        Includes everything that changes the actual fitted state: source
        config (sans slice fields), train inputs, fit_scope, kernel /
        scaling / centering choices, etc.
        """

    # ── Optional persistence (default no-ops for stateless processors) ──

    def save_cache(self, cache_dir: str, key: str) -> None:
        """Persist fitted state to `cache_dir`, keyed by `key`.

        Default is a no-op — override only when fit() produces state that
        needs to be restored in predict mode.
        """

    def load_fitted(self, cache_dir: str, key: str) -> None:
        """Restore fitted state from `cache_dir/key`.

        After this call, `transform()` works without a prior `fit()`.
        Default is a no-op — pairs with `save_cache`'s default no-op for
        stateless processors.
        """

    # ── Reporting ───────────────────────────────────────────────────

    @property
    def feature_dims(self) -> dict[str, int]:
        """Output dim per derived-feature name. Available after `fit()`.

        Returns the empty dict by default (in-place processors); override
        for processors that produce `derived_features` entries. The
        returned dims should already reflect any D1 slicing (clamped
        width under `n_components_max`).
        """
        return {}

    # ── Coverage filter ─────────────────────────────────────────────

    @classmethod
    def coverage_mask(
        cls,
        metadata_df: "pd.DataFrame",
        proc_config: dict,
    ) -> np.ndarray | None:
        """Boolean keep_mask shaping which samples this processor can serve.

        Called by `G2FDataset` before processor instantiation, hence a
        classmethod operating on raw metadata + config. Default returns
        `None` (no coverage filtering); override only for processors with
        partial coverage — e.g. genomic returns `False` for pedigrees not
        present in the dosage CSV. Returning `None` is equivalent to
        returning `np.ones(len(metadata_df), dtype=bool)` but cheaper to
        detect.
        """
        return None
