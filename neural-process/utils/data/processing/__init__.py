"""Data processing layer: feature extraction orchestrator and processor exports.

The :class:`FeatureProcessor` orchestrator instantiates enabled processors
from a config dict, runs them in priority order via the registry, and
attaches computed features to dataset sample dicts. Downstream code (DL
models and FPCA baselines) consumes enriched sample dicts without
knowing how features were derived.

Adding a new processor: write a :class:`BaseProcessor` subclass with the
required ClassVars (`name`, `priority`, ...), decorate with
`@register_processor`, and import it here so the decorator runs. The
orchestrator dispatches to it uniformly via the registry — no
processor-specific branches anywhere.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from .axis_source import AxisSourceProcessor, AxisTable, build_axis_table
from .base import BaseProcessor, OrchestratorContext
from .enviromic import EnviromicFeatureBuilder, aggregate_weather
from .genomic import GenomicFeatureBuilder, load_genomic_csv
from .interaction import InteractionKernelBuilder
from .metadata_features import MetadataFeaturesProcessor
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
from .phenomic import PhenomicFeatureBuilder
from .raw_vi import RawVISubsetProcessor
from .registry import (
    _REGISTRY,
    get_processor_class,
    names_in_priority_order,
)
from .vi_fpca import VIFPCAProcessor
from .weather import (
    RawWeatherProcessor,
    WeatherConcatProcessor,
    WeatherFPCAProcessor,
    load_weather_csv,
)
from .weather_kernel import WeatherKernelProcessor

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


__all__ = [
    # Orchestrator + ABC
    "FeatureProcessor",
    "BaseProcessor",
    "OrchestratorContext",
    # Individual processors (also registered via @register_processor)
    "AxisSourceProcessor",
    "VIFPCAProcessor",
    "RawVISubsetProcessor",
    "WeatherConcatProcessor",
    "WeatherFPCAProcessor",
    "RawWeatherProcessor",
    "GenomicFeatureBuilder",
    "PhenomicFeatureBuilder",
    "EnviromicFeatureBuilder",
    "MetadataFeaturesProcessor",
    "InteractionKernelBuilder",
    "WeatherKernelProcessor",
    # Utilities
    "AxisTable",
    "build_axis_table",
    "aggregate_weather",
    "load_weather_csv",
    "load_genomic_csv",
    "compute_kernel",
    "center_kernel_train",
    "center_kernel_cross",
    "eigen_decompose",
    "slice_components",
    "read_component_params",
    "project_eigen_train",
    "project_eigen_cross",
]


# ── FeatureProcessor ────────────────────────────────────────────


class FeatureProcessor:
    """Orchestrates all feature extraction, fitted on train split.

    Reads a processing config dict, instantiates enabled processors via
    the registry, and runs them in priority order. Each processor's
    lifecycle is uniform (fit / transform / cache_key / save_cache /
    load_fitted) — there are no processor-specific branches in this
    class.

    Parameters
    ----------
    processing_cfg : dict
        Processing config block (the ``processing:`` section from YAML).
        Each sub-key (``vi_fpca``, ``genomic``, etc.) has an ``enabled``
        flag and processor-specific parameters.
    cache_dir : str | None
        Root cache directory. Forwarded to processors that support
        caching via :class:`OrchestratorContext`. ``None`` disables
        caching everywhere.
    """

    def __init__(
        self,
        processing_cfg: dict,
        cache_dir: str | None = None,
    ):
        self._cfg = dict(processing_cfg)
        self._cache_dir = cache_dir

        # Validate inter-processor dependencies before instantiating
        self._validate_dependencies()

        # Instantiate enabled processors in priority order
        self._processors: list[tuple[str, int, BaseProcessor]] = []
        self._instantiate_processors()

        # Source-name → producer registry, built once after
        # instantiation. Used by InteractionKernelBuilder.raw_kernel
        # to look up each component's producing processor.
        self._feature_to_processor: dict[str, BaseProcessor] = {}
        for _name, _, proc in self._processors:
            for src in getattr(proc, "sources", []) or []:
                src_name = src.get("name") if isinstance(src, dict) else None
                if src_name is not None:
                    self._feature_to_processor[src_name] = proc

        # Populated after fit_transform()
        self._effective_y_dim: int | None = None
        # Processors whose fitted state was NOT written to disk
        # (persist_cache: false) — their recorded cache keys have no
        # backing file, so predict mode must refuse them up front.
        self._unpersisted: list[str] = []

    # ── Public API ──────────────────────────────────────────────

    def fit_transform(self, dataset) -> None:
        """Train mode: fit on train split, transform all splits.

        Modifies dataset sample dicts in-place by attaching
        ``derived_features`` entries (and, for in-place processors,
        widening ``channels`` or attaching weather
        sequences). One uniform loop drives every processor; per-
        processor logic lives entirely in the processor's `fit` /
        `transform` / `cache_key` methods.

        Parameters
        ----------
        dataset : G2FDataset
            Must have ``_train``, ``_val``, ``_test``, ``metadata_df``,
            and ``split_indices`` attributes.
        """
        for proc_name, _, proc in self._processors:
            logger.info("Running processor: %s", proc_name)
            ctx = self._make_context(dataset, proc, proc_name)
            proc.fit(ctx)
            if self._cache_dir is not None:
                key = proc.cache_key(ctx)
                # `persist_cache: false` on a processor block skips the
                # on-disk write while still recording the key. Use it for
                # the derived N x N kernel processors (phenomic /
                # interaction / weather_kernel) in train-only CV runs:
                # their caches are read back only by predict-mode
                # `load_fitted`, never at fit time, and each split's
                # entry is ~1-2 GB of dead weight (a 400-job campaign
                # wrote 850 GB of them).
                if self._cfg.get(proc_name, {}).get("persist_cache", True):
                    proc.save_cache(self._cache_dir, key)
                else:
                    logger.info(
                        "Processor %s: persist_cache=false, skipping "
                        "cache write (key %s).", proc_name, key[:16],
                    )
                    if proc_name not in self._unpersisted:
                        self._unpersisted.append(proc_name)
                proc._last_cache_key = key
            for split, _ in ctx.iter_splits():
                features = proc.transform(ctx, split)
                if features is not None:
                    self._attach(dataset, {split: features})
            logger.info("Processor %s completed.", proc_name)

        self._record_effective_y_dim(dataset)

    def load_and_transform(
        self,
        dataset,
        cache_keys: dict[str, str],
    ) -> None:
        """Predict mode: load fitted state from cache, transform all splits.

        For each enabled processor, resolves its cache key from
        ``cache_keys``, calls ``load_fitted``, then runs the same
        per-split transform logic as :meth:`fit_transform` — but without
        refitting. External inputs (genomic CSV, weather CSV, etc.) are
        lazily loaded inside the processor's `transform()` because the
        cache provides only the *fitted state*, not the raw data.

        Raises
        ------
        RuntimeError
            If `cache_dir` was not configured at construction.
        KeyError
            If an enabled processor has no matching entry in `cache_keys`.
        FileNotFoundError
            If a processor's cache file is missing at the expected path.
        """
        if self._cache_dir is None:
            raise RuntimeError(
                "load_and_transform requires cache_dir; pass it at "
                "FeatureProcessor construction time."
            )

        for proc_name, _, proc in self._processors:
            if proc_name not in cache_keys:
                raise KeyError(
                    f"No cache key provided for enabled processor "
                    f"{proc_name!r}. Available keys: "
                    f"{sorted(cache_keys.keys())}"
                )
            key = cache_keys[proc_name]
            logger.info(
                "Loading processor %s from cache key %s",
                proc_name, key[:16],
            )
            proc.load_fitted(self._cache_dir, key)
            proc._last_cache_key = key

            ctx = self._make_context(dataset, proc, proc_name)
            for split, _ in ctx.iter_splits():
                features = proc.transform(ctx, split)
                if features is not None:
                    self._attach(dataset, {split: features})
            logger.info("Processor %s loaded + transformed.", proc_name)

        self._record_effective_y_dim(dataset)

    # ── Reporting ───────────────────────────────────────────────

    @property
    def feature_dims(self) -> dict[str, int]:
        """Aggregated output dimensions from all sub-processors.

        In-place processors (`axis_source`, `weather_concat`,
        `raw_weather`, `raw_vi`) return an
        empty dict from `feature_dims` by default, so they contribute
        nothing here without a name-based filter.
        """
        dims: dict[str, int] = {}
        for _, _, proc in self._processors:
            dims.update(proc.feature_dims)
        return dims

    @property
    def effective_ranks(self) -> dict[str, dict]:
        """Per-eigen-source record describing the consumed width + K knobs.

        Built by querying each registered processor for its
        ``eigen_source_types`` (a ClassVar). Replaces the previously
        hardcoded `{processor_name: {types}}` map. See
        :meth:`feature_dims`.
        """
        records: dict[str, dict] = {}
        for proc_name, _, proc in self._processors:
            eligible = getattr(
                type(proc), "eigen_source_types", frozenset(),
            )
            if not eligible:
                continue
            sources = getattr(proc, "sources", None)
            if not sources:
                continue
            dims = proc.feature_dims
            for src in sources:
                stype = src["type"]
                if stype not in eligible:
                    continue
                name = src["name"]
                n_comp, n_comp_max = read_component_params(src)
                if n_comp is not None:
                    mode = "hard"
                elif n_comp_max is not None:
                    mode = "max"
                else:
                    mode = "full"
                records[name] = {
                    "dim": int(dims[name]),
                    "mode": mode,
                    "requested_k": n_comp,
                    "requested_k_max": n_comp_max,
                    "processor": proc_name,
                }
        return records

    @property
    def effective_y_dim(self) -> int | None:
        """Width of ``channels`` after processing.

        The VI-subset width (37 by default, fewer under `raw_vi`), plus W
        when weather concat is enabled.
        Available after ``fit_transform()`` completes.
        """
        return self._effective_y_dim

    @property
    def weather_concat_n_vars(self) -> int:
        """W weather columns appended to ``channels`` by
        ``weather_concat``, or 0 if the processor is not enabled.

        Used by ``G2FDataset._normalize_dataset`` to decide whether to
        split per-column normalization between the native VI prefix and
        the weather-concat tail (the
        ``dataset.normalize.channels`` and
        ``dataset.normalize.weather_concat_values`` toggles).
        """
        for name, _, proc in self._processors:
            if name == "weather_concat":
                return proc.n_weather_vars
        return 0

    @property
    def processors(self) -> list[tuple[str, int, BaseProcessor]]:
        """List of (name, priority, processor) tuples in execution order."""
        return list(self._processors)

    @property
    def enabled_names(self) -> list[str]:
        """Names of enabled processors in execution order."""
        return [name for name, _, _ in self._processors]

    @property
    def cache_keys(self) -> dict[str, str]:
        """Per-processor cache keys recorded during ``fit_transform()``.

        Populated by the orchestrator after every successful
        :meth:`fit_transform`. The result is serialized to
        ``processing_metadata.json`` so predict-mode can restore the
        same cache entries via :meth:`load_and_transform`.
        """
        keys: dict[str, str] = {}
        for name, _, proc in self._processors:
            key = getattr(proc, "_last_cache_key", None)
            if key is not None:
                keys[name] = key
        return keys

    @property
    def unpersisted_processors(self) -> list[str]:
        """Processors that ran with ``persist_cache: false``.

        Their :attr:`cache_keys` entries have no backing cache file, so
        ``load_and_transform`` can never restore them — predict mode
        checks this list (via ``processing_metadata.json``) to fail with
        an actionable message instead of a bare ``FileNotFoundError``.
        """
        return list(self._unpersisted)

    # ── Private: instantiation ──────────────────────────────────

    def _validate_dependencies(self) -> None:
        """Generic dependency check via `cls.requires` per processor.

        Replaces the previous hardcoded "phenomic depends on vi_fpca"
        check. Any processor that declares `requires = ("foo", ...)`
        gets the same uniform treatment.
        """
        cfg = self._cfg
        for name, cls in _REGISTRY.items():
            if not _is_enabled(cfg, name):
                continue
            for required in cls.requires:
                if not _is_enabled(cfg, required):
                    raise ValueError(
                        f"{name} processing requires {required} to be "
                        f"enabled (declared via "
                        f"{cls.__name__}.requires)."
                    )

    def _instantiate_processors(self) -> None:
        """Create processor instances for each enabled config section.

        Iterates the registry in priority order and instantiates every
        processor whose corresponding config block has `enabled: true`,
        via the class's `from_config` factory.
        """
        procs: list[tuple[str, int, BaseProcessor]] = []
        for name in names_in_priority_order():
            if not _is_enabled(self._cfg, name):
                continue
            cls = get_processor_class(name)
            # Pass cache_dir to processors via config so from_config can
            # forward it to constructors that take it (vi_fpca,
            # weather_fpca). Processors that don't read it ignore it.
            proc_config = dict(self._cfg.get(name, {}))
            proc_config.setdefault("cache_dir", self._cache_dir)
            proc = cls.from_config(proc_config)
            procs.append((name, cls.priority, proc))
        self._processors = procs

    def _make_context(
        self, dataset, proc: BaseProcessor, proc_name: str,
    ) -> OrchestratorContext:
        """Build the orchestrator context for one processor's lifecycle.

        Resolves upstream-processor handles via `cls.requires`. Upstream
        must already be in `self._processors` (the registry's priority
        ordering enforces this — declaring `requires=("vi_fpca",)` with
        a priority lower than vi_fpca's would crash here, which is the
        intended fail-fast). Uses `getattr` with default `()` so duck-typed
        processors (that don't subclass `BaseProcessor`) still work
        without explicit requires-setup.

        ``proc_name`` is passed explicitly (rather than read from
        ``proc.name``) so duck-typed processors work without a name
        attribute.
        """
        requires = getattr(type(proc), "requires", ())
        upstream = {req: self._proc_by_name(req) for req in requires}
        return OrchestratorContext(
            dataset=dataset,
            proc_config=self._cfg.get(proc_name, {}),
            cache_dir=self._cache_dir,
            upstream=upstream,
            feature_to_processor=self._feature_to_processor,
        )

    def _proc_by_name(self, name: str) -> BaseProcessor:
        for n, _, proc in self._processors:
            if n == name:
                return proc
        raise KeyError(
            f"Processor {name!r} not in enabled set; "
            f"declared as a requirement but not enabled."
        )

    @property
    def feature_to_processor(self) -> dict[str, BaseProcessor]:
        """Source-name → producing processor (used by BGLR raw-K path)."""
        return dict(self._feature_to_processor)

    def build_ctx(self, dataset, proc_name: str) -> OrchestratorContext:
        """Build an OrchestratorContext for a named processor.

        Public wrapper around :meth:`_make_context` for callers outside
        the orchestrator's own fit/transform loop that need a context
        with `feature_to_processor` populated (e.g. to call
        `raw_kernel(...)` on a producer).
        """
        proc = self._proc_by_name(proc_name)
        return self._make_context(dataset, proc, proc_name)

    def _record_effective_y_dim(self, dataset) -> None:
        if dataset._train:
            self._effective_y_dim = (
                dataset._train[0]["channels"].shape[-1]
            )

    # ── Public utilities (broadcast helpers for processors) ────

    @staticmethod
    def _entity_to_splits(
        entity_features: dict[str, dict[str, np.ndarray]],
        metadata_df: "pd.DataFrame",
        split_indices: dict[str, list[int]],
        entity_column: str,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Convert per-entity features to per-sample per-split features.

        Useful sibling of `_env_dict_to_per_sample` keyed by an
        arbitrary metadata column (e.g. ``"Env"`` or ``"Pedigree"``)
        rather than the canonical Env.Year series. Retained as a public
        utility for processor implementations.

        Parameters
        ----------
        entity_features : ``{entity_name: {feature_name: array(dim,)}}``
        metadata_df : dataset metadata.
        split_indices : ``{split_name: [row_indices]}``.
        entity_column : column in ``metadata_df`` (e.g., ``"Env"``).

        Returns
        -------
        Per-split dict: ``{split_name: {feature_name: array(N, dim)}}``.
        """
        any_entity = next(iter(entity_features))
        feat_names = list(entity_features[any_entity].keys())

        result = {}
        for split_name, indices in split_indices.items():
            if not indices:
                continue
            entities = metadata_df.loc[indices, entity_column].tolist()
            n_samples = len(entities)

            split_feats: dict[str, np.ndarray] = {}
            for feat_name in feat_names:
                dim = entity_features[any_entity][feat_name].shape[0]
                arr = np.zeros((n_samples, dim), dtype=np.float32)
                for i, ent in enumerate(entities):
                    arr[i] = entity_features[ent][feat_name]
                split_feats[feat_name] = arr
            result[split_name] = split_feats

        return result

    @staticmethod
    def _env_dict_to_per_sample(
        env_features: dict[str, dict[str, np.ndarray]],
        metadata_df: "pd.DataFrame",
        split_indices: dict[str, list[int]],
        env_series: "pd.Series",
    ) -> dict[str, dict[str, np.ndarray]]:
        """Convert per-env feature dicts to per-sample feature arrays.

        Direct env→vector lookup — no row ordering assumptions. Useful
        for processors that compute features at env granularity and
        need to expand to per-sample arrays for the orchestrator's
        attach step. The orchestrator no longer calls this directly
        (each processor's `transform(ctx, split)` does its own
        per-sample broadcast), but it's retained as a public utility
        for processor implementations.

        Parameters
        ----------
        env_features : ``{feat_name: {env_name: vector(dim,)}}``
            Flat per-env mapping.
        metadata_df : dataset metadata DataFrame.
        split_indices : ``{split_name: [row_indices]}``.
        env_series : Series indexed like ``metadata_df`` whose values are
            the env identifiers used as keys in ``env_features``.

        Returns
        -------
        Per-sample dict: ``{split_name: {feat_name: array(n_samples, dim)}}``.
        """
        result: dict[str, dict[str, np.ndarray]] = {}
        for split_name in ("train", "val", "test"):
            indices = split_indices.get(split_name, [])
            if not indices:
                continue
            envs = env_series.loc[indices].tolist()
            n_samples = len(envs)

            per_sample: dict[str, np.ndarray] = {}
            for feat_name, env_dict in env_features.items():
                if not env_dict:
                    continue
                dim = next(iter(env_dict.values())).shape[0]
                arr = np.zeros((n_samples, dim), dtype=np.float32)
                for i, env in enumerate(envs):
                    if env not in env_dict:
                        raise KeyError(
                            f"Env {env!r} present in {split_name} split but "
                            f"not in {feat_name} env_features (have "
                            f"{sorted(env_dict.keys())[:5]}...)"
                        )
                    arr[i] = env_dict[env]
                per_sample[feat_name] = arr

            result[split_name] = per_sample
        return result

    # ── Private: feature attachment ─────────────────────────────

    @staticmethod
    def _attach(
        dataset,
        features: dict[str, dict[str, np.ndarray]],
    ) -> None:
        """Attach per-split features to sample dicts as ``derived_features``.

        Parameters
        ----------
        dataset : G2FDataset
        features : ``{split_name: {feature_name: array(N, dim)}}``
            where N matches the number of samples in that split.
        """
        for split_name, split_data in (
            ("train", dataset._train),
            ("val", dataset._val),
            ("test", dataset._test),
        ):
            if split_name not in features or not split_data:
                continue

            split_feats = features[split_name]
            if not split_feats:
                continue
            any_key = next(iter(split_feats))
            n_features = len(split_feats[any_key])
            if n_features != len(split_data):
                raise ValueError(
                    f"Feature count mismatch for split '{split_name}': "
                    f"got {n_features} features but "
                    f"{len(split_data)} samples."
                )

            for i, sample in enumerate(split_data):
                if "derived_features" not in sample:
                    sample["derived_features"] = {}
                for feat_name, feat_array in split_feats.items():
                    sample["derived_features"][feat_name] = torch.from_numpy(
                        feat_array[i].astype(np.float32)
                    )


# ── Helpers ─────────────────────────────────────────────────────


def _is_enabled(cfg: dict, key: str) -> bool:
    """Check if a processor section is enabled in the config."""
    return key in cfg and cfg[key].get("enabled", False)
