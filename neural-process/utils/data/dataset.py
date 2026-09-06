"""G2F dataset classes for loading, splitting, normalizing, and collating crop yield data."""

import logging
import random
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .base import G2FBatch, ViewBatch

if TYPE_CHECKING:
    from .views import View
from .processing import FeatureProcessor
from .reader import DataReader
from .splitter import DatasetSplitter

logger = logging.getLogger(__name__)


def _read_pedigree_index(csv_path: str) -> set[str]:
    """Read pedigree names from the genomic CSV without loading the full matrix.

    The genomic CSV is ~566 MB.  Only the first column (``Pedigree``) is read
    to keep memory usage and I/O time minimal.

    Parameters
    ----------
    csv_path : str
        Path to the genomic CSV.  Expected to have a ``Pedigree`` column as
        the first column, followed by SNP dosage columns.

    Returns
    -------
    set[str]
        Unique pedigree names found in the CSV.
    """
    pedigrees = pd.read_csv(csv_path, usecols=[0]).iloc[:, 0]
    return set(pedigrees.astype(str))


class _SplitDataset(torch.utils.data.Dataset):
    """Internal Dataset class for train/val/test splits."""

    def __init__(self, data_list: list[dict[str, torch.Tensor]]):
        """
        Args:
            data_list: List of sample dicts with keys 'dap',
                'channels', and 'yield_value'.
        """
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.data_list[idx]


class G2FDataset:
    """
    Base class for handling data loading, validation, dataset splitting,
    and normalization for Genome-to-Field (G2F) datasets.
    """

    def __init__(
        self,
        data_dir: str = "./data/Pedigrees_Wide_Format_BLUEs",
        *,
        validate: bool = True,
        case_sensitive: bool = False,
        delete_invalid: bool = True,
        data_df_columns: list[str | tuple] = [
            "Pedigree",
            "Vegetation.Index",
            "Year",
            "Env",
            ("VI.BLUE.", "prefix"),
            "Yield.t.ha.BLUE",
        ],
        metadata_df_columns: list[str] = ["Pedigree", "Year", "Env", "File"],
        data_df_fields_to_convert: dict | None = None,
        metadata_df_fields_to_convert: dict | None = None,
        to_tensor: bool = True,
        cache_dir: str | None = None,
        test_filter_columns: list[dict] | None = None,
        val_filter_columns: list[dict] | None = None,
        random_state: int | None = None,
        test_ratio: float = 0.1,
        val_seed: int = 0,
        eval_streams: list | None = None,
        normalize: dict[str, bool] = {
            "dap": False,
            "channels": False,
            "weather_concat_values": False,
            "yield_value": False,
        },
        smoke_n: int | None = None,
        processing: dict | None = None,
        processing_cache_keys: dict[str, str] | None = None,
        cv_spec: dict | None = None,
        views: dict | None = None,
    ):
        """
        Initialize the G2F dataset: load data, split, compute statistics, and normalize.

        Args:
            data_dir: Path to the directory containing per-pedigree CSV files.
            validate: Whether to validate file content against filename metadata.
            case_sensitive: Whether comparisons during validation are case-sensitive.
            delete_invalid: Whether to delete files that fail validation.
            data_df_columns: Expected columns (or prefix tuples) in data CSVs.
            metadata_df_columns: Expected columns in the parsed metadata DataFrame.
            data_df_fields_to_convert: Case conversion rules for data DataFrames.
            metadata_df_fields_to_convert: Case conversion rules for the metadata DataFrame.
            to_tensor: Whether to convert extracted arrays to PyTorch tensors.
            cache_dir: Directory for caching processed data. None (default)
                places the cache at ``{parent_of_data_dir}/.cache/`` (i.e.
                sibling to ``data_dir``, not inside it). Set to False to
                disable caching.
            test_filter_columns: Filter criteria for selecting the test set.
                List of dicts, each specifying one column condition. All
                conditions in the list are AND'd (intersected). Multiple
                values within a single dict are OR'd at the value level.
                Supports an optional "_expand_" key to include related files.
                When provided, test_ratio is ignored. Examples:
                  [{"Env": "DEH1"}]                      # single-env test set (all years)
                  [{"Env": "DEH1"}, {"Year": 2020}]      # DEH1 in 2020 only
                  [{"Env": ["DEH1", "IAH4"]}]            # DEH1 or IAH4 (any year)
            val_filter_columns: Filter criteria for selecting the validation set.
                Same format as test_filter_columns. When provided, val_ratio
                is ignored.
            random_state: Seed for reproducible shuffling during splits.
            test_ratio: Proportion of data allocated to the test set.
            normalize: Dict indicating which data keys to z-score normalize.
            smoke_n: If not None, randomly retain at most this many samples per
                split after normalization. If a split is smaller than smoke_n,
                the full split is kept. Intended for local smoke tests where a
                full dataset run is too slow. Sampling is seeded by random_state
                for reproducibility.
            processing: Config dict for FeatureProcessor (VI FPCA, genomic,
                weather, etc.). None disables all feature processing.
            processing_cache_keys: If provided, switches to predict mode —
                processors restore fitted state from cache instead of fitting
                on the train split. Keys map processor names to cache keys.
                None (default) uses train mode (fit + transform).

        Data splitting:
            Splitting is a two-step sequential process via DatasetSplitter:

            1. **Test split** — If test_filter_columns is provided, all files
               matching the filter become the test set (test_ratio is ignored).
               Otherwise, test_ratio determines a random proportion.
            2. **Train split** — Everything left after the test carve-out.
               The validation set is carved later from the training pool by
               the eval-streams ``carve`` stream (seeded by ``val_seed``);
               there is no flat val split here.

            Example with the shipped defaults
            (test_filter_columns=[{Env: DEH1},{Year: 2020}]):
              - Test:  all files where Env == DEH1 and Year == 2020
              - Train: all remaining files (non-DEH1 environments)
        """
        self.data_dir = data_dir
        self.validate = validate
        self.case_sensitive = case_sensitive
        self.delete_invalid = delete_invalid
        self.data_df_columns = data_df_columns
        self.metadata_df_columns = metadata_df_columns
        self.data_df_fields_to_convert = data_df_fields_to_convert
        self.metadata_df_fields_to_convert = metadata_df_fields_to_convert
        self.to_tensor = to_tensor
        self.cache_dir = cache_dir
        self.test_filter_columns = test_filter_columns
        self.val_filter_columns = val_filter_columns
        self.random_state = random_state
        self.test_ratio = test_ratio
        self.val_seed = val_seed
        # Evaluation streams (DL eval-streams plan): the ordered list of
        # named, read-only streams evaluated every val epoch. One ``carve``
        # stream (the monitor) carves the DL early-stopping holdout from the
        # train pool — seeded by ``val_seed`` (decoupled from ``misc.seed``),
        # replacing the legacy flat ``val_ratio`` carve. ``observe`` streams
        # (``test`` / ``cv0`` / …) are read-only subsets of the assigned rows.
        self.eval_streams_cfg = eval_streams
        from utils.data.eval_streams import parse_eval_streams

        self.eval_stream_specs = parse_eval_streams(eval_streams)
        self.normalize = normalize
        if smoke_n is not None and (not isinstance(smoke_n, int) or smoke_n < 1):
            raise ValueError(
                f"smoke_n must be a positive integer, got {smoke_n!r}"
            )
        self.smoke_n = smoke_n
        self.processing = processing
        self.processing_cache_keys = processing_cache_keys
        self.cv_spec = cv_spec

        # View specs: config-driven coords/channels selection per
        # set-encoder peer. Built into View objects here; the DL-target subset
        # (`self.dl_views`) is handed to the collate fn, and `_inject_view_dims`
        # reads `self.views["main"]` for the VI branch coord/channel widths.
        from .views import View  # local import to keep import order one-way

        self.views_config = views
        self.views: dict[str, View] = {}
        if views:
            for _vname, _vspec in views.items():
                self.views[_vname] = View.from_config(_vname, _vspec)
        # Drop any view whose backing processor is disabled, so a view
        # declared in the base config (e.g. `weather`, owned by raw_weather)
        # is inert unless its processor is enabled. Ownerless views (`main`,
        # collate-produced) are always kept. This keeps single-view configs
        # from carrying an un-assemblable weather view (assemble_view /
        # _inject_view_dims would otherwise raise on the missing weather_dap).
        self.views = self._filter_views_to_enabled_processors(self.views)
        self.dl_views: list[View] = [
            v for v in self.views.values() if v.target == "dl"
        ]

        # Explicit VI-FPCA fit-mask + per-row cv_label (set by the
        # role-based split path; None under the legacy filter/ratio path).
        self.fit_mask: np.ndarray | None = None
        self.cv_labels: np.ndarray | None = None

        # 1. Init DataReader
        self._initialize_reader()
        # 2. Load raw VI + yield data
        self._load_data()
        # 2.5. Row-alignment invariant: split_indices are stored as
        #      metadata_df.index labels yet consumed positionally
        #      (keep_mask[idx], .iloc), so label must equal position.
        self._assert_metadata_rangeindex()

        if self.cv_spec is not None:
            # Role-based split (cv_2_1 / cv_0_00 / env_year_loo): coverage
            # filter FIRST so the common-female set is computed on the
            # genotyped universe (= R's `order`, D6), then assign roles.
            self._build_role_based_split()
        else:
            # 3. Init splitter
            self._initialize_splitter()
            # 4. Train / test split (NO val carve here — the val carve is now
            #    an eval-streams `carve` strategy seeded by `val_seed`, applied
            #    below over the train pool, replacing the legacy `val_ratio`).
            self._split_data()
            # 5. Pre-filter to modality coverage (drops samples without
            #    genotype data when genomic processing is enabled)
            self._filter_to_modality_coverage(self.processing)
            # 5b. Synthesize the (role, cv_label) columns the eval-stream
            #     selectors read, then carve the DL val holdout from the train
            #     pool via the configured strategy — the legacy path's
            #     equivalent of `assign_roles` doing it inline (role-based).
            self._synthesize_legacy_assignment_columns()
            self._carve_val_from_train()

        # 5.5. Smoke subsample before processing so FPCA/kernel fits
        #      run on tiny data during local smoke tests.
        if self.smoke_n is not None:
            self._subsample_smoke()

        # 6. Feature processing BEFORE normalization (needs raw values)
        self.processor = None
        if self.processing:
            self.processor = FeatureProcessor(
                self.processing, cache_dir=self.reader.cache_dir
            )
            if self.processing_cache_keys is not None:
                self.processor.load_and_transform(self, self.processing_cache_keys)
            else:
                self.processor.fit_transform(self)

        # 7. Stats on (possibly widened) raw data
        self._compute_train_statistics()
        # 8. Normalize all splits
        self._normalize_datasets()
        # 9. Materialize the eval-stream row lists (carve → val; observe →
        #    read-only subsets of the already-normalized train/test rows).
        self._build_eval_stream_splits()

    def _initialize_reader(self):
        """Initializes the DataReader object."""
        self.reader = DataReader(
            data_dir=self.data_dir,
            validate=self.validate,
            case_sensitive=self.case_sensitive,
            delete_invalid=self.delete_invalid,
            data_df_columns=self.data_df_columns,
            metadata_df_columns=self.metadata_df_columns,
            data_df_fields_to_convert=self.data_df_fields_to_convert,
            metadata_df_fields_to_convert=self.metadata_df_fields_to_convert,
            to_tensor=self.to_tensor,
            cache_dir=self.cache_dir,
        )

    def _assert_metadata_rangeindex(self) -> None:
        """Guard the row-alignment invariant: ``metadata_df`` index == position.

        The whole splitting/scoring system silently assumes ``metadata_df``
        carries a contiguous ``RangeIndex`` (label == position):
        ``split_indices`` are stored as ``.index`` *labels* yet consumed
        *positionally* (``keep_mask[idx]`` in ``_apply_coverage_filter``,
        ``.iloc`` in the BGLR block dump), and the BGLR env-block path mixes
        ``.loc`` with ``.iloc`` on the same indices. A violation mislabels
        rows with **no error**, so we fail loudly at the dataset-build
        boundary instead.
        """
        idx = self.metadata_df.index
        if not idx.equals(pd.RangeIndex(len(self.metadata_df))):
            raise AssertionError(
                "metadata_df must carry a contiguous RangeIndex "
                "(label == position) — split_indices / .loc / .iloc all "
                f"depend on it. Got index of type {type(idx).__name__} that "
                "is not RangeIndex(0..n). A metadata filter that omits "
                "reset_index(drop=True) is the likely cause."
            )

    def _initialize_splitter(self):
        """Initializes the DatasetSplitter."""
        self.splitter = DatasetSplitter(
            metadata_df=self.metadata_df,
            data_dict=self.data_dict,
            random_state=self.random_state,
        )

    def _load_data(self):
        """Loads the dataset using the reader."""
        self.data_dict, self.metadata_df = self.reader.load()
        # Canonical VI ordering for the per-sample
        # `channels[:, j]` axis. Used by VIFPCAProcessor's
        # `vi_subset` post-load slice.
        self.vi_names: list[str] = list(self.reader.vi_names_ or [])

    def _filter_views_to_enabled_processors(
        self, views: "dict[str, View]"
    ) -> "dict[str, View]":
        """Keep only views whose backing processor is enabled.

        A view is dropped when its owning processor (per the registry's
        ``view_owners()``: e.g. ``weather`` -> ``raw_weather``) is absent or
        disabled in ``self.processing``. Views with no owner — ``main``, which
        the collate path produces from the VI sample dict — are always kept.

        This lets the base config declare optional views (the ``weather`` peer)
        unconditionally while keeping single-view configs from carrying an
        un-assemblable view: with ``raw_weather`` off there is no
        ``weather_dap`` on the sample, so assembling the ``weather`` view (in
        collate or ``_inject_view_dims``) would raise.
        """
        from .processing.registry import view_owners

        owners = view_owners()
        processing = self.processing or {}
        kept: dict[str, View] = {}
        for name, view in views.items():
            owner = owners.get(name)
            if owner is None:
                kept[name] = view  # collate-produced (e.g. "main")
                continue
            proc_cfg = processing.get(owner)
            enabled = bool(proc_cfg) and bool(
                proc_cfg.get("enabled", False)
                if hasattr(proc_cfg, "get")
                else getattr(proc_cfg, "enabled", False)
            )
            if enabled:
                kept[name] = view
        return kept

    def _split_data(self):
        """Split into train / test (the val holdout is carved separately).

        The legacy flat ``val_ratio`` / ``val_filter_columns`` carve is gone:
        ``split_data`` now produces only the test split (filter or ratio) and
        the train pool (everything else); ``val`` comes from the configured
        eval-streams ``carve`` strategy applied afterward
        (:meth:`_carve_val_from_train`), seeded by ``val_seed``.
        """
        self._train, self._val, self._test, self.split_indices = (
            self.splitter.split_data(
                test_ratio=self.test_ratio,
                val_ratio=0.0,
                test_filter_columns=self.test_filter_columns,
                val_filter_columns=None,
            )
        )

    def _compute_train_statistics(self):
        """
        Computes mean and standard deviation for training data.
        """

        def _compute_mean_std(
            tensor: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute mean and standard deviation with safety checks."""
            if tensor.numel() == 0:
                print(
                    "[WARNING] Empty tensor encountered while computing statistics."
                )
                return torch.tensor(0.0), torch.tensor(
                    1.0
                )  # Default values to prevent errors
            # ddof=1 (unbiased=True) keeps z-scoring consistent across the
            # data pipeline — score_scalers, weather processors, and
            # enviromic aggregation all use ddof=1 (R `sd()` semantics).
            mean = tensor.mean(dim=0)
            # A single-row tensor makes the ddof=1 std ill-defined; torch warns
            # ("degrees of freedom is <= 0") and returns NaN. We intentionally
            # handle that NaN below, so silence the expected warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                std = tensor.std(dim=0, unbiased=True)
            # A single-row split makes unbiased std NaN; clamp any non-finite
            # std to 1.0 (mirroring score_scalers._safe_std_ddof1). Left as-is,
            # NaN would slip through _normalize's `std.ne(0)` guard (NaN != 0)
            # and z-score the entire split to NaN. A genuine zero std (constant
            # column) stays 0 and is handled by that same guard (pass-through).
            std = torch.where(torch.isfinite(std), std, torch.ones_like(std))
            return mean, std

        # Define keys for dataset attributes
        keys = ["dap", "channels", "yield_value"]

        # Compute statistics dynamically
        train_stats = {}
        for key in keys:
            tensor_values = [data[key] for data in self._train if key in data]
            if not tensor_values:
                print(f"[WARNING] No valid data found for key: {key}")
                continue

            tensor = torch.cat(tensor_values, dim=0)
            mean, std = _compute_mean_std(tensor)

            setattr(self, f"{key}_train_mean", mean)
            setattr(self, f"{key}_train_std", std)
            train_stats[key] = (mean, std)

        # # Log computed statistics in a structured way
        # for key, (mean, std) in train_stats.items():
        #     mean_str = (
        #         f"{mean.tolist()}"
        #         if mean.numel() > 1
        #         else f"{mean.item():.6f}"
        #     )
        #     std_str = (
        #         f"{std.tolist()}" if std.numel() > 1 else f"{std.item():.6f}"
        #     )
        #     print(
        #         f"[INFO] {key.replace('_', ' ').title()} "
        #         f"-> Train Mean: {mean_str} | Std: {std_str}"
        #     )

    def _normalize_datasets(self):
        """Normalizes the train, validation, and test datasets if requested."""
        self._train = self._normalize_dataset(self._train)
        self._val = self._normalize_dataset(self._val)
        self._test = self._normalize_dataset(self._test)

    def _normalize_dataset(
        self, data_list: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        """Normalize each split per ``self.normalize``, using train stats.

        Per-key behaviour:

        - ``dap`` and ``yield_value`` use the scalar
          ``self.normalize[key]`` flag.
        - ``channels`` is split into the native VI prefix
          (first ``effective_y_dim - W`` columns) and the weather-concat
          tail (last ``W`` columns, when ``weather_concat`` is enabled).
          Each portion is gated independently by
          ``self.normalize["channels"]`` and
          ``self.normalize["weather_concat_values"]`` respectively.
        """
        if not data_list:
            return data_list

        def _normalize(
            value: torch.Tensor,
            mean: torch.Tensor,
            std: torch.Tensor,
            normalize_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Per-last-dim z-score where mask is True AND std>0;
            pass through otherwise."""
            if not normalize_mask.any():
                return value
            # Require a finite, non-zero std: a NaN/inf std (e.g. a single-row
            # split) must NOT normalize that column (NaN != 0 would otherwise
            # pass the guard and z-score to NaN). Std is clamped at the source
            # in _compute_train_statistics, so isfinite() is defence-in-depth.
            apply = normalize_mask & torch.isfinite(std) & std.ne(0)
            if not apply.any():
                return value
            eff_mean = torch.where(apply, mean, torch.zeros_like(mean))
            eff_std = torch.where(apply, std, torch.ones_like(std))
            return (value - eff_mean) / eff_std

        # Build per-key normalize masks.
        n_widened_vi = data_list[0]["channels"].shape[-1]
        n_wc = (
            self.processor.weather_concat_n_vars
            if self.processor is not None
            else 0
        )
        n_native_vi = n_widened_vi - n_wc

        vi_flag = bool(self.normalize["channels"])
        wc_flag = bool(self.normalize["weather_concat_values"])
        vi_mask = torch.tensor(
            [vi_flag] * n_native_vi + [wc_flag] * n_wc
        )

        masks = {
            "dap": torch.tensor(
                [bool(self.normalize["dap"])]
            ),
            "channels": vi_mask,
            "yield_value": torch.tensor(
                [bool(self.normalize["yield_value"])]
            ),
        }

        return [
            {
                **data,
                **{
                    key: _normalize(
                        data[key],
                        getattr(self, f"{key}_train_mean"),
                        getattr(self, f"{key}_train_std"),
                        masks[key],
                    )
                    for key in masks
                },
            }
            for data in data_list
        ]

    def _subsample_smoke(self) -> None:
        """Randomly retain at most smoke_n samples per split (for local smoke tests)."""
        rng = random.Random(self.random_state)
        split_attr_to_name = {"_train": "train", "_val": "val", "_test": "test"}
        for attr, name in split_attr_to_name.items():
            split = getattr(self, attr)
            if not split:
                continue
            k = min(self.smoke_n, len(split))
            idx = sorted(rng.sample(range(len(split)), k))
            setattr(self, attr, [split[i] for i in idx])
            if hasattr(self, "split_indices") and name in self.split_indices:
                orig = self.split_indices[name]
                self.split_indices[name] = [orig[i] for i in idx]
        print(
            f"[G2FDataset] smoke_n={self.smoke_n}: "
            f"train={len(self._train)}, val={len(self._val)}, test={len(self._test)}"
        )

    def _filter_to_modality_coverage(self, processing_cfg: dict | None) -> None:
        """Restrict dataset to samples with complete coverage for all enabled modalities.

        Each registered processor declares its coverage filter via the
        `BaseProcessor.coverage_mask(metadata_df, proc_config)` classmethod.
        This method iterates every enabled processor, calls its
        `coverage_mask`, and AND's the resulting boolean masks. A
        processor that returns `None` contributes no constraint.

        Replaces the previously-hardcoded genomic-pedigree check.
        Modifies ``_train``, ``_val``, ``_test``, ``split_indices``, and
        ``metadata_df`` in-place. No-op when all enabled modalities
        have full coverage or when no processing is configured.
        """
        keep_mask = self._coverage_keep_mask(processing_cfg)
        if keep_mask is None or keep_mask.all():
            return  # no-op: all enabled modalities have full coverage

        self._apply_coverage_filter(keep_mask)

    def _coverage_keep_mask(
        self, processing_cfg: dict | None
    ) -> np.ndarray | None:
        """AND of every enabled processor's coverage mask over ``metadata_df``.

        Returns ``None`` when no processing is configured (no constraint).
        Shared by the legacy split path (``_filter_to_modality_coverage``)
        and the role-based path (``_build_role_based_split``), so both drop
        exactly the same rows.
        """
        if not processing_cfg:
            return None

        from utils.data.processing.registry import (
            _REGISTRY as _PROC_REGISTRY,
        )

        keep_mask = np.ones(len(self.metadata_df), dtype=bool)
        for name, cls in _PROC_REGISTRY.items():
            proc_block = processing_cfg.get(name, {})
            mask = cls.coverage_mask(self.metadata_df, proc_block)
            if mask is None:
                continue
            n_drop = int((~mask & keep_mask).sum())
            if n_drop > 0:
                logger.info(
                    "%s coverage filter: dropping %d samples. "
                    "Remaining: %d samples.",
                    name, n_drop, int(mask.sum()),
                )
            keep_mask &= mask
        return keep_mask

    def _build_val_strategies(self, assignment=None) -> list:
        """Validation carve strategies from the eval-streams ``carve`` specs.

        Each ``source: carve`` stream resolves to a
        :class:`ValidationStrategy` via the carve registry; OOD strategies
        (``grouped_variety`` / ``holdout_env`` / ``fold``) and the IID
        ``random`` strategy are all reachable. No carve specs (BGLR baselines,
        predict-mode) → ``[]`` → no carve. The carve is seeded by the
        dedicated ``val_seed`` (decoupled from ``misc.seed``), applied by the
        splitter over the train pool. A ``fold`` strategy needs the run's
        ``fold_map``, reconstructed from the scheme assignment.
        """
        carves = [s for s in self.eval_stream_specs if s.is_carve]
        if not carves:
            return []
        from utils.data.eval_streams import get_val_strategy

        fold_map = self._fold_map_from_assignment(assignment)
        return [
            get_val_strategy(c.strategy or {}, fold_map=fold_map) for c in carves
        ]

    def _fold_map_from_assignment(self, assignment) -> dict[str, int] | None:
        """``{female_id: fold}`` from a scheme assignment, or ``None``.

        The ``fold`` carve strategy reuses the committed female→fold partition.
        It is not exposed on :class:`SchemeAssignment`, so rebuild it from the
        per-row ``folds`` array (NaN for non-common rows) joined positionally
        with the normalized maternal-line ids of ``metadata_df`` (the
        assignment is built from the same coverage-filtered frame). ``None``
        when there is no fold structure (legacy path, or ``env_year_loo``) —
        the VALIDATE gate rejects ``fold`` there before this is consulted.
        """
        if assignment is None or getattr(assignment, "folds", None) is None:
            return None
        from cv.schemes.identifiers import female_series

        fem = female_series(self.metadata_df).to_numpy()
        fmap: dict[str, int] = {}
        for f, fold in zip(fem, assignment.folds):
            if fold == fold:  # not NaN
                fmap[str(f)] = int(fold)
        return fmap or None

    def _synthesize_legacy_assignment_columns(self) -> None:
        """Attach ``role`` / ``cv_label`` columns under the legacy split path.

        The role-based path carries these from the ``SchemeAssignment``; the
        legacy filter/ratio path has none, so the eval-stream selectors
        (``role: PREDICT`` → ``test``) would have nothing to read. Synthesize
        them from ``split_indices``: ``PREDICT`` for the held-out test rows,
        ``FIT`` for the train pool; ``cv_label`` is ``None`` everywhere (no
        scheme labels exist — ``by_label`` streams are gate-rejected without a
        fold-bearing scheme). Runs after the coverage filter so positions align
        with the final ``metadata_df``.

        NB: only the ``metadata_df`` columns are written (the observe selectors
        read those). ``self.cv_labels`` is deliberately left ``None`` so the
        offline scorer's legacy fallback still fires —
        ``cv/scoring.label_array`` labels the held-out test rows ``"test"`` only
        when ``dataset.cv_labels is None`` (the degenerate one-label case). An
        all-``None`` array here would defeat that and score nothing.
        """
        from cv.schemes.base import Role

        n = len(self.metadata_df)
        role = np.full(n, Role.FIT.value, dtype=object)
        for p in self.split_indices.get("test", []):
            role[p] = Role.PREDICT.value
        cv_label = np.full(n, None, dtype=object)
        self.metadata_df = self.metadata_df.assign(role=role, cv_label=cv_label)

    def _carve_val_from_train(self) -> None:
        """Carve the DL val holdout from the train pool (legacy path).

        Applies the configured ``carve`` strategies over the train positions
        (a single ``np.random.default_rng(val_seed)`` shared across them, no
        global-RNG dance), moving the carved rows from ``_train`` to ``_val``,
        updating ``split_indices`` and clearing the carved rows' ``cv_label``.
        Multiple carves select sequentially from the shrinking pool (disjoint
        by construction). No carve specs → ``_val`` stays empty.
        """
        strategies = self._build_val_strategies()
        if not strategies:
            return
        train_pos = list(self.split_indices["train"])
        rng = np.random.default_rng(self.val_seed)
        remaining = np.array(train_pos, dtype=int)
        carved: list[int] = []
        for strat in strategies:
            sel = np.asarray(
                strat.select(self.metadata_df, remaining, rng=rng), dtype=int
            )
            remaining_set = set(remaining.tolist())
            outside = [int(p) for p in sel if int(p) not in remaining_set]
            if outside:
                raise AssertionError(
                    f"val strategy {strat.name!r} selected positions outside "
                    f"the remaining train pool: {outside[:5]}."
                )
            sel_set = {int(p) for p in sel}
            carved.extend(sorted(sel_set))
            remaining = np.array(
                [p for p in remaining if p not in sel_set], dtype=int
            )
        if not carved:
            # Carve strategies WERE configured (the no-strategy case returned
            # earlier) but selected zero rows — e.g. a `random` frac that rounds
            # to int(len·frac)==0 on a tiny train pool. Failing loudly here
            # mirrors run.py's empty-monitor-stream guard: an empty val carve
            # leaves the monitor dataloader empty, so early-stopping and
            # best-checkpoint would silently have nothing to watch.
            raise ValueError(
                "DL val carve produced 0 rows — the monitor dataloader would "
                "be empty and early stopping / best-checkpoint would watch a "
                "metric that is never logged. Check the carve strategy (a "
                "`frac` that rounds to zero on a small train pool, or a spec "
                "matching no train rows)."
            )
        carved_set = set(carved)
        keep_local, carve_local = [], []
        for i, p in enumerate(train_pos):
            (carve_local if p in carved_set else keep_local).append(i)

        self._val = [self._train[i] for i in carve_local]
        self._train = [self._train[i] for i in keep_local]
        self.split_indices["val"] = [train_pos[i] for i in carve_local]
        self.split_indices["train"] = [train_pos[i] for i in keep_local]

        # A carved val row is an early-stopping holdout, never scored.
        cv = self.metadata_df["cv_label"].to_numpy().copy()
        for p in carved:
            cv[p] = None
        self.metadata_df["cv_label"] = cv

    def _build_eval_stream_splits(self) -> None:
        """Materialize the per-stream row lists consumed by ``get_split``.

        Carve streams reuse the already-built ``_val`` rows. Observe streams
        select positions over the post-carve, coverage-filtered ``metadata_df``
        and map them — via a position→normalized-row index over the base
        train/val/test splits — to a read-only row list. Observe streams carve
        nothing and may overlap each other (e.g. ``cv0 ∪ cv00 ⊆ test``); a
        ``cv2`` stream resolves to in-sample FIT rows. Built after
        normalization, so every stream row is train-stat-normalized exactly
        like ``test``.
        """
        self._stream_splits: dict[str, list[dict]] = {}
        if not self.eval_stream_specs:
            return
        from utils.data.eval_streams import get_observe_selector

        pos_to_row: dict[int, dict] = {}
        for split in ("train", "val", "test"):
            idxs = self.split_indices.get(split, [])
            rows = getattr(self, f"_{split}")
            for local, pos in enumerate(idxs):
                pos_to_row[int(pos)] = rows[local]

        for spec in self.eval_stream_specs:
            if spec.is_carve:
                # Single carve slot → the materialized val rows.
                self._stream_splits[spec.name] = self._val
                continue
            selector = get_observe_selector(spec.selector or {})
            positions = np.asarray(selector.select(self.metadata_df), dtype=int)
            rows = [
                pos_to_row[int(p)] for p in positions if int(p) in pos_to_row
            ]
            self._stream_splits[spec.name] = rows
            if not rows:
                logger.warning(
                    "eval stream %r (selector %s) selected 0 rows for this "
                    "split job — its val dataloader will be skipped.",
                    spec.name,
                    (spec.selector or {}).get("name"),
                )

    def _build_role_based_split(self) -> None:
        """Coverage-filter metadata, then split by explicit scheme roles.

        Unlike the legacy flow (split → coverage filter on splits), the
        role-based path filters ``metadata_df`` to the genotyped universe
        *first* so the common-female determination (D6) runs on the same
        rows R used (its ``order``). The scheme assignment is then computed
        over the coverage-filtered frame and the splitter partitions by
        role. Sets ``self.fit_mask`` and ``self.cv_labels`` and carries
        ``role`` / ``cv_label`` columns onto ``metadata_df`` for the
        label-based scorer (Step 3 / D8).
        """
        from cv.schemes import resolve_cv_assignment

        keep_mask = self._coverage_keep_mask(self.processing)
        # The common-female universe is the FULL pre-coverage frame — the
        # single source of truth for "common", matching R's Metadata script
        # (full phenotype, before the genotype filter), independent of
        # fold_source. Roles are still assigned over the coverage-filtered
        # genotyped frame (R's ``order``).
        full_metadata_df = self.metadata_df
        if keep_mask is not None and not keep_mask.all():
            self.metadata_df = self.metadata_df[keep_mask].reset_index(
                drop=True
            )
        self._assert_metadata_rangeindex()

        assignment = resolve_cv_assignment(
            self.metadata_df,
            self.cv_spec,
            common_universe_df=full_metadata_df,
        )

        self.splitter = DatasetSplitter(
            metadata_df=self.metadata_df,
            data_dict=self.data_dict,
            random_state=self.random_state,
        )
        self._train, self._val, self._test, self.split_indices = (
            self.splitter.assign_roles(
                assignment,
                val_strategies=self._build_val_strategies(assignment),
                val_seed=self.val_seed,
            )
        )
        self.fit_mask = self.splitter.fit_mask
        self.cv_labels = self.splitter.cv_labels

        # Carry role/label (+ Female / Fold) onto metadata_df for the
        # label-based scorer and the D8 all-row predictions.csv.
        from cv.schemes.identifiers import female_series

        self.metadata_df = self.metadata_df.assign(
            Female=female_series(self.metadata_df).to_numpy(),
            Fold=assignment.folds,
            role=assignment.roles,
            cv_label=self.cv_labels,
        )

    def _apply_coverage_filter(self, keep_mask: np.ndarray) -> None:
        """Apply a boolean mask to drop samples from all splits and metadata.

        Parameters
        ----------
        keep_mask : np.ndarray of bool, shape (n_total,)
            Boolean mask over ``metadata_df`` rows.  ``True`` = keep.
        """
        # Build old → new index mapping for metadata_df
        new_index = np.full(len(keep_mask), -1, dtype=int)
        new_index[keep_mask] = np.arange(keep_mask.sum())

        for split_name in ("train", "val", "test"):
            old_idxs = self.split_indices[split_name]
            if not old_idxs:
                continue

            # Which positions in this split's data list survive?
            surviving = [i for i, idx in enumerate(old_idxs) if keep_mask[idx]]

            data_list = getattr(self, f"_{split_name}")
            setattr(self, f"_{split_name}", [data_list[i] for i in surviving])

            # Update split_indices with new metadata_df positions
            self.split_indices[split_name] = [
                int(new_index[old_idxs[i]]) for i in surviving
            ]

        # Filter metadata_df and reset index
        self.metadata_df = self.metadata_df[keep_mask].reset_index(drop=True)

    def get_split(self, split: str) -> torch.utils.data.Dataset:
        """Retrieve a split (or eval stream) as a PyTorch map-style Dataset.

        Resolves, in order:

        1. a named eval stream (``cv0`` / ``cv00`` / ``cv1`` / ``cv2`` / any
           configured observe or carve stream) from ``self._stream_splits``
           (case-sensitive — stream names are explicit config identifiers);
        2. one of the base splits ``train`` / ``val`` / ``test``
           (case-insensitive).

        **Stream names take priority over base splits.** A configured stream
        whose name equals a base split (the shipped bundles name the PREDICT
        observe probe ``test``) therefore SHADOWS that base split — during
        training ``get_split("test")`` returns the *observe stream*, not
        ``self._test``. This is intentional and safe in the current wiring:
        ``run_training`` only fetches ``train`` and the streams by their own
        names, and the offline path (``eval.py`` / ``evaluate_model``) runs in
        predict-mode, which clears ``eval_streams`` (so ``_stream_splits`` is
        empty and the base ``test`` is reached) and otherwise reads
        ``dataset._test`` directly. A future *training-path* caller that wants
        the base ``test``/``val`` split by name while a same-named stream is
        configured would get the stream instead — read the attribute
        (``self._test``) directly in that case.

        ``_SplitDataset`` instances are cached per resolved key.
        """
        stream_splits = getattr(self, "_stream_splits", {})
        if split in stream_splits:
            key = split
            data_list = stream_splits[split]
        else:
            key = split.lower()
            if key not in {"train", "val", "test"}:
                raise ValueError(
                    f"Invalid split name: '{split}'. Must be one of "
                    f"['train', 'val', 'test'] (case-insensitive) or a "
                    f"configured eval stream "
                    f"{sorted(stream_splits)}."
                )
            data_list = getattr(self, f"_{key}")

        attr = f"_split_ds__{key}"
        if not hasattr(self, attr):
            setattr(self, attr, _SplitDataset(data_list))
        return getattr(self, attr)


def _pad_view_sequences(
    coords_list: list[torch.Tensor],
    chan_list: list[torch.Tensor],
    pad_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of per-sample ``(coords, channels)`` to a common length.

    Generalises the VI padding to any coordinate width ``D`` (multi-axis
    views) and channel width ``C``. Returns ``(coords[B,T,D],
    channels[B,T,C], pad_mask[B,T])`` with ``pad_mask`` True for padded rows.
    Operates on copies of the list (the caller's tensors are not mutated in
    place beyond the local list rebinding).
    """
    coords_list = list(coords_list)
    chan_list = list(chan_list)
    seq_lengths = [c.shape[0] for c in coords_list]
    max_seq_len = max(seq_lengths)
    batch_size = len(coords_list)
    pad_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    for i in range(batch_size):
        pad_len = max_seq_len - coords_list[i].shape[0]
        if pad_len > 0:
            coords_list[i] = F.pad(
                coords_list[i], (0, 0, 0, pad_len), value=pad_value
            )
            chan_list[i] = F.pad(
                chan_list[i], (0, 0, 0, pad_len), value=pad_value
            )
            pad_mask[i, -pad_len:] = True
    return (
        torch.stack(coords_list, dim=0),
        torch.stack(chan_list, dim=0),
        pad_mask,
    )


def g2f_collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_value: float = 0.0,
    views: "list[View] | None" = None,
) -> G2FBatch:
    """
    Collate function for G2F data that pads variable-length sequences,
    constructs padding masks, and assembles the per-modality set views.

    View assembly:

    - ``views`` given (the production path): each :class:`~utils.data.views.View`
      is realised per sample via :func:`~utils.data.views.assemble_view`
      (config-driven coords/channels selection + axis transform), then padded
      into a :class:`ViewBatch`. Multi-axis coords (``D>1``) and
      axis-as-channel selectors flow through here.
    - ``views=None`` (the legacy default): a single ``"main"`` view is
      built directly from ``dap`` (coords) + ``channels`` (vegetation indices),
      byte-equivalent to the pre-Views path and not requiring per-sample
      ``channel_names``.

    The ``raw_weather`` ``"weather"`` peer is now an ordinary config-driven
    view (``coords:[weather_dap] channels:[weather.*]``) assembled through the
    same :func:`~utils.data.views.assemble_view` path — no
    special-casing. Fixed-size derived features feed the MLP peers.

    Parameters:
        batch: List of sample dicts from the Dataset.
        pad_value: Value used to pad shorter sequences.
        views: DL-target View specs to assemble, or None for the legacy main.

    Returns:
        G2FBatch with named views, scalar target ``s``, and derived features.
    """
    s = torch.stack([data["yield_value"] for data in batch])

    out_views: dict[str, ViewBatch] = {}
    if views:
        from .views import assemble_view

        for view in views:
            assembled = [assemble_view(data, view) for data in batch]
            coords, channels, pad_mask = _pad_view_sequences(
                [a.coords for a in assembled],
                [a.channels for a in assembled],
                pad_value,
            )
            out_views[view.name] = ViewBatch(
                coords=coords,
                channels=channels,
                pad_mask=pad_mask,
                coord_names=assembled[0].coord_names,
                channel_names=assembled[0].channel_names,
            )
    else:
        # Legacy default: VI "main" view from DAP coords + VI channels.
        coords, channels, pad_mask = _pad_view_sequences(
            [data["dap"] for data in batch],
            [data["channels"] for data in batch],
            pad_value,
        )
        out_views["main"] = ViewBatch(
            coords=coords,
            channels=channels,
            pad_mask=pad_mask,
            coord_names=["dap"],
            channel_names=batch[0].get("channel_names"),
        )

    # The weather peer is no longer special-cased: when raw_weather is
    # enabled, the dataset's `dl_views` include a `weather` view
    # (coords:[weather_dap] channels:[weather.*]) that the loop above
    # assembles through assemble_view like any other view.

    # Stack derived features (fixed-size per sample, stored in sub-dict)
    derived_features = {}
    if "derived_features" in batch[0]:
        for key in batch[0]["derived_features"]:
            derived_features[key] = torch.stack(
                [sample["derived_features"][key] for sample in batch]
            )

    return G2FBatch(views=out_views, s=s, derived_features=derived_features)
