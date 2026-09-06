# Data Pipeline

This module handles reading, processing, splitting, and batching the G2F (Genomes to Fields) crop yield dataset for model training.

## Overview

Each data sample is a CSV file representing one pedigree in one environment/year. It contains a variable-length time series of 37 vegetation indices measured at irregular days after planting (DAPs), plus a scalar yield target.

**Pipeline summary:**

```
Raw CSV files
  → DataReader.load()             # read, validate, interpolate, extract
  → DatasetSplitter.split_data()  # train/val/test partitioning
  → _filter_to_modality_coverage  # drop samples missing required modality (e.g. genotype)
  → FeatureProcessor.fit_transform (train mode)
       or .load_and_transform      (predict mode, via cache keys)
  → G2FDataset.normalize          # z-score using train statistics
  → DataLoader + collate_fn       # assemble per-View (coords, channels), pad, stack features
  → G2FBatch                      # named Views + derived features, ready for the model
```

`FeatureProcessor` is the optional feature-extraction stage. It hosts
twelve registered processors — the `axis_source` derived-axis builder
(GDD / AGDD, priority −1), VI FPCA, phenomic / genomic / enviromic
kernels, the env-level `weather_kernel` (K_W from weather FPC scores),
three weather variants, metadata one-hots, the `raw_vi` VI-subset
slicer, and a generic Hadamard-kernel
interaction builder — dispatched uniformly via the processor registry
(`utils/data/processing/registry.py`). It runs on
the raw (pre-normalization) data so train-only fitting shares the
same statistical protocol as the FPCA baselines, then attaches
per-sample features under `derived_features` (or widens / slices the
`channels` matrix in place for weather concat / `raw_vi`, or attaches
`weather_dap` / `weather_values` / `weather_channel_names` for
raw-weather, or attaches `gdd` / `agdd` axes for `axis_source`). Adding a
new processor is a single `@register_processor`-decorated `BaseProcessor`
subclass — the orchestrator, validator, rank check, and coverage filter
pick it up through the registry without code changes. Details:
[`utils/data/processing/README.md`](processing/README.md).

**Views (which columns are coordinates vs. channels).** Each DL
set-encoder peer reads a named **View** — a config-driven selection of
*coordinate* axes (`coords`, e.g. `[dap]` or `[agdd]`) and *signal*
channels (`channels`, e.g. `["vi.*"]` or `["weather.*"]`) plus an optional
axis transform (`identity` / `dedup` / `warp`). Views are realised by
`utils/data/views.py::assemble_view` at collate time and carried in the
batch as `ViewBatch`es. This makes "which quantity is the time axis"
(DAP vs GDD vs AGDD) and "which signals are channels" a config decision —
see `utils/data/views.py`, `utils/data/axis_transforms/`, and the
`conf/axis/` group.

## Dataset

19 unique (environment, year) combinations across 12 environments and 2 years (2020–2021).
Each CSV file in `dataset-files/g2f/Pedigrees_Wide_Format_BLUEs/` represents one pedigree in one (Env, Year).

| Environment | 2020 | 2021 |
|-------------|------|------|
| DEH1        | 1142 | —    |
| IAH4        | —    | 1128 |
| MIH1        | 414  | —    |
| MNH1        | 395  | 415  |
| MOH1        | 1006 | —    |
| NEH1        | —    | 246  |
| TXH1        | 246  | 246  |
| TXH2        | 246  | 246  |
| TXH3        | 266  | 245  |
| WIH1        | 394  | 398  |
| WIH2        | 1127 | 1137 |
| WIH3        | 415  | 397  |

*Numbers are file counts (= number of pedigrees) per (Env, Year).*

- **Both years**: MNH1, TXH1, TXH2, TXH3, WIH1, WIH2, WIH3
- **2020 only**: DEH1, MIH1, MOH1
- **2021 only**: IAH4, NEH1

## Module Structure

| File | Purpose |
|------|---------|
| `base.py` | `BaseBatch` (ABC), `ViewBatch` (coords/channels/pad_mask + names), and `G2FBatch` dataclass (`views`, `s`, `derived_features`) |
| `reader.py` | `DataReader` — reads raw CSVs, validates, interpolates, extracts tensors |
| `helpers.py` | Utility functions: interpolation, case conversion, string matching |
| `views.py` | `View` spec + `resolve_channels` + `assemble_view` — selects per-View coords/channels by name from the flat sample dict and applies the axis transform |
| `axis_transforms/` | Axis-transform registry + `identity` / `dedup` / `warp` (the per-View curve reshapers) |
| `dataset.py` | `G2FDataset`, `_SplitDataset`, `g2f_collate_fn`, `_filter_to_modality_coverage` |
| `splitter.py` | `DatasetSplitter` — test/train splitting by filter or ratio (+ role-based `assign_roles`), returns `split_indices`; the DL val holdout is carved separately via eval-streams (`val_seed`) |
| `eval_streams/` | Eval-stream building blocks: `carve` validation strategies (the val holdout) + `observe` selectors (read-only probes over assigned rows) |
| `tasks.py` | Task-level data layer for the TNP: `NPTaskBatch`, `PrecomputedPool`, `TaskSampler`, `np_collate` (context/query task batching) |
| `augment.py` | Train-only per-modality timepoint subsampling (task- and sample-level collate augmentation) |
| `processing/` | `FeatureProcessor` orchestrator + `BaseProcessor` ABC + registry + 12 feature processors (incl. `axis_source`) — see [`processing/README.md`](processing/README.md) |

## Step-by-Step Pipeline

### 1. Reading Raw Data (`DataReader`)

`DataReader.load()` processes all CSV files in the data directory through these stages:

1. **Filename parsing** — Extracts metadata (environment, year, pedigree) from filenames using the pattern `{Env}.{Year}.{Inbred}.{Tester}.csv`.

2. **CSV loading** — Reads each file with pandas (supports `.csv`, `.xlsx`, `.xls`, `.json`).

3. **Case normalization** — Converts column names, index values, and cell values to lowercase for consistent matching (when `case_sensitive=False`).

4. **Validation** — Checks that pedigree/year/environment in each file match the filename metadata, that yield is unique per file, and that the vegetation index set is consistent across all files. Invalid files are optionally deleted.

5. **Canonical VI ordering** — VI rows are sorted alphabetically by name before tensorization, so column `j` of the `channels` matrix corresponds to the same VI across every file. The ordered names are emitted per sample as `channel_names` and exposed as `G2FDataset.vi_names`, consumed by `VIFPCAProcessor.vi_subset` (post-load score slice) and `RawVISubsetProcessor` (raw-curve column slice for the DL set encoder).

6. **Missing value interpolation** — Fills NaN holes using linear interpolation via `scipy.interpolate.interp1d`. **Important:** this only fills gaps within existing time steps; it does **not** regularize the grid. Each sample retains its own irregular DAP sequence. The pre-interpolation NaN positions are captured as a per-sample `vi_nan_mask` so consumers can recover the originally-real cells (used by `VIFPCAProcessor.missing_values='skip'` to keep interpolated cells out of fdapace fits — the R reference scheme).

7. **Tensor extraction** — Extracts five fields per file into a **flat** sample dict:
   - `dap`: shape `(T, 1)` — days-after-plant values parsed from column suffixes (the native coordinate axis)
   - `channels`: shape `(T, 37)` — VI measurements at each time step (post-interpolation)
   - `channel_names`: list of 37 VI names (the canonical column ordering of `channels`)
   - `vi_nan_mask`: shape `(T, 37)`, bool — True wherever the original cell was NaN
   - `yield_value`: shape `(1,)` — scalar yield from the `Yield.t.ha.BLUE` column

8. **Caching** — Processed data is cached to `{parent_of_data_dir}/.cache/g2f_data_{hash}.pt` (i.e. sibling to the CSV directory, not inside it) keyed on config parameters and file modification times. Set `cache_dir=False` to disable. The key is split-independent, so concurrent SLURM jobs share one file; the write is parallel-safe (PID-tagged temp + atomic `os.replace`, same pattern as the processor caches — see `processing/README.md` §"Parallel-safe writes").

**Returns:** `(data_dict, metadata_df)` where `data_dict[filename]` maps to the flat sample dict (`dap`, `channels`, `channel_names`, `vi_nan_mask`, `yield_value`). Derived axes (`gdd` / `agdd`) and weather fields (`weather_dap` / `weather_values` / `weather_channel_names`) are attached later by the `axis_source` / `raw_weather` processors when enabled.

### 2. Splitting (`DatasetSplitter`)

`DatasetSplitter.split_data()` carves out the **test** set and leaves the rest
as the **train** pool:

1. **Test split** — Selected by filter columns (e.g., `[{"Env": "IAH4"}]`) or by ratio.
2. **Train** — Everything left.

The DL **validation** holdout is no longer carved here. The legacy flat
`val_ratio` / `val_filter_columns` carve was removed (the DL eval-streams plan):
`_split_data` now calls `split_data(..., val_ratio=0.0, val_filter_columns=None)`,
and the validation set is carved from the train pool afterwards by the
eval-streams **`carve`** strategy (`_carve_val_from_train`, seeded by the
dedicated `dataset.val_seed`, default `0`). See `conf/eval_streams/` for the
stream bundles and [`processing/README.md`](processing/README.md) /
[`../../docs/cv_schemes.md`](../../docs/cv_schemes.md) for how `val` relates to
the FIT/OBSERVE/PREDICT roles.

Test filter format examples:
```yaml
# Hold out one (Env, Year) combination — AND of two conditions
test_filter_columns:
  - Env: DEH1
  - Year: 2020

# Hold out multiple environments across all years (OR within one dict)
test_filter_columns:
  - Env: [DEH1, IAH4]

# Match by pedigree, expand to all environments sharing that pedigree
test_filter_columns:
  - Pedigree: W10004_0086/PHP02
    _expand_: Env
```

Each dict in the list is one column condition. All dicts are AND'd (intersected).
Multiple values within a single dict are OR'd at the value level.

The test split is deterministic when `random_state` is set; the DL val carve is
deterministic under `dataset.val_seed` (decoupled from `random_state` and
`misc.seed`).

#### Role-based splits (`cv_spec`) and the common-female universe

When a `cv_spec` block is present (`+cv_spec=cv_2_1|cv_0_00`), the dataset
uses an explicit **role-based** split (`_build_role_based_split`) that
mirrors the R reference instead of the filter/ratio carve-out. Folds
are built on the **common-female set** — females present in *every*
environment — and only common females are masked and scored
(CV1/CV2/CV0/CV00). Two frames are in play, and the order they are used
matters:

- **The common-female universe is the full pre-coverage frame.** The set
  (and its `total_envs` denominator) is derived from `metadata_df`
  *before* the genotype coverage filter, matching R's `Metadata` script
  (`filter(n_envs == total_envs)` over the full phenotype). This is the
  single source of truth for "common"
  (`resolve_cv_assignment(common_universe_df=...)`).
- **Roles are assigned on the coverage-filtered (genotyped) frame** =
  R's `order`. The genotype filter (`genomic.coverage_mask`, which drops
  pedigrees absent from the dosage CSV) runs first; the surviving rows
  get FIT / OBSERVE / PREDICT roles.

**Consequence — `total_envs` is counted on the full frame.** Because the
universe is the full pre-coverage frame, ungenotyped plants can add
environments (or env-incidences) that the genotyped frame does not have:

- A female must appear in **every** environment of the *full trial
  design* to be common — including an environment that survives only via
  ungenotyped plants. A female genotyped in every *genotyped* env can
  therefore be **non-common** if it is absent from an all-ungenotyped
  env. This is intentional: it matches R, which derives `total_envs` from
  the full env list, not the genotyped subset.
- Ungenotyped plants influence only the *definition* of "common"; they
  are dropped by the coverage filter and never reach the splits, kernels,
  or model. A common female with no surviving genotyped row simply
  contributes zero rows.
- Rows with a missing `Pedigree` are excluded from the count (R's
  `filter(!is.na(Female))`), so the broadened universe cannot seed a
  spurious `"nan"` female. (Current G2F metadata has no null pedigrees —
  `Pedigree` is built as `Inbred/Tester` from the filename.)

Corollary: do not target a `cv_0_00` `heldout_env` that has no genotyped
rows — the held-out column would be empty in the genotyped frame, leaving
no CV0/CV00 rows to score (R has the same degeneracy).

### 3. Normalization (`G2FDataset`)

After splitting and any optional feature processing (Stage 2 in the
root README — feature processors run on raw, pre-normalization data so
their train-only fitting matches the FPCA baselines' protocol),
`G2FDataset` computes mean and standard deviation from the **training
set only**, then applies z-score normalization to all splits:

```
x_normalized = (x - train_mean) / train_std
```

Which keys are normalized is controlled by the `normalize` config dict
(keyed by the flat sample-dict field names):
```yaml
normalize:
  dap: true
  channels: true
  weather_concat_values: true    # consulted only when weather_concat is enabled (g2f.yaml ships true)
  yield_value: false
```

### 4. Batching and Padding (`g2f_collate_fn`)

The collate function is **View-driven**. It takes the dataset's
`dl_views` (the `target: dl` Views built from `dataset.views` config) and,
for each View, calls `assemble_view` per sample to produce a
`(coords[T,D], channels[T,C])` pair, then pads each View's sequences
independently to that View's max length with `pad_value` (default `0.0`),
recording a boolean `pad_mask` (`True` = padding). Each padded View
becomes a `ViewBatch`; fixed-size derived features are stacked unchanged.
(With no `views` argument the collate falls back to a single legacy `main`
View built directly from `dap`/`channels` — byte-equivalent to the
config-driven `main`.)

**Output:** a `G2FBatch` with:

| Field | Type / shape | Description |
|-------|--------------|-------------|
| `views` | `dict[str, ViewBatch]` | One entry per DL View (e.g. `"main"`, and `"weather"` when `raw_weather` is enabled) |
| `s` | `(B, 1)` | Yield target |
| `derived_features` | `dict[str, (B, d_name)]` | Optional fixed-size features from the processing layer (`vi_fpc_scores`, `genomic_add`, `phenomic_relmat`, `weather_fpc_scores`, `enviromic_relmat`, ...) |

Each `ViewBatch` carries:

| Field | Shape | Description |
|-------|-------|-------------|
| `coords` | `(B, T_max, D)` | The View's coordinate axes (`D=1` for `[dap]`/`[agdd]`; `D>1` for multi-axis coords) |
| `channels` | `(B, T_max, C)` | The View's signal channels (`C=37` for VI `vi.*`; `W` for weather `weather.*`; widened by axis-as-channel selectors) |
| `pad_mask` | `(B, T_max)` | `True` at padded positions |
| `coord_names` | `list[str]` | Names of the `D` coordinate columns |
| `channel_names` | `list[str]` | Names of the `C` channel columns |

`derived_features` defaults to `{}`. The set of Views present depends on
config: the VI `main` View is always built; the `weather` View appears
only when `raw_weather` is enabled (the dataset drops Views whose backing
processor is disabled). Each of the TNP `SampleTokenizer`'s per-modality
set encoders is bound to one View by name (`vi_view="main"` /
`weather_view="weather"`) and reads `batch.views[<name>]` — so adding a
modality is a second View + a second sub-encoder (fused by the tokenizer's
`fuse_x`), no bespoke plumbing.

## Usage

```python
from utils.data import G2FDataset, g2f_collate_fn
from torch.utils.data import DataLoader

# Initialize dataset (reads, splits, normalizes). The held-out test set is the
# filter; the DL val holdout is configured via `eval_streams` (a `carve`
# stream seeded by `val_seed`), not a val filter.
dataset = G2FDataset(
    data_dir="./dataset-files/g2f/Pedigrees_Wide_Format_BLUEs",
    normalize={"dap": True, "channels": True, "yield_value": False},
    test_filter_columns=[{"Env": "IAH4"}],
    val_seed=0,
)

# Get a split as a PyTorch Dataset
train_ds = dataset.get_split("train")

# Create DataLoader with the custom collate function. Production passes the
# dataset's DL Views so collate assembles them; here we let it fall back to
# the legacy `main` View.
train_loader = DataLoader(
    train_ds,
    batch_size=...,  # see conf/data/g2f.yaml
    shuffle=True,
    collate_fn=lambda batch: g2f_collate_fn(
        batch, pad_value=0.0, views=dataset.dl_views
    ),
)

# Iterate
for batch in train_loader:
    main = batch.views["main"]
    # main.coords:   (B, T_max, D)   # D=1 for coords=[dap]
    # main.channels: (B, T_max, 37)
    # main.pad_mask: (B, T_max)
    # batch.s:       (B, 1)
    ...
```

## Configuration

The canonical defaults live in `conf/data/g2f.yaml` — every processor
under `dataset.processing.*` is `enabled: false` there, so any
production model fragment opts in by overriding individual blocks.
See [`processing/README.md`](processing/README.md) for the per-processor
parameter contract.
