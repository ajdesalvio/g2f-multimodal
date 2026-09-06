# G2F Multimodal

Crop yield prediction from vegetation index (VI) time series using a Transformer Neural Process and functional data analysis baselines.

## Overview

This project uses data from the **Genomes to Fields (G2F)** initiative (2020-2021) to predict crop yield from remotely sensed vegetation index curves. Each sample is a pedigree-by-environment combination with a variable-length time series of **37 vegetation indices** measured at irregular days after planting (DAPs), paired with a scalar yield target.

Two modeling approaches are implemented:

1. **Transformer Neural Process (TNP)** — `set_func.models.TransformerNeuralProcess`. The unit of computation is a *task*: a **context set** of samples with observed yields plus a **query set** to predict. A `SampleTokenizer` treats each sample's irregular time series as a *set* of (DAP, VI-vector) elements, encodes it with a per-sample `TransformerSetEncoder` (plus optional weather-view and genomic-feature branches), and fuses the result with a `(y, density)` encoding into one token per sample — the density channel flags whether the yield is observed (context) or missing (query). A cross-sample transformer encoder (`ISTransformerEncoder` with pseudo-token memory in the shipped configs; `PerceiverEncoder` and `EfficientQueryTransformerEncoder` are also available) refines the query tokens against the context, and a decoder maps each to a yield prediction via an output **head**: a distributional head (e.g. `HeteroscedasticNormalLikelihood`, returning a `GaussianPrediction`) or a `PointHead` (returning a `PointPrediction` — a point estimate with no density).
2. **FPCA + Regression (Baselines)** — Functional Principal Component Analysis (via R's `fdapace`) applied independently to each of the 37 VIs, producing FPC score features fed into a regression model (OLS, GPR, GBLUP, or BGLR — a multi-kernel Bayesian RKHS regressor; the shipped env-GxE/weather-GxE composers use BGLR and GBLUP). FPCA is fitted on training data only; test scores are projected via the CE/BLUP formula. Auxiliary modalities (genotype, phenomic RKHS, weather FPCA, enviromic similarity) are available as the same processors the DL pipeline uses — see the Stage 2 listing below.

## Pipeline

Data ingestion, optional feature processing, and model consumption all run
through the same layered path. The feature-processing stage is entirely
opt-in — a vanilla VI-only run skips every processor and goes straight from
the splitter to normalization. Multi-modal combos toggle processors on via
`dataset.processing.*.enabled`, and the orchestrator attaches their outputs
to each sample as either `derived_features` (fixed-size MLP inputs), extra
per-sample fields backing a second set-encoder **View** (`weather_dap` /
`weather_values` for the weather branch), a widened `channels` matrix
(weather-concat / `raw_vi`), or derived time axes (`gdd` / `agdd` from
`axis_source`).

The full pipeline is broken into three stages below — ingestion,
optional feature processing, and batching + model — so each piece is
small enough to read without zooming.

### Stage 1 · Ingestion and splitting

```
Raw G2F CSV files (dataset-files/g2f/Pedigrees_Wide_Format_BLUEs/)
  → DataReader: load · validate · interpolate · extract (DAP, VI, yield) tensors
  → DatasetSplitter: train / test by filter_columns or ratio (val carved later via eval-streams, seeded by val_seed)
  → if no processor enabled → Stage 3 (skip feature processing)
    else                    → Stage 2
```

### Stage 2 · Feature processing *(optional)*

Runs only when at least one `dataset.processing.*.enabled` flag is
set. Each processor writes its output into `derived_features` (fixed
feature dicts consumed by MLP feature branches), or in place on the
per-sample `channels` matrix (weather-concat / `raw_vi`), or as extra
per-sample fields (`weather_*` for the weather View, `gdd` / `agdd` axes).

```
From Stage 1
  → _filter_to_modality_coverage: drop samples missing required modality
                                  (e.g. genomic drops pedigrees not in the dosage CSV)
  → FeatureProcessor: train  · .fit_transform(ds)
                       predict · .load_and_transform(ds, cache_keys)
  → run enabled processors in priority order:
       -1. axis_source       → attach gdd / agdd derived time axes (from weather GDD)
        0. metadata_features → derived_features[env_onehot, pedigree_onehot, ...]
        1. vi_fpca           → derived_features[vi_fpc_scores]              (R fdapace, train-only fit; axis: dap|agdd)
        2. phenomic          → derived_features[phenomic_*]                 (kernel over vi_fpc_scores)
        3. weather_fpca      → derived_features[weather_fpc_scores]         (per-var FPCA; axis: dap|agdd)
        4. weather_kernel    → derived_features[weather_relmat, ...]        (env-level K_W eigen V·√D)
        5. genomic           → derived_features[genomic_*]                  (Van Raden / Vitezica GRM)
        6. enviromic         → derived_features[enviromic_*]                (env similarity; axis: dap|agdd)
        7. raw_weather       → weather_dap / weather_values / weather_channel_names (the weather View; axis: dap|agdd)
        8. raw_vi            → slice the channels matrix to a VI subset (in place; DL)
        9. weather_concat    → widen the channels matrix from 37 to 37+W (in place)
       10. interaction       → derived_features[gxea, gxwa, pxe, pxw, ...]  (Hadamard k-way kernels)
  → To Stage 3
```

### Stage 3 · Normalize, collate, and model forward

```
From Stage 1 or 2
  → G2FDataset._normalize_datasets: z-score using train statistics
  → per DL View, assemble_view → (coords, channels), pad + mask; stack derived_features
  → TaskSampler / PrecomputedPool (utils/data/tasks.py): draw tasks — a context set
    + a query set — from the training-visible pool
  → NPTaskBatch: context + query, each a canonical [B, n, ...] G2FBatch of
              views{ "main": ViewBatch, ["weather": ViewBatch] } · s · derived_features
              (ViewBatch = coords · channels · pad_mask · coord_names · channel_names)
  → TransformerNeuralProcess (tokenize → cross-sample encode → decode → head):
        - SampleTokenizer — one token per sample:
            vi branch      — TransformerSetEncoder over views["main"] (coords, channels, pad_mask)
            weather branch — optional; TransformerSetEncoder over views["weather"]
            geno branch    — optional; MLP over derived_features (e.g. genomic_add)
            fuse_x(concat branches) + (y, density) encoding → token_proj → [B, n, d_model]
        - transformer_encoder — cross-sample: ISTransformerEncoder (pseudo-token memory)
                          | PerceiverEncoder | EfficientQueryTransformerEncoder;
                          refines query tokens against context tokens
        → MLPDecoder → head (BaseHead): HeteroscedasticNormalLikelihood → GaussianPrediction Normal(μ, σ)
                                          (or PointHead → PointPrediction for a point estimate)
  → Yield prediction for the query samples
```

Single-modality VI models use only the `vi` branch; multi-modal
experiments under `conf/experiment/dl/` add the optional weather
set-encoder branch and the genomic MLP branch. See
[`utils/data/processing/README.md`](utils/data/processing/README.md)
for per-processor details.

## Project Structure

```
neural-process/
├── train.py / eval.py     # DL training + evaluation entry points (@hydra.main shims)
├── set_func/              # TNP model library
│   ├── core/              # Building blocks: encoders/ (TransformerSetEncoder), np/ cross-sample
│   │                      #   encoders (ISTransformer / Perceiver / EfficientQueryTransformer),
│   │                      #   attentions/, attention_layers/, transformers/, tokenizers/, decoders/, mlp
│   ├── models/            # neural_process.py — TransformerNeuralProcess + SampleTokenizer
│   ├── heads/, likelihoods/, predictions/   # Output heads + predictive distributions
│   └── utils/             # aggregate.py (Aggregator / PMAAggregator), helpers.py
├── utils/
│   ├── data/              # G2FDataset, DataReader, DatasetSplitter, G2FBatch, Views + axis_transforms,
│   │   │                  #   tasks.py (NPTaskBatch, TaskSampler, PrecomputedPool)
│   │   └── processing/    # FeatureProcessor orchestrator + 12 processors — see processing/README.md
│   └── experiment/        # Hydra setup, Lightning wrapper, metrics, run helpers
├── baselines/             # FPCA + sklearn regressor baseline (R fdapace + Python) — see baselines/README.md
├── conf/                  # Hydra config tree (data, optim, cv_spec, model, experiment, axis, augment, vi_subset, weather_subset)
├── cv/                    # CV orchestration (env_year_loo / cv_2_1 / cv_0_00 / random_kfold): schemes/, scoring, folds, slug_util — see cv/README.md
├── scripts/
│   ├── jobs/cv/           # SLURM job templates + submit scripts (cv_0_00_2_1 unified launcher, env_year_loo, random_kfold)
│   ├── cv/                # make_female_folds.py — alternative (native-RNG) fold table
│   ├── export_raw_metrics.py            # per-split metrics -> tidy_all_metrics.csv + wide/ tables (the shared results layout)
│   ├── compare_tnp_vs_kernel_paired.py  # paired TNP-vs-kernel comparison on identical splits (macro/pooled RMSE, Tiezzi r)
│   └── patch_bglr.sh      # rank-1 RKHS patch for BGLR (see baselines/R_HPC_SETUP.md)
├── environments/          # Captured python+R version pins (production vs dev)
├── docs/                  # CV-scheme docs
├── dataset-files/         # Data directory (not tracked in git)
└── artifacts/             # Output directory for checkpoints, metrics, results
```

Detailed structure of the data-processing layer is in
[`utils/data/processing/README.md`](utils/data/processing/README.md).
Per-config-group purpose is in the `conf/` table below.

**DL config file naming.** Names under `conf/{model,experiment}/dl/` read
`<family>_<modalities>_<geno>[__<facet>…]__[<size>]__<loss>` — `__`-separated
facets, with the **loss (`__mse`/`__nll`) always the final facet** and an
**optional size letter (`__S`/`__L`) immediately before it** (omitted when a
family ships a single size, as the `istnp` configs do). See
[`baselines/README.md`](baselines/README.md) ("DL slug grammar") for the full
token grammar.

## Data

**The dataset is not part of this repository — it is shared separately**
(see the paper's data availability statement). The code expects it under
`dataset-files/g2f/` at the repository root. All paths below are the
defaults from [`conf/data/g2f.yaml`](conf/data/g2f.yaml); every one of
them is a Hydra config value, so a dataset stored elsewhere just needs
`dataset.data_dir=...` (and the `dataset.processing.*.csv_path=...`
entries for the modalities you enable) overridden on the CLI or in the
config.

### Expected layout

```
dataset-files/g2f/
├── Pedigrees_Wide_Format_BLUEs/          # REQUIRED — the core VI + yield samples
│   └── {Env}.{Year}.{Parent1}.{Parent2}.csv   (one file per pedigree × env-year, ~10k files)
├── EnvRtype_Weather_Data_Cleaned_V2.csv  # required for weather / enviromic / GDD-AGDD features
├── G2F_2020_2021_Genomic_Data.csv        # required for genomic features
├── Pedigree_Overlap_Genomic_Phenomic.csv # provenance only — not read by the code
└── EnvRType_Weather_Variable_Descriptions.xlsx  # documentation only — not read by the code
```

The two auxiliary CSVs are only needed when the corresponding processors
are enabled: a plain VI-only run (`dataset.processing.*` all disabled)
touches nothing but `Pedigrees_Wide_Format_BLUEs/`. The shipped
env-GxE/weather-GxE composers and the multi-modal TNP configs need both.

### File formats

**1. `Pedigrees_Wide_Format_BLUEs/{Env}.{Year}.{Parent1}.{Parent2}.csv`
(one sample each).** A sample is one pedigree grown in one
(environment, year). There are **19 (Env, Year) combinations** across 12
environments and 2 years (2020–2021), ~10k files total — see
[utils/data/README.md](utils/data/README.md) for the per-env inventory.
Each file has **37 rows — one per vegetation index** — and these columns:

| Column | Content |
|---|---|
| `Pedigree` | `"{Parent1}/{Parent2}"` — must match the filename |
| `Vegetation.Index` | VI name (`BI`, `GLI`, `NGRDI`, `VARI`, … 37 total) |
| `Year`, `Env` | `{Year}` and `"{Env}.{Year}"` — must match the filename |
| `VI.BLUE.<DAP>` | one column per flight date, e.g. `VI.BLUE.25 … VI.BLUE.103` — the VI BLUE at that day-after-planting. The DAP grid is irregular and differs per (Env, Year) |
| `Yield.t.ha.BLUE` | scalar yield target (t/ha), repeated on every row |

`DataReader` validates file content against the filename metadata and
handles missing VI cells by linear interpolation (recording them in
`vi_nan_mask`; the FPCA pipeline's default `missing_values: skip` policy
drops them again before fdapace fits).

**2. `EnvRtype_Weather_Data_Cleaned_V2.csv` (daily env-level weather).**
One row per (Env, day): key columns `DAP`, `Env` (`"{Env}.{Year}"`)
followed by ~40 daily weather variables (`T2M*`, `PRECTOT`, `VPD`,
`GDD`, `PTR`, `PTT`, …; the xlsx documents each). Consumed by the
weather processors (`weather_fpca`, `weather_kernel`, `raw_weather`,
`weather_concat`), `enviromic`, and by `axis_source`, which reads the
`GDD` column to build the cumulative AGDD time axis. The shipped
configs use `PTR` as the single weather variable (`weather_subset: ptr`)
and the CSV `GDD` column for the `__t-agdd` variants.

**3. `G2F_2020_2021_Genomic_Data.csv` (SNP dosage matrix).** One row per
pedigree: a `Pedigree` column (same `"P1/P2"` format) followed by
~239k SNP columns with additive dosage codes `{0, 1, 2}`. Consumed by
the `genomic` processor (VanRaden additive / Vitezica dominance GRMs,
raw-PCA, or literal dosage). Samples whose pedigree is absent from this
CSV are dropped by the modality-coverage filter when genomic features
are enabled. The first read parses the full matrix (several minutes)
and is then content-cached.

### Cache

On first use the pipeline creates `dataset-files/g2f/.cache/`
(`{parent_of_data_dir}/.cache/`) holding content-addressed processor
caches (parsed genomic dosage, fitted FPCA bases, kernels, …). It is
safe to delete at any time — everything regenerates; cache keys include
the fitted-state config, so config changes never reuse stale entries.

**Data splitting** is configured in `conf/data/g2f.yaml` (with the held-out
fold set per run — `dataset.test_filter_columns=[{Env: …},{Year: …}]` plus
`misc.fold_env`, or the role-based `+cv_spec=env_year_loo` group) and handled by
`DatasetSplitter`. Under LOO cross-validation, the held-out (environment,
year) pair is the test fold and all remaining (env, year) combinations form
the training set.

Splitting supports both metadata-based filters (`test_filter_columns`) and random ratio-based splits (`test_ratio`). When filters are provided, ratios are ignored.

## Quick Start

### Prerequisites

**Python** (3.11+; production runs on 3.11.5 on HPRC Grace, dev on 3.13.1 — see [environments/](environments/)):
```bash
pip install torch lightning hydra-core omegaconf scikit-learn scipy pandas numpy joblib wandb
```

**R** (for FPCA baseline only — see [baselines/README.md](baselines/README.md) for details):
```r
install.packages(c("fdapace", "optparse", "jsonlite", "BGLR", "sommer"))
# BGLR then needs the one-line rank-1 patch: bash scripts/patch_bglr.sh
```

> **Exact environment pins.** The commands above install whatever is
> current on CRAN / PyPI. For the exact versions that produced the
> reported metrics, see
> [`environments/`](environments/): `hprc-grace/` has the production
> pin on Texas A&M HPRC Grace (Python 3.11.5 + PyTorch 2.5.1 CUDA 12.4
> + R 4.4.1 + fdapace 0.6.0 + 30 transitive R deps), and
> `local-macbook/` has the dev/debug pin on macOS (Python 3.13.1 +
> PyTorch 2.6.0 + R 4.5.2).

### Training a Deep Learning Model

Training entry is `train.py`, a three-line `@hydra.main` shim that
composes the root config in `conf/config.yaml` and delegates to
`run_training`. Pick an experiment via the `+experiment=dl/<slug>`
override; each experiment composer pulls its `cv` + `data` +
`model/dl` fragments from the `conf/` tree.

**TNP, VI only:**
```bash
python train.py +experiment=dl/istnp_vi__vienc-tfm__aug-vi__nll
```

**TNP, VI + genomic GRM rows** (or the full VI + weather + genomic
composer, `istnp_vi_wthr_ga_gd__grm-rows__vienc-tfm__wthrenc-tfm__aug-vi-wthr__nll`):
```bash
python train.py +experiment=dl/istnp_vi_ga_gd__grm-rows__vienc-tfm__aug-vi__nll
```

**Override config values from the command line** (bare `key=value`
— no `--` prefix; `@hydra.main` parses these natively):
```bash
python train.py +experiment=dl/istnp_vi__vienc-tfm__aug-vi__nll misc.seed=42 misc.epochs=200 optimizer.lr=1e-3
```

**Resume from a checkpoint:**
```bash
python train.py +experiment=dl/istnp_vi__vienc-tfm__aug-vi__nll \
    misc.checkpointing.resume_from=artifacts/.../checkpoints/last.ckpt
```

Artifacts (checkpoints, metrics) are saved to `artifacts/<name>/seed=<seed>/`.

### Evaluating a Trained Model

```bash
python eval.py +experiment=dl/istnp_vi__vienc-tfm__aug-vi__nll
```

The evaluation script evaluates **every present checkpoint in one process** (the dataset + processing are built once). Checkpoints are discovered as per-metric `best_<metric>.ckpt`, `best_frozen_<metric>.ckpt` (when `freeze_on_plateau` was active), and `last.ckpt`; **every** checkpoint's outputs are suffixed by its label — `metrics_<label>.json` / `predictions_<label>.csv` (e.g. `metrics_best_pearson_r.json`, `metrics_last.json`) — so there is no bare `metrics.json` for a DL run and a metric-selected checkpoint is always attributable to its metric. Test metrics are RMSE, Pearson r, Spearman r, `r_w`/`rho_w`, and — when the head is distributional — `loglik` (mean predictive log-likelihood).

### Running the FPCA Baseline

For full usage instructions (training, prediction, CLI options, output artifacts)
and a detailed explanation of the FPCA pipeline, see
[baselines/README.md](baselines/README.md).

```bash
python -m baselines.fpca_train \
  +experiment=fpca/fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse
```

## Configuration System

Configs are YAML files under `conf/`, composed and resolved by
Hydra 1.3 + OmegaConf. Entry points (`train.py`, `eval.py`,
`baselines/fpca_train.py`, `baselines/fpca_predict.py`) are
`@hydra.main`-decorated shims that compose the root
`conf/config.yaml` and delegate to a non-decorated helper function.

Composition works via `defaults:` lists on experiment composers
under `conf/experiment/{dl,fpca}/<slug>.yaml`, each pulling in
an `optim` fragment (`conf/optim/default.yaml`), a `data` fragment
(`conf/data/g2f.yaml`), and a `model` fragment
(`conf/model/{dl,fpca}/<slug>.yaml`). Every fragment uses
`# @package _global_` so its keys land at root — runtime consumers
read `optimizer` / `scheduler` / `dataset` / `model` as root keys.

The system supports:

- **Experiment composers**: select via `+experiment=<dir>/<slug>`
  (e.g. `+experiment=dl/istnp_vi__vienc-tfm__aug-vi__nll`, or one of the shipped FPCA composers
  under `conf/experiment/fpca/` — 7 BGLR, 4 GBLUP, and 1 OLS composer;
  the env-GxE / weather-GxE BGLR/GBLUP families each ship a default
  `dap` and a `__t-agdd` (AGDD) axis variant).
- **CLI overrides**: bare `key=value` (no `--` prefix) — Hydra parses
  them natively. Use `+key=value` to force-add a key not declared in
  the schema (e.g. `+dataset.smoke_n=100` for a runtime-only subsample).
- **Interpolation**: `${params.embed_dim}`, `${model_name}`, relative
  references `${.seed}`.
- **Custom resolvers**: `${eval:'2 * ${params.embed_dim}'}` for
  arithmetic; `${cond:${bool},"true_str","false_str"}` for
  conditional strings (used by `model_run_name` to suffix
  `-fold=DEH1.2020` when `misc.fold_env` is set).

### `conf/` layout

- **`conf/config.yaml`** — Root composer — pulls `misc: base`, `_self_`.
  `hydra.run.dir` set to `${misc.artifacts_dir}`.
- **`conf/misc/base.yaml`** — Shared `misc.*` block — experiment name, seed,
  epochs, checkpointing (D1-normalized form), W&B settings, artifact paths.
- **`conf/data/g2f.yaml`** — G2F dataset target — file glob, metadata columns,
  processing defaults (all processors `enabled: false` by default).
- **`conf/optim/default.yaml`** — DL optimization schedule — optimizer,
  scheduler, `sched_name` / `hp_tag` slug tags.
- **`conf/model/dl/<slug>.yaml`** — DL model fragment — architecture, params,
  task sampler, phase configs. One self-contained TNP fragment per modality
  combo (VI-only / +genomic GRM rows / +weather+genomic).
- **`conf/model/fpca/<slug>.yaml`** — FPCA model fragment — processors,
  feature_keys, regressor, loss. R-mirror modality-ablation family (see
  `baselines/README.md`).
- **`conf/experiment/dl/<slug>.yaml`** — DL experiment composer — `defaults:`
  list pulls `optim` + `data` + `model/dl/<slug>` + `vi_subset` +
  `weather_subset` + `axis` + `augment`. One per DL model.
- **`conf/experiment/fpca/<slug>.yaml`** — FPCA experiment composer — same
  shape. One per FPCA model.
- **`conf/vi_subset/{full,ngrdi}.yaml`** — VI ablation axis — `full` (default)
  is a no-op, `ngrdi` clobbers both `vi_fpca.vi_subset` (FPCA score slice) and
  `raw_vi.vi_subset` (DL raw-curve slice) to `[NGRDI]`. Sets the `vi_chan` slug
  piece → the VI encoder's channel token reads `vi.c=ngrdi`.
- **`conf/weather_subset/{full,ptr}.yaml`** — Weather ablation axis — sets the
  weather variable subset atomically across all four weather-consuming
  processors (`weather_vars` on enviromic / weather_concat / raw_weather,
  `emit_vars` on weather_fpca). `ptr` reproduces the R reference's env-GxE/weather-GxE (PTR only).
  Sets the `wthr_chan` slug piece → the weather encoder's channel token reads
  `wthr.c=ptr`. (Restricting weather_fpca's *fit* set — the compute knob,
  distinct from `emit_vars` — is `fit_vars`, set per-experiment, e.g. on the
  expensive AGDD axis.)
- **`conf/axis/*.yaml`** — Time-axis ablation. Names follow
  `<family>_t-<time>[_c-<chan>][_<transform>]` — `family` ∈
  `shared`/`fpca`/`dl`, axes after `t-` are time/coords, axes after `c-` are
  channels. `shared_t-dap` (default) is a no-op; the others flip the
  fdapace/View coordinate (`shared_t-agdd` flips every FPCA + weather
  consumer). Each sets the per-encoder slug pieces (`vi_time`/`vi_aug`,
  `wthr_time`/`wthr_aug`) that render into the `vi.t=…c=…` / `wthr.t=…c=…`
  tokens of `model_name`. `dl_t-dap-gdd-agdd` feeds gdd+agdd to both
  set-encoders as extra **coordinates** (rendered in the `vi.t=`/`wthr.t=` time
  token, e.g. `vi.t=dap+gdd+agdd`); the same axes placed as **channels** instead
  render under the `.c=` token (`vi.t=dap.c=…+gdd+agdd`) — `.t=` vs `.c=`
  placement is the only slug difference. `dl_t-agdd_wthr-c-agdd` is the mixed
  axis (VI coords [agdd]; weather coords [weather_agdd] + abs-agdd channel).
  Composed via `axis=<name>`.

### Feature-subset ablation

The `vi_subset` and `weather_subset` config groups are independent
ablation axes layered on top of the existing model fragments. Both
default to `full` (no-op — the model fragment's hardcoded values
persist). Switching to a non-default group clobbers the relevant
processor keys *after* the model fragment composes, and adds a slug
suffix to `model_run_name` so artifact paths and W&B run names stay
collision-safe.

```bash
# Full features (default)
python -m baselines.fpca_train +experiment=fpca/<slug>

# The R-reference env-GxE/weather-GxE reproduction — single VI (NGRDI), single weather var (PTR)
python -m baselines.fpca_train +experiment=fpca/<slug> \
    vi_subset=ngrdi weather_subset=ptr

# Mix and match — independent axes
python -m baselines.fpca_train +experiment=fpca/<slug> vi_subset=ngrdi
```

The chosen subset names come from the R reference's V2 LOEO bundle:
`FPC_Scores_BLUEs_{DAP,AGDD}_LOEO_Projected.csv` contain FPC scores
for **NGRDI** only, and `Weather_FPC_Scores_{DAP,AGDD}_LOEO_Projected.csv`
contain scores for **PTR** only — confirming the R reference's env-GxE/weather-GxE setup
runs on those single-variable subsets, not the script-default fulls.

To add a new ablation point (e.g., a top-5 VI subset), drop a new
`conf/vi_subset/<name>.yaml` setting both `dataset.processing.vi_fpca.vi_subset`
(FPCA models) and `dataset.processing.raw_vi.vi_subset` (DL models) to
your list, plus `vi_chan: <name>` (the VI channel-subset slug piece, which
renders into the `vi.c=<name>` token of `model_name`). No edits to existing
configs needed.

## Model Architecture

The deep learning model is a **Transformer Neural Process**
(`set_func.models.TransformerNeuralProcess`). The unit of computation is a
*task* — a context set of samples with observed yields plus a query set to
predict. A `SampleTokenizer` turns each sample into one token, a
cross-sample transformer encoder refines the query tokens against the
context, and a decoder + output head produce the predictive distribution
over the query yields.

```
Input: NPTaskBatch — context + query, each a canonical [B, n, ...] G2FBatch:
  views["main"]: ViewBatch    — coords [B, n, T, D] (D=1 for [dap]) · channels [B, n, T, 37] · pad_mask [B, n, T]
  views["weather"]: ViewBatch — optional; coords [B, n, Tw, 1] · channels [B, n, Tw, W] · pad_mask [B, n, Tw]
  s: [B, n]                   — yield target (observed for context, withheld for query)
  derived_features: dict      — e.g. {"genomic_add": [B, n, 50]}

SampleTokenizer — one token per sample, [B, n, d_model]:
  vi branch:      TransformerSetEncoder over views["main"] (coords, channels, mask=pad_mask)
  weather branch: TransformerSetEncoder over views["weather"]              — optional
  geno branch:    MLP over derived_features[geno_key] (e.g. genomic_add)   — optional
  fuse_x: concat branch outputs → MLP → d_x (= d_model)
  y path: (y, density) — density flags a missing y (1 = query, 0 = context; query y zeroed)
          → y_encoder → d_y
  token_proj: [d_x + d_y] → d_model

transformer_encoder — cross-sample, config-selected (set_func/core/np/):
  ISTransformerEncoder             — pseudo-token (induced-set) memory, O(Nc·K); the shipped istnp_* configs
  PerceiverEncoder                 — latent-bottleneck cross-attention
  EfficientQueryTransformerEncoder — context self-attention + query cross-attention
  zq_out = transformer_encoder(zc, zq)                     → [B, nq, d_model]

decoder (MLPDecoder):
  z_decoder → [B, nq, 2]                                   — map to (mean | log_scale)

head (HeteroscedasticNormalLikelihood):
  → Normal(mu=softplus(mean), sigma=0.99·softplus(log_scale)+0.01)  — per-row predictive variance
```

A single-modality VI model has just the `vi` branch. Multi-modal combos
(see configs under `conf/model/dl/`) add the genomic MLP branch (GRM
rows) and the weather set-encoder branch. The `dataset.processing`
config block enables the feature processors that populate
`derived_features` / the weather View — see
[`utils/data/processing/README.md`](utils/data/processing/README.md).
Task sampling (context/query sizes, batches per epoch, the val/test
conditioning pool) is configured per-model under `task_sampler:`.

### Set aggregation

The per-sample set encoders (`TransformerSetEncoder`) pool their set-level
output into a fixed-size embedding via a configurable aggregator.

- **`Aggregator`** (`set_func.utils.Aggregator`) — parameter-free NaN-safe
  reductions: `mean`, `sum`, `min`, `max`, `quantile`. Can compose multiple
  (concatenates along the feature dim).
- **`PMAAggregator`** (`set_func.utils.PMAAggregator`) — Pooling by Multihead
  Attention from the Set Transformer paper (Lee et al. 2019). Learnable seed
  vectors attend over the set; `num_seeds=1` is a drop-in replacement for
  `mean`. Separate from the set encoder's own attention layers — the pool's
  heads are configured independently. The shipped `istnp_*` configs use PMA
  with `num_seeds=1` for the VI (and weather) sub-encoders.

## Metrics

Up to six metrics are reported per scored `cv_label`: **RMSE**, **Pearson r**, **Spearman r**, two within-environment (`Env.Year`) inverse-variance weighted correlations — **`r_w`** (Pearson) and **`rho_w`** (Spearman) — and, for DL models with a distributional head, **`loglik`** (mean predictive log-likelihood; absent for the point-estimate and FPCA/BGLR paths, which carry no predictive density). All metrics operate on raw, unnormalized yield values — `yield_value` normalization is disabled in `conf/data/g2f.yaml` and the FPCA baseline explicitly overrides it — so RMSE values are in the same units (t/ha BLUEs) and are directly comparable across pipelines.

### Implemented metrics

| Metric | Symbol | DL location | Baseline location |
|---|---|---|---|
| Root Mean Squared Error | RMSE | `utils/experiment/metrics.py` — `RMSE` | `cv/scoring.py` — `compute_metrics` (shared) |
| Pearson correlation coefficient | r | `utils/experiment/metrics.py` — `PearsonCorrelation` (key: `pearson_r`) | `cv/scoring.py` — `compute_metrics` (shared) |
| Spearman rank correlation coefficient | ρ | `utils/experiment/metrics.py` — `SpearmanCorrelation` (key: `spearman_r`) | `cv/scoring.py` — `compute_metrics` (shared) |
| Mean predictive log-likelihood | `loglik` | `utils/experiment/metrics.py` — `LogLikelihood` (key: `loglik`; distributional heads only) | n/a (baselines carry no predictive density) |
| Within-block weighted Pearson (Tiezzi et al. 2017) | `r_w` | `cv/scoring.py` — `weighted_block_correlation` (shared, via `score_by_label`) | `cv/scoring.py` — `weighted_block_correlation` (shared) |
| Within-block weighted Spearman | `rho_w` | `cv/scoring.py` — `weighted_block_correlation(method="spearman")` (shared) | `cv/scoring.py` — `weighted_block_correlation` (shared) |

**`r_w` / `rho_w`** are the within-environment, inverse-variance weighted correlations of Tiezzi et al. (2017): a correlation is computed *within* each `Env.Year` block and the per-block values pooled by inverse-variance weight, measuring how well the model ranks genotypes inside an environment (removing between-environment mean differences). `r_w` uses Pearson, `rho_w` uses Spearman (Pearson on within-block ranks). Each carries a `*_n_blocks` count of contributing environments. There is intentionally no weighted RMSE. See [cv/README.md](cv/README.md) for the formula and skip rules.

**RMSE** is `sqrt(mean((y_pred - y_true)^2))` in both pipelines.

**Pearson r** is the standard correlation coefficient. The baseline uses `scipy.stats.pearsonr`; the DL implementation computes it directly from centered predictions and targets.

**Spearman r** is Pearson correlation applied to the ranks of the inputs — it captures monotonic (not necessarily linear) agreement between predictions and targets. The baseline uses `scipy.stats.spearmanr`; the DL implementation computes it directly from `argsort().argsort()` ordinal ranks (ties broken by position, adequate for continuous yield targets where ties are vanishingly rare; a constant-input guard short-circuits to 0.0 to avoid spurious correlations from rank-promotion of constant vectors). Spearman is useful as a selection metric in breeding applications where correctly ranking pedigrees matters more than absolute yield prediction.

### Batch-level metric aggregation (pooled, batch-size invariant)

Epoch metrics in the deep learning pipeline are computed **once over the
pooled epoch predictions**, so they match the split-level values the FPCA
baseline computes and are independent of the dataloader batch size:

- `train`/`val`/`test` steps log per-batch values under `*_step` keys (the
  live curves) and buffer the raw `(predicted-mean, target)` tensors.
- At epoch end, `LitWrapper._eval_epoch_end` concatenates the buffer and runs
  the metric once → `train_*`/`val_*`/`test_*` epoch keys (e.g.
  `val_pearson_r`, the early-stopping / LR-scheduler / checkpoint monitor).
- The DL **eval** path (`evaluate_model`) likewise accumulates raw
  predictions across all `trainer.predict` batches and scores the
  `by_metric` block once.

This matters because per-batch aggregation (a weighted mean of per-batch
metrics, the previous behaviour) is mathematically wrong for correlations —
Pearson r is not linear under averaging, and Spearman re-ranks within each
batch — and biased even for RMSE (`mean(sqrt(batch_MSE)) ≠
sqrt(mean(all_errors²))` by Jensen). Pooling avoids all of it. Consequently
`dataloader.{val,test}.batch_size` (in `conf/data/g2f.yaml`) is a pure
**memory** knob: `null` keeps the whole split in one batch, but any finite
value yields identical reported numbers.

## HPC (SLURM) Submission

The repo implements four CV split groups, each selected via the
`+cv_spec=<scheme>` config group (`conf/cv_spec/`) and each with its own
job-script directory under `scripts/jobs/cv/`:

- **`env_year_loo`** — leave-one-(Env,Year)-out (single `test` metric).
- **`cv_2_1`** — one female-fold held out across all envs → `CV2` + `CV1`.
- **`cv_0_00`** — a held-out env **plus** a female-fold → `CV0` + `CV00`.
- **`random_kfold`** — uniform random row-level k-fold, the in-distribution
  baseline (single `test` metric; FPCA-only job scripts under
  `scripts/jobs/cv/random_kfold/`).

For `env_year_loo`:

```bash
cd scripts/jobs/cv/env_year_loo
# Edit ENV_YEAR_FOLDS, DL_CONFIGS, FPCA_CONFIGS, DL_SEEDS/FPCA_SEEDS at the
# top (AGDD = the `__t-agdd` FPCA composers, listed in FPCA_CONFIGS), then:
./submit_fseq.sh
```

`submit_fseq.sh` submits one SLURM job per (config, seed); each job runs all
folds **sequentially** in one allocation. Use `submit_fpar.sh` for one job per
(fold, config, seed). For `cv_2_1` / `cv_0_00`, the unified
`scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh` submits **every** model class
(neural_process — the TNP — and FPCA/BGLR) over **both** schemes from one script
(shared `SEED_PAIRS`/`FOLDS`/`FOLD_CSV` so the pinned fold map and the paired
comparison are structural; edit its `CLASS_ORDER` registry to add/skip a class).
The committed fold table `cv/data/female_folds.csv` is the R reference pipeline's
own `set.seed` assignment, so TNP and kernel-model splits are identical. All
submitters are idempotent (skip when the run's marker exists).
See [cv/README.md](cv/README.md) for the full artifact schema, the metric
definitions, and the paired TNP-vs-FPCA protocol.

## Known Issues and TODOs

### DL pipeline

- **Checkpoint resuming is local-only**
  W&B checkpoint resuming was removed to simplify the pipeline. If remote checkpoint retrieval is needed in the future, it will need to be re-added to `utils/experiment/utils.py:find_checkpoint_path`.

### FPCA baseline

- **Per-split asymmetric FPC score caching (deferred)**
  Currently VI FPC scores for all splits are cached in a single content-addressed file keyed on the train split. A finer-grained scheme — caching train scores by `hash(train)` and test scores by `hash(train) + hash(test)` — would allow the R `predict` mode to be used when only the test split changes, avoiding a full FPCA re-fit.

- **Subset reuse of a cached full weather FPCA fit (deferred)**
  A `WeatherFPCAProcessor` run with `fit_vars=[PTR]` does **not** reuse an existing `fit_vars=null` (all-columns) cache, even though the full entry already stores `PTR`'s scores (and slicing them out is bit-identical by per-variable FPCA independence). The restricted run misses because `fit_vars` is folded into the cache key (and the `env_content_hash` only covers the loaded columns), so it recomputes — needlessly expensive on the AGDD axis. Two ways to fix, both gated on making the data-identity hash variable-set-independent (cheap: the AGDD axis table already loads every column):
  1. **Superset-aware lookup (small):** probe the full-fit key (same identity, no `fit_vars` tag); if it exists and `var_names_full ⊇` the request, load and slice via the existing `emit_vars` path. Reuses only the full (or an exactly-probed) entry.
  2. **Per-variable cache granularity (principled):** key entries by `(identity, variable)` so `fit_vars=null` warms every variable, any subset hits, and a superset computes only the missing vars in one batched R call. Makes the expensive AGDD fit a one-time, incrementally-extensible cost.
  Both require relaxing the hard `fit_vars` equality check in `WeatherFPCAProcessor.load_fitted` to a superset check. Bit-identical reuse holds only under the existing per-variable-independence + per-column-scaler invariant. See `utils/data/processing/README.md` (`fit_vars` vs `emit_vars`).

- **(Resolved) `wandb.init` >64-char tag crash** — long slugs (e.g. the 98-char
  weather-GxE BGLR slug) used to trip wandb's 1–64 char `run_tags` validator. Both
  pipelines now sanitize tags through `utils/experiment/utils.py::sanitize_wandb_tags`
  (truncate + hash any tag >64 chars); `fpca_core.train` and the DL `_log_to_wandb`
  path both call it before `wandb.init`.

- **BGLR needs a one-line source patch for rank-1 RKHS kernels**
  Stock CRAN BGLR 1.1.4 aborts with `invalid 'times' argument` whenever a kernel block reduces to a single super-`tolD` eigenvalue — a **rank-1 kernel**. weather-GxE triggers this: its weather kernel `K_W` is rank-1 when built from one PTR weather FPC (`weather_fpca.n_components=1`). BGLR's RKHS handler does `LT$V = LT$V[, tmp]` and R drops the resulting 1-column matrix to a vector, so the outer product fails (confirmed independently upstream).
  **Fix:** re-install BGLR with the `drop = FALSE` patch via `bash scripts/patch_bglr.sh` — see [`baselines/R_HPC_SETUP.md`](baselines/R_HPC_SETUP.md) ("Patch BGLR for rank-1 RKHS terms") and [`baselines/bglr_rank1.patch`](baselines/bglr_rank1.patch). The patch lives only in `R_LIBS_USER` (the CRAN pin still reads `BGLR 1.1.4`), so re-run it after any BGLR re-install or environment rebuild, or weather-GxE silently fails in BGLR.

### Cross-validation

See [cv/README.md](cv/README.md) for usage, artifact schema, and design notes.
