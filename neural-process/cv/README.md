# Cross-Validation (LOO + structured G×E + random k-fold)

(Environment, Year)-level LOO CV: each fold holds out one (Env, Year) pair as
the test set and trains on all remaining data. No fixed global test set — the
LOO score is the evaluation. With 12 environments across 2 years there are
19 valid (Env, Year) combinations (some envs appear in only one year).

This module hosts **four** CV schemes (`cv/schemes/`, selected via `+cv_spec=`):

| scheme | held out | unit | cv_label(s) |
|---|---|---|---|
| `env_year_loo` | one (Env, Year) column | env | `test` |
| `cv_2_1` | one female fold across all envs | female (maternal line) | `CV1`/`CV2` |
| `cv_0_00` | a female fold **+** a held-out env | female + env | `CV0`/`CV00` |
| `random_kfold` | a uniform random sample of **all rows** | **row** | `test` |

`random_kfold` is the in-distribution baseline (no female grouping, no env
masking): `RandomKFoldScheme` (`fold_unit="row"`) draws a seeded balanced
per-row partition via `random_row_fold_array` (`schemes/fold_source.py`), which
`resolve_cv_assignment` threads in as a `row_folds` array rather than a
maternal-line `fold_map`. Like `env_year_loo` it has no OBSERVE rows (so
`fit_mask is None`) and a single `"test"` label. Config: `conf/cv_spec/
random_kfold.yaml` (cv_seed, fold, k_folds; no heldout_env). FPCA runs
launch via `scripts/jobs/cv/random_kfold/submit_fpca.sh`.

## Directory Layout

```
cv/
├── README.md              # this file
├── folds.py               # validate user-specified Env.Year folds against data
├── scoring.py             # shared scorer (score_by_label, block_array, weighted_block_correlation → r_w/rho_w) — DL + FPCA
├── slug_util.py           # run identity: resolve_slug (model_slug) + fold_dir_token
├── rank_scan.py           # per-fold kernel eigen-rank scan (reference CSV for n_components choices)
├── validate_kernel_rank.py # submit-side pre-flight: rank-consistency check over experiment configs
├── schemes/               # cv_2_1 / cv_0_00 / env_year_loo / random_kfold role+label assignment, fold sources
└── data/female_folds.csv  # PINNED female→fold map (10 seeds; the R reference's table) for the paired DL↔FPCA design

scripts/cv/
└── make_female_folds.py   # generate an alternative native-RNG fold table (reproducible, validated)

conf/
├── optim/default.yaml      # DL optimization fragment: optimizer, scheduler,
│                           # sched_name / hp_tag (FPCA experiments pull optim/none)
├── data/g2f.yaml           # dataset + dataloader + params (x_dim, y_dim, s_dim)
├── misc/base.yaml          # misc.* schema (seed, checkpointing, W&B, etc.)
├── experiment/dl/*.yaml    # per-model DL composers
└── experiment/fpca/*.yaml  # per-model FPCA composers

scripts/jobs/cv/env_year_loo/
├── submit_fseq.sh        # sequential: all folds in one job per (config, seed)
├── submit_fpar.sh        # parallel: one job per (fold, config, seed)
├── dl_job.job       # SLURM: DL fold (GPU, train → eval in one allocation)
├── fpca_job.job     # SLURM: FPCA fold (CPU only, no GPU waste)
└── eval_only.job    # SLURM: re-score an existing checkpoint (eval pass only)

scripts/jobs/cv/cv_0_00_2_1/   # unified launcher for the CV1/CV2 + CV0/CV00 schemes
├── submit_cv.sh          # one launcher, ALL model classes over BOTH schemes.
│                         #   Per-class MODEL SPECS blocks (NP_*/FPCA_* knobs +
│                         #   CLASS_ORDER) fan out neural_process / fpca; shared
│                         #   SEED_PAIRS/FOLDS[/HELDOUT_ENVS]/FOLD_CSV make the
│                         #   pairing structural. Artifacts under
│                         #   artifacts/cv/cv_0_00_2_1/<class>/<scheme>/...
├── dl_job.job            # SLURM: DL split (GPU) — neural_process
└── fpca_job.job          # SLURM: FPCA split (CPU + R), +cv_spec=${SCHEME}

scripts/jobs/cv/random_kfold/  # in-distribution per-row k-fold baseline
├── submit_fpca.sh        # FPCA/BGLR over the random_kfold scheme (CV_SEEDS × FOLDS)
└── fpca_job.job          # SLURM: FPCA split (CPU + R)
```

## Artifact Structure

```
artifacts/cv/env_year_loo/{fold_env}.{fold_year}/{model_slug}/seed={N}/
├── metrics.json              # FPCA only: by_metric schema (see below); the FPCA skip marker
├── predictions.csv           # FPCA only: D8 all-row schema (Pedigree.Env, Pedigree, Env, Female, Fold, role, cv_label, Actual, Predicted)
├── metrics_<ckpt>.json       # DL: same by_metric schema, one per evaluated checkpoint —
│                             #   metrics_best_<metric>.json, metrics_best_frozen_<metric>.json
│                             #   (freeze_on_plateau only), metrics_last.json. No bare metrics.json for a DL run.
├── predictions_<ckpt>.csv    # DL: D8 all-row schema, one per checkpoint (predictions_best_<metric>.csv, …)
├── processing_metadata.json  # per-processor cache keys + feature dims (written when a processor is enabled)
├── training_curve.jsonl      # DL only: one JSON line per validation epoch
├── checkpoints/              # DL only: one best_<metric>.ckpt per monitored metric (the monitor
│                             #   stream's checkpoint_metrics — [pearson_r, rmse, loglik] in the
│                             #   shipped bundles) + last.ckpt + best_frozen_<metric>.ckpt when
│                             #   freeze_on_plateau is enabled (best up to that metric's plateau),
│                             #   plus best_frozen_<profile>_<metric>.ckpt per configured
│                             #   extra freeze profile (misc.checkpointing extra_profiles)
└── regression_model.joblib   # FPCA only
```

The CV1/CV2/CV0/CV00 split groups (D10) use fully isolated trees with a
fold-dir token from `cv.slug_util.fold_dir_token` instead of the bare env:
```
artifacts/cv/cv_0_00_2_1/{class}/cv_2_1/Seed{NN}.Fold{K}/{model_slug}/seed={N}/...          # CV2 + CV1
artifacts/cv/cv_0_00_2_1/{class}/cv_0_00/Seed{NN}.Fold{K}.{Env.Year}/{model_slug}/seed={N}/...  # CV0 + CV00
```
(`{class}` = `neural_process` | `fpca`, the unified `submit_cv.sh` root.)
Same per-run files as above; `model_slug` stays scheme-agnostic (the split
identity lives in the dir token, not the slug).

`processing_metadata.json` is written by `train.py` /
`baselines.fpca_train` when the config enables any processor. It
stores per-processor content-addressed cache keys (plus enabled
processors, feature dims, and effective y_dim). `eval.py`'s predict
mode reads it back via `setup._resolve_processing_cache_keys` so
`FeatureProcessor.load_and_transform` can restore the same fitted state
the training run produced. Fitted state itself lives under the
dataset's `.cache/` directory — see
[`utils/data/processing/README.md`](../utils/data/processing/README.md).
For vanilla (all-disabled) LOO runs the file is harmless — eval skips
cache resolution entirely when no processor is enabled.

`training_curve.jsonl` is written by `TrainingCurveCallback` (enabled via
`misc.save_training_curve: true` in `conf/misc/base.yaml`). Each line records
**every eval stream's** epoch metrics (one namespaced `{stream}_{metric}` column
per stream, plus `train_*`), not just `val_*`:
```json
{"epoch": 42, "val_rmse": 1.23, "val_pearson_r": 0.81, "val_spearman_r": 0.79, "test_rmse": 1.44, "test_pearson_r": 0.62, "train_rmse": 0.95}
```
The file is overwritten (truncated at `on_train_start`) by default, mirroring
the checkpoint convention, so a rerun of the same (model, seed) replaces the
prior curve; set `misc.training_curve.enable_version_counter: true` to preserve
attempts as `training_curve-v1.jsonl`, …. The `val_*` series is the DL
early-stopping signal — an eval-streams **`carve`** stream over `FIT ∪ OBSERVE`
(default a `random` `frac: 0.1`), seeded by `dataset.val_seed` and dropped at
eval (predict-mode clears `eval_streams`). The `test_*` series is an
**observe-only** probe over the held-out PREDICT rows: logged for trajectory
observation but structurally barred from driving selection; the authoritative
test score still comes only from `eval.py`.

Example: `artifacts/cv/env_year_loo/DEH1.2020/istnp_L4_H4x16_D64_FF64_ist.k256_vi.enc=tfm.d64.L2.h4_vi.t=dap.c=full_loss=nll/seed=0/metrics_best_pearson_r.json`

**`model_slug` encoding** — one unified template
(`conf/misc/base.yaml`), computed by `cv/slug_util.py::resolve_slug`:
`model_name + hp_tag`. Run identity is carried *inside* `model_name` via the
**unified per-encoder grammar**: each set encoder / FPCA block is one peer
token carrying its time axis and channel set inline —
`vi.t=<time>.c=<channels>` (VI) and `wthr.t=<time>.c=<channels>` (weather),
with FPCA peers adding kernel facets (`.k=` upstream FPC count, `.s=` scaling,
`.rank=`, `.uc`). Genomic / interaction kernels are sibling peers
(`ga.rows`/`ga.rank=full`, `gaXeid`, `gaXwthr`, …), alphabetically sorted.
- DL e.g. `istnp_L4_H4x16_D64_FF64_ist.k256_vi.enc=tfm.d64.L2.h4_vi.t=dap.c=full_wthr.t=dap.c=full_loss=nll`;
  FPCA e.g. `fpca_ga.rank=full.uc_..._vi.t=agdd.c=ngrdi.k=5.s=wez.rank=full.uc_wthr.t=agdd.c=ptr.k=1.s=gz.rank=full.uc_bglr_loss=mse`.
- `hp_tag` (`_sched=...wd=...`) is **DL-only**: DL experiments select
  `/optim: default` (which defines `hp_tag`), while FPCA/BGLR/GBLUP experiments
  select `/optim: none`, leaving `hp_tag` undefined — so FPCA slugs end at
  `..._loss=mse` with no scheduler/weight-decay suffix.
- The per-encoder pieces come from the `vi_subset` / `weather_subset` groups
  (`*_chan`) and the `axis` group (`*_time` / `*_aug`); their defaults (`full`
  channels, `dap` time, no augmentation) render `vi.t=dap.c=full`. A dedicated
  experiment composer bakes in non-defaults (e.g. the `..._bglr_mse__t-agdd` FPCA
  configs pull `/axis: shared_t-agdd` → `vi.t=agdd` / `wthr.t=agdd`) so each axis run
  gets its own slug / artifact dir. (Submit scripts just list configs; an
  ad-hoc `axis=<name>` CLI override also works.)

See `baselines/README.md` for the per-token model-name grammar and
`ls conf/experiment/fpca/` / `ls conf/experiment/dl/` for the built configs.

Changing `optimizer.weight_decay` or `sched_name` in `conf/optim/default.yaml`
updates `hp_tag` automatically, producing new DL artifact paths without
overwriting existing ones.

Changing any default hyperparam (layers, dims, heads) produces a new distinct
slug automatically — no manual naming or path collision.

**`metrics*.json` `by_metric` schema (both DL and FPCA):** one entry per
scored `cv_label` (D8). `env_year_loo` and `random_kfold` are the degenerate
one-label `"test"` case; `cv_2_1` emits `CV1`+`CV2`; `cv_0_00` emits
`CV0`+`CV00`.
```json
{
  "meta": {"model_name": "...", "model_slug": "...", "seed": 42, ...},
  "by_metric": {
    "test": {"n": 412, "rmse": 1.23, "pearson_r": 0.87, "spearman_r": 0.85, "loglik": -2.14,
             "r_w": 0.41, "r_w_n_blocks": 19, "rho_w": 0.38, "rho_w_n_blocks": 19}
  }
}
```
Each label block carries `n` (row count), `rmse`, `pearson_r`, `spearman_r`,
`loglik` (mean predictive log-likelihood — DL distributional heads only; absent
for point-estimate heads and the FPCA/BGLR baselines, which carry no density),
the within-block weighted correlations `r_w` (Pearson) and `rho_w` (Spearman)
with their contributing-block counts (see below), each computed **once** over
the rows carrying that label (the DL path pools
all `trainer.predict` batches; the FPCA path scatters train+test predictions
back to `metadata_df` positions). Both pipelines share the scorer in
`cv/scoring.py`.

## (Env, Year) Folds — 19 Total

| 2020 only       | Both years                              | 2021 only   |
|-----------------|-----------------------------------------|-------------|
| DEH1, MIH1, MOH1 | MNH1, TXH1, TXH2, TXH3, WIH1, WIH2, WIH3 | IAH4, NEH1  |

## HPC Submission

```bash
# 1. Edit ENV_YEAR_FOLDS, DL_SEEDS, FPCA_SEEDS, DL_CONFIGS, FPCA_CONFIGS in
#    submit_fseq.sh (AGDD = the `...__t-agdd` composers, listed in FPCA_CONFIGS)
# 2. Validate folds against data (runs before any sbatch):
python -m cv.folds validate \
    --data-dir ./dataset-files/g2f/Pedigrees_Wide_Format_BLUEs \
    --folds DEH1.2020 IAH4.2021 WIH1.2020 ...

# 3. Submit (idempotent — a (config, seed) pair is skipped when every fold's
#    marker exists: metrics_last.json for DL runs, metrics.json for FPCA;
#    EVAL_ONLY=1 instead re-evaluates any fold whose label blocks lack spearman_r):
bash scripts/jobs/cv/env_year_loo/submit_fseq.sh
```

Monitor progress while jobs are in flight:
```bash
squeue --me
```

## Config Layering

Every LOO fold (DL and FPCA) is selected by a single Hydra experiment
override (`+experiment=dl/<slug>` or `+experiment=fpca/<slug>`). The
chosen experiment composer under `conf/experiment/{dl,fpca}/` has a
`defaults:` list that pulls the group fragments below into root via
`# @package _global_` (the `misc/base` block is supplied separately by the
root `conf/config.yaml` defaults, not by the composer):

```
conf/misc/base.yaml         ← (from root config.yaml) shared misc.* block (seed, epochs,
                                checkpointing D1 shape, early_stopping, W&B settings, artifacts_dir)
conf/data/g2f.yaml          ← dataset target (file glob, metadata cols, processing defaults)
conf/optim/default.yaml   ← DL optimization schedule: optimizer (AdamW, weight_decay),
                                scheduler (CosineAnnealingLR), sched_name / hp_tag template
                                (FPCA/BGLR experiments pull conf/optim/none.yaml instead)
conf/model/{family}/<slug>.yaml ← architecture / regressor, n_components (FPCA)
conf/vi_subset/<name>.yaml      ← VI ablation axis (default `full` = no-op; `ngrdi` overrides
                                    to [NGRDI]); sets the `vi_chan` slug piece → `vi.c=ngrdi`
conf/weather_subset/<name>.yaml ← weather ablation axis (default `full` = no-op; `ptr`
                                    overrides all four weather processors to [PTR]); sets the
                                    `wthr_chan` slug piece → `wthr.c=ptr`
conf/axis/<name>.yaml           ← time-axis ablation, named `<family>_t-<time>[_c-<chan>]`
                                    (default `shared_t-dap` = no-op; `shared_t-agdd`,
                                    `fpca_t-agdd_dedup`, `fpca_t-dap_c-agdd`, `fpca_t-agdd_c-dap`, `fpca_t-agdd_warp`,
                                    `dl_t-dap-agdd`, `dl_t-dap-gdd-agdd`,
                                    `dl_t-agdd_wthr-c-agdd`,
                                    … flip the fdapace/view coordinate); set the `*_time`/`*_aug`
                                    slug pieces →
                                    `vi.t=<time>` / `vi.c=<chan><aug>` (and weather analogs)
conf/augment/<name>.yaml        ← DL-only train-time augmentation (the shipped DL composers pull
                                    `vi-tsub` VI time-subsampling; `none` = no-op); sets the
                                    `vi_tsub` / `wthr_tsub` slug pieces → `.tsub=0.75-1.0`
CLI overrides               ← fold_env (drives run name via model_run_name interpolation),
                                test_filter_columns, misc.artifacts_dir, misc.seed, tags,
                                vi_subset=<name>, weather_subset=<name>, axis=<name>
```

The `vi_subset` and `weather_subset` groups compose **after** the
model fragment, so non-default selections clobber the model's
hardcoded subset values. Default `full` files are intentionally empty
of value-overrides (only contributing the empty `_tag` strings) so
vanilla full-feature runs preserve whatever the model fragment set.

`conf/optim/default.yaml` is the DL-specific fragment, separate from
`conf/misc/base.yaml`. Changes to either take effect only on the experiments
whose composers pull that fragment in their `defaults:` list — the `conf/`
tree has no inheritance hierarchy, only declarative composition.

## Filter Semantics: AND across dicts

Each fold passes a two-dict list to `test_filter_columns`:
```yaml
# Hold out DEH1 in 2020 only — AND of two conditions
test_filter_columns:
  - Env: DEH1
  - Year: 2020
```
All dicts in the list are AND'd (intersected). Multiple values within a single
dict are OR'd at the value level. This is implemented in `DatasetSplitter`.

## FPCA: every eigen K is baked into the slug

Each FPCA config encodes every eigen K as an integer literal in the
slug itself — no `???` fields, no runtime K overrides. To sweep a K,
create a new config following the R-mirror token grammar (see
`baselines/README.md`):

Twelve FPCA configs are pre-generated today (`ls conf/experiment/fpca/`),
covering the G+P main-effects / env-GxE / weather-GxE analogs: `bglr` and `gblup`
variants of env-GxE and weather-GxE (each in a DAP and an AGDD (`__t-agdd`) flavor), the
main-effects (`fpca_ga_gd_vi`) BGLR pair, an env-free weather-GxE BGLR, and an `ols` sklearn
endpoint for weather-GxE (DAP).

```bash
# Full main effects + GxE + PxE interactions — the R reference's env-GxE (shipped)
python -m baselines.fpca_train \
    +experiment=fpca/fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse

# Full main effects + GxWeather + PxWeather interactions — the R reference's weather-GxE (shipped)
python -m baselines.fpca_train \
    +experiment=fpca/fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse

# LOO (submit_fseq.sh iterates over FPCA_CONFIGS, each slug's Ks are baked in)
```

To add a new K value or modality combination, create a new composer
at `conf/experiment/fpca/fpca_<feat_codes>_<reg>_mse.yaml` (with
literal eigen Ks inline per the token grammar) and add the slug to
`FPCA_CONFIGS` in `submit_fseq.sh`.

## W&B Integration

W&B logging has separate **training** and **evaluation** toggles in
`conf/misc/base.yaml`:

- **`wandb_logging_enabled`** (default `true`) — the training run. The CV submit
  scripts set it **per family** via `DL_WANDB` / `FPCA_WANDB` (exported to the
  job as `WANDB_LOGGING`): DL training logs by default; FPCA is off (BGLR writes
  everything to `metrics.json`).
- **`wandb_eval_logging_enabled`** (default `false`) — the `eval.py` scoring
  pass (`run_eval` / `log_results`). Evaluation is silent by default and
  independent of the training toggle, so DL eval never logs unless explicitly
  opted in (`misc.wandb_eval_logging_enabled=true`).

When training logging is enabled, runs go to one W&B project per CV scheme,
`G2F-<scheme>` (e.g. `G2F-ey_loo` for `env_year_loo`; `conf/misc/base.yaml`
`project: ${.data}-${scheme_label:${oc.select:cv_spec.scheme,env_year_loo}}`).

All metrics are always persisted to `metrics.json` regardless of W&B settings.
FPCA jobs route through `wandb_logging_enabled` — `fpca_core.py` calls
`wandb.init` when enabled and writes per-label test metrics as `<label>/<metric>`
keys (e.g. `test/rmse`, `test/pearson_r`, `test/spearman_r`) to `run.summary`,
matching the DL pipeline's `_log_to_wandb` path.

## Utility Scripts

### `cv.folds validate`

```bash
python -m cv.folds validate \
    --data-dir ./dataset-files/g2f/Pedigrees_Wide_Format_BLUEs \
    --folds DEH1.2020 IAH4.2021 WIH1.2020
```

Exits 0 if all fold identifiers are present in the data directory; exits 1 with
a "did you mean?" hint for any unrecognised fold. Always run before `submit_fseq.sh`.

## Metric aggregation (batch-size invariant)

Epoch metrics are computed **once over the pooled epoch predictions**, in
both the mid-training curves and the final `metrics.json`. `LitWrapper`
buffers raw `(predicted-mean, target)` per step and reduces once in
`on_{train,validation,test}_epoch_end`; the DL eval path accumulates raw
`trainer.predict` outputs and scores `by_metric` once. So
`train_*`/`val_*`/`test_*` (incl. `val_pearson_r`, the early-stopping /
LR-scheduler / checkpoint monitor) are independent of the dataloader batch
size — `dataloader.{val,test}.batch_size` is a pure memory knob. (Previously
these were a weighted mean of per-batch metrics, biased for correlations and
RMSE; that path is gone.)

## Within-block predictive ability (`r_w` Pearson, `rho_w` Spearman)

Alongside `rmse` / `pearson_r` / `spearman_r`, each `by_metric` entry carries
**two** within-block, inverse-variance weighted correlations (Tiezzi et al.
2017, *Genotype by environment (climate) interaction…*, J. Dairy Sci.
100:2042): `r_w` (Pearson) and `rho_w` (Spearman / rank). Where the pooled
`pearson_r`/`spearman_r` mix between-environment mean differences into the
score, these measure how well the model ranks genotypes **inside** an
environment:

1. Per block (here **`Env.Year`**, the analog of Tiezzi's herd), compute the
   correlation `r_j` between observed and predicted yield over that block's
   rows, with sampling variance `V(r_j) = (1 − r_j²)/(n_j − 2)`.
2. Pool by inverse-variance weight: `r_w = Σ r_j/V(r_j) ÷ Σ 1/V(r_j)` — so
   larger, more reliable environments dominate.

**`r_w` vs `rho_w`.** `r_w` uses a Pearson `r_j` (linear within-env ability);
`rho_w` uses a Spearman `r_j` — i.e. Pearson on within-block ranks. Spearman's
ρ shares the same `(1 − ρ²)/(n − 2)` variance approximation (it is what the
Spearman significance t-test uses), so the identical inverse-variance pooling
is valid; `rho_w` captures within-env **rank** ability (selection-relevant) and
is robust to monotone nonlinearity / outliers that depress `r_w`. There is
deliberately **no** weighted RMSE: the inverse-variance scheme is the sampling
variance of a *correlation*, and a row-weighted block-pooled RMSE just recovers
the global RMSE — so error is reported via `rmse` only.

Blocks that cannot yield a finite, positive-variance correlation are skipped
(`n_j < 3`, constant observed/predicted within the block, or `|r_j| = 1`);
`r_w_n_blocks` / `rho_w_n_blocks` in `metrics.json` record how many contributed
to each. The
implementation is `cv.scoring.weighted_block_correlation(..., method=)`; the
block id comes from `cv.scoring.block_array` (`Env.Year`).

## Paired DL vs FPCA comparison (`cv_2_1` / `cv_0_00`)

The `cv_2_1` (CV2/CV1) and `cv_0_00` (CV0/CV00) schemes are designed so a deep
model and the FPCA/BGLR baseline can be compared **fairly and pairwise** on the
same held-out genotypes, with the same metrics. Fairness rests on three layers
that are a **single source of truth** shared by both pipelines:

| Layer | Single source of truth | Shared by |
|-------|------------------------|-----------|
| **Split** (who is FIT/OBSERVE/PREDICT, what is masked) | `cv_spec` → `G2FDataset._build_role_based_split` | DL (`+cv_spec=...` threaded by `setup.py`) and FPCA (same) |
| **Metrics** (`rmse` / `pearson_r` / `spearman_r` / `r_w` / `rho_w`) | `cv/scoring.py::score_by_label` + `block_array` | DL `evaluate_model` and `baselines.fpca_core` call it verbatim |

So a DL run and an FPCA run differ only in the **model** — never in the split
or the scorer.

### Pinned folds (the paired design)

The common-female *set* is deterministic (RNG-free; `cv/schemes/common_females.py`),
but the fold *assignment* over that set is seeded. To hold out **identical**
genotypes per seed across both pipelines, the assignment is frozen into a
committed CSV and read back via `cv_spec.fold_source=r_csv`. The committed
`cv/data/female_folds.csv` is the **R reference pipeline's own table**
(`set.seed(seed_num)` + `sample` over the sorted 223 common females, 10 seeds),
so every TNP split is identical to the corresponding kernel-model split.

An alternative table can be generated with the native RNG (reproducible;
byte-identical to `cv_spec.fold_source=native` for the same seed, since the
generator reuses `NativeFoldSource`):

```bash
# Common-female set is computed over the FULL pre-coverage phenotype frame —
# the same universe the runtime validates against.
python scripts/cv/make_female_folds.py \
    --data-dir ./dataset-files/g2f/Pedigrees_Wide_Format_BLUEs \
    --seeds 1-10 --k-folds 5 --out cv/data/female_folds_native.csv
export G2F_FEMALE_FOLDS_CSV=$PWD/cv/data/female_folds_native.csv
```

`RCsvFoldSource` (strict) hard-fails if the data's common set ever disagrees
with the CSV — so a stale CSV or a different data version is caught, never
silently mis-split. The unified `submit_cv.sh` (all model classes) defaults
`FOLD_CSV` to `cv/data/female_folds.csv` for every family.

> **Cluster note:** the committed CSV is generated from one data copy. If the
> cluster's phenotype data version differs, the common set differs and strict
> validation fails by design — regenerate on the cluster with the same script.

### Two kinds of seed: shared fold seed vs per-pipeline init seed

There are two distinct sources of stochasticity, carried by deliberately
distinct config keys (`cv_spec.cv_seed` ≠ `misc.seed` — conflating them is a
latent bug class):

- **`CV_SEED`** (`cv_spec.cv_seed`, R `Seed_Num`) — the female-fold partition
  (which genotypes are held out). This is the only seed **shared** across
  pipelines: the unified `submit_cv.sh` feeds one `SEED_PAIRS` list
  (`cv_seed:model_seed` entries) to every model class, so they hold out the
  same genotypes per replicate (paired by construction — no cross-script list
  to keep in sync).
- **`SEED`** (`misc.seed`) — the per-pipeline model-init seed (DL weight
  init + batch order; BGLR `set.seed`). In `submit_cv.sh` it is the
  `model_seed` half of each `SEED_PAIRS` entry (the LOO `submit_fseq.sh` keeps
  per-family lists, `DL_SEEDS` / `FPCA_SEEDS`). It drives an independent RNG
  in each pipeline and need not match across them; only the fold seed must.

So a replicate is defined by `CV_SEED` (shared); the init seed is an
orthogonal, pipeline-local knob.

The DL early-stopping val carve uses the dedicated `dataset.val_seed`
(decoupled from both `misc.seed` and `random_state`); fixing `val_seed` while
varying `misc.seed` yields the same validation set across randomizations.
Because the train pool itself changes with `CV_SEED`, the carve still varies
across replicates without adding a third uncontrolled RNG.

### Two protocol choices to disclose in Methods

- **D1 — small fixed val carve.** DL carves ~10% of FIT∪OBSERVE for early
  stopping / checkpoint selection via the eval-streams `val` `carve` stream
  (default `random` `frac: 0.1`, seeded by `dataset.val_seed`; the shipped
  `cv_2_1` / `cv_0_00` bundles use a `grouped_variety` `n_groups: 30` carve —
  the `_val=gv30` slug tag); those rows are **unscored**
  (`cv_label` cleared) and the carve is **dropped at eval** (predict-mode
  clears `eval_streams`), so CV2/CV0 in-sample metrics still cover every FIT
  row — matching R/BGLR, which fit on 100%. Net: DL trains on ~10% fewer rows
  for early stopping; with pinned folds
  this asymmetry is identical across seeds and does not bias the paired contrast.
- **D2 — each-its-best inputs.** DL consumes the **raw** modalities (VI curves,
  weather series, genomic) and learns its own representation; FPCA consumes its
  hand-built FPCA-score / kernel features. The comparison is of *full pipelines*,
  not a shared-feature predictor swap. Note also that BGLR is **transductive**
  (it sees held-out rows' features, masks only `y`) while DL is **inductive**
  (never sees held-out rows in training); the scored held-out rows and metrics
  are identical, so this model-class difference is a footnote, not a confound.

### Run + analyze

```bash
# Submit ALL model classes on the same seeds/folds (pinned, paired) — one script:
bash scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh     # classes per CLASS_ORDER (neural_process by default;
#                                                     enable the FPCA line for FPCA/BGLR), both schemes
#   (edit the MODEL SPECS blocks / CLASS_ORDER, or use ONLY_* filters to scope it.)
```

**Analysis unit = seed.** Pool each model's per-fold predictions per
`(model, seed, cv_label)` into one metric as one paired observation, then compare DL vs FPCA
across the 10 seeds with a **paired** test (Wilcoxon signed-rank), reporting the
per-seed paired deltas. Pinned folds are what make the comparison paired —
removing the between-split variance that dominates genomic CV. Both
pipelines share `cv/scoring.py`, so identical predictions yield an
identical `by_metric` block by construction.

## Extensibility

- **More DL seeds**: extend `DL_SEEDS` in `submit_fseq.sh` (the only FPCA-side randomness is the BGLR Gibbs `set.seed`, driven by `FPCA_SEEDS`).
- **New DL hyperparams**: create a new named config file → new slug, no path collision with existing results.
- **New FPCA regressor**: add the config slug to `FPCA_CONFIGS` in `submit_fseq.sh`; no other changes needed.
- **Other CV strategies**: plain per-row random k-fold is implemented as `random_kfold` (see the schemes table above); a new scheme is a `CVScheme` subclass + a `registry.py` entry + a `conf/cv_spec/<name>.yaml`. Group-LOO / other variants follow the same pattern, with isolated `artifacts/cv/<name>/` paths.
