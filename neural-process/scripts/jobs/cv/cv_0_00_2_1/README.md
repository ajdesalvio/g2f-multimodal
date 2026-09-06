# Unified CV train/test — `cv_0_00_2_1`

One launcher + two engines that run **every model class** over **both** CV
schemes, writing all artifacts under a single root that diverges on model class.

```
artifacts/cv/cv_0_00_2_1/<model_class>/<scheme>/<token>/<slug>/seed=<n>
                         └ neural_process | fpca
```

This supersedes the earlier per-family launchers: the split-parity invariants
(`cv_seed`, `FOLD_CSV`, `val_seed=0`) now live in one place, so results across
all classes stay directly comparable.

## Files

| File | Role |
| --- | --- |
| `submit_cv.sh` | Launcher. Per-class spec blocks fan out over schemes × `SEED_PAIRS` (cv_seed:model_seed) × folds × [envs] and dispatch each split to the right engine. |
| `dl_job.job` | **DL engine** (GPU) — serves `neural_process`. Const LR, no early stop, streams the scheme's CV folds each epoch, keeps per-metric `best_<metric>` + `best_frozen_<metric>` checkpoints plus `last`, then `eval.py` scores each. Marker: `curve.done`. |
| `fpca_job.job` | **FPCA/BGLR engine** (CPU + R, Grace-only) — endpoint BGLR / GBLUP / sklearn baselines. Marker: `metrics.json`. |

## Model specs (top of `submit_cv.sh`)

There's no flat registry. Everything you edit is in the **`MODEL SPECS` section
at the top of the file**; everything below it is machinery. Each class is one
block of `<PREFIX>_*` knobs (shared by the whole family) plus its `<PREFIX>_CONFIGS`
list:

```bash
# ── neural_process ──
NP_ENGINE=dl
NP_GRES=gpu:a100:1
NP_TIME=12:00:00   # full-scale campaign: mean 3.3 h, max 7.5 h per split
NP_EPOCHS=250
NP_PATIENCE=25
NP_PROJECT=G2F-NP
NP_CONFIGS=(
  "istnp_..."
)
```

Block knobs (`NP_`, `FPCA_` prefixes):
- `_ENGINE` — `dl` (→ `dl_job.job`) or `fpca` (→ `fpca_job.job`)
- `_TIME` — sbatch `--time` (all engines; shipped: `NP_TIME=12:00:00`, `FPCA_TIME=24:00:00` — FPC scores are recomputed from scratch per split)
- `_GRES` / `_EPOCHS` / `_PATIENCE` / `_PROJECT` — DL only (ignored by the fpca engine)
- `_CONFIGS` — the models in this family: `conf/experiment/<engine>/<config>.yaml` (no `.yaml`)

`CLASS_ORDER` (same section) lists which blocks run, in order — comment a line to
skip a class. **cpus / mem / partition are uniform engine infra** and live in the
machinery (`submit_class`), *not* in the per-class specs, so a class's settings
are all in one place.

**Adding a model** = add one line to that block's `_CONFIGS`. The spec is shared
across every config in the block; **if a specific model needs different
resources, bump the block's knobs or add a second block** — that's on you.
Runtime isolation is preserved: each class keeps its own W&B project
(`G2F-NP-<scheme>`) and its own artifact subtree.

## Run

```bash
# Preview the full plan (no sbatch; safe off-cluster):
DRY_RUN=1 bash scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh

# 1-split smoke per class before full scale:
ONLY_CLASS=neural_process ONLY_SCHEME=cv_0_00 ONLY_FOLD=1 ONLY_ENV=DEH1.2020 \
  EPOCHS=3 SMOKE_N=512 ARTIFACT_BASE=artifacts/cv/cv_0_00_2_1-smoke \
  bash scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh

# One class only / one config (targeted backfill):
ONLY_CLASS=fpca bash scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh
ONLY_CONFIG=fpca_ga_gd_vi_bglr_mse bash scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh

# Full submit:
bash scripts/jobs/cv/cv_0_00_2_1/submit_cv.sh
```

Idempotent: a split whose marker (`curve.done` for DL, `metrics.json` for FPCA)
already exists is skipped, so re-running only fills gaps.

## Filters / env knobs

`ONLY_CLASS`, `ONLY_CONFIG`, `ONLY_SCHEME`, `ONLY_FOLD`, `ONLY_ENV`, `ONLY_PAIR`
(stage one `cv_seed:model_seed` replicate at a time — a bare submit fans out
every `SEED_PAIRS` entry and can exceed `MaxSubmitJobs`), `EPOCHS`
(override every DL block's epochs), `SMOKE_N`, `ARTIFACT_BASE`, `DRY_RUN`,
`FOLD_CSV` (via `G2F_FEMALE_FOLDS_CSV`), `ACCOUNT`. Scale up by editing the
grid arrays at the top: `FOLDS` and `SEED_PAIRS` — a list of `cv_seed:model_seed`
replicates (iterated as PAIRS, not a cross-product), shared across classes so the
`cv_seed`s stay aligned for the paired comparison.

> **FPCA is Grace-only** — BGLR / GBLUP need R (`R/4.4.1` → full `GCC/13.2.0`).
> The DL engine runs on Grace or FASTER (hostname-branched module loads).
