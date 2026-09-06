#!/bin/bash
# LOO CV — one SLURM job per (config, seed), with all folds running
# sequentially inside that allocation. Idempotent: skips a (config,
# seed) pair only when every fold's metrics.json exists.
# Use submit_fpar.sh for per-fold parallel submission.

set -euo pipefail

# ── User config ────────────────────────────────────────────────────────────
ENV_YEAR_FOLDS=(
  "DEH1 2020"
  "IAH4 2021"
  "MIH1 2020"
  "MNH1 2020"  "MNH1 2021"
  "MOH1 2020"
  "NEH1 2021"
  "TXH1 2020"  "TXH1 2021"
  "TXH2 2020"  "TXH2 2021"
  "TXH3 2020"  "TXH3 2021"
  "WIH1 2020"  "WIH1 2021"
  "WIH2 2020"  "WIH2 2021"
  "WIH3 2020"  "WIH3 2021"
)

DL_SEEDS=(0)
FPCA_SEEDS=(0)

CLUSTER="${CLUSTER:-GRACE}"                        # GRACE or FASTER (see CLUSTER_ACCOUNTS)
SLURM_MAIL_USER="your-email@example.com"
SMOKE_N=""

# W&B logging per family, applied to the training run (eval is always silent).
# Passed to the job as WANDB_LOGGING.
DL_WANDB="true"      # log DL training to W&B
FPCA_WANDB="false"   # FPCA/BGLR: no W&B

# EVAL_ONLY=1 ./submit_fseq.sh — skip train.py, only re-eval against
# existing best.ckpt. Per-fold idempotent skip uses splits.test.spearman_r
# presence so pre-Spearman runs are re-evaluated; DL_TIME shortened,
# DL_SEEDS expanded to (0 1), FPCA configs cleared (already carry it).
EVAL_ONLY="${EVAL_ONLY:-}"

# ── SLURM resources ────────────────────────────────────────────────────────
# DL: ~1.5h/fold × 19 folds ≈ 29h budget; 48h gives headroom.
DL_TIME="48:00:00";  DL_CPUS=2;   DL_MEM="50G";  DL_GRES="gpu:1"; DL_PARTITION="gpu"
# FPCA: BGLR with eigen-full can be a few min/fold × 19 → 72h ceiling.
FPCA_TIME="72:00:00"; FPCA_CPUS=2; FPCA_MEM="50G"

# ── Model configs (longer lists, after resources) ───────────────────────────
# Defined before the EVAL_ONLY block below, which may clear FPCA_CONFIGS.
# Names match conf/experiment/{dl,fpca}/*.yaml. Variable subsets are baked into
# the composer (one per 2x2 cell), never passed as a CLI override.
DL_CONFIGS=(
  "istnp_vi__vienc-tfm__aug-vi__nll"                                        # raw VI
  "istnp_vi_ga_gd__grm-rows__vienc-tfm__aug-vi__nll"                        # + GRM rows
  "istnp_vi_wthr_ga_gd__grm-rows__vienc-tfm__wthrenc-tfm__aug-vi-wthr__nll" # + weather set
)

FPCA_CONFIGS=(
  "fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse"                 # env-GxE  (DAP)
  "fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse__t-agdd"            # env-GxE  (AGDD)
  "fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse"       # weather-GxE (DAP)
  "fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_bglr_mse__t-agdd"  # weather-GxE (AGDD)
  # GBLUP — frequentist (REML/BLUP via sommer) twins of the four BGLR configs above.
  "fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_gblup_mse"                # env-GxE  (DAP)
  "fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_gblup_mse__t-agdd"           # env-GxE  (AGDD)
  "fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_gblup_mse"      # weather-GxE (DAP)
  "fpca_eid_ga_gd_gaXwthr_gdXwthr_vi_viXwthr_wthr_gblup_mse__t-agdd" # weather-GxE (AGDD)
)

# EVAL_ONLY: eval is forward-pass only; backfill both seeds; FPCA skipped.
if [[ -n "${EVAL_ONLY}" ]]; then
  DL_TIME="4:00:00"
  DL_SEEDS=(0 1)
  FPCA_CONFIGS=()
  echo "*** EVAL_ONLY: train.py skipped; eval.py only ***"
fi

declare -A CLUSTER_ACCOUNTS=(["GRACE"]="YOUR_GRACE_ACCOUNT" ["FASTER"]="YOUR_FASTER_ACCOUNT")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATA_DIR="${REPO_ROOT}/dataset-files/g2f/Pedigrees_Wide_Format_BLUEs"

if [[ -n "${SMOKE_N}" ]]; then
  ENV_YEAR_LOO_ARTIFACT_ROOT="artifacts/cv/env_year_loo-smoke"
else
  ENV_YEAR_LOO_ARTIFACT_ROOT="artifacts/cv/env_year_loo"
fi

# ── Pre-flight ─────────────────────────────────────────────────────────────
CLUSTER_UPPER=$(echo "${CLUSTER}" | tr '[:lower:]' '[:upper:]')
ACCOUNT="${CLUSTER_ACCOUNTS[${CLUSTER_UPPER}]:-}"
if [[ -z "${ACCOUNT}" ]]; then
  echo "Error: No account configured for cluster '${CLUSTER_UPPER}'. Add it to CLUSTER_ACCOUNTS."
  exit 1
fi

# Build Env.Year fold identifier strings for validation
fold_strs=()
for fold in "${ENV_YEAR_FOLDS[@]}"; do
  fold_env=$(echo "${fold}" | cut -d' ' -f1)
  fold_year=$(echo "${fold}" | cut -d' ' -f2)
  fold_strs+=("${fold_env}.${fold_year}")
done

echo "============================================================"
echo "LOO CV Submission"
echo "============================================================"
echo "Cluster:     ${CLUSTER_UPPER}  (account: ${ACCOUNT})"
echo "Folds:       ${fold_strs[*]}"
echo "DL seeds:    ${DL_SEEDS[*]}"
echo "FPCA seeds:  ${FPCA_SEEDS[*]}"
echo "DL configs:   ${DL_CONFIGS[*]}"
echo "FPCA configs: ${FPCA_CONFIGS[*]}"
if [[ -n "${SMOKE_N}" ]]; then
  echo "*** SMOKE MODE: dataset.smoke_n=${SMOKE_N}  output → ${ENV_YEAR_LOO_ARTIFACT_ROOT} ***"
fi
echo "------------------------------------------------------------"
echo "DL resources:   time=${DL_TIME}  mem=${DL_MEM}  gres=${DL_GRES}  partition=${DL_PARTITION}"
echo "FPCA resources: time=${FPCA_TIME}  mem=${FPCA_MEM}  partition=<default>"
echo "------------------------------------------------------------"

cd "${REPO_ROOT}"

# ── Environment ────────────────────────────────────────────────────────────
ENV_PATH="/scratch/user/$USER/neural-processes/env/bin/activate"
# `set +eu` shields the HPRC profile (PS1 nounset + `grep -q` errexit).
if ! command -v module &>/dev/null; then
  set +eu
  source /etc/profile.d/hprc_profile.sh
  set -eu
fi
module purge
module load GCCcore/13.2.0
module load Python/3.11.5
[[ -f "${ENV_PATH}" ]] || { echo "Error: venv not found at ${ENV_PATH}"; exit 1; }
source "${ENV_PATH}"

python -m cv.folds validate \
    --data-dir "${DATA_DIR}" \
    --folds "${fold_strs[@]}" \
  || { echo "Fold validation failed. No jobs submitted."; exit 1; }

# D1 rank-consistency mutual-exclusion check on every config.
python -m cv.validate_kernel_rank \
    --configs "${DL_CONFIGS[@]}" "${FPCA_CONFIGS[@]}" \
    --folds "${fold_strs[@]}" \
  || { echo "Kernel rank pre-flight failed. No jobs submitted."; exit 1; }

echo ""

# ── Slug pre-computation (cfg.misc.model_slug, single source of truth) ─────
declare -A SLUG_MAP

for config_name in "${DL_CONFIGS[@]}"; do
  SLUG_MAP["${config_name}"]=$(python -c "
from cv.slug_util import resolve_slug
print(resolve_slug('dl/${config_name}'))
")
done

for config_name in "${FPCA_CONFIGS[@]}"; do
  SLUG_MAP["${config_name}"]=$(python -c "
from cv.slug_util import resolve_slug
print(resolve_slug('fpca/${config_name}'))
")
done

echo "Resolved slugs:"
for config_name in "${DL_CONFIGS[@]}" "${FPCA_CONFIGS[@]}"; do
  echo "  ${config_name} → ${SLUG_MAP[${config_name}]}"
done
echo "------------------------------------------------------------"

# ── Submission loop ────────────────────────────────────────────────────────
# One sbatch per (config, seed). Each job iterates FOLDS internally and
# writes per-checkpoint metrics files per fold. A pair is skipped only
# when every fold's metrics_last.json (DL) / metrics.json (FPCA) exists; partial completion still re-submits the full list.
mkdir -p "${SCRIPT_DIR}/logs"

FOLDS_STR="${fold_strs[*]}"

SUBMITTED=0
SKIPPED=0

# ── DL models ────────────────────────────────────────────────────────────────
for config_name in "${DL_CONFIGS[@]}"; do
  slug="${SLUG_MAP[${config_name}]}"
  for seed in "${DL_SEEDS[@]}"; do
    # All-folds-done check — skip submission if every fold is complete.
    # EVAL_ONLY mode flips the completeness test to "has spearman_r" so
    # older runs that predate the Spearman metric get re-evaluated.
    all_done=1
    for fold_id in "${fold_strs[@]}"; do
      mpath="${ENV_YEAR_LOO_ARTIFACT_ROOT}/${fold_id}/${slug}/seed=${seed}/metrics_last.json"
      if [[ ! -f "${mpath}" ]]; then
        all_done=0
        break
      fi
      if [[ -n "${EVAL_ONLY}" ]]; then
        python -c "
import json, sys
d = json.load(open('${mpath}'))
# by_metric is keyed by cv_label ('test', ...); spearman_r lives inside
# each label's block, so check every block.
bm = d.get('by_metric') or {}
sys.exit(0 if bm and all('spearman_r' in v for v in bm.values()) else 1)
" 2>/dev/null || { all_done=0; break; }
      fi
    done
    if (( all_done == 1 )); then
      echo "SKIP   ${slug}/seed=${seed}  (all folds complete)"
      (( ++SKIPPED ))
      continue
    fi
    # `--export` cannot inline FOLDS="…" because the space inside the
    # fold list trips its own delimiter parsing — pre-export and list names.
    export FOLDS="${FOLDS_STR}"
    export ARTIFACT_ROOT="${ENV_YEAR_LOO_ARTIFACT_ROOT}"
    export CONFIG="${config_name}"
    export SLUG="${slug}"
    export SEED="${seed}"
    export SMOKE_N="${SMOKE_N}"
    export EVAL_ONLY="${EVAL_ONLY}"
    export WANDB_LOGGING="${DL_WANDB}"
    sbatch \
      --job-name="loo_${slug}_s${seed}" \
      --account="${ACCOUNT}" \
      --time="${DL_TIME}" \
      --nodes=1 \
      --ntasks-per-node=1 \
      --cpus-per-task="${DL_CPUS}" \
      --mem="${DL_MEM}" \
      --gres="${DL_GRES}" \
      --partition="${DL_PARTITION}" \
      --mail-type=FAIL \
      --mail-user="${SLURM_MAIL_USER}" \
      --export=FOLDS,ARTIFACT_ROOT,CONFIG,SLUG,SEED,SMOKE_N,EVAL_ONLY,WANDB_LOGGING,PATH,HOME \
      "${SCRIPT_DIR}/dl_job.job"
    echo "SUBMIT DL   ${slug}/seed=${seed}  (folds: ${#fold_strs[@]})"
    (( ++SUBMITTED ))
    sleep 1
  done
done

# ── FPCA baselines ───────────────────────────────────────────────────────────
for config_name in "${FPCA_CONFIGS[@]}"; do
  slug="${SLUG_MAP[${config_name}]}"
  for seed in "${FPCA_SEEDS[@]}"; do
    all_done=1
    for fold_id in "${fold_strs[@]}"; do
      if [[ ! -f "${ENV_YEAR_LOO_ARTIFACT_ROOT}/${fold_id}/${slug}/seed=${seed}/metrics.json" ]]; then
        all_done=0
        break
      fi
    done
    if (( all_done == 1 )); then
      echo "SKIP   ${slug}/seed=${seed}  (all folds complete)"
      (( ++SKIPPED ))
      continue
    fi
    export FOLDS="${FOLDS_STR}"
    export ARTIFACT_ROOT="${ENV_YEAR_LOO_ARTIFACT_ROOT}"
    export CONFIG="${config_name}"
    export SLUG="${slug}"
    export SEED="${seed}"
    export SMOKE_N="${SMOKE_N}"
    export WANDB_LOGGING="${FPCA_WANDB}"
    sbatch \
      --job-name="loo_${slug}_s${seed}" \
      --account="${ACCOUNT}" \
      --time="${FPCA_TIME}" \
      --nodes=1 \
      --ntasks-per-node=1 \
      --cpus-per-task="${FPCA_CPUS}" \
      --mem="${FPCA_MEM}" \
      --mail-type=FAIL \
      --mail-user="${SLURM_MAIL_USER}" \
      --export=FOLDS,ARTIFACT_ROOT,CONFIG,SLUG,SEED,SMOKE_N,WANDB_LOGGING,PATH,HOME \
      "${SCRIPT_DIR}/fpca_job.job"
    echo "SUBMIT FPCA ${slug}/seed=${seed}  (folds: ${#fold_strs[@]})"
    (( ++SUBMITTED ))
    sleep 1
  done
done

echo ""
echo "============================================================"
echo "Done.  Submitted: ${SUBMITTED}   Skipped (already complete): ${SKIPPED}"
echo "============================================================"
echo "Monitor:   squeue --me"
