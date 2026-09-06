#!/bin/bash
# LOO CV — one SLURM job per (fold, config, seed). Idempotent: skips
# any combo whose metrics.json already exists.
#
# Run:        ./submit_fpar.sh
# Monitor:    squeue --me

set -euo pipefail

# ── User configuration ─────────────────────────────────────────────────────

# Folds to hold out, one at a time. Format: "ENV YEAR" per entry.
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

DL_SEEDS=(0 1)
FPCA_SEEDS=(0)

CLUSTER="${CLUSTER:-GRACE}"                        # GRACE or FASTER (see CLUSTER_ACCOUNTS)
SLURM_MAIL_USER="your-email@example.com"

# Smoke mode: positive int subsamples each split to N. Results land
# under artifacts/cv/env_year_loo-smoke/ to keep them out of prod.
SMOKE_N=""

# W&B logging per family, applied to the training run (eval is always silent).
# Passed to the job as WANDB_LOGGING.
DL_WANDB="true"      # log DL training to W&B
FPCA_WANDB="false"   # FPCA/BGLR: no W&B

# ── SLURM resources ────────────────────────────────────────────────────────
DL_TIME="3:00:00";   DL_CPUS=2;   DL_MEM="50G";  DL_GRES="gpu:1"; DL_PARTITION="gpu"
# FPCA: 15h covers weather FPCA (dense V2 daily curves, ~1.5h) + kernel
# build + 10k-iter Gibbs sampler at G2F scale, with margin for weather-GxE.
FPCA_TIME="15:00:00"; FPCA_CPUS=2; FPCA_MEM="50G"

declare -A CLUSTER_ACCOUNTS=(["GRACE"]="YOUR_GRACE_ACCOUNT" ["FASTER"]="YOUR_FASTER_ACCOUNT")

# ── Model configs (longer lists, kept last) ─────────────────────────────────
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

fold_strs=()
for fold in "${ENV_YEAR_FOLDS[@]}"; do
  fold_env=$(echo "${fold}" | cut -d' ' -f1)
  fold_year=$(echo "${fold}" | cut -d' ' -f2)
  fold_strs+=("${fold_env}.${fold_year}")
done

N_DL=$(( ${#fold_strs[@]} * ${#DL_CONFIGS[@]} * ${#DL_SEEDS[@]} ))
N_FPCA=$(( ${#fold_strs[@]} * ${#FPCA_CONFIGS[@]} * ${#FPCA_SEEDS[@]} ))
N_TOTAL=$(( N_DL + N_FPCA ))

echo "============================================================"
echo "LOO CV — Per-Fold Parallel Submission"
echo "============================================================"
echo "Cluster:      ${CLUSTER_UPPER}  (account: ${ACCOUNT})"
echo "Folds:        ${#fold_strs[@]}  (${fold_strs[*]})"
echo "DL configs:   ${#DL_CONFIGS[@]}   seeds: ${DL_SEEDS[*]}"
echo "FPCA configs: ${#FPCA_CONFIGS[@]}  seeds: ${FPCA_SEEDS[*]}"
echo "Max jobs:     ${N_TOTAL}  (${N_DL} DL + ${N_FPCA} FPCA)"
if [[ -n "${SMOKE_N}" ]]; then
  echo "*** SMOKE MODE: dataset.smoke_n=${SMOKE_N}  output → ${ENV_YEAR_LOO_ARTIFACT_ROOT} ***"
fi
echo "------------------------------------------------------------"
echo "DL resources:   time=${DL_TIME}  cpus=${DL_CPUS}  mem=${DL_MEM}  gres=${DL_GRES}  partition=${DL_PARTITION}"
echo "FPCA resources: time=${FPCA_TIME}  cpus=${FPCA_CPUS}  mem=${FPCA_MEM}  partition=<default>"
echo "------------------------------------------------------------"

cd "${REPO_ROOT}"

# ── Environment ────────────────────────────────────────────────────────────
ENV_PATH="/scratch/user/$USER/neural-processes/env/bin/activate"
# `set +eu` wraps the HPRC profile source: it references PS1 (nounset
# trip) and runs a `grep -q` that exits 1 on a fresh shell (errexit trip).
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

python -m cv.folds validate --data-dir "${DATA_DIR}" --folds "${fold_strs[@]}" \
  || { echo "Fold validation failed. No jobs submitted."; exit 1; }

# Kernel-rank pre-flight intentionally disabled here; the enabled
# configs are validated by prior runs (see cv/validate_kernel_rank.py).

echo ""

# ── Slug pre-computation ───────────────────────────────────────────────────
# Slug = cfg.misc.model_slug from conf/misc/base.yaml (single source of
# truth). Feature subsets are owned by the experiment composer.
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
mkdir -p "${SCRIPT_DIR}/logs"

# Line-buffered tee so non-interactive runs (ssh, CI) leave a live audit
# trail and partial progress survives a dropped SSH channel.
LOG_FILE="${SCRIPT_DIR}/logs/submit_$(date +%Y%m%dT%H%M%S)_$$.log"
exec > >(stdbuf -oL tee -a "${LOG_FILE}") 2>&1
echo "Mirroring submission output to ${LOG_FILE}"
echo "------------------------------------------------------------"

# 3× retry with 5/10/15s backoff — a transient slurmctld hiccup would
# otherwise kill the whole loop under `set -e`.
submit_with_retry() {
  local rc
  for attempt in 1 2 3; do
    set +e
    sbatch "$@"
    rc=$?
    set -e
    [[ $rc -eq 0 ]] && return 0
    echo "  sbatch failed (rc=$rc); retry ${attempt}/3 in $((attempt * 5))s"
    sleep $((attempt * 5))
  done
  echo "  Giving up after 3 retries."
  return 1
}

SUBMITTED=0
SKIPPED=0
FAILED=0

for fold in "${ENV_YEAR_FOLDS[@]}"; do
  fold_env=$(echo "${fold}" | cut -d' ' -f1)
  fold_year=$(echo "${fold}" | cut -d' ' -f2)
  fold_id="${fold_env}.${fold_year}"

  # ── DL models ──────────────────────────────────────────────────────────────
  for config_name in "${DL_CONFIGS[@]}"; do
    slug="${SLUG_MAP[${config_name}]}"
    for seed in "${DL_SEEDS[@]}"; do
      out="${ENV_YEAR_LOO_ARTIFACT_ROOT}/${fold_id}/${slug}/seed=${seed}"
      if [[ -f "${out}/metrics_last.json" ]]; then
        echo "SKIP   ${fold_id}/${slug}/seed=${seed}  (metrics_last.json exists)"
        (( ++SKIPPED ))
        continue
      fi

      SMOKE_EXPORT=""
      if [[ -n "${SMOKE_N}" ]]; then
        SMOKE_EXPORT=",SMOKE_N=${SMOKE_N}"
      fi

      if submit_with_retry \
        --job-name="loo_${fold_id}_${slug}_s${seed}" \
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
        --export=FOLD_ENV="${fold_env}",FOLD_YEAR="${fold_year}",CONFIG="${config_name}",SLUG="${slug}",SEED="${seed}",OUT_DIR="${out}",WANDB_LOGGING="${DL_WANDB}"${SMOKE_EXPORT} \
        "${SCRIPT_DIR}/dl_job.job"; then
        echo "SUBMIT DL   ${fold_id}/${slug}/seed=${seed}"
        (( ++SUBMITTED ))
      else
        echo "FAILED DL   ${fold_id}/${slug}/seed=${seed}"
        (( ++FAILED ))
      fi
      sleep 0.2
    done
  done

  # ── FPCA baselines ─────────────────────────────────────────────────────────
  for config_name in "${FPCA_CONFIGS[@]}"; do
    slug="${SLUG_MAP[${config_name}]}"
    for seed in "${FPCA_SEEDS[@]}"; do
      out="${ENV_YEAR_LOO_ARTIFACT_ROOT}/${fold_id}/${slug}/seed=${seed}"
      if [[ -f "${out}/metrics_last.json" ]]; then
        echo "SKIP   ${fold_id}/${slug}/seed=${seed}  (metrics_last.json exists)"
        (( ++SKIPPED ))
        continue
      fi

      SMOKE_EXPORT=""
      if [[ -n "${SMOKE_N}" ]]; then
        SMOKE_EXPORT=",SMOKE_N=${SMOKE_N}"
      fi

      if submit_with_retry \
        --job-name="loo_${fold_id}_${slug}_s${seed}" \
        --account="${ACCOUNT}" \
        --time="${FPCA_TIME}" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task="${FPCA_CPUS}" \
        --mem="${FPCA_MEM}" \
        --mail-type=FAIL \
        --mail-user="${SLURM_MAIL_USER}" \
        --export=FOLD_ENV="${fold_env}",FOLD_YEAR="${fold_year}",CONFIG="${config_name}",SLUG="${slug}",SEED="${seed}",OUT_DIR="${out}",WANDB_LOGGING="${FPCA_WANDB}"${SMOKE_EXPORT} \
        "${SCRIPT_DIR}/fpca_job.job"; then
        echo "SUBMIT FPCA ${fold_id}/${slug}/seed=${seed}"
        (( ++SUBMITTED ))
      else
        echo "FAILED FPCA ${fold_id}/${slug}/seed=${seed}"
        (( ++FAILED ))
      fi
      sleep 0.2
    done
  done

done

echo ""
echo "============================================================"
echo "Done.  Submitted: ${SUBMITTED}   Skipped (already complete): ${SKIPPED}   Failed (after retries): ${FAILED}"
echo "============================================================"
echo "Monitor:   squeue --me"
