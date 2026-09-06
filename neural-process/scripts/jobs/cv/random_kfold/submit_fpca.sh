#!/bin/bash
# random_kfold — FPCA/BGLR submission, one SLURM job per split
# (cv_seed × fold × config × seed). Plain per-ROW random k-fold over ALL rows
# (no env removed, no female grouping). The BGLR counterpart of the DL
# curve-track random_kfold runs: the SAME cv_seed + k_folds reproduce the SAME
# random_row_fold_array over the identical coverage-filtered universe
# (axis_source ∩ genomic), so DL and FPCA hold out the SAME rows per fold and
# are directly comparable. BGLR fits once → metrics.json (no curve tracking;
# tracking is not meaningful for FPCA). Idempotent (skips splits whose
# metrics.json exists).
#
# Run:        ./submit_fpca.sh
#
# NB: K_FOLDS MUST match any paired DL sweep (default 5) or the
# partitions won't line up.

set -euo pipefail

# ── User configuration ─────────────────────────────────────────────────────
CV_SEEDS=(1)                      # seeds the per-row fold partition (match DL)
FOLDS=(1 2 3 4 5)                 # held-out folds (match DL)
K_FOLDS=5                         # number of per-row folds (MUST match DL)

FPCA_SEEDS=(0)                    # FPCA-only model-init seed (BGLR set.seed)

CLUSTER="${CLUSTER:-GRACE}"
SLURM_MAIL_USER="your-email@example.com"
SMOKE_N=""
FPCA_WANDB="false"                # FPCA/BGLR: no W&B (tracking not meaningful)

# No --partition: SLURM routes the CPU/BGLR job by walltime to the default
# partition, matching the working production FPCA submits (cv_2_1/cv_0_00).
# FPCA_TIME is env-overridable; default 3h is ample because the shipped
# composers pin the VI subset (NGRDI) inside the experiment config, so R
# fits one VI, not ~37. Bump it for a heavier full-fit run, e.g.
# `FPCA_TIME=15:00:00 ./submit_fpca.sh`.
FPCA_TIME="${FPCA_TIME:-03:00:00}"; FPCA_CPUS=2; FPCA_MEM="50G"
declare -A CLUSTER_ACCOUNTS=(["GRACE"]="YOUR_GRACE_ACCOUNT" ["FASTER"]="YOUR_FASTER_ACCOUNT")

# Names match conf/experiment/fpca/*.yaml. env-GxE (env-identity interactions) is the
# headline baseline; G+P main-effects (genomic + VI, NO GxE, NO env)
# is the like-for-like match to the weather-free DL `vi_ga_gd` trio for the
# in-distribution capacity comparison. Both on the DAP and AGDD time axes.
FPCA_CONFIGS=(
  "fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse"                   # env-GxE (DAP)
  "fpca_eid_ga_gd_gaXeid_gdXeid_vi_viXeid_bglr_mse__t-agdd"           # env-GxE (AGDD)
  "fpca_ga_gd_vi_bglr_mse"                                        # G+P main-effects (DAP)
  "fpca_ga_gd_vi_bglr_mse__t-agdd"                                # G+P main-effects (AGDD)
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ -n "${SMOKE_N}" ]]; then
  ARTIFACT_ROOT="artifacts/cv/random_kfold-smoke"
else
  ARTIFACT_ROOT="artifacts/cv/random_kfold"
fi

# ── Pre-flight ─────────────────────────────────────────────────────────────
CLUSTER_UPPER=$(echo "${CLUSTER}" | tr '[:lower:]' '[:upper:]')
ACCOUNT="${CLUSTER_ACCOUNTS[${CLUSTER_UPPER}]:-}"
[[ -z "${ACCOUNT}" ]] && { echo "Error: no account for '${CLUSTER_UPPER}'."; exit 1; }

N_BASE=$(( ${#CV_SEEDS[@]} * ${#FOLDS[@]} ))
N_FPCA=$(( N_BASE * ${#FPCA_CONFIGS[@]} * ${#FPCA_SEEDS[@]} ))

echo "============================================================"
echo "random_kfold — FPCA/BGLR per-split submission"
echo "============================================================"
echo "Cluster:      ${CLUSTER_UPPER}  (account: ${ACCOUNT})"
echo "CV seeds:     ${#CV_SEEDS[@]}  folds: ${#FOLDS[@]}  (k_folds=${K_FOLDS})  → ${N_BASE} splits"
echo "FPCA configs: ${#FPCA_CONFIGS[@]}  FPCA seeds: ${FPCA_SEEDS[*]}"
echo "Max jobs:     ${N_FPCA}"
echo "Fold source:  native per-row (cv_seed=${CV_SEEDS[*]}); matches the DL random_kfold sweep"
[[ -n "${SMOKE_N}" ]] && echo "*** SMOKE MODE: smoke_n=${SMOKE_N} → ${ARTIFACT_ROOT} ***"
echo "------------------------------------------------------------"

cd "${REPO_ROOT}"

ENV_PATH="/scratch/user/$USER/neural-processes/env/bin/activate"
if ! command -v module &>/dev/null; then
  set +eu; source /etc/profile.d/hprc_profile.sh; set -eu
fi
module purge
module load GCCcore/13.2.0
module load Python/3.11.5
[[ -f "${ENV_PATH}" ]] || { echo "Error: venv not found at ${ENV_PATH}"; exit 1; }
source "${ENV_PATH}"

# Slug pre-computation (FPCA composers pull /optim: none — no scheduler tag).
declare -A SLUG_MAP
for config_name in "${FPCA_CONFIGS[@]}"; do
  SLUG_MAP["${config_name}"]=$(python -c "from cv.slug_util import resolve_slug; print(resolve_slug('fpca/${config_name}'))")
done
echo "Resolved slugs:"
for config_name in "${FPCA_CONFIGS[@]}"; do
  echo "  ${config_name} → ${SLUG_MAP[${config_name}]}"
done
echo "------------------------------------------------------------"

mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${SCRIPT_DIR}/logs/submit_fpca_$(date +%Y%m%dT%H%M%S)_$$.log"
exec > >(stdbuf -oL tee -a "${LOG_FILE}") 2>&1

submit_with_retry() {
  local rc
  for attempt in 1 2 3; do
    set +e; sbatch "$@"; rc=$?; set -e
    [[ $rc -eq 0 ]] && return 0
    echo "  sbatch failed (rc=$rc); retry ${attempt}/3 in $((attempt*5))s"
    sleep $((attempt * 5))
  done
  echo "  Giving up after 3 retries."; return 1
}

SUBMITTED=0; SKIPPED=0; FAILED=0

for cv_seed in "${CV_SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    token=$(python -c "from cv.slug_util import fold_dir_token; print(fold_dir_token(cv_seed=${cv_seed}, fold=${fold}))")

    for config_name in "${FPCA_CONFIGS[@]}"; do
      slug="${SLUG_MAP[${config_name}]}"
      for fpca_seed in "${FPCA_SEEDS[@]}"; do
        out="${ARTIFACT_ROOT}/${token}/${slug}/seed=${fpca_seed}"
        if [[ -f "${out}/metrics.json" ]]; then
          echo "SKIP   ${token}/${slug}/seed=${fpca_seed}"; (( ++SKIPPED )); continue
        fi
        EXPORTS="CV_SEED=${cv_seed},FOLD=${fold},K_FOLDS=${K_FOLDS},CONFIG=${config_name},SLUG=${slug},SEED=${fpca_seed},OUT_DIR=${out},WANDB_LOGGING=${FPCA_WANDB}"
        [[ -n "${SMOKE_N}" ]] && EXPORTS="${EXPORTS},SMOKE_N=${SMOKE_N}"
        if submit_with_retry \
          --job-name="rkfpca_${token}_${slug}_s${fpca_seed}" \
          --account="${ACCOUNT}" --time="${FPCA_TIME}" \
          --nodes=1 --ntasks-per-node=1 --cpus-per-task="${FPCA_CPUS}" \
          --mem="${FPCA_MEM}" \
          --mail-type=FAIL --mail-user="${SLURM_MAIL_USER}" \
          --export="${EXPORTS}" \
          "${SCRIPT_DIR}/fpca_job.job"; then
          echo "SUBMIT ${token}/${slug}/seed=${fpca_seed}"; (( ++SUBMITTED ))
        else
          echo "FAILED ${token}/${slug}/seed=${fpca_seed}"; (( ++FAILED ))
        fi
        sleep 0.2
      done
    done

  done
done

echo "------------------------------------------------------------"
echo "Submitted: ${SUBMITTED}  Skipped: ${SKIPPED}  Failed: ${FAILED}"
[[ ${FAILED} -gt 0 ]] && exit 1 || exit 0
