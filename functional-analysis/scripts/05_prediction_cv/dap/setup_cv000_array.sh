#!/bin/bash

#SBATCH --job-name=DAP_CV000_Setup
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=medium
#SBATCH --array=1-950%30
#SBATCH --output=%x.out.%A_%a
#SBATCH --error=%x.err.%A_%a

set -euo pipefail

SCRIPT_DIR="${G2F_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONFIG_FILE="${G2F_CONFIG_FILE:-${SCRIPT_DIR}/config.sh}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Could not find config.sh at ${CONFIG_FILE}. Set G2F_SCRIPT_DIR or G2F_CONFIG_FILE." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
SCRIPT_DIR="${G2F_SCRIPT_DIR:-${G2F_PIPELINE_DIR:-${SCRIPT_DIR}}}"

module load GCC/13.3.0 OpenMPI/5.0.3 R_tamu/4.4.2

PARAM_FILE="${PARAM_FILE:-${G2F_SETUP_PARAM_FILE_CV000}}"
if [[ ! -s "${PARAM_FILE}" ]]; then
  echo "Missing parameter file: ${PARAM_FILE}" >&2
  exit 1
fi

TOTAL_LINES="$(wc -l < "${PARAM_FILE}")"
if (( SLURM_ARRAY_TASK_ID > TOTAL_LINES )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is beyond ${TOTAL_LINES} lines in ${PARAM_FILE}; exiting."
  exit 0
fi

PARAM_LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PARAM_FILE}")"
read -r SEED FOLD SPLIT_GROUP HELDOUT_ENV <<< "${PARAM_LINE}"

if [[ "${SPLIT_GROUP}" != "CV_0_00" ]]; then
  echo "Expected CV_0_00 in parameter file but found ${SPLIT_GROUP}" >&2
  exit 1
fi

SPLIT_ID="$(compose_split_id "${SEED}" "${FOLD}" "${SPLIT_GROUP}" "${HELDOUT_ENV}")"
SPLIT_FILE="${G2F_CV_OUT_PATH}/bundles/split_bundles/${SPLIT_ID}.rds"

echo "JOB ELEMENT ID: ${SLURM_ARRAY_TASK_ID}"
echo "Split: ${SPLIT_ID}"
echo "Output split file: ${SPLIT_FILE}"

if truthy "${G2F_SKIP_COMPLETED}" && [[ -s "${SPLIT_FILE}" ]]; then
  echo "Split bundle already exists; skipping."
  exit 0
fi

Rscript --rtamuenvs="${G2F_R_ENV_NAME}" "${SCRIPT_DIR}/build_fold_kernels.R" \
  "${SEED}" "${FOLD}" "${SPLIT_GROUP}" "${HELDOUT_ENV}"
