#!/bin/bash

#SBATCH --job-name=G2F_LOEO_DAP_noZe
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --partition=medium
#SBATCH --array=1-266%30
#SBATCH --output=%x.out.%A_%a
#SBATCH --error=%x.err.%A_%a

set -euo pipefail

module load GCC/13.3.0 OpenMPI/5.0.3 R_tamu/4.4.2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export G2F_TIME_DOMAIN=DAP
# shellcheck source=/dev/null
source "${G2F_CONFIG_FILE:-${SCRIPT_DIR}/config.sh}"
PARAM_FILE="${G2F_PARAM_FILE:-${SCRIPT_DIR}/prediction_job_array.txt}"
R_SCRIPT="${SCRIPT_DIR}/run_dap_prediction.R"

export G2F_INCLUDE_ZE=FALSE
export G2F_MODEL_PATH="${G2F_MODEL_PATH:-${G2F_KERNEL_OUT_PATH}/noZe/models}"

PARAM_LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PARAM_FILE}" | tr -d '\r')"
read -r MODEL HELDOUT_ENV <<< "${PARAM_LINE}"

if [[ -z "${MODEL:-}" || -z "${HELDOUT_ENV:-}" ]]; then
  echo "Could not parse task ${SLURM_ARRAY_TASK_ID} from ${PARAM_FILE}." >&2
  exit 1
fi

echo "Model: ${MODEL}"
echo "Held-out environment: ${HELDOUT_ENV}"
echo "Include Ze: ${G2F_INCLUDE_ZE}"
echo "Model path: ${G2F_MODEL_PATH}"

Rscript --rtamuenvs="${G2F_R_ENV_NAME}" "${R_SCRIPT}" "${MODEL}" "${HELDOUT_ENV}"
