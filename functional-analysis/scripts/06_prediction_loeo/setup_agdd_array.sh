#!/bin/bash

## NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=AGDD_LOEO_Setup
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=37G
#SBATCH --partition=long
#SBATCH --array=1-19%10
#SBATCH --output=%x.out.%A_%a
#SBATCH --error=%x.err.%A_%a

set -euo pipefail

module load GCC/13.3.0 OpenMPI/5.0.3 R_tamu/4.4.2

SCRIPT_DIR="${G2F_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export G2F_TIME_DOMAIN=AGDD
# shellcheck source=/dev/null
source "${G2F_CONFIG_FILE:-${SCRIPT_DIR}/config.sh}"
PARAM_FILE="${G2F_PARAM_FILE:-${SCRIPT_DIR}/agdd_heldout_environments.txt}"
R_SCRIPT="${SCRIPT_DIR}/build_agdd_kernels.R"

HELDOUT_ENV="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PARAM_FILE}")"

if [[ -z "${HELDOUT_ENV}" ]]; then
  echo "No held-out environment found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

export G2F_INCLUDE_ZE=FALSE

echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Held-out environment: ${HELDOUT_ENV}"
echo "R script: ${R_SCRIPT}"
echo "Parameter file: ${PARAM_FILE}"

Rscript --rtamuenvs="${G2F_R_ENV_NAME}" "${R_SCRIPT}" "${HELDOUT_ENV}"
