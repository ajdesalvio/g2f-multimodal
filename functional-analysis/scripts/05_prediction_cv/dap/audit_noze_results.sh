#!/bin/bash

#SBATCH --job-name=DAP_CV_Audit_noZe
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=medium
#SBATCH --output=%x.out.%j
#SBATCH --error=%x.err.%j

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

export G2F_INCLUDE_ZE=FALSE

echo "Auditing no-Ze prediction outputs under ${G2F_CV_OUT_PATH}/results"
Rscript --rtamuenvs="${G2F_R_ENV_NAME}" "${SCRIPT_DIR}/verify_results.R"
