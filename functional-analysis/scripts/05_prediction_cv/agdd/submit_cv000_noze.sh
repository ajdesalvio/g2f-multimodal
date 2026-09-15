#!/bin/bash

#SBATCH --job-name=AGDD_CV000_noZe
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --array=1-950%30
#SBATCH --output=%x.out.%A_%a
#SBATCH --error=%x.err.%A_%a

set -euo pipefail
SCRIPT_DIR="${G2F_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=/dev/null
source "${G2F_CONFIG_FILE:-${SCRIPT_DIR}/config.sh}"
module load GCC/13.3.0 OpenMPI/5.0.3 R_tamu/4.4.2
export G2F_INCLUDE_ZE=FALSE

bash "${G2F_PROJECT_DIR}/scripts/05_prediction_cv/prediction_array_task.sh" \
  "CV_0_00" "${G2F_PRED_PARAM_FILE_CV000}" "${SCRIPT_DIR}"
