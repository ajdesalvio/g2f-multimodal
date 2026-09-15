#!/bin/bash

#SBATCH --job-name=AGDD_CV_FPCA_Prep
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
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

mkdir -p "${G2F_CV_OUT_PATH}" "${G2F_JOBS_DIR}" "${G2F_METADATA_DIR}"

echo "Pipeline directory: ${G2F_PIPELINE_DIR}"
echo "Output root: ${G2F_CV_OUT_PATH}"
echo "Data path: ${G2F_DATA_PATH}"
echo "Climate file: ${G2F_CLIMATE_FILE}"
echo "VI source file: ${G2F_VI_SOURCE_FILE}"
echo "AGDD VI file: ${G2F_VI_FILE}"
echo "Weather traits: ${G2F_WEATHER_TRAITS}"
echo "Weather FPCs: ${G2F_WEATHER_NFPCS}"
echo "VI names: ${G2F_VI_NAMES}"
echo "VI FPCs: ${G2F_VI_NFPCS}"

Rscript --rtamuenvs="${G2F_R_ENV_NAME}" "${SCRIPT_DIR}/preprocess_agdd_inputs.R"
Rscript --rtamuenvs="${G2F_R_ENV_NAME}" "${SCRIPT_DIR}/build_metadata.R"
Rscript --rtamuenvs="${G2F_R_ENV_NAME}" "${SCRIPT_DIR}/build_constant_kernels.R"
Rscript --rtamuenvs="${G2F_R_ENV_NAME}" "${SCRIPT_DIR}/build_weather_inputs.R"

COMBINED_SETUP_FILE="${G2F_JOBS_DIR}/AGDD_CV_Setup_Jobs.txt"
COMBINED_PRED_FILE="${G2F_JOBS_DIR}/AGDD_CV_Prediction_Jobs.txt"
LEGACY_JOBS_DIR="${G2F_CV_OUT_PATH}/job_tables"

if [[ ! -s "${COMBINED_SETUP_FILE}" ]]; then
  if [[ -s "${LEGACY_JOBS_DIR}/AGDD_CV_Setup_Jobs.txt" ]]; then
    COMBINED_SETUP_FILE="${LEGACY_JOBS_DIR}/AGDD_CV_Setup_Jobs.txt"
    echo "Using legacy setup job table: ${COMBINED_SETUP_FILE}"
  else
    echo "Missing setup job table: ${COMBINED_SETUP_FILE}" >&2
    exit 1
  fi
fi

if [[ ! -s "${COMBINED_PRED_FILE}" ]]; then
  if [[ -s "${LEGACY_JOBS_DIR}/AGDD_CV_Prediction_Jobs.txt" ]]; then
    COMBINED_PRED_FILE="${LEGACY_JOBS_DIR}/AGDD_CV_Prediction_Jobs.txt"
    echo "Using legacy prediction job table: ${COMBINED_PRED_FILE}"
  else
    echo "Missing prediction job table: ${COMBINED_PRED_FILE}" >&2
    exit 1
  fi
fi

awk '$3=="CV_2_1"{print}' "${COMBINED_SETUP_FILE}" > "${G2F_SETUP_PARAM_FILE_CV21}"
awk '$3=="CV_0_00"{print}' "${COMBINED_SETUP_FILE}" > "${G2F_SETUP_PARAM_FILE_CV000}"
awk '$3=="CV_2_1"{print}' "${COMBINED_PRED_FILE}" > "${G2F_PRED_PARAM_FILE_CV21}"
awk '$3=="CV_0_00"{print}' "${COMBINED_PRED_FILE}" > "${G2F_PRED_PARAM_FILE_CV000}"

echo "Prep completed."
echo "CV_2_1 setup jobs: $(wc -l < "${G2F_SETUP_PARAM_FILE_CV21}")"
echo "CV_0_00 setup jobs: $(wc -l < "${G2F_SETUP_PARAM_FILE_CV000}")"
echo "CV_2_1 prediction jobs: $(wc -l < "${G2F_PRED_PARAM_FILE_CV21}")"
echo "CV_0_00 prediction jobs: $(wc -l < "${G2F_PRED_PARAM_FILE_CV000}")"
