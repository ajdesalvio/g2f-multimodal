#!/bin/bash

# Shared task body for the four initial no-Ze CV prediction arrays. The calling
# launcher loads R first; this child shell re-sources the domain configuration
# so its helper functions are available here as well.

set -euo pipefail

EXPECTED_SPLIT="${1:?Expected split group is required.}"
PARAM_FILE="${2:?Prediction parameter file is required.}"
DOMAIN_SCRIPT_DIR="${3:?Domain script directory is required.}"

CONFIG_FILE="${G2F_CONFIG_FILE:-${DOMAIN_SCRIPT_DIR}/config.sh}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Could not find domain config at ${CONFIG_FILE}. Set G2F_CONFIG_FILE." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONFIG_FILE}"
SCRIPT_DIR="${G2F_SCRIPT_DIR:-${DOMAIN_SCRIPT_DIR}}"

if truthy "${G2F_INCLUDE_ZE}"; then
  echo "This manuscript launcher is restricted to the final no-Ze analysis." >&2
  exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/run_prediction.R" ]]; then
  echo "Missing prediction script: ${SCRIPT_DIR}/run_prediction.R" >&2
  exit 1
fi

if [[ ! -s "${PARAM_FILE}" ]]; then
  echo "Missing prediction parameter file: ${PARAM_FILE}" >&2
  exit 1
fi

TOTAL_LINES="$(wc -l < "${PARAM_FILE}")"
if (( SLURM_ARRAY_TASK_ID > TOTAL_LINES )); then
  echo "Array task ${SLURM_ARRAY_TASK_ID} is beyond ${TOTAL_LINES} lines; exiting."
  exit 0
fi

PARAM_LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${PARAM_FILE}")"
read -r SEED FOLD SPLIT_GROUP HELDOUT_ENV <<< "${PARAM_LINE}"

if [[ "${SPLIT_GROUP}" != "${EXPECTED_SPLIT}" ]]; then
  echo "Expected ${EXPECTED_SPLIT} but found ${SPLIT_GROUP}." >&2
  exit 1
fi

SPLIT_ID="$(compose_split_id "${SEED}" "${FOLD}" "${SPLIT_GROUP}" "${HELDOUT_ENV}")"
SPLIT_FILE="${G2F_CV_OUT_PATH}/bundles/split_bundles/${SPLIT_ID}.rds"
METRICS_FILE="${G2F_CV_OUT_PATH}/results/${SPLIT_ID}.noZe.metrics.csv"

if [[ ! -s "${SPLIT_FILE}" ]]; then
  echo "Missing split bundle: ${SPLIT_FILE}" >&2
  exit 1
fi
if truthy "${G2F_SKIP_COMPLETED}" && [[ -s "${METRICS_FILE}" ]]; then
  echo "Metrics already exist; skipping ${SPLIT_ID}."
  exit 0
fi

echo "Split: ${SPLIT_ID}"
echo "Include Ze: ${G2F_INCLUDE_ZE}"
Rscript --rtamuenvs="${G2F_R_ENV_NAME}" "${SCRIPT_DIR}/run_prediction.R" \
  "${SEED}" "${FOLD}" "${SPLIT_GROUP}" "${HELDOUT_ENV}"
