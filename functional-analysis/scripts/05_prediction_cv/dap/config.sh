#!/bin/bash

# Shared HPRC configuration for the DAP CV FPCA projection pipeline. Override
# any value by exporting it before submission.

export G2F_SCRIPT_DIR="${G2F_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export G2F_PIPELINE_DIR="${G2F_PIPELINE_DIR:-${G2F_SCRIPT_DIR}}"
export G2F_PROJECT_DIR="${G2F_PROJECT_DIR:-$(cd "${G2F_SCRIPT_DIR}/../../.." && pwd)}"

export G2F_R_ENV_NAME="${G2F_R_ENV_NAME:-BGLR_Mod_4.4.2}"
export G2F_DATA_PATH="${G2F_DATA_PATH:-${G2F_DATA_DIR:-${G2F_PROJECT_DIR}/data}}"
export G2F_CV_OUT_PATH="${G2F_CV_OUT_PATH:-${G2F_RESULTS_DIR:-${G2F_PROJECT_DIR}/results}/prediction/cv/dap}"

export G2F_JOBS_DIR="${G2F_JOBS_DIR:-${G2F_CV_OUT_PATH}/jobs}"
export G2F_METADATA_DIR="${G2F_METADATA_DIR:-${G2F_CV_OUT_PATH}/metadata}"
export G2F_LOEO_SCORE_PATH="${G2F_LOEO_SCORE_PATH:-${G2F_RESULTS_DIR:-${G2F_PROJECT_DIR}/results}/prediction/loeo/dap/projected_scores}"

export G2F_WEATHER_ALL_FILE="${G2F_WEATHER_ALL_FILE:-Weather_FPCA_Scores_ALL.csv}"
export G2F_WEATHER_LOEO_FILE="${G2F_WEATHER_LOEO_FILE:-Weather_FPC_Scores_DAP_LOEO_Projected.csv}"
export G2F_WEATHER_TRAITS="${G2F_WEATHER_TRAITS:-PTR}"
export G2F_WEATHER_NFPCS="${G2F_WEATHER_NFPCS:-1}"

# NGRDI and PTR were selected a priori for the manuscript analysis.
export G2F_VI_NAMES="${G2F_VI_NAMES:-NGRDI}"
export G2F_VI_NFPCS="${G2F_VI_NFPCS:-5}"

export G2F_N_ITER="${G2F_N_ITER:-10000}"
export G2F_BURN_IN="${G2F_BURN_IN:-1000}"
export G2F_SAVE_PRED_VALUES="${G2F_SAVE_PRED_VALUES:-TRUE}"
export G2F_SAVE_EVAL_ONLY="${G2F_SAVE_EVAL_ONLY:-TRUE}"
export G2F_SAVE_VI_SCORES="${G2F_SAVE_VI_SCORES:-FALSE}"
export G2F_INCLUDE_ZE="${G2F_INCLUDE_ZE:-FALSE}"

# G2F_R_ENV_NAME must name an environment containing the documented
# drop = FALSE BGLR patch required by the single-PTR-predictor model.

export G2F_SKIP_COMPLETED="${G2F_SKIP_COMPLETED:-TRUE}"

export G2F_SETUP_PARAM_FILE_CV21="${G2F_SETUP_PARAM_FILE_CV21:-${G2F_JOBS_DIR}/DAP_CV_Setup_Jobs_CV_2_1.txt}"
export G2F_SETUP_PARAM_FILE_CV000="${G2F_SETUP_PARAM_FILE_CV000:-${G2F_JOBS_DIR}/DAP_CV_Setup_Jobs_CV_0_00.txt}"
export G2F_PRED_PARAM_FILE_CV21="${G2F_PRED_PARAM_FILE_CV21:-${G2F_JOBS_DIR}/DAP_CV_Prediction_Jobs_CV_2_1.txt}"
export G2F_PRED_PARAM_FILE_CV000="${G2F_PRED_PARAM_FILE_CV000:-${G2F_JOBS_DIR}/DAP_CV_Prediction_Jobs_CV_0_00.txt}"

truthy() {
  case "${1^^}" in
    TRUE|1|YES|Y) return 0 ;;
    *) return 1 ;;
  esac
}

sanitize_component() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

compose_split_id() {
  local seed="$1"
  local fold="$2"
  local split_group="$3"
  local heldout_env="$4"
  local safe_env
  safe_env="$(sanitize_component "${heldout_env}")"
  printf 'Seed%02d.Fold%d.%s.%s' "${seed}" "${fold}" "${split_group}" "${safe_env}"
}
