#!/bin/bash

# Shared configuration for LOEO setup and prediction jobs. Override any value
# below by exporting the corresponding environment variable before submission.

G2F_SCRIPT_DIR="${G2F_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
G2F_PROJECT_DIR="${G2F_PROJECT_DIR:-$(cd "${G2F_SCRIPT_DIR}/../.." && pwd)}"
G2F_TIME_DOMAIN="${G2F_TIME_DOMAIN:-DAP}"
G2F_TIME_DOMAIN_LOWER="${G2F_TIME_DOMAIN,,}"

export G2F_SCRIPT_DIR G2F_PROJECT_DIR G2F_TIME_DOMAIN
export G2F_R_ENV_NAME="${G2F_R_ENV_NAME:-BGLR_Mod_4.4.2}"
export G2F_DATA_PATH="${G2F_DATA_PATH:-${G2F_DATA_DIR:-${G2F_PROJECT_DIR}/data}}"

G2F_LOEO_ROOT="${G2F_LOEO_ROOT:-${G2F_RESULTS_DIR:-${G2F_PROJECT_DIR}/results}/prediction/loeo/${G2F_TIME_DOMAIN_LOWER}}"
export G2F_LOEO_SCORE_PATH="${G2F_LOEO_SCORE_PATH:-${G2F_LOEO_ROOT}/projected_scores}"
export G2F_KERNEL_OUT_PATH="${G2F_KERNEL_OUT_PATH:-${G2F_LOEO_ROOT}/kernels}"
export G2F_RESULTS_PATH="${G2F_RESULTS_PATH:-${G2F_LOEO_ROOT}/predictions}"
export G2F_WORK_PATH="${G2F_WORK_PATH:-${TMPDIR:-${G2F_LOEO_ROOT}/work}}"

# The manuscript reports only the no-Ze analysis. NGRDI and PTR were selected
# a priori. Override only for an explicitly separate sensitivity analysis.
export G2F_INCLUDE_ZE="${G2F_INCLUDE_ZE:-FALSE}"
export G2F_VI_NAMES="${G2F_VI_NAMES:-NGRDI}"
export G2F_WEATHER_TRAITS="${G2F_WEATHER_TRAITS:-PTR}"
export G2F_WEATHER_NFPCS="${G2F_WEATHER_NFPCS:-1}"

# BGLR_Mod_4.4.2 must include patches/BGLR-single-predictor-drop-false.patch. The
# drop = FALSE change is required when PTR contributes only one predictor.
