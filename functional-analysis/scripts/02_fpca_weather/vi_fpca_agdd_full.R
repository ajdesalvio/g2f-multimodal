# Unified AGDD-domain FPCA for plot-level vegetation-index BLUE trajectories.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(fdapace)
  library(tidyr)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
source(file.path(project_dir, "R", "utils", "fpca.R"))
source(file.path(project_dir, "R", "utils", "vi_fpca.R"))

paths <- g2f_paths()
weather <- data.table::fread(
  g2f_resolve_input(
    paths,
    "EnvRtype_Weather_Data_Cleaned_V2.csv",
    result_subdirs = "02_fpca_weather/envrtype_dap"
  )
) |>
  dplyr::arrange(.data$Env, .data$DAP) |>
  dplyr::group_by(.data$Env) |>
  dplyr::mutate(AGDD = cumsum(.data$GDD)) |>
  dplyr::ungroup() |>
  dplyr::select(.data$Env, .data$DAP, .data$AGDD)

vi_data <- data.table::fread(
  g2f_resolve_input(
    paths,
    "VI_BLUEs_G2F_2020_2021.CSV",
    result_subdirs = "01_blue_variance"
  )
) |>
  dplyr::left_join(weather, by = c("Env", "DAP"))

g2f_run_vi_fpca(
  vi_data,
  time_col = "AGDD",
  paths = paths,
  output_subdir = "vi_agdd",
  score_filename = "FPC_Scores_BLUEs_AGDD_Unified_FPCA_Max_FPCs_2020_2021_G2F_VIs.csv",
  curve_filename = "FPCA_Predicted_Curves_BLUEs_AGDD_Unified_FPCA.csv"
)
