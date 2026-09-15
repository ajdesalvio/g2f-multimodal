# Functional principal components for in-field air-temperature trajectories.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(fdapace)
  library(tidyr)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
source(file.path(project_dir, "R", "utils", "fpca.R"))
source(file.path(project_dir, "R", "utils", "station_weather.R"))

paths <- g2f_paths()
analysis <- g2f_analysis_config()
trait <- "Temperature [C]"

weather <- g2f_prepare_station_weather(paths, analysis)
weather <- g2f_apply_dap_windows(
  weather,
  trait,
  analysis$weather$station_air_temperature
)
cleaned <- g2f_prioritize_station_observations(weather, trait)

fit <- g2f_fit_sparse_fpca(
  cleaned,
  id_col = "Env",
  time_col = "Weather.DAP",
  value_col = trait,
  max_components = 4L,
  n_reg_grid = 20000L,
  plot = FALSE
)

g2f_write_station_fpca(
  fit,
  cleaned_data = cleaned,
  paths = paths,
  trait = trait,
  filename_stem = "Weather_FPCA_Air_Temp",
  tall_value_name = "Temperature.C"
)
