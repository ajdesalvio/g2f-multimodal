# Fit AGDD-domain FPCA models to the cleaned EnvRtype weather data.
# Historical source: 1.5_Weather_GDD_FPCA/Weather_GDD_Domain_FPCA_V2.R

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(fdapace)
  library(tidyr)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
source(file.path(project_dir, "R", "utils", "fpca.R"))

paths <- g2f_paths()
output_dir <- g2f_results_dir(paths, "02_fpca_weather", "envrtype_agdd")
model_dir <- file.path(output_dir, "models")
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

cleaned_weather_file <- g2f_resolve_input(
  paths,
  "EnvRtype_Weather_Data_Cleaned_V2.csv",
  result_subdirs = file.path("02_fpca_weather", "envrtype_dap")
)
g2f_require_file(cleaned_weather_file, "Cleaned EnvRtype weather data")
weather <- fread(cleaned_weather_file)

required_columns <- c("Env", "DAP", "GDD")
missing_columns <- setdiff(required_columns, names(weather))
if (length(missing_columns) > 0L) {
  stop("Cleaned weather data are missing: ", paste(missing_columns, collapse = ", "))
}

weather <- weather |>
  arrange(.data$Env, .data$DAP) |>
  group_by(.data$Env) |>
  mutate(AGDD = cumsum(.data$GDD)) |>
  ungroup()

weather_variables <- intersect(as.character(paths$weather_traits), names(weather))
if (length(weather_variables) == 0L) {
  stop(
    "None of the configured weather traits are present. Configured: ",
    paste(paths$weather_traits, collapse = ", ")
  )
}

score_list <- list()
wide_curve_list <- list()
tall_curve_list <- list()
mean_curve_list <- list()

for (weather_variable in weather_variables) {
  message("Fitting AGDD weather FPCA: ", weather_variable)
  fpca <- g2f_fit_sparse_fpca(
    data = weather,
    id_col = "Env",
    time_col = "AGDD",
    value_col = weather_variable,
    max_components = 4L,
    n_reg_grid = 100L,
    plot = FALSE
  )

  scores <- fpca$scores |>
    mutate(
      Year = sub("^.*\\.", "", .data$Env),
      Env.Weather.Var = paste(.data$Env, weather_variable, sep = "."),
      Weather.Var = weather_variable
    )

  curves_wide <- fpca$predicted_curves_wide |>
    mutate(
      Year = sub("^.*\\.", "", .data$Env),
      Env.Weather.Var = paste(.data$Env, weather_variable, sep = "."),
      Weather.Var = weather_variable
    ) |>
    relocate(.data$Env, .data$Year, .data$Env.Weather.Var, .data$Weather.Var)

  curves_tall <- fpca$predicted_curves_tall |>
    mutate(
      Year = sub("^.*\\.", "", .data$Env),
      Env.Weather.Var = paste(.data$Env, weather_variable, sep = "."),
      Weather.Var = weather_variable
    ) |>
    relocate(.data$Env, .data$Year, .data$Env.Weather.Var, .data$Weather.Var)

  mean_curve <- fpca$mean_curve |>
    rename(AGDD = .data$Time) |>
    mutate(Weather.Var = weather_variable)

  score_list[[weather_variable]] <- scores
  wide_curve_list[[weather_variable]] <- curves_wide
  tall_curve_list[[weather_variable]] <- curves_tall
  mean_curve_list[[weather_variable]] <- mean_curve

  model_name <- gsub("[^A-Za-z0-9_.-]", "_", weather_variable)
  saveRDS(fpca$model, file.path(model_dir, paste0("AGDD_Weather_FPCA_", model_name, ".rds")))
}

scores <- bind_rows(score_list)
curves_wide <- bind_rows(wide_curve_list)
curves_tall <- bind_rows(tall_curve_list)
mean_curves <- bind_rows(mean_curve_list)

fwrite(scores, file.path(output_dir, "Weather_FPCA_AGDD_EnvRtype_Data_Scores_PTR_ONLY_V1.csv"))
fwrite(curves_wide, file.path(output_dir, "Weather_FPCA_AGDD_EnvRtype_Data_Predcurves_Wide_PTR_ONLY_V1.csv"))
fwrite(curves_tall, file.path(output_dir, "Weather_FPCA_AGDD_EnvRtype_Data_Predcurves_Tall_PTR_ONLY_V1.csv"))
fwrite(mean_curves, file.path(output_dir, "Weather_FPCA_AGDD_EnvRtype_Data_Mean_Curves_PTR_ONLY_V1.csv"))

message("AGDD-domain EnvRtype weather outputs written to: ", output_dir)
