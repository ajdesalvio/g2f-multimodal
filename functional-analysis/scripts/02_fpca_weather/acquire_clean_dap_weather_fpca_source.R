# Acquire, clean, and fit DAP-domain FPCA models to EnvRtype weather data.
# Historical source: 1.5_Weather_GDD_FPCA/Weather_GDD_PTR_V8.R

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(EnvRtype)
  library(fdapace)
  library(lubridate)
  library(tidyr)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
source(file.path(project_dir, "R", "utils", "fpca.R"))

paths <- g2f_paths()
analysis <- g2f_analysis_config()
output_dir <- g2f_results_dir(paths, "02_fpca_weather", "envrtype_dap")
model_dir <- file.path(output_dir, "models")
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

coordinates_file <- g2f_data_file(paths, "Collaborator_Files", "Envs_Coordinates.csv")
max_flight_dap_file <- g2f_data_file(paths, "Collaborator_Files", "Max.Flight.DAP.csv")
planting_dates_file <- g2f_data_file(paths, "Collaborator_Files", "Planting.Dates.csv")
archived_raw_weather_file <- g2f_data_file(paths, "EnvRtype_Raw_Weather.csv")
downloaded_raw_weather_file <- file.path(output_dir, "EnvRtype_Raw_Weather.csv")

invisible(lapply(
  c(coordinates_file, max_flight_dap_file, planting_dates_file),
  g2f_require_file
))

refresh_weather <- g2f_bool(Sys.getenv("G2F_REFRESH_WEATHER", unset = "FALSE"))
if (refresh_weather || !file.exists(archived_raw_weather_file)) {
  coordinates <- fread(coordinates_file) |>
    left_join(fread(max_flight_dap_file), by = "Env") |>
    left_join(fread(planting_dates_file), by = "Env") |>
    mutate(
      Planting.Date.POSIXct = ymd(.data$Planting.Date, tz = "UTC"),
      Final.Flight.Date = as.character(
        .data$Planting.Date.POSIXct + days(.data$Max.Flight.DAP)
      )
    )

  raw_weather <- EnvRtype::get_weather(
    env.id = coordinates$Env,
    lat = coordinates$Latitude,
    lon = coordinates$Longitude,
    start.day = coordinates$Planting.Date,
    end.day = coordinates$Final.Flight.Date
  )
  fwrite(raw_weather, downloaded_raw_weather_file)
} else {
  raw_weather <- fread(archived_raw_weather_file)
}

weather <- EnvRtype::processWTH(
  env.data = as.data.frame(raw_weather),
  Tbase1 = 8,
  Tbase2 = 45,
  Topt1 = 30,
  Topt2 = 37
)

weather$PTR <- weather$N / weather$GDD
weather$PTT <- weather$GDD * weather$N

clean_outliers <- function(values, multiplier, absolute = FALSE) {
  values <- as.numeric(values)
  values[!is.finite(values)] <- NA_real_
  threshold <- multiplier * stats::sd(values, na.rm = TRUE)
  remove <- if (absolute) abs(values) > threshold else values > threshold
  remove[is.na(remove)] <- FALSE
  list(values = replace(values, remove, NA_real_), threshold = threshold, removed = sum(remove))
}

ptr_rule <- analysis$weather$ptr_outlier_rule
ptr_clean <- clean_outliers(
  weather$PTR,
  multiplier = ptr_rule$standard_deviation_multiplier,
  absolute = identical(ptr_rule$direction, "absolute")
)
weather$PTR <- ptr_clean$values

par_clean <- NULL
if ("PAR_TEMP" %in% names(weather)) {
  par_rule <- analysis$weather$par_temp_outlier_rule
  par_clean <- clean_outliers(
    weather$PAR_TEMP,
    multiplier = par_rule$standard_deviation_multiplier,
    absolute = identical(par_rule$direction, "absolute")
  )
  weather$PAR_TEMP <- par_clean$values
}

weather$DAP <- weather$daysFromStart
weather <- weather[, c("DAP", setdiff(names(weather), "DAP")), drop = FALSE]

cleaning_summary <- data.frame(
  variable = c("PTR", if (!is.null(par_clean)) "PAR_TEMP"),
  direction = c(ptr_rule$direction, if (!is.null(par_clean)) par_rule$direction),
  sd_multiplier = c(
    ptr_rule$standard_deviation_multiplier,
    if (!is.null(par_clean)) par_rule$standard_deviation_multiplier
  ),
  threshold = c(ptr_clean$threshold, if (!is.null(par_clean)) par_clean$threshold),
  observations_removed = c(ptr_clean$removed, if (!is.null(par_clean)) par_clean$removed)
)

cleaned_weather_file <- file.path(output_dir, "EnvRtype_Weather_Data_Cleaned_V2.csv")
fwrite(weather, cleaned_weather_file)
fwrite(cleaning_summary, file.path(output_dir, "EnvRtype_Weather_Cleaning_Summary.csv"))

metadata_columns <- c("DAP", "Env", "YYYYMMDD", "LON", "LAT", "DOY", "daysFromStart")
excluded_variables <- c("T2M", "T2M_MAX", "T2M_MIN", "FROST_DAYS")
weather_variables <- setdiff(names(weather), c(metadata_columns, excluded_variables))
weather_variables <- weather_variables[vapply(weather[weather_variables], is.numeric, logical(1))]

score_list <- list()
wide_curve_list <- list()
tall_curve_list <- list()
mean_curve_list <- list()

for (weather_variable in weather_variables) {
  message("Fitting DAP weather FPCA: ", weather_variable)
  fpca <- g2f_fit_sparse_fpca(
    data = weather,
    id_col = "Env",
    time_col = "DAP",
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
    rename(DAP = .data$Time) |>
    mutate(Weather.Var = weather_variable)

  score_list[[weather_variable]] <- scores
  wide_curve_list[[weather_variable]] <- curves_wide
  tall_curve_list[[weather_variable]] <- curves_tall
  mean_curve_list[[weather_variable]] <- mean_curve

  model_name <- gsub("[^A-Za-z0-9_.-]", "_", weather_variable)
  saveRDS(fpca$model, file.path(model_dir, paste0("DAP_Weather_FPCA_", model_name, ".rds")))
}

scores <- bind_rows(score_list)
curves_wide <- bind_rows(wide_curve_list)
curves_tall <- bind_rows(tall_curve_list)
mean_curves <- bind_rows(mean_curve_list)

fwrite(scores, file.path(output_dir, "Weather_FPCA_EnvRtype_Data_Scores_V2.csv"))
fwrite(curves_wide, file.path(output_dir, "Weather_FPCA_EnvRtype_Data_Predcurves_Wide_V2.csv"))
fwrite(curves_tall, file.path(output_dir, "Weather_FPCA_EnvRtype_Data_Predcurves_Tall_V2.csv"))
fwrite(mean_curves, file.path(output_dir, "Weather_FPCA_EnvRtype_Data_Mean_Curves_V2.csv"))

message("DAP-domain EnvRtype weather outputs written to: ", output_dir)
