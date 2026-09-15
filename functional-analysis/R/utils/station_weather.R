g2f_parse_dates <- function(x, formats) {
  x <- as.character(x)
  parsed <- as.Date(rep(NA_character_, length(x)))

  for (format in formats) {
    missing <- is.na(parsed) & !is.na(x) & nzchar(x)
    parsed[missing] <- as.Date(x[missing], format = format)
  }

  parsed
}

g2f_parse_weather_datetimes <- function(x, tz = "UTC") {
  x <- as.character(x)
  parsed <- as.POSIXct(rep(NA_character_, length(x)), tz = tz)
  formats <- c(
    "%m/%d/%Y %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ"
  )

  for (format in formats) {
    missing <- is.na(parsed) & !is.na(x) & nzchar(x)
    parsed[missing] <- as.POSIXct(x[missing], format = format, tz = tz)
  }

  parsed
}

g2f_modal_value <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0L) return(NA)
  counts <- sort(table(x), decreasing = TRUE)
  names(counts)[[1L]]
}

g2f_prepare_station_weather <- function(paths, analysis) {
  phenotype_2020 <- data.table::fread(
    g2f_data_file(paths, "g2f_2020_phenotypic_clean_data.csv")
  )
  phenotype_2021 <- data.table::fread(
    g2f_data_file(paths, "g2f_2021_phenotypic_clean_data.csv")
  )

  phenotype_2020$Planting.Date <- g2f_parse_dates(
    phenotype_2020[["Date Plot Planted [MM/DD/YY]"]],
    c("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d")
  )
  phenotype_2021$Planting.Date <- g2f_parse_dates(
    phenotype_2021[["Date Plot Planted [MM/DD/YY]"]],
    c("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y")
  )

  phenotype_columns <- c("Year", "Field-Location", "Planting.Date")
  phenotype <- dplyr::bind_rows(
    phenotype_2020[, ..phenotype_columns],
    phenotype_2021[, ..phenotype_columns]
  ) |>
    dplyr::mutate(
      Env = paste(.data[["Field-Location"]], .data$Year, sep = ".")
    )

  environments <- unlist(analysis$drone$environments, use.names = FALSE)
  planting_dates <- phenotype |>
    dplyr::filter(.data$Env %in% environments) |>
    dplyr::group_by(.data$Env) |>
    dplyr::summarise(
      Planting.Date = as.Date(g2f_modal_value(.data$Planting.Date)),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      Planting.Date.Time = as.POSIXct(
        paste(.data$Planting.Date, "09:00"),
        format = "%Y-%m-%d %H:%M",
        tz = "UTC"
      )
    )

  weather_2020 <- data.table::fread(
    g2f_data_file(paths, "g2f_2020_weather_cleaned.csv")
  )
  weather_2021 <- data.table::fread(
    g2f_data_file(paths, "g2f_2021_weather_cleaned.csv")
  )
  weather_2020[["CO2 [ppm]"]] <- NULL

  weather_columns <- c(
    "Date_key",
    "Temperature [C]",
    "Relative Humidity [%]",
    "Soil Temperature [C]",
    "Field Location",
    "Year",
    "NWS Network"
  )
  missing_weather_columns <- setdiff(
    weather_columns,
    intersect(names(weather_2020), names(weather_2021))
  )
  if (length(missing_weather_columns) > 0L) {
    stop(
      "Weather inputs are missing required columns: ",
      paste(missing_weather_columns, collapse = ", ")
    )
  }

  weather <- dplyr::bind_rows(
    weather_2020[, ..weather_columns],
    weather_2021[, ..weather_columns]
  ) |>
    dplyr::mutate(
      EnvW = paste(.data[["Field Location"]], .data$Year, sep = ".")
    )

  # TXH1 and TXH3 share a station but have different planting dates. Duplicate
  # the shared record before assigning DAP, as in the original analysis.
  weather_environments <- environments
  weather_environments[grepl("^TXH[13]\\.", weather_environments)] <- sub(
    "^TXH[13]\\.", "TXH1_TXH3.",
    weather_environments[grepl("^TXH[13]\\.", weather_environments)]
  )
  weather <- dplyr::filter(weather, .data$EnvW %in% unique(weather_environments))

  shared_texas <- dplyr::filter(
    weather,
    .data$EnvW %in% c("TXH1_TXH3.2020", "TXH1_TXH3.2021")
  )
  other_weather <- dplyr::filter(
    weather,
    !.data$EnvW %in% c("TXH1_TXH3.2020", "TXH1_TXH3.2021")
  ) |>
    dplyr::mutate(Env = paste(.data[["Field Location"]], .data$Year, sep = "."))

  texas_weather <- dplyr::bind_rows(
    dplyr::mutate(
      dplyr::filter(shared_texas, .data$Year == 2020),
      Env = "TXH1.2020"
    ),
    dplyr::mutate(
      dplyr::filter(shared_texas, .data$Year == 2020),
      Env = "TXH3.2020"
    ),
    dplyr::mutate(
      dplyr::filter(shared_texas, .data$Year == 2021),
      Env = "TXH1.2021"
    ),
    dplyr::mutate(
      dplyr::filter(shared_texas, .data$Year == 2021),
      Env = "TXH3.2021"
    )
  )

  weather <- dplyr::bind_rows(other_weather, texas_weather) |>
    dplyr::filter(.data$Env != "MNH1.2020") |>
    dplyr::mutate(Date_key = as.character(.data$Date_key))

  mnh1 <- data.table::fread(
    g2f_resolve_input(
      paths,
      "2020_g2f_waseca_raw_weather_stacked_FPCA.csv",
      result_subdirs = "02_fpca_weather/station"
    )
  ) |>
    dplyr::mutate(
      Date_key = as.character(.data$Date_key),
      Env = paste(.data[["Field Location"]], .data$Year, sep = ".")
    )

  weather <- dplyr::bind_rows(mnh1, weather) |>
    dplyr::mutate(
      Weather.Date.Time = g2f_parse_weather_datetimes(.data$Date_key)
    ) |>
    dplyr::left_join(planting_dates, by = "Env") |>
    dplyr::mutate(
      Weather.DAP = as.numeric(
        difftime(.data$Weather.Date.Time, .data$Planting.Date.Time, units = "days")
      )
    ) |>
    dplyr::filter(
      is.finite(.data$Weather.DAP),
      .data$Weather.DAP >= 0,
      .data$Weather.DAP <= analysis$weather$max_drone_dap
    ) |>
    dplyr::arrange(.data$Env, .data$Weather.DAP)

  observed <- sort(unique(weather$Env))
  if (!setequal(observed, environments)) {
    stop(
      "Station weather environments do not match config/analysis.yml. Missing: ",
      paste(setdiff(environments, observed), collapse = ", "),
      "; unexpected: ", paste(setdiff(observed, environments), collapse = ", ")
    )
  }

  weather
}

g2f_apply_dap_windows <- function(data, trait, windows) {
  for (window in windows) {
    rows <- data$Env == window$environment
    if (!is.null(window$min_dap)) {
      rows <- rows & data$Weather.DAP < as.numeric(window$min_dap)
    } else {
      rows <- rep(FALSE, nrow(data))
    }

    if (!is.null(window$max_dap)) {
      rows <- rows | (
        data$Env == window$environment &
          data$Weather.DAP > as.numeric(window$max_dap)
      )
    }
    data[[trait]][rows] <- NA_real_
  }
  data
}

g2f_apply_hampel_by_environment <- function(
    data,
    trait,
    environments,
    k,
    t0) {
  for (environment in environments) {
    rows <- which(data$Env == environment & !is.na(data[[trait]]))
    if (length(rows) < (2L * as.integer(k) + 1L)) next
    outlier_positions <- pracma::hampel(
      data[[trait]][rows],
      k = as.integer(k),
      t0 = as.numeric(t0)
    )$ind
    data[[trait]][rows[outlier_positions]] <- NA_real_
  }
  data
}

g2f_prioritize_station_observations <- function(data, trait) {
  if (!"NWS Network" %in% names(data)) data[["NWS Network"]] <- NA_character_

  data |>
    dplyr::mutate(
      .in_field = is.na(.data[["NWS Network"]]) |
        !nzchar(trimws(as.character(.data[["NWS Network"]])))
    ) |>
    dplyr::group_by(.data$Env, .data$Weather.DAP) |>
    dplyr::filter(!any(.data$.in_field) | .data$.in_field) |>
    dplyr::summarise(
      Value = if (all(is.na(.data[[trait]]))) {
        NA_real_
      } else {
        mean(.data[[trait]], na.rm = TRUE)
      },
      .groups = "drop"
    ) |>
    dplyr::rename(!!trait := "Value")
}

g2f_write_station_fpca <- function(
    fit,
    cleaned_data,
    paths,
    trait,
    filename_stem,
    tall_value_name) {
  output_dir <- g2f_results_dir(paths, "02_fpca_weather", "station")

  scores <- fit$scores |>
    dplyr::mutate(
      Year = sub("^.*\\.", "", .data$Env),
      Env.Weather.Var = paste(.data$Env, trait, sep = "."),
      Weather.Var = trait
    )

  curves_wide <- fit$predicted_curves_wide |>
    dplyr::mutate(
      Year = sub("^.*\\.", "", .data$Env),
      Env.Weather.Var = paste(.data$Env, trait, sep = "."),
      Weather.Var = trait
    ) |>
    dplyr::relocate(
      .data$Env, .data$Year, .data$Env.Weather.Var, .data$Weather.Var
    )

  curves_tall <- fit$predicted_curves_tall |>
    dplyr::rename(!!tall_value_name := .data$Value) |>
    dplyr::mutate(
      Year = sub("^.*\\.", "", .data$Env),
      Env.Weather.Var = paste(.data$Env, trait, sep = "."),
      Weather.Var = trait
    ) |>
    dplyr::relocate(
      .data$Env, .data$Year, .data$Env.Weather.Var, .data$Weather.Var
    )

  data.table::fwrite(
    cleaned_data,
    file.path(output_dir, paste0(filename_stem, "_Cleaned_Tall.csv"))
  )
  data.table::fwrite(
    scores,
    file.path(output_dir, paste0(filename_stem, "_Scores.csv"))
  )
  data.table::fwrite(
    curves_wide,
    file.path(output_dir, paste0(filename_stem, "_Predcurves_Wide.csv"))
  )
  data.table::fwrite(
    curves_tall,
    file.path(output_dir, paste0(filename_stem, "_Predcurves_Tall.csv"))
  )
  saveRDS(
    fit$model,
    file.path(output_dir, paste0(filename_stem, "_Model.rds"))
  )

  invisible(list(scores = scores, curves_wide = curves_wide, curves_tall = curves_tall))
}
