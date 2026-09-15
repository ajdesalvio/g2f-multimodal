g2f_prediction_result_paths <- function(paths, analysis) {
  config <- analysis$prediction_results
  if (is.null(config)) stop("config/analysis.yml has no prediction_results section.")

  project_path <- function(...) file.path(paths$project_dir, ...)
  result_path <- function(relative_path) file.path(paths$results_dir, relative_path)
  prefer_existing <- function(candidates) {
    existing <- candidates[file.exists(candidates) | dir.exists(candidates)]
    if (length(existing)) existing[[1L]] else candidates[[1L]]
  }
  artifact_path <- function(...) project_path("results", ...)
  workbook_path <- function(filename) {
    prefer_existing(c(
      artifact_path("prediction", "workbooks", filename),
      file.path(paths$data_dir, filename)
    ))
  }
  result_or_fallback <- function(primary, fallback) {
    prefer_existing(c(primary, result_path(fallback)))
  }

  list(
    cv_xlsx = workbook_path(config$workbooks$cv),
    cv_csv = workbook_path(sub("[.]xlsx$", ".csv", config$workbooks$cv)),
    loeo_xlsx = workbook_path(config$workbooks$loeo),
    loeo_csv = workbook_path(sub("[.]xlsx$", ".csv", config$workbooks$loeo)),
    cv0_dap = result_or_fallback(
      result_path("prediction/cv/dap/results"),
      config$cv$dap_metrics
    ),
    cv0_agdd = result_or_fallback(
      result_path("prediction/cv/agdd/results"),
      config$cv$agdd_metrics
    ),
    cv21_dap = result_or_fallback(
      result_path("prediction/cv/dap/results/prediction_values"),
      config$cv$dap_prediction_values
    ),
    cv21_agdd = result_or_fallback(
      result_path("prediction/cv/agdd/results/prediction_values"),
      config$cv$agdd_prediction_values
    ),
    loeo_dap = result_or_fallback(
      result_path("prediction/loeo/dap/predictions"),
      config$loeo$dap
    ),
    loeo_agdd = result_or_fallback(
      result_path("prediction/loeo/agdd/predictions"),
      config$loeo$agdd
    ),
    tnp = prefer_existing(c(
      artifact_path("prediction", "tnp", "tidy_all_metrics.csv"),
      result_path(config$tnp_metrics),
      file.path(paths$data_dir, basename(config$tnp_metrics))
    )),
    primary_tnp_checkpoint = config$primary_tnp_checkpoint
  )
}

g2f_model_labels <- function() {
  c(
    "M1.G" = "M1.G",
    "M1.P" = "M1.P",
    "M2.G" = "M2.G",
    "M3.G" = "M3.G-Int",
    "M3.P" = "M3.P-Int",
    "M4.G" = "M4.G.W",
    "M4.P" = "M4.P.W",
    "M5.G" = "M5.G.W-Int",
    "M5.P" = "M5.P.W-Int",
    "M6.G.P" = "M6.G.P",
    "M7.G.P" = "M7.G.P",
    "M8.G.P" = "M8.G.P-Int",
    "M9.G.P" = "M9.G.P.W",
    "M10.G.P" = "M10.G.P.W-Int"
  )
}

g2f_read_csv_directory <- function(
  directory,
  pattern,
  expected_minimum = 1L,
  expected_count = NULL
) {
  if (!dir.exists(directory)) stop("Prediction result directory is missing: ", directory)
  files <- list.files(directory, pattern = pattern, full.names = TRUE)
  if (!is.null(expected_count) && length(files) != expected_count) {
    stop(
      "Expected exactly ", expected_count, " files matching ", pattern,
      " in ", directory, "; found ", length(files), "."
    )
  }
  if (length(files) < expected_minimum) {
    stop(
      "Expected at least ", expected_minimum, " files matching ", pattern,
      " in ", directory, "; found ", length(files), "."
    )
  }

  data.table::rbindlist(lapply(files, function(file) {
    data <- data.table::fread(file)
    data$File.Name <- basename(file)
    data
  }), fill = TRUE)
}

g2f_assert_no_ze <- function(data) {
  if ("Include_Ze" %in% names(data)) {
    values <- unique(data$Include_Ze[!is.na(data$Include_Ze)])
    if (length(values) > 0L && any(as.logical(values))) {
      stop("Prediction inputs contain Include_Ze = TRUE results.")
    }
  }
  invisible(data)
}
