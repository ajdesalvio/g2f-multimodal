suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
})

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "05_prediction_cv", "path_helpers.R"))
cv_paths <- g2f_cv_paths("agdd")

get_path <- function(env, default) {
  val <- Sys.getenv(env, unset = "")
  if (nzchar(val)) normalizePath(val, winslash = "/", mustWork = FALSE) else default
}

safe_mean <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else mean(x)
}

collapse_agdd_weather_duplicates <- function(df_clim) {
  dup_summary <- df_clim %>%
    count(Env, AGDD, name = "N") %>%
    filter(N > 1L) %>%
    arrange(Env, AGDD)

  if (nrow(dup_summary) == 0L) {
    return(list(clean = df_clim, duplicates = dup_summary))
  }

  dup_keys <- dup_summary %>%
    mutate(Key = paste(Env, AGDD, sep = "||")) %>%
    pull(Key)

  df_clim <- df_clim %>%
    mutate(Key = paste(Env, AGDD, sep = "||"))

  df_keep <- df_clim %>%
    filter(!Key %in% dup_keys) %>%
    select(-Key)

  df_dup <- df_clim %>%
    filter(Key %in% dup_keys) %>%
    select(-Key)

  meta_cols <- c("Env", "AGDD", "DAP", "YYYYMMDD", "LON", "LAT", "DOY", "daysFromStart")
  avg_cols <- setdiff(names(df_dup), meta_cols)
  avg_cols <- avg_cols[vapply(df_dup[avg_cols], is.numeric, logical(1))]

  df_dup_avg <- df_dup %>%
    group_by(Env, AGDD) %>%
    summarize(
      across(all_of(avg_cols), safe_mean),
      DAP = first(DAP),
      YYYYMMDD = first(YYYYMMDD),
      LON = first(LON),
      LAT = first(LAT),
      DOY = first(DOY),
      daysFromStart = first(daysFromStart),
      .groups = "drop"
    )

  clean <- bind_rows(df_keep, df_dup_avg) %>%
    arrange(Env, AGDD)

  list(clean = clean, duplicates = dup_summary)
}

make_agdd_logger <- function(log_file = NULL, emit = TRUE) {
  function(...) {
    msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste(..., collapse = ""))
    if (isTRUE(emit)) {
      message(msg)
    }
    if (!is.null(log_file)) {
      cat(msg, "\n", file = log_file, append = TRUE)
    }
  }
}

run_agdd_preprocess <- function(
    data_path = get_path("G2F_DATA_PATH", cv_paths$data_path),
    out_root = get_path("G2F_CV_OUT_PATH", cv_paths$out_root),
    derived_path = get_path("G2F_DERIVED_DATA_PATH", cv_paths$derived_path),
    loeo_score_path = get_path("G2F_LOEO_SCORE_PATH", cv_paths$loeo_score_path),
    climate_file = Sys.getenv("G2F_CLIMATE_FILE", unset = "EnvRtype_Weather_Data_Cleaned_V2.csv"),
    vi_source_file = Sys.getenv("G2F_VI_SOURCE_FILE", unset = "VI_BLUEs_G2F_2020_2021.csv"),
    vi_output_file = Sys.getenv("G2F_VI_FILE", unset = "VI_BLUEs_G2F_2020_2021_AGDD.csv"),
    weather_all_file = Sys.getenv("G2F_WEATHER_ALL_FILE", unset = "Weather_FPCA_AGDD_EnvRtype_Data_Scores_PTR_ONLY_V1.csv"),
    weather_loeo_file = Sys.getenv("G2F_WEATHER_LOEO_FILE", unset = "Weather_FPC_Scores_AGDD_LOEO_Projected.csv"),
    write_outputs = TRUE,
    return_objects = FALSE,
    log_to_console = TRUE) {

  metadata_dir <- file.path(out_root, "metadata")
  logs_dir <- file.path(out_root, "logs")
  if (isTRUE(write_outputs)) {
    dir.create(metadata_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(derived_path, recursive = TRUE, showWarnings = FALSE)
  }

  log_file <- if (isTRUE(write_outputs)) file.path(logs_dir, "AGDD_CV_Preprocess_V1.log") else NULL
  log_message <- make_agdd_logger(log_file = log_file, emit = log_to_console)

  climate_path <- file.path(data_path, climate_file)
  vi_source_path <- file.path(data_path, vi_source_file)
  vi_output_path <- file.path(derived_path, vi_output_file)
  weather_all_path <- file.path(data_path, weather_all_file)
  weather_loeo_path <- file.path(loeo_score_path, weather_loeo_file)
  if (!file.exists(weather_loeo_path)) {
    weather_loeo_path <- file.path(data_path, weather_loeo_file)
  }

  log_message("Starting AGDD preprocessing")
  log_message("data_path = ", data_path)
  log_message("derived_path = ", derived_path)
  log_message("loeo_score_path = ", loeo_score_path)
  log_message("climate_file = ", climate_file)
  log_message("vi_source_file = ", vi_source_file)
  log_message("vi_output_file = ", vi_output_file)

  if (!file.exists(climate_path)) {
    stop("Climate file not found: ", climate_path)
  }
  if (!file.exists(vi_source_path)) {
    stop("VI source file not found: ", vi_source_path)
  }
  if (!file.exists(weather_all_path)) {
    stop("AGDD weather ALL score file not found: ", weather_all_path)
  }
  if (!file.exists(weather_loeo_path)) {
    stop("AGDD weather LOEO score file not found: ", weather_loeo_path)
  }

  df_clim <- fread(climate_path) %>%
    as.data.frame() %>%
    arrange(Env, DAP) %>%
    group_by(Env) %>%
    mutate(AGDD = cumsum(GDD)) %>%
    ungroup()

  weather_dup <- collapse_agdd_weather_duplicates(df_clim)
  df_clim_clean <- weather_dup$clean
  weather_dup_summary <- weather_dup$duplicates

  dap_agdd_map <- df_clim %>%
    select(Env, DAP, AGDD, YYYYMMDD) %>%
    distinct() %>%
    arrange(Env, DAP)

  if (anyDuplicated(dap_agdd_map[, c("Env", "DAP")]) > 0L) {
    stop("The DAP-to-AGDD map contains duplicated (Env, DAP) keys.")
  }

  vi_raw <- fread(vi_source_path) %>%
    as.data.frame() %>%
    rename(DAP_Original = DAP)

  vi_agdd_raw <- vi_raw %>%
    left_join(dap_agdd_map, by = c("Env", "DAP_Original" = "DAP"))

  missing_map <- vi_agdd_raw %>%
    filter(is.na(AGDD)) %>%
    distinct(Env, DAP_Original) %>%
    arrange(Env, DAP_Original)

  if (nrow(missing_map) > 0L) {
    stop(
      "Some VI rows could not be mapped from DAP to AGDD. Examples: ",
      paste(utils::head(paste(missing_map$Env, missing_map$DAP_Original, sep = "@"), 10L), collapse = ", ")
    )
  }

  vi_dup_summary <- vi_agdd_raw %>%
    count(Pedigree, Year, Env, Vegetation.Index, AGDD, name = "N") %>%
    filter(N > 1L) %>%
    arrange(Env, Vegetation.Index, Pedigree, AGDD)

  vi_agdd <- vi_agdd_raw %>%
    group_by(Pedigree, Year, Env, Vegetation.Index, AGDD) %>%
    summarize(
      VI.BLUE = safe_mean(VI.BLUE),
      DAP.Original = paste(sort(unique(DAP_Original)), collapse = ";"),
      N.Original.Rows = dplyr::n(),
      .groups = "drop"
    ) %>%
    rename(DAP = AGDD) %>%
    mutate(Pedigree.Env = paste(Pedigree, Env, sep = ".")) %>%
    arrange(Env, Vegetation.Index, Pedigree, DAP)

  summary_df <- data.frame(
    Climate_File = climate_file,
    VI_Source_File = vi_source_file,
    VI_Output_File = vi_output_file,
    Weather_ALL_File = weather_all_file,
    Weather_LOEO_File = weather_loeo_file,
    Climate_Rows = nrow(df_clim),
    Weather_Duplicate_AGDD_Rows = nrow(weather_dup_summary),
    VI_Source_Rows = nrow(vi_raw),
    VI_AGDD_Rows = nrow(vi_agdd),
    VI_Duplicate_AGDD_Rows = nrow(vi_dup_summary),
    stringsAsFactors = FALSE
  )

  if (isTRUE(write_outputs)) {
    fwrite(vi_agdd, vi_output_path)
    fwrite(dap_agdd_map, file.path(metadata_dir, "DAP_AGDD_Map.csv"))
    fwrite(weather_dup_summary, file.path(metadata_dir, "AGDD_Weather_Duplicates.csv"))
    fwrite(vi_dup_summary, file.path(metadata_dir, "AGDD_VI_Duplicates.csv"))
    fwrite(summary_df, file.path(metadata_dir, "AGDD_Preprocess_Summary.csv"))
    saveRDS(
      list(
        summary_df = summary_df,
        dap_agdd_map = dap_agdd_map,
        weather_dup_summary = weather_dup_summary,
        vi_dup_summary = vi_dup_summary
      ),
      file.path(metadata_dir, "AGDD_Preprocess_Summary.rds")
    )
  }

  log_message("Completed AGDD preprocessing")
  log_message("Weather duplicate AGDD rows = ", nrow(weather_dup_summary))
  log_message("VI duplicate AGDD rows = ", nrow(vi_dup_summary))
  log_message("AGDD VI file written to ", vi_output_path)

  if (isTRUE(return_objects)) {
    return(list(
      summary_df = summary_df,
      climate_raw = df_clim,
      climate_clean = df_clim_clean,
      weather_dup_summary = weather_dup_summary,
      dap_agdd_map = dap_agdd_map,
      vi_raw = vi_raw,
      vi_agdd = vi_agdd,
      vi_dup_summary = vi_dup_summary,
      vi_output_file = vi_output_path
    ))
  }

  invisible(summary_df)
}

if (sys.nframe() == 0) {
  run_agdd_preprocess()
}
