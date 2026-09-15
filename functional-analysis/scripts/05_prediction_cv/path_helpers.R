g2f_prediction_root <- function() {
  configured <- Sys.getenv("G2F_PROJECT_DIR", unset = "")
  candidate <- if (nzchar(configured)) configured else getwd()
  candidate <- normalizePath(candidate, winslash = "/", mustWork = TRUE)

  marker <- file.path(candidate, "R", "utils", "paths.R")
  if (!file.exists(marker)) {
    stop(
      "Could not locate functional-analysis. Run from that directory ",
      "or set G2F_PROJECT_DIR. Expected: ", marker
    )
  }

  candidate
}

g2f_prediction_root_dir <- g2f_prediction_root()
source(file.path(g2f_prediction_root_dir, "R", "utils", "paths.R"))

g2f_cv_paths <- function(time_domain) {
  time_domain <- tolower(time_domain)
  if (!time_domain %in% c("dap", "agdd")) {
    stop("time_domain must be either 'dap' or 'agdd'.")
  }

  configured_data_path <- Sys.getenv("G2F_DATA_PATH", unset = "")
  paths <- g2f_paths(require_data = !nzchar(configured_data_path))
  data_path <- if (nzchar(configured_data_path)) configured_data_path else paths$data_dir
  out_root <- Sys.getenv(
    "G2F_CV_OUT_PATH",
    unset = file.path(paths$results_dir, "prediction", "cv", time_domain)
  )

  include_ze_env <- Sys.getenv("G2F_INCLUDE_ZE", unset = "")
  common_female_candidates <- c(
    file.path(data_path, "Common_Females.csv"),
    file.path(paths$project_dir, "data", "supplementary", "Common_Females.csv")
  )
  common_female_existing <- common_female_candidates[file.exists(common_female_candidates)]

  list(
    data_path = normalizePath(data_path, winslash = "/", mustWork = FALSE),
    common_females_file = if (length(common_female_existing)) common_female_existing[[1L]] else common_female_candidates[[1L]],
    out_root = normalizePath(out_root, winslash = "/", mustWork = FALSE),
    derived_path = normalizePath(file.path(out_root, "derived_data"), winslash = "/", mustWork = FALSE),
    loeo_score_path = normalizePath(
      file.path(paths$results_dir, "prediction", "loeo", time_domain, "projected_scores"),
      winslash = "/",
      mustWork = FALSE
    ),
    include_ze = if (nzchar(include_ze_env)) g2f_bool(include_ze_env) else paths$include_ze,
    vi_names = Sys.getenv("G2F_VI_NAMES", unset = paste(paths$vi_names, collapse = ",")),
    weather_traits = Sys.getenv(
      "G2F_WEATHER_TRAITS",
      unset = paste(paths$weather_traits, collapse = ",")
    )
  )
}
