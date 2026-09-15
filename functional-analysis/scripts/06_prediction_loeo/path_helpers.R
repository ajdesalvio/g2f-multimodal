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

g2f_loeo_paths <- function(time_domain) {
  time_domain <- tolower(time_domain)
  if (!time_domain %in% c("dap", "agdd")) {
    stop("time_domain must be either 'dap' or 'agdd'.")
  }

  configured_data_path <- Sys.getenv("G2F_DATA_PATH", unset = "")
  paths <- g2f_paths(require_data = !nzchar(configured_data_path))
  domain_root <- file.path(paths$results_dir, "prediction", "loeo", time_domain)
  data_path <- if (nzchar(configured_data_path)) configured_data_path else paths$data_dir
  score_path <- Sys.getenv(
    "G2F_LOEO_SCORE_PATH",
    unset = file.path(domain_root, "projected_scores")
  )
  kernel_path <- Sys.getenv(
    "G2F_KERNEL_OUT_PATH",
    unset = file.path(domain_root, "kernels")
  )
  results_path <- Sys.getenv(
    "G2F_RESULTS_PATH",
    unset = file.path(domain_root, "predictions")
  )
  work_path <- Sys.getenv(
    "G2F_WORK_PATH",
    unset = file.path(domain_root, "work")
  )

  include_ze_env <- Sys.getenv("G2F_INCLUDE_ZE", unset = "")

  list(
    data_path = normalizePath(data_path, winslash = "/", mustWork = FALSE),
    score_path = normalizePath(score_path, winslash = "/", mustWork = FALSE),
    kernel_path = normalizePath(kernel_path, winslash = "/", mustWork = FALSE),
    matrix_path = normalizePath(file.path(domain_root, "relationship_matrices"), winslash = "/", mustWork = FALSE),
    results_path = normalizePath(results_path, winslash = "/", mustWork = FALSE),
    work_path = normalizePath(work_path, winslash = "/", mustWork = FALSE),
    include_ze = if (nzchar(include_ze_env)) g2f_bool(include_ze_env) else paths$include_ze,
    vi_names = Sys.getenv("G2F_VI_NAMES", unset = paste(paths$vi_names, collapse = ",")),
    weather_traits = Sys.getenv(
      "G2F_WEATHER_TRAITS",
      unset = paste(paths$weather_traits, collapse = ",")
    )
  )
}
