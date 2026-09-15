`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) return(y)
  if (is.character(x) && length(x) == 1L && !nzchar(x)) return(y)
  x
}

g2f_bool <- function(value, default = FALSE) {
  if (is.null(value) || length(value) == 0L || !nzchar(as.character(value[[1L]]))) {
    return(default)
  }

  normalized <- toupper(trimws(as.character(value[[1L]])))
  if (normalized %in% c("TRUE", "T", "1", "YES", "Y")) return(TRUE)
  if (normalized %in% c("FALSE", "F", "0", "NO", "N")) return(FALSE)
  stop("Cannot interpret logical value: ", value[[1L]])
}

g2f_find_project_dir <- function(start = getwd()) {
  current <- normalizePath(start, winslash = "/", mustWork = TRUE)

  repeat {
    marker <- file.path(current, "config", "paths.example.yml")
    if (file.exists(marker)) return(current)

    parent <- dirname(current)
    if (identical(parent, current)) break
    current <- parent
  }

  stop(
    "Could not locate the functional-analysis root from ", start, ". Run from within ",
    "functional-analysis or set G2F_PROJECT_DIR to that directory."
  )
}

g2f_project_dir <- function() {
  configured <- Sys.getenv("G2F_PROJECT_DIR", unset = "")
  if (nzchar(configured)) {
    root <- normalizePath(configured, winslash = "/", mustWork = TRUE)
    marker <- file.path(root, "config", "paths.example.yml")
    if (!file.exists(marker)) stop("G2F_PROJECT_DIR is not the functional-analysis root: ", root)
    return(root)
  }

  g2f_find_project_dir()
}

g2f_paths <- function(config_file = NULL, require_data = TRUE) {
  root <- g2f_project_dir()
  config_file <- config_file %||% Sys.getenv(
    "G2F_CONFIG_FILE",
    unset = file.path(root, "config", "paths.yml")
  )

  config <- list()
  if (file.exists(config_file)) {
    if (!requireNamespace("yaml", quietly = TRUE)) {
      stop("Package 'yaml' is required to read ", config_file)
    }
    config <- yaml::read_yaml(config_file) %||% list()
  }

  project_config <- config$project %||% list()
  prediction_config <- config$prediction %||% list()

  data_dir <- Sys.getenv(
    "G2F_DATA_DIR",
    unset = project_config$data_dir %||% file.path(root, "data")
  )
  results_dir <- Sys.getenv(
    "G2F_RESULTS_DIR",
    unset = project_config$results_dir %||% file.path(root, "results")
  )

  data_dir <- normalizePath(data_dir, winslash = "/", mustWork = require_data)
  results_dir <- normalizePath(results_dir, winslash = "/", mustWork = FALSE)

  include_ze_env <- Sys.getenv("G2F_INCLUDE_ZE", unset = "")
  include_ze <- if (nzchar(include_ze_env)) {
    g2f_bool(include_ze_env)
  } else {
    g2f_bool(prediction_config$include_ze, default = FALSE)
  }

  list(
    project_dir = root,
    data_dir = data_dir,
    results_dir = results_dir,
    include_ze = include_ze,
    vi_names = prediction_config$vi_names %||% "NGRDI",
    weather_traits = prediction_config$weather_traits %||% "PTR"
  )
}

g2f_analysis_config <- function(path = NULL) {
  root <- g2f_project_dir()
  path <- path %||% file.path(root, "config", "analysis.yml")
  g2f_require_file(path, "Analysis configuration")
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("Package 'yaml' is required to read ", path)
  }
  yaml::read_yaml(path)
}

g2f_data_file <- function(paths, ...) {
  file.path(paths$data_dir, ...)
}

g2f_results_file <- function(paths, ...) {
  file.path(paths$results_dir, ...)
}

g2f_results_dir <- function(paths, ..., create = TRUE) {
  directory <- file.path(paths$results_dir, ...)
  if (create) dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  normalizePath(directory, winslash = "/", mustWork = create)
}

g2f_resolve_input <- function(paths, filename, result_subdirs = character()) {
  candidates <- c(
    unlist(lapply(result_subdirs, function(x) file.path(paths$results_dir, x, filename))),
    file.path(paths$data_dir, filename),
    file.path(paths$project_dir, "data", "supplementary", filename)
  )
  existing <- candidates[file.exists(candidates)]

  if (length(existing) == 0L) {
    stop(
      "Required input was not found: ", filename, "\nSearched:\n- ",
      paste(candidates, collapse = "\n- ")
    )
  }

  normalizePath(existing[[1L]], winslash = "/", mustWork = TRUE)
}

g2f_require_file <- function(path, description = "Required input") {
  if (!file.exists(path)) {
    stop(description, " does not exist: ", path)
  }
  invisible(path)
}
