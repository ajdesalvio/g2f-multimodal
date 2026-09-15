qtl_project_paths <- function(analysis, create = FALSE) {
  analysis <- match.arg(analysis, c("dap", "agdd", "flowering_yield"))

  project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
  paths_script <- file.path(project_dir, "R", "utils", "paths.R")
  if (!file.exists(paths_script)) {
    stop(
      "Cannot locate R/utils/paths.R. Run from functional-analysis or set ",
      "G2F_PROJECT_DIR to that directory."
    )
  }
  source(paths_script, local = environment())
  base <- g2f_paths(require_data = TRUE)

  analysis_key <- toupper(analysis)
  analysis_key <- gsub("[^A-Z0-9]", "_", analysis_key)
  configured_dir <- function(kind, default) {
    specific <- Sys.getenv(paste0("G2F_QTL_", analysis_key, "_", kind, "_DIR"), unset = "")
    generic <- Sys.getenv(paste0("G2F_QTL_", kind, "_DIR"), unset = "")
    value <- if (nzchar(specific)) specific else if (nzchar(generic)) generic else default
    normalizePath(value, winslash = "/", mustWork = FALSE)
  }

  qtl_root <- file.path(base$results_dir, "qtl")
  analysis_root <- file.path(qtl_root, analysis)
  result <- c(
    base,
    list(
      analysis = analysis,
      reference_dir = configured_dir(
        "REFERENCE",
        file.path(base$data_dir, "Michel_2022_Supplementary")
      ),
      input_dir = configured_dir("INPUT", file.path(analysis_root, "input")),
      output_dir = configured_dir("OUTPUT", file.path(analysis_root, "output")),
      compiled_dir = normalizePath(
        Sys.getenv("G2F_QTL_COMPILED_DIR", unset = file.path(qtl_root, "compiled")),
        winslash = "/",
        mustWork = FALSE
      ),
      refinement_dir = normalizePath(
        Sys.getenv("G2F_QTL_REFINEMENT_DIR", unset = file.path(qtl_root, "refinement")),
        winslash = "/",
        mustWork = FALSE
      )
    )
  )

  if (create) {
    for (directory in result[c("input_dir", "output_dir", "compiled_dir", "refinement_dir")]) {
      dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    }
  }
  result
}

qtl_require_file <- function(path, description = "Required input") {
  if (!file.exists(path)) stop(description, " does not exist: ", path)
  invisible(path)
}

qtl_require_argument <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (length(args) != 1L || !nzchar(args[[1L]])) {
    stop("Expected exactly one argument in the form ENV.YEAR.TESTER.")
  }
  if (!grepl("^.+\\.[0-9]{4}\\.[^.]+$", args[[1L]])) {
    stop("Invalid environment-tester identifier: ", args[[1L]])
  }
  args[[1L]]
}

qtl_split_env_tester <- function(x) {
  matches <- regexec("^(.+\\.[0-9]{4})\\.([^.]+)$", x)
  pieces <- regmatches(x, matches)
  if (any(lengths(pieces) != 3L)) {
    stop("Could not split environment-tester identifier(s): ", paste(x[lengths(pieces) != 3L], collapse = ", "))
  }
  data.frame(
    Env = vapply(pieces, `[[`, character(1), 2L),
    Tester = vapply(pieces, `[[`, character(1), 3L),
    stringsAsFactors = FALSE
  )
}

qtl_result_rds <- function(output_dir, env_tester, must_work = TRUE) {
  candidates <- file.path(
    output_dir,
    c(
      paste0(env_tester, ".QTL.Outputs.rds"),
      paste0(env_tester, "QTL.Outputs.rds")
    )
  )
  existing <- candidates[file.exists(candidates)]
  if (length(existing)) return(existing[[1L]])
  if (must_work) stop("QTL result file not found. Tried: ", paste(candidates, collapse = "; "))
  candidates[[1L]]
}

qtl_env_tester_from_rds <- function(path) {
  sub("\\.?QTL\\.Outputs\\.rds$", "", basename(path))
}
