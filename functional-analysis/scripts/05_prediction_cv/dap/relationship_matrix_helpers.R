suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
})

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "05_prediction_cv", "path_helpers.R"))
cv_paths <- g2f_cv_paths("dap")

################################################################################
# DAP CV relationship-matrix setup for models M8.G.P and M10.G.P
################################################################################
#
# Purpose
# -------
# This script builds the raw relationship matrices used by
# models M8.G.P and M10.G.P under the leakage-safe CV2/CV1/CV0/CV00 split
# system.
#
# This script's job is to:
#
#   1. Run the existing DAP CV setup code for one split.
#   2. Reconstruct the raw relationship matrices used by M8.G.P and M10.G.P.
#   3. Save an explainable RDS bundle plus small description CSV files.
#
# Model components
# ----------------
# M8.G.P uses:
#   KG_G_A, KG_G_D, KG_GE_A, KG_GE_D, KP, KP_PE
#
# M10.G.P uses:
#   KG_G_A, KG_G_D, KP, KE_W, KG_GE_AW, KG_GE_DW, KP_PW
#
# The original BGLR prediction script also included Ze as a fixed environment
# incidence term. Ze is saved in the output bundle under design_matrices, but it
# is not a relationship matrix and is therefore not included in the M8/M10
# relationship-matrix lists.
#
# Important leakage-control assumption
# ------------------------------------
# The underlying build_fold_kernels.R script controls the leakage-safe
# FPCA logic:
#
#   CV_2_1:
#     Fit VI FPCA on the 4 training folds across all environments, then project
#     the fifth fold and environment-specific hybrids. Weather uses the
#     all-environment weather FPC scores.
#
#   CV_0_00:
#     Fit VI FPCA on the 4 training folds while excluding the held-out
#     environment, then project excluded curves. Weather uses the LOEO weather
#     FPC scores for the held-out environment.
#
# Example usage from RStudio
# --------------------------
# source("scripts/05_prediction_cv/dap/relationship_matrix_helpers.R")
#
# bundle <- build_dap_cv_m8_m10_matrices(
#   seed_num = 1,
#   fold_num = 1,
#   split_group = "CV_2_1",
#   heldout_env = "None",
#   data_path = cv_paths$data_path,
#   out_root = cv_paths$out_root,
#   vi_names = "NGRDI",
#   n_vi_fpcs = 5,
#   write_outputs = TRUE,
#   return_objects = TRUE
# )
#
# m8_matrices <- bundle$relationship_matrices[bundle$model_components[["M8.G.P"]]]
# m10_matrices <- bundle$relationship_matrices[bundle$model_components[["M10.G.P"]]]
#
# Example command-line usage
# --------------------------
# Rscript scripts/05_prediction_cv/dap/relationship_matrix_helpers.R 1 1 CV_2_1 None
# Rscript scripts/05_prediction_cv/dap/relationship_matrix_helpers.R 1 1 CV_0_00 MIH1.2020
#
# Optional environment variables
# ------------------------------
#   G2F_DATA_PATH:
#     Folder containing the input data files.
#
#   G2F_CV_OUT_PATH:
#     Pipeline output folder containing or receiving metadata, constant bundles,
#     weather bundles, and collaborator matrix bundles.
#
#   G2F_PIPELINE_DIR:
#     Folder containing the companion DAP CV scripts. Defaults to this script's
#     folder when possible.
#
#   G2F_VI_NAMES:
#     Comma-separated vegetation indices. Example: NGRDI. Blank means all
#     non-"new" VIs, matching the original setup script behavior.
#
#   G2F_VI_NFPCS:
#     Number of VI FPCs retained per selected VI. Default: 5.
#
#   G2F_WEATHER_TRAITS, G2F_WEATHER_NFPCS, G2F_WEATHER_ALL_FILE,
#   G2F_WEATHER_LOEO_FILE:
#     Passed through to the weather setup stage if prerequisite weather bundles
#     need to be created.
#
#   G2F_PREPARE_IF_MISSING:
#     TRUE/FALSE. If TRUE, missing metadata/constant/weather setup artifacts are
#     created automatically. Default: TRUE.
#
#   G2F_SAVE_MATRIX_CSV:
#     TRUE/FALSE. If TRUE, writes every relationship matrix as CSV. Default:
#     FALSE because the matrices are large.
#
#   G2F_SAVE_MODEL_MATRIX_RDS:
#     TRUE/FALSE. If TRUE, writes separate M8 and M10 matrix-list RDS files.
#     Default: FALSE to avoid duplicating large matrices on disk.
#
#   G2F_COMPUTE_EIGS:
#     TRUE/FALSE. If FALSE, the V2 helper scripts skip eigendecomposition and
#     build only the raw relationship matrices needed by this handoff script.
#     Default: FALSE.
#
#   G2F_VERBOSE:
#     TRUE/FALSE. If TRUE, prints detailed progress messages. Default: TRUE.
#
################################################################################

get_path <- function(env, default) {
  val <- Sys.getenv(env, unset = "")
  if (nzchar(val)) normalizePath(val, winslash = "/", mustWork = FALSE) else default
}

sanitize_arg <- function(x) {
  trimws(gsub("[\r\n]+", "", x, perl = TRUE))
}

sanitize_file_component <- function(x) {
  gsub("[^A-Za-z0-9._-]", "_", x)
}

parse_csv_env <- function(env, default) {
  val <- Sys.getenv(env, unset = default)
  out <- trimws(strsplit(val, ",", fixed = TRUE)[[1]])
  out[nzchar(out)]
}

is_truthy_env <- function(env, default = "FALSE") {
  val <- toupper(Sys.getenv(env, unset = default))
  val %in% c("TRUE", "1", "YES", "Y")
}

log_progress <- function(verbose, ...) {
  if (isTRUE(verbose)) {
    message(
      format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      " | ",
      paste(..., collapse = "")
    )
  }
}

summarize_dims <- function(x) {
  paste0(nrow(x), " x ", ncol(x))
}

detect_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0L) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = FALSE)))
  }

  for (frame_i in rev(sys.frames())) {
    if (!is.null(frame_i$ofile)) {
      return(dirname(normalizePath(frame_i$ofile, winslash = "/", mustWork = FALSE)))
    }
  }

  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

source_pipeline_script <- function(pipeline_env, pipeline_dir, file_name, verbose) {
  file_path <- file.path(pipeline_dir, file_name)
  if (!file.exists(file_path)) {
    stop("Could not find required pipeline script: ", file_path)
  }
  log_progress(verbose, "Sourcing ", file_name)
  sys.source(file_path, envir = pipeline_env)
}

make_matrix_manifest <- function(matrices, model_membership) {
  do.call(rbind, lapply(names(matrices), function(name_i) {
    mat <- matrices[[name_i]]
    data.frame(
      Matrix = name_i,
      Rows = nrow(mat),
      Columns = ncol(mat),
      Row_Role = "Pedigree.Env observations",
      Col_Role = "Pedigree.Env observations",
      In_M8_G_P = name_i %in% model_membership[["M8.G.P"]],
      In_M10_G_P = name_i %in% model_membership[["M10.G.P"]],
      stringsAsFactors = FALSE
    )
  }))
}

write_matrix_csvs <- function(matrices, out_dir) {
  csv_dir <- file.path(out_dir, "matrix_csv")
  dir.create(csv_dir, recursive = TRUE, showWarnings = FALSE)

  invisible(lapply(names(matrices), function(name_i) {
    mat <- matrices[[name_i]]
    out_df <- data.frame(ID = rownames(mat), mat, check.names = FALSE)
    fwrite(out_df, file.path(csv_dir, paste0(name_i, ".csv")))
  }))
}

make_split_id <- function(seed_num, fold_num, split_group, heldout_env) {
  paste(
    sprintf("Seed%02d", as.integer(seed_num)),
    sprintf("Fold%d", as.integer(fold_num)),
    split_group,
    sanitize_file_component(heldout_env),
    sep = "."
  )
}

ensure_cv_prerequisites <- function(
    pipeline_env,
    pipeline_dir,
    data_path,
    out_root,
    split_group,
    heldout_env,
    compute_eigs,
    prepare_if_missing,
    verbose) {

  bundles_dir <- file.path(out_root, "bundles")
  metadata_dir <- file.path(out_root, "metadata")

  metadata_file <- file.path(metadata_dir, "female_folds.csv")
  constant_file <- file.path(bundles_dir, "constant_kernel_bundle.rds")
  weather_all_file <- file.path(bundles_dir, "weather", "weather_all_bundle.rds")
  weather_loeo_file <- file.path(
    bundles_dir,
    "weather",
    "loeo",
    paste0("weather_", sanitize_file_component(heldout_env), ".rds")
  )

  need_metadata <- !file.exists(metadata_file)
  need_constant <- !file.exists(constant_file)
  need_weather <- if (split_group == "CV_2_1") {
    !file.exists(weather_all_file)
  } else {
    !file.exists(weather_loeo_file)
  }

  if (!need_metadata && !need_constant && !need_weather) {
    log_progress(verbose, "Required metadata, constant bundle, and weather bundle already exist")
    return(invisible(TRUE))
  }

  if (!isTRUE(prepare_if_missing)) {
    missing <- c(
      if (need_metadata) metadata_file else character(0),
      if (need_constant) constant_file else character(0),
      if (need_weather && split_group == "CV_2_1") weather_all_file else character(0),
      if (need_weather && split_group == "CV_0_00") weather_loeo_file else character(0)
    )
    stop("Missing required setup artifact(s): ", paste(missing, collapse = "; "))
  }

  if (need_metadata) {
    source_pipeline_script(pipeline_env, pipeline_dir, "build_metadata.R", verbose)
    log_progress(verbose, "Creating CV metadata/fold files")
    pipeline_env$run_cv_metadata(
      data_path = data_path,
      out_root = out_root,
      write_outputs = TRUE,
      return_objects = FALSE,
      log_to_console = verbose
    )
  }

  if (need_constant) {
    source_pipeline_script(pipeline_env, pipeline_dir, "build_constant_kernels.R", verbose)
    log_progress(verbose, "Creating constant genomic/incidence kernel bundle")
    pipeline_env$run_cv_constant_kernels(
      data_path = data_path,
      out_root = out_root,
      write_outputs = TRUE,
      return_objects = FALSE,
      log_to_console = verbose,
      compute_eigs = compute_eigs
    )
  }

  if (need_weather) {
    source_pipeline_script(pipeline_env, pipeline_dir, "build_weather_inputs.R", verbose)
    log_progress(verbose, "Creating weather kernel bundle(s)")
    pipeline_env$run_cv_weather_setup(
      data_path = data_path,
      out_root = out_root,
      write_outputs = TRUE,
      return_objects = FALSE,
      log_to_console = verbose,
      compute_eigs = compute_eigs
    )
  }

  invisible(TRUE)
}

build_dap_cv_m8_m10_matrices <- function(
    seed_num,
    fold_num,
    split_group,
    heldout_env,
    data_path = get_path("G2F_DATA_PATH", cv_paths$data_path),
    out_root = get_path("G2F_CV_OUT_PATH", cv_paths$out_root),
    pipeline_dir = get_path("G2F_PIPELINE_DIR", detect_script_dir()),
    vi_names = parse_csv_env("G2F_VI_NAMES", "NGRDI"),
    n_vi_fpcs = suppressWarnings(as.integer(Sys.getenv("G2F_VI_NFPCS", unset = "5"))),
    prepare_if_missing = is_truthy_env("G2F_PREPARE_IF_MISSING", "TRUE"),
    write_outputs = TRUE,
    return_objects = FALSE,
    save_matrix_csv = is_truthy_env("G2F_SAVE_MATRIX_CSV", "FALSE"),
    save_model_matrix_rds = is_truthy_env("G2F_SAVE_MODEL_MATRIX_RDS", "FALSE"),
    compute_eigs = is_truthy_env("G2F_COMPUTE_EIGS", "FALSE"),
    verbose = is_truthy_env("G2F_VERBOSE", "TRUE")) {

  seed_num <- as.integer(seed_num)
  fold_num <- as.integer(fold_num)
  split_group <- sanitize_arg(split_group)
  heldout_env <- sanitize_arg(heldout_env)
  vi_names <- trimws(vi_names)
  vi_names <- vi_names[nzchar(vi_names)]

  if (is.na(seed_num) || is.na(fold_num)) {
    stop("seed_num and fold_num must be integers.")
  }
  if (!split_group %in% c("CV_2_1", "CV_0_00")) {
    stop("split_group must be one of: CV_2_1, CV_0_00")
  }
  if (split_group == "CV_2_1" && !identical(heldout_env, "None")) {
    stop("Use heldout_env = 'None' for CV_2_1 splits.")
  }
  if (is.na(n_vi_fpcs) || n_vi_fpcs < 1L) {
    stop("G2F_VI_NFPCS must be a positive integer.")
  }

  pipeline_dir <- normalizePath(pipeline_dir, winslash = "/", mustWork = FALSE)
  split_id <- make_split_id(seed_num, fold_num, split_group, heldout_env)

  log_progress(verbose, "Starting DAP CV M8/M10 relationship-matrix setup")
  log_progress(verbose, "split_id = ", split_id)
  log_progress(verbose, "data_path = ", data_path)
  log_progress(verbose, "out_root = ", out_root)
  log_progress(verbose, "pipeline_dir = ", pipeline_dir)
  log_progress(verbose, "VI selection = ", if (length(vi_names) > 0L) paste(vi_names, collapse = ",") else "<all VIs>")
  log_progress(verbose, "VI FPCs retained per selected VI = ", n_vi_fpcs)
  log_progress(verbose, "compute_eigs = ", compute_eigs)

  pipeline_env <- new.env(parent = globalenv())

  ensure_cv_prerequisites(
    pipeline_env = pipeline_env,
    pipeline_dir = pipeline_dir,
    data_path = data_path,
    out_root = out_root,
    split_group = split_group,
    heldout_env = heldout_env,
    compute_eigs = compute_eigs,
    prepare_if_missing = prepare_if_missing,
    verbose = verbose
  )

  source_pipeline_script(pipeline_env, pipeline_dir, "build_fold_kernels.R", verbose)

  log_progress(verbose, "Running leakage-safe CV split setup")
  setup <- pipeline_env$run_cv_prediction_setup(
    seed_num = seed_num,
    fold_num = fold_num,
    split_group = split_group,
    heldout_env = heldout_env,
    data_path = data_path,
    out_root = out_root,
    vi_names = vi_names,
    n_vi_fpcs = n_vi_fpcs,
    save_vi_scores = FALSE,
    write_outputs = FALSE,
    return_objects = TRUE,
    log_to_console = verbose,
    compute_eigs = compute_eigs
  )

  constant_bundle <- setup$constant_bundle
  weather_bundle <- setup$weather_bundle
  order <- constant_bundle$order

  log_progress(verbose, "Reconstructing raw relationship matrices from setup objects")
  KG_G_A <- constant_bundle$right_add
  KG_G_D <- constant_bundle$right_dom
  KG_GE_A <- constant_bundle$left_add_base * KG_G_A
  KG_GE_D <- constant_bundle$left_add_base * KG_G_D
  KP <- setup$KP
  KP_PE <- setup$KP_PE
  KE_W <- setup$KE_W
  KG_GE_AW <- KE_W * KG_G_A
  KG_GE_DW <- KE_W * KG_G_D
  KP_PW <- setup$KP_PW

  relationship_matrices <- list(
    KG_G_A = KG_G_A,
    KG_G_D = KG_G_D,
    KG_GE_A = KG_GE_A,
    KG_GE_D = KG_GE_D,
    KP = KP,
    KP_PE = KP_PE,
    KE_W = KE_W,
    KG_GE_AW = KG_GE_AW,
    KG_GE_DW = KG_GE_DW,
    KP_PW = KP_PW
  )

  bad_order <- names(relationship_matrices)[!vapply(relationship_matrices, function(K) {
    identical(rownames(K), order) && identical(colnames(K), order)
  }, logical(1))]
  if (length(bad_order) > 0L) {
    stop("The following matrices are not aligned to the expected Pedigree.Env order: ", paste(bad_order, collapse = ", "))
  }

  log_progress(verbose, "All relationship matrices align to Pedigree.Env order")
  log_progress(verbose, "Matrix dimension = ", summarize_dims(KP))

  model_membership <- list(
    "M8.G.P" = c("KG_G_A", "KG_G_D", "KG_GE_A", "KG_GE_D", "KP", "KP_PE"),
    "M10.G.P" = c("KG_G_A", "KG_G_D", "KP", "KE_W", "KG_GE_AW", "KG_GE_DW", "KP_PW")
  )

  manifest <- make_matrix_manifest(relationship_matrices, model_membership)

  matrix_descriptions <- data.frame(
    Matrix = names(relationship_matrices),
    Description = c(
      "Additive genomic main-effect kernel projected from hybrid-level K_A to observation-level Pedigree.Env rows.",
      "Dominance genomic main-effect kernel projected from hybrid-level K_D to observation-level Pedigree.Env rows.",
      "Additive genotype-by-environment kernel: same-environment indicator multiplied elementwise by KG_G_A.",
      "Dominance genotype-by-environment kernel: same-environment indicator multiplied elementwise by KG_G_D.",
      "Phenomic kernel from leakage-safe CV VI FPC scores, scaled within environment.",
      "Phenomic-by-environment kernel: same-environment indicator multiplied elementwise by KP.",
      "Observation-level weather/enviromic kernel from DAP weather FPC scores for this CV regime.",
      "Additive genotype-by-weather kernel: KE_W multiplied elementwise by KG_G_A.",
      "Dominance genotype-by-weather kernel: KE_W multiplied elementwise by KG_G_D.",
      "Phenomic-by-weather kernel: KE_W multiplied elementwise by KP."
    ),
    stringsAsFactors = FALSE
  )

  metadata <- list(
    split_id = split_id,
    seed_num = seed_num,
    fold_num = fold_num,
    split_group = split_group,
    heldout_env = heldout_env,
    data_path = data_path,
    out_root = out_root,
    vi_names_requested = if (length(vi_names) > 0L) vi_names else "<all VIs>",
    n_vi_fpcs_requested = n_vi_fpcs,
    vi_summary = setup$vi_summary,
    vi_fve = setup$vi_fve,
    train_ids = setup$train_ids,
    weather_regime = weather_bundle$regime,
    weather_heldout_env = weather_bundle$heldout_env,
    weather_traits = weather_bundle$weather_traits,
    weather_score_cols = weather_bundle$score_cols,
    row_order = order,
    environment_order = constant_bundle$envs,
    created_at = as.character(Sys.time())
  )

  output_bundle <- list(
    metadata = metadata,
    matrix_manifest = manifest,
    matrix_descriptions = matrix_descriptions,
    model_components = model_membership,
    relationship_matrices = relationship_matrices,
    design_matrices = list(Ze = constant_bundle$Ze),
    environment_kernel = list(
      K_W = weather_bundle$K_W,
      weather_score_matrix = weather_bundle$weather_score_mat
    ),
    phenomic_scores = list(
      VI_all = setup$VI_all,
      vi_score_mat = setup$vi_score_mat
    ),
    row_data = setup$row_data
  )

  if (isTRUE(write_outputs)) {
    out_dir <- file.path(out_root, "collaborator_matrices")
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    prefix <- paste0(split_id, "_M8_M10")
    bundle_file <- file.path(out_dir, paste0(prefix, "_relationship_matrices.rds"))
    manifest_file <- file.path(out_dir, paste0(prefix, "_matrix_manifest.csv"))
    descriptions_file <- file.path(out_dir, paste0(prefix, "_matrix_descriptions.csv"))
    m8_file <- file.path(out_dir, paste0(prefix, "_M8.G.P_matrices.rds"))
    m10_file <- file.path(out_dir, paste0(prefix, "_M10.G.P_matrices.rds"))

    log_progress(verbose, "Saving collaborator matrix bundle")
    saveRDS(output_bundle, bundle_file)
    fwrite(manifest, manifest_file)
    fwrite(matrix_descriptions, descriptions_file)

    if (isTRUE(save_model_matrix_rds)) {
      log_progress(verbose, "Saving optional model-specific matrix RDS files")
      saveRDS(relationship_matrices[model_membership[["M8.G.P"]]], m8_file)
      saveRDS(relationship_matrices[model_membership[["M10.G.P"]]], m10_file)
    }

    if (isTRUE(save_matrix_csv)) {
      log_progress(verbose, "Saving optional matrix CSV files")
      write_matrix_csvs(relationship_matrices, file.path(out_dir, prefix))
    }

    log_progress(verbose, "Wrote bundle: ", bundle_file)
  }

  if (isTRUE(return_objects)) output_bundle else invisible(output_bundle)
}

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  args <- vapply(args, sanitize_arg, FUN.VALUE = character(1))
  if (length(args) != 4L) {
    stop("Expected exactly 4 arguments: seed_num fold_num split_group heldout_env")
  }

  list(
    seed_num = as.integer(args[[1]]),
    fold_num = as.integer(args[[2]]),
    split_group = args[[3]],
    heldout_env = args[[4]]
  )
}

main <- function() {
  parsed <- parse_args()
  build_dap_cv_m8_m10_matrices(
    seed_num = parsed$seed_num,
    fold_num = parsed$fold_num,
    split_group = parsed$split_group,
    heldout_env = parsed$heldout_env
  )
}

if (sys.nframe() == 0L) {
  main()
}
