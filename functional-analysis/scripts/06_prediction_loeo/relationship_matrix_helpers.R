suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
})

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "06_prediction_loeo", "path_helpers.R"))

################################################################################
# LOEO relationship-matrix setup for models M8.G.P and M10.G.P
################################################################################
#
# Purpose
# -------
# This script builds the relationship matrices used by two leave-one-environment-
# out (LOEO) prediction models:
#
#   M8.G.P:
#     KG_G_A, KG_G_D, KG_GE_A, KG_GE_D, KP, KP_PE
#
#   M10.G.P:
#     KG_G_A, KG_G_D, KP, KE_W, KG_GE_AW, KG_GE_DW, KP_PW
#
# In our original BGLR workflow, these matrices were eigendecomposed and then
# passed into BGLR as RKHS terms. This script stops at the matrix stage
# and saves the raw matrices in an RDS bundle so another prediction
# method can eigendecompose or otherwise transform them independently.
#
# Important leakage-control assumption
# ------------------------------------
# This script assumes the VI and weather FPC score files have already been built
# using the leakage-safe LOEO FPCA projection workflow:
#
#   1. Fit FPCA bases using all environments except the held-out environment.
#   2. Project the held-out environment's sparse curves onto those bases.
#   3. Save one row per Pedigree.Env or Env for each Heldout.Environment.
#
# This script does not refit FPCA. It consumes the projected score files produced
# by the earlier DAP or AGDD LOEO FPCA scripts.
#
# Example usage
# -------------
# From a terminal:
#
#   Rscript scripts/06_prediction_loeo/relationship_matrix_helpers.R DAP MIH1.2020
#   Rscript scripts/06_prediction_loeo/relationship_matrix_helpers.R AGDD MIH1.2020
#
# After loading the output bundle:
#
#   bundle <- readRDS("DAP_LOEO_MIH1.2020_M8_M10_relationship_matrices.rds")
#   m8_matrices <- bundle$relationship_matrices[bundle$model_components[["M8.G.P"]]]
#   m10_matrices <- bundle$relationship_matrices[bundle$model_components[["M10.G.P"]]]
#
# Optional environment variables
# ------------------------------
#   G2F_DATA_PATH:
#     Folder containing input data files.
#
#   G2F_MATRIX_OUT_PATH:
#     Folder where output RDS/CSV files should be written.
#
#   G2F_LOEO_SCORE_PATH:
#     Folder containing leakage-safe projected VI and weather score files.
#
#   G2F_VI_NFPCS:
#     Number of VI FPCs to retain per vegetation index. Default: 5.
#
#   G2F_VI_NAMES:
#     Comma-separated vegetation indices to retain. Example: NGRDI.
#     Leave blank to use every VI present in the projected score file.
#
#   G2F_WEATHER_NFPCS:
#     Number of weather FPCs to retain per weather trait. Default: 1.
#
#   G2F_WEATHER_TRAITS:
#     Comma-separated weather traits to retain. Default: PTR.
#
#   G2F_SAVE_MATRIX_CSV:
#     TRUE/FALSE. If TRUE, also writes each large relationship matrix as CSV.
#     Default: FALSE because these matrices are large (~2 gigabytes).
#
#   G2F_SAVE_MODEL_MATRIX_RDS:
#     TRUE/FALSE. If TRUE, also writes separate M8 and M10 RDS files containing
#     only the matrices for those models. Default: FALSE to avoid duplicating
#     very large matrices on disk.
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

safe_scale_matrix <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"

  centers <- colMeans(x, na.rm = TRUE)
  sds <- apply(x, 2, sd, na.rm = TRUE)
  centers[!is.finite(centers)] <- 0
  sds[!is.finite(sds) | sds == 0] <- 1

  out <- sweep(x, 2, centers, "-")
  out <- sweep(out, 2, sds, "/")
  out[!is.finite(out)] <- 0
  out
}

split_pedigree_env <- function(pedigree_env) {
  pieces <- strsplit(as.character(pedigree_env), "\\.")
  parsed <- lapply(pieces, function(x) {
    if (length(x) < 3L) {
      stop("Could not parse Pedigree.Env value: ", paste(x, collapse = "."))
    }
    env <- paste(tail(x, 2L), collapse = ".")
    pedigree <- paste(head(x, -2L), collapse = ".")
    c(Pedigree = pedigree, Env = env)
  })
  as.data.frame(do.call(rbind, parsed), stringsAsFactors = FALSE)
}

load_relationship_matrix <- function(file_path) {
  mat_df <- fread(file_path) %>% as.data.frame()
  rownames(mat_df) <- mat_df[[1]]
  mat <- as.matrix(mat_df[, -1, drop = FALSE])
  storage.mode(mat) <- "double"
  mat
}

make_matrix_manifest <- function(matrices, row_role, model_membership) {
  do.call(rbind, lapply(names(matrices), function(name_i) {
    mat <- matrices[[name_i]]
    data.frame(
      Matrix = name_i,
      Rows = nrow(mat),
      Columns = ncol(mat),
      Row_Role = row_role[[name_i]],
      Col_Role = row_role[[name_i]],
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

build_weather_score_matrix <- function(W, heldout_env, envs, weather_traits, n_weather_fpcs) {
  weather_fpc_cols <- as.vector(unlist(lapply(weather_traits, function(trait_i) {
    paste0("FPC", seq_len(n_weather_fpcs), ".", trait_i)
  })))

  missing_weather_cols <- setdiff(weather_fpc_cols, names(W))
  if (length(missing_weather_cols) > 0L) {
    stop("Missing weather FPC columns: ", paste(missing_weather_cols, collapse = ", "))
  }

  W_i <- W %>%
    filter(Heldout.Environment == heldout_env, Weather.Variable %in% weather_traits)

  rows_by_env <- lapply(envs, function(env_i) {
    env_df <- W_i %>% filter(Env == env_i)
    if (nrow(env_df) == 0L) {
      stop("Weather score file is missing environment ", env_i, " for held-out environment ", heldout_env)
    }

    vals <- unlist(lapply(weather_traits, function(trait_i) {
      trait_df <- env_df %>% filter(Weather.Variable == trait_i)
      if (nrow(trait_df) == 0L) {
        stop("Weather score file is missing trait ", trait_i, " for environment ", env_i)
      }
      if (nrow(trait_df) > 1L) {
        stop("Weather score file has duplicate rows for trait ", trait_i, " in environment ", env_i)
      }
      cols_i <- paste0("FPC", seq_len(n_weather_fpcs), ".", trait_i)
      as.numeric(trait_df[1, cols_i, drop = TRUE])
    }))

    vals
  })

  score_mat <- do.call(rbind, rows_by_env)
  rownames(score_mat) <- envs
  colnames(score_mat) <- weather_fpc_cols

  list(
    raw_rows = W_i,
    score_matrix = score_mat,
    score_cols = weather_fpc_cols
  )
}

resolve_vi_feature_cols <- function(VI, vi_names, n_vi_fpcs) {
  vi_names <- trimws(vi_names)
  vi_names <- vi_names[nzchar(vi_names)]

  if (length(vi_names) > 0L) {
    expected_cols <- as.vector(unlist(lapply(vi_names, function(vi_i) {
      paste0("FPC", seq_len(n_vi_fpcs), ".", vi_i)
    })))

    missing_cols <- setdiff(expected_cols, names(VI))
    if (length(missing_cols) > 0L) {
      stop(
        "The requested VI/FPC score columns were not found: ",
        paste(missing_cols, collapse = ", ")
      )
    }

    return(expected_cols)
  }

  fpc_prefixes <- paste0("FPC", seq_len(n_vi_fpcs), ".")
  cols <- names(VI)[vapply(
    names(VI),
    function(col_i) any(startsWith(col_i, fpc_prefixes)),
    logical(1)
  )]

  if (length(cols) == 0L) {
    stop("No VI FPC score columns were found for the requested number of VI FPCs.")
  }

  cols
}

set_single_thread <- function() {
  suppressWarnings({
    if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
      RhpcBLASctl::blas_set_num_threads(1)
      RhpcBLASctl::omp_set_num_threads(1)
    } else {
      Sys.setenv(
        OMP_NUM_THREADS = "1",
        MKL_NUM_THREADS = "1",
        OPENBLAS_NUM_THREADS = "1",
        VECLIB_MAXIMUM_THREADS = "1"
      )
    }
  })
}

build_loeo_m8_m10_matrices <- function(
    time_domain,
    heldout_env,
    data_path = NULL,
    score_path = NULL,
    out_path = NULL,
    n_vi_fpcs = NULL,
    vi_names = parse_csv_env("G2F_VI_NAMES", "NGRDI"),
    n_weather_fpcs = suppressWarnings(as.integer(Sys.getenv("G2F_WEATHER_NFPCS", unset = "1"))),
    weather_traits = parse_csv_env("G2F_WEATHER_TRAITS", "PTR"),
    write_outputs = TRUE,
    return_objects = FALSE,
    save_matrix_csv = identical(toupper(Sys.getenv("G2F_SAVE_MATRIX_CSV", unset = "FALSE")), "TRUE"),
    save_model_matrix_rds = identical(toupper(Sys.getenv("G2F_SAVE_MODEL_MATRIX_RDS", unset = "FALSE")), "TRUE"),
    verbose = is_truthy_env("G2F_VERBOSE", "TRUE")) {

  time_domain <- toupper(sanitize_arg(time_domain))
  heldout_env <- sanitize_arg(heldout_env)

  if (!time_domain %in% c("DAP", "AGDD")) {
    stop("time_domain must be either DAP or AGDD.")
  }
  if (!nzchar(heldout_env)) {
    stop("heldout_env must be a non-empty environment name, for example MIH1.2020.")
  }
  if (is.na(n_vi_fpcs) || n_vi_fpcs < 1L) {
    stop("G2F_VI_NFPCS must be a positive integer.")
  }
  if (is.null(n_vi_fpcs)) {
    n_vi_fpcs <- suppressWarnings(as.integer(Sys.getenv(
      "G2F_VI_NFPCS",
      unset = if (time_domain == "DAP") "5" else "7"
    )))
  }
  vi_names <- trimws(vi_names)
  vi_names <- vi_names[nzchar(vi_names)]
  if (is.na(n_weather_fpcs) || n_weather_fpcs < 1L) {
    stop("G2F_WEATHER_NFPCS must be a positive integer.")
  }
  if (length(weather_traits) == 0L) {
    stop("At least one weather trait must be supplied.")
  }

  set_single_thread()

  defaults <- g2f_loeo_paths(time_domain)
  data_path <- if (is.null(data_path)) get_path("G2F_DATA_PATH", defaults$data_path) else data_path
  score_path <- if (is.null(score_path)) get_path("G2F_LOEO_SCORE_PATH", defaults$score_path) else score_path
  out_path <- if (is.null(out_path)) get_path("G2F_MATRIX_OUT_PATH", defaults$matrix_path) else out_path

  vi_score_file <- if (time_domain == "DAP") {
    "FPC_Scores_BLUEs_DAP_LOEO_Projected.csv"
  } else {
    "FPC_Scores_BLUEs_AGDD_LOEO_Projected.csv"
  }

  weather_score_file <- if (time_domain == "DAP") {
    "Weather_FPC_Scores_DAP_LOEO_Projected.csv"
  } else {
    "Weather_FPC_Scores_AGDD_LOEO_Projected.csv"
  }

  log_progress(verbose, "Starting LOEO relationship-matrix setup")
  log_progress(verbose, "time_domain = ", time_domain)
  log_progress(verbose, "heldout_env = ", heldout_env)
  log_progress(verbose, "data_path = ", data_path)
  log_progress(verbose, "score_path = ", score_path)
  log_progress(verbose, "out_path = ", out_path)
  log_progress(verbose, "VI selection = ", if (length(vi_names) > 0L) paste(vi_names, collapse = ",") else "<all VIs>")
  log_progress(verbose, "VI FPCs retained per selected VI = ", n_vi_fpcs)
  log_progress(verbose, "Weather traits = ", paste(weather_traits, collapse = ","))
  log_progress(verbose, "Weather FPCs retained per selected trait = ", n_weather_fpcs)

  if (isTRUE(write_outputs)) {
    dir.create(out_path, recursive = TRUE, showWarnings = FALSE)
  }

  log_progress(verbose, "Loading environment names and Pedigree.Env order")
  envs <- read.csv(file.path(data_path, "Env_Names_G2F_2020_2021.csv"))$Env
  if (!heldout_env %in% envs) {
    stop("Held-out environment not found in Env_Names_G2F_2020_2021.csv: ", heldout_env)
  }

  order <- read.csv(file.path(data_path, "G2F.2020.2021.Pedigrees.csv"))$Pedigree.Env
  log_progress(verbose, "Environment count = ", length(envs))
  log_progress(verbose, "Observation count = ", length(order))

  log_progress(verbose, "Loading phenotype data and aligning rows to Pedigree.Env order")
  pheno <- fread(file.path(data_path, "Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv")) %>%
    as.data.frame() %>%
    mutate(Pedigree.Env = paste(Pedigree, Env, sep = ".")) %>%
    filter(Pedigree.Env %in% order) %>%
    arrange(match(Pedigree.Env, order))

  if (!identical(pheno$Pedigree.Env, order)) {
    stop("Phenotype rows are not aligned to G2F.2020.2021.Pedigrees.csv.")
  }
  log_progress(verbose, "Phenotype rows aligned: ", nrow(pheno))

  log_progress(verbose, "Loading additive and dominance genomic relationship matrices")
  K_A <- load_relationship_matrix(file.path(data_path, "GENOMIC.RELAT.MAT.ADD.csv"))
  K_D <- load_relationship_matrix(file.path(data_path, "GENOMIC.RELAT.MAT.DOM.csv"))
  if (!identical(rownames(K_A), rownames(K_D))) {
    stop("Additive and dominance genomic matrices do not have identical row names.")
  }
  log_progress(verbose, "K_A dimensions = ", summarize_dims(K_A))
  log_progress(verbose, "K_D dimensions = ", summarize_dims(K_D))

  log_progress(verbose, "Building genotype and environment incidence matrices")
  Ze <- model.matrix(~ Env - 1, pheno)
  Za <- model.matrix(~ Pedigree - 1, pheno)
  za_cols <- gsub("^Pedigree", "", colnames(Za))

  if (!identical(za_cols, colnames(K_A))) {
    stop("Pedigree design matrix columns are not aligned to the additive genomic matrix.")
  }
  if (!identical(za_cols, colnames(K_D))) {
    stop("Pedigree design matrix columns are not aligned to the dominance genomic matrix.")
  }
  log_progress(verbose, "Ze dimensions = ", summarize_dims(Ze))
  log_progress(verbose, "Za dimensions = ", summarize_dims(Za))

  # Observation-level genomic main-effect kernels.
  log_progress(verbose, "Building KG_G_A and KG_G_D")
  KG_G_A <- Za %*% K_A %*% t(Za)
  KG_G_D <- Za %*% K_D %*% t(Za)
  rownames(KG_G_A) <- order
  colnames(KG_G_A) <- order
  rownames(KG_G_D) <- order
  colnames(KG_G_D) <- order

  # Same-environment indicator kernel. Multiplying by this isolates GxE terms.
  log_progress(verbose, "Building same-environment indicator kernel")
  KE_identity <- tcrossprod(Ze)
  rownames(KE_identity) <- order
  colnames(KE_identity) <- order

  log_progress(verbose, "Building KG_GE_A and KG_GE_D")
  KG_GE_A <- KE_identity * KG_G_A
  KG_GE_D <- KE_identity * KG_G_D
  rownames(KG_GE_A) <- order
  colnames(KG_GE_A) <- order
  rownames(KG_GE_D) <- order
  colnames(KG_GE_D) <- order

  log_progress(verbose, "Loading leakage-safe LOEO VI FPC scores from ", vi_score_file)
  VI <- fread(file.path(score_path, vi_score_file)) %>% as.data.frame()
  parsed_vi <- split_pedigree_env(VI$Pedigree.Env)
  VI$Pedigree <- parsed_vi$Pedigree
  VI$Env <- parsed_vi$Env

  ped_cols <- c("Pedigree", "Env", "Pedigree.Env")
  vi_feature_cols <- resolve_vi_feature_cols(VI, vi_names = vi_names, n_vi_fpcs = n_vi_fpcs)
  log_progress(verbose, "VI FPC columns selected = ", length(vi_feature_cols))
  log_progress(verbose, "First VI columns: ", paste(utils::head(vi_feature_cols, 10L), collapse = ", "))

  VI_i <- VI %>%
    filter(Heldout.Environment == heldout_env) %>%
    select(any_of(ped_cols), all_of(vi_feature_cols)) %>%
    arrange(match(Pedigree.Env, order))

  if (nrow(VI_i) != length(order)) {
    stop(
      "Unexpected number of VI score rows for held-out environment ", heldout_env,
      ". Expected ", length(order), ", found ", nrow(VI_i), "."
    )
  }

  # Scale VI FPC scores within each environment, matching the original LOEO setup.
  log_progress(verbose, "Scaling selected VI FPC scores within each environment")
  VI_by_env <- lapply(envs, function(env_i) {
    temp <- VI_i %>% filter(Env == env_i)
    if (nrow(temp) == 0L) {
      stop("No VI rows found for environment ", env_i, " under held-out environment ", heldout_env)
    }
    scaled <- safe_scale_matrix(temp[, vi_feature_cols, drop = FALSE])
    rownames(scaled) <- temp$Pedigree.Env
    scaled
  })

  VI_all <- do.call(rbind, VI_by_env)
  VI_all <- VI_all[order, , drop = FALSE]
  log_progress(verbose, "VI_all dimensions = ", summarize_dims(VI_all))

  log_progress(verbose, "Building KP")
  KP <- tcrossprod(VI_all) / ncol(VI_all)
  rownames(KP) <- order
  colnames(KP) <- order

  log_progress(verbose, "Building KP_PE")
  KP_PE <- KE_identity * KP
  rownames(KP_PE) <- order
  colnames(KP_PE) <- order

  log_progress(verbose, "Loading leakage-safe LOEO weather FPC scores from ", weather_score_file)
  W <- fread(file.path(score_path, weather_score_file)) %>% as.data.frame()
  weather_var_col <- intersect(c("Weather.Variable", "Weather_Variable", "Weather.Var"), names(W))
  if (length(weather_var_col) == 0L) {
    stop("Weather score file is missing a weather-variable column.")
  }
  weather_var_col <- weather_var_col[[1]]
  names(W)[names(W) == weather_var_col] <- "Weather.Variable"

  weather_prepared <- build_weather_score_matrix(
    W = W,
    heldout_env = heldout_env,
    envs = envs,
    weather_traits = weather_traits,
    n_weather_fpcs = n_weather_fpcs
  )
  weather_fpc_cols <- weather_prepared$score_cols
  weather_score_matrix <- weather_prepared$score_matrix
  log_progress(verbose, "Weather FPC columns selected = ", paste(weather_fpc_cols, collapse = ", "))
  log_progress(verbose, "Weather score matrix dimensions before scaling = ", summarize_dims(weather_score_matrix))
  weather_score_matrix <- safe_scale_matrix(weather_score_matrix)
  rownames(weather_score_matrix) <- envs

  log_progress(verbose, "Building environment-level K_W")
  K_W <- tcrossprod(weather_score_matrix) / ncol(weather_score_matrix)
  rownames(K_W) <- envs
  colnames(K_W) <- envs

  # Project the environment-level weather kernel to the observation level.
  log_progress(verbose, "Projecting K_W to observation-level KE_W")
  KE_W <- Ze %*% K_W %*% t(Ze)
  rownames(KE_W) <- order
  colnames(KE_W) <- order

  log_progress(verbose, "Building KG_GE_AW, KG_GE_DW, and KP_PW")
  KG_GE_AW <- KE_W * KG_G_A
  KG_GE_DW <- KE_W * KG_G_D
  KP_PW <- KE_W * KP
  rownames(KG_GE_AW) <- order
  colnames(KG_GE_AW) <- order
  rownames(KG_GE_DW) <- order
  colnames(KG_GE_DW) <- order
  rownames(KP_PW) <- order
  colnames(KP_PW) <- order

  log_progress(verbose, "Packaging relationship matrices for M8.G.P and M10.G.P")
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

  model_membership <- list(
    "M8.G.P" = c("KG_G_A", "KG_G_D", "KG_GE_A", "KG_GE_D", "KP", "KP_PE"),
    "M10.G.P" = c("KG_G_A", "KG_G_D", "KP", "KE_W", "KG_GE_AW", "KG_GE_DW", "KP_PW")
  )

  row_role <- setNames(rep("Pedigree.Env observations", length(relationship_matrices)), names(relationship_matrices))
  manifest <- make_matrix_manifest(relationship_matrices, row_role, model_membership)

  matrix_descriptions <- data.frame(
    Matrix = names(relationship_matrices),
    Description = c(
      "Additive genomic main-effect kernel projected from hybrid-level K_A to observation-level Pedigree.Env rows.",
      "Dominance genomic main-effect kernel projected from hybrid-level K_D to observation-level Pedigree.Env rows.",
      "Additive genotype-by-environment kernel: same-environment indicator multiplied elementwise by KG_G_A.",
      "Dominance genotype-by-environment kernel: same-environment indicator multiplied elementwise by KG_G_D.",
      "Phenomic kernel from leakage-safe LOEO VI FPC scores, scaled within environment.",
      "Phenomic-by-environment kernel: same-environment indicator multiplied elementwise by KP.",
      "Observation-level weather/enviromic kernel from leakage-safe LOEO weather FPC scores.",
      "Additive genotype-by-weather kernel: KE_W multiplied elementwise by KG_G_A.",
      "Dominance genotype-by-weather kernel: KE_W multiplied elementwise by KG_G_D.",
      "Phenomic-by-weather kernel: KE_W multiplied elementwise by KP."
    ),
    stringsAsFactors = FALSE
  )

  metadata <- list(
    time_domain = time_domain,
    heldout_environment = heldout_env,
    data_path = data_path,
    score_path = score_path,
    vi_score_file = vi_score_file,
    weather_score_file = weather_score_file,
    vi_names_requested = if (length(vi_names) > 0L) vi_names else "<all VIs>",
    n_vi_fpcs_requested = n_vi_fpcs,
    vi_fpc_columns_used = vi_feature_cols,
    weather_traits = weather_traits,
    n_weather_fpcs_requested = n_weather_fpcs,
    weather_fpc_columns_used = weather_fpc_cols,
    row_order = order,
    environment_order = envs,
    created_at = as.character(Sys.time())
  )

  output_bundle <- list(
    metadata = metadata,
    matrix_manifest = manifest,
    matrix_descriptions = matrix_descriptions,
    model_components = model_membership,
    relationship_matrices = relationship_matrices,
    design_matrices = list(Ze = Ze, Za = Za),
    environment_kernel = list(K_W = K_W, weather_score_matrix = weather_score_matrix),
    phenomic_scores = list(VI_all = VI_all),
    phenotype_rows = pheno
  )

  if (isTRUE(write_outputs)) {
    safe_env <- sanitize_file_component(heldout_env)
    prefix <- paste(time_domain, "LOEO", safe_env, sep = "_")

    bundle_file <- file.path(out_path, paste0(prefix, "_M8_M10_relationship_matrices.rds"))
    manifest_file <- file.path(out_path, paste0(prefix, "_matrix_manifest.csv"))
    descriptions_file <- file.path(out_path, paste0(prefix, "_matrix_descriptions.csv"))
    m8_file <- file.path(out_path, paste0(prefix, "_M8.G.P_matrices.rds"))
    m10_file <- file.path(out_path, paste0(prefix, "_M10.G.P_matrices.rds"))

    log_progress(verbose, "Saving main relationship-matrix bundle")
    saveRDS(output_bundle, bundle_file)
    fwrite(manifest, manifest_file)
    fwrite(matrix_descriptions, descriptions_file)

    if (isTRUE(save_model_matrix_rds)) {
      log_progress(verbose, "Saving optional model-specific matrix RDS files")
      saveRDS(relationship_matrices[model_membership[["M8.G.P"]]], m8_file)
      saveRDS(relationship_matrices[model_membership[["M10.G.P"]]], m10_file)
      log_progress(verbose, "Wrote M8 matrix list: ", m8_file)
      log_progress(verbose, "Wrote M10 matrix list: ", m10_file)
    }

    if (isTRUE(save_matrix_csv)) {
      log_progress(verbose, "Saving optional matrix CSV files")
      write_matrix_csvs(relationship_matrices, file.path(out_path, prefix))
    }

    log_progress(verbose, "Wrote bundle: ", bundle_file)
  }

  if (isTRUE(return_objects)) output_bundle else invisible(output_bundle)
}

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  args <- vapply(args, sanitize_arg, FUN.VALUE = character(1))
  if (length(args) != 2L) {
    stop("Expected exactly 2 arguments: time_domain heldout_env")
  }
  list(time_domain = args[[1]], heldout_env = args[[2]])
}

main <- function() {
  parsed <- parse_args()
  build_loeo_m8_m10_matrices(
    time_domain = parsed$time_domain,
    heldout_env = parsed$heldout_env
  )
}

if (sys.nframe() == 0L) {
  main()
}
