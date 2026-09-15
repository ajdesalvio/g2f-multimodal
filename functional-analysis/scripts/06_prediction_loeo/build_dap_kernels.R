suppressPackageStartupMessages({
  library(fastmatrix)
  library(dplyr)
  library(data.table)
})

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "06_prediction_loeo", "path_helpers.R"))
loeo_paths <- g2f_loeo_paths("dap")

#### USAGE ####
# Array-oriented HPRC setup script.
# Expects exactly one positional argument:
#   1) heldout_env
#
# Optional environment variables:
#   G2F_DATA_PATH
#   G2F_LOEO_SCORE_PATH
#   G2F_KERNEL_OUT_PATH
#   G2F_KERNEL_RUN_NAME
#   G2F_INCLUDE_ZE

#### HELPERS ####
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

parse_bool_env <- function(env, default = TRUE) {
  val <- toupper(trimws(Sys.getenv(env, unset = "")))
  if (!nzchar(val)) {
    return(default)
  }
  if (val %in% c("TRUE", "T", "1", "YES", "Y")) {
    return(TRUE)
  }
  if (val %in% c("FALSE", "F", "0", "NO", "N")) {
    return(FALSE)
  }
  stop(env, " must be TRUE/FALSE, YES/NO, or 1/0.")
}

args <- commandArgs(trailingOnly = TRUE)
args <- vapply(args, sanitize_arg, FUN.VALUE = character(1))

if (length(args) != 1 || !nzchar(args[1])) {
  stop("Expected exactly 1 argument: heldout_env")
}

heldout_env_i <- args[1]

slurm_array_job_id <- Sys.getenv("SLURM_ARRAY_JOB_ID", unset = "")
slurm_job_id <- Sys.getenv("SLURM_JOB_ID", unset = "local")
slurm_task_id <- Sys.getenv("SLURM_ARRAY_TASK_ID", unset = "local")
include_ze <- parse_bool_env("G2F_INCLUDE_ZE", default = FALSE)

#### THREAD CONTROL ####
# This array design uses one held-out environment per SLURM task and no internal R parallelism.
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

#### PATHS ####
data_path <- get_path(
  "G2F_DATA_PATH",
  loeo_paths$data_path
)

score_path <- get_path("G2F_LOEO_SCORE_PATH", loeo_paths$score_path)

kernel_path_change <- get_path(
  "G2F_KERNEL_OUT_PATH",
  loeo_paths$kernel_path
)

default_run_name <- if (isTRUE(include_ze)) "withZe" else "noZe"
run_name <- Sys.getenv("G2F_KERNEL_RUN_NAME", unset = default_run_name)
run_dir <- file.path(kernel_path_change, run_name)
models_dir <- file.path(run_dir, "models")
logs_dir <- file.path(run_dir, "logs")
summaries_dir <- file.path(run_dir, "summaries")

dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(summaries_dir, recursive = TRUE, showWarnings = FALSE)

safe_env <- sanitize_file_component(heldout_env_i)
job_tag <- paste0("job_", slurm_job_id, "_task_", slurm_task_id)
log_file <- file.path(logs_dir, paste0("heldout_", safe_env, "_", job_tag, ".log"))
summary_file <- file.path(summaries_dir, paste0("summary_", safe_env, ".csv"))
summary_rds_file <- file.path(summaries_dir, paste0("summary_", safe_env, ".rds"))
manifest_file <- file.path(summaries_dir, paste0("manifest_", safe_env, ".csv"))

log_message <- function(...) {
  msg <- paste0(
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    " | pid=", Sys.getpid(),
    " | host=", Sys.info()[["nodename"]],
    " | env=", heldout_env_i,
    " | ",
    paste(..., collapse = "")
  )
  message(msg)
  cat(msg, "\n", file = log_file, append = TRUE)
}

main <- function() {
  log_message("Starting LOEO prediction setup task")
  log_message("data_path = ", data_path)
  log_message("score_path = ", score_path)
  log_message("run_dir = ", run_dir)
  log_message("models_dir = ", models_dir)
  log_message("Include Ze environment fixed-effect BRR term = ", include_ze)
  log_message("SLURM_ARRAY_JOB_ID = ", if (nzchar(slurm_array_job_id)) slurm_array_job_id else "unset")
  log_message("SLURM_JOB_ID = ", slurm_job_id)
  log_message("SLURM_ARRAY_TASK_ID = ", slurm_task_id)

  #### LOAD INPUTS ####
  VI <- fread(file.path(score_path, "FPC_Scores_BLUEs_DAP_LOEO_Projected.csv")) %>% as.data.frame()
  parsed_id <- strsplit(as.character(VI$Pedigree.Env), "\\.")
  VI$Env <- vapply(parsed_id, function(x) paste(tail(x, 2L), collapse = "."), character(1))
  VI$Pedigree <- vapply(parsed_id, function(x) paste(head(x, -2L), collapse = "."), character(1))

  W <- fread(file.path(score_path, "Weather_FPC_Scores_DAP_LOEO_Projected.csv")) %>% as.data.frame()
  envs <- read.csv(file.path(data_path, "Env_Names_G2F_2020_2021.csv"))$Env

  if (!heldout_env_i %in% envs) {
    stop("Held-out environment not found in Env_Names_G2F_2020_2021.csv: ", heldout_env_i)
  }

  order <- read.csv(file.path(data_path, "G2F.2020.2021.Pedigrees.csv"))$Pedigree.Env

  pheno <- fread(file.path(data_path, "Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv")) %>% as.data.frame()
  pheno$Pedigree.Env <- paste(pheno$Pedigree, pheno$Env, sep = ".")
  pheno <- pheno %>% filter(Pedigree.Env %in% order) %>% arrange(Env, Pedigree)
  stopifnot(identical(pheno$Pedigree.Env, order))

  ped.org.cols <- c("Pedigree", "Env", "Pedigree.Env")
  w.org.cols <- c("Env", "Weather.Variable")

  nFPCs <- suppressWarnings(as.integer(Sys.getenv("G2F_VI_NFPCS", unset = "5")))
  viTraits <- trimws(strsplit(Sys.getenv("G2F_VI_NAMES", unset = "NGRDI"), ",", fixed = TRUE)[[1]])
  viTraits <- viTraits[nzchar(viTraits)]
  viFPCs_retain <- as.vector(unlist(lapply(viTraits, function(x) paste0("FPC", seq_len(nFPCs), ".", x))))

  nwFPCs <- suppressWarnings(as.integer(Sys.getenv("G2F_WEATHER_NFPCS", unset = "1")))
  wTraits <- trimws(strsplit(Sys.getenv("G2F_WEATHER_TRAITS", unset = "PTR"), ",", fixed = TRUE)[[1]])
  wTraits <- wTraits[nzchar(wTraits)]
  wFPCs_retain <- paste0("FPC", 1:nwFPCs)
  wFPCs_retain <- paste(expand.grid(wFPCs_retain, wTraits)$Var1, expand.grid(wFPCs_retain, wTraits)$Var2, sep = ".")

  log_message("Loading genomic relationship matrices")
  K_A <- fread(file.path(data_path, "GENOMIC.RELAT.MAT.ADD.csv")) %>% as.data.frame()
  rownames(K_A) <- K_A$V1
  K_A <- as.matrix(K_A[, -1])

  K_D <- fread(file.path(data_path, "GENOMIC.RELAT.MAT.DOM.csv")) %>% as.data.frame()
  rownames(K_D) <- K_D$V1
  K_D <- as.matrix(K_D[, -1])

  stopifnot(identical(rownames(K_A), rownames(K_D)))

  Ze <- model.matrix(~ Env - 1, pheno)
  Za <- model.matrix(~ Pedigree - 1, pheno)
  Za.cols <- gsub("Pedigree", "", colnames(Za))

  stopifnot(identical(Za.cols, colnames(K_A)))
  stopifnot(identical(Za.cols, colnames(K_D)))

  left_add_base <- Ze %*% diag(1, length(envs), length(envs)) %*% t(Ze)
  right_add <- Za %*% K_A %*% t(Za)
  right_dom <- Za %*% K_D %*% t(Za)

  constant_kernels <- list(
    KG_G_A = right_add,
    KG_G_D = right_dom,
    KG_GE_A = left_add_base * right_add,
    KG_GE_D = left_add_base * right_dom
  )
  eig_list_constant <- lapply(constant_kernels, eigen, symmetric = TRUE)
  names(eig_list_constant) <- paste0(names(constant_kernels), ".eig")

  #### BUILD CHANGING KERNELS FOR THIS HELD-OUT ENVIRONMENT ####
  log_message("Filtering phenomic and weather inputs for held-out environment")

  VI_i <- VI %>%
    filter(Heldout.Environment == heldout_env_i) %>%
    select(any_of(ped.org.cols), all_of(viFPCs_retain)) %>%
    arrange(match(Pedigree.Env, order))

  if (nrow(VI_i) != length(order)) {
    stop("Unexpected number of VI rows for held-out environment ", heldout_env_i,
         ". Expected ", length(order), ", found ", nrow(VI_i))
  }

  VI.list <- lapply(envs, function(scaling_env_i) {
    temp <- VI_i %>% filter(Env == scaling_env_i)
    if (nrow(temp) == 0) {
      stop("No VI rows found for scaling environment ", scaling_env_i,
           " under held-out environment ", heldout_env_i)
    }
    rownames(temp) <- paste(temp$Pedigree, scaling_env_i, sep = ".")
    scale(temp[, -c(1:3)], scale = TRUE, center = TRUE)
  })
  names(VI.list) <- envs

  VI.all <- do.call(rbind, VI.list)
  VI.all <- VI.all[order, ]

  log_message("Building phenomic kernels")
  KP <- tcrossprod(as.matrix(VI.all)) / ncol(VI.all)
  rownames(KP) <- order
  colnames(KP) <- order

  KP_PE <- hadamard(left_add_base, KP)
  rownames(KP_PE) <- order
  colnames(KP_PE) <- order

  W_i <- W %>%
    filter(Heldout.Environment == heldout_env_i) %>%
    select(any_of(w.org.cols), all_of(wFPCs_retain)) %>%
    arrange(match(Env, envs))

  if (!identical(W_i$Env, envs)) {
    stop("Weather rows are not aligned to the expected environment order for ", heldout_env_i)
  }

  wfpc.wide.mat <- W_i %>% select(!any_of(w.org.cols)) %>% as.matrix()
  storage.mode(wfpc.wide.mat) <- "double"

  train_weather_idx <- W_i$Env != heldout_env_i
  if (sum(train_weather_idx) != length(envs) - 1L) {
    stop("Expected ", length(envs) - 1L,
         " training environments for weather scaling, found ",
         sum(train_weather_idx), ".")
  }

  log_message("Scaling weather FPCs using training environments only")
  weather_center <- colMeans(wfpc.wide.mat[train_weather_idx, , drop = FALSE], na.rm = TRUE)
  weather_scale <- apply(wfpc.wide.mat[train_weather_idx, , drop = FALSE], 2, sd, na.rm = TRUE)

  weather_center[!is.finite(weather_center)] <- 0
  weather_scale[!is.finite(weather_scale) | weather_scale == 0] <- 1

  scaled_wfpc <- scale(wfpc.wide.mat, center = weather_center, scale = weather_scale)
  scaled_wfpc <- as.matrix(scaled_wfpc)
  scaled_wfpc[!is.finite(scaled_wfpc)] <- 0
  rownames(scaled_wfpc) <- W_i$Env

  K_W <- tcrossprod(scaled_wfpc) / ncol(scaled_wfpc)
  rownames(K_W) <- envs
  colnames(K_W) <- envs

  log_message("Building enviromic interaction kernels")
  KE_W <- Ze %*% K_W %*% t(Ze)
  KG_GE_AW <- hadamard(KE_W, right_add)
  KG_GE_DW <- hadamard(KE_W, right_dom)
  KP_PW <- hadamard(KE_W, KP)

  rownames(KE_W) <- order
  colnames(KE_W) <- order
  rownames(KG_GE_AW) <- order
  colnames(KG_GE_AW) <- order
  rownames(KG_GE_DW) <- order
  colnames(KG_GE_DW) <- order
  rownames(KP_PW) <- order
  colnames(KP_PW) <- order

  log_message("Running eigendecompositions for held-out environment")
  kernels_change <- list(
    KP = KP,
    KP_PE = KP_PE,
    KE_W = KE_W,
    KG_GE_AW = KG_GE_AW,
    KG_GE_DW = KG_GE_DW,
    KP_PW = KP_PW
  )

  eig_list_change <- lapply(kernels_change, eigen)
  eig_list_change <- setNames(eig_list_change, paste0(names(kernels_change), ".eig"))

  KG_G_A.eig <- eig_list_constant[["KG_G_A.eig"]]
  KG_G_D.eig <- eig_list_constant[["KG_G_D.eig"]]
  KG_GE_A.eig <- eig_list_constant[["KG_GE_A.eig"]]
  KG_GE_D.eig <- eig_list_constant[["KG_GE_D.eig"]]
  KP.eig <- eig_list_change[["KP.eig"]]
  KP_PE.eig <- eig_list_change[["KP_PE.eig"]]
  KE_W.eig <- eig_list_change[["KE_W.eig"]]
  KG_GE_AW.eig <- eig_list_change[["KG_GE_AW.eig"]]
  KG_GE_DW.eig <- eig_list_change[["KG_GE_DW.eig"]]
  KP_PW.eig <- eig_list_change[["KP_PW.eig"]]

  #### BUILD MODELS ####
  log_message("Assembling BGLR ETA model objects")
  eta_components <- list(
    KG_G_A = list(V = KG_G_A.eig$vectors, d = KG_G_A.eig$values, model = "RKHS"),
    KG_G_D = list(V = KG_G_D.eig$vectors, d = KG_G_D.eig$values, model = "RKHS"),
    KG_GE_A = list(V = KG_GE_A.eig$vectors, d = KG_GE_A.eig$values, model = "RKHS"),
    KG_GE_D = list(V = KG_GE_D.eig$vectors, d = KG_GE_D.eig$values, model = "RKHS"),
    KP = list(V = KP.eig$vectors, d = KP.eig$values, model = "RKHS"),
    KP_PE = list(V = KP_PE.eig$vectors, d = KP_PE.eig$values, model = "RKHS"),
    KE_W = list(V = KE_W.eig$vectors, d = KE_W.eig$values, model = "RKHS"),
    KG_GE_AW = list(V = KG_GE_AW.eig$vectors, d = KG_GE_AW.eig$values, model = "RKHS"),
    KG_GE_DW = list(V = KG_GE_DW.eig$vectors, d = KG_GE_DW.eig$values, model = "RKHS"),
    KP_PW = list(V = KP_PW.eig$vectors, d = KP_PW.eig$values, model = "RKHS"),
    Ze = list(X = Ze, model = "BRR")
  )

  model_specs <- list(
    "M1.G" = c("KG_G_A", "Ze"),
    "M1.P" = c("KP", "Ze"),
    "M2.G" = c("KG_G_A", "KG_G_D", "Ze"),
    "M3.G" = c("KG_G_A", "KG_G_D", "KG_GE_A", "KG_GE_D", "Ze"),
    "M3.P" = c("KP", "KP_PE", "Ze"),
    "M4.G" = c("KG_G_A", "KG_G_D", "KE_W", "Ze"),
    "M4.P" = c("KP", "KE_W", "Ze"),
    "M5.G" = c("KG_G_A", "KG_G_D", "KE_W", "KG_GE_AW", "KG_GE_DW", "Ze"),
    "M5.P" = c("KP", "KE_W", "KP_PW", "Ze"),
    "M6.G.P" = c("KG_G_A", "KP", "Ze"),
    "M7.G.P" = c("KG_G_A", "KG_G_D", "KP", "Ze"),
    "M8.G.P" = c("KG_G_A", "KG_G_D", "KG_GE_A", "KG_GE_D", "KP", "KP_PE", "Ze"),
    "M9.G.P" = c("KG_G_A", "KG_G_D", "KP", "KE_W", "Ze"),
    "M10.G.P" = c("KG_G_A", "KG_G_D", "KP", "KE_W", "KG_GE_AW", "KG_GE_DW", "KP_PW", "Ze")
  )
  if (!isTRUE(include_ze)) {
    model_specs <- lapply(model_specs, setdiff, "Ze")
  }

  Model.Names <- names(model_specs)
  Models <- lapply(model_specs, function(component_names) {
    lapply(component_names, function(component_name) eta_components[[component_name]])
  })

  Model.Names.Heldout <- paste(Model.Names, heldout_env_i, sep = ".")
  names(Models) <- Model.Names.Heldout

  log_message("Saving ", length(Model.Names.Heldout), " model files")
  for (model_name in Model.Names.Heldout) {
    saveRDS(Models[[model_name]], file = file.path(models_dir, paste0(model_name, ".rds")))
  }

  manifest <- data.frame(
    Heldout.Environment = heldout_env_i,
    Include.Ze = include_ze,
    Model.Name = Model.Names.Heldout,
    Model.File = file.path(models_dir, paste0(Model.Names.Heldout, ".rds")),
    stringsAsFactors = FALSE
  )
  write.csv(manifest, file = manifest_file, row.names = FALSE)

  summary_df <- data.frame(
    Heldout.Environment = heldout_env_i,
    Include.Ze = include_ze,
    Models.Saved = length(Model.Names.Heldout),
    Run.Directory = run_dir,
    Models.Directory = models_dir,
    Log.File = log_file,
    Manifest.File = manifest_file,
    SLURM.Array.Job.ID = if (nzchar(slurm_array_job_id)) slurm_array_job_id else NA_character_,
    SLURM.Job.ID = slurm_job_id,
    SLURM.Array.Task.ID = slurm_task_id,
    stringsAsFactors = FALSE
  )

  write.csv(summary_df, file = summary_file, row.names = FALSE)
  saveRDS(summary_df, file = summary_rds_file)

  log_message("Completed successfully")
  print(summary_df)
  invisible(summary_df)
}

tryCatch(
  main(),
  error = function(e) {
    if (exists("log_message", mode = "function")) {
      log_message("ERROR: ", conditionMessage(e))
    }
    stop(e)
  }
)
