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

load_pheno <- function(data_path) {
  pheno_rds <- file.path(data_path, "Pheno_Data.rds")
  if (file.exists(pheno_rds)) {
    pheno <- readRDS(pheno_rds)
  } else {
    pheno <- fread(file.path(data_path, "Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv")) %>%
      as.data.frame()
    if (!"Pedigree.Env" %in% names(pheno)) {
      pheno$Pedigree.Env <- paste(pheno$Pedigree, pheno$Env, sep = ".")
    }
  }

  pheno <- as.data.frame(pheno)

  if (!"Female" %in% names(pheno)) {
    pheno$Female <- vapply(
      strsplit(as.character(pheno$Pedigree), "/"),
      function(x) if (length(x) >= 1L) x[[1]] else NA_character_,
      FUN.VALUE = character(1)
    )
  }

  pheno
}

make_constant_logger <- function(log_file = NULL, emit = TRUE) {
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

run_cv_constant_kernels <- function(
    data_path = get_path("G2F_DATA_PATH", cv_paths$data_path),
    out_root = get_path("G2F_CV_OUT_PATH", cv_paths$out_root),
    write_outputs = TRUE,
    return_objects = FALSE,
    log_to_console = TRUE) {

  bundles_dir <- file.path(out_root, "bundles")
  logs_dir <- file.path(out_root, "logs")

  if (isTRUE(write_outputs)) {
    dir.create(bundles_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
  }

  log_file <- if (isTRUE(write_outputs)) file.path(logs_dir, "AGDD_CV_Constant_Kernels.log") else NULL
  log_message <- make_constant_logger(log_file = log_file, emit = log_to_console)

  log_message("Starting constant kernel build")

  envs <- read.csv(file.path(data_path, "Env_Names_G2F_2020_2021.csv"))$Env
  order <- read.csv(file.path(data_path, "G2F.2020.2021.Pedigrees.csv"))$Pedigree.Env

  pheno <- load_pheno(data_path) %>%
    filter(Pedigree.Env %in% order) %>%
    arrange(match(Pedigree.Env, order))

  if (!identical(pheno$Pedigree.Env, order)) {
    stop("Phenotypic row order does not match G2F.2020.2021.Pedigrees.csv")
  }

  K_A <- fread(file.path(data_path, "GENOMIC.RELAT.MAT.ADD.csv")) %>% as.data.frame()
  rownames(K_A) <- K_A$V1
  K_A <- as.matrix(K_A[, -1, drop = FALSE])

  K_D <- fread(file.path(data_path, "GENOMIC.RELAT.MAT.DOM.csv")) %>% as.data.frame()
  rownames(K_D) <- K_D$V1
  K_D <- as.matrix(K_D[, -1, drop = FALSE])

  stopifnot(identical(rownames(K_A), rownames(K_D)))

  Ze <- model.matrix(~ Env - 1, pheno)
  Za <- model.matrix(~ Pedigree - 1, pheno)
  Za_cols <- gsub("^Pedigree", "", colnames(Za))

  if (!identical(Za_cols, colnames(K_A))) {
    stop("Pedigree design matrix columns are not aligned to the additive genomic matrix.")
  }
  if (!identical(Za_cols, colnames(K_D))) {
    stop("Pedigree design matrix columns are not aligned to the dominance genomic matrix.")
  }

  left_add_base <- tcrossprod(Ze)
  right_add <- Za %*% K_A %*% t(Za)
  right_dom <- Za %*% K_D %*% t(Za)

  rownames(left_add_base) <- order
  colnames(left_add_base) <- order
  rownames(right_add) <- order
  colnames(right_add) <- order
  rownames(right_dom) <- order
  colnames(right_dom) <- order

  constant_kernels <- list(
    KG_G_A = right_add,
    KG_G_D = right_dom,
    KG_GE_A = left_add_base * right_add,
    KG_GE_D = left_add_base * right_dom
  )

  log_message("Running eigendecomposition on constant kernels")
  eig_list_constant <- lapply(constant_kernels, eigen, symmetric = TRUE)
  names(eig_list_constant) <- paste0(names(constant_kernels), ".eig")

  row_cols <- intersect(
    c("Pedigree.Env", "Pedigree", "Env", "Female", "Yield.t.ha.BLUE", "Year"),
    names(pheno)
  )

  constant_bundle <- list(
    envs = envs,
    order = order,
    row_data = pheno[, row_cols, drop = FALSE],
    Ze = Ze,
    left_add_base = left_add_base,
    right_add = right_add,
    right_dom = right_dom,
    eig_list_constant = eig_list_constant
  )

  bundle_file <- file.path(bundles_dir, "constant_kernel_bundle.rds")
  eig_file <- file.path(bundles_dir, "Constant_Kernels_Eigendecomp.rds")
  summary_file <- file.path(bundles_dir, "constant_kernel_summary.csv")

  summary_df <- data.frame(
    Environments = length(envs),
    Observations = length(order),
    Additive.Genotypes = ncol(K_A),
    Dominance.Genotypes = ncol(K_D),
    Bundle_File = bundle_file,
    Eig_File = eig_file,
    stringsAsFactors = FALSE
  )

  if (isTRUE(write_outputs)) {
    saveRDS(constant_bundle, bundle_file)
    saveRDS(eig_list_constant, eig_file)
    fwrite(summary_df, summary_file)
  }

  log_message("Completed constant kernel build")
  print(summary_df)

  result <- list(
    summary_df = summary_df,
    constant_bundle = constant_bundle,
    constant_kernels = constant_kernels,
    eig_list_constant = eig_list_constant,
    pheno = pheno,
    K_A = K_A,
    K_D = K_D,
    data_path = data_path,
    out_root = out_root
  )

  if (isTRUE(return_objects)) result else invisible(summary_df)
}

main <- function() {
  run_cv_constant_kernels()
}

if (sys.nframe() == 0L) {
  main()
}
