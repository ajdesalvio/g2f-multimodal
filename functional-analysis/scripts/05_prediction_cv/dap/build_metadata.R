suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
})

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "05_prediction_cv", "path_helpers.R"))
cv_paths <- g2f_cv_paths("dap")

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

  pheno$Female <- trimws(as.character(pheno$Female))
  pheno
}

load_canonical_females <- function(common_females_file, expected_n) {
  if (!file.exists(common_females_file)) {
    stop("Missing canonical common-female file: ", common_females_file)
  }

  x <- fread(common_females_file)
  if (!"Female" %in% names(x)) {
    stop("Canonical common-female file must contain a Female column: ", common_females_file)
  }

  females <- trimws(as.character(x$Female))
  if (anyNA(females) || any(!nzchar(females))) {
    stop("Canonical common-female file contains missing or blank Female values.")
  }
  if (anyDuplicated(females)) {
    stop("Canonical common-female file contains duplicate Female values.")
  }

  females <- sort(females)
  if (length(females) != expected_n) {
    stop(
      "Expected ", expected_n, " canonical common females but found ", length(females),
      " in ", common_females_file
    )
  }

  females
}

derive_common_females <- function(pheno, envs) {
  observed_envs <- sort(unique(as.character(pheno$Env)))
  expected_envs <- sort(unique(as.character(envs)))

  if (!setequal(observed_envs, expected_envs)) {
    stop(
      "Phenotype and environment files disagree. Only in phenotype data: ",
      paste(setdiff(observed_envs, expected_envs), collapse = ", "),
      "; only in environment file: ",
      paste(setdiff(expected_envs, observed_envs), collapse = ", ")
    )
  }

  pheno %>%
    distinct(Env, Female) %>%
    filter(!is.na(Female), nzchar(Female)) %>%
    add_count(Female, name = "n_envs") %>%
    filter(n_envs == length(expected_envs)) %>%
    pull(Female) %>%
    unique() %>%
    sort()
}

make_fold_map <- function(common_females, seed_num, k_folds) {
  set.seed(seed_num)
  obs_per_fold <- ceiling(length(common_females) / k_folds)
  folds <- sample(rep(seq_len(k_folds), each = obs_per_fold, length.out = length(common_females)))

  data.frame(
    Seed_Num = seed_num,
    Female = common_females,
    Fold = folds,
    stringsAsFactors = FALSE
  )
}

validate_fold_map <- function(female_folds, common_females, seeds, k_folds) {
  required <- c("Seed_Num", "Female", "Fold")
  if (!all(required %in% names(female_folds))) {
    stop("Fold map is missing required columns: ", paste(setdiff(required, names(female_folds)), collapse = ", "))
  }
  if (nrow(female_folds) != length(common_females) * length(seeds)) {
    stop("Fold map has an unexpected number of rows.")
  }
  if (anyDuplicated(female_folds[, c("Seed_Num", "Female")])) {
    stop("Fold map contains duplicate Seed_Num/Female combinations.")
  }
  if (!setequal(unique(female_folds$Seed_Num), seeds)) {
    stop("Fold map seed values do not match the requested seeds.")
  }
  if (anyNA(female_folds$Fold) || !all(female_folds$Fold %in% seq_len(k_folds))) {
    stop("Fold map contains missing or invalid fold values.")
  }

  by_seed <- split(female_folds, female_folds$Seed_Num)
  bad_seed <- vapply(
    by_seed,
    function(x) nrow(x) != length(common_females) || !setequal(x$Female, common_females),
    logical(1)
  )
  if (any(bad_seed)) {
    stop("Fold map does not contain the canonical female set for seed(s): ", paste(names(bad_seed)[bad_seed], collapse = ", "))
  }

  obs_per_fold <- ceiling(length(common_females) / k_folds)
  expected_sizes <- table(rep(seq_len(k_folds), each = obs_per_fold, length.out = length(common_females)))
  observed_sizes <- female_folds %>% count(Seed_Num, Fold, name = "n")
  bad_sizes <- observed_sizes %>%
    mutate(expected_n = as.integer(expected_sizes[as.character(Fold)])) %>%
    filter(n != expected_n)
  if (nrow(bad_sizes) > 0L) {
    stop("Fold sizes do not match the original R fold-generation algorithm.")
  }

  invisible(TRUE)
}

sanitize_file_component <- function(x) {
  gsub("[^A-Za-z0-9._-]", "_", x)
}

make_metadata_logger <- function(log_file = NULL, emit = TRUE) {
  function(...) {
    msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste(..., collapse = ""))
    if (isTRUE(emit)) message(msg)
    if (!is.null(log_file)) cat(msg, "\n", file = log_file, append = TRUE)
  }
}

run_cv_metadata <- function(
    data_path = get_path("G2F_DATA_PATH", cv_paths$data_path),
    out_root = get_path("G2F_CV_OUT_PATH", cv_paths$out_root),
    common_females_file = get_path("G2F_COMMON_FEMALES_FILE", cv_paths$common_females_file),
    expected_common_females = suppressWarnings(as.integer(Sys.getenv("G2F_EXPECTED_COMMON_FEMALES", unset = "223"))),
    expected_environments = suppressWarnings(as.integer(Sys.getenv("G2F_EXPECTED_ENVIRONMENTS", unset = "19"))),
    seeds = seq_len(10L),
    k_folds = 5L,
    write_outputs = TRUE,
    return_objects = FALSE,
    log_to_console = TRUE) {

  if (is.na(expected_common_females) || expected_common_females < 1L) {
    stop("G2F_EXPECTED_COMMON_FEMALES must be a positive integer.")
  }
  if (is.na(expected_environments) || expected_environments < 1L) {
    stop("G2F_EXPECTED_ENVIRONMENTS must be a positive integer.")
  }

  metadata_dir <- file.path(out_root, "metadata")
  jobs_dir <- file.path(out_root, "jobs")
  logs_dir <- file.path(out_root, "logs")

  if (isTRUE(write_outputs)) {
    dir.create(metadata_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(jobs_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
  }

  log_file <- if (isTRUE(write_outputs)) file.path(logs_dir, "DAP_CV_Metadata_V2.log") else NULL
  log_message <- make_metadata_logger(log_file = log_file, emit = log_to_console)

  log_message("Starting metadata build")
  log_message("data_path = ", data_path)
  log_message("out_root = ", out_root)
  log_message("common_females_file = ", common_females_file)

  pheno <- load_pheno(data_path)
  env_file <- file.path(data_path, "Env_Names_G2F_2020_2021.csv")
  envs <- if (file.exists(env_file)) read.csv(env_file)$Env else sort(unique(pheno$Env))
  envs <- sort(unique(trimws(as.character(envs))))

  if (length(envs) != expected_environments) {
    stop("Expected ", expected_environments, " environments but found ", length(envs), ".")
  }

  common_females <- load_canonical_females(common_females_file, expected_common_females)
  derived_common_females <- derive_common_females(pheno, envs)

  if (!identical(common_females, derived_common_females)) {
    stop(
      "Canonical common females disagree with the phenotype intersection. ",
      "Only in canonical file: ", paste(setdiff(common_females, derived_common_females), collapse = ", "),
      "; only in phenotype intersection: ", paste(setdiff(derived_common_females, common_females), collapse = ", ")
    )
  }

  female_folds <- do.call(
    rbind,
    lapply(seeds, make_fold_map, common_females = common_females, k_folds = k_folds)
  )
  validate_fold_map(female_folds, common_females, seeds, k_folds)

  model_names <- c(
    "M1.G", "M1.P", "M2.G", "M3.G", "M3.P", "M4.G", "M4.P",
    "M5.G", "M5.P", "M6.G.P", "M7.G.P", "M8.G.P", "M9.G.P", "M10.G.P"
  )

  jobs_cv_2_1 <- expand.grid(
    Seed_Num = seeds,
    Fold_Num = seq_len(k_folds),
    Split_Group = "CV_2_1",
    Heldout_Env = "None",
    stringsAsFactors = FALSE
  )
  jobs_cv_0_00 <- expand.grid(
    Seed_Num = seeds,
    Fold_Num = seq_len(k_folds),
    Split_Group = "CV_0_00",
    Heldout_Env = envs,
    stringsAsFactors = FALSE
  )

  split_jobs <- bind_rows(jobs_cv_2_1, jobs_cv_0_00) %>%
    mutate(
      Split_ID = paste(
        sprintf("Seed%02d", Seed_Num), sprintf("Fold%d", Fold_Num),
        Split_Group, sanitize_file_component(Heldout_Env), sep = "."
      ),
      Weather_Regime = ifelse(Split_Group == "CV_2_1", "ALL", "LOEO"),
      Metric_Train = ifelse(Split_Group == "CV_2_1", "CV2", "CV0"),
      Metric_Test = ifelse(Split_Group == "CV_2_1", "CV1", "CV00")
    ) %>%
    arrange(Split_Group, Heldout_Env, Seed_Num, Fold_Num)

  summary_df <- data.frame(
    Common_Females = length(common_females),
    Common_Females_File = normalizePath(common_females_file, winslash = "/", mustWork = TRUE),
    Common_Females_MD5 = unname(tools::md5sum(common_females_file)),
    Seeds = length(seeds),
    Folds = k_folds,
    Environments = length(envs),
    Split_Jobs = nrow(split_jobs),
    Models = length(model_names),
    stringsAsFactors = FALSE
  )

  if (isTRUE(write_outputs)) {
    fwrite(female_folds, file.path(metadata_dir, "female_folds.csv"))
    saveRDS(female_folds, file.path(metadata_dir, "female_folds.rds"))
    fwrite(data.frame(Female = common_females), file.path(metadata_dir, "common_females.csv"))
    fwrite(split_jobs, file.path(metadata_dir, "split_jobs.csv"))
    saveRDS(split_jobs, file.path(metadata_dir, "split_jobs.rds"))
    fwrite(
      split_jobs[, c("Seed_Num", "Fold_Num", "Split_Group", "Heldout_Env")],
      file.path(jobs_dir, "DAP_CV_Setup_Jobs.txt"), sep = " ", col.names = FALSE
    )
    fwrite(
      split_jobs[, c("Seed_Num", "Fold_Num", "Split_Group", "Heldout_Env")],
      file.path(jobs_dir, "DAP_CV_Prediction_Jobs.txt"), sep = " ", col.names = FALSE
    )
    fwrite(data.frame(Model_Name = model_names), file.path(metadata_dir, "model_names.csv"))
    fwrite(summary_df, file.path(metadata_dir, "metadata_summary.csv"))
  }

  log_message("Validated ", length(common_females), " common females across ", length(envs), " environments")
  log_message("Completed metadata build with ", nrow(split_jobs), " split jobs")
  print(summary_df)

  result <- list(
    summary_df = summary_df,
    pheno = pheno,
    envs = envs,
    common_females = common_females,
    derived_common_females = derived_common_females,
    female_folds = female_folds,
    split_jobs = split_jobs,
    model_names = model_names,
    common_females_file = common_females_file,
    data_path = data_path,
    out_root = out_root
  )

  if (isTRUE(return_objects)) result else invisible(summary_df)
}

main <- function() run_cv_metadata()

if (sys.nframe() == 0L) main()
