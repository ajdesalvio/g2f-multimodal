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

sanitize_file_component <- function(x) {
  gsub("[^A-Za-z0-9._-]", "_", x)
}

make_metadata_logger <- function(log_file = NULL, emit = TRUE) {
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

run_cv_metadata <- function(
    data_path = get_path("G2F_DATA_PATH", cv_paths$data_path),
    out_root = get_path("G2F_CV_OUT_PATH", cv_paths$out_root),
    seeds = seq_len(10L),
    k_folds = 5L,
    write_outputs = TRUE,
    return_objects = FALSE,
    log_to_console = TRUE) {

  metadata_dir <- file.path(out_root, "metadata")
  jobs_dir <- file.path(out_root, "jobs")
  logs_dir <- file.path(out_root, "logs")

  if (isTRUE(write_outputs)) {
    dir.create(metadata_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(jobs_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
  }

  log_file <- if (isTRUE(write_outputs)) file.path(logs_dir, "AGDD_CV_Metadata.log") else NULL
  log_message <- make_metadata_logger(log_file = log_file, emit = log_to_console)

  log_message("Starting metadata build")
  log_message("data_path = ", data_path)
  log_message("out_root = ", out_root)

  env_file <- file.path(data_path, "Env_Names_G2F_2020_2021.csv")
  envs <- if (file.exists(env_file)) read.csv(env_file)$Env else sort(unique(load_pheno(data_path)$Env))

  pheno <- load_pheno(data_path)
  total_envs <- length(unique(envs))

  common_females <- pheno %>%
    distinct(Env, Female) %>%
    filter(!is.na(Female)) %>%
    add_count(Female, name = "n_envs") %>%
    filter(n_envs == total_envs) %>%
    pull(Female) %>%
    unique() %>%
    sort()

  if (length(common_females) == 0L) {
    stop("No common females were identified across all environments.")
  }

  # The finalized archived phenotype input yields the same canonical set used
  # by DAP and TNP. Fail early if an incompatible phenotype version is supplied.
  canonical_file <- get_path("G2F_COMMON_FEMALES_FILE", cv_paths$common_females_file)
  canonical <- sort(trimws(as.character(fread(canonical_file)$Female)))
  if (length(canonical) != 223L || anyDuplicated(canonical) ||
      !identical(sort(common_females), canonical)) {
    stop("AGDD common females do not match the canonical 223-female cohort.")
  }

  female_folds <- do.call(
    rbind,
    lapply(seeds, make_fold_map, common_females = common_females, k_folds = k_folds)
  )

  model_names <- c(
    "M1.G", "M1.P",
    "M2.G",
    "M3.G", "M3.P",
    "M4.G", "M4.P",
    "M5.G", "M5.P",
    "M6.G.P",
    "M7.G.P",
    "M8.G.P",
    "M9.G.P",
    "M10.G.P"
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
        sprintf("Seed%02d", Seed_Num),
        sprintf("Fold%d", Fold_Num),
        Split_Group,
        sanitize_file_component(Heldout_Env),
        sep = "."
      ),
      Weather_Regime = ifelse(Split_Group == "CV_2_1", "ALL", "LOEO"),
      Metric_Train = ifelse(Split_Group == "CV_2_1", "CV2", "CV0"),
      Metric_Test = ifelse(Split_Group == "CV_2_1", "CV1", "CV00")
    ) %>%
    arrange(Split_Group, Heldout_Env, Seed_Num, Fold_Num)

  summary_df <- data.frame(
    Common_Females = length(common_females),
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

    fwrite(
      data.frame(Female = common_females, stringsAsFactors = FALSE),
      file.path(metadata_dir, "common_females.csv")
    )

    fwrite(split_jobs, file.path(metadata_dir, "split_jobs.csv"))
    saveRDS(split_jobs, file.path(metadata_dir, "split_jobs.rds"))

    fwrite(
      split_jobs[, c("Seed_Num", "Fold_Num", "Split_Group", "Heldout_Env")],
      file.path(jobs_dir, "AGDD_CV_Setup_Jobs.txt"),
      sep = " ",
      col.names = FALSE
    )

    fwrite(
      split_jobs[, c("Seed_Num", "Fold_Num", "Split_Group", "Heldout_Env")],
      file.path(jobs_dir, "AGDD_CV_Prediction_Jobs.txt"),
      sep = " ",
      col.names = FALSE
    )

    fwrite(
      data.frame(Model_Name = model_names, stringsAsFactors = FALSE),
      file.path(metadata_dir, "model_names.csv")
    )

    fwrite(summary_df, file.path(metadata_dir, "metadata_summary.csv"))
  }

  log_message("Completed metadata build with ", nrow(split_jobs), " split jobs")
  print(summary_df)

  result <- list(
    summary_df = summary_df,
    pheno = pheno,
    envs = envs,
    common_females = common_females,
    female_folds = female_folds,
    split_jobs = split_jobs,
    model_names = model_names,
    data_path = data_path,
    out_root = out_root
  )

  if (isTRUE(return_objects)) result else invisible(summary_df)
}

main <- function() {
  run_cv_metadata()
}

if (sys.nframe() == 0L) {
  main()
}
