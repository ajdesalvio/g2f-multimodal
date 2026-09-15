suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "05_prediction_cv", "path_helpers.R"))
cv_paths <- g2f_cv_paths("agdd")

get_path <- function(env, default) {
  val <- Sys.getenv(env, unset = "")
  if (nzchar(val)) normalizePath(val, winslash = "/", mustWork = FALSE) else default
}

parse_bool_env <- function(env, default = FALSE) {
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

sanitize_file_component <- function(x) {
  gsub("[^A-Za-z0-9._-]", "_", x)
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

read_job_file <- function(file_path) {
  if (!file.exists(file_path)) {
    return(NULL)
  }

  job_df <- fread(file_path, header = FALSE, sep = " ", fill = TRUE) %>%
    as.data.frame()

  if (ncol(job_df) < 4L) {
    stop("Job file does not contain four columns: ", file_path)
  }

  job_df <- job_df[, seq_len(4L), drop = FALSE]
  names(job_df) <- c("Seed_Num", "Fold_Num", "Split_Group", "Heldout_Env")
  job_df$Seed_Num <- as.integer(job_df$Seed_Num)
  job_df$Fold_Num <- as.integer(job_df$Fold_Num)
  job_df$Split_ID <- make_split_id(
    seed_num = job_df$Seed_Num,
    fold_num = job_df$Fold_Num,
    split_group = job_df$Split_Group,
    heldout_env = job_df$Heldout_Env
  )
  job_df
}

load_expected_jobs <- function(jobs_dir) {
  split_files <- c(
    file.path(jobs_dir, "AGDD_CV_Prediction_Jobs_CV_2_1.txt"),
    file.path(jobs_dir, "AGDD_CV_Prediction_Jobs_CV_0_00.txt")
  )

  job_list <- Filter(Negate(is.null), lapply(split_files, read_job_file))
  if (length(job_list) > 0L) {
    return(bind_rows(job_list) %>% distinct(Split_ID, .keep_all = TRUE))
  }

  combined_file <- file.path(jobs_dir, "AGDD_CV_Prediction_Jobs.txt")
  job_df <- read_job_file(combined_file)
  if (is.null(job_df)) {
    stop("Could not find prediction job tables in ", jobs_dir)
  }
  job_df %>% distinct(Split_ID, .keep_all = TRUE)
}

empty_inspection <- function(split_id, output_split_id, csv_file, rds_file) {
  data.frame(
    Split_ID = split_id,
    Output_Split_ID = output_split_id,
    Metrics_File = normalizePath(csv_file, winslash = "/", mustWork = FALSE),
    Metrics_RDS_File = normalizePath(rds_file, winslash = "/", mustWork = FALSE),
    Metrics_CSV_Present = file.exists(csv_file),
    Metrics_RDS_Present = file.exists(rds_file),
    CSV_Readable = FALSE,
    RDS_Readable = FALSE,
    Rows = NA_integer_,
    Models = NA_integer_,
    Metrics = NA_character_,
    Expected_Metrics = NA_character_,
    Split_ID_OK = FALSE,
    Include_Ze_OK = FALSE,
    Model_Metric_Coverage_OK = FALSE,
    All_Status_Ok = FALSE,
    Any_Error_Status = NA,
    Any_Missing_Cor = NA,
    Inspection_Error = NA_character_,
    stringsAsFactors = FALSE
  )
}

inspect_expected_output <- function(
    split_id,
    output_split_id,
    split_group,
    results_dir,
    include_ze,
    expected_models) {
  csv_file <- file.path(results_dir, paste0(output_split_id, ".metrics.csv"))
  rds_file <- file.path(results_dir, paste0(output_split_id, ".metrics.rds"))
  ans <- empty_inspection(split_id, output_split_id, csv_file, rds_file)

  expected_metrics <- if (identical(split_group, "CV_2_1")) {
    c("CV2", "CV1")
  } else if (identical(split_group, "CV_0_00")) {
    c("CV0", "CV00")
  } else {
    character(0)
  }
  ans$Expected_Metrics <- paste(expected_metrics, collapse = ";")

  if (!ans$Metrics_CSV_Present) {
    return(ans)
  }

  metrics_df <- tryCatch(
    fread(csv_file) %>% as.data.frame(),
    error = function(e) e
  )
  if (inherits(metrics_df, "error")) {
    ans$Inspection_Error <- paste("CSV:", conditionMessage(metrics_df))
    return(ans)
  }

  ans$CSV_Readable <- TRUE
  ans$Rows <- nrow(metrics_df)
  ans$Models <- if ("Model_Name" %in% names(metrics_df)) {
    dplyr::n_distinct(metrics_df$Model_Name)
  } else {
    0L
  }
  ans$Metrics <- if ("Metric" %in% names(metrics_df)) {
    paste(sort(unique(metrics_df$Metric)), collapse = ";")
  } else {
    ""
  }
  ans$Split_ID_OK <- "Split_ID" %in% names(metrics_df) &&
    nrow(metrics_df) > 0L &&
    all(metrics_df$Split_ID == split_id)
  ans$Include_Ze_OK <- "Include_Ze" %in% names(metrics_df) &&
    nrow(metrics_df) > 0L &&
    all(!is.na(metrics_df$Include_Ze)) &&
    all(as.logical(metrics_df$Include_Ze) == include_ze)
  ans$All_Status_Ok <- "Status" %in% names(metrics_df) &&
    nrow(metrics_df) > 0L &&
    all(metrics_df$Status == "ok")
  ans$Any_Error_Status <- if ("Status" %in% names(metrics_df)) {
    any(is.na(metrics_df$Status) | metrics_df$Status != "ok")
  } else {
    TRUE
  }
  ans$Any_Missing_Cor <- if ("Cor" %in% names(metrics_df)) {
    any(is.na(metrics_df$Cor))
  } else {
    TRUE
  }

  if (all(c("Model_Name", "Metric") %in% names(metrics_df))) {
    coverage <- metrics_df %>% count(Model_Name, Metric, name = "N")
    expected_grid <- expand.grid(
      Model_Name = unique(metrics_df$Model_Name),
      Metric = expected_metrics,
      stringsAsFactors = FALSE
    )
    coverage_check <- expected_grid %>%
      left_join(coverage, by = c("Model_Name", "Metric"))
    ans$Model_Metric_Coverage_OK <-
      ans$Models == expected_models &&
      nrow(metrics_df) == expected_models * length(expected_metrics) &&
      nrow(coverage_check) == expected_models * length(expected_metrics) &&
      all(!is.na(coverage_check$N) & coverage_check$N == 1L) &&
      setequal(unique(metrics_df$Metric), expected_metrics)
  }

  if (ans$Metrics_RDS_Present) {
    metrics_rds <- tryCatch(readRDS(rds_file), error = function(e) e)
    if (inherits(metrics_rds, "error")) {
      rds_error <- paste("RDS:", conditionMessage(metrics_rds))
      ans$Inspection_Error <- if (is.na(ans$Inspection_Error)) {
        rds_error
      } else {
        paste(ans$Inspection_Error, rds_error, sep = " | ")
      }
    } else {
      ans$RDS_Readable <- is.data.frame(metrics_rds) && nrow(metrics_rds) == nrow(metrics_df)
    }
  }

  ans
}

make_problem_reason <- function(df) {
  reasons <- character(0)
  if (!df$Metrics_CSV_Present) reasons <- c(reasons, "missing CSV")
  if (!df$Metrics_RDS_Present) reasons <- c(reasons, "missing RDS")
  if (df$Metrics_CSV_Present && !df$CSV_Readable) reasons <- c(reasons, "unreadable CSV")
  if (df$Metrics_RDS_Present && !df$RDS_Readable) reasons <- c(reasons, "unreadable/inconsistent RDS")
  if (df$CSV_Readable && !df$Split_ID_OK) reasons <- c(reasons, "Split_ID mismatch")
  if (df$CSV_Readable && !df$Include_Ze_OK) reasons <- c(reasons, "Include_Ze mismatch")
  if (df$CSV_Readable && !df$Model_Metric_Coverage_OK) reasons <- c(reasons, "incomplete model/metric coverage")
  if (df$CSV_Readable && !df$All_Status_Ok) reasons <- c(reasons, "non-ok model status")
  if (df$CSV_Readable && isTRUE(df$Any_Missing_Cor)) reasons <- c(reasons, "missing correlation")
  if (!is.na(df$Inspection_Error)) reasons <- c(reasons, df$Inspection_Error)
  paste(unique(reasons), collapse = "; ")
}

write_restart_job_file <- function(df, file_path) {
  restart_jobs <- df %>%
    select(Seed_Num, Fold_Num, Split_Group, Heldout_Env)
  fwrite(restart_jobs, file_path, sep = " ", col.names = FALSE)
}

run_prediction_results_audit <- function(
    out_root = get_path("G2F_CV_OUT_PATH", cv_paths$out_root),
    include_ze = parse_bool_env("G2F_INCLUDE_ZE", default = FALSE),
    expected_models = 14L,
    write_outputs = TRUE,
    return_objects = FALSE) {
  jobs_dir <- file.path(out_root, "jobs")
  results_dir <- file.path(out_root, "results")
  audit_dir <- file.path(out_root, "audits")
  variant <- if (isTRUE(include_ze)) "withZe" else "noZe"
  output_suffix <- if (isTRUE(include_ze)) "" else ".noZe"
  output_prefix <- paste0("prediction_results_", variant)

  if (isTRUE(write_outputs)) {
    dir.create(audit_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(jobs_dir, recursive = TRUE, showWarnings = FALSE)
  }

  expected_jobs <- load_expected_jobs(jobs_dir) %>%
    arrange(Split_Group, Heldout_Env, Seed_Num, Fold_Num) %>%
    mutate(Output_Split_ID = paste0(Split_ID, output_suffix))

  inspection_df <- bind_rows(lapply(seq_len(nrow(expected_jobs)), function(i) {
    inspect_expected_output(
      split_id = expected_jobs$Split_ID[[i]],
      output_split_id = expected_jobs$Output_Split_ID[[i]],
      split_group = expected_jobs$Split_Group[[i]],
      results_dir = results_dir,
      include_ze = include_ze,
      expected_models = expected_models
    )
  }))

  summary_df <- expected_jobs %>%
    left_join(inspection_df, by = c("Split_ID", "Output_Split_ID")) %>%
    mutate(
      Complete_Output = Metrics_CSV_Present & Metrics_RDS_Present &
        CSV_Readable & RDS_Readable & Split_ID_OK & Include_Ze_OK &
        Model_Metric_Coverage_OK & All_Status_Ok & !Any_Missing_Cor
    )
  summary_df$Problem_Reason <- vapply(
    seq_len(nrow(summary_df)),
    function(i) make_problem_reason(summary_df[i, , drop = FALSE]),
    FUN.VALUE = character(1)
  )

  remaining_df <- summary_df %>% filter(!Complete_Output)
  problem_df <- remaining_df %>%
    filter(Metrics_CSV_Present | Metrics_RDS_Present)

  counts_df <- summary_df %>%
    group_by(Split_Group) %>%
    summarise(
      Expected_Splits = n(),
      CSV_Present = sum(Metrics_CSV_Present),
      RDS_Present = sum(Metrics_RDS_Present),
      Complete_Splits = sum(Complete_Output),
      Remaining_Splits = sum(!Complete_Output),
      .groups = "drop"
    )
  counts_df <- bind_rows(
    counts_df,
    counts_df %>%
      summarise(
        Split_Group = "TOTAL",
        across(where(is.numeric), sum)
      )
  )

  all_metrics_files <- list.files(
    results_dir,
    pattern = "\\.metrics\\.csv$",
    full.names = FALSE
  )
  is_noze_file <- grepl("\\.noZe\\.metrics\\.csv$", all_metrics_files)
  observed_files <- if (isTRUE(include_ze)) {
    all_metrics_files[!is_noze_file]
  } else {
    all_metrics_files[is_noze_file]
  }
  observed_ids <- sub("\\.metrics\\.csv$", "", observed_files)
  unexpected_df <- data.frame(
    Output_Split_ID = setdiff(observed_ids, expected_jobs$Output_Split_ID),
    stringsAsFactors = FALSE
  )

  restart_files <- c(
    CV_2_1 = file.path(jobs_dir, paste0("AGDD_CV_Prediction_Jobs_CV_2_1_", variant, "_Missing_Only.txt")),
    CV_0_00 = file.path(jobs_dir, paste0("AGDD_CV_Prediction_Jobs_CV_0_00_", variant, "_Missing_Only.txt"))
  )

  if (isTRUE(write_outputs)) {
    fwrite(summary_df, file.path(audit_dir, paste0(output_prefix, "_audit_summary.csv")))
    fwrite(remaining_df, file.path(audit_dir, paste0(output_prefix, "_remaining.csv")))
    fwrite(problem_df, file.path(audit_dir, paste0(output_prefix, "_problems.csv")))
    fwrite(counts_df, file.path(audit_dir, paste0(output_prefix, "_counts.csv")))
    fwrite(unexpected_df, file.path(audit_dir, paste0(output_prefix, "_unexpected.csv")))
    write_restart_job_file(remaining_df %>% filter(Split_Group == "CV_2_1"), restart_files[["CV_2_1"]])
    write_restart_job_file(remaining_df %>% filter(Split_Group == "CV_0_00"), restart_files[["CV_0_00"]])
  }

  message("Audited prediction variant: ", variant)
  print(counts_df)
  if (nrow(remaining_df) > 0L) {
    print(remaining_df %>% select(Output_Split_ID, Split_Group, Problem_Reason))
  } else {
    message("All expected ", variant, " prediction outputs are complete.")
  }
  if (isTRUE(write_outputs)) {
    message("Restart job tables:")
    message("  ", restart_files[["CV_2_1"]])
    message("  ", restart_files[["CV_0_00"]])
  }

  result <- list(
    summary_df = summary_df,
    remaining_df = remaining_df,
    problem_df = problem_df,
    counts_df = counts_df,
    unexpected_df = unexpected_df,
    restart_files = restart_files,
    include_ze = include_ze,
    jobs_dir = jobs_dir,
    results_dir = results_dir,
    audit_dir = audit_dir
  )

  if (isTRUE(return_objects)) result else invisible(summary_df)
}

main <- function() {
  run_prediction_results_audit()
}

if (sys.nframe() == 0L) {
  main()
}
