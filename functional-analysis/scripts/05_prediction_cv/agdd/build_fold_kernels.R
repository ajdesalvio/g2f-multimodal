suppressPackageStartupMessages({
  library(dplyr)
  library(data.table)
  library(tidyr)
  library(fdapace)
})

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "05_prediction_cv", "path_helpers.R"))
cv_paths <- g2f_cv_paths("agdd")

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

format_time_key <- function(x) {
  x_num <- suppressWarnings(as.numeric(x))
  out <- ifelse(
    is.finite(x_num),
    format(x_num, scientific = FALSE, trim = TRUE, digits = 15),
    as.character(x)
  )
  trimws(out)
}

safe_scale_matrix <- function(x) {
  x <- as.matrix(x)
  if (!is.numeric(x)) {
    storage.mode(x) <- "double"
  }

  centers <- colMeans(x, na.rm = TRUE)
  sds <- apply(x, 2, sd, na.rm = TRUE)

  centers[!is.finite(centers)] <- 0
  sds[!is.finite(sds) | sds == 0] <- 1

  out <- sweep(x, 2, centers, "-")
  out <- sweep(out, 2, sds, "/")
  out[!is.finite(out)] <- 0
  out
}

clip_fpca_lists <- function(Ly, Lt, lo, hi) {
  out <- Map(function(y, t) {
    keep <- is.finite(y) & is.finite(t) & t >= lo & t <= hi
    list(y = y[keep], t = t[keep])
  }, Ly, Lt)

  list(
    Ly = lapply(out, `[[`, "y"),
    Lt = lapply(out, `[[`, "t")
  )
}

score_matrix_from_object <- function(score_obj, ids, feature_stub, n_fpcs) {
  ids <- as.character(ids)
  out <- matrix(0, nrow = length(ids), ncol = n_fpcs)

  if (!is.null(score_obj) && length(score_obj) > 0L) {
    score_mat <- as.matrix(score_obj)
    keep_n <- min(ncol(score_mat), n_fpcs)
    out[, seq_len(keep_n)] <- score_mat[, seq_len(keep_n), drop = FALSE]
  }

  rownames(out) <- ids
  colnames(out) <- paste0("FPC", seq_len(n_fpcs), ".", feature_stub)
  out
}

collapse_duplicate_vi_times <- function(VI) {
  VI %>%
    group_by(Pedigree, Env, Year, Vegetation.Index, DAP) %>%
    summarize(
      VI.BLUE = mean(VI.BLUE, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(Pedigree.Env = paste(Pedigree, Env, sep = "."))
}

build_vi_score_matrix <- function(vi_name, VI, row_data, order, train_ids, dap_sorted, n_vi_fpcs) {
  template <- row_data[, c("Pedigree.Env", "Pedigree", "Env"), drop = FALSE]
  dap_labels <- format_time_key(dap_sorted)

  VI_i <- VI %>%
    filter(Vegetation.Index == vi_name, Pedigree.Env %in% order) %>%
    mutate(Time_Label = format_time_key(DAP)) %>%
    select(Pedigree.Env, Time_Label, VI.BLUE) %>%
    group_by(Pedigree.Env, Time_Label) %>%
    summarize(VI.BLUE = mean(VI.BLUE, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(
      id_cols = Pedigree.Env,
      names_from = Time_Label,
      values_from = VI.BLUE,
      names_glue = "VI.BLUE.{Time_Label}",
      values_fill = NA
    ) %>%
    as.data.frame()

  VI_i <- template %>%
    left_join(VI_i, by = "Pedigree.Env") %>%
    arrange(match(Pedigree.Env, order))

  dap_cols <- paste0("VI.BLUE.", dap_labels)
  missing_dap_cols <- setdiff(dap_cols, names(VI_i))
  if (length(missing_dap_cols) > 0L) {
    for (col_i in missing_dap_cols) {
      VI_i[[col_i]] <- NA_real_
    }
  }

  VI_i <- VI_i[, c("Pedigree", "Env", "Pedigree.Env", dap_cols), drop = FALSE]

  fpca_input <- MakeFPCAInputs(
    IDs = rep(VI_i$Pedigree.Env, each = length(dap_sorted)),
    tVec = rep(dap_sorted, length(VI_i$Pedigree.Env)),
    t(VI_i[, dap_cols, drop = FALSE])
  )

  idx_train <- fpca_input$Lid %in% train_ids
  idx_hold <- !idx_train

  if (!any(idx_train)) {
    stop("No FPCA training observations were found for VI ", vi_name)
  }

  fpca_obj <- FPCA(
    fpca_input$Ly[idx_train],
    fpca_input$Lt[idx_train],
    list(
      dataType = "Sparse",
      methodMuCovEst = "smooth",
      methodBwCov = "GCV",
      methodBwMu = "GCV",
      plot = FALSE
    )
  )

  train_id_order <- unique(fpca_input$Lid[idx_train])
  score_train <- score_matrix_from_object(fpca_obj$xiEst, train_id_order, vi_name, n_vi_fpcs)

  hold_id_order <- unique(fpca_input$Lid[idx_hold])
  if (length(hold_id_order) > 0L) {
    lo <- min(fpca_obj$workGrid)
    hi <- max(fpca_obj$workGrid)
    held <- clip_fpca_lists(fpca_input$Ly[idx_hold], fpca_input$Lt[idx_hold], lo, hi)
    pred <- predict(fpca_obj, newLy = held$Ly, newLt = held$Lt)
    score_hold <- score_matrix_from_object(pred$scores, hold_id_order, vi_name, n_vi_fpcs)
  } else {
    score_hold <- score_matrix_from_object(NULL, character(0), vi_name, n_vi_fpcs)
  }

  score_all <- rbind(score_train, score_hold)
  score_all <- score_all[order, , drop = FALSE]

  fve <- fpca_obj$cumFVE
  retained_fve <- rep(NA_real_, n_vi_fpcs)
  if (length(fve) > 0L) {
    keep_n <- min(length(fve), n_vi_fpcs)
    retained_fve[1] <- fve[1]
    if (keep_n > 1L) {
      retained_fve[2:keep_n] <- diff(fve[seq_len(keep_n)])
    }
  }

  list(
    scores = score_all,
    summary = data.frame(
      Vegetation.Index = vi_name,
      Train_IDs = length(train_id_order),
      Projected_IDs = length(hold_id_order),
      Retained_FPCs = n_vi_fpcs,
      stringsAsFactors = FALSE
    ),
    fve = retained_fve
  )
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

make_setup_logger <- function(split_id, log_file = NULL, emit = TRUE) {
  function(...) {
    msg <- paste0(
      format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      " | pid=", Sys.getpid(),
      " | split=", split_id,
      " | ",
      paste(..., collapse = "")
    )
    if (isTRUE(emit)) {
      message(msg)
    }
    if (!is.null(log_file)) {
      cat(msg, "\n", file = log_file, append = TRUE)
    }
  }
}

run_cv_prediction_setup <- function(
    seed_num,
    fold_num,
    split_group,
    heldout_env,
    data_path = get_path("G2F_DATA_PATH", cv_paths$data_path),
    out_root = get_path("G2F_CV_OUT_PATH", cv_paths$out_root),
    derived_path = get_path("G2F_DERIVED_DATA_PATH", cv_paths$derived_path),
    vi_file = Sys.getenv("G2F_VI_FILE", unset = "VI_BLUEs_G2F_2020_2021_AGDD.csv"),
    vi_names = strsplit(Sys.getenv("G2F_VI_NAMES", unset = "NGRDI"), ",", fixed = TRUE)[[1]],
    n_vi_fpcs = suppressWarnings(as.integer(Sys.getenv("G2F_VI_NFPCS", unset = "5"))),
    save_vi_scores = identical(toupper(Sys.getenv("G2F_SAVE_VI_SCORES", unset = "FALSE")), "TRUE"),
    write_outputs = TRUE,
    return_objects = FALSE,
    log_to_console = TRUE) {
  seed_num <- as.integer(seed_num)
  fold_num <- as.integer(fold_num)

  if (is.na(seed_num) || is.na(fold_num)) {
    stop("seed_num and fold_num must be integers.")
  }
  if (!split_group %in% c("CV_2_1", "CV_0_00")) {
    stop("split_group must be one of: CV_2_1, CV_0_00")
  }
  if (is.na(n_vi_fpcs) || n_vi_fpcs < 1L) {
    stop("G2F_VI_NFPCS must be a positive integer.")
  }

  vi_names <- trimws(vi_names)
  vi_names <- vi_names[nzchar(vi_names)]

  set_single_thread()

  slurm_array_job_id <- Sys.getenv("SLURM_ARRAY_JOB_ID", unset = "")
  slurm_job_id <- Sys.getenv("SLURM_JOB_ID", unset = "local")
  slurm_task_id <- Sys.getenv("SLURM_ARRAY_TASK_ID", unset = "local")
  run_job_id <- if (nzchar(slurm_array_job_id)) slurm_array_job_id else slurm_job_id

  bundles_dir <- file.path(out_root, "bundles")
  metadata_dir <- file.path(out_root, "metadata")
  split_dir <- file.path(bundles_dir, "split_bundles")
  logs_dir <- file.path(out_root, "logs")
  summaries_dir <- file.path(out_root, "summaries")

  if (isTRUE(write_outputs)) {
    dir.create(split_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(summaries_dir, recursive = TRUE, showWarnings = FALSE)
  }

  split_id <- paste(
    sprintf("Seed%02d", seed_num),
    sprintf("Fold%d", fold_num),
    split_group,
    sanitize_file_component(heldout_env),
    sep = "."
  )

  log_file <- if (isTRUE(write_outputs)) {
    file.path(logs_dir, paste0("setup_", split_id, "_job_", run_job_id, "_task_", slurm_task_id, ".log"))
  } else {
    NULL
  }
  summary_file <- file.path(summaries_dir, paste0("setup_summary_", split_id, ".csv"))
  summary_rds_file <- file.path(summaries_dir, paste0("setup_summary_", split_id, ".rds"))
  log_message <- make_setup_logger(split_id = split_id, log_file = log_file, emit = log_to_console)

  log_message("Starting split setup")
  log_message("data_path = ", data_path)
  log_message("derived_path = ", derived_path)
  log_message("vi_file = ", vi_file)
  log_message("out_root = ", out_root)

  constant_bundle <- readRDS(file.path(bundles_dir, "constant_kernel_bundle.rds"))
  female_folds <- fread(file.path(metadata_dir, "female_folds.csv")) %>% as.data.frame()

  row_data <- constant_bundle$row_data %>%
    left_join(
      female_folds %>% filter(Seed_Num == seed_num) %>% select(Female, Fold),
      by = "Female"
    )

  envs <- constant_bundle$envs
  order <- constant_bundle$order

  if (split_group == "CV_0_00" && !heldout_env %in% envs) {
    stop("Held-out environment not found in constant bundle: ", heldout_env)
  }

  is_common <- !is.na(row_data$Fold)
  in_training <- is_common & row_data$Fold != fold_num
  if (split_group == "CV_0_00") {
    in_training <- in_training & row_data$Env != heldout_env
  }

  train_ids <- row_data$Pedigree.Env[in_training]
  if (length(train_ids) == 0L) {
    stop("No training Pedigree.Env entries were selected for split ", split_id)
  }

  log_message("Loading VI BLUEs")
  VI <- fread(file.path(derived_path, vi_file)) %>%
    as.data.frame() %>%
    collapse_duplicate_vi_times()

  available_vi_names <- sort(unique(VI$Vegetation.Index))
  available_vi_names <- available_vi_names[!grepl("new", available_vi_names, ignore.case = TRUE)]

  if (length(vi_names) == 0L) {
    vi_names <- available_vi_names
  } else {
    missing_vi_names <- setdiff(vi_names, available_vi_names)
    if (length(missing_vi_names) > 0L) {
      stop(
        "The following requested vegetation indices were not found in ", vi_file, ": ",
        paste(missing_vi_names, collapse = ", ")
      )
    }
    VI <- VI %>% filter(Vegetation.Index %in% vi_names)
  }

  dap_sorted <- sort(unique(VI$DAP))

  log_message("Selected VIs = ", paste(vi_names, collapse = ","))
  log_message("Running VI FPCA projections for ", length(vi_names), " vegetation indices")
  vi_results <- lapply(vi_names, function(vi_name) {
    log_message("FPCA setup for VI = ", vi_name)
    build_vi_score_matrix(
      vi_name = vi_name,
      VI = VI,
      row_data = row_data,
      order = order,
      train_ids = train_ids,
      dap_sorted = dap_sorted,
      n_vi_fpcs = n_vi_fpcs
    )
  })

  vi_score_mat <- do.call(cbind, lapply(vi_results, `[[`, "scores"))
  rownames(vi_score_mat) <- order

  log_message("Building phenomic kernels")
  VI_all <- do.call(
    rbind,
    lapply(envs, function(env_i) {
      idx <- which(row_data$Env == env_i)
      temp <- safe_scale_matrix(vi_score_mat[idx, , drop = FALSE])
      rownames(temp) <- row_data$Pedigree.Env[idx]
      temp
    })
  )
  VI_all <- VI_all[order, , drop = FALSE]

  KP <- tcrossprod(VI_all) / ncol(VI_all)
  KP_PE <- constant_bundle$left_add_base * KP
  rownames(KP) <- order
  colnames(KP) <- order
  rownames(KP_PE) <- order
  colnames(KP_PE) <- order

  weather_bundle_file <- if (split_group == "CV_2_1") {
    file.path(bundles_dir, "weather", "weather_all_bundle.rds")
  } else {
    file.path(
      bundles_dir,
      "weather",
      "loeo",
      paste0("weather_", sanitize_file_component(heldout_env), ".rds")
    )
  }

  log_message("Loading weather bundle ", weather_bundle_file)
  weather_bundle <- readRDS(weather_bundle_file)

  KE_W <- constant_bundle$Ze %*% weather_bundle$K_W %*% t(constant_bundle$Ze)
  rownames(KE_W) <- order
  colnames(KE_W) <- order

  KP_PW <- KE_W * KP
  rownames(KP_PW) <- order
  colnames(KP_PW) <- order

  log_message("Running eigendecomposition on split-specific kernels")
  dynamic_eigs <- list(
    KP.eig = eigen(KP, symmetric = TRUE),
    KP_PE.eig = eigen(KP_PE, symmetric = TRUE),
    KP_PW.eig = eigen(KP_PW, symmetric = TRUE)
  )

  vi_summary <- bind_rows(lapply(vi_results, `[[`, "summary"))
  vi_fve <- do.call(
    rbind,
    lapply(seq_along(vi_names), function(i) {
      data.frame(
        Vegetation.Index = vi_names[i],
        FPC = seq_len(n_vi_fpcs),
        FVE = vi_results[[i]]$fve,
        stringsAsFactors = FALSE
      )
    })
  )

  row_cols <- unique(c(
    "Pedigree.Env", "Pedigree", "Env", "Female", "Fold",
    intersect(c("Yield.t.ha.BLUE", "Yield"), names(row_data))
  ))

  split_bundle <- list(
    split_info = list(
      split_id = split_id,
      seed_num = seed_num,
      fold_num = fold_num,
      split_group = split_group,
      heldout_env = heldout_env,
      weather_bundle_file = weather_bundle_file
    ),
    row_data = row_data[, row_cols, drop = FALSE],
    dynamic_eigs = dynamic_eigs,
    vi_summary = vi_summary,
    vi_fve = vi_fve,
    train_ids = train_ids
  )

  split_file <- file.path(split_dir, paste0(split_id, ".rds"))

  if (isTRUE(write_outputs)) {
    saveRDS(split_bundle, split_file)

    if (isTRUE(save_vi_scores)) {
      score_dir <- file.path(split_dir, "debug_vi_scores")
      dir.create(score_dir, recursive = TRUE, showWarnings = FALSE)
      vi_score_df <- data.frame(Pedigree.Env = order, vi_score_mat, check.names = FALSE)
      saveRDS(vi_score_df, file.path(score_dir, paste0(split_id, ".vi_scores.rds")))
    }
  }

  summary_df <- data.frame(
    Split_ID = split_id,
    Seed_Num = seed_num,
    Fold_Num = fold_num,
    Split_Group = split_group,
    Heldout_Env = heldout_env,
    Training_IDs = length(train_ids),
    Vegetation_Indices = length(vi_names),
    VI_Names = paste(vi_names, collapse = ";"),
    VI_FPCs_Retained = n_vi_fpcs,
    Split_File = split_file,
    Weather_Bundle = weather_bundle_file,
    stringsAsFactors = FALSE
  )

  if (isTRUE(write_outputs)) {
    fwrite(summary_df, summary_file)
    saveRDS(summary_df, summary_rds_file)
  }

  log_message("Completed split setup")
  print(summary_df)

  result <- list(
    summary_df = summary_df,
    split_bundle = split_bundle,
    constant_bundle = constant_bundle,
    female_folds = female_folds,
    row_data = row_data,
    train_ids = train_ids,
    VI = VI,
    vi_file = vi_file,
    vi_names = vi_names,
    vi_summary = vi_summary,
    vi_fve = vi_fve,
    dap_sorted = dap_sorted,
    vi_results = vi_results,
    vi_score_mat = vi_score_mat,
    VI_all = VI_all,
    KP = KP,
    KP_PE = KP_PE,
    weather_bundle = weather_bundle,
    KE_W = KE_W,
    KP_PW = KP_PW,
    dynamic_eigs = dynamic_eigs,
    split_id = split_id,
    data_path = data_path,
    out_root = out_root
  )

  if (isTRUE(return_objects)) result else invisible(summary_df)
}

parse_setup_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  args <- vapply(args, sanitize_arg, FUN.VALUE = character(1))
  if (length(args) != 4L) {
    stop("Expected exactly 4 arguments: seed_num fold_num split_group heldout_env")
  }

  list(
    seed_num = as.integer(args[1]),
    fold_num = as.integer(args[2]),
    split_group = args[3],
    heldout_env = args[4]
  )
}

main <- function() {
  parsed <- parse_setup_args()

  tryCatch(
    run_cv_prediction_setup(
      seed_num = parsed$seed_num,
      fold_num = parsed$fold_num,
      split_group = parsed$split_group,
      heldout_env = parsed$heldout_env
    ),
    error = function(e) {
      split_id <- paste(
        sprintf("Seed%02d", parsed$seed_num),
        sprintf("Fold%d", parsed$fold_num),
        parsed$split_group,
        sanitize_file_component(parsed$heldout_env),
        sep = "."
      )
      msg <- paste0(
        format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        " | pid=", Sys.getpid(),
        " | split=", split_id,
        " | FAILED: ",
        conditionMessage(e)
      )
      message(msg)
      stop(e)
    }
  )
}

if (sys.nframe() == 0L) {
  main()
}
