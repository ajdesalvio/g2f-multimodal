suppressPackageStartupMessages({
  library(BGLR)
  library(dplyr)
  library(data.table)
})

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "05_prediction_cv", "path_helpers.R"))
cv_paths <- g2f_cv_paths("dap")

assert_single_predictor_bglr_patch <- function() {
  helper <- tryCatch(
    getFromNamespace("setLT.RKHS", "BGLR"),
    error = function(error) NULL
  )
  # Inspect the affected assignment, not unrelated drop=FALSE expressions.
  target_subsets <- list()
  inspect <- function(expr) {
    if (!is.call(expr)) return(invisible(NULL))
    if (
      is.symbol(expr[[1L]]) &&
      as.character(expr[[1L]]) %in% c("=", "<-") &&
      length(expr) == 3L && identical(expr[[2L]], quote(LT$V))
    ) {
      rhs <- expr[[3L]]
      if (
        is.call(rhs) && identical(rhs[[1L]], as.name("[")) &&
        length(rhs) >= 4L && identical(rhs[[2L]], quote(LT$V)) &&
        identical(rhs[[4L]], as.name("tmp"))
      ) {
        target_subsets[[length(target_subsets) + 1L]] <<- rhs
      }
    }
    for (index in seq_along(expr)[-1L]) {
      if (is.call(expr[[index]])) inspect(expr[[index]])
    }
    invisible(NULL)
  }
  if (is.function(helper)) inspect(body(helper))
  expected <- quote(LT$V[, tmp, drop = FALSE])
  patched <- length(target_subsets) > 0L &&
    all(vapply(target_subsets, identical, logical(1), y = expected))
  if (!patched) {
    stop(
      "The installed BGLR setLT.RKHS() does not contain the verified ",
      "LT$V[, tmp, drop = FALSE] patch. See patches/README.md; ",
      "use the patched package from the final HPRC environment."
    )
  }
  invisible(TRUE)
}

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

# Produce a stable positive integer seed from the complete prediction identity.
# A small rolling hash avoids depending on optional packages and keeps the seed
# unchanged when a filtered subset of models is rerun.
derive_bglr_seed <- function(
    seed_num,
    fold_num,
    split_group,
    heldout_env,
    model_name,
    include_ze,
    namespace = "G2F_CV_BGLR_V1") {
  key <- paste(
    namespace,
    as.integer(seed_num),
    as.integer(fold_num),
    split_group,
    heldout_env,
    model_name,
    isTRUE(include_ze),
    sep = "|"
  )

  hash <- 104729
  modulus <- 2147483646
  for (byte in utf8ToInt(enc2utf8(key))) {
    hash <- (hash * 31 + byte) %% modulus
  }

  as.integer(hash + 1)
}

eta_from_eig <- function(eig_obj) {
  list(V = eig_obj$vectors, d = eig_obj$values, model = "RKHS")
}

mask_yields <- function(df, split_group, fold_num, heldout_env) {
  fold_match <- !is.na(df$Fold) & df$Fold == fold_num
  env_match <- as.character(df$Env) == as.character(heldout_env)

  if (split_group == "CV_0_00") {
    mask <- fold_match | env_match
  } else {
    mask <- fold_match
  }

  mask[is.na(mask)] <- FALSE
  mask
}

evaluate_metrics <- function(pred_df, split_group, fold_num, heldout_env) {
  common_rows <- !is.na(pred_df$Fold)

  metric_sets <- if (split_group == "CV_2_1") {
    list(
      CV2 = common_rows & pred_df$Fold != fold_num,
      CV1 = common_rows & pred_df$Fold == fold_num
    )
  } else {
    list(
      CV0 = common_rows & pred_df$Env == heldout_env & pred_df$Fold != fold_num,
      CV00 = common_rows & pred_df$Env == heldout_env & pred_df$Fold == fold_num
    )
  }

  bind_rows(lapply(names(metric_sets), function(metric_name) {
    idx <- metric_sets[[metric_name]] & complete.cases(pred_df$Actual, pred_df$Predicted)
    data.frame(
      Metric = metric_name,
      N = sum(idx, na.rm = TRUE),
      Cor = if (sum(idx, na.rm = TRUE) >= 2L) {
        cor(pred_df$Actual[idx], pred_df$Predicted[idx])
      } else {
        NA_real_
      },
      RMSE = if (any(idx)) sqrt(mean((pred_df$Actual[idx] - pred_df$Predicted[idx])^2)) else NA_real_,
      stringsAsFactors = FALSE
    )
  }))
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

make_prediction_logger <- function(split_id, log_file = NULL, emit = TRUE) {
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

run_cv_prediction <- function(
    seed_num,
    fold_num,
    split_group,
    heldout_env,
    out_root = get_path("G2F_CV_OUT_PATH", cv_paths$out_root),
    n_iter = suppressWarnings(as.integer(Sys.getenv("G2F_N_ITER", unset = "10000"))),
    burn_in = suppressWarnings(as.integer(Sys.getenv("G2F_BURN_IN", unset = "1000"))),
    save_pred_values = identical(toupper(Sys.getenv("G2F_SAVE_PRED_VALUES", unset = "TRUE")), "TRUE"),
    save_eval_only = parse_bool_env("G2F_SAVE_EVAL_ONLY", default = TRUE),
    model_filter = Sys.getenv("G2F_MODEL_FILTER", unset = ""),
    include_ze = parse_bool_env("G2F_INCLUDE_ZE", default = FALSE),
    write_outputs = TRUE,
    return_objects = FALSE,
    return_fits = FALSE,
    log_to_console = TRUE) {
  seed_num <- as.integer(seed_num)
  fold_num <- as.integer(fold_num)

  if (is.na(seed_num) || is.na(fold_num)) {
    stop("seed_num and fold_num must be integers.")
  }
  if (!split_group %in% c("CV_2_1", "CV_0_00")) {
    stop("split_group must be one of: CV_2_1, CV_0_00")
  }

  set_single_thread()
  assert_single_predictor_bglr_patch()

  slurm_job_id <- Sys.getenv("SLURM_JOB_ID", unset = "local")
  slurm_task_id <- Sys.getenv("SLURM_ARRAY_TASK_ID", unset = "local")

  bundles_dir <- file.path(out_root, "bundles")
  split_dir <- file.path(bundles_dir, "split_bundles")
  results_dir <- file.path(out_root, "results")
  pred_values_dir <- file.path(results_dir, "prediction_values")
  logs_dir <- file.path(out_root, "logs")
  wd_root <- if (isTRUE(write_outputs)) file.path(out_root, "working_directories") else file.path(tempdir(), "DAP_CV_debug_wd")

  if (isTRUE(write_outputs)) {
    dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(pred_values_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
  }
  dir.create(wd_root, recursive = TRUE, showWarnings = FALSE)

  split_id <- paste(
    sprintf("Seed%02d", seed_num),
    sprintf("Fold%d", fold_num),
    split_group,
    sanitize_file_component(heldout_env),
    sep = "."
  )
  output_split_id <- if (isTRUE(include_ze)) split_id else paste(split_id, "noZe", sep = ".")

  log_file <- if (isTRUE(write_outputs)) {
    file.path(logs_dir, paste0("prediction_", output_split_id, "_job_", slurm_job_id, "_task_", slurm_task_id, ".log"))
  } else {
    NULL
  }
  metrics_file <- file.path(results_dir, paste0(output_split_id, ".metrics.csv"))
  metrics_rds_file <- file.path(results_dir, paste0(output_split_id, ".metrics.rds"))
  log_message <- make_prediction_logger(split_id = output_split_id, log_file = log_file, emit = log_to_console)

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

  if (length(model_filter) == 1L && nzchar(model_filter)) {
    model_filter <- trimws(strsplit(model_filter, ",", fixed = TRUE)[[1]])
  }
  if (length(model_filter) > 0L && any(nzchar(model_filter))) {
    model_filter <- model_filter[nzchar(model_filter)]
    missing_models <- setdiff(model_filter, names(model_specs))
    if (length(missing_models) > 0L) {
      stop("Unknown model_filter value(s): ", paste(missing_models, collapse = ", "))
    }
    model_specs <- model_specs[model_filter]
  }
  if (!isTRUE(include_ze)) {
    model_specs <- lapply(model_specs, setdiff, "Ze")
  }

  constant_bundle <- readRDS(file.path(bundles_dir, "constant_kernel_bundle.rds"))
  split_bundle <- readRDS(file.path(split_dir, paste0(split_id, ".rds")))
  weather_bundle <- readRDS(split_bundle$split_info$weather_bundle_file)

  row_data <- split_bundle$row_data
  yield_col <- intersect(c("Yield.t.ha.BLUE", "Yield"), names(row_data))
  if (length(yield_col) == 0L) {
    stop("No yield column was found in the split bundle row data.")
  }
  yield_col <- yield_col[[1]]

  constant_eigs <- constant_bundle$eig_list_constant
  dynamic_eigs <- split_bundle$dynamic_eigs
  weather_eigs <- weather_bundle$eig_list

  eta_components <- list(
    KG_G_A = eta_from_eig(constant_eigs[["KG_G_A.eig"]]),
    KG_G_D = eta_from_eig(constant_eigs[["KG_G_D.eig"]]),
    KG_GE_A = eta_from_eig(constant_eigs[["KG_GE_A.eig"]]),
    KG_GE_D = eta_from_eig(constant_eigs[["KG_GE_D.eig"]]),
    KP = eta_from_eig(dynamic_eigs[["KP.eig"]]),
    KP_PE = eta_from_eig(dynamic_eigs[["KP_PE.eig"]]),
    KE_W = eta_from_eig(weather_eigs[["KE_W.eig"]]),
    KG_GE_AW = eta_from_eig(weather_eigs[["KG_GE_AW.eig"]]),
    KG_GE_DW = eta_from_eig(weather_eigs[["KG_GE_DW.eig"]]),
    KP_PW = eta_from_eig(dynamic_eigs[["KP_PW.eig"]]),
    Ze = list(X = constant_bundle$Ze, model = "BRR")
  )

  log_message("Include Ze environment fixed-effect BRR term = ", include_ze)
  log_message("Starting prediction run with ", length(model_specs), " models")
  metric_template <- if (split_group == "CV_2_1") c("CV2", "CV1") else c("CV0", "CV00")

  model_results <- lapply(names(model_specs), function(model_name) {
    wd_job <- file.path(wd_root, paste(output_split_id, sanitize_file_component(model_name), sep = "_"))
    if (isTRUE(write_outputs)) {
      dir.create(wd_job, recursive = TRUE, showWarnings = FALSE)
    }

    bglr_seed <- derive_bglr_seed(
      seed_num = seed_num,
      fold_num = fold_num,
      split_group = split_group,
      heldout_env = heldout_env,
      model_name = model_name,
      include_ze = include_ze
    )
    set.seed(
      bglr_seed,
      kind = "Mersenne-Twister",
      normal.kind = "Inversion",
      sample.kind = "Rejection"
    )

    log_message("Fitting model ", model_name, " | BGLR RNG seed = ", bglr_seed)

    tryCatch({
      eta_i <- lapply(model_specs[[model_name]], function(component_name) eta_components[[component_name]])

      y_vec <- row_data[[yield_col]]
      mask <- mask_yields(row_data, split_group, fold_num, heldout_env)
      y_vec[mask] <- NA_real_
      y_vec <- as.numeric(y_vec)

      fit <- BGLR(
        y = y_vec,
        ETA = eta_i,
        nIter = n_iter,
        burnIn = burn_in,
        saveAt = file.path(wd_job, "BGLR_")
      )

      pred_df <- data.frame(
        Pedigree.Env = row_data$Pedigree.Env,
        Pedigree = row_data$Pedigree,
        Env = row_data$Env,
        Female = row_data$Female,
        Fold = row_data$Fold,
        Actual = row_data[[yield_col]],
        Predicted = fit$yHat,
        stringsAsFactors = FALSE
      )

      if (isTRUE(save_pred_values) && isTRUE(write_outputs)) {
        pred_out <- pred_df
        if (isTRUE(save_eval_only)) {
          keep <- !is.na(pred_df$Fold)
          if (split_group == "CV_0_00") keep <- keep & pred_df$Env == heldout_env
          pred_out <- pred_df[keep, , drop = FALSE]
        }
        fwrite(
          pred_out,
          file.path(pred_values_dir, paste0(output_split_id, ".", model_name, ".prediction_values.csv"))
        )
      }

      metrics_i <- evaluate_metrics(pred_df, split_group, fold_num, heldout_env) %>%
        mutate(
          Split_ID = split_id,
          Seed_Num = seed_num,
          Fold_Num = fold_num,
          Split_Group = split_group,
          Heldout_Env = heldout_env,
          Model_Name = model_name,
          Include_Ze = include_ze,
          Status = "ok",
          Error = NA_character_
        ) %>%
        select(
          Split_ID, Seed_Num, Fold_Num, Split_Group, Heldout_Env,
          Model_Name, Include_Ze, Metric, N, Cor, RMSE, Status, Error
        )

      list(
        metrics = metrics_i,
        pred_df = if (isTRUE(return_objects)) pred_df else NULL,
        fit = if (isTRUE(return_fits)) fit else NULL,
        eta = if (isTRUE(return_objects)) eta_i else NULL
      )
    }, error = function(e) {
      log_message("Model failed: ", model_name, " | ", conditionMessage(e))
      list(
        metrics = data.frame(
          Split_ID = split_id,
          Seed_Num = seed_num,
          Fold_Num = fold_num,
          Split_Group = split_group,
          Heldout_Env = heldout_env,
          Model_Name = model_name,
          Include_Ze = include_ze,
          Metric = metric_template,
          N = NA_integer_,
          Cor = NA_real_,
          RMSE = NA_real_,
          Status = "error",
          Error = conditionMessage(e),
          stringsAsFactors = FALSE
        ),
        pred_df = NULL,
        fit = NULL,
        eta = NULL
      )
    })
  })
  names(model_results) <- names(model_specs)

  metrics_df <- bind_rows(lapply(model_results, `[[`, "metrics"))
  if (isTRUE(write_outputs)) {
    fwrite(metrics_df, metrics_file)
    saveRDS(metrics_df, metrics_rds_file)
  }

  log_message("Completed prediction run")
  print(metrics_df)

  result <- list(
    metrics_df = metrics_df,
    model_results = model_results,
    model_specs = model_specs,
    constant_bundle = constant_bundle,
    split_bundle = split_bundle,
    weather_bundle = weather_bundle,
    eta_components = eta_components,
    split_id = split_id,
    output_split_id = output_split_id,
    include_ze = include_ze,
    out_root = out_root
  )

  if (isTRUE(return_objects)) result else invisible(metrics_df)
}

parse_prediction_args <- function(args = commandArgs(trailingOnly = TRUE)) {
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
  parsed <- parse_prediction_args()

  tryCatch(
    run_cv_prediction(
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
