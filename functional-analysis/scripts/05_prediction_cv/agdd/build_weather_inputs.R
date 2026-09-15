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

scale_weather_score_matrix <- function(score_mat, envs, regime, heldout_env) {
  score_mat <- as.matrix(score_mat)
  if (!is.numeric(score_mat)) {
    storage.mode(score_mat) <- "double"
  }

  if (!identical(rownames(score_mat), envs)) {
    stop("Weather score matrix rows are not aligned to the expected environment order.")
  }

  if (!identical(regime, "LOEO")) {
    return(safe_scale_matrix(score_mat))
  }

  if (!heldout_env %in% envs) {
    stop("Held-out environment not found in environment order: ", heldout_env)
  }

  train_env_idx <- envs != heldout_env
  if (sum(train_env_idx) != length(envs) - 1L) {
    stop(
      "Expected ", length(envs) - 1L,
      " training environments for LOEO weather scaling, found ",
      sum(train_env_idx), "."
    )
  }

  centers <- colMeans(score_mat[train_env_idx, , drop = FALSE], na.rm = TRUE)
  sds <- apply(score_mat[train_env_idx, , drop = FALSE], 2, sd, na.rm = TRUE)

  centers[!is.finite(centers)] <- 0
  sds[!is.finite(sds) | sds == 0] <- 1

  out <- sweep(score_mat, 2, centers, "-")
  out <- sweep(out, 2, sds, "/")
  out[!is.finite(out)] <- 0
  rownames(out) <- rownames(score_mat)
  colnames(out) <- colnames(score_mat)
  out
}

normalize_weather_variable_col <- function(weather_df) {
  weather_df <- as.data.frame(weather_df)

  if ("Weather.Variable" %in% names(weather_df)) {
    return(weather_df)
  }

  if ("Weather_Variable" %in% names(weather_df)) {
    names(weather_df)[names(weather_df) == "Weather_Variable"] <- "Weather.Variable"
    return(weather_df)
  }

  if ("Weather.Var" %in% names(weather_df)) {
    names(weather_df)[names(weather_df) == "Weather.Var"] <- "Weather.Variable"
    return(weather_df)
  }

  stop(
    "Weather input is missing a recognizable weather-variable column. ",
    "Expected one of: Weather.Variable, Weather_Variable, Weather.Var."
  )
}

resolve_weather_score_cols <- function(weather_df, weather_traits, nw_fpcs) {
  expected_trait_cols <- as.vector(
    unlist(lapply(weather_traits, function(trait) paste0("FPC", seq_len(nw_fpcs), ".", trait)))
  )

  trait_cols <- intersect(expected_trait_cols, names(weather_df))
  if (length(trait_cols) > 0L) {
    return(trait_cols)
  }

  generic_cols <- intersect(paste0("FPC", seq_len(nw_fpcs)), names(weather_df))
  if (length(generic_cols) > 0L) {
    return(generic_cols)
  }

  character(0)
}

align_wide_weather_scores <- function(weather_df, envs, score_cols) {
  weather_df <- as.data.frame(weather_df)

  rows_by_env <- lapply(envs, function(env_i) {
    env_df <- weather_df[weather_df$Env == env_i, , drop = FALSE]
    if (nrow(env_df) == 0L) {
      stop("Weather input is missing environment ", env_i)
    }

    values <- vapply(score_cols, function(col_i) {
      if (!col_i %in% names(env_df)) {
        return(NA_real_)
      }

      vals <- env_df[[col_i]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        return(NA_real_)
      }

      vals_num <- suppressWarnings(as.numeric(vals))
      vals_num <- vals_num[!is.na(vals_num)]
      if (length(vals_num) == 0L) {
        return(NA_real_)
      }

      uniq_vals <- unique(vals_num)
      if (length(uniq_vals) > 1L) {
        stop(
          "Weather input has multiple non-identical values for environment ", env_i,
          " and column ", col_i, "."
        )
      }

      uniq_vals[[1]]
    }, numeric(1))

    values
  })

  score_mat <- do.call(rbind, rows_by_env)
  rownames(score_mat) <- envs
  colnames(score_mat) <- score_cols
  score_mat
}

align_stacked_weather_scores <- function(weather_df, envs, weather_traits, nw_fpcs) {
  generic_cols <- intersect(paste0("FPC", seq_len(nw_fpcs)), names(weather_df))

  if (length(generic_cols) == 0L) {
    stop("Stacked weather input is missing generic FPC columns.")
  }

  score_cols <- as.vector(
    unlist(lapply(weather_traits, function(trait_i) paste0(generic_cols, ".", trait_i)))
  )

  rows_by_env <- lapply(envs, function(env_i) {
    env_df <- weather_df[weather_df$Env == env_i, , drop = FALSE]
    if (nrow(env_df) == 0L) {
      stop("Weather input is missing environment ", env_i)
    }

    env_values <- unlist(lapply(weather_traits, function(trait_i) {
      trait_df <- env_df[env_df$Weather.Variable == trait_i, generic_cols, drop = FALSE]
      if (nrow(trait_df) == 0L) {
        stop(
          "Weather input is missing trait ", trait_i,
          " for environment ", env_i, "."
        )
      }
      if (nrow(trait_df) > 1L) {
        stop(
          "Weather input has duplicate rows for trait ", trait_i,
          " in environment ", env_i, "."
        )
      }

      vals <- suppressWarnings(as.numeric(trait_df[1, generic_cols, drop = TRUE]))
      names(vals) <- paste0(generic_cols, ".", trait_i)
      vals
    }))

    env_values[score_cols]
  })

  score_mat <- do.call(rbind, rows_by_env)
  rownames(score_mat) <- envs
  colnames(score_mat) <- score_cols
  score_mat
}

prepare_weather_score_matrix <- function(weather_df, envs, weather_traits, nw_fpcs) {
  weather_df <- normalize_weather_variable_col(weather_df)
  weather_df <- weather_df %>%
    filter(Weather.Variable %in% weather_traits)

  trait_specific_cols <- as.vector(
    unlist(lapply(weather_traits, function(trait_i) paste0("FPC", seq_len(nw_fpcs), ".", trait_i)))
  )
  available_trait_cols <- intersect(trait_specific_cols, names(weather_df))
  generic_cols <- intersect(paste0("FPC", seq_len(nw_fpcs)), names(weather_df))

  if (length(available_trait_cols) > 0L) {
    score_mat <- align_wide_weather_scores(
      weather_df = weather_df,
      envs = envs,
      score_cols = available_trait_cols
    )
  } else if (length(generic_cols) > 0L) {
    score_mat <- align_stacked_weather_scores(
      weather_df = weather_df,
      envs = envs,
      weather_traits = weather_traits,
      nw_fpcs = nw_fpcs
    )
  } else {
    stop("No retained weather score columns were found after filtering weather traits.")
  }

  missing_envs <- setdiff(envs, unique(weather_df$Env))
  if (length(missing_envs) > 0L) {
    stop(
      "Weather input is missing the following environments: ",
      paste(missing_envs, collapse = ", ")
    )
  }

  score_mat <- score_mat[envs, , drop = FALSE]

  list(
    weather_df = weather_df,
    score_cols = colnames(score_mat),
    score_mat = score_mat
  )
}

build_weather_bundle <- function(weather_df, constant_bundle, envs, weather_traits, nw_fpcs, regime, heldout_env) {
  prepared <- prepare_weather_score_matrix(
    weather_df = weather_df,
    envs = envs,
    weather_traits = weather_traits,
    nw_fpcs = nw_fpcs
  )

  score_mat <- prepared$score_mat
  score_mat <- scale_weather_score_matrix(
    score_mat = score_mat,
    envs = envs,
    regime = regime,
    heldout_env = heldout_env
  )

  K_W <- tcrossprod(score_mat) / ncol(score_mat)
  rownames(K_W) <- envs
  colnames(K_W) <- envs

  KE_W <- constant_bundle$Ze %*% K_W %*% t(constant_bundle$Ze)
  KG_GE_AW <- KE_W * constant_bundle$right_add
  KG_GE_DW <- KE_W * constant_bundle$right_dom

  rownames(KE_W) <- constant_bundle$order
  colnames(KE_W) <- constant_bundle$order
  rownames(KG_GE_AW) <- constant_bundle$order
  colnames(KG_GE_AW) <- constant_bundle$order
  rownames(KG_GE_DW) <- constant_bundle$order
  colnames(KG_GE_DW) <- constant_bundle$order

  eig_list <- list(
    KE_W.eig = eigen(KE_W, symmetric = TRUE),
    KG_GE_AW.eig = eigen(KG_GE_AW, symmetric = TRUE),
    KG_GE_DW.eig = eigen(KG_GE_DW, symmetric = TRUE)
  )

  list(
    regime = regime,
    heldout_env = heldout_env,
    weather_traits = weather_traits,
    score_cols = prepared$score_cols,
    weather_score_mat = score_mat,
    K_W = K_W,
    eig_list = eig_list
  )
}

make_weather_logger <- function(log_file = NULL, emit = TRUE) {
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

run_cv_weather_setup <- function(
    data_path = get_path("G2F_DATA_PATH", cv_paths$data_path),
    out_root = get_path("G2F_CV_OUT_PATH", cv_paths$out_root),
    loeo_score_path = get_path("G2F_LOEO_SCORE_PATH", cv_paths$loeo_score_path),
    weather_all_file = Sys.getenv("G2F_WEATHER_ALL_FILE", unset = "Weather_FPCA_AGDD_EnvRtype_Data_Scores_PTR_ONLY_V1.csv"),
    weather_loeo_file = Sys.getenv("G2F_WEATHER_LOEO_FILE", unset = "Weather_FPC_Scores_AGDD_LOEO_Projected.csv"),
    weather_traits = strsplit(Sys.getenv("G2F_WEATHER_TRAITS", unset = "PTR"), ",", fixed = TRUE)[[1]],
    nw_fpcs = suppressWarnings(as.integer(Sys.getenv("G2F_WEATHER_NFPCS", unset = "1"))),
    write_outputs = TRUE,
    return_objects = FALSE,
    log_to_console = TRUE) {

  weather_traits <- trimws(weather_traits)
  weather_traits <- weather_traits[nzchar(weather_traits)]

  if (is.na(nw_fpcs) || nw_fpcs < 1L) {
    stop("G2F_WEATHER_NFPCS must be a positive integer.")
  }

  bundles_dir <- file.path(out_root, "bundles")
  weather_dir <- file.path(bundles_dir, "weather")
  weather_loeo_dir <- file.path(weather_dir, "loeo")
  logs_dir <- file.path(out_root, "logs")

  if (isTRUE(write_outputs)) {
    dir.create(weather_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(weather_loeo_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
  }

  log_file <- if (isTRUE(write_outputs)) file.path(logs_dir, "AGDD_CV_Weather_Setup_V2.log") else NULL
  log_message <- make_weather_logger(log_file = log_file, emit = log_to_console)

  constant_bundle <- readRDS(file.path(bundles_dir, "constant_kernel_bundle.rds"))
  envs <- constant_bundle$envs

  log_message("Weather traits = ", paste(weather_traits, collapse = ","))
  log_message("Weather FPCs retained = ", nw_fpcs)

  weather_all <- fread(file.path(data_path, weather_all_file)) %>% normalize_weather_variable_col()

  all_bundle <- build_weather_bundle(
    weather_df = weather_all,
    constant_bundle = constant_bundle,
    envs = envs,
    weather_traits = weather_traits,
    nw_fpcs = nw_fpcs,
    regime = "ALL",
    heldout_env = "None"
  )

  weather_loeo_path <- file.path(loeo_score_path, weather_loeo_file)
  if (!file.exists(weather_loeo_path)) {
    weather_loeo_path <- file.path(data_path, weather_loeo_file)
  }
  g2f_require_file(weather_loeo_path, "AGDD LOEO weather score file")
  log_message("LOEO weather scores = ", weather_loeo_path)
  weather_loeo <- fread(weather_loeo_path) %>% normalize_weather_variable_col()

  loeo_results <- lapply(envs, function(heldout_env) {
    log_message("Building LOEO weather bundle for ", heldout_env)

    weather_i <- weather_loeo %>%
      filter(Heldout.Environment == heldout_env)

    bundle_i <- build_weather_bundle(
      weather_df = weather_i,
      constant_bundle = constant_bundle,
      envs = envs,
      weather_traits = weather_traits,
      nw_fpcs = nw_fpcs,
      regime = "LOEO",
      heldout_env = heldout_env
    )

    out_file <- file.path(weather_loeo_dir, paste0("weather_", gsub("[^A-Za-z0-9._-]", "_", heldout_env), ".rds"))

    list(
      manifest = data.frame(
        Regime = "LOEO",
        Heldout_Env = heldout_env,
        Bundle_File = out_file,
        Score_Cols = paste(bundle_i$score_cols, collapse = ";"),
        stringsAsFactors = FALSE
      ),
      bundle = bundle_i,
      out_file = out_file
    )
  })

  manifest_df <- bind_rows(
    data.frame(
      Regime = "ALL",
      Heldout_Env = "None",
      Bundle_File = file.path(weather_dir, "weather_all_bundle.rds"),
      Score_Cols = paste(all_bundle$score_cols, collapse = ";"),
      stringsAsFactors = FALSE
    ),
    bind_rows(lapply(loeo_results, `[[`, "manifest"))
  )

  if (isTRUE(write_outputs)) {
    saveRDS(all_bundle, file.path(weather_dir, "weather_all_bundle.rds"))
    invisible(lapply(loeo_results, function(x) saveRDS(x$bundle, x$out_file)))
    fwrite(manifest_df, file.path(weather_dir, "weather_bundle_manifest.csv"))
  }

  log_message("Completed weather bundle setup")
  print(manifest_df)

  result <- list(
    manifest_df = manifest_df,
    constant_bundle = constant_bundle,
    weather_all = weather_all,
    weather_loeo = weather_loeo,
    all_bundle = all_bundle,
    loeo_bundles = setNames(lapply(loeo_results, `[[`, "bundle"), envs),
    data_path = data_path,
    out_root = out_root
  )

  if (isTRUE(return_objects)) result else invisible(manifest_df)
}

main <- function() {
  run_cv_weather_setup()
}

if (sys.nframe() == 0L) {
  main()
}
