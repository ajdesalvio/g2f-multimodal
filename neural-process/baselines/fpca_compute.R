#!/usr/bin/env Rscript
#
# FPCA computation for sparse VI curves using fdapace.
#
# Two modes:
#   train   — fit FPCA per VI on training data only, project all splits via CE/BLUP, save models
#   predict — load saved FPCA models, project new curves via CE/BLUP
#
# Usage:
#   Rscript fpca_compute.R --mode train   --input_csv <path> --output_csv <path> --model_dir <path> --n_components 4
#   Rscript fpca_compute.R --mode predict --input_csv <path> --output_csv <path> --model_dir <path> --n_components 4

suppressPackageStartupMessages({
  library(fdapace)
  library(optparse)
})

# ── CLI argument parsing ─────────────────────────────────────────────────────

option_list <- list(
  make_option("--mode", type = "character", default = "train",
              help = "Mode: 'train' or 'predict' [default: train]"),
  make_option("--input_csv", type = "character",
              help = "Path to input CSV (tall format)"),
  make_option("--output_csv", type = "character",
              help = "Path to write FPC scores CSV"),
  make_option("--model_dir", type = "character",
              help = "Directory for saving/loading fpca_models.rds"),
  make_option("--n_components", type = "integer", default = 4L,
              help = "Number of FPC scores to extract [default: 4]"),
  make_option("--n_reg_grid", type = "integer", default = 0L,
              help = paste("nRegGrid for FPCA. 0 (default) leaves it unset, so",
                           "fdapace uses its own default of 51 — what the",
                           "reference LOEO generators and every VI fit use.",
                           "The reference sets 100 for the FULL 19-env weather",
                           "fit only; the Python caller passes 100 there and",
                           "nothing elsewhere."))
)

opt <- parse_args(OptionParser(option_list = option_list))

stopifnot(!is.null(opt$input_csv), !is.null(opt$output_csv), !is.null(opt$model_dir))

# ── Helper: build fdapace inputs from a subset of the tall dataframe ─────────

#' Convert a tall dataframe (one row per observation) into the list-of-lists
#' format required by fdapace::FPCA().
#'
#' The input dataframe must already be filtered to a single VI index and a
#' single split (e.g. "train").  It must contain columns: sample_id, t, value.
#'
#' For each unique sample_id, this function extracts the observed values and
#' timepoints, sorted by the time axis ``t``, into separate vectors. ``t`` is
#' the generic FPCA time coordinate — it carries DAP, GDD, or AGDD depending on
#' the configured view; fdapace is axis-name agnostic. The result is:
#'   - Ly: list of N numeric vectors, each containing one sample's VI values
#'   - Lt: list of N numeric vectors, each containing the corresponding t values
#'   - ids: vector of N sample_id values (preserves ordering for score matching)
#'
#' Samples may have different numbers of observations (sparse/irregular design),
#' which fdapace handles natively.
#'
#' @param df  Data frame with columns: sample_id, t, value
#' @return    Named list with elements Ly (values), Lt (timepoints), ids
make_fpca_inputs <- function(df) {
  ids <- unique(df$sample_id)
  Ly <- list()
  Lt <- list()
  for (i in seq_along(ids)) {
    sub <- df[df$sample_id == ids[i], ]
    sub <- sub[order(sub$t), ]
    Ly[[i]] <- sub$value
    Lt[[i]] <- sub$t
  }
  list(Ly = Ly, Lt = Lt, ids = ids)
}

# ── Helper: truncate curves to the fitted FPCA domain ────────────────────────

#' Drop observations outside the training domain before calling predict().
#'
#' fdapace::predict() hard-errors if any newLt value falls outside the domain
#' of the fitted mean/covariance functions [min(workGrid), max(workGrid)].
#' This happens in LOO CV when the held-out fold contains curves measured at
#' DAPs beyond the range seen during training (e.g. an environment with a
#' longer growing season than all training environments).
#'
#' For each curve, observations with DAP < t_min or DAP > t_max are silently
#' dropped.  A single summary message is printed when any truncation occurs so
#' the user can see which VI / split was affected and how many points were lost.
#' Curves that end up with fewer than 2 observations after truncation cannot be
#' scored by fdapace and are removed entirely (with a per-sample warning).
#'
#' @param fpca_obj  Fitted fdapace FPCA object
#' @param Ly        List of N numeric vectors — observed values
#' @param Lt        List of N numeric vectors — observed DAPs
#' @param ids       Vector of N sample_id values
#' @param context   Short string included in messages, e.g. "VI 5, split 'test'"
#' @return  Named list(Ly, Lt, ids) after truncation and filtering
truncate_to_domain <- function(fpca_obj, Ly, Lt, ids, context = "") {
  t_min <- min(fpca_obj$workGrid)
  t_max <- max(fpca_obj$workGrid)
  ctx   <- if (nchar(context) > 0) sprintf(" (%s)", context) else ""

  n_obs_dropped    <- 0L
  n_curves_clipped <- 0L
  keep <- rep(TRUE, length(ids))

  for (i in seq_along(Lt)) {
    # Mirror the reference's clip_fpca_lists(): drop non-finite VALUES as
    # well as out-of-domain times. Under missing_values="drop" upstream the
    # curves carry NaN where the CSV had gaps; predict() would otherwise
    # propagate those straight into the scores.
    mask    <- is.finite(Ly[[i]]) & is.finite(Lt[[i]]) &
               Lt[[i]] >= t_min & Lt[[i]] <= t_max
    n_drop  <- sum(!mask)
    if (n_drop > 0L) {
      n_obs_dropped    <- n_obs_dropped    + n_drop
      n_curves_clipped <- n_curves_clipped + 1L
      Lt[[i]] <- Lt[[i]][mask]
      Ly[[i]] <- Ly[[i]][mask]
    }
    if (length(Lt[[i]]) < 2L) {
      cat(sprintf(
        "[R] WARNING: sample_id=%s has fewer than 2 observations within the training domain [%.1f, %.1f]%s — zero scores will be used for this sample.\n",
        ids[i], t_min, t_max, ctx))
      keep[i] <- FALSE
    }
  }

  if (n_obs_dropped > 0L) {
    cat(sprintf(
      "[R] WARNING: Truncated curves to training DAP domain [%.1f, %.1f]%s: dropped %d observation(s) from %d curve(s) that extended beyond the training range.\n",
      t_min, t_max, ctx, n_obs_dropped, n_curves_clipped))
  }

  list(Ly = Ly[keep], Lt = Lt[keep], ids = ids[keep], unscored_ids = ids[!keep])
}

# ── Helper: extract FPC scores from a fitted FPCA object for given data ──────

#' Project a set of sparse curves onto a fitted FPCA basis and return the
#' K-dimensional score vectors as a tidy data frame.
#'
#' Uses fdapace::predict() to compute the Conditional Expectation (CE/BLUP)
#' scores for each sample's curve given the learned mean function,
#' eigenfunctions, and noise variance.
#'
#' Curves are first truncated to the training DAP domain via
#' truncate_to_domain() — fdapace::predict() hard-errors on out-of-range DAPs.
#'
#' If fdapace estimated fewer than n_components eigenfunctions (low-rank data),
#' the missing dimensions are zero-padded so every VI produces a vector of the
#' same length.  If more were estimated, only the first n_components are kept.
#'
#' @param fpca_obj     Fitted fdapace FPCA object (from training)
#' @param Ly           List of N numeric vectors — observed VI values per sample
#' @param Lt           List of N numeric vectors — observed DAPs per sample
#' @param ids          Vector of N sample_id values (for output matching)
#' @param n_components Number of FPC scores to extract (K)
#' @param vi_idx       Integer VI index (0–36), included in output for joining
#' @param split_name   Split label ("train", "val", "test"), included in output
#' @return  Data frame with columns: sample_id, vi_index, FPC1..FPCK, split
extract_scores <- function(fpca_obj, Ly, Lt, ids, n_components, vi_idx, split_name) {
  truncated <- truncate_to_domain(fpca_obj, Ly, Lt, ids,
                                  context = sprintf("VI %d, split '%s'", vi_idx, split_name))
  Ly  <- truncated$Ly
  Lt  <- truncated$Lt
  ids <- truncated$ids

  unscored_ids <- truncated$unscored_ids

  if (length(ids) == 0L) {
    cat(sprintf("[R] WARNING: No scorable samples remain for VI %d, split '%s' after domain truncation — all samples will receive zero scores.\n",
                vi_idx, split_name))
    zero_rows <- data.frame(
      sample_id      = unscored_ids,
      vi_index       = vi_idx,
      matrix(0, nrow = length(unscored_ids), ncol = n_components),
      split          = split_name,
      is_zero_padded = TRUE,
      stringsAsFactors = FALSE
    )
    colnames(zero_rows)[3:(2 + n_components)] <- paste0("FPC", 1:n_components)
    return(zero_rows)
  }

  pred <- predict(fpca_obj, newLy = Ly, newLt = Lt)
  scores <- pred$scores
  # fdapace::predict() may return a named vector (not a matrix) when only one
  # sample remains after truncation; coerce defensively to avoid ncol()/nrow()
  # returning NULL on a vector.
  if (!is.matrix(scores)) scores <- matrix(scores, nrow = 1L)

  # Ensure we have enough columns; pad with 0 if fewer components estimated
  n_avail <- ncol(scores)
  if (n_avail < n_components) {
    padding <- matrix(0, nrow = nrow(scores), ncol = n_components - n_avail)
    scores <- cbind(scores, padding)
  } else {
    scores <- scores[, 1:n_components, drop = FALSE]
  }

  result <- data.frame(
    sample_id = ids,
    vi_index = vi_idx,
    scores,
    split = split_name,
    is_zero_padded = FALSE,
    stringsAsFactors = FALSE
  )
  colnames(result)[3:(2 + n_components)] <- paste0("FPC", 1:n_components)

  # Append zero rows for samples that had < 2 obs after truncation
  if (length(unscored_ids) > 0L) {
    zero_rows <- data.frame(
      sample_id      = unscored_ids,
      vi_index       = vi_idx,
      matrix(0, nrow = length(unscored_ids), ncol = n_components),
      split          = split_name,
      is_zero_padded = TRUE,
      stringsAsFactors = FALSE
    )
    colnames(zero_rows)[3:(2 + n_components)] <- paste0("FPC", 1:n_components)
    result <- rbind(result, zero_rows)
  }

  result
}

# ── Helper: in-sample scores for the split the basis was fit on ─────────────

#' Take the FPCA object's own ``xiEst`` for the training samples.
#'
#' The reference never re-projects the fitting set: it reads
#' ``fpca_obj$xiEst`` directly (weather ``project_dap_fpca.R``; VI
#' ``DAP_CV_Prediction_Setup_V2.R``) and calls ``predict()`` only for
#' held-out curves. Re-projecting training curves gives *nearly* the same
#' numbers but not identical ones, and it also subjects those curves to
#' domain clipping the reference never applies to them.
#'
#' Only valid in train mode, where the basis was fit on exactly these
#' curves. In predict mode the basis came from elsewhere, so every split
#' must go through extract_scores().
#'
#' ``xiEst`` rows follow the order fdapace received the curves, i.e. the
#' order of ``ids`` from make_fpca_inputs() — so the two align 1:1.
scores_from_xi <- function(fpca_obj, ids, n_components, vi_idx, split_name) {
  scores <- fpca_obj$xiEst
  if (!is.matrix(scores)) scores <- matrix(scores, nrow = 1L)

  if (nrow(scores) != length(ids)) {
    stop(sprintf(
      "[R] xiEst has %d rows but %d training ids for VI %d — refusing to guess the alignment.",
      nrow(scores), length(ids), vi_idx))
  }

  n_avail <- ncol(scores)
  if (n_avail < n_components) {
    scores <- cbind(scores, matrix(0, nrow = nrow(scores),
                                   ncol = n_components - n_avail))
  } else {
    scores <- scores[, 1:n_components, drop = FALSE]
  }

  result <- data.frame(
    sample_id = ids,
    vi_index = vi_idx,
    scores,
    split = split_name,
    is_zero_padded = FALSE,
    stringsAsFactors = FALSE
  )
  colnames(result)[3:(2 + n_components)] <- paste0("FPC", 1:n_components)
  result
}

# ── TRAIN mode ───────────────────────────────────────────────────────────────

#' Fit one FPCA model per VI on training data, then project all splits onto
#' the learned bases to produce FPC score vectors.
#'
#' Pipeline for each of the 37 VIs:
#'   1. Filter the tall CSV to this VI's training rows
#'   2. Build fdapace inputs (Ly, Lt) via make_fpca_inputs()
#'   3. Fit fdapace::FPCA() with sparse smoothed covariance (GCV bandwidths)
#'   4. For every split (train/val/test), project curves via extract_scores()
#'
#' Outputs saved to disk:
#'   - fpca_models.rds: named list of 37 fitted FPCA objects (keyed by VI index)
#'   - fve.json: per-VI cumulative fraction of variance explained
#'   - output_csv: tall data frame of FPC scores (sample_id, vi_index, FPC1..K, split)
#'
#' @param opt  Parsed CLI options (input_csv, output_csv, model_dir, n_components)
run_train <- function(opt) {
  cat("[R] Loading input CSV:", opt$input_csv, "\n")
  data <- read.csv(opt$input_csv, stringsAsFactors = FALSE)

  vi_indices <- sort(unique(data$vi_index))
  n_components <- opt$n_components

  fpca_models <- list()
  all_scores <- list()
  fve_list <- list()
  failed_vis <- integer(0)
  # Whole-split extract_scores() FAILURES (the tryCatch below): each such
  # block is an all-zero feature column masquerading as a real projection.
  # Recorded here and surfaced in fpca_summary.json so the Python caller
  # can fail loud instead of silently ingesting a zero column. (Distinct
  # from benign per-sample <2-obs padding inside extract_scores.)
  zero_padded_blocks <- character(0)

  for (vi_idx in vi_indices) {
    cat(sprintf("[R] Fitting FPCA for VI %d ...\n", vi_idx))
    vi_data <- data[data$vi_index == vi_idx, ]

    # ── Fit on training data only ──
    train_data <- vi_data[vi_data$split == "train", ]
    train_inputs <- make_fpca_inputs(train_data)

    fpca_optns <- list(dataType = "Sparse",
                       methodMuCovEst = "smooth",
                       methodBwCov = "GCV",
                       methodBwMu = "GCV",
                       plot = FALSE)
    # Left unset unless the caller asks, so fdapace's default (51) applies —
    # matching the reference everywhere except its full 19-env weather fit.
    if (!is.null(opt$n_reg_grid) && opt$n_reg_grid > 0L) {
      fpca_optns$nRegGrid <- as.integer(opt$n_reg_grid)
    }

    fpca_obj <- tryCatch({
      FPCA(train_inputs$Ly, train_inputs$Lt, fpca_optns)
    }, error = function(e) {
      cat(sprintf("[R] WARNING: FPCA failed for VI %d: %s. Skipping.\n", vi_idx, e$message))
      return(NULL)
    })

    if (is.null(fpca_obj)) {
      failed_vis <- c(failed_vis, vi_idx)
      next
    }

    fpca_models[[as.character(vi_idx)]] <- fpca_obj

    # Save FVE info
    cum_fve <- fpca_obj$cumFVE
    fve_list[[as.character(vi_idx)]] <- cum_fve[1:min(length(cum_fve), n_components)]

    # ── Project all splits ──
    splits <- unique(vi_data$split)
    for (sp in splits) {
      sp_data <- vi_data[vi_data$split == sp, ]
      sp_inputs <- make_fpca_inputs(sp_data)

      scores_df <- tryCatch({
        if (identical(sp, "train")) {
          # The basis was fit on exactly these curves — read the in-sample
          # scores rather than re-projecting them (reference behaviour).
          scores_from_xi(fpca_obj, sp_inputs$ids, n_components, vi_idx, sp)
        } else {
          extract_scores(fpca_obj, sp_inputs$Ly, sp_inputs$Lt,
                         sp_inputs$ids, n_components, vi_idx, sp)
        }
      }, error = function(e) {
        cat(sprintf("[R] WARNING: scoring failed for VI %d, split '%s': %s — zero scores will be used for all %d sample(s).\n",
                    vi_idx, sp, e$message, length(sp_inputs$ids)))
        zero_padded_blocks <<- c(zero_padded_blocks,
                                 sprintf("vi=%d,split=%s", vi_idx, sp))
        zero_df <- data.frame(
          sample_id      = sp_inputs$ids,
          vi_index       = vi_idx,
          matrix(0, nrow = length(sp_inputs$ids), ncol = n_components),
          split          = sp,
          is_zero_padded = TRUE,
          stringsAsFactors = FALSE
        )
        colnames(zero_df)[3:(2 + n_components)] <- paste0("FPC", 1:n_components)
        zero_df
      })

      all_scores[[length(all_scores) + 1]] <- scores_df
    }
  }

  # Save FPCA models
  dir.create(opt$model_dir, showWarnings = FALSE, recursive = TRUE)
  model_path <- file.path(opt$model_dir, "fpca_models.rds")
  saveRDS(fpca_models, model_path)
  cat("[R] Saved FPCA models to:", model_path, "\n")

  # Save FVE info
  fve_path <- file.path(opt$model_dir, "fve.json")
  fve_json <- paste0("{\n",
    paste(sapply(names(fve_list), function(nm) {
      vals <- paste(fve_list[[nm]], collapse = ", ")
      sprintf('  "%s": [%s]', nm, vals)
    }), collapse = ",\n"),
    "\n}")
  writeLines(fve_json, fve_path)
  cat("[R] Saved FVE to:", fve_path, "\n")

  # Save FPCA fitting summary — read by Python, which fails loud on any
  # failed VI or zero-padded (all-zero) score block.
  summary_path <- file.path(opt$model_dir, "fpca_summary.json")
  failed_vis_str <- if (length(failed_vis) > 0L) paste(failed_vis, collapse = ", ") else ""
  zpb_str <- if (length(zero_padded_blocks) > 0L) {
    paste(sprintf('"%s"', zero_padded_blocks), collapse = ", ")
  } else ""
  summary_json <- sprintf(
    '{\n  "mode": "train",\n  "n_vis_attempted": %d,\n  "n_vis_fitted": %d,\n  "failed_vis": [%s],\n  "n_zero_padded_blocks": %d,\n  "zero_padded_blocks": [%s]\n}\n',
    length(vi_indices),
    length(fpca_models),
    failed_vis_str,
    length(zero_padded_blocks),
    zpb_str
  )
  writeLines(summary_json, summary_path)
  cat("[R] Saved FPCA fitting summary to:", summary_path, "\n")

  # Write scores CSV
  result_df <- do.call(rbind, all_scores)
  write.csv(result_df, opt$output_csv, row.names = FALSE)
  cat("[R] Wrote FPC scores to:", opt$output_csv, "\n")
  cat(sprintf("[R] Total rows: %d, VIs fitted: %d\n", nrow(result_df), length(fpca_models)))
}

# ── PREDICT mode ─────────────────────────────────────────────────────────────

#' Project new sparse curves onto previously saved FPCA bases.
#'
#' Loads the fitted FPCA models from fpca_models.rds (saved during training),
#' then for each VI projects the input curves using fdapace::predict() to
#' obtain FPC scores via the Conditional Expectation formula.
#'
#' The fitted basis carries every estimated component; the Python caller
#' applies its configured n_components as a post-load slice.
#'
#' If the input contains a VI index that was not fitted during training (e.g.
#' because FPCA failed for that VI), it is skipped with a warning.
#'
#' @param opt  Parsed CLI options (input_csv, output_csv, model_dir, n_components)
run_predict <- function(opt) {
  cat("[R] Loading input CSV:", opt$input_csv, "\n")
  data <- read.csv(opt$input_csv, stringsAsFactors = FALSE)

  model_path <- file.path(opt$model_dir, "fpca_models.rds")
  cat("[R] Loading FPCA models from:", model_path, "\n")
  fpca_models <- readRDS(model_path)

  n_components <- opt$n_components
  all_scores <- list()
  # Same fail-loud markers as train mode (see run_train).
  zero_padded_blocks <- character(0)
  failed_vis <- integer(0)          # VIs with no saved basis → dropped

  vi_indices <- sort(unique(data$vi_index))

  for (vi_idx in vi_indices) {
    vi_key <- as.character(vi_idx)
    if (!(vi_key %in% names(fpca_models))) {
      cat(sprintf("[R] WARNING: No saved FPCA model for VI %d. Skipping.\n", vi_idx))
      failed_vis <- c(failed_vis, vi_idx)
      next
    }

    fpca_obj <- fpca_models[[vi_key]]
    vi_data <- data[data$vi_index == vi_idx, ]

    splits <- unique(vi_data$split)
    for (sp in splits) {
      sp_data <- vi_data[vi_data$split == sp, ]
      sp_inputs <- make_fpca_inputs(sp_data)

      scores_df <- tryCatch({
        extract_scores(fpca_obj, sp_inputs$Ly, sp_inputs$Lt,
                       sp_inputs$ids, n_components, vi_idx, sp)
      }, error = function(e) {
        cat(sprintf("[R] WARNING: scoring failed for VI %d, split '%s': %s — zero scores will be used for all %d sample(s).\n",
                    vi_idx, sp, e$message, length(sp_inputs$ids)))
        zero_padded_blocks <<- c(zero_padded_blocks,
                                 sprintf("vi=%d,split=%s", vi_idx, sp))
        zero_df <- data.frame(
          sample_id      = sp_inputs$ids,
          vi_index       = vi_idx,
          matrix(0, nrow = length(sp_inputs$ids), ncol = n_components),
          split          = sp,
          is_zero_padded = TRUE,
          stringsAsFactors = FALSE
        )
        colnames(zero_df)[3:(2 + n_components)] <- paste0("FPC", 1:n_components)
        zero_df
      })

      all_scores[[length(all_scores) + 1]] <- scores_df
    }
  }

  result_df <- do.call(rbind, all_scores)
  write.csv(result_df, opt$output_csv, row.names = FALSE)
  cat("[R] Wrote FPC scores to:", opt$output_csv, "\n")
  cat(sprintf("[R] Total rows: %d\n", nrow(result_df)))

  # Fail-loud summary — Python raises on any dropped VI or zero-padded block.
  summary_path <- file.path(opt$model_dir, "fpca_summary.json")
  failed_vis_str <- if (length(failed_vis) > 0L) paste(failed_vis, collapse = ", ") else ""
  zpb_str <- if (length(zero_padded_blocks) > 0L) {
    paste(sprintf('"%s"', zero_padded_blocks), collapse = ", ")
  } else ""
  summary_json <- sprintf(
    '{\n  "mode": "predict",\n  "n_vis_attempted": %d,\n  "n_vis_fitted": %d,\n  "failed_vis": [%s],\n  "n_zero_padded_blocks": %d,\n  "zero_padded_blocks": [%s]\n}\n',
    length(vi_indices),
    length(vi_indices) - length(failed_vis),
    failed_vis_str,
    length(zero_padded_blocks),
    zpb_str
  )
  writeLines(summary_json, summary_path)
  cat("[R] Saved FPCA fitting summary to:", summary_path, "\n")
}

# ── Main dispatch ────────────────────────────────────────────────────────────

if (opt$mode == "train") {
  run_train(opt)
} else if (opt$mode == "predict") {
  run_predict(opt)
} else {
  stop(sprintf("Unknown mode: '%s'. Use 'train' or 'predict'.", opt$mode))
}

cat("[R] Done.\n")
