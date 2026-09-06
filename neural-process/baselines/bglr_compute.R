#!/usr/bin/env Rscript
#
# Multi-kernel Bayesian RKHS regression via BGLR's Gibbs sampler.
#
# RKHS(K) parameterization: one ETA term per feature_keys block.
# This script eigen-decomposes each kernel in R (`eigen(K, symmetric =
# TRUE)`) and passes `list(V = $vectors, d = $values, model = 'RKHS')`
# straight into ETA — matching the R reference rather than handing the
# raw K to BGLR. The optional terminal env block uses
# `list(X = Z_e, model = 'BRR')`.
#
# This script is transductive: it consumes train + test rows in a
# single fit, masks test y to NA, and emits fit$yHat for the full
# vector. Slicing into train/test predictions is the caller's job.
#
# Usage:
#   Rscript bglr_compute.R --work_dir <path>
#
# Working directory layout (caller-provided):
#   <work_dir>/
#     manifest.json          {block_names, env_brr_key, n_iter, burn_in,
#                             seed, n_train, n_test}
#     y_full.csv             header "y", N rows. Test rows are
#                            placeholders — overwritten via train_mask.
#     train_mask.csv         header "is_train", values in {0, 1}.
#     blocks/
#       <block_name>.bin     raw float64, N*N doubles, row-major.
#                            Symmetric N×N kernel matrix K_k. One per
#                            block in `block_names`.
#       <env_brr_key>.csv    optional, only when env_brr_key set;
#                            header z_0..z_{p-1}, N rows. Env design
#                            matrix Z_e (kept as CSV — small, dense,
#                            non-square).
#
# Outputs (written back into <work_dir>):
#   yhat.csv         header "y_hat", N rows (train fitted + test predicted).
#   bglr_meta.json   {varE, terms[{name,var,dim,model}], n_iter,
#                     burn_in, seed, elapsed_s}.

suppressPackageStartupMessages({
  library(BGLR)
  library(optparse)
  library(jsonlite)
})

# Threading is pinned by the Python parent (sets OMP / OPENBLAS / MKL /
# VECLIB env vars before Rscript launches). The R script does NOT set
# thread env vars itself — OpenBLAS / MKL / Accelerate size their thread
# pools at first BLAS call, and Sys.setenv() inside R may run after that
# point.

option_list <- list(
  make_option("--work_dir", type = "character",
              help = "Path to BGLR work directory")
)

opt <- parse_args(OptionParser(option_list = option_list))
stopifnot(!is.null(opt$work_dir))

work_dir <- opt$work_dir
if (!dir.exists(work_dir)) {
  stop(sprintf("[R-BGLR] ERROR: work_dir does not exist: %s", work_dir))
}

# ── Read manifest ────────────────────────────────────────────────────────────

manifest_path <- file.path(work_dir, "manifest.json")
if (!file.exists(manifest_path)) {
  stop(sprintf("[R-BGLR] ERROR: manifest.json not found at %s",
               manifest_path))
}
manifest <- fromJSON(manifest_path, simplifyVector = TRUE)

required_fields <- c("block_names", "n_iter", "burn_in", "seed",
                     "n_train", "n_test")
missing_fields <- setdiff(required_fields, names(manifest))
if (length(missing_fields) > 0) {
  stop(sprintf(
    "[R-BGLR] ERROR: manifest.json missing required fields: %s",
    paste(missing_fields, collapse = ", ")
  ))
}

block_names <- as.character(manifest$block_names)
n_iter      <- as.integer(manifest$n_iter)
burn_in     <- as.integer(manifest$burn_in)
seed        <- as.integer(manifest$seed)
n_train     <- as.integer(manifest$n_train)
n_test      <- as.integer(manifest$n_test)
N           <- n_train + n_test

env_brr_key <- if (is.null(manifest$env_brr_key)) {
  NULL
} else {
  as.character(manifest$env_brr_key)
}

# Reproducibility: canonical CRAN BGLR has no `seed=` argument
# (would error with `unused argument`); rely on set.seed() alone.
set.seed(seed)

# ── Read y and train mask ────────────────────────────────────────────────────

y_full   <- read.csv(file.path(work_dir, "y_full.csv"))$y
is_train <- read.csv(file.path(work_dir, "train_mask.csv"))$is_train == 1

if (length(y_full) != N) {
  stop(sprintf(
    "[R-BGLR] ERROR: y_full has %d rows, manifest says n_train+n_test=%d",
    length(y_full), N
  ))
}
if (length(is_train) != N) {
  stop(sprintf(
    "[R-BGLR] ERROR: train_mask has %d rows, expected %d",
    length(is_train), N
  ))
}

# Mask test y to NA — matches the R reference's transductive pattern. The
# placeholder zeros that Python writes for test rows in y_full.csv
# are unconditionally overwritten here, so their numeric value
# never reaches BGLR.
y_t <- y_full
y_t[!is_train] <- NA_real_

# ── Build ETA ────────────────────────────────────────────────────────────────

read_kernel_bin <- function(block_name) {
  path <- file.path(work_dir, "blocks", paste0(block_name, ".bin"))
  if (!file.exists(path)) {
    stop(sprintf("[R-BGLR] ERROR: kernel file not found: %s", path))
  }
  con <- file(path, "rb")
  on.exit(close(con))
  vals <- readBin(con, what = "double", n = N * N, size = 8L,
                  endian = "little")
  if (length(vals) != N * N) {
    stop(sprintf(
      "[R-BGLR] ERROR: kernel %s has %d doubles, expected %d (N=%d)",
      block_name, length(vals), N * N, N
    ))
  }
  # Python writes K row-major; R fills column-major. K is symmetric
  # by construction (raw obs-level kernel from a producer) so the
  # orientation doesn't matter, but we explicitly transpose-equivalent
  # fill via byrow = TRUE for clarity.
  matrix(vals, nrow = N, ncol = N, byrow = TRUE)
}

read_env_csv <- function(block_name) {
  path <- file.path(work_dir, "blocks", paste0(block_name, ".csv"))
  if (!file.exists(path)) {
    stop(sprintf("[R-BGLR] ERROR: env block file not found: %s", path))
  }
  X <- as.matrix(read.csv(path))
  if (nrow(X) != N) {
    stop(sprintf(
      "[R-BGLR] ERROR: env block %s has %d rows, expected %d",
      block_name, nrow(X), N
    ))
  }
  X
}

ETA <- list()
eta_models <- character(0)   # parallel to ETA, used during meta readback
eta_eigen_rank <- integer(0) # parallel; eigen rank passed to BGLR
# No upstream filter: the raw K is eigen-decomposed once and the
# full result (vectors + values, including numerical-noise negatives)
# is handed to BGLR. BGLR's internal RKHS handler (tolD=1e-10) is the
# sole arbiter — matches the R reference (DAP_LOEO_Prediction_Setup_V3.R:250,
# 267-276) which passes `eigen(K)$vectors`/`$values` straight into ETA.
for (block_name in block_names) {
  K <- read_kernel_bin(block_name)
  evd <- eigen(K, symmetric = TRUE)
  if (length(evd$values) == 0) {
    stop(sprintf("[R-BGLR] ERROR: kernel %s eigen-decomp returned no values",
                 block_name))
  }
  if (max(evd$values) <= 0) {
    stop(sprintf(
      "[R-BGLR] ERROR: kernel %s has no positive eigenvalues (max=%g) — non-PSD?",
      block_name, max(evd$values)
    ))
  }
  V_k <- evd$vectors
  d_k <- evd$values
  # Report the EFFECTIVE positive rank (every positive mode of K), not the raw
  # N = length(d_k). The full V/d (including numerical-noise negatives) is still
  # handed to BGLR below — BGLR's internal tolD filters those — but the reported
  # `dim` must be the positive-mode count so it stays consistent with the
  # processor's advertised feature_dims (a low-rank kernel has rank < N).
  rank_k <- sum(d_k > 0)
  rm(evd)
  ETA[[length(ETA) + 1]] <- list(V = V_k, d = d_k, model = "RKHS")
  eta_models <- c(eta_models, "RKHS")
  eta_eigen_rank <- c(eta_eigen_rank, rank_k)
  cat(sprintf(
    "[R-BGLR] Kernel block %s: K shape=(%d, %d), eigen rank=%d [RKHS]\n",
    block_name, nrow(K), ncol(K), rank_k
  ))
  rm(K)
}

# Env BRR term goes LAST in ETA — Z_e ETA-last convention. We
# append via length(ETA) + 1 (positional, not named) so env always
# lands strictly after every kernel block regardless of how
# manifest$block_names was serialized.
if (!is.null(env_brr_key)) {
  X_env <- read_env_csv(env_brr_key)
  ETA[[length(ETA) + 1]] <- list(X = X_env, model = "BRR")
  eta_models <- c(eta_models, "BRR")
  eta_eigen_rank <- c(eta_eigen_rank, ncol(X_env))
  cat(sprintf("[R-BGLR] Env block %s: p=%d (ETA-last) [BRR]\n",
              env_brr_key, ncol(X_env)))
}

# Ordered ETA names: kernel blocks then env (matches the order of
# ETA terms above).
eta_names <- if (is.null(env_brr_key)) {
  block_names
} else {
  c(block_names, env_brr_key)
}

# ── Fit BGLR ─────────────────────────────────────────────────────────────────

# saveAt is a path *prefix*, BGLR does not mkdir parents — so create
# the scratch dir before BGLR's first write or it would error.
scratch_dir <- file.path(work_dir, "bglr_scratch")
dir.create(scratch_dir, showWarnings = FALSE, recursive = TRUE)
saveAt_prefix <- file.path(scratch_dir, "BGLR_")

t0 <- Sys.time()

# Requires the rank-1-patched BGLR (R/BGLR.R: `LT$V[, tmp, drop = FALSE]`).
# Stock CRAN BGLR 1.1.4 aborts with "invalid 'times' argument" when a kernel
# block has a single super-tolD eigenvalue (e.g. weather-GxE's rank-1 K_W under
# weather_fpca.n_components=1): R drops the 1-column eigenvector
# matrix to a vector and the outer product fails. Install the fix via
# scripts/patch_bglr.sh; see baselines/R_HPC_SETUP.md and bglr_rank1.patch.
fit <- tryCatch(
  BGLR(
    y       = y_t,
    ETA     = ETA,
    nIter   = n_iter,
    burnIn  = burn_in,
    saveAt  = saveAt_prefix,
    verbose = FALSE
  ),
  error = function(e) {
    cat(sprintf("[R-BGLR] ERROR: %s\n", e$message))
    quit(status = 1)
  }
)

elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

# ── Write outputs ────────────────────────────────────────────────────────────

# yhat.csv: BGLR's posterior mean over the full N-vector. Train
# rows are fitted, test rows are predicted via the masked-y
# transductive call.
yhat_df <- data.frame(y_hat = fit$yHat)
write.csv(yhat_df, file.path(work_dir, "yhat.csv"), row.names = FALSE)

# Per-term variance + effective dim. RKHS terms expose `varU` (the
# kernel variance σ²_u). `dim` is the eigen rank handed to BGLR
# (every positive mode of K). BRR terms expose `varB` and use
# ncol(X). The unified `var` field downstream lets fpca_core stay
# regressor-agnostic.
terms_meta <- vector("list", length(ETA))
for (k in seq_along(ETA)) {
  model_k <- eta_models[k]
  if (model_k == "RKHS") {
    var_k <- fit$ETA[[k]]$varU
  } else {
    var_k <- fit$ETA[[k]]$varB
  }
  terms_meta[[k]] <- list(
    name  = eta_names[k],
    var   = var_k,
    dim   = eta_eigen_rank[k],
    model = model_k
  )
}

meta <- list(
  varE      = fit$varE,
  terms     = terms_meta,
  n_iter    = n_iter,
  burn_in   = burn_in,
  seed      = seed,
  elapsed_s = elapsed
)

writeLines(
  toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
  file.path(work_dir, "bglr_meta.json")
)

cat(sprintf("[R-BGLR] Done in %.2f s\n", elapsed))
