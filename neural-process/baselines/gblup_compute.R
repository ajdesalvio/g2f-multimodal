#!/usr/bin/env Rscript
#
# Multi-kernel frequentist GBLUP — the REML/BLUP twin of bglr_compute.R.
#
# Same generative model as the Bayesian RKHS path
#   y = 1·mu + sum_k u_k + e,   u_k ~ N(0, K_k·sigma2_k),  e ~ N(0, sigma2_e·I)
# but the variance components {sigma2_k, sigma2_e} are estimated by
# REML (via `sommer::mmer`) on the TRAIN rows instead of drawn by a
# Gibbs sampler under scaled-inverse-chi2 priors. Predictions are the
# BLUP (= the posterior mean conditional on the REML point estimates),
# which for a multi-kernel mixed model is the explicit kriging form
#   yhat = 1·mu_hat + sum_k sigma2_k · K_k[, tr] · V_tr^{-1} (y_tr - 1·mu_hat)
#   V_tr = sum_k sigma2_k · K_k[tr, tr] + sigma2_e · I_ntr,
#   mu_hat = (1' V_tr^{-1} y_tr) / (1' V_tr^{-1} 1)   (GLS intercept).
#
# Division of labour:
#   * sommer estimates the variance components by REML (the part that
#     makes this the *frequentist* twin — not a CV-tuned ridge).
#   * the BLUP/kriging step is done here explicitly from the full
#     train+test kernel blocks. Doing it ourselves (rather than via
#     sommer's predict) keeps transductive prediction unambiguous
#     across every CV scheme (the K_k[te, tr] cross-blocks carry the
#     coupling) and sidesteps sommer's NA-row handling. It also makes
#     rank-1 kernels (weather-GxE's weather K_W) a non-event — no BGLR-style
#     rank-1 patch is needed, since V_tr is SPD for any sigma2_e > 0.
#
# The env BRR(Z_e) term of the Bayesian path is folded UPSTREAM (in
# GBLUPRegressor) into a linear kernel block K_env = Z_e Z_e', which is
# exactly the RKHS-equivalent of a ridge on the env one-hot. So this
# script sees a flat list of N×N kernel blocks and treats env like any
# other; `env_brr_key` from the manifest is used only to label that
# term's `model` field in the meta output.
#
# This script is transductive: it consumes train + test rows together,
# fits REML on train, and emits yhat for the full N-vector. Slicing
# into train/test predictions is the caller's job.
#
# Usage:
#   Rscript gblup_compute.R --work_dir <path>
#
# Working directory layout (caller-provided):
#   <work_dir>/
#     manifest.json          {block_names, env_brr_key, reml_max_iter,
#                             tol, seed, n_train, n_test}
#     y_full.csv             header "y", N rows. Test rows are
#                            placeholders — never read (train_mask gates).
#     train_mask.csv         header "is_train", values in {0, 1}.
#     blocks/
#       <block_name>.bin     raw float64, N*N doubles, row-major.
#                            Symmetric N×N kernel matrix K_k. One per
#                            block in `block_names` (env block included).
#
# Outputs (written back into <work_dir>):
#   yhat.csv          header "y_hat", N rows (train fitted + test predicted).
#   gblup_meta.json   {varE, terms[{name,var,dim,model}], reml_max_iter,
#                      reml_converged, seed, elapsed_s, method}.

suppressPackageStartupMessages({
  library(sommer)
  library(optparse)
  library(jsonlite)
})

# Threading: unlike the BGLR path, the Python parent does NOT pin
# BLAS/OMP threads for this script — sommer's REML is dense-matrix
# bound, so the BLAS backend sizes its own pool (see
# gblup_regressor.py). The R script sets no thread env vars either.

option_list <- list(
  make_option("--work_dir", type = "character",
              help = "Path to GBLUP work directory")
)

opt <- parse_args(OptionParser(option_list = option_list))
stopifnot(!is.null(opt$work_dir))

work_dir <- opt$work_dir
if (!dir.exists(work_dir)) {
  stop(sprintf("[R-GBLUP] ERROR: work_dir does not exist: %s", work_dir))
}

# ── Read manifest ────────────────────────────────────────────────────────────

manifest_path <- file.path(work_dir, "manifest.json")
if (!file.exists(manifest_path)) {
  stop(sprintf("[R-GBLUP] ERROR: manifest.json not found at %s",
               manifest_path))
}
manifest <- fromJSON(manifest_path, simplifyVector = TRUE)

required_fields <- c("block_names", "reml_max_iter", "seed",
                     "n_train", "n_test")
missing_fields <- setdiff(required_fields, names(manifest))
if (length(missing_fields) > 0) {
  stop(sprintf(
    "[R-GBLUP] ERROR: manifest.json missing required fields: %s",
    paste(missing_fields, collapse = ", ")
  ))
}

block_names   <- as.character(manifest$block_names)
reml_max_iter <- as.integer(manifest$reml_max_iter)
seed          <- as.integer(manifest$seed)
n_train       <- as.integer(manifest$n_train)
n_test        <- as.integer(manifest$n_test)
# REML log-likelihood convergence tolerance (sommer's tolParConvLL).
# Optional in the manifest for backward compatibility; sommer's own
# default (1e-4) is used when absent.
reml_tol      <- if (is.null(manifest$tol)) 1e-4 else as.numeric(manifest$tol)
N             <- n_train + n_test
B             <- length(block_names)

env_brr_key <- if (is.null(manifest$env_brr_key)) {
  NULL
} else {
  as.character(manifest$env_brr_key)
}

if (B == 0) {
  stop("[R-GBLUP] ERROR: no kernel blocks listed in manifest$block_names")
}

# REML is deterministic, but set.seed() for parity with the BGLR path
# and in case any sommer internals touch the RNG (starting values).
set.seed(seed)

# ── Read y and train mask ────────────────────────────────────────────────────

y_full   <- read.csv(file.path(work_dir, "y_full.csv"))$y
is_train <- read.csv(file.path(work_dir, "train_mask.csv"))$is_train == 1

if (length(y_full) != N) {
  stop(sprintf(
    "[R-GBLUP] ERROR: y_full has %d rows, manifest says n_train+n_test=%d",
    length(y_full), N
  ))
}
if (length(is_train) != N) {
  stop(sprintf(
    "[R-GBLUP] ERROR: train_mask has %d rows, expected %d",
    length(is_train), N
  ))
}

tr <- which(is_train)
te <- which(!is_train)
n_tr <- length(tr)
if (n_tr != n_train) {
  stop(sprintf(
    "[R-GBLUP] ERROR: train_mask has %d train rows, manifest says n_train=%d",
    n_tr, n_train
  ))
}
y_tr <- y_full[tr]

# ── Read kernel blocks ───────────────────────────────────────────────────────

read_kernel_bin <- function(block_name) {
  path <- file.path(work_dir, "blocks", paste0(block_name, ".bin"))
  if (!file.exists(path)) {
    stop(sprintf("[R-GBLUP] ERROR: kernel file not found: %s", path))
  }
  con <- file(path, "rb")
  on.exit(close(con))
  vals <- readBin(con, what = "double", n = N * N, size = 8L,
                  endian = "little")
  if (length(vals) != N * N) {
    stop(sprintf(
      "[R-GBLUP] ERROR: kernel %s has %d doubles, expected %d (N=%d)",
      block_name, length(vals), N * N, N
    ))
  }
  # Python writes K row-major; K is symmetric by construction so the
  # byrow fill is exact (mirrors bglr_compute.R).
  matrix(vals, nrow = N, ncol = N, byrow = TRUE)
}

K_list <- vector("list", B)
names(K_list) <- block_names
for (k in seq_len(B)) {
  K_list[[k]] <- read_kernel_bin(block_names[k])
}

# ── Estimate variance components by REML (sommer) ─────────────────────────────

# Resolve the variance-structure helper across sommer versions: `vsr`
# (sommer >= 4.1) supersedes the older `vs`. Both take (factor, Gu=K).
vsf_name <- if ("vsr" %in% getNamespaceExports("sommer")) {
  "vsr"
} else if ("vs" %in% getNamespaceExports("sommer")) {
  "vs"
} else {
  stop("[R-GBLUP] ERROR: neither sommer::vsr nor sommer::vs is available")
}

# One random effect per kernel block. Each individual is its own factor
# level (one train obs per level) so the incidence Z is the identity and
# the random-effect covariance is exactly the train sub-kernel Gu =
# K_k[tr, tr] — i.e. textbook GBLUP. Distinct factor columns id1..idB
# (each over the same 1..n_tr levels) avoid sommer term-name collisions.
# The Gu_k matrices are bound in this top-level frame so mmer resolves
# them when it evaluates the random formula.
df <- data.frame(y = y_tr)
block_terms <- character(B)
level_labels <- as.character(seq_len(n_tr))
for (k in seq_len(B)) {
  Ktr <- K_list[[k]][tr, tr, drop = FALSE]
  rownames(Ktr) <- colnames(Ktr) <- level_labels
  assign(paste0("Gu_", k), Ktr)
  df[[paste0("id", k)]] <- factor(level_labels, levels = level_labels)
  block_terms[k] <- sprintf("%s(id%d, Gu = Gu_%d)", vsf_name, k, k)
}

random_formula <- as.formula(paste("~", paste(block_terms, collapse = " + ")))

# Assemble the mmer call. The REML iteration-cap argument was renamed
# `iters` → `nIters` across sommer versions; pick whichever this build
# exposes so the cap is honoured rather than silently dropped.
mmer_formals <- names(formals(sommer::mmer))
mmer_args <- list(
  fixed       = y ~ 1,
  random      = random_formula,
  rcov        = ~ units,
  data        = df,
  dateWarning = FALSE,
  verbose     = FALSE
)
iter_arg <- intersect(c("nIters", "iters"), mmer_formals)
if (length(iter_arg) > 0) {
  mmer_args[[iter_arg[1]]] <- reml_max_iter
}
# Convergence tolerance on the REML log-likelihood (renamed across
# sommer versions; set whichever this build exposes).
tol_arg <- intersect(c("tolParConvLL", "tolpar"), mmer_formals)
if (length(tol_arg) > 0) {
  mmer_args[[tol_arg[1]]] <- reml_tol
}

t0 <- Sys.time()
ans <- tryCatch(
  do.call(sommer::mmer, mmer_args),
  error = function(e) {
    cat(sprintf("[R-GBLUP] ERROR: sommer REML failed: %s\n", e$message))
    quit(status = 1)
  }
)

# Variance components: bind each sigma to its kernel BY NAME, not by
# position. Positional binding (sigma_k <- vcvals[seq_len(B)]) silently
# mis-maps if a sommer version returns varcomp reordered (by name or
# magnitude). Each block k is fitted with random term `vsr(id{k}, ...)`,
# and sommer names its varcomp row after that factor — e.g. "u:id1.y-y",
# with the residual (rcov = ~units) named "units.y-y". We match each
# kernel's `id{k}` token and the residual's `units` token against the
# rownames, and stop() loudly if any cannot be matched uniquely.
vc <- summary(ans)$varcomp
vcol <- if ("VarComp" %in% colnames(vc)) "VarComp" else colnames(vc)[1]
vc_names <- rownames(vc)
vcvals <- as.numeric(vc[[vcol]])
if (is.null(vc_names) || any(!nzchar(vc_names))) {
  stop(sprintf(
    "[R-GBLUP] ERROR: summary(ans)$varcomp lacks usable rownames; cannot map variance components to kernels by name (rows: %s)",
    paste(vc_names, collapse = ", ")
  ))
}
if (length(vcvals) != B + 1) {
  stop(sprintf(
    "[R-GBLUP] ERROR: expected %d variance components (%d kernels + residual), got %d",
    B + 1, B, length(vcvals)
  ))
}

# Residual row: the `~units` rcov term. Match its `units` token.
resid_pat <- "(^|[^[:alnum:]])units([^[:alnum:]]|$)"
resid_idx <- grep(resid_pat, vc_names)
if (length(resid_idx) != 1L) {
  stop(sprintf(
    "[R-GBLUP] ERROR: expected exactly one residual (units) variance component, matched %d in {%s}",
    length(resid_idx), paste(vc_names, collapse = ", ")
  ))
}

# Each block k -> its `id{k}` factor. Use a digit boundary after the token
# so `id1` does not spuriously match `id10`, `id11`, ...
matched <- logical(length(vc_names))
matched[resid_idx] <- TRUE
sigma_k <- numeric(B)
for (k in seq_len(B)) {
  tok <- sprintf("id%d", k)
  pat <- sprintf("(^|[^[:alnum:]])%s([^[:digit:]]|$)", tok)
  hit <- setdiff(grep(pat, vc_names), resid_idx)
  if (length(hit) != 1L) {
    stop(sprintf(
      "[R-GBLUP] ERROR: could not uniquely match variance component for kernel block '%s' (factor %s); matched %d row(s) in {%s}",
      block_names[k], tok, length(hit), paste(vc_names, collapse = ", ")
    ))
  }
  sigma_k[k] <- vcvals[hit]
  matched[hit] <- TRUE
}
if (!all(matched)) {
  stop(sprintf(
    "[R-GBLUP] ERROR: unmatched variance component row(s) after name mapping: {%s}",
    paste(vc_names[!matched], collapse = ", ")
  ))
}
sigma_k[!is.finite(sigma_k)] <- 0           # NaN/Inf component → drop it
sigma_k <- pmax(sigma_k, 0)                  # REML boundary can hit 0
sigma_e <- vcvals[resid_idx]
if (!is.finite(sigma_e) || sigma_e <= 0) {
  # Guard a degenerate / non-finite residual so V_tr stays SPD. Note
  # max(NaN, x) is NaN in R, so this must be a direct assignment rather
  # than max(sigma_e, floor).
  sigma_e <- 1e-8 * stats::var(y_tr)
}

reml_converged <- tryCatch(isTRUE(ans$convergence),
                           error = function(e) NA)

for (k in seq_len(B)) {
  cat(sprintf(
    "[R-GBLUP] Block %-16s sigma2=%.6g%s\n",
    block_names[k], sigma_k[k],
    if (!is.null(env_brr_key) && block_names[k] == env_brr_key) " [env/BRR]" else ""
  ))
}
cat(sprintf("[R-GBLUP] residual sigma2_e=%.6g  reml_converged=%s\n",
            sigma_e, as.character(reml_converged)))

# ── BLUP / kriging prediction over all N rows ─────────────────────────────────

# V_tr = sum_k sigma2_k K_k[tr, tr] + sigma2_e I   (n_tr × n_tr, SPD).
V_tr <- matrix(0, n_tr, n_tr)
for (k in seq_len(B)) {
  if (sigma_k[k] > 0) {
    V_tr <- V_tr + sigma_k[k] * K_list[[k]][tr, tr, drop = FALSE]
  }
}
diag(V_tr) <- diag(V_tr) + sigma_e

# Cholesky solve; add a tiny ridge if V_tr is numerically indefinite.
U <- tryCatch(
  chol(V_tr),
  error = function(e) {
    jit <- 1e-6 * mean(diag(V_tr))
    cat(sprintf("[R-GBLUP] V_tr chol failed; adding ridge %.3g\n", jit))
    chol(V_tr + diag(jit, n_tr))
  }
)
solveV <- function(b) backsolve(U, forwardsolve(t(U), b))

ones <- rep(1.0, n_tr)
Vinv_y <- solveV(y_tr)
Vinv_1 <- solveV(ones)
mu_hat <- sum(Vinv_y) / sum(Vinv_1)            # GLS intercept
a <- solveV(y_tr - mu_hat)                     # V_tr^{-1} (y_tr - 1 mu_hat)

# yhat_i = mu_hat + sum_k sigma2_k K_k[i, tr] %*% a, for ALL rows i.
# The test-row predictions ride entirely on the K_k[te, tr] cross-blocks.
g_full <- numeric(N)
for (k in seq_len(B)) {
  if (sigma_k[k] > 0) {
    g_full <- g_full + sigma_k[k] * as.numeric(K_list[[k]][, tr, drop = FALSE] %*% a)
  }
}
yhat <- mu_hat + g_full

elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

# ── Write outputs ────────────────────────────────────────────────────────────

write.csv(data.frame(y_hat = yhat),
          file.path(work_dir, "yhat.csv"), row.names = FALSE)

# Per-term variance, mirroring bglr_meta's `terms` schema so fpca_core
# can stay regressor-agnostic. `var` is the REML variance component
# sigma2_k; `model` labels the env term to match the Bayesian path's
# BRR(Z_e). `dim` is left null: BGLR reports the eigen rank it actually
# uses, but GBLUP's dense REML never eigen-decomposes the kernel, and
# computing the rank purely for the meta would cost a full O(n^3)
# decomposition per block — not worth it.
terms_meta <- vector("list", B)
for (k in seq_len(B)) {
  is_env <- !is.null(env_brr_key) && block_names[k] == env_brr_key
  terms_meta[[k]] <- list(
    name  = block_names[k],
    var   = sigma_k[k],
    dim   = NA,
    model = if (is_env) "BRR" else "RKHS"
  )
}

meta <- list(
  varE           = sigma_e,
  terms          = terms_meta,
  reml_max_iter  = reml_max_iter,
  reml_converged = reml_converged,
  mu             = mu_hat,
  seed           = seed,
  elapsed_s      = elapsed,
  method         = sprintf("sommer_reml(%s)", vsf_name)
)

writeLines(
  toJSON(meta, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null"),
  file.path(work_dir, "gblup_meta.json")
)

cat(sprintf("[R-GBLUP] Done in %.2f s\n", elapsed))
