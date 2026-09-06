#!/usr/bin/env bash
# Re-install BGLR 1.1.4 with the rank-1 RKHS patch (LT$V[, tmp, drop = FALSE]).
#
# WHY: stock CRAN BGLR 1.1.4 aborts with "invalid 'times' argument" whenever a
# kernel block reduces to a single super-tolD eigenvalue (a rank-1 kernel) —
# its RKHS handler does `LT$V = LT$V[, tmp]` and R drops the 1-column matrix to
# a vector, so the outer product fails. weather-GxE hits this: its weather kernel
# K_W is rank-1 when built from one PTR weather FPC (
# weather_fpca.n_components=1). See
# baselines/R_HPC_SETUP.md and baselines/bglr_rank1.patch.
#
# RUN with the HPRC R module stack loaded and R_LIBS_USER pointing at the
# project R library, e.g.:
#   module load GCC/13.2.0 R/4.4.1
#   export R_LIBS_USER=/scratch/user/$USER/neural-processes/env/r_libs
#   bash scripts/patch_bglr.sh
#
# Idempotent: rebuilds from a fresh CRAN download each run. Re-run after ANY
# `install.packages("BGLR")` or R-env rebuild, or weather-GxE silently breaks again
# (the patch lives in R_LIBS_USER, not in the CRAN version pin).
set -euo pipefail

BGLR_VERSION="1.1.4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_FILE="${REPO_ROOT}/baselines/bglr_rank1.patch"

: "${R_LIBS_USER:?Set R_LIBS_USER to the project R library dir (e.g. /scratch/user/$USER/neural-processes/env/r_libs)}"
command -v Rscript >/dev/null \
  || { echo "ERROR: Rscript not on PATH — load the R module first (e.g. module load GCC/13.2.0 R/4.4.1)."; exit 1; }
[[ -f "${PATCH_FILE}" ]] || { echo "ERROR: patch file not found: ${PATCH_FILE}"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
cd "${WORK}"

echo "[patch_bglr] Downloading BGLR ${BGLR_VERSION} source from CRAN ..."
Rscript -e "options(repos=c(CRAN='https://cloud.r-project.org')); download.packages('BGLR', destdir='.', type='source')" >/dev/null
TARBALL="$(ls BGLR_*.tar.gz | head -1)"
echo "[patch_bglr] Got ${TARBALL}"
if [[ "${TARBALL}" != "BGLR_${BGLR_VERSION}.tar.gz" ]]; then
  echo "[patch_bglr] WARNING: CRAN served ${TARBALL}, not BGLR_${BGLR_VERSION}.tar.gz."
  echo "             The patch targets the rank-1 line by content and should still"
  echo "             apply, but review the installed result before trusting weather-GxE."
fi

tar xzf "${TARBALL}"
TARGET="BGLR/R/BGLR.R"

PATCHED_RE="LT\$V = LT\$V\[, tmp, drop = FALSE\]"
if grep -q "${PATCHED_RE}" "${TARGET}"; then
  echo "[patch_bglr] Source already patched (unexpected for a fresh download) — continuing."
elif patch -p1 -d BGLR --forward --ignore-whitespace < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[patch_bglr] Applied ${PATCH_FILE} via patch(1)."
else
  echo "[patch_bglr] patch(1) did not apply cleanly; falling back to content replace."
  perl -0777 -pi -e 's/LT\$V = LT\$V\[, tmp\]/LT\$V = LT\$V[, tmp, drop = FALSE]/' "${TARGET}"
fi

grep -q "${PATCHED_RE}" "${TARGET}" \
  || { echo "ERROR: patch did not take — '${TARGET}' still has the unpatched line."; exit 1; }
echo "[patch_bglr] Verified patched line in ${TARGET}."

echo "[patch_bglr] Installing into R_LIBS_USER=${R_LIBS_USER} ..."
mkdir -p "${R_LIBS_USER}"
R CMD INSTALL -l "${R_LIBS_USER}" BGLR

echo "[patch_bglr] Verifying a rank-1 RKHS fit ..."
R_LIBS_USER="${R_LIBS_USER}" Rscript -e '
  suppressPackageStartupMessages(library(BGLR))
  z <- rnorm(200); K <- tcrossprod(z); e <- eigen(K, symmetric = TRUE)
  # saveAt into a tempdir so this check never litters mu.dat/varE.dat/ETA_*.dat
  # into the caller cwd (BGLR default saveAt="" writes to getwd()).
  fit <- BGLR(y = rnorm(200),
              ETA = list(list(V = e$vectors, d = e$values, model = "RKHS")),
              nIter = 40, burnIn = 10, verbose = FALSE,
              saveAt = file.path(tempdir(), "bglr_verify_"))
  cat("[patch_bglr] rank-1 RKHS OK (BGLR", as.character(packageVersion("BGLR")), "patched)\n")'

echo "[patch_bglr] Done. weather-GxE (rank-1 K_W) will now fit."
