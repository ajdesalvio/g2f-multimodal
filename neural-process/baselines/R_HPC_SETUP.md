# Installing R Packages on HPC (TAMU Grace)

> **Looking for the exact versions used by this repository's published runs?**
> See [`../environments/hprc-grace/r-packages.txt`](../environments/hprc-grace/r-packages.txt)
> for the full pin (`fdapace 0.6.0`, `optparse 1.7.5`, `BGLR 1.1.4`,
> `jsonlite 2.0.0`, plus transitive dependencies). The instructions
> below are the **install recipe** for setting up a fresh `r_libs`
> directory; the `environments/` snapshot is the **outcome** of running
> that recipe at a specific date.

## Prerequisites

This guide assumes:
- Your Python virtual environment is built against `GCCcore/13.2.0`
- Your project is located at `/scratch/user/$USER/neural-processes/`

---

## Step 1: Load Modules

```bash
module load GCC/13.2.0
module load R/4.4.1
```

> **Note:** `R/4.4.1` requires `GCC/13.2.0`, which subsumes `GCCcore/13.2.0` and is
> therefore compatible with your existing Python virtual environment.
> Do **not** use `R/4.4.2` as it requires `GCC/13.3.0`, which conflicts with the Python venv.

---

## Step 2: Create R Library Directory

Place the R library inside the project virtual environment folder to keep everything self-contained:

```bash
mkdir -p /scratch/user/$USER/neural-processes/env/r_libs
```

---

## Step 3: Install Packages

```bash
Rscript -e "install.packages('package_name', repos='https://cran.r-project.org', lib='/scratch/user/$USER/neural-processes/env/r_libs')"
```

> **Note:** The explicit `lib=` argument tells R where to install the package.
> `R_LIBS_USER` does **not** need to be set at install time when `lib=` is specified.

### `fdapace`

```bash
Rscript -e "install.packages('fdapace', repos='https://cran.r-project.org', lib='/scratch/user/$USER/neural-processes/env/r_libs')"
```

> Installation can take **5–15 minutes** due to C++ compilation (Rcpp/RcppEigen).

### `optparse`

```bash
Rscript -e "install.packages('optparse', repos='https://cran.r-project.org', lib='/scratch/user/$USER/neural-processes/env/r_libs')"
```

> Pure R package — installs in seconds.

### `BGLR`

```bash
Rscript -e "install.packages('BGLR', repos='https://cran.r-project.org', lib='/scratch/user/$USER/neural-processes/env/r_libs', dependencies=TRUE)"
```

> Installation can take **5+ minutes** due to compiled deps. CRAN
> pulls `truncnorm` and `pROC` automatically; `MASS` is part of base
> R recommended packages and is already present from the `fdapace`
> transitive set. Used by `baselines/bglr_compute.R` (called from the
> `bglr` regressor in `baselines/regressors.py`) to run the
> multi-kernel Bayesian RKHS Gibbs sampler that mirrors the R reference's
> reference setup.

#### Patch BGLR for rank-1 RKHS terms (REQUIRED for weather-GxE)

> ⚠️ **Stock CRAN BGLR 1.1.4 cannot fit a rank-1 kernel.** It aborts with
> `invalid 'times' argument` whenever a kernel block reduces to a **single**
> super-`tolD` eigenvalue. weather-GxE triggers this: its weather kernel `K_W` is
> rank-1 when built from one PTR weather FPC (
> `weather_fpca.n_components=1`). Root cause: BGLR's RKHS handler does
> `LT$V = LT$V[, tmp]`, and when `tmp` selects one column R silently drops the
> 1-column matrix to a vector, so the downstream outer product fails. (The
> reference R pipeline carries the same patch.)

Re-install BGLR 1.1.4 with the one-line `drop = FALSE` fix. The repo ships an
automated installer — from the repo root with the module stack loaded and
`R_LIBS_USER` exported:

```bash
bash scripts/patch_bglr.sh
```

It downloads BGLR 1.1.4, applies [`bglr_rank1.patch`](bglr_rank1.patch)
(`R/BGLR.R`: `LT$V = LT$V[, tmp]` → `LT$V = LT$V[, tmp, drop = FALSE]`),
`R CMD INSTALL`s it into `R_LIBS_USER` (replacing the stock build), and runs a
rank-1 smoke fit. Verify manually any time with:

```bash
Rscript -e 'library(BGLR); z<-rnorm(200); K<-tcrossprod(z); e<-eigen(K,symmetric=TRUE);
  BGLR(y=rnorm(200), ETA=list(list(V=e$vectors,d=e$values,model="RKHS")),
       nIter=40, burnIn=10, verbose=FALSE,
       saveAt=file.path(tempdir(),"bglr_verify_")); cat("rank-1 RKHS OK\n")'
```

Stock BGLR errors here; the patched build prints `rank-1 RKHS OK`.

> **The patch is not in the version pin.** `environments/hprc-grace/r-packages.txt`
> still reports `BGLR 1.1.4`, and the patch lives only inside `R_LIBS_USER`.
> Re-run `scripts/patch_bglr.sh` after **any** `install.packages("BGLR")` or
> R-environment rebuild, or weather-GxE will silently start failing in BGLR again.

#### Applying the BGLR patch on a local / non-HPC machine (laptop, CI, reviewer)

The rank-1 patch is **environment state, not tracked source** — it lives in an
installed R package, so cloning the repo is *not* enough to run weather-GxE locally.
Stock CRAN BGLR fails the same way off the cluster. env-GxE and the non-BGLR steps
work unpatched; only **weather-GxE** (rank-1 `K_W`) needs this.

`scripts/patch_bglr.sh` is cluster-agnostic — it only requires `Rscript` on
`PATH` and `R_LIBS_USER` set to *some* writable library dir. The cleanest
target is **R's own default user library**, because R auto-prepends it to
`.libPaths()`, so the patched build shadows the stock one with **no env var
exported at run time** (matching how `baselines/*` invoke `Rscript` as a
subprocess):

```bash
# 1. Discover R's default user library (create it if missing):
export R_LIBS_USER="$(Rscript -e 'cat(Sys.getenv("R_LIBS_USER"))')"
mkdir -p "$R_LIBS_USER"

# 2. Install the patched BGLR there:
bash scripts/patch_bglr.sh

# 3. Verify with a PLAIN Rscript (no env overrides — proves auto-pickup):
Rscript -e 'cat(find.package("BGLR")); library(BGLR);
  z<-rnorm(200); K<-tcrossprod(z); e<-eigen(K,symmetric=TRUE);
  BGLR(y=rnorm(200), ETA=list(list(V=e$vectors,d=e$values,model="RKHS")),
       nIter=40, burnIn=10, verbose=FALSE,
       saveAt=file.path(tempdir(),"bglr_verify_")); cat("\nrank-1 RKHS OK\n")'
```

`find.package("BGLR")` should now point inside `$R_LIBS_USER`, and the fit
prints `rank-1 RKHS OK`.

> **macOS gotcha — `ld: library 'emutls_w' not found` during install.**
> CRAN's R for macOS hard-codes `FLIBS` to the *official* CRAN gfortran at
> `/opt/gfortran/lib` (with CRAN-only runtime libs `-lemutls_w -lheapt_w`). If
> your gfortran is from **Homebrew** instead (`gcc` formula), those paths don't
> exist and the BGLR link step fails. Fix it for this one install with a
> throwaway Makevars that points `FLIBS` at the Homebrew gfortran — **no global
> R config change**:
>
> ```bash
> # adjust the path if `gfortran -print-file-name=libgfortran.dylib` differs
> printf 'FLIBS = -L/opt/homebrew/lib/gcc/current -lgfortran -lquadmath\n' > /tmp/bglr_makevars
> R_MAKEVARS_USER=/tmp/bglr_makevars R_LIBS_USER="$R_LIBS_USER" bash scripts/patch_bglr.sh
> ```
>
> The CRAN-only `-lemutls_w`/`-lheapt_w` are unnecessary with Homebrew gcc and
> are simply dropped. (Verified on Apple-silicon, R 4.5.2 + Homebrew gcc 15.)

### `jsonlite`

```bash
Rscript -e "install.packages('jsonlite', repos='https://cran.r-project.org', lib='/scratch/user/$USER/neural-processes/env/r_libs')"
```

> Pure R package — installs in seconds. Used by `bglr_compute.R` for
> the `manifest.json` / `bglr_meta.json` interchange between Python
> and R. Also used by `gblup_compute.R`.

### `sommer`

```bash
Rscript -e "install.packages('sommer', repos='https://cran.r-project.org', lib='/scratch/user/$USER/neural-processes/env/r_libs', dependencies=TRUE)"
```

> Installation can take **5+ minutes** due to compiled deps (`Rcpp`,
> `RcppArmadillo`). Used by `baselines/gblup_compute.R` (called from the
> `gblup` regressor in `baselines/regressors.py`) to estimate the
> multi-kernel REML variance components — the frequentist twin of the
> `bglr` Gibbs sampler. No source patch is required: GBLUP handles
> rank-1 kernels (weather-GxE's `K_W`) natively, so the BGLR rank-1 patch above
> is **not** needed for `gblup` runs. `gblup_compute.R` resolves the
> variance-structure helper across versions (`vsr` on sommer ≥ 4.1,
> falling back to `vs`); pin sommer ≥ 4.1 to match the tested path.

A successful install ends with:
```
* DONE (package_name)
```

---

## Step 4: Verify Installation

```bash
export R_LIBS_USER=/scratch/user/$USER/neural-processes/env/r_libs
Rscript -e "library(fdapace); cat('fdapace loaded OK\n')"
Rscript -e "library(optparse); cat('optparse loaded OK\n')"
Rscript -e "library(BGLR); cat('BGLR loaded OK\n')"
Rscript -e "library(jsonlite); cat('jsonlite loaded OK\n')"
Rscript -e "library(sommer); cat('sommer loaded OK\n')"
```

---

## Step 5: Configure Your Job Script

Add the following to your SLURM job script before running any Python code that invokes R:

```bash
#!/bin/bash
#SBATCH ...

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module purge
module load GCC/13.2.0
module load Python/3.11.5
module load WebProxy/0000
module load git/2.42.0
module load R/4.4.1

export R_LIBS_USER=/scratch/user/$USER/neural-processes/env/r_libs
source /scratch/user/$USER/neural-processes/env/bin/activate
```

> **Important:** `R_LIBS_USER` must be exported **before** running your Python script,
> otherwise R will not find the installed packages when called via `subprocess`.

---

## Project Directory Structure

After setup, your `env/` folder will look like:

```
env/
├── bin/
├── include/
├── lib/
├── lib64/
├── r_libs/         ← R packages live here
├── pyvenv.cfg
└── share/
```

---

## Notes

- **Compilation warnings** about `ignoring attributes on template argument '__m512d'`
  are harmless Eigen/AVX-512 mismatches. They do not affect correctness.
- **R packages and Python packages are separate ecosystems.** R packages cannot
  be installed inside a Python virtual environment.
- `GCCcore` vs `GCC`: `GCCcore` provides only the compiler binaries and is used
  as a base for tools like Python. `GCC` is the full toolchain (including OpenMP
  and Fortran runtime) required by R and compiled R packages.

---

## Disk: phenomic cache footprint after the BGLR rollout

A `fit_scope=train` (inductive, e.g. `ols`) and a `fit_scope=all`
(transductive, `bglr`/`gblup`) variant of the same model produce
**separate** phenomic cache entries — `fit_scope` is part of the cache
key. Both end up on disk if both variants run on the same fold.

V1 measurement on DEH1.2020 (K=4 vi_fpca, the production
phenomic config):

| Cache entry | `fit_scope` | N | Size |
| ----------- | ----------- | ---- | ----- |
| `phenomic_4e5f8d66f8250136.npz` | `train` (inductive)    | 8,967 | **663 MB** |
| `phenomic_242c7a263fc0d81f.npz` | `all` (transductive)   | 10,109 | **840 MB** |

Ratio 1.27× (≈ (10109/8967)² ≈ 1.27, matches the N² scaling — the
on-disk size is dominated by the centered kernel `K_centered`,
which is N×N float64 ≈ 8N² bytes).

Per-fold combined: **~1.5 GB**.

Across the 19 LOEO folds, once both ablations have populated their
caches on every fold:

- `fit_scope=train` phenomic cache: 19 × ~663 MB ≈ **12.6 GB**
- `fit_scope=all` phenomic cache:   19 × ~840 MB ≈ **16.0 GB**
- **Combined incremental footprint: ~28.6 GB**

If only the BGLR ablation runs (not both), the impact is just the
~16 GB row.

Multiple variants reuse the same cache entry when their phenomic
config is bit-identical — all phenomic-bearing `_bglr_` configs
in the production family share one `fit_scope=all` entry per
fold, so the row above is **one entry per fold**, not per variant.

If your Grace `$SCRATCH` quota is tight, the ~16 GB BGLR-side
delta is the number to compare against. For typical TAMU 1 TB
quotas this is well within rounding error; document the headroom
before pushing the BGLR rollout into a long-running cron.
