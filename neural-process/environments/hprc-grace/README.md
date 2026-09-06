# HPRC Grace — Environment Pin

Package versions used by this repository on the **Texas A&M HPRC Grace**
cluster for the LOO cross-validation and multi-modal training/evaluation
runs. These files are **not** a generic `requirements.txt` — they document
the exact software stack that produced the reported metrics (written under
`artifacts/`, which is generated at runtime and not part of this repo).
Last re-verified against the live cluster environment on 2026-08-19.

If you are running on a different cluster (Launch, FASTER, Terra, etc.)
or a local workstation, treat the version numbers here as a known-good
baseline, not a strict requirement.

## Files

- `modules.txt` — Lmod modules loaded before activating the Python venv.
- `requirements.txt` — Python packages from the virtual environment at
  `/scratch/user/$USER/neural-processes/env/`, filtered to this project's
  dependency closure (the venv is shared with unrelated projects; their
  packages are omitted).
- `r-packages.txt` — Installed R packages (and their versions) in
  `R_LIBS_USER=/scratch/user/$USER/neural-processes/env/r_libs`,
  loaded on top of the system R from the `R/4.4.1` module.

## Minimum module stack to reproduce

This is what the job scripts in `scripts/jobs/cv/env_year_loo/*.job` load:

```bash
module purge
module load GCC/13.2.0          # toolchain (also required for R)
module load Python/3.11.5       # Python interpreter
module load WebProxy/0000       # outbound network (for W&B, pip, etc.)
module load git/2.42.0
module load R/4.4.1              # provides Rscript used by vi_fpca / weather_fpca
export R_LIBS_USER=/scratch/user/$USER/neural-processes/env/r_libs
source /scratch/user/$USER/neural-processes/env/bin/activate
```

## Hardware

- **OS**: Red Hat Enterprise Linux 8.10 (Ootpa), x86_64.
- **GPU jobs** (`dl_job.job`): `--partition=gpu --gres=gpu:1` with no GPU
  model constraint. Grace's `gpu` partition is A100-dominated (100 nodes
  with 2× A100 each, vs 9 RTX 6000 and 8 T4 nodes), so production DL runs
  landed on A100s. PyTorch is the CUDA 12.4 wheel build (`nvidia-*-cu12`).
- **FPCA/BGLR/GBLUP jobs** (`fpca_job.job`): CPU-only — no GPU requested;
  R is the bottleneck, not CUDA.

The `GCCcore/13.2.0` stack is incompatible with the `R/4.4.1` module
(it depends on the full `GCC/13.2.0` toolchain, not `GCCcore`), so all
`*.job` scripts (which load R) use `GCC/13.2.0` uniformly. The
`submit_*.sh` driver scripts only load `GCCcore/13.2.0` + Python for
slug pre-computation and never touch R, so they keep `GCCcore`.

## Key dependencies worth calling out

| Component | Version | Notes |
|---|---|---|
| Python | 3.11.5 | HPRC module |
| PyTorch | 2.5.1 (CUDA 12.4) | CUDA wheel build (`nvidia-*-cu12`) |
| Lightning | 2.5.0.post0 | Both `lightning` and `pytorch-lightning` metapackages installed |
| Hydra | 1.3.2 | Config composition |
| OmegaConf | 2.3.0 | Backing config engine |
| NumPy | 2.2.1 | — |
| pandas | 2.2.3 | — |
| scikit-learn | 1.6.0 | FPCA baseline regressors |
| W&B | 0.25.1 | Logging; local dir redirected to `/tmp/wandb` |
| R | 4.4.1 | HPRC module |
| fdapace | 0.6.0 | Sparse FPCA — used by `vi_fpca`, `weather_fpca` |
| BGLR | 1.1.4 **(patched)** | Multi-kernel Bayesian RKHS. Carries the rank-1 `drop = FALSE` source patch (apply via `scripts/patch_bglr.sh`) so weather-GxE's rank-1 `K_W` fits instead of aborting with `invalid 'times' argument`; re-apply after any re-install. See `../../baselines/R_HPC_SETUP.md`. |
| optparse | 1.7.5 | R CLI parsing in `fpca_compute.R` / `bglr_compute.R` / `gblup_compute.R` |
| sommer | 4.4.5 | Multi-kernel REML/BLUP (`sommer::mmer`) — the `gblup` regressor engine (`gblup_compute.R`) |
| jsonlite | 2.0.0 | Manifest/metadata interchange between the Python regressor wrappers and the R scripts |

## Regenerating these files

From a Grace login or compute node with the full module stack loaded and
the venv activated:

```bash
module list 2>&1 > environments/hprc-grace/modules.txt
pip freeze       > environments/hprc-grace/requirements.txt
# then FILTER requirements.txt to this project's dependency closure —
# the venv is shared, so a raw freeze picks up unrelated projects'
# packages (jax, xarray, Cartopy, ...); keep only pyproject.toml's
# direct deps and their transitive requirements
Rscript -e 'ip <- installed.packages(lib.loc=Sys.getenv("R_LIBS_USER"));
            df <- data.frame(Package=ip[,"Package"], Version=ip[,"Version"]);
            df <- df[order(df$Package),];
            write.table(df, "environments/hprc-grace/r-packages.txt",
                        quote=FALSE, row.names=FALSE, sep="  ")'
```

Then commit the updated files together. Do **not** regenerate these in
isolation across partial environments — if you only update `requirements.txt`
after changing one package without refreshing the R lib listing, the pin
will drift from "known-good snapshot" to "a moment in someone's shell".

> **BGLR is patched, and regeneration cannot capture that.** The installed
> `BGLR 1.1.4` carries the rank-1 RKHS source patch
> (`baselines/bglr_rank1.patch`, applied by `scripts/patch_bglr.sh`).
> `installed.packages()` only records the version string, so regenerating
> `r-packages.txt` drops the "PATCHED BUILD" annotation — re-add it by hand,
> and re-run `scripts/patch_bglr.sh` after any BGLR re-install. Without the
> patch, weather-GxE aborts in BGLR with `invalid 'times' argument`.
