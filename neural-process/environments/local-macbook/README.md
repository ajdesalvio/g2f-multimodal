# Local MacBook — Development / Debugging Environment

Package versions on the developer workstation used for **writing,
debugging, and smoke-running** this repository. This is *not* the environment
that produced any published metrics — production runs happen on HPRC
Grace (see `../hprc-grace/`). Use this snapshot to reproduce local
smoke runs and to understand which versions drift from Grace.

## Files

- `sysinfo.txt` — macOS version, arch, Python interpreter location.
- `requirements.txt` — Python packages from the local dev venv,
  filtered to this project's dependency closure (the venv is shared
  across projects; unrelated packages are omitted).
- `r-packages.txt` — Direct R dependencies (`fdapace`, `optparse`,
  `BGLR`, `sommer`, `jsonlite`) and their versions from the system R
  install at `/usr/local/bin/R`.

## Scope of this environment

Used for:
- Writing code, CPU debugging, and local smoke runs of the FPCA and
  TNP pipelines (the R steps use the local `fdapace` / `BGLR` / `sommer`).

Not used for:
- Any reported result (LOO CV or the CV0/CV00/CV1/CV2 sweeps). All of
  those come from Grace under `../hprc-grace/`.
- GPU training (Mac has no NVIDIA GPU — all local runs are CPU-only or
  MPS fallback).

## Known differences vs. Grace

These are the packages whose versions differ between this Mac dev env
and the `hprc-grace` pin. Differences are benign for correctness but
worth knowing when debugging behavior that differs between the local
machine and Grace:

| Package | Local MacBook | HPRC Grace | Notes |
|---|---|---|---|
| Python | **3.13.1** | 3.11.5 | Grace tied to the `Python/3.11.5` Lmod module |
| macOS / OS | macOS 26.5.2 arm64 | RHEL 8.10 | — |
| torch | 2.6.0 (MPS) | 2.5.1 (CUDA 12.4) | Different accelerator backends |
| scikit-learn | 1.6.1 | 1.6.0 | Minor |
| scipy | 1.15.1 | 1.14.1 | Minor |
| numpy | 2.2.2 | 2.2.1 | Patch version |
| lightning-utilities | 0.12.0 | 0.11.9 | — |
| R | 4.5.2 | 4.4.1 | Both have `fdapace 0.6.0` + `optparse 1.7.5` (FPCA + CLI parsing). The `bglr` baselines also use `BGLR` — see the weather-GxE note below |

## Running weather-GxE (`bglr`) baselines locally

`r-packages.txt` lists `fdapace`, `optparse`, `BGLR`, `sommer`, and `jsonlite` with
their pinned versions, but the pin alone is not enough to run the **weather-GxE**
`bglr` baselines locally: `BGLR` must be the **rank-1-patched** build,
since stock CRAN BGLR 1.1.4 aborts with `invalid 'times' argument` on
weather-GxE's rank-1 weather kernel. That patch is
*environment state* (it lives in the installed BGLR build, not in tracked
source, and the version string still reads 1.1.4), so `r-packages.txt`
can only note it in a comment. Install it with `scripts/patch_bglr.sh` — see
[`../../baselines/R_HPC_SETUP.md`](../../baselines/R_HPC_SETUP.md)
§"Applying the BGLR patch on a local / non-HPC machine", which also
covers the macOS Homebrew-gfortran `FLIBS` gotcha. env-GxE and the non-BGLR
steps run unpatched.

## Shared-venv packages omitted from the pin

The dev venv is shared with other projects, so it also contains linting
and formatting tools, notebook tooling, and unrelated side-project
packages (jax, gpytorch, matplotlib, seaborn, sktime, ...). None of them
are needed to run this repository, and `requirements.txt` deliberately
omits them — it lists only the dependency closure of `pyproject.toml`.

## Regenerating these files

From the dev machine with the same venv active:

```bash
sw_vers > environments/local-macbook/sysinfo.txt
echo "arch: $(uname -m)" >> environments/local-macbook/sysinfo.txt
echo "python: $(which python) ($(python --version))" >> environments/local-macbook/sysinfo.txt
pip freeze > environments/local-macbook/requirements.txt
# then filter to this project's dependency closure (pyproject.toml
# direct deps and their transitive requirements) — the shared venv
# contains unrelated projects' packages
Rscript -e 'for (p in c("BGLR","fdapace","jsonlite","optparse","sommer"))
            cat(p, as.character(packageVersion(p)), "\n")' \
            > environments/local-macbook/r-packages.txt   # header comments are hand-maintained
```
