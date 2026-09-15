# Cross-validation prediction workflows

This directory contains the DAP- and AGDD-domain prediction workflows used for
the manuscript. Both use NGRDI and PTR, which were selected a priori. The
manuscript result is the no-Ze analysis; `G2F_INCLUDE_ZE=FALSE` is therefore the
default in both domain configurations.

The finalized comparison uses 223 maternal lines present in all 19 environments.
The canonical list is in `data/supplementary/Common_Females.csv`; metadata
preparation checks it against the phenotype input. Both domains use the same
R-generated fold assignments (10 seeds × 5 folds). TNP paired comparisons use
the shared subset of seeds 1–5.

## Configuration

Local R runs read `config/paths.yml` through `R/utils/paths.R`. HPRC launchers
also accept environment overrides:

- `G2F_PROJECT_DIR`: the `functional-analysis/` directory within the repository
- `G2F_DATA_PATH`: directory containing the Dryad input files
- `G2F_CV_OUT_PATH`: domain-specific output root
- `G2F_LOEO_SCORE_PATH`: projected weather-score directory produced by the
  matching LOEO workflow
- `G2F_COMMON_FEMALES_FILE`: optional override of the canonical maternal-line CSV
- `G2F_R_ENV_NAME`: patched R environment; defaults to `BGLR_Mod_4.4.2`
- `G2F_VI_NAMES=NGRDI` and `G2F_WEATHER_TRAITS=PTR`

Account and email options are intentionally not embedded. Supply them to
`sbatch` or through the user's HPRC defaults.

## Execution order

For either `dap/` or `agdd/`:

1. Submit `prepare_workflow.sh` to build metadata, constant kernels, weather
   inputs, and the parameter tables.
2. Submit `setup_cv21_array.sh` and `setup_cv000_array.sh` to build the
   split-specific FPCA and kernel bundles.
3. Submit `submit_cv21_noze.sh` and `submit_cv000_noze.sh` for the initial
   no-Ze prediction runs.
4. Submit `audit_noze_results.sh` to verify completeness.
5. If the audit creates restart tables, submit the corresponding
   `submit_*_noze_missing.sh` launchers.

The AGDD preprocessing output is written below `G2F_CV_OUT_PATH/derived_data`;
the original Dryad input directory is not modified.

Prediction rows and RMSE are saved by default. Every model fit receives a
deterministic seed based on its seed/fold/environment/model identity. FPCA
bases are fitted to training records and held-out records are projected.
Phenomic scores are standardized within each environment using all available
predictor records; weather-score scaling uses training environments. These
are the procedures used for the released results.

## Patched BGLR requirement

PTR contributes one predictor. Prediction scripts verify that BGLR contains
the documented `drop = FALSE` fix before fitting models. See
[the patch and installation instructions](../../patches/README.md).
