# Leave-one-environment-out prediction workflow

This directory implements leakage-safe DAP and AGDD FPCA projection, kernel
construction, and no-Ze BGLR prediction. NGRDI and PTR are the manuscript
defaults because they were selected a priori.

## Output layout

R scripts read `config/paths.yml` through `R/utils/paths.R`. Unless overridden,
each domain writes beneath `<results_dir>/prediction/loeo/<domain>/`:

- `projected_scores/`: leakage-safe VI and weather FPC scores
- `kernels/noZe/models/`: no-Ze BGLR model bundles
- `predictions/`: prediction values and environment-level correlations
- `work/`: BGLR temporary files

The same paths can be overridden on HPRC with `G2F_DATA_PATH`,
`G2F_LOEO_SCORE_PATH`, `G2F_KERNEL_OUT_PATH`, `G2F_RESULTS_PATH`, and
`G2F_WORK_PATH`.

## Execution order

1. Run `project_dap_fpca.R` and `project_agdd_fpca.R`. Each script fits FPCA
   bases without the held-out environment, projects the held-out curves, and
   writes one complete score file per domain. In particular, the AGDD weather
   output is written directly as
   `Weather_FPC_Scores_AGDD_LOEO_Projected.csv`.
2. Submit `setup_dap_array.sh` and `setup_agdd_array.sh` to build model bundles
   for every held-out environment.
3. Submit `submit_dap_noze.sh` and `submit_agdd_noze.sh` to fit the 14 models
   listed in `prediction_job_array.txt`.
4. Run the stage 8 result compilers. They require the complete set of 266
   environment-level correlation files per domain; the unified summarizer also
   requires all 266 prediction-value files.

The launchers contain resource requests but no personal account, email, or
filesystem locations. Override paths through `config.sh` environment variables.
The prediction stage requires `BGLR_Mod_4.4.2` with the repository's documented
single-predictor `drop = FALSE` patch.
