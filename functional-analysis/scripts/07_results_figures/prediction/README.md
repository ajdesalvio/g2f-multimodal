# Prediction summaries

Run from `functional-analysis/` after configuring the two local roots:

```sh
Rscript scripts/07_results_figures/prediction/summarize_all_prediction_results.R
```

This is the primary entry point. It implements the finalized V6 calculation
and writes four tables below `results_dir/07_results_figures/final_prediction_summaries`.
The checked-in copies are in `results/prediction/final_summaries/`.

## Required inputs and calculations

- BGLR: 14 models, DAP and AGDD, 10 seeds × 5 folds for CV, and 19 LOEO environments.
- TNP: September 6 matched-fold metrics, 3 configurations, 5 seeds × 5 folds.
- CV0/CV00: inverse-variance (Tiezzi) pooling of within-environment correlations;
  observation-weighted pooling of squared errors for RMSE.
- CV1/CV2: within-environment correlations pooled within each seed/fold; RMSE
  computed over scored records. Average folds within seeds, then seeds.
- LOEO: correlations from the retained calculation workbook; RMSE from only
  the held-out environment in each prediction file. Training rows are excluded.

All primary kernel results are no-Ze. CV2 is in-sample. Paired comparisons with
TNP use only shared seeds 1–5, not all 10 BGLR seeds.

`compile_cv_results.R` optionally exports the unaggregated CV0/CV00 metric
table. `compile_loeo_results.R` exports the per-environment correlation table
underlying the retained LOEO workbook. Neither is required before the primary
summary when using the deposited task-level results and workbook.
