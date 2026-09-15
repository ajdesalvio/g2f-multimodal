# Prediction results

Start with [the final prediction summary](final_summaries/G2F_Final_Prediction_Summary.csv):
152 rows containing all primary BGLR and TNP correlations and RMSE values.

- `final_summaries/`: exact finalized V6 outputs, including seed-level results,
  pooled-Pearson sensitivity, and TNP checkpoint sensitivity.
- `tnp/tidy_all_metrics.csv`: the September 6 matched-fold TNP release. The
  primary checkpoint is `best_frozen_pearson_r`; training ran all 250 epochs.
- `workbooks/`: unchanged Tiezzi calculation workbooks and CSV mirrors. The
  LOEO workbook supplies primary LOEO correlations. The July CV workbook is
  retained for provenance only; finalized CV results are calculated directly
  from corrected task-level outputs instead of this historical workbook.

Reproduce these summaries with
[`summarize_all_prediction_results.R`](../../scripts/07_results_figures/prediction/summarize_all_prediction_results.R).
Raw BGLR task-level results are external (planned for Dryad); exact release
paths are in [`config/analysis.yml`](../../config/analysis.yml).

CV1, CV0, and CV00 score held-out records. **CV2 is in-sample fit**, not a test
of generalization. LOEO RMSE uses only rows where `Env == Untested.Env`.
