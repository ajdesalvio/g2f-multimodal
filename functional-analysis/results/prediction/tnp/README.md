# TNP evaluation metrics

`tidy_all_metrics.csv` is the unchanged September 6, 2026 collaborator export:
30,000 rows for 3 modality configurations, 5 seeds, 100 training splits per
seed, 10 checkpoint rules, and 2 evaluation scenarios per training split.

The maternal-line folds match the BGLR R-generated fold table. All models ran
250 epochs; `best_frozen_pearson_r` freezes checkpoint selection after a
validation plateau, not training. It is the prespecified primary checkpoint.
CV2 evaluates training records and is an in-sample reference.

`pearson_r` is the correlation across scored rows; `r_w` pools correlations
within environments. Use `r_w` for CV1/CV2 and pool environment-specific
`pearson_r` values for CV0/CV00, then average folds within seeds and seeds.
The primary summary script performs these steps. Training, preprocessing, and
evaluation code are in the sibling [neural-process workflow](../../../../neural-process/).
The retained metric export has not been independently regenerated from that
workflow.
