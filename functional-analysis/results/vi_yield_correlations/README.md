# Vegetation-index FPC correlations with grain yield

These tables document the descriptive VI FPC–yield correlations reported in
the manuscript, calculated from existing full-data FPCA scores and grain-yield
BLUEs. They are distinct from the **weather** FPC correlations in Figure 3
and from held-out prediction performance.

| File | Contents |
| --- | --- |
| [VI_FPC_Yield_Correlations_Pooled.csv](VI_FPC_Yield_Correlations_Pooled.csv) | 444 Pearson correlations across the 10,109 matched genotype–environment records |
| [VI_FPC_Yield_Correlations_Within_Environment.csv](VI_FPC_Yield_Correlations_Within_Environment.csv) | 8,436 Pearson correlations calculated separately within each of 19 environments |
| [VI_FPC_Yield_Correlation_Inputs.csv](VI_FPC_Yield_Correlation_Inputs.csv) | Exact input filenames, byte sizes, and MD5 checksums |
| [sessionInfo.txt](sessionInfo.txt) | R and package versions used for these exports |

Both tables cover all 37 VIs, retaining FPC1–5 for DAP and FPC1–7 for AGDD.
`Domain`, `Vegetation.Index`, and `FPC` identify each comparison; `FPC_ID`
retains the historical combined label. The within-environment table adds `Env`.
`Cor` is Pearson's r, `N` is the number of cohort records, `N_complete` counts
finite score–yield pairs, and `Status` describes whether r is defined.
No pairs are silently discarded: missing/nonfinite values or zero variance
produce `NA` and an explanatory status. All deposited correlations have
`Status = ok` and `N_complete = N`.

## NGRDI values reported in the manuscript

| Quantity | Pearson r |
| --- | ---: |
| DAP FPC1, pooled | 0.550262 |
| AGDD FPC3, pooled | 0.596590 |
| DAP within-environment range, FPC1–5 | −0.290217 to 0.719954 |
| AGDD within-environment range, FPC1–7 | −0.497419 to 0.635254 |

The ranges span **all retained NGRDI components across the 19 environments**,
not FPC1 alone. Pooled r uses individual genotype–environment records without
environment weighting or centering. It is not an average of within-environment
correlations and can reflect differences among environments. These are
descriptive associations, not causal effects or cross-validated prediction
accuracies. FPC signs refer to the saved score bases.

## Recalculate

Run the portable adaptation of `Unified_FPCA_Yield_Cor_V6.R`:

```sh
cd functional-analysis
Rscript --vanilla scripts/02_fpca_weather/vi_yield_correlations.R
```

The [script](../../scripts/02_fpca_weather/vi_yield_correlations.R) uses the
configured data and working-results roots. Place these inputs in `data_dir`:

- `G2F.2020.2021.Pedigrees.csv` — the canonical 10,109-record cohort.
- `Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv` — yield BLUEs.
- `FPC_Scores_BLUEs_Unified_FPCA_Max_FPCs_2020_2021_G2F_VIs.csv` — DAP scores.
- `FPC_Scores_BLUEs_AGDD_Unified_FPCA_Max_FPCs_2020_2021_G2F_VIs.csv` — AGDD scores.

The yield and score inputs can also be resolved from the corresponding
`01_blue_variance`, `02_fpca_weather/vi_dap`, and `02_fpca_weather/vi_agdd`
working-results folders. These inputs remain external; see
[data availability](../../docs/DATA_AVAILABILITY.md). The deposited result
tables can be downloaded without those inputs.

Recalculation writes to
`<results_dir>/02_fpca_weather/vi_yield_correlations/`, leaving these reference
tables unchanged. It does not refit FPCA or prediction models. The adaptation
runs both time domains explicitly, selects score columns by name, aligns
records by identifier, and exports the pooled and within-environment results
directly instead of relying on interactive object selection or older plot CSVs.
