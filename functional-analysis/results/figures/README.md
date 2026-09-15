# Manuscript figures

Start with the PDFs below. These are exact copies of the final
figure exports, renamed by paper order; their source versions are recorded in
[RESULT_MANIFEST.csv](../RESULT_MANIFEST.csv). Historical alternatives and
unused plots are not included.

| Paper item | PDF | Generating script |
| --- | --- | --- |
| Figure 1: variance components | [Figure 1](Figure_01_Variance_Components.pdf) | [Variance components](../../scripts/07_results_figures/figures/plot_variance_components.R) |
| Figure 2: DAP/AGDD alignment and FPCA | [Figure 2](Figure_02_FPCA_Trajectories_Biplots.pdf) | [Trajectories and biplots](../../scripts/07_results_figures/figures/plot_fpca_trajectories_biplots.R) |
| Figure 3: weather FPCs and yield | [Figure 3](Figure_03_Weather_Yield_Correlations.pdf) | [Weather correlations](../../scripts/07_results_figures/figures/plot_weather_yield_correlations.R) |
| Figure 4: prediction | [Figure 4](Figure_04_Prediction.pdf) | [Prediction plot](../../scripts/07_results_figures/figures/plot_prediction_results.R) |
| Figure 5: QTL presence and LD | [Figure 5](Figure_05_QTL_and_LD.pdf) | [QTL panels](../../scripts/07_results_figures/figures/plot_qtl_presence.R), [LD workflow](../../scripts/04_qtl/ld/README.md), and [manual assembly](../qtl/ld/README.md) |
| Figure 6: chromosome 7 trajectories | [Figure 6](Figure_06_QTL_Trajectories.pdf) | [QTL trajectories](../../scripts/07_results_figures/figures/plot_qtl_trajectories.R) |
| Supplementary Figure 1: AGDD biplots | [Supplementary 1](Supplementary_01_AGDD_Biplots.pdf) | [AGDD biplots](../../scripts/07_results_figures/figures/plot_agdd_biplots.R) |
| Supplementary Figure 2: environment differences | [Supplementary 2](Supplementary_02_Environment_Differences.pdf) | [Paired comparisons](../../scripts/07_results_figures/figures/plot_paired_prediction_comparisons.R) |
| Supplementary Figure 3: fold distributions | [Supplementary 3](Supplementary_03_Fold_Distributions.pdf) | Same paired-comparison script |
| Supplementary Figure 4: RMSE weighting | [Supplementary 4](Supplementary_04_RMSE_Weighting.pdf) | Same paired-comparison script |

Figure 1 uses the V6 export with "Genotype" labels. Figure 5 includes the LD
heatmaps and preserves its original manual PowerPoint assembly.

## Source data

- Figure 3: [all 151 weather correlations](source_data/Yield_FPC_Correlations_All_DAP.csv), using all-DAP FPCs and the 10,109 matched yield/phenomic records in 19 environments. `T2M_MIN` FPC2 is -0.637662 (displayed as -0.64).
- Figure 4: [final prediction summary](../prediction/final_summaries/G2F_Final_Prediction_Summary.csv) and [25 LOEO bars](source_data/Figure_04_LOEO_Bars.csv). Genetic-only runs are averaged over DAP/AGDD for display; the complete summary retains both runs.
- Supplementary Figures 2-4: `source_data/CV*.csv` contains environment-, fold-, and seed-level metrics and paired differences. These use shared seeds 1-5 and five folds; the main kernel summary uses ten seeds.

Run scripts from `functional-analysis/` after configuring the two local roots.
Rebuilt files go to `<results_dir>/07_results_figures/figures/`, leaving these
manuscript copies unchanged. Small typography/layout differences can occur
across R/graphics versions. Compare the plotted values as well as the appearance.
Prediction figures use the retained final summaries by default. To inspect a
new compilation, set `G2F_PREDICTION_SUMMARY` and
`G2F_PREDICTION_SEED_SUMMARY` to its files explicitly; old local reruns will not
silently override the release summaries.

To replot Supplementary Figures 2-4 using only the small source tables already
in this repository (no analysis archives or external data needed):

```sh
Rscript scripts/07_results_figures/figures/plot_paired_prediction_comparisons.R --from-source-data
```

Omit `--from-source-data` to recompute the source tables from raw predictions and
verify their seed-level metrics against the final summary.
