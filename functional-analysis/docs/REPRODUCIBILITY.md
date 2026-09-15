# Methods and reproducibility

These scripts preserve the final analyses, including their limitations. They use explicit files and configuration rather than saved R sessions or selectively executed console blocks. [Analysis configuration](ANALYSIS_CONFIGURATION.md) summarizes the encoded filtering and execution settings.

## Prediction design

- The matched dataset contains 10,109 hybrid–environment records and 1,180 hybrids across 19 environments. The common maternal line fold scheme uses 223 maternal lines.
- NGRDI and PTR were chosen a priori. Five DAP and seven AGDD NGRDI components were chosen in exploratory prediction comparisons, then held fixed across splits. Those trials are not archived. Component count selection was not nested within the reported validation.
- Full data FPCA is used for descriptive figures and QTL. For prediction, FPCA bases are fitted to training curves and held-out curves are projected onto the fitted basis.
- **Phenomic standardization uses all predictor scores within each environment**, including held-out scores. Held-out yields are excluded. Weather scores use training environment centering and scaling when environments are withheld. These are different procedures.
- Genomic kernels receive no additional centering or normalization after GRM construction.
- Final models omit the separate categorical environment main effect (no-Ze), retain the intercept, and include interaction/weather kernels as specified. Predictions are exported directly.
- **CV2 is an in-sample reference.**

## Metrics

The primary [compiler](../scripts/07_results_figures/prediction/summarize_all_prediction_results.R) is adapted from the final analysis that produced the manuscript figures.

| Output | Calculation |
|---|---|
| Fold-based kernel correlation | Within-environment correlations pooled by the Tiezzi procedure within each seed/fold; folds averaged within seed, then seeds averaged |
| Fold-based RMSE | Across scored predictions within each seed/fold; folds averaged within seed, then seeds averaged |
| LOEO metrics | Held-out environment rows only (`Env == Untested.Env`); within-environment correlations pooled and squared errors pooled across held-out predictions |
| Reported SD | Between-seed SD of seed-level summaries; absent when the analysis has no seed replication |
| Paired model comparisons | Matched seeds 1–5 and five folds per seed |
| RMSE sensitivity | Pooled RMSE versus the arithmetic mean of the 19 separate environment RMSEs |

RMSE is in t/ha. Pooled Pearson correlation and Tiezzi-pooled within-environment correlation are different estimands. Missing values must be treated as such (NAs).

The archived outputs saved in this repository supersede earlier summary versions. Historical Excel calculations are retained as records, but do not supersede the current CV calculations.

TNP runs used all 250 training epochs. Plateau-based freezing controls checkpoint selection and does not dictate training termination. The primary checkpoint is `best_frozen_pearson_r`. The TNP final metrics are included here; training, preprocessing, and evaluation are documented in the sibling [neural-process workflow](../../neural-process/).

## Other analysis decisions

- Weather/yield correlations use all-DAP FPCs and median yield BLUEs from the **10,109 matched records**, across 19 environments.
- NGRDI QTL results contain **167 peak detections** (71 DAP, 96 AGDD), counted separately by environment, tester, component, and time domain.
- The exploratory chromosome 7 annotation window is 126,763,800–138,089,853 bp, excluding the broad WIH3.2020.PHP02 interval when choosing its bounds. MaizeMine results include multiple reference assemblies, are filtered/grouped by symbol, but are not evidence of causal genes.
- Raster pixel size is the exported orthomosaic resolution. Photogrammetry processing reports are unavailable for the orthomosaics saved in Data 2 Science as they were not exported at the time of processing.
- Figure 5 includes squared genotype dosage correlations (LD r-squared) for chromosome 3 (139–186 Mb) and chromosome 7 (125–140 Mb), using 345 inbred lines. Retained matrices were regenerated from the available genotype data. Both LD panels reproduced the original rendered pixels. The composite was assembled manually using PowerPoint. Its editable layout is retained with the [LD results](../results/qtl/ld/).
- Figure 1 displays "Genotype". Historical input columns such as `Pedigree` and `Pedigree.Env` remain unchanged to preserve file compatibility.

## What validation establishes

Static checks cover syntax, paths, inventories, and hashes. Release checks cover retained summary dimensions and key numerical invariants. Recompiling summaries from task outputs checks aggregation without refitting models.

Targeted aggregation checks for `M10.G.P.W-Int` reproduced the retained DAP CV1, AGDD CV00, and DAP LOEO correlation and RMSE summaries from their task-level inputs, within floating-point precision. These checks do not establish agreement for every model. Full end-to-end model refitting from a fresh repository download has not been tested.

The historical prediction environment used R 4.4.2. Numerical and PDF identity can depend on package versions, BLAS, Java, random-number generation, and fonts.
