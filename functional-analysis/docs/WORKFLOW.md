# Reproduction workflow

Run from `functional-analysis/` after configuring [two local roots](../config/README.md). In these instructions, the analysis root is this directory (not the outer Git repository). Generated files go to the external results root, and checked-in publication outputs remain unchanged. Input resolvers look for earlier stage outputs first, then downloaded data and bundled reference files.

## Choose a starting point

| Goal | Start here | Required material |
|---|---|---|
| Inspect the manuscript | [results](../results/README.md) and [resource inventory](RESOURCE_INVENTORY.md) | Repository only |
| Validate repository files | [validation commands](../README.md#configure-local-paths) | Repository; base R and Python |
| Recalculate reported metrics | Stage 8 below | Final CV/LOEO task outputs, LOEO workbook, TNP metrics |
| Replot the paper | Stage 9 below | Retained source tables; FPCA/QTL objects for some figures |
| Refit models from processed data | Stages 5–7 | Downloaded processed inputs and a suitable cluster |
| Rebuild from source data | Stages 1–9 | Raw sources, software environment, and cluster |

## 1. Assemble plot and phenotype data

In `scripts/00_data_prep/`, run:

1. `combine_vi_pht.R`
2. `fix_jpg_dates.R`
3. `merge_vi_pht_jpg.R`
4. `build_genomic_phenomic_overlap.R`
5. `build_pedigree_environment_overlap.R`
6. `build_prediction_phenotypes.R`

VI/PHT tables for each location belong in `data_dir/vi_pht_location_files/`, or use `G2F_VI_PHT_INPUT_DIR`. The prediction phenotype order contains 10,109 matched records.

## 2. Estimate BLUEs and variance components

In `scripts/01_blue_variance/`, run `yield_blues.R`, `dta_dts_blues.R`, and `vi_blues.R`; then `yield_variance_components_source.R` and `vi_variance_components_source.R`.

The shared VI helper applies the filters in [config/analysis.yml](../config/analysis.yml). Unused historical BLUP branches are not part of these scripts.

## 3. Process weather and full data FPCA

In `scripts/02_fpca_weather/`:

- Station weather: `clean_mnh1_2020_weather.R` → `station_air_weather_fpca.R` and `station_soil_weather_fpca.R`.
- EnvRtype weather: `acquire_clean_dap_weather_fpca_source.R` → `agdd_weather_fpca_source.R`.
- VI trajectories: `vi_fpca_dap_full.R` and `vi_fpca_agdd_full.R`.
- Weather/yield source table: `weather_yield_correlations_all_dap.R`.

The last step uses all DAP scores and yield medians from the matched 10,109-record cohort. Set `G2F_REFRESH_WEATHER=TRUE` only when a new NASA POWER download is intended. Normally use archived observations.

## 4. Prepare genomic relationships

Run `scripts/03_genomics/genomic_imputation.R`, then `genomic_relationship_matrices.R`. This stage requires Java/rTASSEL and externally stored genotype data. Preserve the pedigree ordering used downstream.

## 5. Run QTL mapping

Follow [scripts/04_qtl/README.md](../scripts/04_qtl/README.md) for the DAP and AGDD manuscript branches. The flowering/yield branch is optional supplementary material. Each branch prepares inputs, launches scans, and compiles results. The 29 environment/tester combinations are in `scripts/04_qtl/hprc/env_testers.txt`.

After scans finish, run the [refinement workflow](../scripts/04_qtl/refinement/README.md). Interval recalculation requires per-cross `QTL.Outputs.rds` files. Compiled CSVs alone are not sufficient to replace them. All 58 DAP/AGDD objects are planned to be released via Dryad. Flowering/yield objects are needed only to regenerate the additional 39 supplemental peaks. Without that branch, recalculation writes only available FPC results. Check the 71 DAP and 96 AGDD peaks before using a rerun.

For a **results-only** rebuild, the offline publication table exporter uses retained CSVs to produce the 167-peak FPC subset and exploratory named gene tables. The complete 206-row interval table remains available, including flowering/yield results.

Figure 5's [regional LD workflow](../scripts/04_qtl/ld/README.md) is independent of the QTL scan objects. Replot its retained matrices with `Rscript scripts/04_qtl/ld/calculate_regional_ld.R --from-source-data`, or recompute from the external genotype CSV. The [original composite and editable assembly](../results/qtl/ld/README.md) preserve the manual layout step.

## 6. Run fold-based kernel prediction

Follow [scripts/05_prediction_cv/README.md](../scripts/05_prediction_cv/README.md), separately for DAP and AGDD:

1. Metadata and preprocessing.
2. Constant kernels and weather inputs.
3. Fold-specific FPCA and kernels.
4. Prediction and verification.

The common maternal line CV scheme uses 223 maternal lines. Launchers enforce `G2F_INCLUDE_ZE=FALSE`. DAP uses NGRDI FPC1–5; AGDD uses FPC1–7; both use PTR FPC1. Read [methods and limitations](REPRODUCIBILITY.md) for further details.

## 7. Run leave-one-environment-out prediction

In `scripts/06_prediction_loeo/`, run the domain's `project_*_fpca.R`, `build_*_kernels.R`, and `run_*_prediction.R`, in that order. See the [stage README](../scripts/06_prediction_loeo/README.md) for launch commands.

The AGDD projection writes `Weather_FPC_Scores_AGDD_LOEO_Projected.csv` directly. Each domain produces 266 model/environment prediction files. The final compiler scores only each run's held-out environment.

## 8. Compile finalized prediction metrics

Run:

```sh
Rscript --vanilla scripts/07_results_figures/prediction/summarize_all_prediction_results.R
```

This is the portable version of **Summarize_All_Prediction_Results_V6.R**. The primary output is `G2F_Final_Prediction_Summary.csv`; accompanying seed-level, pooled Pearson sensitivity, and checkpoint sensitivity tables preserve the aggregation trail. Fold-level source tables for the paired comparisons are retained with the figures. See [prediction results](../results/prediction/README.md) for exact input batches and output locations.

The compiler resolves canonical new-run folders first and final historical batches from `prediction_results` in [config/analysis.yml](../config/analysis.yml) as fallbacks. Keep the final DAP/AGDD batches intact. The LOEO workbook supplies retained correlation inputs; current CV correlation/RMSE are calculated from final task outputs. TNP metrics come from the finalized collaborator export.

The older `compile_cv_results.R` and `compile_loeo_results.R` are supporting diagnostic compilers.

## 9. Reproduce figures and flight metadata

Use the [figure index](../results/figures/README.md): it maps each manuscript figure to its generating script and source files. Paired prediction plots use the matched final five-seed comparisons.

`scripts/07_results_figures/drone/build_drone_data_table_source.R` generates the flight/date source table. `extract_orthomosaic_gsd.R` reads an explicit local raster manifest and reports exported pixel dimensions. The supplied final flight/GSD workbook is retained unchanged.

## 10. Validate regenerated outputs

Compare regenerated tables with the retained publication results, using the aggregation definitions in [methods and reproducibility](REPRODUCIBILITY.md). Publication figures may differ byte-for-byte after rendering because of fonts/software; compare both numerical values and visual appearance. See [data availability](DATA_AVAILABILITY.md) for external inputs and their checksums.
