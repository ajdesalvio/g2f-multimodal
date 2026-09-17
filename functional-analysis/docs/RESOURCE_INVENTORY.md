# Supplementary resource inventory

| Resource | File or location |
|---|---|
| Environment metadata | [Envs_Coordinates.csv](../data/supplementary/Envs_Coordinates.csv) |
| Flight metadata and exported raster resolution | [Drone_Data_Description_V6.xlsx](../results/drone/Drone_Data_Description_V6.xlsx) |
| Imagery collections and DOIs | [D2S_STAC_Links.xlsx](../results/drone/D2S_STAC_Links.xlsx) |
| Cleaned plot observations | External: `G2F_2020_2021_Cleaned_Data.csv` (~515 MB); final download URL pending |
| 10,109 phenomic–phenotypic records | [Pedigree_Overlap_Phenomic_Phenotypic_BLUEs.csv](../data/supplementary/Pedigree_Overlap_Phenomic_Phenotypic_BLUEs.csv) |
| Weather variable definitions and units | [EnvRType_Weather_Variable_Descriptions.xlsx](../data/supplementary/EnvRType_Weather_Variable_Descriptions.xlsx) |
| Air temperature | External: `Weather_AirTemp_Clean_Tall.csv` (~18.5 MB) |
| Soil temperature | External: `Weather_SoilTemp_Clean_Tall.csv` (~18.5 MB) |
| EnvRtype weather data | [EnvRtype_Weather_Data_Cleaned_V2.csv](../data/supplementary/EnvRtype_Weather_Data_Cleaned_V2.csv) |
| 1,180 genomic–phenomic hybrids | [Pedigree_Overlap_Genomic_Phenomic.csv](../data/supplementary/Pedigree_Overlap_Genomic_Phenomic.csv) |
| Figure 1 VI variance components and model-fit statistics | [VarComp_G2F_2020_2021.csv](../data/supplementary/VarComp_G2F_2020_2021.csv) |
| Figure 1 yield variance components and model-fit statistics | [VarComp_Yield_G2F_2020_2021.csv](../data/supplementary/VarComp_Yield_G2F_2020_2021.csv) |
| VI FPC–yield correlation calculations | [vi_yield_correlations.R](../scripts/02_fpca_weather/vi_yield_correlations.R), adapted from `Unified_FPCA_Yield_Cor_V6.R` |
| VI FPC–yield correlations, DAP and AGDD | [Pooled](../results/vi_yield_correlations/VI_FPC_Yield_Correlations_Pooled.csv) and [within-environment](../results/vi_yield_correlations/VI_FPC_Yield_Correlations_Within_Environment.csv) tables; [definitions and inputs](../results/vi_yield_correlations/) |
| QTL/LD genotypes and R/qtl2 reference files | [Michel et al. (2022) dataset](https://doi.org/10.25386/genetics.19439684), File S5; [download and setup](../data/MICHEL_2022_DATA.md) |
| FPC-only QTL peaks and full QTL table | [QTL results](../results/qtl/) — the separate FPC-only file has 167 peaks; the full source is preserved |
| DAP/AGDD QTL scan objects | External: 58 RDS files, 32.80 GB; [checksums and folder mapping](../data/QTL_SCAN_ARCHIVE.md); deposit links pending |
| Prediction calculations | [summarize_all_prediction_results.R](../scripts/07_results_figures/prediction/summarize_all_prediction_results.R), adapted from `Summarize_All_Prediction_Results_V6.R` |
| Prediction outputs | [final summaries](../results/prediction/final_summaries/), including model-level and seed-level CSVs |
| Main and supplementary figures | [figure index and source data](../results/figures/) |

The flight workbook has 400 metadata rows representing **356 unique environment/DAP combinations**. All 356 match the retained flight-date conversion table. The 44 repeated keys arise from MOH1 C5a/C5b field records. The image access workbook lists 16 collections covering the 19 analysis environments through combined Texas collections and split MOH1 fields.

The prediction compiler requires the final task-level outputs and retained inputs described in its [README](../results/prediction/README.md).

Exact copy provenance is in [results/RESULT_MANIFEST.csv](../results/RESULT_MANIFEST.csv) and [data/manifest.csv](../data/manifest.csv). Historical-to-clean script names are in [scripts/SOURCE_MAP.csv](../scripts/SOURCE_MAP.csv).
