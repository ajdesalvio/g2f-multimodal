# G2F multimodal maize yield prediction

Code, data access, and results for **“An open field phenomics resource for multimodal maize yield prediction across divergent environments.”** The study combines genomic markers, temporal drone phenotypes, and weather data for 1,180 maize hybrids across 19 Genomes to Fields environments.

## Choose a workflow

| Folder | Contents | Start here |
| --- | --- | --- |
| [functional-analysis](functional-analysis/) | Data preparation, DAP/AGDD FPCA, QTL and LD, kernel prediction, and manuscript figures/results | [Workflow](functional-analysis/docs/WORKFLOW.md) · [Results](functional-analysis/results/) |
| [neural-process](neural-process/) | Transformer neural process preprocessing, training, evaluation, and associated baselines | [TNP documentation](neural-process/README.md) |

Each workflow has its own environment and setup instructions. For the paper's final combined prediction summaries, go to [prediction results](functional-analysis/results/prediction/). For figures and supplementary resources, use the [figure index](functional-analysis/results/figures/) and [resource inventory](functional-analysis/docs/RESOURCE_INVENTORY.md).

Direct supplementary downloads: Figure 1 [VI variance components](functional-analysis/data/supplementary/VarComp_G2F_2020_2021.csv) and [yield variance components](functional-analysis/data/supplementary/VarComp_Yield_G2F_2020_2021.csv); [VI FPC–yield correlation tables and calculation code](functional-analysis/results/vi_yield_correlations/).

## QTL and LD reference data

Download the **[Michel et al. (2022) supplementary dataset](https://doi.org/10.25386/genetics.19439684)** for the QTL/LD genotype and R/qtl2 reference files. The required archive is [File S5 control.tar.gz (12.7 MB)](https://ndownloader.figshare.com/files/34591628). Follow the [download and folder setup instructions](functional-analysis/data/MICHEL_2022_DATA.md) before running QTL or LD analyses.

## Geospatial data

All study geospatial data are deposited in **Data 2 Science** with DOIs. Browse the STAC collections below or follow a DOI to its dataset record. Links are transcribed from the [D2S_STAC_Links.xlsx](functional-analysis/results/drone/D2S_STAC_Links.xlsx) workbook.

| Collection | Data to Science | DOI |
| --- | --- | --- |
| DEH1.2020 | [Open collection](https://stac.d2s.org/collections/224e1ad0-d192-4861-95e8-7e774d2a4c5e?.language=en) | [10.6084/m9.figshare.33300057](https://doi.org/10.6084/m9.figshare.33300057) |
| IAH4.2021 | [Open collection](https://stac.d2s.org/collections/819b21bd-e84c-4dca-baec-ac9ecab69286?.language=en) | [10.6084/m9.figshare.33301599](https://doi.org/10.6084/m9.figshare.33301599) |
| MIH1.2020 | [Open collection](https://stac.d2s.org/collections/294b403c-d4dd-4d84-a4df-d5b56f23b4a6?.language=en) | [10.6084/m9.figshare.33301605](https://doi.org/10.6084/m9.figshare.33301605) |
| MNH1.2020 | [Open collection](https://stac.d2s.org/collections/91a9bdc8-295a-40d4-a99a-9c5888f4551a?.language=en) | [10.6084/m9.figshare.33301611](https://doi.org/10.6084/m9.figshare.33301611) |
| MNH1.2021 | [Open collection](https://stac.d2s.org/collections/6212cf56-3772-4356-9e4a-916d001eaedc?.language=en) | [10.6084/m9.figshare.33301701](https://doi.org/10.6084/m9.figshare.33301701) |
| MOH1.2020.C5a | [Open collection](https://stac.d2s.org/collections/ca7ef0c3-152a-4e9f-9a51-ab66cb439656?.language=en) | [10.6084/m9.figshare.33301704](https://doi.org/10.6084/m9.figshare.33301704) |
| MOH1.2020.C5b | [Open collection](https://stac.d2s.org/collections/abf3b468-2eb9-40d2-a6c9-c333bb0124bc?.language=en) | [10.6084/m9.figshare.33301707](https://doi.org/10.6084/m9.figshare.33301707) |
| NEH1.2021 | [Open collection](https://stac.d2s.org/collections/4cf79550-0412-452d-a3ea-f4ccf929cee9?.language=en) | [10.6084/m9.figshare.33301713](https://doi.org/10.6084/m9.figshare.33301713) |
| TXH123.2020 | [Open collection](https://stac.d2s.org/collections/a79191ad-2f59-4f83-a8c1-a3b046d20f18?.language=en) | [10.6084/m9.figshare.33301716](https://doi.org/10.6084/m9.figshare.33301716) |
| TXH123.2021 | [Open collection](https://stac.d2s.org/collections/a2261fac-7a57-43ea-9325-636d24a03b59?.language=en) | [10.6084/m9.figshare.33301719](https://doi.org/10.6084/m9.figshare.33301719) |
| WIH1.2020 | [Open collection](https://stac.d2s.org/collections/c454cc53-b8e5-4f09-add3-788217ac3813?.language=en) | [10.6084/m9.figshare.33301749](https://doi.org/10.6084/m9.figshare.33301749) |
| WIH1.2021 | [Open collection](https://stac.d2s.org/collections/863345f6-ec59-4513-b514-3a5dcb791c85?.language=en) | [10.6084/m9.figshare.33301755](https://doi.org/10.6084/m9.figshare.33301755) |
| WIH2.2020 | [Open collection](https://stac.d2s.org/collections/6958c1f9-a2e6-4d7d-aa70-f26ec816f4bd?.language=en) | [10.6084/m9.figshare.33438316](https://doi.org/10.6084/m9.figshare.33438316) |
| WIH2.2021 | [Open collection](https://stac.d2s.org/collections/aef44205-f1d1-4c69-878b-5f1b6cf60ea7?.language=en) | [10.6084/m9.figshare.33301758](https://doi.org/10.6084/m9.figshare.33301758) |
| WIH3.2020 | [Open collection](https://stac.d2s.org/collections/49d5771b-c179-4fa4-ac87-140d38a3b15d?.language=en) | [10.6084/m9.figshare.33301764](https://doi.org/10.6084/m9.figshare.33301764) |
| WIH3.2021 | [Open collection](https://stac.d2s.org/collections/a3794e12-0c8f-4889-b971-72e40a834d13?.language=en) | [10.6084/m9.figshare.33301776](https://doi.org/10.6084/m9.figshare.33301776) |

These 16 collections cover 19 analysis environments: `TXH123` combines Texas trials 1–3 within each year, and `MOH1.2020` is split into fields C5a and C5b. The [flight metadata](functional-analysis/results/drone/) describe cameras, dates, and exported raster resolution. Orthomosaics are hosted externally on Purdue's D2S servers.

## Other data and reproducibility

Compact results and source tables are included in `functional-analysis/`. Large non-geospatial inputs and task-level outputs are planned to be hosted on Dryad; its DOI and download mapping remain pending. See [data availability](functional-analysis/docs/DATA_AVAILABILITY.md) for the current inventory and [reproducibility notes](functional-analysis/docs/REPRODUCIBILITY.md) for analysis assumptions and remaining provenance gaps.

To check the functional-analysis files without downloading the study data:

```sh
cd functional-analysis
Rscript --vanilla scripts/validate_repository.R
python scripts/validate_manifests.py
python scripts/validate_release.py
```

## Citation and licenses

[CITATION.cff](CITATION.cff) records the manuscript title and author list. The manuscript DOI will be added when available. Please also cite the relevant dataset DOI when reusing a geospatial collection.

Original software is subject to an MIT license in both [functional-analysis](functional-analysis/LICENSE) and [neural-process](neural-process/LICENSE). Our documentation and figure artwork use [CC BY 4.0](functional-analysis/LICENSE-DOCUMENTATION.md). Planned Dryad data use CC0. Third party software and datasets retain their own terms. See [license scope and exceptions](LICENSING.md).
