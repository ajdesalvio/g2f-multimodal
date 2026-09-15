# Functional analysis and manuscript results

Code and results for **“An open field phenomics resource for multimodal maize yield prediction across divergent environments.”**

This workflow prepares the study data, fits DAP/AGDD functional principal components, maps QTL, runs kernel prediction, and assembles the manuscript results. TNP training and preprocessing are documented separately in [neural-process](../neural-process/). Return to the [repository overview](../README.md) for both workflows and geospatial data links.

## Start here

- **Read the results:** [prediction summaries](results/prediction/), [manuscript figures](results/figures/), and [QTL tables](results/qtl/).
- **Find a supplementary file:** [resource inventory](docs/RESOURCE_INVENTORY.md).
- **Reproduce an analysis:** [execution order](docs/WORKFLOW.md) and [software requirements](environment/README.md).

## Repository layout

| Directory | Purpose |
|---|---|
| [scripts/00_data_prep/](scripts/00_data_prep/) | Assemble plot, image date, and pedigree data |
| [scripts/01_blue_variance/](scripts/01_blue_variance/) | Estimate BLUEs and variance components |
| [scripts/02_fpca_weather/](scripts/02_fpca_weather/) | Clean weather and fit descriptive DAP/AGDD FPCA |
| [scripts/03_genomics/](scripts/03_genomics/) | Impute markers and build genomic relationships |
| [scripts/04_qtl/](scripts/04_qtl/) | Map FPC QTL; retain flowering/yield QTL as an optional supplement |
| [scripts/05_prediction_cv/](scripts/05_prediction_cv/) | Prepare and run fold-based kernel prediction |
| [scripts/06_prediction_loeo/](scripts/06_prediction_loeo/) | Leave-one-environment-out prediction |
| [scripts/07_results_figures/](scripts/07_results_figures/) | Compile final results and reproduce figures |
| [results/](results/) | Publication tables, source data, and PDFs |
| [data/](data/) | Small reference inputs and external-data inventory |
| [config/](config/) · [R/utils/](R/utils/) | Scientific settings, local paths, and shared functions |

## Configure local paths

Copy `config/paths.example.yml` to the ignored `config/paths.yml`. Set two directories:

```yaml
project:
  data_dir: /path/to/downloaded-data
  results_dir: /path/to/working-results
```

Run commands below from **`functional-analysis/`**, not the outer Git repository directory. `G2F_DATA_DIR` and `G2F_RESULTS_DIR` can override these settings; `G2F_PROJECT_DIR` must point to `functional-analysis/` when running elsewhere. Generated files go to the working results directory. Keep it separate from the checked-in publication results.

To check the repository without downloading the study data:

```sh
Rscript --vanilla scripts/validate_repository.R
python scripts/validate_manifests.py
python scripts/validate_release.py
```

The [workflow](docs/WORKFLOW.md) distinguishes lightweight result compilation from full-scale model fitting. You do not need to rerun every stage to inspect the paper's results.

## Analysis conventions

Only final **no-Ze** predictions are reported: the separate categorical environment main effect is omitted. CV2 is an **in-sample reference**. NGRDI and PTR were selected a priori; five DAP and seven AGDD NGRDI components were selected in exploratory prediction comparisons and then fixed. FPCA bases are fitted on training curves. Phenomic score standardization uses all predictor records within each environment. See [methods and limitations](docs/REPRODUCIBILITY.md) for these distinctions and metric definitions.

## Data, citation, and reuse

Geospatial data are deposited in Data 2 Science with DOIs. The [linked collection table](../README.md#geospatial-data) covers all 19 study environments. Larger analysis inputs and task-level outputs are planned for release via Dryad. Its DOI and download links will be added at a later date. Final TNP metrics are included here and its complete workflow is in the sibling `neural-process/` directory. See [data availability](docs/DATA_AVAILABILITY.md).

[CITATION.cff](CITATION.cff) follows the current manuscript title and author order. The manuscript DOI will be added when available. Original software uses [MIT](LICENSE), and original documentation and figure artwork use [CC BY 4.0](LICENSE-DOCUMENTATION.md). Third party software and data retain their own terms; see the [licensing overview](../LICENSING.md).
