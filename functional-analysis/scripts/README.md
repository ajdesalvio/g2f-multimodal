# Scripts

Scripts are organized by dependency order rather than by their historical sandbox numbering.

```text
00_data_prep/
01_blue_variance/
02_fpca_weather/
03_genomics/
04_qtl/
05_prediction_cv/
06_prediction_loeo/
07_results_figures/
```

Cleaned entry points use configured inputs and explicit output directories.
Stage READMEs document execution context and ordering; original
filename/version provenance is centralized in `SOURCE_MAP.csv`, and scientific
choices are centralized in `config/analysis.yml`.

Historical alternatives and exploratory analyses are not copied into this
clean repository.
