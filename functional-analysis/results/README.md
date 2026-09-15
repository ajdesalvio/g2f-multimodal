# Publication results

Start with the output you want to check:

| Output | Location |
|---|---|
| Final correlation and RMSE summaries | [prediction/](prediction/) |
| Main and supplementary figures, with source tables | [figures/](figures/) |
| Full QTL results, 167 FPC-only peaks, and exploratory annotations | [qtl/](qtl/) |
| Flight dates, cameras, GSD, and image access | [drone/](drone/) |

These are retained publication artifacts. Regeneration writes to the configured external results root, so it does not overwrite this reference set.

[RESULT_MANIFEST.csv](RESULT_MANIFEST.csv) records exact-copy sources or generators and checksums. [EXTERNAL_DEPENDENCIES.csv](EXTERNAL_DEPENDENCIES.csv) lists material that is not bundled. Consult [data availability](../docs/DATA_AVAILABILITY.md) before attempting a full rerun.

In the dependency catalog, `available_external` means archived outside this repository; it does not mean a public download is available. Public access remains pending for the non-geospatial archives.
