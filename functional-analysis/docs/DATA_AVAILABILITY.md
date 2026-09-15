# Data availability

## Included in GitHub

The repository contains portable scripts, final no-Ze prediction summaries, source tables for the reported figures, the final TNP metric export, preserved calculation workbooks, full and FPC-only QTL tables, exploratory annotation snapshots, manuscript PDFs, flight metadata, and small supplementary reference inputs.

The [resource inventory](RESOURCE_INVENTORY.md) links each manuscript-facing resource. [FILE_MANIFEST.csv](../FILE_MANIFEST.csv) and the specialized data/result manifests provide SHA-256 checksums.

## Imagery: Data to Science

All study geospatial data are deposited in Data to Science with DOIs. Use the [linked collection table](../../README.md#geospatial-data) to browse or cite each collection. The image-access workbook is retained with the flight metadata. The repository does not download or mirror orthomosaics. The automated link check covered collection URLs and DOI redirects, not individual imagery assets.

The flight workbook contains 400 metadata rows and 356 unique environment/DAP combinations, exactly matching the retained flight-date conversion table. Repeated keys reflect MOH1 C5a/C5b field records. [D2S_STAC_Links.xlsx](../results/drone/D2S_STAC_Links.xlsx) lists 16 collection links and DOIs covering the 19 analysis environments. GSD entries describe exported raster resolution, not original processing reports or georeferencing accuracy.

## QTL/LD reference data: Michel et al. (2022)

The [Michel et al. supplementary dataset](https://doi.org/10.25386/genetics.19439684) provides the genotype matrix and R/qtl2 reference files used for QTL mapping and Figure 5 LD. Download File S5 and follow the [folder setup instructions](../data/MICHEL_2022_DATA.md), including the four supporting files needed beside generated QTL cross JSONs. The genotype CSV downloaded from File S5 matches the retained analysis-input checksum. The dataset record lists CC BY 4.0.

## Larger analysis files: planned Dryad deposit

Larger study inputs and intermediates remain external: the cleaned plot table, station-weather time series, prediction genotype/relationship objects, FPCA fits, and task-level CV/LOEO/QTL outputs. The manifests below identify available archives and files. The Dryad DOI and file-level download URLs are not yet recorded here; the Michel reference-data download above is already public.

The HPRC archive collection contains individual data/script/result archives. Archive checksums establish file identity, not completeness or end-to-end model reproducibility.

[EXTERNAL_FILE_MANIFEST.csv](../data/EXTERNAL_FILE_MANIFEST.csv) records local SHA-256 values and sizes for 15 HPRC analysis archives, the verified modified BGLR library archive, the cleaned plot table, both station-weather tables, and the QTL/LD genotype CSV. [QTL_RDS_MANIFEST.csv](../data/QTL_RDS_MANIFEST.csv) adds checksums for the 58 DAP/AGDD scan objects (32.80 GB). These are local reference checksums, not proof that an uploaded deposit has been verified.

The planned Dryad data license is CC0. The BGLR library is third-party software, not CC0 data; preserve its package licenses and corresponding-source requirements before any archive redistribution. See [license scope](../../LICENSING.md).

## Provenance and gaps

- Use the archived NASA POWER responses used through EnvRtype; a future API download may differ.
- The Michel QTL/LD reference-data source is linked above. The separate all-years G2F prediction HapMap source accession and redistribution terms are not yet documented; the Michel DOI does not establish their provenance.
- All 29 DAP and 29 AGDD `QTL.Outputs.rds` scan intermediates are archived locally. See [archive status and folder mapping](../data/QTL_SCAN_ARCHIVE.md); external deposition remains pending. Flowering/yield QTL code/results are retained as optional supplementary material; their scan objects are not included in the external inventory and are required only to regenerate those supplementary analyses.
- Final TNP metrics are included here. Training, preprocessing, and evaluation are documented in [neural-process](../../neural-process/). The retained metric export has not been independently regenerated from that workflow.
- The modified BGLR archive is verified and checksummed; runtime limitations are described in [software environment](../environment/README.md).
