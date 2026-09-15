# Study data

[Supplementary resource inventory](../docs/RESOURCE_INVENTORY.md) maps the manuscript's resource names to exact files.

- `supplementary/` contains small reference inputs copied without changing their contents.
- [Michel et al. (2022) QTL/LD reference data](MICHEL_2022_DATA.md): public download, required files, and folder setup.
- [manifest.csv](manifest.csv) records included files and external input/output classes.
- [EXTERNAL_FILE_MANIFEST.csv](EXTERNAL_FILE_MANIFEST.csv) identifies 15 HPRC analysis archives, the verified modified BGLR library archive, and four larger supplementary/genotype CSVs by size and SHA-256. The QTL/LD genotype row links to its containing Michel archive; other download URLs are pending.
- [QTL_RDS_MANIFEST.csv](QTL_RDS_MANIFEST.csv) identifies all 58 DAP/AGDD scan objects. See [QTL archive notes](QTL_SCAN_ARCHIVE.md) for folder mapping and the optional flowering/yield supplement.
- [orthomosaic_file_manifest.example.csv](orthomosaic_file_manifest.example.csv) shows the local input format for raster-resolution extraction.

Set the two external roots in [config/paths.example.yml](../config/paths.example.yml). Downloaded inputs belong under the data root. Generated intermediates belong under the working-results root. Input resolvers prefer those roots and use bundled supplementary copies as a fallback where supported.

The cleaned plot table (~515 MB), weather station time series, genomic/FPCA objects, and task-level predictions remain external. Public download links are pending; a fresh-download reproduction requires these inputs. Location-level VI/PHT inputs should be under `vi_pht_location_files/`, or selected with `G2F_VI_PHT_INPUT_DIR`.

Imagery is hosted separately through Data 2 Science; see the [availability notes](../docs/DATA_AVAILABILITY.md).
