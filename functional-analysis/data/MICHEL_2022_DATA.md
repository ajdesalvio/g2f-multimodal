# Michel et al. (2022) reference data

The QTL and LD workflows use the **[Michel et al. (2022) supplementary dataset](https://doi.org/10.25386/genetics.19439684)**, associated with *Genetic mapping and prediction of flowering time and plant height in a maize Stiff Stalk MAGIC population* ([paper](https://doi.org/10.1093/genetics/iyac063)).

## Download and extract

1. Download **[File S5 control.tar.gz (12.7 MB)](https://ndownloader.figshare.com/files/34591628)**. This R/qtl2 control archive contains the reference files used here; the full 6+ GB Figshare download is not needed for the QTL/LD setup.
2. Extract it and place the **contents of `control/`** directly in `<data_dir>/Michel_2022_Supplementary/`. Do not leave an extra `control/` level between that directory and its files. Configure `data_dir` using the [two-root setup](../config/README.md).
3. Retain the original filenames. QTL preparation reads `cm.mbp.ss.100k.sites.json` and `cm.mbp.ss.100k.sites.SSpopulation_geno.csv`; LD reads the same genotype CSV.

The **File S5 component** has MD5 `8cbfcb1c038ecf987d72a260c9177f0a`. Its genotype CSV matches the SHA-256 recorded in [EXTERNAL_FILE_MANIFEST.csv](EXTERNAL_FILE_MANIFEST.csv): `D5824BFE86D639D4153CC8B4D097786BED39258E0009A0C72F181DA04557C39E`. These file-identity checks were performed on 15 September 2026; they do not validate every file in the full package or constitute a model refit.

## Additional setup for QTL scans

After running each branch's `prepare_inputs_source.R`, copy these four files from the extracted reference directory **beside the generated cross JSONs**:

- `cm.mbp.ss.100k.sites.SSparental_geno.csv`
- `cm.mbp.ss.100k.sites.mbp_to_cm_map.csv`
- `cm.mbp.ss.100k.sites.physical_marker_map.csv`
- `cm.mbp.ss.100k.sites.ssCross.csv`

The default destination is `<results_dir>/qtl/dap/input/` or `<results_dir>/qtl/agdd/input/`; use `<results_dir>/qtl/flowering_yield/input/` only for that optional supplement. Include these files when transferring the prepared input directory to HPRC. The preparation scripts replace the JSON's population-genotype and phenotype references, but retain relative paths to these four supporting files; they do not copy them automatically. The original `ss.blue.phenotypes.csv` is replaced by this study's generated phenotype files.

The [QTL configuration](../scripts/04_qtl/README.md) supports reference/input directory overrides. The LD script always reads the genotype CSV under `<data_dir>/Michel_2022_Supplementary/`; it does not use `G2F_QTL_REFERENCE_DIR`. To replot the included LD matrices without downloading genotypes, use the [LD source-data mode](../scripts/04_qtl/ld/README.md).

## Citation and scope

The [dataset record](https://doi.org/10.25386/genetics.19439684) lists CC BY 4.0. Cite Michel et al. and the dataset DOI when using these materials. This download supplies the QTL/LD reference inputs; it is not a verified source for the separate all-years G2F prediction HapMap or a replacement for this study's phenotype, FPCA, or prediction archives.
