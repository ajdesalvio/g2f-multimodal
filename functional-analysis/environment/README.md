# Software environment

`DESCRIPTION` lists the R packages imported by the retained workflows. The HPRC prediction environment used R 4.4.2 and a custom environment named `BGLR_Mod_4.4.2`.

## Local analyses

Use R 4.4.2 where possible. From `functional-analysis/`, install the packages in `DESCRIPTION` into an isolated library. Spatial raster GSD extraction also requires the system libraries used by `terra`; genotype imputation requires Java and `rTASSEL`. The LD workflow uses the Bioconductor package `SNPRelate` (and its `gdsfmt` dependency).

An authoritative `renv.lock` is not committed in this repository because the accessible local library is newer than the original analysis environment, and generating a lock from it would misstate the software actually used.

## HPRC prediction environment

The one-predictor PTR model requires this BGLR source change:

```r
LT$V = LT$V[, tmp, drop = FALSE]
```

`BGLR_Mod_4.4.2.tar.gz` contains the library used for final HPRC jobs. Its **BGLR 1.1.5**, built with **R 4.4.2**, contains the exact one-predictor patch; all four prediction guards pass against its serialized helper. See [verified provenance](../patches/verified-bglr-provenance.md) for its checksum, build details, and four included packages, and [patch instructions](../patches/README.md) for a source-based installation.

This archive is an installed Linux library overlay and is not a complete R/OS environment. The `4.4.2` in the archive name identifies the R version, not the BGLR package version.

## Optional Python dependency

The MaizeMine annotation utility requires Python, `pandas`, `intermine`, and network access. It is optional for rerunning the statistical QTL scans and compiled figures. At the time of creating this GitHub repository, MaizeMine's version is v1.6. The analyses described in the manuscript used v1.5.
