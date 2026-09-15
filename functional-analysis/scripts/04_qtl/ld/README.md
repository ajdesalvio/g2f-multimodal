# Regional linkage disequilibrium

For a quick check using the retained matrices (no genotype download or
`SNPRelate` installation needed), run from this workflow's root:

```sh
Rscript scripts/04_qtl/ld/calculate_regional_ld.R --from-source-data
```

To recompute from genotypes, configure the two local roots and run:

```sh
Rscript scripts/04_qtl/ld/calculate_regional_ld.R
```

This runs both Figure 5 windows: chromosome 3 at 139-186 Mb and chromosome 7
at 125-140 Mb. It retains the original 345 W10004 lines, 0/1-to-0/2 genotype
recoding, and `SNPRelate::snpgdsLDMat(method = "corr", slide = -1)` followed by
squaring to obtain r-squared. No SNP pruning is applied. Install `SNPRelate`
through Bioconductor before running; other dependencies are listed in
`DESCRIPTION`. Set `G2F_LD_THREADS` to change the default eight threads.

Download the genotype input from [Michel et al. (2022), File S5](https://doi.org/10.25386/genetics.19439684).
See [download and extraction instructions](../../../data/MICHEL_2022_DATA.md).
Inputs under `data_dir`:

- `Michel_2022_Supplementary/cm.mbp.ss.100k.sites.SSpopulation_geno.csv`
  (large external genotype input).
- `W10004_Unique_Names.csv` (also included in `results/qtl/ld`).

Outputs go to `results_dir/04_qtl/ld`: a regenerated GDS, one r-squared RDS
and PDF per region, legend, region table, and session information. The RDS
records sample IDs and SNP order/positions. All-pairs matrices can require
substantial memory; use the source-data mode to replot the retained matrices.

The old script's QTL-task-RDS import was redundant: it subsequently replaced
those sample IDs with the included CSV. Those task objects are therefore
**not needed for the LD calculation**. See [retained Figure 5 assets](../../../results/qtl/ld/README.md)
for the historical manual assembly and final figure.
