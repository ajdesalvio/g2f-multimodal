# QTL results

Start with [publication/QTL_Results_NGRDI_FPC_167_Peaks.csv](publication/QTL_Results_NGRDI_FPC_167_Peaks.csv): the manuscript's 167 significant NGRDI FPC peaks across analyses (71 DAP, 96 AGDD). The full `QTL_Results_Combined_Revised_Intervals.csv` is preserved separately and also includes 39 flowering/yield peaks that were not discussed in the paper (206 rows total).

Other retained tables contain permutation thresholds, chromosome 7 maxmarg estimates, gene hits, and optional supplementary flowering/yield results. Multi-gigabyte LOD/BLUP tables and per-cross HPRC RDS files belong in the external data archive. Public Dryad access for all 58 manuscript DAP/AGDD objects is pending. Flowering/yield scan objects are not included in the external inventory; they are needed only to regenerate the supplementary analyses.

`annotation_snapshots/` preserves the original chromosome-3 and chromosome-7 MaizeMine responses collapsed by gene identifier. `publication/` contains named gene summaries grouped by symbol. The existing `QTL_chr7_125_140Mb_genes_GO_collapsed_for_table.csv` preserves the 10-row, manually formatted chromosome 7 manuscript table. See the [annotation method and exact intervals](../../scripts/04_qtl/refinement/README.md) before interpretation.
