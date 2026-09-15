# Figure 5: QTL presence and LD

The publication figure is [`Figure_05_QTL_and_LD.pdf`](../../figures/Figure_05_QTL_and_LD.pdf).
It is an unchanged copy of `Presence.Plot.with.LD.V3_cropped.pdf`, not a newly
assembled figure. Its SHA-256 is
`A513272794C8CE9BB6F312E46A91824BB748242B88BE18B977A44798F81CDD2F`.

| Component | Retained asset or rerun |
| --- | --- |
| QTL panels A-C | `QTL_presence_panels_A_C.svg`; rerun `scripts/07_results_figures/figures/plot_qtl_presence.R` |
| Chromosome 3 LD | `LD_Chr3_139_186Mb.pdf` |
| Chromosome 7 LD | `LD_Chr7_125_140Mb.pdf` |
| Final editable assembly | `Figure_05_manual_assembly.pptx` |
| Analysis sample IDs | `W10004_Unique_Names.csv` (345 lines) |
| Regenerated matrices | `LD_Chr3_139_186Mb.rds` (2,009 SNPs); `LD_Chr7_125_140Mb.rds` (840 SNPs) |

The original figure combined the panels manually in PowerPoint and was then
exported/cropped. The unchanged PowerPoint is included to preserve that layout
step; rerunning the R scripts produces component panels, not a byte-identical
final composite. Whole-genome QTL-point jitter was unseeded in the old script;
the portable rerun sets seed 42 for stable display only.

The QTL panels exclude flowering/yield traits, retaining 167 FPC QTL. The LD
panels use the same W10004 panel and SNP-index axes as the original analysis;
indices should not be interpreted as equal physical distances.

Original r-squared matrices were not saved locally. The included RDS matrices
were regenerated on 2026-09-14 with the [portable LD script](../../../scripts/04_qtl/ld/README.md)
and contain SNP positions, sample IDs and method/package metadata. Both rebuilt
heatmaps rendered pixel-identically to the original panels at 700-pixel size.
`LD_regions.csv` and `sessionInfo.txt` document that rerun. The retained PDFs
remain the original outputs. Use the script's `--from-source-data` option to
replot the included matrices without the large external genotype CSV.
