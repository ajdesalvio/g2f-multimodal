# QTL intervals and exploratory gene annotations

1. `recalculate_intervals.R` derives 1.5 LOD support intervals from retained per-cross QTL RDS files.
2. `query_maizemine_interval.py` queries both configured intervals, once each. It is the non-interactive version of `Query_MaizeMine_Specific_Interval_V3.py`.
3. `export_publication_tables.R` exports the 167 NGRDI FPC peaks and groups named genes from the retained annotation snapshots. It works offline.

The manuscript interval rebuild uses the 58 DAP/AGDD task objects in the
[scan archive](../../../data/QTL_SCAN_ARCHIVE.md). Flowering/yield RDS files are
optional and needed only to reproduce the additional 39 peaks in the retained
206-row supplemental table. `recalculate_intervals.R` processes available files
and does not reject incomplete branches. Confirm task coverage first and check
the regenerated FPC subset against the retained 71 DAP and 96 AGDD peaks.

From `functional-analysis/`:

```sh
Rscript scripts/04_qtl/refinement/export_publication_tables.R
# Optional live annotation refresh (requires pandas and intermine):
python scripts/04_qtl/refinement/query_maizemine_interval.py
```

The exploratory queries use chromosome 7: **126,763,800–138,089,853 bp** and chromosome 3: **112,559,472–184,794,196 bp**. The broad chromosome-7 interval from WIH3.2020.PHP02 was excluded when selecting the lower bound. These are fixed exploratory intervals, not automatically optimized intervals.

The query retains overlapping records across assemblies without coordinate harmonization. Genes lacking a symbol or whose symbol contains `LOC` are excluded from the named-gene summaries; remaining records are grouped by symbol, retaining their identifiers and descriptions. These tables are exploratory annotations, not a complete single-assembly gene inventory or evidence that a gene is causal.

The historical source assigned the MaizeMine v1.5 endpoint and then overwrote it with `http://maizemine.rnet.missouri.edu:8080/maizemine/service`. The retained script preserves the effective endpoint and exposes `--service-url` for an explicit change. Publication tables use the saved responses in [annotation_snapshots](../../../results/qtl/annotation_snapshots/), so they do not depend on the current live service. Historical snapshot filenames have approximate bounds, whereas the exact bounds above are authoritative. New live queries use exact coordinates in their filenames.
