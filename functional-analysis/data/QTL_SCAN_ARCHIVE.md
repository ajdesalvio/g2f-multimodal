# QTL scan archive

The manuscript uses DAP and AGDD NGRDI FPC QTL. All **58 expected scan files**
are available locally: 29 environment/tester tasks per domain, totaling
**32,804,251,447 bytes (32.80 GB)**. The large RDS files are not stored in GitHub;
their external deposit DOI and download links remain pending.

[QTL_RDS_MANIFEST.csv](QTL_RDS_MANIFEST.csv) records each file's task, size,
SHA-256 checksum, original archive-relative path, and expected path under the
working-results root. All 58 filenames match the retained task list with no
missing, duplicate, or unexpected tasks, and no files changed during hashing
on 15 September 2026. These are local reference checksums, not a comparison
against independently supplied HPRC-side checksums.

All 58 files also passed sequential `readRDS()` and structural checks on
15 September 2026: task keys, 13 required scientific fields, expected FPC
columns, 100,000 scan positions, 1,000 permutations, and probability/map/sample
dimensions across 10 chromosomes. File sizes and modification times remained
unchanged during inspection. These checks did not rerun models, recompute
intervals, or validate numerical scan values.

## Folder mapping

| Archive folder | Analysis | Canonical location under `results_dir` |
| --- | --- | --- |
| `Standard_FPCA/` | DAP NGRDI FPC1–5 | `qtl/dap/output/` |
| `AGDD_FPCA/` | AGDD NGRDI FPC1–7 | `qtl/agdd/output/` |

Keep the original filenames, including names without a dot before
`QTL.Outputs.rds`; the scripts accept both naming conventions. Alternatively,
leave the files where they are and set `G2F_QTL_DAP_OUTPUT_DIR` and
`G2F_QTL_AGDD_OUTPUT_DIR` to their respective directories. Do not set one generic
QTL output directory for both domains.

The objects retain genotype probabilities, maps, phenotypes, kinship, genome
scans, permutation results, and peaks. Historical objects lack the newer
`analysis` and `permutation.seed` metadata fields; the retained downstream
scripts do not require those fields. Their absence does not invalidate the
saved scientific objects or establish a historical random seed.

## Optional flowering/yield supplement

Flowering/yield QTL code and results remain as supplementary repository
material, not manuscript analyses. Their RDS files are not included in this
archive and are required only to regenerate those supplementary analyses.

To regenerate the full supplemental 206-row interval table, also provide those
scan files through `G2F_QTL_FLOWERING_YIELD_OUTPUT_DIR`. Without them, the
interval recalculation script processes the available DAP/AGDD files only;
its output does not reconstruct the additional 39 flowering/yield peaks.
The checked-in full table remains unchanged. The script does not reject partial
branches: verify task coverage first, then compare the regenerated FPC results
with the retained 71 DAP and 96 AGDD peaks.
