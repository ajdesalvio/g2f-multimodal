# QTL mapping workflow

This directory contains the manuscript's DAP-based and AGDD-based NGRDI FPC
QTL workflow, plus optional supplementary flowering/yield analyses.
Figure 5's [regional LD workflow](ld/README.md)
and original panel/composite outputs are included.

## Configuration

The [scan archive notes](../../data/QTL_SCAN_ARCHIVE.md) map the 58 archived
DAP/AGDD files to these paths. Flowering/yield task objects are needed only if
rebuilding that optional supplement, not for the manuscript's FPC analyses.

Run R scripts from `functional-analysis/`, or set `G2F_PROJECT_DIR` to that directory.
Local data and result roots come from `config/paths.yml` through
`R/utils/paths.R`. The following optional environment variables override QTL
subdirectories:

- `G2F_QTL_INPUT_DIR` and `G2F_QTL_OUTPUT_DIR` for the active job;
- `G2F_QTL_DAP_INPUT_DIR`, `G2F_QTL_AGDD_INPUT_DIR`, or
  `G2F_QTL_FLOWERING_YIELD_INPUT_DIR` (and corresponding `OUTPUT_DIR` variants)
  when several analyses are compiled together;
- `G2F_QTL_REFERENCE_DIR` for the Michel et al. (2022) qtl2 files;
- `G2F_QTL_COMPILED_DIR` and `G2F_QTL_REFINEMENT_DIR` for final tables;
- `G2F_QTL_SEED` for the 1,000 permutation scan (default `20250707`);
- `G2F_QTL_INTERVAL_FILE` to override the revised interval table (by default,
  `rerun_maxmarg.R` reads the table written to `G2F_QTL_REFINEMENT_DIR` by
  `recalculate_intervals.R`).

Download the raw genotype matrix and qtl2 JSON template from the
[Michel et al. (2022) dataset](https://doi.org/10.25386/genetics.19439684), File S5.
Follow the [download and folder setup instructions](../../data/MICHEL_2022_DATA.md).
Files are expected under `<data_dir>/Michel_2022_Supplementary` unless
`G2F_QTL_REFERENCE_DIR` (or an analysis-specific reference override) is set.

## Execution order

1. Run each analysis's `prepare_inputs_source.R` locally.
2. Copy the [four supporting reference files](../../data/MICHEL_2022_DATA.md#additional-setup-for-qtl-scans)
   beside the generated cross JSONs, then transfer the complete input directory
   to HPRC storage.
3. On HPRC, submit the selected `hprc/<analysis>/scan_array.sh` launcher under this QTL directory.
4. After all scans finish, submit `hprc/<analysis>/downstream_array.sh`.
5. Run `shared/compile_vi_qtl_results.R` to compile the manuscript FPC tables.
   Run `flowering_yield/compile_results.R` only to rebuild the optional supplement.
6. Confirm that all 29 expected tasks in each of the DAP and AGDD branches have a
   `QTL.Outputs.rds` file in their configured per-analysis output directory.
   Flowering/yield task files are needed only for the optional supplement.
   These task-level scan objects, rather than the compiled CSV tables, are the
   inputs to interval recalculation.
   Each RDS contains retained samples/phenotypes, genotype probabilities, maps,
   kinship, genome scan curves, permutation results, and peaks for one
   environment/tester/analysis. For example, `DEH1.2020.PHK76.QTL.Outputs.rds`
   holds one scan task. A compact results CSV cannot reconstruct these objects.
7. Run `refinement/recalculate_intervals.R`. It reads available files across the
   configured branches and
   writes `QTL_Results_Combined_Revised_Intervals.csv` to the configured
   refinement result directory. Missing branches or tasks are not rejected, so
   check coverage first. The 58 DAP/AGDD objects support the 167 manuscript peaks
   (71 DAP, 96 AGDD). Recreating the full 206-row supplemental table also needs
   the flowering/yield objects.
8. Run `refinement/rerun_maxmarg.R` for the genotype effect follow-up configured
   under `qtl.genotype_effect_follow_up` in `config/analysis.yml`. It reads the
   interval table from step 7 by default and reuses the corresponding DAP/AGDD
   `QTL.Outputs.rds` files. Set `G2F_QTL_INTERVAL_FILE` only to use a different
   revised-interval table deliberately.

The Slurm launchers contain no account number, email address, or user path. Pass
site-specific options to `sbatch` (for example, `sbatch -A ACCOUNT ...`). On HPRC,
`G2F_QTL_ROOT` defaults to `$SCRATCH/g2f-multimodal-qtl`; it may be overridden.
Set `G2F_R_ENVIRONMENT` only if the scripts use a TAMU custom R environment.

## Python annotation dependency

`refinement/query_maizemine_interval.py` requires `pandas` and `intermine` plus
network access to MaizeMine. Make sure to check the current MaizeMine version as
it may be higher than the archived version.
Its defaults encode both exploratory
windows: chromosome 7 (126,763,800–138,089,853 bp) and chromosome 3
(112,559,472–184,794,196 bp). See the [refinement README](refinement/README.md)
for selecting a window, retaining the multi-assembly interpretation, or rebuilding
publication tables offline from the included response snapshots.
