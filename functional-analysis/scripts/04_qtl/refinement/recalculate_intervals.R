library(data.table)
library(qtl2)

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "scripts", "04_qtl", "shared", "qtl_paths.R"))

analyses <- c("dap", "agdd", "flowering_yield")
analysis_labels <- c(dap = "DAP", agdd = "AGDD", flowering_yield = "Flowering.Yield")
interval_tables <- list()

for (analysis in analyses) {
  paths <- qtl_project_paths(analysis, create = TRUE)
  result_files <- list.files(
    paths$output_dir,
    pattern = "QTL\\.Outputs\\.rds$",
    full.names = TRUE
  )
  for (result_file in result_files) {
    env_tester <- qtl_env_tester_from_rds(result_file)
    message("Recalculating intervals from ", result_file)
    container <- readRDS(result_file)
    result <- container[[env_tester]]
    if (is.null(result) && length(container) == 1L) result <- container[[1L]]
    if (is.null(result)) stop("Cannot identify result object in ", result_file)

    peaks <- qtl2::find_peaks(
      scan1_output = result$scan.result,
      map = result$pmap,
      threshold = summary(result$permutation.result),
      drop = 1.5
    )
    if (!nrow(peaks)) next
    parts <- qtl_split_env_tester(env_tester)
    peaks$Env.Tester <- env_tester
    peaks$FPCA_type <- analysis_labels[[analysis]]
    peaks$Env <- parts$Env
    peaks$Tester <- parts$Tester
    interval_tables[[paste(analysis, env_tester, sep = ":")]] <- peaks
  }
}

if (!length(interval_tables)) {
  stop(
    "No QTL result RDS files were found. Configure the per-analysis output ",
    "directories with G2F_QTL_<ANALYSIS>_OUTPUT_DIR."
  )
}
intervals <- rbindlist(interval_tables, fill = TRUE)
intervals$ci_width <- intervals$ci_hi - intervals$ci_lo

paths <- qtl_project_paths("dap", create = TRUE)
output_file <- file.path(
  paths$refinement_dir,
  "QTL_Results_Combined_Revised_Intervals.csv"
)
fwrite(intervals, output_file)
message("Saved ", output_file)
