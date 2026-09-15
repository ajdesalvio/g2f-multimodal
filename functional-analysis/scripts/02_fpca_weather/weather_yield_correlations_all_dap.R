# Correlate all-DAP weather FPC scores with median environment grain yield.
# Match Figure 3: calculate yield medians from the phenomic/phenotypic BLUE
# overlap cohort, rather than every hybrid with a yield estimate.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))

paths <- g2f_paths()
output_dir <- g2f_results_dir(paths, "02_fpca_weather", "correlations")

score_files <- c(
  g2f_resolve_input(
    paths,
    "Weather_FPCA_Air_Temp_Scores.csv",
    result_subdirs = "02_fpca_weather/station"
  ),
  g2f_resolve_input(
    paths,
    "Weather_FPCA_Soil_Temp_Scores.csv",
    result_subdirs = "02_fpca_weather/station"
  ),
  g2f_resolve_input(
    paths,
    "Weather_FPCA_EnvRtype_Data_Scores_V2.csv",
    result_subdirs = "02_fpca_weather/envrtype_dap"
  )
)

scores <- lapply(score_files, data.table::fread) |>
  dplyr::bind_rows()

required_score_columns <- c("Env", "Weather.Var", paste0("FPC", 1:4))
missing_score_columns <- setdiff(required_score_columns, names(scores))
if (length(missing_score_columns) > 0L) {
  stop("Weather score files are missing: ", paste(missing_score_columns, collapse = ", "))
}

scores_wide <- scores |>
  dplyr::select(dplyr::all_of(required_score_columns)) |>
  tidyr::pivot_wider(
    names_from = "Weather.Var",
    values_from = dplyr::all_of(paste0("FPC", 1:4))
  ) |>
  dplyr::select(where(~ !any(is.na(.x))))

yield <- data.table::fread(
  g2f_resolve_input(
    paths,
    "Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv",
    result_subdirs = "01_blue_variance"
  )
)
cohort <- data.table::fread(
  g2f_resolve_input(paths, "Pedigree_Overlap_Phenomic_Phenotypic_BLUEs.csv")
)
required_yield_columns <- c("Pedigree.Env", "Env", "Yield.t.ha.BLUE")
if (!all(required_yield_columns %in% names(yield)) ||
    !"Pedigree.Env" %in% names(cohort)) {
  stop("Yield and overlap-cohort inputs are missing required columns.")
}
if (nrow(cohort) == 0L || anyNA(cohort$Pedigree.Env) ||
    anyDuplicated(cohort$Pedigree.Env)) {
  stop("The Figure 3 overlap cohort must contain unique, nonmissing record IDs.")
}
yield <- dplyr::semi_join(yield, cohort, by = "Pedigree.Env")
if (nrow(yield) != nrow(cohort) || anyDuplicated(yield$Pedigree.Env) ||
    anyNA(yield$Env) || any(!is.finite(yield$Yield.t.ha.BLUE))) {
  stop("Every Figure 3 overlap record must match exactly one finite yield BLUE.")
}
message("Figure 3 yield cohort: ", nrow(yield), " matched records in ",
        dplyr::n_distinct(yield$Env), " environments.")
environment_yield <- yield |>
  dplyr::group_by(.data$Env) |>
  dplyr::summarise(
    Median_Yield_t_ha = stats::median(.data$Yield.t.ha.BLUE, na.rm = TRUE),
    .groups = "drop"
  )

analysis_data <- dplyr::inner_join(environment_yield, scores_wide, by = "Env")
fpc_columns <- setdiff(names(analysis_data), c("Env", "Median_Yield_t_ha"))
if (nrow(analysis_data) < 3L || length(fpc_columns) == 0L) {
  stop("Too few complete environments or FPC columns for correlation analysis.")
}

correlations <- data.frame(
  FPC_ID = fpc_columns,
  Cor = vapply(
    fpc_columns,
    function(column) stats::cor(
      analysis_data$Median_Yield_t_ha,
      analysis_data[[column]],
      use = "complete.obs"
    ),
    numeric(1)
  ),
  Domain = "DAP",
  Environments = vapply(
    fpc_columns,
    function(column) sum(stats::complete.cases(
      analysis_data$Median_Yield_t_ha,
      analysis_data[[column]]
    )),
    integer(1)
  ),
  Yield_Records = nrow(yield)
) |>
  dplyr::arrange(.data$Cor)

data.table::fwrite(
  correlations,
  file.path(output_dir, "Yield_FPC_Correlations_All_DAP.csv")
)
