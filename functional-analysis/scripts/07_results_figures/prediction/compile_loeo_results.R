# Compile the final no-Ze DAP and AGDD LOEO correlation files for Tiezzi summaries.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
source(file.path(project_dir, "R", "utils", "prediction_results.R"))

paths <- g2f_paths()
analysis <- g2f_analysis_config()
inputs <- g2f_prediction_result_paths(paths, analysis)
output_dir <- g2f_results_dir(paths, "07_results_figures", "prediction_inputs")

read_domain <- function(directory, domain) {
  result <- g2f_read_csv_directory(
    directory,
    pattern = "Cor.*[.]csv$",
    expected_count = 266L
  )
  g2f_assert_no_ze(result)
  result$FPCA_Type <- domain
  result
}

correlations <- dplyr::bind_rows(
  read_domain(inputs$loeo_dap, "DAP"),
  read_domain(inputs$loeo_agdd, "AGDD")
)

parse_model <- function(filename) {
  model_codes <- names(g2f_model_labels())
  matches <- model_codes[startsWith(filename, paste0(model_codes, "."))]
  if (length(matches) != 1L) NA_character_ else matches[[1L]]
}

full_model <- vapply(correlations$File.Name, parse_model, character(1))
correlations$Model <- vapply(
  strsplit(correlations$File.Name, ".", fixed = TRUE),
  function(parts) paste(parts[1:2], collapse = "."),
  character(1)
)
model_labels <- g2f_model_labels()
correlations$ModelW <- unname(model_labels[full_model])
if (anyNA(correlations$ModelW)) stop("At least one LOEO filename has an unknown model code.")

phenotypes <- readRDS(
  g2f_resolve_input(paths, "Pheno_Data.rds", result_subdirs = "03_genomics")
) |>
  dplyr::group_by(.data$Env) |>
  dplyr::summarise(n_hybrids = dplyr::n_distinct(.data$Pedigree), .groups = "drop")

compiled <- correlations |>
  dplyr::left_join(phenotypes, by = c("Untested.Env" = "Env")) |>
  dplyr::select(dplyr::all_of(c(
    "Untested.Env", "Model", "ModelW", "n_hybrids", "FPCA_Type", "Cor"
  ))) |>
  dplyr::arrange(.data$FPCA_Type, .data$ModelW, .data$Untested.Env)

if (anyNA(compiled$n_hybrids)) stop("An LOEO environment is absent from Pheno_Data.rds.")

data.table::fwrite(
  compiled,
  file.path(output_dir, "G2F_LOEO_Results_Projections_DAP_AGDD_PTR_FPC1_Tiezzi_noZe_V3.csv")
)
