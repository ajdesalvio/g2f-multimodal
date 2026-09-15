library(data.table)
library(dplyr)
library(tidyr)

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "scripts", "04_qtl", "shared", "qtl_paths.R"))
source(file.path(project_dir, "scripts", "04_qtl", "shared", "prepare_cross_inputs.R"))
paths <- qtl_project_paths("dap")

score_file <- file.path(
  paths$data_dir,
  "FPC_Scores_BLUEs_Unified_FPCA_Max_FPCs_2020_2021_G2F_VIs.csv"
)
qtl_require_file(score_file, "DAP FPCA scores")
scores <- fread(score_file, data.table = FALSE) |>
  arrange(Env, Pedigree, Vegetation.Index)

fpc_names <- paste0("FPC", 1:5)
phenotype_columns <- paste0(fpc_names, "_NGRDI")
scores_wide <- scores |>
  pivot_wider(
    id_cols = c(Pedigree.Env, Env, Year, Pedigree),
    names_from = Vegetation.Index,
    names_glue = "{.value}_{Vegetation.Index}",
    values_from = all_of(fpc_names),
    values_fill = NA
  ) |>
  as.data.frame()

prepare_qtl_crosses(
  analysis = "dap",
  phenotype_data = scores_wide,
  phenotype_columns = phenotype_columns,
  phenotype_suffix = ".subset.phenotype.FPC.data.csv"
)
