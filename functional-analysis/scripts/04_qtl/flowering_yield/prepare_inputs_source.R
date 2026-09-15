library(data.table)
library(dplyr)

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "scripts", "04_qtl", "shared", "qtl_paths.R"))
source(file.path(project_dir, "scripts", "04_qtl", "shared", "prepare_cross_inputs.R"))
paths <- qtl_project_paths("flowering_yield")

phenotype_file <- file.path(paths$data_dir, "Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv")
qtl_require_file(phenotype_file, "Flowering and yield BLUEs")
phenotype_columns <- c("DTA.BLUE", "DTS.BLUE", "ASI", "Yield.t.ha.BLUE")
phenotypes <- fread(phenotype_file, data.table = FALSE) |>
  arrange(Env, Pedigree) |>
  select(Pedigree, Pedigree.Env, Env, all_of(phenotype_columns))

prepare_qtl_crosses(
  analysis = "flowering_yield",
  phenotype_data = phenotypes,
  phenotype_columns = phenotype_columns,
  phenotype_suffix = ".subset.phenotype.yield.flowering.data.csv"
)
