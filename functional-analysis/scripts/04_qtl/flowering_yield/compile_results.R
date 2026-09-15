library(data.table)
library(dplyr)
library(tidyr)

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "scripts", "04_qtl", "shared", "qtl_paths.R"))
source(file.path(project_dir, "scripts", "04_qtl", "shared", "compile_qtl_results.R"))

compile_qtl_results(
  analyses = "flowering_yield",
  output_stem = "QTL_Results_Flowering_Yield"
)
