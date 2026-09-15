library(data.table)
library(dplyr)
library(qtl2)
library(stringr)
library(tidyr)

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "scripts", "04_qtl", "shared", "qtl_paths.R"))
source(file.path(project_dir, "scripts", "04_qtl", "shared", "run_qtl_downstream.R"))

run_qtl_downstream("agdd", qtl_require_argument(), output_prefix = ".AGDD")
