library(qtl2)

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "scripts", "04_qtl", "shared", "qtl_paths.R"))
source(file.path(project_dir, "scripts", "04_qtl", "shared", "run_qtl_scan.R"))

run_qtl_scan("flowering_yield", qtl_require_argument())
