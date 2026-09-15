# Find pedigree-environment observations shared by phenomic and phenotype data.

suppressPackageStartupMessages({
  library(data.table)
})

source(file.path("R", "utils", "paths.R"))
paths <- g2f_paths()

prep_dir <- g2f_results_file(paths, "00_data_prep")
blue_dir <- g2f_results_file(paths, "01_blue_variance")
fpca_dir <- g2f_results_file(paths, "02_fpca_weather")
dir.create(prep_dir, recursive = TRUE, showWarnings = FALSE)

prefer_generated <- function(filename, generated_dir) {
  generated <- file.path(generated_dir, filename)
  archived <- g2f_data_file(paths, filename)
  if (file.exists(generated)) generated else archived
}

common_file <- prefer_generated(
  "Pedigree_Overlap_Genomic_Phenomic.csv",
  prep_dir
)
phenotype_file <- prefer_generated(
  "Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv",
  blue_dir
)
phenomic_file <- prefer_generated(
  "FPC_Scores_BLUEs_Unified_FPCA_Max_FPCs_2020_2021_G2F_VIs.csv",
  fpca_dir
)

g2f_require_file(common_file, "Genomic/phenomic pedigree overlap")
g2f_require_file(phenotype_file, "Combined phenotype BLUEs")
g2f_require_file(phenomic_file, "Full-data VI FPC scores")

common <- fread(common_file, select = "Pedigree")$Pedigree
phenotypes <- fread(
  phenotype_file,
  select = c("Pedigree", "Env", "Pedigree.Env")
)
phenomic <- fread(
  phenomic_file,
  select = c("Pedigree", "Env", "Pedigree.Env")
)
phenomic <- unique(phenomic[Pedigree %in% common])
phenotypes <- unique(phenotypes)

overlap <- intersect(phenomic$Pedigree.Env, phenotypes$Pedigree.Env)
ped_env <- unique(
  phenomic[Pedigree.Env %in% overlap, .(Pedigree.Env, Pedigree, Env)]
)
setorder(ped_env, Env, Pedigree)
ped_env[, Env := NULL]

if (!setequal(unique(ped_env$Pedigree), common)) {
  stop(
    "The pedigree-environment overlap does not retain every pedigree in the ",
    "genomic/phenomic overlap."
  )
}

output_file <- file.path(
  prep_dir,
  "Pedigree_Overlap_Phenomic_Phenotypic_BLUEs.csv"
)
fwrite(ped_env, output_file)
message("Wrote ", nrow(ped_env), " pedigree-environment rows to ", output_file)
