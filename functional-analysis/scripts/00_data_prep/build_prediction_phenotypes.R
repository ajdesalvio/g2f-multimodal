# Build the ordered phenotype object used by CV and LOEO prediction scripts.

suppressPackageStartupMessages({
  library(data.table)
})

source(file.path("R", "utils", "paths.R"))
paths <- g2f_paths()

prep_dir <- g2f_results_file(paths, "00_data_prep")
blue_dir <- g2f_results_file(paths, "01_blue_variance")
dir.create(prep_dir, recursive = TRUE, showWarnings = FALSE)

prefer_generated <- function(filename, generated_dir) {
  generated <- file.path(generated_dir, filename)
  archived <- g2f_data_file(paths, filename)
  if (file.exists(generated)) generated else archived
}

phenotype_file <- prefer_generated(
  "Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv",
  blue_dir
)
overlap_file <- prefer_generated(
  "Pedigree_Overlap_Phenomic_Phenotypic_BLUEs.csv",
  prep_dir
)
g2f_require_file(phenotype_file, "Combined phenotype BLUEs")
g2f_require_file(overlap_file, "Phenomic/phenotypic overlap")

phenotypes <- as.data.frame(fread(phenotype_file))
overlap <- fread(overlap_file, select = "Pedigree.Env")$Pedigree.Env

nonmissing_ids <- phenotypes$Pedigree.Env[!is.na(phenotypes$Pedigree.Env)]
if (anyDuplicated(nonmissing_ids)) {
  stop("Combined phenotype BLUEs contain duplicate Pedigree.Env values.")
}
if (anyDuplicated(overlap)) {
  stop("The overlap file contains duplicate Pedigree.Env values.")
}

row_index <- match(overlap, phenotypes$Pedigree.Env)
if (anyNA(row_index)) {
  stop("Some overlap rows are absent from the combined phenotype BLUEs.")
}
phenotypes <- phenotypes[row_index, , drop = FALSE]
rownames(phenotypes) <- NULL

output_file <- file.path(prep_dir, "Pheno_Data.rds")
saveRDS(phenotypes, output_file)
message("Wrote ", nrow(phenotypes), " ordered phenotype rows to ", output_file)
