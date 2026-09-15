# Construct additive and dominance genomic relationship matrices.

suppressPackageStartupMessages({
  library(data.table)
  library(AGHmatrix)
})

source(file.path("R", "utils", "paths.R"))
paths <- g2f_paths()

prep_dir <- g2f_results_file(paths, "00_data_prep")
output_dir <- g2f_results_file(paths, "03_genomics")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

dosage_file <- file.path(
  output_dir,
  "5_Genotype_Data_All_Years_hapmap_commonped_nomissing.csv"
)
if (!file.exists(dosage_file)) {
  dosage_file <- g2f_data_file(
    paths,
    "5_Genotype_Data_All_Years_hapmap_commonped_nomissing.csv"
  )
}
overlap_file <- file.path(prep_dir, "Pedigree_Overlap_Genomic_Phenomic.csv")
if (!file.exists(overlap_file)) {
  overlap_file <- g2f_data_file(
    paths,
    "Pedigree_Overlap_Genomic_Phenomic.csv"
  )
}
g2f_require_file(dosage_file, "Complete marker-dosage matrix")
g2f_require_file(overlap_file, "Genomic/phenomic pedigree overlap")

dosage_table <- fread(dosage_file, check.names = FALSE)
pedigrees <- dosage_table[[1L]]
dosage <- as.matrix(dosage_table[, -1L])
storage.mode(dosage) <- "numeric"
rownames(dosage) <- pedigrees
rm(dosage_table)

if (anyNA(dosage)) stop("The marker-dosage matrix contains missing values.")
common <- sort(unique(fread(overlap_file, select = "Pedigree")$Pedigree))
if (!setequal(rownames(dosage), common)) {
  stop("Dosage-matrix taxa do not match the pedigree overlap.")
}
dosage <- dosage[common, , drop = FALSE]

# MAF filtering was completed in genomic_imputation.R.
g_add <- Gmatrix(dosage, method = "VanRaden")
g_dom <- Gmatrix(dosage, method = "Vitezica")
g_add <- g_add[common, common, drop = FALSE]
g_dom <- g_dom[common, common, drop = FALSE]

additive_file <- file.path(output_dir, "GENOMIC.RELAT.MAT.ADD.csv")
dominance_file <- file.path(output_dir, "GENOMIC.RELAT.MAT.DOM.csv")
fwrite(as.data.frame(g_add), additive_file, row.names = TRUE)
fwrite(as.data.frame(g_dom), dominance_file, row.names = TRUE)
message("Wrote additive and dominance GRMs to ", output_dir)
