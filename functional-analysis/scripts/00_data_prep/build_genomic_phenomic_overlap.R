# Build the pedigree intersection between genotype samples and VI FPC scores.

suppressPackageStartupMessages({
  library(data.table)
})

source(file.path("R", "utils", "paths.R"))
paths <- g2f_paths()

vcf_file <- g2f_data_file(paths, "5_Genotype_Data_All_Years.vcf")
phenomic_file <- g2f_data_file(
  paths,
  "FPC_Scores_BLUEs_Unified_FPCA_Max_FPCs_2020_2021_G2F_VIs.csv"
)
g2f_require_file(vcf_file, "Genotype VCF")
g2f_require_file(phenomic_file, "Full-data VI FPC scores")

# Read only the VCF header; loading the multi-gigabyte marker body is unnecessary
# for determining sample overlap.
connection <- file(vcf_file, open = "r")
on.exit(close(connection), add = TRUE)
header <- NULL
repeat {
  line <- readLines(connection, n = 1L, warn = FALSE)
  if (length(line) == 0L) break
  if (startsWith(line, "#CHROM")) {
    header <- line
    break
  }
}
if (is.null(header)) stop("The VCF does not contain a #CHROM header line.")

vcf_columns <- strsplit(header, "\t", fixed = TRUE)[[1L]]
if (length(vcf_columns) < 10L) stop("The VCF contains no genotype samples.")
genotype_pedigrees <- vcf_columns[10:length(vcf_columns)]

phenomic_pedigrees <- unique(
  fread(phenomic_file, select = "Pedigree")$Pedigree
)
common <- data.table(
  Pedigree = sort(intersect(phenomic_pedigrees, genotype_pedigrees))
)
if (nrow(common) == 0L) stop("No pedigree overlap was found.")

output_dir <- g2f_results_file(paths, "00_data_prep")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_file <- file.path(
  output_dir,
  "Pedigree_Overlap_Genomic_Phenomic.csv"
)
fwrite(common, output_file)
message("Wrote ", nrow(common), " shared pedigrees to ", output_file)
