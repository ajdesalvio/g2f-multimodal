# Filter G2F markers, create a complete dosage matrix, and calculate genotype PCA.

# Java memory must be configured before rTASSEL (and therefore rJava) is loaded.
java_max_gb <- Sys.getenv("G2F_JAVA_MAX_GB", unset = "80")
java_initial_gb <- Sys.getenv("G2F_JAVA_INITIAL_GB", unset = "5")
options(java.parameters = c(
  paste0("-Xmx", java_max_gb, "g"),
  paste0("-Xms", java_initial_gb, "g")
))

suppressPackageStartupMessages({
  library(data.table)
  library(rTASSEL)
  library(ggplot2)
  library(cowplot)
})

source(file.path("R", "utils", "paths.R"))
paths <- g2f_paths()

prep_dir <- g2f_results_file(paths, "00_data_prep")
output_dir <- g2f_results_file(paths, "03_genomics")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input_hapmap <- g2f_data_file(
  paths,
  "5_Genotype_Data_All_Years_TASSEL_hapmap.hmp.txt"
)
overlap_file <- file.path(prep_dir, "Pedigree_Overlap_Genomic_Phenomic.csv")
if (!file.exists(overlap_file)) {
  overlap_file <- g2f_data_file(
    paths,
    "Pedigree_Overlap_Genomic_Phenomic.csv"
  )
}
g2f_require_file(input_hapmap, "Unfiltered TASSEL HapMap")
g2f_require_file(overlap_file, "Genomic/phenomic pedigree overlap")

common <- sort(unique(fread(overlap_file, select = "Pedigree")$Pedigree))
hmp <- as.data.frame(
  fread(input_hapmap, header = TRUE, check.names = FALSE),
  check.names = FALSE
)

metadata_columns <- names(hmp)[seq_len(11L)]
missing_pedigrees <- setdiff(common, names(hmp))
if (length(missing_pedigrees) > 0L) {
  stop(
    "HapMap is missing pedigrees in the overlap file: ",
    paste(missing_pedigrees, collapse = ", ")
  )
}
hmp <- hmp[, c(metadata_columns, common), drop = FALSE]

# Retain marker rows with exactly two allele states (the historical biallelic
# SNP rule), then write a standards-compatible intermediate HapMap.
allele_counts <- lengths(strsplit(hmp$alleles, "/", fixed = TRUE))
hmp <- hmp[allele_counts == 2L, , drop = FALSE]
filtered_hapmap_file <- file.path(
  output_dir,
  "5_Genotype_Data_All_Years_hapmap_filtered.hmp.txt"
)
fwrite(hmp, filtered_hapmap_file, sep = "\t", quote = FALSE)
rm(hmp)
invisible(gc())

genotypes <- readGenotypeTableFromPath(filtered_hapmap_file)

# The final genotype table used no missing calls and MAF >= 0.05.
n_taxa <- length(common)
genotypes_filtered <- filterGenotypeTableSites(
  genotypes,
  siteMinAlleleFreq = 0.05,
  siteMaxAlleleFreq = 1.0,
  siteMinCount = n_taxa
)

dosage <- as.matrix(genotypes_filtered)
if (anyNA(dosage)) stop("The filtered dosage matrix still contains missing calls.")
if (!setequal(rownames(dosage), common)) {
  stop("Filtered dosage-matrix taxa do not match the pedigree overlap.")
}
dosage <- dosage[common, , drop = FALSE]

dosage_file <- file.path(
  output_dir,
  "5_Genotype_Data_All_Years_hapmap_commonped_nomissing.csv"
)
fwrite(as.data.frame(dosage), dosage_file, row.names = TRUE)

genotype_pca <- pca(genotypes_filtered)
pc_scores <- genotype_pca@results[["PC_Datum"]]
pc_scores$Tester <- ifelse(
  grepl("PHK76", pc_scores$Taxa),
  "PHK76",
  ifelse(
    grepl("PHP02", pc_scores$Taxa),
    "PHP02",
    ifelse(grepl("PHZ51", pc_scores$Taxa), "PHZ51", "Hybrid Checks")
  )
)
pc_scores$Tester <- factor(
  pc_scores$Tester,
  levels = c("PHK76", "PHP02", "PHZ51", "Hybrid Checks")
)

pve <- genotype_pca@results[["Eigenvalues_Datum"]]
eigenvectors <- genotype_pca@results[["Eigenvectors_Datum"]]
fwrite(pc_scores, file.path(output_dir, "PCA_Scores_Genotype_Data.csv"))
fwrite(
  pve,
  file.path(output_dir, "PCA_Percent_Variation_Explained_Genotype_Data.csv")
)
fwrite(
  eigenvectors,
  file.path(output_dir, "PCA_Eigenvectors_Genotype_Data.csv")
)

pc_percent <- 100 * pve$proportion_of_total[1:3]
tester_colors <- c(
  PHK76 = "#648fff",
  PHP02 = "#785ef0",
  PHZ51 = "#dc267f",
  `Hybrid Checks` = "#fe6100"
)
tester_shapes <- c(PHK76 = 21, PHP02 = 22, PHZ51 = 23, `Hybrid Checks` = 24)

make_pc_plot <- function(x_name, y_name, x_index, y_index) {
  ggplot(pc_scores, aes(x = .data[[x_name]], y = .data[[y_name]])) +
    geom_point(aes(shape = Tester, fill = Tester), size = 2, stroke = 0) +
    scale_shape_manual(values = tester_shapes) +
    scale_fill_manual(values = tester_colors) +
    theme_classic() +
    labs(
      x = sprintf("PC%d (%.1f%%)", x_index, pc_percent[x_index]),
      y = sprintf("PC%d (%.1f%%)", y_index, pc_percent[y_index])
    )
}

pc12 <- make_pc_plot("PC1", "PC2", 1, 2)
pc13 <- make_pc_plot("PC1", "PC3", 1, 3)
pc23 <- make_pc_plot("PC2", "PC3", 2, 3)
legend <- get_legend(pc12 + theme(legend.position = "right"))
panel <- plot_grid(
  pc12 + theme(legend.position = "none"),
  pc13 + theme(legend.position = "none"),
  pc23 + theme(legend.position = "none"),
  legend,
  nrow = 1,
  rel_widths = c(1, 1, 1, 0.65)
)
ggsave(
  file.path(output_dir, "PCA_Plots_Genotypic_Data.jpg"),
  panel,
  width = 12,
  height = 4,
  units = "in",
  dpi = 800
)
message("Wrote filtered genotype and PCA outputs to ", output_dir)
