# Figure 5: LD in the 345 W10004 lines used in QTL mapping.
# Adapted from LD_V6.R; both published regions run without interactive edits.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(cowplot)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
paths <- g2f_paths()
output_dir <- g2f_results_dir(paths, "04_qtl", "ld")

render_ld <- function(ld_result, stem) {
  ld_matrix <- ld_result$r_squared
  stopifnot(nrow(ld_matrix) == ncol(ld_matrix),
            nrow(ld_matrix) == length(ld_result$snp_id))
  # Match the original lower-triangle heatmap: axes are SNP index, not Mb.
  cells <- as.data.table(which(!is.na(ld_matrix), arr.ind = TRUE))
  setnames(cells, c("i", "j"))
  cells[, r2 := ld_matrix[cbind(i, j)]]
  heatmap <- ggplot(cells[j <= i], aes(x = i, y = j, fill = r2)) +
    geom_raster() + coord_equal(expand = FALSE) +
    scale_fill_gradient2(low = "grey90", mid = "lightpink", high = "red",
                         limits = c(0, 1), midpoint = 0.5, na.value = "white") +
    labs(fill = expression(r^2)) + theme_void()
  legend <- cowplot::get_legend(heatmap)
  ggsave(file.path(output_dir, paste0(stem, ".pdf")),
         heatmap + theme(legend.position = "none"),
         device = cairo_pdf, width = 6, height = 6)
  if (ld_result$region$chromosome == 3L) {
    ggsave(file.path(output_dir, "LD_legend.pdf"), cowplot::ggdraw(legend),
           device = cairo_pdf, width = 1.5, height = 3)
  }
}

if ("--from-source-data" %in% commandArgs(trailingOnly = TRUE)) {
  # Quick review: no genotype download or SNPRelate installation is needed.
  for (stem in c("LD_Chr3_139_186Mb", "LD_Chr7_125_140Mb")) {
    input_file <- file.path(paths$project_dir, "results", "qtl", "ld", paste0(stem, ".rds"))
    g2f_require_file(input_file, "Retained LD matrix")
    render_ld(readRDS(input_file), stem)
    message("Replotted ", stem, " from the retained LD matrix.")
  }
} else {
suppressPackageStartupMessages(library(SNPRelate))

genotype_file <- g2f_data_file(
  paths, "Michel_2022_Supplementary", "cm.mbp.ss.100k.sites.SSpopulation_geno.csv"
)
sample_candidates <- c(
  g2f_data_file(paths, "W10004_Unique_Names.csv"),
  file.path(paths$project_dir, "results", "qtl", "ld", "W10004_Unique_Names.csv")
)
sample_file <- sample_candidates[file.exists(sample_candidates)][1L]
g2f_require_file(genotype_file, "Michel et al. genotype matrix")
if (is.na(sample_file)) stop("Missing W10004_Unique_Names.csv.")

sample_ids <- read.csv(sample_file, stringsAsFactors = FALSE)$W10004_Unique_Names
if (length(sample_ids) != 345L || anyNA(sample_ids) || anyDuplicated(sample_ids)) {
  stop("Expected the 345 distinct W10004 lines used for the published LD analysis.")
}
threads <- as.integer(Sys.getenv("G2F_LD_THREADS", unset = "8"))
if (is.na(threads) || threads < 1L) stop("G2F_LD_THREADS must be a positive integer.")

# The original script extracted sample IDs from QTL task objects, but then
# replaced them with this archived CSV. Only the CSV is needed for this analysis.
genotypes <- as.data.frame(data.table::fread(genotype_file))
if (!identical(names(genotypes)[1L], "lines") || anyDuplicated(genotypes$lines)) {
  stop("The first genotype column must contain unique sample IDs named 'lines'.")
}
missing_samples <- setdiff(sample_ids, genotypes$lines)
if (length(missing_samples)) {
  stop("W10004 lines missing from genotype matrix: ", paste(missing_samples, collapse = ", "))
}
snp_ids <- names(genotypes)[-1L]
if (anyDuplicated(snp_ids) || any(!grepl("^S[0-9]+_[0-9]+.*$", snp_ids))) {
  stop("Expected distinct SNP IDs with chromosome/position encoded as S<chr>_<bp>.")
}
chromosome <- as.integer(sub("^S([0-9]+)_.*$", "\\1", snp_ids))
position <- as.numeric(sub("^S[0-9]+_([0-9]+).*$", "\\1", snp_ids))
genotype_matrix <- as.matrix(genotypes[match(sample_ids, genotypes$lines), -1L, drop = FALSE])
if (any(!is.na(genotype_matrix) & !(genotype_matrix %in% c(0, 1)))) {
  stop("Expected the original 0/1 homozygous genotype coding (or missing values).")
}
# Match LD_V6.R: recode homozygous 0/1 calls to 0/2 for SNPRelate.
genotype_matrix <- genotype_matrix * 2L

main <- function() {
  # Store regenerated data under the results root, never overwrite the raw GDS.
  gds_path <- file.path(output_dir, "All_W10004.gds")
  SNPRelate::snpgdsCreateGeno(
    gds.fn = gds_path, genmat = t(genotype_matrix), sample.id = sample_ids,
    snp.id = snp_ids, snp.chromosome = chromosome, snp.position = position,
    snpfirstdim = TRUE
  )
  genofile <- SNPRelate::snpgdsOpen(gds_path)
  on.exit(SNPRelate::snpgdsClose(genofile), add = TRUE)

  regions <- data.frame(
    chromosome = c(3L, 7L), start_bp = c(139e6, 125e6), end_bp = c(186e6, 140e6)
  )
  for (region_index in seq_len(nrow(regions))) {
    region <- regions[region_index, ]
    selected <- chromosome == region$chromosome &
      position >= region$start_bp & position <= region$end_bp
    if (sum(selected) < 2L) stop("Fewer than two SNPs in chromosome ", region$chromosome)
    # Preserve the published estimator and all-pairs window; no LD pruning.
    ld <- SNPRelate::snpgdsLDMat(
      genofile, snp.id = snp_ids[selected], method = "corr", slide = -1,
      num.thread = threads
    )
    ld_matrix <- ld$LD^2
    stopifnot(nrow(ld_matrix) == ncol(ld_matrix))
    stem <- paste0("LD_Chr", region$chromosome, "_", region$start_bp / 1e6,
                   "_", region$end_bp / 1e6, "Mb")
    ld_result <- list(
        r_squared = ld_matrix, snp_id = ld$snp.id,
        snp_position_bp = position[match(ld$snp.id, snp_ids)],
        sample_id = sample_ids, region = region, method = "corr", slide = -1,
        SNPRelate_version = as.character(packageVersion("SNPRelate"))
    )
    saveRDS(ld_result, file.path(output_dir, paste0(stem, ".rds")))
    render_ld(ld_result, stem)
    message("Saved ", stem, " (", nrow(ld_matrix), " SNPs).")
  }
  write.csv(regions, file.path(output_dir, "LD_regions.csv"), row.names = FALSE)
  writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
}

main()
}
