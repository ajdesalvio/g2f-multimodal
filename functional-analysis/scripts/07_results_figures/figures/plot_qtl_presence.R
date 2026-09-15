# Reproduce Figure 5 panels A-C (QTL presence and chromosome 3/7 intervals).
# Adapted from QTL_with_CIs_V4.R. LD panels and final layout are documented in
# results/qtl/ld/README.md; historical manual assembly is preserved unchanged.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(cowplot)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
paths <- g2f_paths(require_data = FALSE)
output_dir <- g2f_results_dir(paths, "07_results_figures", "figures")

filename <- "QTL_Results_Combined_Revised_Intervals.csv"
candidates <- c(
  file.path(paths$results_dir, "qtl", filename),
  file.path(paths$project_dir, "results", "qtl", filename),
  g2f_data_file(paths, filename)
)
input_file <- candidates[file.exists(candidates)][1L]
if (is.na(input_file)) stop("Missing ", filename)
peaks <- read.csv(input_file, check.names = FALSE)
required_columns <- c("lodcolumn", "chr", "pos", "ci_lo", "ci_hi", "Env.Tester", "FPCA_type")
if (!all(required_columns %in% names(peaks))) stop("Unexpected QTL interval-table columns.")
peaks <- peaks |>
  rename(Phenotype = lodcolumn, FPCA_Type = FPCA_type) |>
  filter(!Phenotype %in% c("ASI", "DTA.BLUE", "DTS.BLUE", "Yield.t.ha.BLUE"))
if (nrow(peaks) != 167L) stop("Expected the 167 FPC QTL in the published figure.")

# The original whole-genome points used unseeded jitter. Set its seed for a
# stable rerun; positions and interval estimates are unchanged.
whole_genome <- ggplot(peaks, aes(pos / 1e6, Env.Tester, colour = Phenotype, shape = FPCA_Type)) +
  geom_point(size = 3, position = position_jitter(seed = 42)) +
  facet_grid(~ chr, scales = "free_x") +
  labs(x = "Genomic Position (Mb)", y = "Environment", shape = "FPCA Type") +
  theme_bw() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5))
legend <- cowplot::get_legend(whole_genome)
whole_genome <- whole_genome + theme(legend.position = "none")

interval_panel <- function(target_chr, limits) {
  ggplot(filter(peaks, chr == target_chr),
         aes(pos / 1e6, Env.Tester, colour = Phenotype, shape = FPCA_Type)) +
    geom_pointrange(aes(xmin = ci_lo / 1e6, xmax = ci_hi / 1e6),
                    position = position_jitter(seed = 42),
                    linetype = "solid", linewidth = 0.75) +
    facet_grid(~ chr, scales = "free_x") +
    labs(x = "Genomic Position (Mb)", y = "Environment", shape = "FPCA Type") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5),
          axis.text.y = element_text(angle = 0), legend.position = "none") +
    coord_cartesian(xlim = limits)
}

chr3 <- interval_panel(3, c(139, 186))
chr7 <- interval_panel(7, c(125, 140))
bottom <- cowplot::plot_grid(
  chr3, chr7 + labs(y = NULL), legend, labels = c("B)", "C)", ""),
  nrow = 1, rel_widths = c(0.457, 0.443, 0.1)
)
panels <- cowplot::plot_grid(
  whole_genome, bottom, labels = c("A)"), nrow = 2, rel_heights = c(0.6, 0.4)
)
ggsave(file.path(output_dir, "Figure_05_QTL_presence_panels_A_C.pdf"),
       plot = panels, device = cairo_pdf, width = 11.2, height = 8.5)
message("Saved Figure 5 QTL panels from ", nrow(peaks), " FPC QTL.")
