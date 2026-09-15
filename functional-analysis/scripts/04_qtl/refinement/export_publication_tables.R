# Export publication subsets without changing the full QTL or annotation tables.
# Source rules: Inspect_MaizeMine_Results_V2.R and author-confirmed Sept 14 inventory.
project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
paths <- g2f_paths(require_data = FALSE)
output_dir <- g2f_results_dir(paths, "qtl", "publication")

read_retained <- function(filename) {
  candidates <- c(file.path(project_dir, "results", "qtl", filename),
                  file.path(project_dir, "results", "qtl", "annotation_snapshots", filename),
                  file.path(paths$data_dir, filename))
  found <- candidates[file.exists(candidates)]
  if (!length(found)) stop("Required retained input missing: ", filename)
  read.csv(found[[1L]], stringsAsFactors = FALSE, check.names = FALSE)
}

peaks <- read_retained("QTL_Results_Combined_Revised_Intervals.csv")
fpc <- peaks[peaks$FPCA_type %in% c("DAP", "AGDD") &
               grepl("^FPC[0-9]+_NGRDI$", peaks$lodcolumn), , drop = FALSE]
stopifnot(nrow(fpc) == 167L, sum(fpc$FPCA_type == "DAP") == 71L,
          sum(fpc$FPCA_type == "AGDD") == 96L)
write.csv(fpc, file.path(output_dir, "QTL_Results_NGRDI_FPC_167_Peaks.csv"),
          row.names = FALSE, na = "")

for (chromosome in c("chr7", "chr3")) {
  source_file <- if (chromosome == "chr7") {
    "QTL_chr7_125_140Mb_genes_GO_collapsed.csv"
  } else {
    "QTL_chr3_113_185Mb_genes_GO_collapsed.csv"
  }
  genes <- read_retained(source_file)
  # Preserve the original, case-sensitive LOC exclusion and symbol grouping.
  keep <- !is.na(genes$symbol) & nzchar(genes$symbol) & !grepl("LOC", genes$symbol)
  genes <- genes[keep, , drop = FALSE]
  symbols <- sort(unique(genes$symbol))
  selected <- do.call(rbind, lapply(symbols, function(symbol) {
    rows <- genes[genes$symbol == symbol, , drop = FALSE]
    data.frame(symbol = symbol,
               primaryIdentifier = paste(unique(rows$primaryIdentifier), collapse = ", "),
               description = paste(unique(rows$description), collapse = ", "),
               stringsAsFactors = FALSE)
  }))
  write.csv(selected, file.path(output_dir, paste0("QTL_", chromosome, "_Named_Gene_Symbols.csv")),
            row.names = FALSE, na = "")
  message(chromosome, ": ", nrow(genes), " identifiers grouped into ", length(symbols), " symbols")
}
message("Exported 167 NGRDI FPC peaks (71 DAP; 96 AGDD) to ", output_dir)
