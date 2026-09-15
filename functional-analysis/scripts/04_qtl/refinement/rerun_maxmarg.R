library(data.table)
library(qtl2)

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
source(file.path(project_dir, "scripts", "04_qtl", "shared", "qtl_paths.R"))

# Read the version-controlled manuscript follow-up rather than duplicating it here.
analysis_config <- g2f_analysis_config()
follow_up <- analysis_config$qtl$genotype_effect_follow_up
if (!is.list(follow_up)) {
  stop(
    "config/analysis.yml must define qtl.genotype_effect_follow_up with ",
    "chromosome and trait values."
  )
}

target_chromosome <- follow_up$chromosome
if (
  length(target_chromosome) != 1L ||
    !is.numeric(target_chromosome) ||
    is.na(target_chromosome) ||
    !is.finite(target_chromosome) ||
    target_chromosome < 1 ||
    target_chromosome != floor(target_chromosome)
) {
  stop("qtl.genotype_effect_follow_up.chromosome must be one positive integer.")
}
target_chromosome <- as.integer(target_chromosome)

target_trait <- follow_up$trait
if (
  length(target_trait) != 1L ||
    !is.character(target_trait) ||
    is.na(target_trait) ||
    !nzchar(trimws(target_trait))
) {
  stop("qtl.genotype_effect_follow_up.trait must be one non-empty string.")
}
target_trait <- trimws(target_trait)

paths <- qtl_project_paths("dap", create = TRUE)
default_interval_file <- file.path(
  paths$refinement_dir,
  "QTL_Results_Combined_Revised_Intervals.csv"
)
interval_file <- Sys.getenv("G2F_QTL_INTERVAL_FILE", unset = default_interval_file)
qtl_require_file(interval_file, "Revised QTL intervals")

intervals <- fread(interval_file, data.table = FALSE)
targets <- intervals[
  intervals$chr == target_chromosome & intervals$lodcolumn == target_trait,
  ,
  drop = FALSE
]
if (!nrow(targets)) {
  stop(
    "No chromosome ", target_chromosome, " / ", target_trait,
    " loci were found in ", interval_file
  )
}
targets$analysis <- ifelse(
  targets$FPCA_type == "DAP",
  "dap",
  ifelse(targets$FPCA_type == "AGDD", "agdd", NA_character_)
)
if (anyNA(targets$analysis)) {
  stop("The confirmed follow-up only supports DAP and AGDD loci.")
}

effect_tables <- lapply(seq_len(nrow(targets)), function(index) {
  locus <- targets[index, , drop = FALSE]
  analysis_paths <- qtl_project_paths(locus$analysis, create = TRUE)
  result_file <- qtl_result_rds(analysis_paths$output_dir, locus$Env.Tester)
  message("Calculating genotype effects from ", result_file)
  container <- readRDS(result_file)
  result <- container[[locus$Env.Tester]]
  if (is.null(result)) stop("Missing key ", locus$Env.Tester, " in ", result_file)
  if (!target_trait %in% colnames(result$pheno.loop.final)) {
    stop("Missing ", target_trait, " phenotype in ", result_file)
  }

  genotype <- qtl2::maxmarg(
    result$probs.final,
    result$pmap,
    chr = target_chromosome,
    pos = locus$pos,
    return_char = TRUE,
    cores = 0
  )
  common_ids <- intersect(names(genotype), rownames(result$pheno.loop.final))
  if (!length(common_ids)) stop("No common genotype and phenotype IDs in ", result_file)

  data.frame(
    id = common_ids,
    chr = target_chromosome,
    pos = locus$pos,
    Trait = target_trait,
    Env.Tester = locus$Env.Tester,
    FPCA_Type = ifelse(locus$analysis == "dap", "Standard_FPCA", "AGDD"),
    genotype = unname(genotype[common_ids]),
    Phenotype = target_trait,
    Value = result$pheno.loop.final[common_ids, target_trait],
    stringsAsFactors = FALSE
  )
})

effects <- rbindlist(effect_tables, fill = TRUE)
parent_names <- c(
  AA = "B73",
  BB = "B84",
  CC = "LH145",
  DD = "NKH8431",
  EE = "PHB47",
  FF = "PHJ40"
)
effects$DH6.Parent.Names <- unname(parent_names[effects$genotype])

trait_file_label <- gsub("[^A-Za-z0-9.-]+", "_", sub("_.*$", "", target_trait))
output_file <- file.path(
  paths$refinement_dir,
  sprintf(
    "QTL_Results_Maxmarg_Chr%d_%s.csv",
    target_chromosome,
    trait_file_label
  )
)
fwrite(effects, output_file)
message("Saved ", output_file)
