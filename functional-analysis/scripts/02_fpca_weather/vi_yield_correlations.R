# Descriptive VI FPC-yield correlations, adapted from Unified_FPCA_Yield_Cor_V6.R.
# Run from functional-analysis/ after configuring the two local roots.
# Uses existing full-data FPCA scores; does not refit FPCA or prediction models.

suppressPackageStartupMessages(library(data.table))
project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
paths <- g2f_paths()

require_columns <- function(x, columns, label) {
  missing <- setdiff(columns, names(x))
  if (length(missing)) stop(label, " is missing: ", paste(missing, collapse = ", "))
}

input_files <- c(
  cohort = g2f_resolve_input(paths, "G2F.2020.2021.Pedigrees.csv"),
  yield = g2f_resolve_input(paths, "Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv",
                          "01_blue_variance"),
  DAP = g2f_resolve_input(paths,
    "FPC_Scores_BLUEs_Unified_FPCA_Max_FPCs_2020_2021_G2F_VIs.csv",
    "02_fpca_weather/vi_dap"),
  AGDD = g2f_resolve_input(paths,
    "FPC_Scores_BLUEs_AGDD_Unified_FPCA_Max_FPCs_2020_2021_G2F_VIs.csv",
    "02_fpca_weather/vi_agdd")
)
cohort <- fread(input_files[["cohort"]])
require_columns(cohort, c("Pedigree.Env", "Env"), "Cohort")
if (nrow(cohort) != 10109L || anyNA(cohort$Pedigree.Env) ||
    anyDuplicated(cohort$Pedigree.Env) || anyNA(cohort$Env) ||
    uniqueN(cohort$Env) != 19L) {
  stop("Expected 10,109 unique genotype-environment records in 19 environments.")
}
yield <- fread(input_files[["yield"]])
require_columns(yield, c("Pedigree.Env", "Env", "Yield.t.ha.BLUE"), "Yield")
yield <- yield[yield$Pedigree.Env %in% cohort$Pedigree.Env, ]
if (nrow(yield) != nrow(cohort) || anyDuplicated(yield$Pedigree.Env)) {
  stop("Each cohort record must match exactly one yield record.")
}
yield <- yield[match(cohort$Pedigree.Env, yield$Pedigree.Env), ]
if (anyNA(yield$Env) || !all(yield$Env == cohort$Env) ||
    !is.numeric(yield$Yield.t.ha.BLUE) || any(!is.finite(yield$Yield.t.ha.BLUE))) {
  stop("Cohort/yield environments disagree, or yield BLUEs are not finite numbers.")
}

# Preserve V6's cor() missing-value policy: do not silently delete pairs.
# A missing/nonfinite score or zero variance is reported as NA with a reason.
pearson <- function(x, y) {
  complete <- is.finite(x) & is.finite(y)
  status <- if (!all(complete)) "missing_or_nonfinite_values" else if (
    length(x) < 2L) "insufficient_records" else if (
    stats::sd(x) == 0 || stats::sd(y) == 0) "zero_variance" else "ok"
  data.table(N = length(x), N_complete = sum(complete),
             Cor = if (status == "ok") stats::cor(x, y, method = "pearson",
                                                    use = "everything") else NA_real_,
             Status = status)
}

pooled <- list()
within <- list()
retained_fpcs <- c(DAP = 5L, AGDD = 7L)
envs <- sort(unique(cohort$Env), method = "radix")
expected_vis <- NULL
for (domain in names(retained_fpcs)) {
  fpcs <- paste0("FPC", seq_len(retained_fpcs[[domain]]))
  columns <- c("Pedigree.Env", "Env", "Vegetation.Index", fpcs)
  header <- names(fread(input_files[[domain]], nrows = 0L))
  if (!all(columns %in% header)) stop(domain, " score file lacks required columns.")
  scores <- fread(input_files[[domain]], select = columns)
  scores <- scores[scores$Pedigree.Env %in% cohort$Pedigree.Env, ]
  if (anyNA(scores$Vegetation.Index) || any(!nzchar(scores$Vegetation.Index)) ||
      anyDuplicated(scores, by = c("Pedigree.Env", "Vegetation.Index"))) {
    stop(domain, " scores must have one record per genotype-environment and VI.")
  }
  if (!all(vapply(scores[, ..fpcs], is.numeric, logical(1)))) {
    stop(domain, " FPC scores must be numeric.")
  }
  vis <- sort(unique(scores$Vegetation.Index), method = "radix")
  if (length(vis) != 37L || !"NGRDI" %in% vis) {
    stop(domain, " must contain the 37 study VIs, including NGRDI.")
  }
  if (is.null(expected_vis)) expected_vis <- vis
  if (!identical(vis, expected_vis)) stop("DAP and AGDD VI inventories disagree.")
  for (vi in vis) {
    vi_scores <- scores[scores$Vegetation.Index == vi, ]
    matched <- match(cohort$Pedigree.Env, vi_scores$Pedigree.Env)
    # Align each VI to the complete cohort; absent scores remain missing.
    present <- !is.na(matched)
    if (anyNA(vi_scores$Env[matched[present]]) ||
        any(vi_scores$Env[matched[present]] != cohort$Env[present])) {
      stop(domain, "/", vi, " score environments disagree with the cohort.")
    }
    for (fpc in fpcs) {
      x <- vi_scores[[fpc]][matched]
      identity <- data.table(Domain = domain, Vegetation.Index = vi, FPC = fpc,
                             FPC_ID = paste(fpc, vi, sep = "_"))
      pooled[[length(pooled) + 1L]] <- cbind(
        identity, pearson(x, yield$Yield.t.ha.BLUE))
      for (env in envs) {
        keep <- cohort$Env == env
        within[[length(within) + 1L]] <- cbind(
          identity, data.table(Env = env), pearson(x[keep], yield$Yield.t.ha.BLUE[keep]))
      }
    }
  }
  message(domain, ": ", length(vis), " VIs x ", length(fpcs),
          " FPCs; ", nrow(cohort), " records; ", length(envs), " environments.")
}
pooled <- rbindlist(pooled)
within <- rbindlist(within)
if (any(pooled[Vegetation.Index == "NGRDI", Status] != "ok") ||
    any(within[Vegetation.Index == "NGRDI", Status] != "ok")) {
  stop("NGRDI correlations are incomplete; inspect the score inputs before publication.")
}
output_dir <- g2f_results_dir(paths, "02_fpca_weather", "vi_yield_correlations")
fwrite(pooled, file.path(output_dir, "VI_FPC_Yield_Correlations_Pooled.csv"),
       na = "NA")
fwrite(within, file.path(output_dir, "VI_FPC_Yield_Correlations_Within_Environment.csv"),
       na = "NA")
input_manifest <- data.table(
  Role = names(input_files), Filename = basename(input_files),
  Size_bytes = as.numeric(file.info(input_files)$size),
  MD5 = unname(tools::md5sum(input_files))
)
fwrite(input_manifest, file.path(output_dir, "VI_FPC_Yield_Correlation_Inputs.csv"))
writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
message("Saved correlation tables to: ", output_dir)
print(pooled[Vegetation.Index == "NGRDI", .(Domain, FPC, N, Cor)])
print(within[Vegetation.Index == "NGRDI",
             .(Minimum_r = min(Cor), Maximum_r = max(Cor)), by = Domain])
