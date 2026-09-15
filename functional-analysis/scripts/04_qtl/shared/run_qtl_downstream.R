qtl_add_env_tester_columns <- function(data, env_tester) {
  parts <- qtl_split_env_tester(env_tester)
  data$Env.Tester <- env_tester
  data$Env <- parts$Env
  data$Tester <- parts$Tester
  data
}

qtl_add_marker_columns <- function(data) {
  marker <- rownames(data)
  parsed <- stringr::str_match(marker, "^S?([^_]+)_([0-9.]+)$")
  if (anyNA(parsed[, 2L]) || anyNA(parsed[, 3L])) {
    stop("Unexpected marker name(s): ", paste(marker[is.na(parsed[, 2L])], collapse = ", "))
  }
  data$chr <- parsed[, 2L]
  data$pos <- as.numeric(parsed[, 3L])
  data
}

run_qtl_downstream <- function(analysis, env_tester, output_prefix = "") {
  paths <- qtl_project_paths(analysis, create = TRUE)
  result_file <- qtl_result_rds(paths$output_dir, env_tester)
  result_container <- readRDS(result_file)
  if (!env_tester %in% names(result_container)) {
    stop("Result object does not contain key ", env_tester, ".")
  }
  result <- result_container[[env_tester]]
  analysis_label <- c(
    dap = "DAP",
    agdd = "AGDD",
    flowering_yield = "Flowering.Yield"
  )[[analysis]]
  include_fpca_type <- analysis != "flowering_yield"

  lod_wide <- qtl_add_marker_columns(as.data.frame(result$scan.result))
  lod_wide <- qtl_add_env_tester_columns(lod_wide, env_tester)
  phenotype_columns <- colnames(result$pheno.loop.final)
  if (include_fpca_type) lod_wide$FPCA_Type <- analysis_label
  lod_wide <- lod_wide[, c(
    phenotype_columns,
    "Env.Tester",
    if (include_fpca_type) "FPCA_Type",
    "chr", "pos", "Env", "Tester"
  ), drop = FALSE]
  lod_long <- tidyr::pivot_longer(
    lod_wide,
    cols = dplyr::all_of(phenotype_columns),
    names_to = "Phenotype",
    values_to = "LOD"
  )

  blup_tables <- lapply(seq_len(nrow(result$LOD.peaks)), function(index) {
    chromosome <- as.character(result$LOD.peaks$chr[[index]])
    trait <- result$LOD.peaks$lodcolumn[[index]]
    message("Running scan1blup for ", env_tester, ", chromosome ", chromosome, ", ", trait)
    coefficients <- as.data.frame(qtl2::scan1blup(
      genoprobs = result$probs.final[, chromosome],
      pheno = result$pheno.loop.final[, trait, drop = FALSE],
      kinship = result$kinship[[chromosome]],
      cores = 0
    ))
    coefficient_columns <- names(coefficients)
    coefficients <- qtl_add_marker_columns(coefficients)
    coefficients <- qtl_add_env_tester_columns(coefficients, env_tester)
    if (include_fpca_type) coefficients$FPCA_Type <- analysis_label
    coefficients$Trait <- trait
    coefficients <- coefficients[, c(
      coefficient_columns,
      "Env.Tester",
      if (include_fpca_type) "FPCA_Type",
      "Trait", "chr", "pos", "Env", "Tester"
    ), drop = FALSE]
    list(
      wide = coefficients,
      long = tidyr::pivot_longer(
        coefficients,
        cols = dplyr::all_of(coefficient_columns),
        names_to = "DH6.Parent",
        values_to = "BLUP"
      )
    )
  })

  file_stub <- file.path(paths$output_dir, paste0(env_tester, output_prefix))
  data.table::fwrite(lod_wide, paste0(file_stub, ".QTL.LOD.wide.csv"))
  data.table::fwrite(lod_long, paste0(file_stub, ".QTL.LOD.long.csv"))
  if (length(blup_tables)) {
    blup_wide <- data.table::rbindlist(lapply(blup_tables, `[[`, "wide"), fill = TRUE)
    blup_long <- data.table::rbindlist(lapply(blup_tables, `[[`, "long"), fill = TRUE)
    data.table::fwrite(blup_wide, paste0(file_stub, ".QTL.BLUPs.wide.csv"))
    data.table::fwrite(blup_long, paste0(file_stub, ".QTL.BLUPs.long.csv"))
  } else {
    message("No significant peaks for ", env_tester, "; no BLUP tables were written.")
  }
  message("Saved downstream QTL tables for ", env_tester)
  invisible(file_stub)
}
