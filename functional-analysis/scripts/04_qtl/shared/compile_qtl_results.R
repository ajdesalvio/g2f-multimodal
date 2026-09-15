qtl_read_env_testers <- function(data_dir) {
  path <- file.path(data_dir, "Env.Testers.for.QTL.csv")
  qtl_require_file(path, "Environment-tester list")
  values <- data.table::fread(path, data.table = FALSE)$Env.Tester
  sort(unique(values[!is.na(values) & nzchar(values)]))
}

qtl_allow_incomplete <- function() {
  tolower(Sys.getenv("G2F_QTL_ALLOW_INCOMPLETE", unset = "false")) %in%
    c("true", "t", "1", "yes", "y")
}

qtl_combine_files <- function(files, description) {
  existing <- files[file.exists(files)]
  if (!length(existing)) stop("No ", description, " files were found.")
  data.table::rbindlist(lapply(existing, data.table::fread), fill = TRUE)
}

compile_qtl_results <- function(analyses, output_stem) {
  analyses <- match.arg(
    analyses,
    choices = c("dap", "agdd", "flowering_yield"),
    several.ok = TRUE
  )
  paths_by_analysis <- lapply(analyses, qtl_project_paths, create = TRUE)
  names(paths_by_analysis) <- analyses
  env_testers <- qtl_read_env_testers(paths_by_analysis[[1L]]$data_dir)

  permutation_tables <- list()
  missing_rds <- character()
  missing_downstream <- character()
  lod_files <- character()
  blup_files <- character()
  for (analysis in analyses) {
    paths <- paths_by_analysis[[analysis]]
    prefix <- if (analysis == "agdd") ".AGDD" else ""
    for (env_tester in env_testers) {
      rds_file <- qtl_result_rds(paths$output_dir, env_tester, must_work = FALSE)
      if (!file.exists(rds_file)) {
        legacy_file <- file.path(paths$output_dir, paste0(env_tester, "QTL.Outputs.rds"))
        if (file.exists(legacy_file)) rds_file <- legacy_file
      }
      if (!file.exists(rds_file)) {
        missing_rds <- c(missing_rds, paste(analysis, env_tester, sep = ":"))
      } else {
        container <- readRDS(rds_file)
        result <- container[[env_tester]]
        if (is.null(result)) stop("Missing key ", env_tester, " in ", rds_file)
        permutation <- as.data.frame(summary(result$permutation.result))
        phenotype_columns <- colnames(result$pheno.loop.final)
        if (analysis != "flowering_yield") {
          permutation$FPCA_Type <- c(
            dap = "Standard_FPCA",
            agdd = "AGDD_FPCA"
          )[[analysis]]
        }
        permutation$Env.Tester <- env_tester
        permutation <- tidyr::pivot_longer(
          permutation,
          cols = dplyr::all_of(phenotype_columns),
          names_to = "Phenotype",
          values_to = "Threshold"
        )
        parts <- qtl_split_env_tester(env_tester)
        permutation$Env <- parts$Env
        permutation$Tester <- parts$Tester
        permutation_tables[[paste(analysis, env_tester, sep = ":")]] <- permutation
      }

      stub <- file.path(paths$output_dir, paste0(env_tester, prefix))
      lod_file <- paste0(stub, ".QTL.LOD.long.csv")
      blup_file <- paste0(stub, ".QTL.BLUPs.long.csv")
      lod_files <- c(lod_files, lod_file)
      if (!file.exists(lod_file)) {
        missing_downstream <- c(missing_downstream, lod_file)
      }
      if (exists("result", inherits = FALSE) && nrow(result$LOD.peaks)) {
        blup_files <- c(blup_files, blup_file)
        if (!file.exists(blup_file)) {
          missing_downstream <- c(missing_downstream, blup_file)
        }
      }
      if (exists("result", inherits = FALSE)) rm(result)
    }
  }

  if (length(missing_rds) && !qtl_allow_incomplete()) {
    stop(
      "Missing ", length(missing_rds), " expected QTL result files. ",
      "Set G2F_QTL_ALLOW_INCOMPLETE=true only to compile a documented partial run. ",
      "First missing result(s): ", paste(head(missing_rds, 5L), collapse = ", ")
    )
  }
  if (length(missing_downstream) && !qtl_allow_incomplete()) {
    stop(
      "Missing ", length(missing_downstream), " expected downstream QTL table(s). ",
      "Run hprc_downstream.R for every completed scan. First missing file(s): ",
      paste(head(missing_downstream, 5L), collapse = ", ")
    )
  }
  if (!length(permutation_tables)) stop("No QTL result RDS files were found.")

  compiled_dir <- paths_by_analysis[[1L]]$compiled_dir
  dir.create(compiled_dir, recursive = TRUE, showWarnings = FALSE)
  permutation_output <- data.table::rbindlist(permutation_tables, fill = TRUE)
  lod_output <- qtl_combine_files(lod_files, "LOD")
  blup_output <- qtl_combine_files(blup_files, "BLUP")

  data.table::fwrite(
    permutation_output,
    file.path(compiled_dir, paste0(output_stem, "_Permutation_Tests.csv"))
  )
  data.table::fwrite(
    lod_output,
    file.path(compiled_dir, paste0(output_stem, "_LOD.csv"))
  )
  data.table::fwrite(
    blup_output,
    file.path(compiled_dir, paste0(output_stem, "_BLUPs.csv"))
  )
  message("Compiled QTL results in ", compiled_dir)
  invisible(compiled_dir)
}
