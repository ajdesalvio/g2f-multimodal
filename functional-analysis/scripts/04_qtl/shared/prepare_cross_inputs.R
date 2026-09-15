prepare_qtl_crosses <- function(
    analysis,
    phenotype_data,
    phenotype_columns,
    phenotype_suffix,
    json_suffix = ".cross.setup.json") {
  paths <- qtl_project_paths(analysis, create = TRUE)

  genotype_file <- file.path(
    paths$reference_dir,
    "cm.mbp.ss.100k.sites.SSpopulation_geno.csv"
  )
  template_file <- file.path(paths$reference_dir, "cm.mbp.ss.100k.sites.json")
  overlap_file <- file.path(paths$data_dir, "Pedigree_Overlap_Genomic_Phenomic.csv")
  invisible(lapply(
    c(genotype_file, template_file, overlap_file),
    qtl_require_file
  ))

  required_columns <- c("Pedigree", "Env", phenotype_columns)
  missing_columns <- setdiff(required_columns, names(phenotype_data))
  if (length(missing_columns)) {
    stop("Phenotype input is missing: ", paste(missing_columns, collapse = ", "))
  }

  genomic <- data.table::fread(genotype_file, data.table = FALSE)
  overlap <- data.table::fread(overlap_file, data.table = FALSE)$Pedigree
  phenotype_data <- phenotype_data[phenotype_data$Pedigree %in% overlap, , drop = FALSE]

  pedigree_parts <- stringr::str_match(phenotype_data$Pedigree, "^([^/]+)/([^/]+)")
  if (anyNA(pedigree_parts[, 2L]) || anyNA(pedigree_parts[, 3L])) {
    stop("Every retained Pedigree must have the form INBRED/TESTER.")
  }
  phenotype_data$Inbred <- pedigree_parts[, 2L]
  phenotype_data$Tester <- pedigree_parts[, 3L]
  phenotype_data <- phenotype_data[startsWith(phenotype_data$Inbred, "W10004"), , drop = FALSE]
  phenotype_data$Env.Tester <- paste(phenotype_data$Env, phenotype_data$Tester, sep = ".")

  env_testers <- sort(unique(phenotype_data$Env.Tester))
  template <- jsonlite::read_json(template_file, simplifyVector = FALSE)

  for (env_tester in env_testers) {
    phenotype_subset <- phenotype_data[
      phenotype_data$Env.Tester == env_tester,
      c("Inbred", phenotype_columns),
      drop = FALSE
    ]
    if (anyDuplicated(phenotype_subset$Inbred)) {
      stop("Duplicate inbred rows in ", env_tester, ".")
    }
    genotype_subset <- genomic[genomic$lines %in% phenotype_subset$Inbred, , drop = FALSE]
    missing_inbreds <- setdiff(phenotype_subset$Inbred, genotype_subset$lines)
    if (length(missing_inbreds)) {
      stop(
        "Genotypes missing for ", env_tester, ": ",
        paste(missing_inbreds, collapse = ", ")
      )
    }
    genotype_subset <- genotype_subset[
      match(phenotype_subset$Inbred, genotype_subset$lines),
      ,
      drop = FALSE
    ]

    genotype_name <- paste0(env_tester, ".subset.genotype.data.csv")
    phenotype_name <- paste0(env_tester, phenotype_suffix)
    data.table::fwrite(genotype_subset, file.path(paths$input_dir, genotype_name))
    data.table::fwrite(phenotype_subset, file.path(paths$input_dir, phenotype_name))

    cross_json <- template
    cross_json$geno <- genotype_name
    cross_json$pheno <- phenotype_name
    jsonlite::write_json(
      cross_json,
      file.path(paths$input_dir, paste0(env_tester, json_suffix)),
      auto_unbox = TRUE,
      pretty = TRUE
    )
  }

  writeLines(env_testers, file.path(paths$input_dir, "env_testers.txt"))
  message("Prepared ", length(env_testers), " QTL crosses in ", paths$input_dir)
  invisible(env_testers)
}
