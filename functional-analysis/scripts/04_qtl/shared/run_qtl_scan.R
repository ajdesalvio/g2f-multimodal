run_qtl_scan <- function(analysis, env_tester, json_suffix = ".cross.setup.json") {
  paths <- qtl_project_paths(analysis, create = TRUE)
  cross_file <- file.path(paths$input_dir, paste0(env_tester, json_suffix))
  qtl_require_file(cross_file, "QTL cross JSON")

  cross <- qtl2::read_cross2(cross_file)
  if (!qtl2::check_cross2(cross)) stop("qtl2::check_cross2 failed for ", env_tester)

  duplicate_summary <- as.data.frame(summary(qtl2::compare_geno(cross)))
  duplicate_ids <- unique(unlist(
    duplicate_summary[duplicate_summary$prop_match > 0.95, c("ind1", "ind2")],
    use.names = FALSE
  ))
  retain_after_duplicates <- if (length(duplicate_ids)) {
    !qtl2::ind_ids(cross) %in% duplicate_ids
  } else {
    rep(TRUE, length(qtl2::ind_ids(cross)))
  }
  cross <- cross[retain_after_duplicates, ]

  phenotype <- cross$pheno
  genetic_map <- cross$gmap
  physical_map <- cross$pmap
  message("Calculating genotype probabilities for ", env_tester)
  probabilities <- qtl2::calc_genoprob(cross, error_prob = 0.01, cores = 0)
  maximum_marginal <- qtl2::maxmarg(probabilities, cores = 0)
  crossover_locations <- qtl2::locate_xo(maximum_marginal, map = physical_map, cores = 0)

  genotype_ids <- qtl2::ind_ids(cross)
  crossover_counts <- vapply(genotype_ids, function(genotype_id) {
    sum(vapply(crossover_locations, function(chromosome) {
      value <- chromosome[[genotype_id]]
      if (is.null(value)) 0L else length(value)
    }, integer(1)))
  }, integer(1))
  suffix <- suppressWarnings(as.integer(sub("^.*_", "", genotype_ids)))
  if (anyNA(suffix)) stop("Could not parse numeric W10004 genotype suffixes.")
  subset_name <- ifelse(suffix < 500L, "A", "B")
  crossover_flag <- (subset_name == "A" & crossover_counts > 150L) |
    (subset_name == "B" & crossover_counts > 250L)
  crossover_table <- data.frame(
    Genotype = genotype_ids,
    TotalXO = crossover_counts,
    Subset = subset_name,
    Flag = crossover_flag
  )
  print(table(crossover_table$Subset, crossover_table$Flag))

  retained_ids <- crossover_table$Genotype[!crossover_table$Flag]
  phenotype_final <- phenotype[retained_ids, , drop = FALSE]
  probabilities_final <- probabilities[retained_ids, ]
  common_ids <- qtl2::get_common_ids(phenotype_final, probabilities_final)
  if (!identical(common_ids, retained_ids)) {
    stop("Phenotype and genotype-probability identifiers are not aligned.")
  }

  message("Calculating kinship and genome scan for ", env_tester)
  kinship <- qtl2::calc_kinship(probabilities_final, type = "loco", cores = 0)
  scan_result <- qtl2::scan1(
    genoprobs = probabilities_final,
    pheno = phenotype_final,
    kinship = kinship,
    cores = 0
  )

  seed <- as.integer(Sys.getenv("G2F_QTL_SEED", unset = "20250707"))
  if (is.na(seed)) stop("G2F_QTL_SEED must be an integer.")
  set.seed(seed)
  message("Running 1,000 permutations for ", env_tester, " with seed ", seed)
  permutation_result <- qtl2::scan1perm(
    genoprobs = probabilities_final,
    pheno = phenotype_final,
    n_perm = 1000,
    cores = 0
  )
  lod_peaks <- qtl2::find_peaks(
    scan1_output = scan_result,
    map = physical_map,
    threshold = summary(permutation_result),
    drop = 5
  )

  qtl_output <- list()
  qtl_output[[env_tester]] <- list(
    sample.duplicates = duplicate_ids,
    xo.locations = crossover_locations,
    xo.dataframe = crossover_table,
    max.marg.prob = maximum_marginal,
    genotypes.retain.final = retained_ids,
    pheno.loop.final = phenotype_final,
    probs.final = probabilities_final,
    gmap = genetic_map,
    pmap = physical_map,
    kinship = kinship,
    scan.result = scan_result,
    permutation.result = permutation_result,
    LOD.peaks = lod_peaks,
    analysis = analysis,
    permutation.seed = seed
  )
  output_file <- qtl_result_rds(paths$output_dir, env_tester, must_work = FALSE)
  saveRDS(qtl_output, output_file)
  message("Saved ", output_file)
  invisible(output_file)
}
