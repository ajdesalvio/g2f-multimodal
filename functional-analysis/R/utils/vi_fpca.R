g2f_run_vi_fpca <- function(
    vi_data,
    time_col,
    paths,
    output_subdir,
    score_filename,
    curve_filename) {
  required <- c(
    "Pedigree", "Year", "Env", "Vegetation.Index", "VI.BLUE", time_col
  )
  missing <- setdiff(required, names(vi_data))
  if (length(missing) > 0L) {
    stop("VI FPCA input is missing: ", paste(missing, collapse = ", "))
  }

  output_dir <- g2f_results_dir(paths, "02_fpca_weather", output_subdir)
  model_dir <- file.path(output_dir, "models")
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

  vi_data <- vi_data |>
    dplyr::mutate(Pedigree.Env = paste(.data$Pedigree, .data$Env, sep = "."))
  metadata <- vi_data |>
    dplyr::distinct(
      .data$Pedigree.Env,
      .data$Pedigree,
      .data$Year,
      .data$Env
    )

  score_list <- list()
  curve_list <- list()
  vegetation_indices <- sort(unique(vi_data$Vegetation.Index))

  for (vegetation_index in vegetation_indices) {
    message("Fitting ", time_col, " VI FPCA: ", vegetation_index)
    current <- dplyr::filter(
      vi_data,
      .data$Vegetation.Index == vegetation_index,
      is.finite(.data[[time_col]])
    )

    fpca <- g2f_fit_sparse_fpca(
      current,
      id_col = "Pedigree.Env",
      time_col = time_col,
      value_col = "VI.BLUE",
      max_components = 20L,
      n_reg_grid = 100L,
      plot = FALSE
    )

    scores <- fpca$scores |>
      dplyr::left_join(metadata, by = "Pedigree.Env") |>
      dplyr::mutate(
        Env.VI = paste(.data$Env, vegetation_index, sep = "."),
        Vegetation.Index = vegetation_index
      ) |>
      dplyr::relocate(
        .data$Pedigree.Env,
        .data$Pedigree,
        .data$Year,
        .data$Env,
        .data$Env.VI,
        .data$Vegetation.Index
      )

    curves <- fpca$predicted_curves_tall |>
      dplyr::rename(Predicted.Value = "Value") |>
      dplyr::left_join(metadata, by = "Pedigree.Env") |>
      dplyr::mutate(
        Env.VI = paste(.data$Env, vegetation_index, sep = "."),
        Vegetation.Index = vegetation_index
      ) |>
      dplyr::relocate(
        .data$Pedigree.Env,
        .data$Pedigree,
        .data$Year,
        .data$Env,
        .data$Env.VI,
        .data$Vegetation.Index
      )

    score_list[[vegetation_index]] <- scores
    curve_list[[vegetation_index]] <- curves
    model_name <- gsub("[^A-Za-z0-9_.-]", "_", vegetation_index)
    saveRDS(
      fpca$model,
      file.path(model_dir, paste0(time_col, "_VI_FPCA_", model_name, ".rds"))
    )
  }

  scores <- dplyr::bind_rows(score_list)
  curves <- dplyr::bind_rows(curve_list)
  data.table::fwrite(scores, file.path(output_dir, score_filename))
  data.table::fwrite(curves, file.path(output_dir, curve_filename))

  invisible(list(scores = scores, curves = curves))
}
