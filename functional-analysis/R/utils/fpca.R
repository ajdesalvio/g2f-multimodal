g2f_fit_sparse_fpca <- function(
    data,
    id_col,
    time_col,
    value_col,
    max_components = 4L,
    n_reg_grid = 100L,
    plot = FALSE) {
  required <- c(id_col, time_col, value_col)
  missing <- setdiff(required, names(data))
  if (length(missing) > 0L) {
    stop("FPCA input is missing columns: ", paste(missing, collapse = ", "))
  }

  input <- data[, required, drop = FALSE]
  names(input) <- c("id", "time", "value")
  input$time <- as.numeric(input$time)
  input$value <- as.numeric(input$value)
  input$value[!is.finite(input$value)] <- NA_real_
  input <- input[!is.na(input$id) & is.finite(input$time), , drop = FALSE]

  # A time point must be unique within an ID for pivot_wider(). Historical
  # station and AGDD workflows used the mean when duplicate times occurred.
  input <- input |>
    dplyr::group_by(.data$id, .data$time) |>
    dplyr::summarise(
      value = if (all(is.na(.data$value))) NA_real_ else mean(.data$value, na.rm = TRUE),
      .groups = "drop"
    )

  observed_ids <- sort(unique(input$id))
  observed_times <- sort(unique(input$time))
  observation_grid <- tidyr::expand_grid(
    id = observed_ids,
    time = observed_times
  ) |>
    dplyr::left_join(input, by = c("id", "time"))

  wide <- observation_grid |>
    tidyr::pivot_wider(names_from = "time", values_from = "value") |>
    as.data.frame()

  time_names <- names(wide)[-1L]
  time_values <- as.numeric(time_names)
  if (anyNA(time_values)) stop("FPCA time values could not be converted to numeric.")
  order_index <- order(time_values)
  time_names <- time_names[order_index]
  time_values <- time_values[order_index]
  wide <- wide[, c("id", time_names), drop = FALSE]
  rownames(wide) <- wide$id

  fpca_input <- fdapace::MakeFPCAInputs(
    IDs = rep(wide$id, each = length(time_values)),
    tVec = rep(time_values, times = nrow(wide)),
    yVec = as.vector(t(as.matrix(wide[, -1L, drop = FALSE])))
  )

  fit <- fdapace::FPCA(
    fpca_input$Ly,
    fpca_input$Lt,
    list(
      dataType = "Sparse",
      plot = plot,
      methodMuCovEst = "smooth",
      methodBwCov = "GCV",
      methodBwMu = "GCV",
      nRegGrid = as.integer(n_reg_grid)
    )
  )

  available <- ncol(fit$xiEst)
  retained <- min(as.integer(max_components), available)
  if (retained < 1L) stop("FPCA returned no component scores for ", value_col)

  scores <- as.data.frame(fit$xiEst[, seq_len(retained), drop = FALSE])
  names(scores) <- paste0("FPC", seq_len(retained))
  scores[[id_col]] <- wide$id

  cumulative_fve <- fit$cumFVE[seq_len(retained)]
  component_fve <- c(cumulative_fve[[1L]], diff(cumulative_fve))
  for (i in seq_len(retained)) {
    scores[[paste0("FPC", i, "_FVE")]] <- component_fve[[i]]
  }

  predictions <- predict(fit, newLy = fpca_input$Ly, newLt = fpca_input$Lt)
  predicted_curves <- as.data.frame(predictions$predCurves)
  predicted_grid <- as.numeric(predictions$predGrid)
  names(predicted_curves) <- format(predicted_grid, scientific = FALSE, trim = TRUE)
  predicted_curves[[id_col]] <- wide$id

  predicted_curves_tall <- tidyr::pivot_longer(
    predicted_curves,
    cols = -dplyr::all_of(id_col),
    names_to = time_col,
    values_to = "Value"
  )
  predicted_curves_tall[[time_col]] <- as.numeric(predicted_curves_tall[[time_col]])

  list(
    model = fit,
    scores = scores,
    predicted_curves_wide = predicted_curves,
    predicted_curves_tall = predicted_curves_tall,
    mean_curve = data.frame(
      Time = predicted_grid,
      Mean_Predicted_Value = colMeans(predictions$predCurves)
    ),
    components_retained = retained
  )
}
