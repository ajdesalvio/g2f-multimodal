# Reproduce the weather-FPCA trajectory and yield-correlation figure.

suppressPackageStartupMessages({
  library(cowplot)
  library(dplyr)
  library(ggplot2)
  library(ggh4x)
  library(tidyr)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))

paths <- g2f_paths()
output_dir <- g2f_results_dir(paths, "07_results_figures", "figures")

correlation_candidates <- c(
  file.path(paths$results_dir, "02_fpca_weather/correlations/Yield_FPC_Correlations_All_DAP.csv"),
  file.path(paths$project_dir, "results/figures/source_data/Yield_FPC_Correlations_All_DAP.csv")
)
correlation_file <- correlation_candidates[file.exists(correlation_candidates)][1L]
if (is.na(correlation_file)) stop("Run weather_yield_correlations_all_dap.R first.")
correlations <- read.csv(correlation_file, check.names = FALSE) |>
  dplyr::arrange(.data$Cor)
stopifnot(all(correlations$Domain == "DAP"), nrow(correlations) == 151L,
          all(correlations$Yield_Records == 10109L), all(correlations$Environments == 19L))

ranked <- correlations |>
  dplyr::mutate(Correlation_Rank = dplyr::row_number())
parameters_per_tail <- min(10L, floor(nrow(ranked) / 2L))
ranked <- ranked |>
  dplyr::filter(
    .data$Correlation_Rank <= parameters_per_tail |
      .data$Correlation_Rank > nrow(ranked) - parameters_per_tail
  ) |>
  dplyr::mutate(FPC_ID = factor(.data$FPC_ID, levels = .data$FPC_ID))

correlation_plot <- ggplot(ranked, aes(x = .data$FPC_ID, y = .data$Cor, fill = .data$Cor > 0)) +
  geom_col(show.legend = FALSE) +
  geom_text(
    aes(label = sprintf("%.2f", .data$Cor)),
    position = position_stack(vjust = 0.5),
    angle = 90,
    color = "black"
  ) +
  scale_fill_manual(values = c("TRUE" = "#648fff", "FALSE" = "#dc267f")) +
  scale_y_continuous(limits = range(correlations$Cor) + c(-0.05, 0.05)) +
  labs(x = "Weather parameter", y = "Correlation with grain yield") +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.6),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.3)
  )

dap_model_file <- g2f_resolve_input(
  paths,
  "DAP_Weather_FPCA_PTR.rds",
  result_subdirs = c(
    "02_fpca_weather/envrtype_dap/models",
    "02_fpca_weather/envrtype_dap"
  )
)
agdd_model_file <- g2f_resolve_input(
  paths,
  "AGDD_Weather_FPCA_PTR.rds",
  result_subdirs = c(
    "02_fpca_weather/envrtype_agdd/models",
    "02_fpca_weather/envrtype_agdd"
  )
)

pc_curves <- function(model, domain, components = 5L) {
  components <- seq_len(min(components, ncol(model$phi), length(model$lambda)))
  dplyr::bind_rows(lapply(components, function(component) {
    standard_deviation <- sqrt(model$lambda[[component]])
    data.frame(
      Time = model$workGrid,
      Mean = model$mu,
      Plus2SD = model$mu + 2 * standard_deviation * model$phi[, component],
      Minus2SD = model$mu - 2 * standard_deviation * model$phi[, component],
      PC = paste0("FPC", component),
      FPCA_Type = domain
    ) |>
      tidyr::pivot_longer(
        cols = c("Mean", "Plus2SD", "Minus2SD"),
        names_to = "Curve",
        values_to = "Value"
      )
  }))
}

curve_data <- dplyr::bind_rows(
  pc_curves(readRDS(dap_model_file), "DAP"),
  pc_curves(readRDS(agdd_model_file), "AGDD")
) |>
  dplyr::mutate(FPCA_Type = factor(.data$FPCA_Type, levels = c("DAP", "AGDD")))

trajectory_plot <- ggplot(
  curve_data,
  aes(x = .data$Time, y = .data$Value, color = .data$Curve)
) +
  geom_line(linewidth = 0.9) +
  scale_color_manual(
    values = c("Mean" = "black", "Plus2SD" = "#648fff", "Minus2SD" = "#fe6100"),
    breaks = c("Mean", "Plus2SD", "Minus2SD"),
    labels = c("Mean", "Mean + 2 SD", "Mean - 2 SD"),
    name = NULL
  ) +
  labs(x = NULL, y = "Photothermal ratio (PTR)") +
  theme_minimal(base_size = 12) +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
    strip.background = element_rect(fill = "grey90", color = "black", linewidth = 0.3),
    legend.position = "bottom"
  ) +
  ggh4x::facet_grid2(.data$FPCA_Type ~ .data$PC, scales = "free_x", independent = "x")

combined <- cowplot::plot_grid(
  trajectory_plot,
  correlation_plot,
  nrow = 2,
  rel_heights = c(0.4, 0.6),
  labels = c("A)", "B)")
)

ggsave(
  file.path(output_dir, "Figure_03_Weather_Yield_Correlations.pdf"),
  plot = combined,
  device = cairo_pdf,
  width = 10,
  height = 8
)
