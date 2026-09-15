# NGRDI and grain-yield variance-component figure used in the manuscript.

suppressPackageStartupMessages({
  library(cowplot)
  library(data.table)
  library(dplyr)
  library(ggplot2)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
paths <- g2f_paths()

vi_variance_file <- g2f_resolve_input(
  paths,
  "VarComp_G2F_2020_2021.csv",
  result_subdirs = "01_blue_variance"
)
yield_variance_file <- g2f_resolve_input(
  paths,
  "VarComp_Yield_G2F_2020_2021.csv",
  result_subdirs = "01_blue_variance"
)
output_dir <- g2f_results_dir(paths, "07_results_figures", "figures")

vi_variance <- fread(vi_variance_file)
yield_variance <- fread(yield_variance_file)

required_vi_columns <- c(
  "grp", "Percent", "Heritability", "R_squared", "Env.DAP",
  "Vegetation.Index", "DAP", "Env"
)
required_yield_columns <- c("grp", "Percent", "Heritability", "R_squared", "Env")
missing_vi_columns <- setdiff(required_vi_columns, names(vi_variance))
missing_yield_columns <- setdiff(required_yield_columns, names(yield_variance))
if (length(missing_vi_columns) > 0L) {
  stop("VI variance input is missing: ", paste(missing_vi_columns, collapse = ", "))
}
if (length(missing_yield_columns) > 0L) {
  stop("Yield variance input is missing: ", paste(missing_yield_columns, collapse = ", "))
}

# Preserve the historical input schema while using manuscript terminology.
group_levels <- c("Genotype", "Pass", "Range", "Replicate", "Residual")
vi_variance[grp == "Pedigree", grp := "Genotype"]
vi_variance[, grp := factor(grp, levels = group_levels)]
vi_variance[, Env := factor(Env, levels = sort(unique(Env)))]
vi_variance[, DAP := factor(DAP, levels = sort(unique(DAP)))]
vi_variance[, Percent := Percent / 100]
yield_variance[grp == "Pedigree", grp := "Genotype"]
yield_variance[, grp := factor(grp, levels = group_levels)]
yield_variance[, Env := factor(Env, levels = sort(unique(Env)))]
yield_variance[, Percent := Percent / 100]

ngrdi_variance <- vi_variance[Vegetation.Index == "NGRDI"]
if (nrow(ngrdi_variance) == 0L) stop("No NGRDI variance-component rows were found.")

# The final figure uses four representative environments. The dense MOH1.2020
# flight series is thinned by removing every second date in its historical order.
figure_environments <- c("MOH1.2020", "MIH1.2020", "TXH3.2020", "WIH3.2021")
missing_environments <- setdiff(figure_environments, as.character(unique(ngrdi_variance$Env)))
if (length(missing_environments) > 0L) {
  stop("NGRDI input is missing figure environments: ", paste(missing_environments, collapse = ", "))
}
moh_dates <- unique(ngrdi_variance[Env == "MOH1.2020", Env.DAP])
moh_dates_to_remove <- moh_dates[seq.int(2L, length(moh_dates), by = 2L)]

common_theme <- theme(
  legend.position.inside = c(0.9, 0.1),
  legend.background = element_rect(
    fill = alpha("azure", 0.5),
    linewidth = 0.1,
    linetype = "solid"
  ),
  legend.title = element_text(size = 11),
  legend.text = element_text(size = 11),
  axis.title.y = element_text(size = 13),
  axis.title.x = element_blank(),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  axis.line = element_line(colour = "black"),
  strip.background = element_rect(fill = "lightgray", colour = "black"),
  strip.text = element_text(colour = "black", face = "plain", size = 10),
  panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.3),
  panel.background = element_rect(fill = "white"),
  plot.background = element_rect(fill = "white")
)

vi_plot <- ngrdi_variance |>
  filter(
    !.data$Env.DAP %in% moh_dates_to_remove,
    as.character(.data$Env) %in% figure_environments
  ) |>
  ggplot(aes(x = .data$DAP, y = .data$Percent)) +
  geom_col(aes(fill = .data$grp), width = 0.7) +
  geom_point(
    aes(y = .data$Heritability, shape = "Heritability"),
    fill = "white",
    colour = "black",
    size = 1.5,
    alpha = 0.5
  ) +
  geom_point(
    aes(y = .data$R_squared, shape = "R-squared"),
    size = 1.5
  ) +
  scale_shape_manual(values = c(Heritability = 1, `R-squared` = 19), name = NULL) +
  facet_wrap(vars(Env), scales = "free_x") +
  guides(fill = guide_legend(title = "Variance\ncomponent")) +
  labs(y = "NGRDI variance components", x = "Days after Planting (DAP)") +
  common_theme +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, size = 10), axis.title.x = element_text())

yield_plot <- ggplot(
  yield_variance,
  aes(x = .data$Env, y = .data$Percent)
) +
  geom_col(aes(fill = .data$grp), width = 0.7) +
  geom_point(
    aes(y = .data$Heritability, shape = "Heritability"),
    fill = "white",
    colour = "black",
    size = 1.5,
    alpha = 0.5
  ) +
  geom_point(
    aes(y = .data$R_squared, shape = "R-squared"),
    size = 1.5
  ) +
  scale_shape_manual(values = c(Heritability = 1, `R-squared` = 19), name = NULL) +
  facet_wrap(vars(Env), scales = "free_x", nrow = 2L) +
  guides(fill = guide_legend(title = "Variance\ncomponent")) +
  labs(y = "Grain yield (t/ha) variance components") +
  common_theme +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

final_plot <- plot_grid(
  vi_plot,
  yield_plot,
  nrow = 2L,
  rel_heights = c(0.5, 0.5)
)

ggsave(
  filename = file.path(output_dir, "Figure_01_Variance_Components.pdf"),
  plot = final_plot,
  device = cairo_pdf,
  width = 12,
  height = 11
)

extreme_rows <- function(data, value_column, source, metric, id_column) {
  valid <- data |>
    filter(is.finite(.data[[value_column]]))
  if (nrow(valid) == 0L) stop("No finite values are available for ", source, " ", metric, ".")

  bind_rows(
    valid |>
      slice_min(order_by = .data[[value_column]], n = 1L, with_ties = FALSE) |>
      mutate(Extreme = "Minimum"),
    valid |>
      slice_max(order_by = .data[[value_column]], n = 1L, with_ties = FALSE) |>
      mutate(Extreme = "Maximum")
  ) |>
    transmute(
      Source = source,
      Metric = metric,
      Extreme = .data$Extreme,
      Environment_Time = as.character(.data[[id_column]]),
      Value = .data[[value_column]]
    )
}

ngrdi_genotype <- ngrdi_variance |>
  filter(.data$grp == "Genotype")
yield_genotype <- yield_variance |>
  filter(.data$grp == "Genotype")
paper_statistics <- bind_rows(
  extreme_rows(ngrdi_genotype, "Percent", "NGRDI", "Genotype variance proportion", "Env.DAP"),
  extreme_rows(ngrdi_genotype, "Heritability", "NGRDI", "Heritability", "Env.DAP"),
  extreme_rows(yield_genotype, "Heritability", "Yield", "Heritability", "Env")
)
fwrite(
  paper_statistics,
  file.path(output_dir, "NGRDI_VarComp_Paper_Statistics.csv")
)
