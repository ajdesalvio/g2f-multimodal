# Figure 4. Final no-Ze kernel and TNP prediction results (source: V6).
# Genetic-only models are averaged over the DAP/AGDD runs for display.
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(cowplot)
})
project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
paths <- g2f_paths()
out_dir <- g2f_results_dir(paths, "07_results_figures", "figures")
source_dir <- file.path(out_dir, "source_data")
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)
summary_candidates <- c(
  Sys.getenv("G2F_PREDICTION_SUMMARY", unset = ""),
  file.path(paths$project_dir, "results/prediction/final_summaries/G2F_Final_Prediction_Summary.csv"),
  file.path(paths$results_dir, "07_results_figures/final_prediction_summaries/G2F_Final_Prediction_Summary.csv"),
  file.path(paths$data_dir, "G2F_Final_Prediction_Summary.csv")
)
summary_file <- summary_candidates[file.exists(summary_candidates)][1L]
if (is.na(summary_file)) stop("Final prediction summary not found; run summarize_prediction_results.R.")
df <- read.csv(summary_file) |>
  select(Method, Model, CV, Time_Domain, Correlation, RMSE)
if (nrow(df) != 152L || any(!is.finite(df$Correlation)) || any(!is.finite(df$RMSE))) {
  stop("Expected the complete 152-row final summary with finite correlations and RMSEs.")
}
model_levels <- c(
  "M1.G", "M1.P", "M2.G", "M3.G-Int", "M3.P-Int", "M4.G.W", "M4.P.W",
  "M5.G.W-Int", "M5.P.W-Int", "M6.G.P", "M7.G.P", "M8.G.P-Int",
  "M9.G.P.W", "M10.G.P.W-Int", "VI-only", "VI+genomic", "VI+genomic+weather"
)
df$Model <- factor(df$Model, levels = model_levels)
df$CV <- factor(df$CV, levels = c("CV2", "CV1", "CV0", "CV00", "LOEO"))
genetic_models <- c("M1.G", "M2.G", "M3.G-Int")
collapse_genetic <- function(data, by_cv = TRUE) {
  grouping <- if (by_cv) c("Model", "CV") else "Model"
  genetic <- data |>
    filter(Model %in% genetic_models) |>
    group_by(across(all_of(grouping))) |>
    summarise(Correlation = mean(Correlation), RMSE = mean(RMSE), .groups = "drop") |>
    mutate(Time_Domain = "Genetic-only")
  bind_rows(genetic, filter(data, !Model %in% genetic_models)) |>
    mutate(Time_Domain = factor(Time_Domain, levels = c("Genetic-only", "DAP", "AGDD")))
}
cv <- collapse_genetic(filter(df, Method == "BGLR", CV != "LOEO"))
loeo <- collapse_genetic(filter(df, Method == "BGLR", CV == "LOEO"), FALSE)
tnp <- filter(df, Method == "TNP")
stopifnot(nrow(cv) == 100L, nrow(loeo) == 25L, nrow(tnp) == 12L)
pd <- position_dodge2(width = 0.8, preserve = "single")
base_theme <- theme_bw() + theme(
  axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
  legend.position = "bottom", panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
  plot.title.position = "plot",
  plot.title = element_text(face = "bold", size = 14, hjust = 0, margin = margin(b = 4))
)
kernel_plot <- function(data) {
  ggplot(data, aes(Model, Correlation, fill = Time_Domain)) +
    geom_col(position = pd, width = 0.8) +
    geom_text(aes(label = sprintf("%.3f", Correlation), group = Time_Domain),
              position = pd, angle = 90, hjust = 1, vjust = 0.5, color = "white", size = 3.5) +
    scale_fill_manual(values = c("Genetic-only" = "#fe6100", "DAP" = "#648fff", "AGDD" = "#dc267f")) +
    guides(fill = guide_legend(title = "Model Type", nrow = 1)) +
    labs(x = "Model Name") + base_theme
}
p1 <- kernel_plot(cv) + facet_wrap(~CV, scales = "free_y") +
  scale_y_continuous("Correlation (Actual vs. Predicted Yield)", expand = expansion(mult = c(0, 0.05))) +
  labs(title = "A) BGLR CV Prediction")
p2 <- kernel_plot(loeo) +
  scale_y_continuous("Correlation", limits = c(0, 0.5), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "C) BGLR LOEO Prediction")
p3 <- ggplot(tnp, aes(Model, Correlation, fill = Model)) +
  geom_col(position = pd, width = 0.8) +
  geom_text(aes(label = sprintf("%.3f", Correlation), group = Model),
            position = pd, angle = 90, hjust = 1, vjust = 0.5, color = "white", size = 3.5) +
  facet_wrap(~CV, scales = "free_x", space = "free_x") +
  scale_y_continuous("Correlation", limits = c(0, 0.8)) +
  scale_fill_manual(values = c("#DC3220", "#005AB5", "#D35FB7")) +
  guides(fill = guide_legend(title = "Model Type", nrow = 1)) +
  labs(x = "Model Name", title = "B) TNP CV Prediction") + base_theme +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
combined <- plot_grid(p1, p3, p2, nrow = 3, rel_heights = c(0.45, 0.27, 0.28))
ggsave(file.path(out_dir, "Figure_04_Prediction.pdf"), combined,
       device = cairo_pdf, width = 8.5, height = 10.5)
write.csv(loeo |> arrange(Model, Time_Domain) |> select(Model, Time_Domain, Correlation),
          file.path(source_dir, "Figure_04_LOEO_Bars.csv"), row.names = FALSE)
write.csv(df, file.path(source_dir, "Figure_04_Prediction_Summary.csv"), row.names = FALSE)
