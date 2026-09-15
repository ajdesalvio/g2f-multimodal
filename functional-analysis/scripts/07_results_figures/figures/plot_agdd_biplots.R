library(RColorBrewer)
library(dplyr)
library(ggplot2)
library(tidyr)
library(data.table)
library(gridExtra)
library(cowplot)

#### FPC biplots ####
project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
paths <- g2f_paths()
output_dir <- g2f_results_dir(paths, "07_results_figures", "figures")

fpc <- fread(g2f_resolve_input(paths, 'FPC_Scores_BLUEs_AGDD_Unified_FPCA_Max_FPCs_2020_2021_G2F_VIs.csv', '02_fpca_weather/vi_agdd')) %>% as.data.frame()

# Filter for a specific vegetation index
fpc.NGRDI <- fpc %>% filter(Vegetation.Index == 'NGRDI')
envs <- sort(unique(fpc$Env))

# Set Env to be a sorted factor
fpc.NGRDI$Env <- factor(fpc.NGRDI$Env, levels = envs)

# Functional variance explained
fve1 <- round(fpc.NGRDI$FPC1_FVE[1] * 100, 1)
fve2 <- round(fpc.NGRDI$FPC2_FVE[1] * 100, 1)
fve3 <- round(fpc.NGRDI$FPC3_FVE[1] * 100, 1)

# Color palette
palette <- c(brewer.pal(8, "Set1"), brewer.pal(8, "Dark2")[1:7], brewer.pal(8, "Pastel1")[1:4])

fpc1_2 <- ggplot(fpc.NGRDI, aes(x = FPC1, y = FPC2, color = Env)) +
  geom_point(size = 2, stroke = 0) +
  scale_color_manual(values = palette) +
  theme_classic() +
  theme(
    legend.position = 'none',
    panel.background = element_rect(fill = "white", color = "white"),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.title = element_text(color = "black"),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = paste0('FPC1 (', fve1, '%)'),
    y = paste0('FPC2 (', fve2, '%)'),
    color = "Environment")

fpc1_3 <- ggplot(fpc.NGRDI, aes(x = FPC1, y = FPC3, color = Env)) +
  geom_point(size = 2, stroke = 0) +
  scale_color_manual(values = palette) +
  theme_classic() +
  theme(
    legend.position = 'none',
    panel.background = element_rect(fill = "white", color = "white"),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.title = element_text(color = "black"),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = paste0('FPC1 (', fve1, '%)'),
    y = paste0('FPC3 (', fve3, '%)'),
    color = "Environment")

fpc2_3 <- ggplot(fpc.NGRDI, aes(x = FPC2, y = FPC3, color = Env)) +
  geom_point(size = 2, stroke = 0) +
  scale_color_manual(values = palette) +
  theme_classic() +
  theme(
    legend.position = 'none',
    panel.background = element_rect(fill = "white", color = "white"),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.title = element_text(color = "black"),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = paste0('FPC2 (', fve2, '%)'),
    y = paste0('FPC3 (', fve3, '%)'),
    color = "Environment")

# Combine the plots
combined_plot <- arrangeGrob(fpc1_2, fpc1_3, fpc2_3, ncol = 3)

# Create one of the plots with a legend
legend <- ggplot(fpc.NGRDI, aes(x = FPC1, y = FPC2, color = Env)) +
  geom_point(size = 2, stroke = 0) +
  scale_color_manual(
    values = palette,
    guide = guide_legend(nrow = 2, byrow = T, override.aes = list(size = 4))  # Specify the number of rows in the legend
  ) +
  theme_classic() +
  theme(
    legend.position = 'bottom',
    panel.background = element_rect(fill = "white", color = "white"),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.title = element_text(color = "black"),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = paste0('FPC1 (', fve1, '%)'),
    y = paste0('FPC2 (', fve2, '%)'),
    color = "Environment")

# Function to extract legend from plot 
get_only_legend <- function(plot) cowplot::get_legend(plot, legend = "bottom")

# Extract the legend
legend <- get_only_legend(legend)

# Save this as its own plot for the stacked plot at the bottom
combined_byEnv <- arrangeGrob(combined_plot, legend, nrow = 2, heights = c(5,0.75))

# Plot each point colored by YIELD now
yield <- fread(g2f_resolve_input(paths, 'Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv', '01_blue_variance')) %>% as.data.frame()

# Merge the FPCs with the yield values (remove rows where yield is NA)
fpc.NGRDI.yield <- fpc.NGRDI %>% left_join(yield, by = 'Pedigree.Env') %>%
  drop_na(Yield.t.ha.BLUE)

fpc1_2_yield <- ggplot(fpc.NGRDI.yield, aes(x = FPC1, y = FPC2, color = Yield.t.ha.BLUE)) +
  geom_point(size = 2) +
  scale_color_viridis_c(name = "Yield (t/ha)", option = "C", end = 0.95) +
  theme_classic() +
  theme(
    legend.position = "none",
    panel.background = element_rect(fill = "white", color = "white"),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.title = element_text(color = "black"),
    axis.text  = element_text(color = "black")
  ) +
  labs(
    x = paste0("FPC1 (", fve1, "%)"),
    y = paste0("FPC2 (", fve2, "%)")
  )

fpc1_3_yield <- ggplot(fpc.NGRDI.yield, aes(x = FPC1, y = FPC3, color = Yield.t.ha.BLUE)) +
  geom_point(size = 2) +
  scale_color_viridis_c(name = "Yield (t/ha)", option = "C", end = 0.95) +
  theme_classic() +
  theme(
    legend.position = 'none',
    panel.background = element_rect(fill = "white", color = "white"),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.title = element_text(color = "black"),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = paste0('FPC1 (', fve1, '%)'),
    y = paste0('FPC3 (', fve3, '%)'),
    color = "Environment")

fpc2_3_yield <- ggplot(fpc.NGRDI.yield, aes(x = FPC2, y = FPC3, color = Yield.t.ha.BLUE)) +
  geom_point(size = 2) +
  scale_color_viridis_c(name = "Yield (t/ha)", option = "C", end = 0.95) +
  theme_classic() +
  theme(
    legend.position = 'none',
    panel.background = element_rect(fill = "white", color = "white"),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.title = element_text(color = "black"),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = paste0('FPC2 (', fve2, '%)'),
    y = paste0('FPC3 (', fve3, '%)'),
    color = "Environment")

# Create one of the plots with a legend
legend_yield <- ggplot(fpc.NGRDI.yield, aes(x = FPC1, y = FPC2, color = Yield.t.ha.BLUE)) +
  geom_point(size = 2, stroke = 0) +
  scale_color_viridis_c(name = "Yield (t/ha)", option = "C", end = 0.95) +
  theme_classic() +
  theme(
    legend.position = 'bottom',
    panel.background = element_rect(fill = "white", color = "white"),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.title = element_text(color = "black"),
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = paste0('FPC1 (', fve1, '%)'),
    y = paste0('FPC2 (', fve2, '%)'),
    color = "Environment")

# Extract the legend
legend_yield <- get_only_legend(legend_yield)

# Combine the plots
combined_plot_yield <- arrangeGrob(fpc1_2_yield, fpc1_3_yield, fpc2_3_yield, ncol = 3)

# Save this as its own plot for combining and comparing with the by-environment plot
combined_byYield <- arrangeGrob(combined_plot_yield, legend_yield, nrow = 2, heights = c(5,0.75))

# Combine everything
fpcs_yield <- cowplot::plot_grid(combined_byEnv, combined_byYield, nrow = 2, rel_heights = c(0.5, 0.5), labels = c('A)', 'B)'))

ggsave(file.path(output_dir, "Supplementary_01_AGDD_Biplots.pdf"),
       plot = fpcs_yield,
       device = cairo_pdf,
       width = 12, height = 8)
