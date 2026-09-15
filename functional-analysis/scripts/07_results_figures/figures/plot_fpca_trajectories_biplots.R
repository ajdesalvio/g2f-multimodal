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
input <- function(filename, subdirs = character()) g2f_resolve_input(paths, filename, subdirs)

score_candidates <- c(
  file.path(paths$results_dir, "02_fpca_weather/vi_dap/FPC_Scores_BLUEs_Unified_FPCA_Max_FPCs_2020_2021_G2F_VIs.csv"),
  file.path(paths$data_dir, "FPC_Scores_BLUEs_Unified_FPCA_2020_2021_G2F_VIs.csv"),
  file.path(paths$data_dir, "FPC_Scores_BLUEs_Unified_FPCA_Max_FPCs_2020_2021_G2F_VIs.csv")
)
score_file <- score_candidates[file.exists(score_candidates)][1L]
if (is.na(score_file)) stop("Run vi_fpca_dap_full.R or supply the released DAP FPC scores.")
fpc <- fread(score_file) %>% as.data.frame()

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
yield <- fread(input('Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv', '01_blue_variance')) %>% as.data.frame()

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
fpcs_yield <- cowplot::plot_grid(combined_byEnv, combined_byYield, nrow = 2, rel_heights = c(0.5, 0.5), labels = c('C)', 'D)'))

#### FPC trajectory figure ####
# Run FPCA using the Unified_FPCA_Blues_V9.R script first, then you can make the figure
# or load the .rds file to avoid having to run FPCA
model_file <- function(domain, archived_name) {
  candidates <- c(
    file.path(paths$results_dir, paste0("02_fpca_weather/vi_", tolower(domain)), "models", paste0(domain, "_VI_FPCA_NGRDI.rds")),
    file.path(paths$data_dir, archived_name)
  )
  found <- candidates[file.exists(candidates)][1L]
  if (is.na(found)) stop("Missing ", domain, " NGRDI FPCA model.")
  found
}
FPCA.DAP <- readRDS(model_file("DAP", "FPCA_NGRDI_Standard.rds"))
FPCA.AGDD <- readRDS(model_file("AGDD", "FPCA_NGRDI_AGDD.rds"))

# Extract relevant components out of the FPCA object for DAP
mu      <- FPCA.DAP$mu
phi     <- FPCA.DAP$phi   # each column = eigenfunction
lambda  <- FPCA.DAP$lambda
timevec <- FPCA.DAP$workGrid

pc_curves_df <- function(pc_num) {
  sd_pc <- sqrt(lambda[pc_num])
  tibble(
    Time      = timevec,
    Mean      = mu,
    Plus2SD   = mu + 2 * sd_pc * phi[, pc_num],
    Minus2SD  = mu - 2 * sd_pc * phi[, pc_num]
  ) |>
    pivot_longer(cols = c(Mean, Plus2SD, Minus2SD),
                 names_to = "Curve", values_to = "Value") |>
    mutate(PC = paste0("FPC", pc_num))
}

df1 <- pc_curves_df(1)
df2 <- pc_curves_df(2)
df3 <- pc_curves_df(3) 
df4 <- pc_curves_df(4)
df5 <- pc_curves_df(5)

# Aggregate DAP data
df.DAP <- bind_rows(df1, df2, df3, df4, df5)

# Extract relevant components out of the FPCA object for AGDD
mu      <- FPCA.AGDD$mu
phi     <- FPCA.AGDD$phi
lambda  <- FPCA.AGDD$lambda
timevec <- FPCA.AGDD$workGrid

df1 <- pc_curves_df(1)
df2 <- pc_curves_df(2)
df3 <- pc_curves_df(3) 
df4 <- pc_curves_df(4)
df5 <- pc_curves_df(5)

# Aggregate AGDD Data
df.AGDD <- bind_rows(df1, df2, df3, df4, df5)
# Aggregate all data
df.DAP$FPCA_Type <- 'DAP'
df.AGDD$FPCA_Type <- 'AGDD'
df.DAP.AGDD <- bind_rows(df.DAP, df.AGDD)
df.DAP.AGDD$FPCA_Type <- factor(df.DAP.AGDD$FPCA_Type, levels = c('DAP', 'AGDD'))

# Individual FPCs +/- 2 SDs with the mean in each facet
p.base <- ggplot(df.DAP.AGDD, aes(x = Time, y = Value, color = Curve)) +
  geom_line(linewidth = 0.9) +
  scale_color_manual(
    values = c(
      "Mean"     = "black",      # or "grey30"
      "Plus2SD"  = "#648fff",    # color 1
      "Minus2SD" = "#fe6100"     # color 2
    ),
    breaks = c("Mean", "Plus2SD", "Minus2SD"),
    labels = c("Mean", "Mean + 2 SD", "Mean - 2 SD"),
    name = NULL
  ) +
  labs(x = NULL, y = "NGRDI") +
  theme_minimal(base_size = 12) +
  theme(
    panel.border    = element_rect(color = "black", fill = NA, linewidth = 0.5),
    strip.text.x = element_text(size = 10, margin = margin(t = 2, r = 1, b = 2, l = 1)),
    strip.text.y = element_text(size = 10, margin = margin(t = 2, r = 1, b = 2, l = 1)),
    strip.background = element_rect(fill = "grey90", color = "black", linewidth = 0.3),
    legend.position = "bottom",
    axis.title.x.top    = element_text(margin = margin(b = 6)),
    axis.title.x.bottom = element_text(margin = margin(t = 6)),
    #legend.title = element_text(size = 12),
    #legend.text = element_text(size = 12),
    #axis.title.y = element_text(size = 13)
  )

p.plus.minus <- p.base +
  ggh4x::facet_grid2(FPCA_Type ~ PC, scales = "free_x", independent = "x") +
  ggh4x::facetted_pos_scales(
    x = list(
      FPCA_Type == "DAP"  ~ scale_x_continuous(),
      FPCA_Type == "AGDD" ~ scale_x_continuous()
    )
  )

#### New addition - AGDD vs. DAP alignment ####
# Read in cleaned weather data to create a DAP / GDD conversion table
df.clim <- read.csv(input('EnvRtype_Weather_Data_Cleaned_V2.csv', '00_data_prep'))
dap.gdd <- df.clim %>% select(Env, DAP, GDD, YYYYMMDD)

# Create the accumulated GDD column (AGDD)
dap.gdd <- dap.gdd %>%
  arrange(Env, DAP) %>%
  group_by(Env) %>%
  mutate(AGDD = cumsum(GDD)) %>%
  ungroup()

# Import entire VI data frame
VI <- fread(input('VI_BLUEs_G2F_2020_2021.CSV', '01_blue_variance')) %>% as.data.frame()
VI$Env.DAP.VI <- paste(VI$Env.DAP, VI$Vegetation.Index, sep = '.')
VI$Pedigree.Env <- paste(VI$Pedigree, VI$Env, sep = '.')

# Left-join the AGDD data frame to the VI data frame
VI <- VI %>% left_join(dap.gdd, by = c('Env', 'DAP'))

# Separate AGDD and DAP data frames
dap <- VI %>% filter(Env %in% c('TXH2.2021', 'TXH3.2021')) %>%
  mutate(Alignment_Type = 'DAP', Time_Axis = DAP) %>%
  filter(Vegetation.Index == 'NGRDI')
agdd <- VI %>% filter(Env %in% c('TXH2.2021', 'TXH3.2021')) %>%
  mutate(Alignment_Type = 'AGDD', Time_Axis = AGDD) %>%
  filter(Vegetation.Index == 'NGRDI')
dap.agdd <- bind_rows(dap, agdd)
dap.agdd$Alignment_Type <- factor(dap.agdd$Alignment_Type, levels = c('DAP', 'AGDD'))

p0 <- dap.agdd %>% 
  filter(Vegetation.Index == 'NGRDI') %>%
  ggplot() +
  aes(x = Time_Axis, y = VI.BLUE,
      group = Pedigree.Env,
      color = Env) +
  geom_line(alpha = 1, linewidth = 0.75) +
  scale_color_manual(values = c('#fe6100', '#dc267f'), name = 'Environment') +
  facet_wrap(~Alignment_Type, scales = 'free_x') +
  ylab('NGRDI BLUEs') +
  xlab('Time Axis (DAP or AGDD)') +
  theme_minimal() +
  theme(
    panel.border    = element_rect(color = "black", fill = NA, linewidth = 0.5),
    strip.text.x = element_text(size = 10, margin = margin(t = 2, r = 1, b = 2, l = 1)),
    strip.text.y = element_text(size = 10, margin = margin(t = 2, r = 1, b = 2, l = 1)),
    strip.background = element_rect(fill = "grey90", color = "black", linewidth = 0.3),
    legend.position = "bottom",
    axis.title.x.top    = element_text(margin = margin(b = 6)),
    axis.title.x.bottom = element_text(margin = margin(t = 6))
  )


p0 <- cowplot::plot_grid(p0, labels = c('A)'))
p.plus.minus <- cowplot::plot_grid(p.plus.minus, labels = c('B)'))


ggsave(file.path(output_dir, "Figure_02_FPCA_Trajectories_Biplots.pdf"),
       plot = cowplot::plot_grid(p0, p.plus.minus, fpcs_yield, nrow = 3, rel_heights = c(0.2, 0.2, 0.6)),
       device = cairo_pdf,
       width = 12, height = 13)
