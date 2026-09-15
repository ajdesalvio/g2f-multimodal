suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(cowplot)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
source(file.path(project_dir, "R", "utils", "prediction_results.R"))
from_source_data <- "--from-source-data" %in% commandArgs(trailingOnly = TRUE)
roots <- g2f_paths(require_data = !from_source_data)
out <- Sys.getenv("G2F_FIGURE_OUTPUT_DIR", unset = "")
if (!nzchar(out)) out <- g2f_results_dir(roots, "07_results_figures", "figures")
dir.create(file.path(out, "source_data"), recursive = TRUE, showWarnings = FALSE)

if (from_source_data) {
  # Lightweight rendering of the retained paper results; no raw HPRC files needed.
  source_dir <- file.path(roots$project_dir, "results", "figures", "source_data")
  read_source <- function(filename, required) {
    file <- file.path(source_dir, filename)
    g2f_require_file(file)
    x <- fread(file)
    if (!all(required %in% names(x))) stop(filename, " is missing required columns.")
    x
  }
  environment_mean <- read_source("CV000_Environment_Mean_Differences.csv",
    c("Domain", "CV", "Environment", "Correlation_Difference", "RMSE_Difference"))
  fold <- read_source("CV_Fold_Metrics.csv",
    c("Series", "CV", "Seed", "Fold", "Correlation", "RMSE"))
  weight_long <- read_source("CV000_RMSE_Weighting_Paired_Differences.csv",
    c("Domain", "CV", "Seed", "Aggregation", "RMSE_Difference"))
  weight_summary <- read_source("CV000_RMSE_Weighting_Summary.csv",
    c("Domain", "CV", "Aggregation", "Mean", "SD"))
  stopifnot(
    nrow(environment_mean) == 76L, uniqueN(environment_mean$Environment) == 19L,
    !anyDuplicated(environment_mean, by = c("Domain", "CV", "Environment")),
    nrow(fold) == 500L, all(fold$Seed %in% 1:5), all(fold$Fold %in% 1:5),
    !anyDuplicated(fold, by = c("Series", "CV", "Seed", "Fold")),
    nrow(weight_long) == 40L,
    !anyDuplicated(weight_long, by = c("Domain", "CV", "Seed", "Aggregation")),
    nrow(weight_summary) == 8L,
    !anyDuplicated(weight_summary, by = c("Domain", "CV", "Aggregation")),
    all(is.finite(environment_mean$Correlation_Difference)), all(is.finite(environment_mean$RMSE_Difference)),
    all(is.finite(fold$Correlation)), all(is.finite(fold$RMSE)),
    all(is.finite(weight_long$RMSE_Difference)), all(is.finite(weight_summary$Mean)),
    all(is.finite(weight_summary$SD)), all(weight_summary$SD >= 0)
  )
} else {
inputs <- g2f_prediction_result_paths(roots, g2f_analysis_config())
seeds <- 1:5
primary_checkpoint <- "best_frozen_pearson_r"
seed_candidates <- c(
  Sys.getenv("G2F_PREDICTION_SEED_SUMMARY", unset = ""),
  file.path(roots$project_dir, "results/prediction/final_summaries/G2F_Final_Prediction_Seed_Summary.csv"),
  file.path(roots$results_dir, "07_results_figures/final_prediction_summaries/G2F_Final_Prediction_Seed_Summary.csv")
)
seed_file <- seed_candidates[file.exists(seed_candidates)][1L]
if (is.na(seed_file)) stop("Run summarize_prediction_results.R before the paired figures.")
paths <- list(
  DAP0 = inputs$cv0_dap, AGDD0 = inputs$cv0_agdd,
  DAP21 = inputs$cv21_dap, AGDD21 = inputs$cv21_agdd,
  TNP = inputs$tnp, Final = seed_file
)
stopifnot(all(file.exists(unlist(paths))))

tiezzi <- function(r, n) {
  keep <- is.finite(r) & is.finite(n) & n > 2
  weighted.mean(r[keep], 1 / pmax((1 - r[keep]^2) / (n[keep] - 2), .Machine$double.eps))
}
pool_rmse <- function(r, n) sqrt(weighted.mean(r^2, n))
save_data <- function(x, name) fwrite(x, file.path(out, "source_data", paste0(name, ".csv")))
keys <- c("CV", "Seed", "Fold", "Environment")

read_cv0 <- function(path, domain) {
  files <- list.files(path, "^Seed0[1-5]\\..*noZe\\.metrics\\.csv$", full.names = TRUE)
  stopifnot(length(files) == 475L)
  x <- rbindlist(lapply(files, fread))[Model_Name == "M10.G.P"]
  stopifnot(nrow(x) == 950L, all(x$Status == "ok"), all(!x$Include_Ze))
  x[, .(Method = "BGLR", Model = "M10.G.P.W-Int", Time_Domain = domain,
        Series = paste("M10", domain), CV = Metric, Seed = Seed_Num, Fold = Fold_Num,
        Environment = Heldout_Env, N, Correlation = Cor, RMSE)]
}

read_cv21 <- function(path, domain) {
  files <- list.files(path, "^Seed0[1-5]\\..*noZe\\.M10\\.G\\.P\\.prediction_values\\.csv$", full.names = TRUE)
  stopifnot(length(files) == 25L)
  rbindlist(lapply(files, function(file) {
    id <- regmatches(basename(file), regexec("^Seed([0-9]+)\\.Fold([0-9]+)\\.", basename(file)))[[1]]
    seed <- as.integer(id[2]); fold <- as.integer(id[3])
    x <- fread(file, select = c("Pedigree.Env", "Env", "Fold", "Actual", "Predicted"))
    x <- x[complete.cases(Actual, Predicted, Fold, Env)] # Same scored rows as the final summary.
    stopifnot(!anyDuplicated(x$Pedigree.Env), all(is.finite(x$Actual)), all(is.finite(x$Predicted)),
              !anyNA(x[, .(Env, Fold)]))
    rbindlist(lapply(c("CV1", "CV2"), function(cv) {
      z <- if (cv == "CV1") x[Fold == fold] else x[Fold != fold]
      e <- z[, .(N = .N, r = cor(Actual, Predicted)), by = Env]
      data.table(Method = "BGLR", Model = "M10.G.P.W-Int", Time_Domain = domain,
                 Series = paste("M10", domain), CV = cv, Seed = seed, Fold = fold,
                 N = nrow(z), Environments = nrow(e), Correlation = tiezzi(e$r, e$N),
                 RMSE = sqrt(mean((z$Actual - z$Predicted)^2)))
    }))
  }))
}

message("Reading shared-seed kernel and TNP results...")
kernel_env <- rbindlist(list(read_cv0(paths$DAP0, "DAP"), read_cv0(paths$AGDD0, "AGDD")))
kernel_fold21 <- rbindlist(list(read_cv21(paths$DAP21, "DAP"), read_cv21(paths$AGDD21, "AGDD")))
tnp <- fread(paths$TNP)[checkpoint == primary_checkpoint & seed %in% seeds]
tnp_names <- c("istnp-full" = "VI+genomic+weather", "weather-free" = "VI+genomic", "vi-only" = "VI-only")
tnp_series <- c("istnp-full" = "TNP VI+G+W", "weather-free" = "TNP VI+G", "vi-only" = "TNP VI")
tnp[, `:=`(Method = "TNP", Model = unname(tnp_names[model]), Time_Domain = "DAP",
           Series = unname(tnp_series[model]), CV = cv_label, Seed = seed,
           Fold = as.integer(sub("^Fold([0-9]+).*$", "\\1", fold)),
           Environment = sub("^Fold[0-9]+\\.", "", fold))]
stopifnot(nrow(tnp) == 3000L, !anyNA(tnp$Model))
tnp_env <- tnp[CV %chin% c("CV0", "CV00"),
               .(Method, Model, Time_Domain, Series, CV, Seed, Fold, Environment,
                 N = n, Correlation = pearson_r, RMSE = rmse)]
env <- rbindlist(list(kernel_env, tnp_env))
stopifnot(!anyDuplicated(env, by = c("Series", keys)),
          all(is.finite(env$Correlation)), all(is.finite(env$RMSE)), all(env$N > 2),
          all(env[, .N, by = .(Series, CV, Seed, Fold)]$N == 19L))

group <- c("Method", "Model", "Time_Domain", "Series", "CV", "Seed", "Fold")
fold0 <- env[, .(N = sum(N), Environments = .N, Correlation = tiezzi(Correlation, N),
                 RMSE = pool_rmse(RMSE, N), Equal_Environment_RMSE = mean(RMSE)), by = group]
tnp_fold21 <- tnp[CV %chin% c("CV1", "CV2"),
                  .(Method, Model, Time_Domain, Series, CV, Seed, Fold, N = n,
                    Environments = r_w_n_blocks, Correlation = r_w, RMSE = rmse)]
fold <- rbindlist(list(fold0, kernel_fold21, tnp_fold21), fill = TRUE)
stopifnot(!anyDuplicated(fold, by = c("Series", "CV", "Seed", "Fold")),
          all(fold[, .N, by = .(Series, CV)]$N == 25L), all(fold$Environments == 19L),
          all(is.finite(fold$Correlation)), all(is.finite(fold$RMSE)))

pair <- merge(kernel_env[, .(Domain = Time_Domain, CV, Seed, Fold, Environment,
                            Kernel_N = N, Kernel_r = Correlation, Kernel_RMSE = RMSE)],
              tnp_env[Series == "TNP VI+G+W", .(CV, Seed, Fold, Environment,
                       TNP_N = N, TNP_r = Correlation, TNP_RMSE = RMSE)], by = keys)
stopifnot(nrow(pair) == 1900L, all(pair$Kernel_N == pair$TNP_N))
pair21 <- merge(kernel_fold21[, .(Series, CV, Seed, Fold, Kernel_N = N)],
                tnp_fold21[Series == "TNP VI+G+W", .(CV, Seed, Fold, TNP_N = N)],
                by = c("CV", "Seed", "Fold"))
stopifnot(nrow(pair21) == 100L, all(pair21$Kernel_N == pair21$TNP_N))

seed <- fold[, .(Correlation = mean(Correlation), RMSE = mean(RMSE)),
             by = .(Method, Model, Time_Domain, Series, CV, Seed)]
reference <- fread(paths$Final)[Seed %in% seeds & (Method == "TNP" | Model == "M10.G.P.W-Int")]
check <- merge(seed, reference, by = c("Method", "Model", "Time_Domain", "CV", "Seed"), suffixes = c("", "_Final"))
stopifnot(nrow(check) == 100L, nrow(reference) == 100L)
errors <- c(Correlation = max(abs(check$Correlation - check$Correlation_Final)),
            RMSE = max(abs(check$RMSE - check$RMSE_Final)))
stopifnot(all(errors < 1e-10))
save_data(data.table(Check = c("CV000 matched cells", "CV21 matched folds", "Final seed comparisons", names(errors)),
                     Value = c(nrow(pair), nrow(pair21), nrow(check), errors)), "Validation")

pair[, `:=`(Correlation_Difference = TNP_r - Kernel_r, RMSE_Difference = TNP_RMSE - Kernel_RMSE)]
environment_seed <- pair[, lapply(.SD, mean), by = .(Domain, CV, Seed, Environment),
                         .SDcols = c("Kernel_N", "Kernel_r", "TNP_r", "Kernel_RMSE", "TNP_RMSE",
                                     "Correlation_Difference", "RMSE_Difference")]
environment_mean <- environment_seed[, lapply(.SD, mean), by = .(Domain, CV, Environment),
                                     .SDcols = c("Kernel_N", "Kernel_r", "TNP_r", "Kernel_RMSE", "TNP_RMSE",
                                                 "Correlation_Difference", "RMSE_Difference")]
setnames(environment_mean, "Kernel_N", "Mean_Scored_N")
weight_seed <- fold0[, .(Pooled = mean(RMSE), Equal_Environment = mean(Equal_Environment_RMSE)),
                    by = .(Series, CV, Seed)]
weight_pair <- merge(weight_seed[grepl("^M10", Series)], weight_seed[Series == "TNP VI+G+W"],
                     by = c("CV", "Seed"), suffixes = c("_M10", "_TNP"))
weight_pair[, `:=`(Domain = sub("M10 ", "", Series_M10),
                   Pooled = Pooled_TNP - Pooled_M10,
                   Equal_Environment = Equal_Environment_TNP - Equal_Environment_M10)]
weight_long <- melt(weight_pair, id.vars = c("Domain", "CV", "Seed"),
                    measure.vars = c("Pooled", "Equal_Environment"),
                    variable.name = "Aggregation", value.name = "RMSE_Difference")
weight_summary <- weight_long[, .(Mean = mean(RMSE_Difference), SD = sd(RMSE_Difference)),
                             by = .(Domain, CV, Aggregation)]
save_data(env, "CV000_Environment_Metrics")
save_data(fold, "CV_Fold_Metrics")
save_data(seed, "CV_Seed_Metrics")
save_data(environment_mean, "CV000_Environment_Mean_Differences")
save_data(weight_seed, "CV000_RMSE_Weighting_Seed_Metrics")
save_data(weight_long, "CV000_RMSE_Weighting_Paired_Differences")
save_data(weight_summary, "CV000_RMSE_Weighting_Summary")
}

colors <- c("M10 DAP" = "#648fff", "M10 AGDD" = "#dc267f", "TNP VI" = "#DC3220",
            "TNP VI+G" = "#005AB5", "TNP VI+G+W" = "#D35FB7")
domain_colors <- c(DAP = "#648fff", AGDD = "#dc267f")
cv_levels <- c("CV2", "CV1", "CV0", "CV00")
cv_labels <- c(CV2 = "CV2 (in-sample)", CV1 = "CV1", CV0 = "CV0", CV00 = "CV00")
style <- theme_bw(base_size = 10, base_family = "sans") +
  theme(panel.grid.minor = element_blank(), panel.grid.major = element_blank(),
        strip.background = element_rect(fill = "grey93", linewidth = 0.4),
        strip.text = element_text(face = "bold"), legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 9, margin = margin(b = 8)),
        plot.caption = element_text(size = 8, hjust = 0), plot.margin = margin(8, 10, 8, 8))
theme_set(style)
save_plot <- function(p, name, width, height) {
  ggsave(file.path(out, paste0(name, ".pdf")), p, device = cairo_pdf, width = width, height = height)
}
assemble <- function(a, b, title, subtitle, caption, heights = c(1, 1)) {
  panels <- plot_grid(a, b, ncol = 1, align = "v", rel_heights = heights)
  plot_grid(ggdraw() + draw_label(title, x = 0.015, hjust = 0, fontface = "bold", size = 14),
            ggdraw() + draw_label(subtitle, x = 0.015, hjust = 0, size = 9), panels,
            ggdraw() + draw_label(caption, x = 0.015, hjust = 0, size = 8),
            ncol = 1, rel_heights = c(0.045, 0.04, 0.855, 0.06))
}

environment_mean[, `:=`(CV = factor(CV, c("CV0", "CV00")),
                         Domain = factor(Domain, c("DAP", "AGDD")),
                         Environment = factor(Environment, rev(sort(unique(Environment)))))]
env_panel <- function(metric, title, axis) {
  ggplot(environment_mean, aes(x = .data[[metric]], y = Environment, color = Domain, shape = Domain)) +
    geom_vline(xintercept = 0, color = "grey55", linewidth = 0.45) +
    geom_point(position = position_dodge(width = 0.45), size = 2.1) +
    facet_wrap(~CV, nrow = 1) + scale_color_manual(values = domain_colors) +
    scale_shape_manual(values = c(16, 17)) +
    scale_x_continuous(expand = expansion(mult = 0.10)) +
    labs(title = title, x = axis, y = NULL, color = "M10 time domain", shape = "M10 time domain") +
    theme(axis.text.y = element_text(size = 8))
}
ep1 <- env_panel("Correlation_Difference", "A  Correlation difference", "TNP - M10 correlation (positive favors TNP)") + theme(legend.position = "none")
ep2 <- env_panel("RMSE_Difference", "B  RMSE difference", "TNP - M10 RMSE (t/ha; negative favors TNP)")
save_plot(assemble(ep1, ep2, "Where do TNP and M10 differ?",
                   "Full TNP versus M10 | 19 environment-years | shared seeds 1-5",
                   "Points are environment means over folds within seed, then over seeds. The same full TNP is used in both comparisons.\nThese environment-level differences are not the headline, across-environment pooled metrics."),
          "Supplementary_02_Environment_Differences", 10, 11)

fold[, `:=`(Series = factor(Series, names(colors)), CV = factor(CV, cv_levels))]
short_labels <- c("M10\nDAP", "M10\nAGDD", "TNP\nVI", "TNP\nVI+G", "TNP\nVI+G+W")
fold_panel <- function(metric, title, axis) {
  ggplot(fold, aes(x = Series, y = .data[[metric]], fill = Series)) +
    geom_violin(trim = TRUE, alpha = 0.3, width = 0.85, linewidth = 0.35) +
    geom_boxplot(width = 0.10, outlier.shape = NA, fill = "white", linewidth = 0.35) +
    geom_point(position = position_jitter(width = 0.10, height = 0, seed = 2),
               shape = 21, size = 0.95, alpha = 0.6, stroke = 0.2) +
    facet_wrap(~CV, nrow = 1, labeller = as_labeller(cv_labels)) +
    scale_fill_manual(values = colors) + scale_x_discrete(labels = short_labels) +
    labs(title = title, x = NULL, y = axis) +
    theme(legend.position = "none", axis.text.x = element_text(size = 7))
}
save_plot(assemble(fold_panel("Correlation", "A  Within-environment predictive ability", "Tiezzi-weighted correlation"),
                   fold_panel("RMSE", "B  Prediction error", "Pooled RMSE (t/ha)"),
                   "Variation in prediction performance across shared splits",
                   "25 fold-level values per model and scenario | 5 shared seeds x 5 folds",
                   "CV0/CV00: correlations and squared errors are pooled across 19 environments within each fold, using exact N.\nCV1/CV2: within-environment correlation and pooled RMSE. CV2 is in-sample. Repeated-CV values are not independent replicates."),
          "Supplementary_03_Fold_Distributions", 11, 7.5)

weight_long[, `:=`(Domain = factor(Domain, c("DAP", "AGDD")), CV = factor(CV, c("CV0", "CV00")),
                   Aggregation = factor(Aggregation, c("Pooled", "Equal_Environment")))]
weight_summary[, `:=`(Domain = factor(Domain, c("DAP", "AGDD")), CV = factor(CV, c("CV0", "CV00")),
                      Aggregation = factor(Aggregation, c("Pooled", "Equal_Environment")))]
wp <- ggplot(weight_long, aes(Aggregation, RMSE_Difference)) +
  geom_hline(yintercept = 0, color = "grey55", linewidth = 0.45) +
  geom_line(aes(group = Seed, color = Domain), alpha = 0.45, linewidth = 0.5) +
  geom_point(aes(color = Domain), size = 2, alpha = 0.7) +
  geom_errorbar(data = weight_summary, aes(y = Mean, ymin = Mean - SD, ymax = Mean + SD),
                width = 0.07, linewidth = 0.5) +
  geom_line(data = weight_summary, aes(y = Mean, group = 1), linewidth = 0.7) +
  geom_point(data = weight_summary, aes(y = Mean), shape = 18, size = 3.3) +
  facet_grid(Domain ~ CV, labeller = labeller(Domain = c(DAP = "M10 DAP", AGDD = "M10 AGDD"))) +
  scale_color_manual(values = domain_colors) +
  scale_x_discrete(labels = c(Pooled = "Hybrid-pooled", Equal_Environment = "Equal-environment"),
                   expand = expansion(mult = 0.25)) +
  labs(title = "Does RMSE aggregation change the comparison?",
       subtitle = "Full TNP minus M10 | negative values favor TNP | shared seeds 1-5",
       x = NULL, y = "Difference in RMSE (t/ha)",
       caption = "Colored points and lines: paired seed means over five folds. Black diamonds and bars: mean +/- SD across five seeds.\nWithin each fold: hybrid-pooled = sqrt(sum(N * RMSE^2) / sum(N)); equal-environment = mean of 19 environment RMSEs.") +
  theme(legend.position = "none")
save_plot(wp, "Supplementary_04_RMSE_Weighting", 8.5, 6.5)
capture.output(sessionInfo(), file = file.path(out, "R_session.txt"))
if (from_source_data) {
  message("Wrote three supplementary figures from retained source data to ", out)
} else {
  message("Validated against final seed summary: max correlation error = ", signif(errors[1], 3),
          "; max RMSE error = ", signif(errors[2], 3), ". Wrote three supplementary figures to ", out)
}
