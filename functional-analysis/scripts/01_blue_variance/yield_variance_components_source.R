# Estimate yield variance components and entry-mean heritability.
# The historical BLUP output is intentionally omitted; only the variance table
# is consumed by the manuscript workflow.

suppressPackageStartupMessages({
  library(data.table)
  library(lme4)
  library(MuMIn)
})

source(file.path("R", "utils", "paths.R"))
paths <- g2f_paths()

input_files <- g2f_data_file(
  paths,
  c(
    "g2f_2020_phenotypic_clean_data.csv",
    "g2f_2021_phenotypic_clean_data.csv"
  )
)
invisible(lapply(input_files, g2f_require_file, description = "Phenotype input"))

drone_environments <- sort(c(
  "DEH1.2020", "IAH4.2021", "MIH1.2020", "MNH1.2020", "MNH1.2021",
  "MOH1.2020", "NEH1.2021", "TXH1.2020", "TXH1.2021", "TXH2.2020",
  "TXH2.2021", "TXH3.2020", "TXH3.2021", "WIH1.2020", "WIH1.2021",
  "WIH2.2020", "WIH2.2021", "WIH3.2020", "WIH3.2021"
))

df <- do.call(rbind, lapply(input_files, function(path) {
  as.data.frame(fread(path))
}))
setDT(df)
df[, `:=`(
  Yield.t.ha = as.numeric(`Grain Yield (bu/A)`) * 0.0673,
  Env = paste(`Field-Location`, Year, sep = "."),
  Range = factor(Range),
  Pass = factor(Pass),
  Replicate = factor(Replicate),
  Pedigree = factor(Pedigree)
)]
df <- df[Env %in% drone_environments]

variance_list <- lapply(drone_environments, function(environment) {
  message("Fitting yield variance model: ", environment)
  subset <- droplevels(as.data.frame(df[Env == environment]))
  if (nrow(subset) == 0L) stop("No phenotype rows found for ", environment)

  model <- lmer(
    Yield.t.ha ~ (1 | Pedigree) + (1 | Range) + (1 | Pass) +
      (1 | Replicate),
    data = subset
  )
  variance <- as.data.frame(VarCorr(model))
  variance$Percent <- round(variance$vcov / sum(variance$vcov) * 100, 2)
  variance$Rmse <- sqrt(mean(residuals(model)^2))
  variance$R_squared <- unname(MuMIn::r.squaredGLMM(model)[, "R2c"])

  vg <- variance$vcov[variance$grp == "Pedigree"]
  ve <- variance$vcov[variance$grp == "Residual"]
  n_rep <- length(unique(subset$Replicate))
  variance$Heritability <- round(vg / (vg + ve / n_rep), 3)
  variance$Env <- environment
  variance
})

variance <- rbindlist(variance_list, fill = TRUE)
variance[, c("var1", "var2") := NULL]
setcolorder(variance, c(
  "grp", "vcov", "sdcor", "Percent", "Rmse", "Heritability",
  "R_squared", "Env"
))
output_dir <- g2f_results_file(paths, "01_blue_variance")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_file <- file.path(output_dir, "VarComp_Yield_G2F_2020_2021.csv")
fwrite(variance, output_file)
message("Wrote yield variance components to ", output_file)
