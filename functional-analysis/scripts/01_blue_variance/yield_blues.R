# Estimate single-environment grain-yield BLUEs for the 2020/2021 trials.

suppressPackageStartupMessages({
  library(data.table)
  library(lmerTest)
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

output_dir <- g2f_results_file(paths, "01_blue_variance")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

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
df[, Yield.t.ha := as.numeric(`Grain Yield (bu/A)`) * 0.0673]
df[, Env := paste(`Field-Location`, Year, sep = ".")]
df <- df[Env %in% drone_environments]
df[, `:=`(
  Env = factor(Env, levels = drone_environments),
  Range = factor(Range),
  Pass = factor(Pass),
  Replicate = factor(Replicate),
  Pedigree = factor(Pedigree)
)]

estimate_environment <- function(environment) {
  subset <- droplevels(as.data.frame(df[Env == environment]))
  if (nrow(subset) == 0L) stop("No phenotype rows found for ", environment)

  model <- lmer(
    Yield.t.ha ~ Pedigree + (1 | Range) + (1 | Pass) + (1 | Replicate),
    data = subset
  )

  variance <- as.data.frame(VarCorr(model))
  variance$Percent <- round(variance$vcov / sum(variance$vcov) * 100, 2)
  variance$Rmse <- sqrt(mean(residuals(model)^2))
  variance$R_squared <- unname(MuMIn::r.squaredGLMM(model)[, "R2c"])
  variance$Env <- environment

  estimates <- as.data.frame(coef(summary(model)))
  estimates$Pedigree <- sub("^Pedigree", "", rownames(estimates))
  estimates$Pedigree[1L] <- levels(subset$Pedigree)[1L]
  estimates$Yield.t.ha.BLUE <- estimates$Estimate
  if (nrow(estimates) > 1L) {
    estimates$Yield.t.ha.BLUE[-1L] <-
      estimates$Estimate[1L] + estimates$Estimate[-1L]
  }
  estimates$Env <- environment
  estimates$Pedigree.Env <- paste(estimates$Pedigree, environment, sep = ".")

  list(estimates = estimates, variance = variance)
}

fits <- lapply(drone_environments, function(environment) {
  message("Fitting yield BLUE model: ", environment)
  estimate_environment(environment)
})

yield_blues <- rbindlist(lapply(fits, `[[`, "estimates"), fill = TRUE)
yield_variance <- rbindlist(lapply(fits, `[[`, "variance"), fill = TRUE)
yield_variance[, c("var1", "var2") := NULL]

blue_file <- file.path(
  output_dir,
  "Single_Env_Yield_BLUEs_G2F_2020_2021.csv"
)
variance_file <- file.path(
  output_dir,
  "Yield_BLUEs_VarComp_G2F_2020_2021.csv"
)
fwrite(yield_blues, blue_file)
fwrite(yield_variance, variance_file)
message("Wrote yield BLUEs to ", blue_file)
