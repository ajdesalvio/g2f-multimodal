# Estimate DTA and DTS BLUEs, then combine them with the grain-yield BLUEs.

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
df[, `:=`(
  DTA = as.numeric(`Anthesis [days]`),
  DTS = as.numeric(`Silking [days]`),
  Env = paste(`Field-Location`, Year, sep = ".")
)]
df <- df[Env %in% drone_environments]
df[, `:=`(
  Env = factor(Env, levels = drone_environments),
  Range = factor(Range),
  Pass = factor(Pass),
  Replicate = factor(Replicate),
  Pedigree = factor(Pedigree)
)]

estimate_trait <- function(trait) {
  fits <- lapply(drone_environments, function(environment) {
    message("Fitting ", trait, " BLUE model: ", environment)
    subset <- droplevels(as.data.frame(df[Env == environment]))
    if (nrow(subset) == 0L) stop("No phenotype rows found for ", environment)

    model <- lmer(
      reformulate(
        c("Pedigree", "(1 | Range)", "(1 | Pass)", "(1 | Replicate)"),
        response = trait
      ),
      data = subset
    )

    variance <- as.data.frame(VarCorr(model))
    variance$Percent <- round(variance$vcov / sum(variance$vcov) * 100, 2)
    variance$Rmse <- sqrt(mean(residuals(model)^2))
    variance$R_squared <- unname(MuMIn::r.squaredGLMM(model)[, "R2c"])
    variance$Env <- environment
    variance$Trait <- trait

    estimates <- as.data.frame(coef(summary(model)))
    estimates$Pedigree <- sub("^Pedigree", "", rownames(estimates))
    estimates$Pedigree[1L] <- levels(subset$Pedigree)[1L]
    estimates$BLUE <- estimates$Estimate
    if (nrow(estimates) > 1L) {
      estimates$BLUE[-1L] <- estimates$Estimate[1L] + estimates$Estimate[-1L]
    }
    names(estimates)[names(estimates) == "BLUE"] <- paste0(trait, ".BLUE")
    estimates$Env <- environment
    estimates$Pedigree.Env <- paste(estimates$Pedigree, environment, sep = ".")

    list(estimates = estimates, variance = variance)
  })

  list(
    estimates = rbindlist(lapply(fits, `[[`, "estimates"), fill = TRUE),
    variance = rbindlist(lapply(fits, `[[`, "variance"), fill = TRUE)
  )
}

dta <- estimate_trait("DTA")
dts <- estimate_trait("DTS")

dta_file <- file.path(output_dir, "DTA_BLUEs_G2F_2020_2021.csv")
dts_file <- file.path(output_dir, "DTS_BLUEs_G2F_2020_2021.csv")
fwrite(dta$estimates, dta_file)
fwrite(dts$estimates, dts_file)
fwrite(
  dta$variance,
  file.path(output_dir, "DTA_VarComp_BLUEs_G2F_2020_2021.csv")
)
fwrite(
  dts$variance,
  file.path(output_dir, "DTS_VarComp_BLUEs_G2F_2020_2021.csv")
)

yield_file <- file.path(
  output_dir,
  "Single_Env_Yield_BLUEs_G2F_2020_2021.csv"
)
if (!file.exists(yield_file)) {
  yield_file <- g2f_data_file(
    paths,
    "Single_Env_Yield_BLUEs_G2F_2020_2021.csv"
  )
}
g2f_require_file(yield_file, "Yield BLUEs")

rename_statistics <- function(table, suffix) {
  statistic_columns <- names(table)[1:5]
  setnames(
    table,
    statistic_columns,
    c(
      paste0("Estimate.BLUE.", suffix),
      paste0("Std.Err.", suffix),
      paste0("df.", suffix),
      paste0("t.val.", suffix),
      paste0("Pr.t.", suffix)
    )
  )
  table
}

yield <- rename_statistics(fread(yield_file), "Yield")
dta_estimates <- rename_statistics(copy(dta$estimates), "DTA")
dts_estimates <- rename_statistics(copy(dts$estimates), "DTS")

join_keys <- c("Pedigree", "Env", "Pedigree.Env")
combined <- merge(yield, dta_estimates, by = join_keys, all = TRUE, sort = FALSE)
combined <- merge(combined, dts_estimates, by = join_keys, all = TRUE, sort = FALSE)

setcolorder(combined, c(
  "Estimate.BLUE.Yield", "Std.Err.Yield", "df.Yield", "t.val.Yield",
  "Pr.t.Yield", "Pedigree", "Yield.t.ha.BLUE", "Env", "Pedigree.Env",
  "Estimate.BLUE.DTA", "Std.Err.DTA", "df.DTA", "t.val.DTA", "Pr.t.DTA",
  "DTA.BLUE", "Estimate.BLUE.DTS", "Std.Err.DTS", "df.DTS", "t.val.DTS",
  "Pr.t.DTS", "DTS.BLUE"
))
combined[, ASI := DTA.BLUE - DTS.BLUE]
combined[, ASI.Abs := abs(ASI)]
setorder(combined, Env, Pedigree)

combined_file <- file.path(
  output_dir,
  "Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv"
)
fwrite(combined, combined_file)
message("Wrote combined yield/DTA/DTS BLUEs to ", combined_file)
