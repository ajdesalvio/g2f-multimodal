# Estimate VI variance components and entry-mean heritability.
# The historical BLUP output is intentionally omitted; only the variance table
# is consumed by the manuscript workflow.

suppressPackageStartupMessages({
  library(data.table)
  library(lme4)
  library(MuMIn)
})

source(file.path("R", "utils", "paths.R"))
source(file.path("scripts", "01_blue_variance", "vi_analysis_helpers.R"))
paths <- g2f_paths()

prep_dir <- g2f_results_file(paths, "00_data_prep")
input_file <- file.path(prep_dir, "2020_2021_G2F_VIs_PHTs_COMBINED.csv")
if (!file.exists(input_file)) {
  input_file <- g2f_data_file(
    paths,
    "2020_2021_G2F_VIs_PHTs_COMBINED.csv"
  )
}
g2f_require_file(input_file, "Combined VI/PHT table")

prepared <- g2f_prepare_vi_data(as.data.frame(fread(input_file)))
df <- prepared$data
variance_list <- list()

for (env_dap in prepared$env_daps) {
  subset <- droplevels(df[df$Env.DAP == env_dap, , drop = FALSE])

  for (vi_name in g2f_vi_names) {
    message("Fitting VI variance model: ", env_dap, " / ", vi_name)
    subset$response <- subset[[vi_name]]
    subset$response[!is.finite(subset$response)] <- NA_real_

    model <- lmer(
      response ~ (1 | Pedigree) + (1 | Range) + (1 | Pass) +
        (1 | Replicate),
      data = subset,
      na.action = na.omit
    )

    variance <- as.data.frame(VarCorr(model))
    variance$Percent <- round(variance$vcov / sum(variance$vcov) * 100, 2)
    variance$Rmse <- sqrt(mean(residuals(model)^2))
    variance$R_squared <- unname(MuMIn::r.squaredGLMM(model)[, "R2c"])

    vg <- variance$vcov[variance$grp == "Pedigree"]
    ve <- variance$vcov[variance$grp == "Residual"]
    n_rep <- length(unique(subset$Replicate[!is.na(subset$response)]))
    variance$Heritability <- round(vg / (vg + ve / n_rep), 3)
    variance$Env.DAP <- env_dap
    variance$Vegetation.Index <- vi_name
    variance_list[[paste(env_dap, vi_name, sep = ".")]] <- variance
  }
}

variance <- rbindlist(variance_list, fill = TRUE)
variance[, c("var1", "var2") := NULL]
split_env_dap <- tstrsplit(variance$Env.DAP, ".", fixed = TRUE)
variance[, `:=`(
  DAP = split_env_dap[[3L]],
  Year = split_env_dap[[2L]],
  Env = paste(split_env_dap[[1L]], split_env_dap[[2L]], sep = ".")
)]
setcolorder(variance, c(
  "grp", "vcov", "sdcor", "Percent", "Rmse", "Heritability",
  "R_squared", "Env.DAP", "Vegetation.Index", "DAP", "Year", "Env"
))

output_dir <- g2f_results_file(paths, "01_blue_variance")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_file <- file.path(output_dir, "VarComp_G2F_2020_2021.csv")
fwrite(variance, output_file)
message("Wrote VI variance components to ", output_file)
