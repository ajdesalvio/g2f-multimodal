# Estimate pointwise, single-environment VI BLUEs.

suppressPackageStartupMessages({
  library(data.table)
  library(lmerTest)
  library(MuMIn)
})

source(file.path("R", "utils", "paths.R"))
source(file.path("scripts", "01_blue_variance", "vi_analysis_helpers.R"))
paths <- g2f_paths()

prep_dir <- g2f_results_file(paths, "00_data_prep")
input_file <- file.path(
  prep_dir,
  "2020_2021_G2F_Data_JPGs_VIs_ALLDATA_V1.csv"
)
if (!file.exists(input_file)) {
  input_file <- g2f_data_file(
    paths,
    "2020_2021_G2F_Data_JPGs_VIs_ALLDATA_V1.csv"
  )
}
g2f_require_file(input_file, "Merged VI/PHT/image table")

prepared <- g2f_prepare_vi_data(as.data.frame(fread(input_file)))
df <- prepared$data

g_list <- list()
variance_list <- list()

for (env_dap in prepared$env_daps) {
  subset <- droplevels(df[df$Env.DAP == env_dap, , drop = FALSE])

  for (vi_name in g2f_vi_names) {
    message("Fitting VI BLUE model: ", env_dap, " / ", vi_name)
    subset$response <- subset[[vi_name]]
    subset$response[!is.finite(subset$response)] <- NA_real_

    model <- lmer(
      response ~ Pedigree + (1 | Range) + (1 | Pass) + (1 | Replicate),
      data = subset,
      na.action = na.omit
    )

    variance <- as.data.frame(VarCorr(model))
    variance$Percent <- round(variance$vcov / sum(variance$vcov) * 100, 2)
    variance$Rmse <- sqrt(mean(residuals(model)^2))
    variance$R_squared <- unname(MuMIn::r.squaredGLMM(model)[, "R2c"])
    variance$Env.DAP <- env_dap
    variance$Vegetation.Index <- vi_name
    variance_list[[paste(env_dap, vi_name, sep = ".")]] <- variance

    estimates <- as.data.frame(coef(summary(model)))
    estimates$Pedigree <- sub("^Pedigree", "", rownames(estimates))
    estimates$Pedigree[1L] <- levels(droplevels(subset$Pedigree))[1L]
    estimates$VI.BLUE <- estimates$Estimate
    if (nrow(estimates) > 1L) {
      estimates$VI.BLUE[-1L] <-
        estimates$Estimate[1L] + estimates$Estimate[-1L]
    }
    estimates$Env.DAP <- env_dap
    estimates$Vegetation.Index <- vi_name
    estimates$Env.DAP.Pedigree <- paste(
      env_dap,
      vi_name,
      rownames(estimates),
      sep = "."
    )
    g_list[[paste(env_dap, vi_name, sep = ".")]] <- estimates
  }
}

vi_blues <- rbindlist(g_list, fill = TRUE)
variance <- rbindlist(variance_list, fill = TRUE)
variance[, c("var1", "var2") := NULL]

split_env_dap <- tstrsplit(variance$Env.DAP, ".", fixed = TRUE)
variance[, `:=`(
  DAP = split_env_dap[[3L]],
  Year = split_env_dap[[2L]],
  Env = paste(split_env_dap[[1L]], split_env_dap[[2L]], sep = ".")
)]

split_model <- tstrsplit(vi_blues$Env.DAP, ".", fixed = TRUE)
vi_blues[, `:=`(
  DAP = split_model[[3L]],
  Year = split_model[[2L]],
  Env = paste(split_model[[1L]], split_model[[2L]], sep = ".")
)]

output_dir <- g2f_results_file(paths, "01_blue_variance")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
fwrite(
  variance,
  file.path(output_dir, "VarComp_BLUEs_G2F_2020_2021.csv")
)
fwrite(
  vi_blues,
  file.path(output_dir, "VI_BLUEs_G2F_2020_2021.csv")
)
message("Wrote VI BLUE and variance-component tables to ", output_dir)
