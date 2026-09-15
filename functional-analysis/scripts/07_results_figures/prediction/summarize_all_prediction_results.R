suppressPackageStartupMessages({
  library(data.table)
  library(readxl)
})


# Portable adaptation of the finalized V6 results compilation.
project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
source(file.path(project_dir, "R", "utils", "prediction_results.R"))
project_paths <- g2f_paths()
paths <- g2f_prediction_result_paths(project_paths, g2f_analysis_config())
out_dir <- g2f_results_dir(project_paths, "07_results_figures", "final_prediction_summaries")
required <- c("loeo_xlsx", "loeo_dap", "loeo_agdd", "cv0_dap", "cv0_agdd",
              "cv21_dap", "cv21_agdd", "tnp")
missing <- required[!file.exists(unlist(paths[required]))]
if (length(missing)) stop("Missing final prediction inputs: ", paste(missing, collapse = ", "))
env_file <- g2f_resolve_input(project_paths, "Env_Names_G2F_2020_2021.csv")

tiezzi <- function(r, n) {
  keep <- is.finite(r) & is.finite(n) & n > 2
  r <- r[keep]
  n <- n[keep]
  if (!length(r)) return(NA_real_)
  v <- pmax((1 - r^2) / (n - 2), .Machine$double.eps)
  weighted.mean(r, 1 / v)
}

pool_rmse <- function(rmse, n) {
  keep <- is.finite(rmse) & is.finite(n) & n > 0
  if (!any(keep)) return(NA_real_)
  sqrt(weighted.mean(rmse[keep]^2, n[keep]))
}

model_key <- c(
  "M1.G" = "M1.G", "M1.P" = "M1.P", "M2.G" = "M2.G",
  "M3.G" = "M3.G-Int", "M3.P" = "M3.P-Int", "M4.G" = "M4.G.W",
  "M4.P" = "M4.P.W", "M5.G" = "M5.G.W-Int", "M5.P" = "M5.P.W-Int",
  "M6.G.P" = "M6.G.P", "M7.G.P" = "M7.G.P", "M8.G.P" = "M8.G.P-Int",
  "M9.G.P" = "M9.G.P.W", "M10.G.P" = "M10.G.P.W-Int"
)

# LOEO
read_loeo_rmse <- function(path, domain) {
  files <- list.files(path, pattern = "PredVals\\.csv$", full.names = TRUE)
  stopifnot(length(files) == 266L)
  expected_envs <- fread(env_file)$Env
  stopifnot(length(expected_envs) == 19L, !anyNA(expected_envs), !anyDuplicated(expected_envs))
  x <- rbindlist(lapply(files, function(file) {
    z <- fread(file, select = c("Pedigree.Env", "Env", "Actual", "Predicted", "Untested.Env"))
    heldout_env <- unique(z$Untested.Env)
    stopifnot(length(heldout_env) == 1L, heldout_env %in% expected_envs)
    # Prediction files contain training rows too; only this run's held-out
    # environment contributes to LOEO error and its pooling denominator.
    z <- z[Env == heldout_env & complete.cases(Actual, Predicted)]
    stopifnot(nrow(z) > 0L, !anyNA(z$Pedigree.Env), !anyDuplicated(z$Pedigree.Env),
              all(is.finite(z$Actual)), all(is.finite(z$Predicted)))
    suffix <- paste0(".", heldout_env, ".PredVals.csv")
    stopifnot(endsWith(basename(file), suffix))
    model <- substr(basename(file), 1L, nchar(basename(file)) - nchar(suffix))
    data.table(Model = unname(model_key[model]), Time_Domain = domain, Environment = heldout_env,
               N = nrow(z), RMSE = sqrt(mean((z$Actual - z$Predicted)^2)))
  }))
  stopifnot(!anyNA(x$Model), !anyDuplicated(x[, .(Model, Environment)]),
            all(x[, .N, by = Model]$N == 19L),
            all(x[, .(complete = setequal(Environment, expected_envs)), by = Model]$complete))
  x[, .(RMSE = pool_rmse(RMSE, N)), by = .(Model, Time_Domain)]
}

loeo <- as.data.table(read_excel(paths$loeo_xlsx))[
  , .(Correlation = tiezzi(Cor, n_hybrids), Environments = .N),
  by = .(Model = ModelW, Time_Domain = FPCA_Type)
][, `:=`(Method = "BGLR", CV = "LOEO", Correlation_SD = NA_real_, RMSE_SD = NA_real_, Seeds = NA_integer_)]

loeo_rmse <- rbindlist(list(read_loeo_rmse(paths$loeo_dap, "DAP"), read_loeo_rmse(paths$loeo_agdd, "AGDD")))
loeo[loeo_rmse, RMSE := i.RMSE, on = .(Model, Time_Domain)]
stopifnot(!anyNA(loeo$RMSE))

# CV0/CV00
read_cv000 <- function(path, domain) {
  files <- list.files(path, pattern = "CV_0_00.*noZe\\.metrics\\.csv$", full.names = TRUE)
  stopifnot(length(files) == 950L)
  x <- rbindlist(lapply(files, fread))[
    Status == "ok" & !Include_Ze & Metric %chin% c("CV0", "CV00"),
    .(Method = "BGLR", Model = unname(model_key[Model_Name]), Time_Domain = domain,
      CV = Metric, Seed = Seed_Num, Fold = Fold_Num, Environment = Heldout_Env,
      N, Correlation = Cor, RMSE)
  ]
  stopifnot(nrow(x) == 26600L, !anyNA(x$Model))
  x
}

cv000 <- rbindlist(list(read_cv000(paths$cv0_dap, "DAP"), read_cv000(paths$cv0_agdd, "AGDD")))
stopifnot(uniqueN(cv000, by = c("Model", "Time_Domain", "CV", "Seed", "Fold", "Environment")) == nrow(cv000))

bglr_cv0_fold <- cv000[, .(
  Correlation = tiezzi(Correlation, N), RMSE = pool_rmse(RMSE, N),
  Environments = uniqueN(Environment)
), by = .(Method, Model, Time_Domain, CV, Seed, Fold)]
stopifnot(all(bglr_cv0_fold$Environments == 19L))

bglr_cv0_seed <- bglr_cv0_fold[, .(
  Correlation = mean(Correlation), RMSE = mean(RMSE)
), by = .(Method, Model, Time_Domain, CV, Seed)]

bglr_cv0 <- bglr_cv0_seed[, .(
  Correlation = mean(Correlation), Correlation_SD = sd(Correlation),
  RMSE = mean(RMSE), RMSE_SD = sd(RMSE), Seeds = .N, Environments = 19L
), by = .(Method, Model, Time_Domain, CV)]

# CV1/CV2
read_cv21 <- function(file, domain) {
  m <- regexec("^Seed([0-9]+)\\.Fold([0-9]+)\\.CV_2_1\\.None\\.noZe\\.(.+)\\.prediction_values\\.csv$", basename(file))
  id <- regmatches(basename(file), m)[[1]]
  stopifnot(length(id) == 4L)
  seed <- as.integer(id[2])
  fold <- as.integer(id[3])
  model <- id[4]
  x <- fread(file, select = c("Env", "Fold", "Actual", "Predicted"))
  x <- x[complete.cases(Actual, Predicted, Fold, Env)]
  rbindlist(lapply(c("CV1", "CV2"), function(cv_label) {
    z <- if (cv_label == "CV1") x[Fold == fold] else x[Fold != fold]
    env <- z[, .(N = .N, r = cor(Actual, Predicted)), by = Env]
    data.table(
      Method = "BGLR", Model = unname(model_key[model]), Time_Domain = domain,
      CV = cv_label, Seed = seed, Fold = fold, Correlation = tiezzi(env$r, env$N),
      Pooled_Pearson = cor(z$Actual, z$Predicted), RMSE = sqrt(mean((z$Actual - z$Predicted)^2)),
      N = nrow(z), Environments = nrow(env)
    )
  }))
}

read_cv21_domain <- function(path, domain) {
  files <- list.files(path, pattern = "prediction_values\\.csv$", full.names = TRUE)
  stopifnot(length(files) == 700L)
  rbindlist(lapply(files, read_cv21, domain = domain))
}

bglr_cv21_fold <- rbindlist(list(
  read_cv21_domain(paths$cv21_dap, "DAP"),
  read_cv21_domain(paths$cv21_agdd, "AGDD")
))
stopifnot(nrow(bglr_cv21_fold) == 2800L, !anyNA(bglr_cv21_fold$Model), all(bglr_cv21_fold$Environments == 19L))

bglr_cv21_seed <- bglr_cv21_fold[, .(
  Correlation = mean(Correlation), Pooled_Pearson = mean(Pooled_Pearson), RMSE = mean(RMSE)
), by = .(Method, Model, Time_Domain, CV, Seed)]

bglr_cv21 <- bglr_cv21_seed[, .(
  Correlation = mean(Correlation), Correlation_SD = sd(Correlation),
  RMSE = mean(RMSE), RMSE_SD = sd(RMSE), Seeds = .N, Environments = 19L
), by = .(Method, Model, Time_Domain, CV)]

# TNP
tnp_names <- c("istnp-full" = "VI+genomic+weather", "weather-free" = "VI+genomic", "vi-only" = "VI-only")
tnp <- fread(paths$tnp)
tnp[, `:=`(Method = "TNP", Model = unname(tnp_names[model]), Time_Domain = "DAP", Fold = as.integer(sub("^Fold([0-9]+).*$", "\\1", fold)))]

tnp_cv0_fold <- tnp[cv_label %chin% c("CV0", "CV00"), .(
  Correlation = tiezzi(pearson_r, n), RMSE = pool_rmse(rmse, n), Environments = .N
), by = .(Method, Model, Time_Domain, CV = cv_label, checkpoint, Seed = seed, Fold)]

tnp_cv21_fold <- tnp[cv_label %chin% c("CV1", "CV2"), .(
  Correlation = r_w, Pooled_Pearson = pearson_r, RMSE = rmse, Environments = r_w_n_blocks
), by = .(Method, Model, Time_Domain, CV = cv_label, checkpoint, Seed = seed, Fold)]

tnp_fold <- rbindlist(list(tnp_cv0_fold, tnp_cv21_fold), fill = TRUE)
tnp_seed <- tnp_fold[, .(
  Correlation = mean(Correlation), Pooled_Pearson = mean(Pooled_Pearson, na.rm = TRUE), RMSE = mean(RMSE)
), by = .(Method, Model, Time_Domain, CV, checkpoint, Seed)]
tnp_seed[is.nan(Pooled_Pearson), Pooled_Pearson := NA_real_]

tnp_all <- tnp_seed[, .(
  Correlation = mean(Correlation), Correlation_SD = sd(Correlation),
  RMSE = mean(RMSE), RMSE_SD = sd(RMSE), Seeds = .N, Environments = 19L
), by = .(Method, Model, Time_Domain, CV, checkpoint)]

primary_checkpoint <- paths$primary_tnp_checkpoint
tnp_primary <- tnp_all[checkpoint == primary_checkpoint]

summary <- rbindlist(list(loeo, bglr_cv0, bglr_cv21, tnp_primary), fill = TRUE)
setnames(summary, "checkpoint", "Checkpoint")
setcolorder(summary, c("Method", "Model", "Time_Domain", "CV", "Checkpoint", "Correlation", "Correlation_SD", "RMSE", "RMSE_SD", "Seeds", "Environments"))
setorder(summary, Method, Time_Domain, Model, CV)

seed_summary <- rbindlist(list(
  bglr_cv0_seed,
  bglr_cv21_seed[, .(Method, Model, Time_Domain, CV, Seed, Correlation, RMSE)],
  tnp_seed[checkpoint == primary_checkpoint, .(Method, Model, Time_Domain, CV, Seed, Correlation, RMSE)]
), fill = TRUE)
setorder(seed_summary, Method, Time_Domain, Model, CV, Seed)

pooled_sensitivity <- rbindlist(list(
  bglr_cv21_seed[, .(Method, Model, Time_Domain, CV, Seed, Within_Environment_r = Correlation, Pooled_Pearson)],
  tnp_seed[checkpoint == primary_checkpoint & CV %chin% c("CV1", "CV2"),
           .(Method, Model, Time_Domain, CV, Seed, Within_Environment_r = Correlation, Pooled_Pearson)]
))
setorder(pooled_sensitivity, Method, Time_Domain, Model, CV, Seed)

fwrite(summary, file.path(out_dir, "G2F_Final_Prediction_Summary.csv"))
fwrite(seed_summary, file.path(out_dir, "G2F_Final_Prediction_Seed_Summary.csv"))
fwrite(pooled_sensitivity, file.path(out_dir, "G2F_CV1_CV2_Pooled_Pearson_Sensitivity.csv"))
fwrite(tnp_all, file.path(out_dir, "G2F_TNP_Checkpoint_Sensitivity.csv"))

cat("Wrote", nrow(summary), "primary summaries to", out_dir, "\n")
