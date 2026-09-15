library(fastmatrix)
library(BGLR)
library(dplyr)
library(data.table)
library(parallel)
library(tidyr)
library(fdapace)

#### SETUP ####
source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "06_prediction_loeo", "path_helpers.R"))
loeo_paths <- g2f_loeo_paths("dap")
data_path <- loeo_paths$data_path
out_path <- loeo_paths$score_path
dir.create(out_path, recursive = TRUE, showWarnings = FALSE)

# All environment names
envs <- read.csv(file.path(data_path, 'Env_Names_G2F_2020_2021.csv'))$Env
# Pedigree.Env order vector
ped.env.order <- read.csv(file.path(data_path, 'G2F.2020.2021.Pedigrees.csv'))$Pedigree.Env

# Organizational columns for VI and weather wide-format data
vi_org_cols <- c('Pedigree', 'Year', 'Env')
w_org_cols <- c('Env', 'Weather_Variable')

# VI BLUEs
VI <- fread(file.path(data_path, 'VI_BLUEs_G2F_2020_2021.csv')) %>%
  as.data.frame() %>%
  mutate(Pedigree.Env = paste(Pedigree, Env, sep = '.'))
# Weather
W <- fread(file.path(data_path, 'EnvRtype_Weather_Data_Cleaned_V2.csv')) %>%
  as.data.frame() %>%
  pivot_longer(cols = T2M:PAR_TEMP,
               names_to = 'Weather_Variable',
               values_to = 'Value') %>%
  select(Env, DAP, Weather_Variable, Value)
# NGRDI and PTR were selected a priori for the manuscript analysis.
vi_names <- trimws(strsplit(loeo_paths$vi_names, ",", fixed = TRUE)[[1]])
vi_names <- intersect(vi_names[nzchar(vi_names)], unique(VI$Vegetation.Index))
# All DAPs sorted for VIs and weather
dap.sorted <- sort(unique(VI$DAP))
w.dap.sorted <- sort(unique(W$DAP))
w_names <- trimws(strsplit(loeo_paths$weather_traits, ",", fixed = TRUE)[[1]])
w_names <- intersect(w_names[nzchar(w_names)], unique(W$Weather_Variable))

if (length(vi_names) == 0L) stop("None of the requested vegetation indices were found.")
if (length(w_names) == 0L) stop("None of the requested weather traits were found.")

#### FPCA ####
# Helper: restrict each (y,t) pair to [lo, hi] for cases where held-out env has longer
# time series than the training domain
clip_fpca_lists <- function(Ly, Lt, lo, hi) {
  out <- Map(function(y, t) {
    keep <- is.finite(y) & is.finite(t) & t >= lo & t <= hi
    list(y = y[keep], t = t[keep])
  }, Ly, Lt)
  
  list(
    Ly = lapply(out, `[[`, "y"),
    Lt = lapply(out, `[[`, "t")
  )
}

##### Vegetation index FPCA #####
VI_FVE_list <- list()
VI_scores_list <- list()

for (vi_i in vi_names) {
  # Filter and sort using Pedigree.Env order
  VI_i <- VI %>% filter(Vegetation.Index == vi_i, Pedigree.Env %in% ped.env.order) %>%
    arrange(match(Pedigree.Env, ped.env.order)) %>%
    pivot_wider(
      id_cols   = c(Pedigree, Year, Env),
      names_from  = DAP,
      values_from = 'VI.BLUE',
      names_glue  = "VI.BLUE.{DAP}",
      values_fill = NA
    ) %>%
    as.data.frame()
  
  # Ensure proper ordering of DAPs for the MakeFPCAInputs function
  VI_i.dap <- paste0('VI.BLUE.', dap.sorted)
  VI_i <- VI_i[,c(vi_org_cols, VI_i.dap)]
  rownames(VI_i) <- paste(VI_i$Pedigree, VI_i$Env, sep = '.')
  
  for (env_i in envs) {
    # Index for lists
    vi_env_i <- paste(vi_i, env_i, sep = '.')
    message(paste('Current VI and held-out env:', vi_env_i))
    # FPCA input preparation
    VI_i_fpca <- MakeFPCAInputs(IDs = rep(rownames(VI_i), each = length(dap.sorted)),
                                tVec = rep(dap.sorted, length(rownames(VI_i))),
                                t(VI_i[,!colnames(VI_i) %in% vi_org_cols]))
    # Pedigree.Env values in training and held-out
    ped.env.h <- rownames(VI_i)[VI_i$Env == env_i]
    ped.env.t <- rownames(VI_i)[VI_i$Env != env_i]
    # Split the MakeFPCAInputs object
    idx.t <- VI_i_fpca$Lid %in% ped.env.t
    idx.h <- VI_i_fpca$Lid %in% ped.env.h
    # FPCA with only the training data
    fpca_obj <- tryCatch({
      FPCA(VI_i_fpca$Ly[idx.t], VI_i_fpca$Lt[idx.t],
           list(dataType = "Sparse",
                methodMuCovEst = "smooth",
                methodBwCov = "GCV",
                methodBwMu = "GCV",
                plot = FALSE))
    }, error = function(e) {
      message(sprintf("FPCA failed for %s and %s: %s", vi_i, env_i, e$message))
      return(NULL)
    })
    if (is.null(fpca_obj)) next
    # Extract scores from training data
    scores.t <- as.data.frame(fpca_obj$xiEst)
    rownames(scores.t) <- ped.env.t
    colnames(scores.t) <- paste0('FPC', 1:ncol(scores.t), '.', vi_i)
    # Extract FVE from training data
    nFPCs <- ncol(scores.t)
    fve.t <- fpca_obj$cumFVE
    nFVE <- min(length(fve.t), nFPCs)
    # Store FVE in a vector
    fpca_variation <- numeric(nFVE)
    if(nFVE >= 1) fpca_variation[1] <- fve.t[1]
    if(nFVE > 1){for(i in 2:nFVE){fpca_variation[i] <- fve.t[i] - fve.t[i-1]}}
    # Save FVE in list
    VI_FVE_list[[vi_env_i]] <- fpca_variation
    # Project held-out curves onto training FPCA bases
    lo <- min(fpca_obj$workGrid)
    hi <- max(fpca_obj$workGrid)
    held <- clip_fpca_lists(VI_i_fpca$Ly[idx.h], VI_i_fpca$Lt[idx.h], lo, hi)
    pred <- predict(fpca_obj, newLy = held$Ly, newLt = held$Lt)
    scores.h <- as.data.frame(pred$scores)
    rownames(scores.h) <- ped.env.h
    colnames(scores.h) <- paste0('FPC', 1:ncol(scores.h), '.', vi_i)
    # Join the training and predicted scores and sort the data frame
    scores.t$Pedigree.Env <- rownames(scores.t)
    scores.h$Pedigree.Env <- rownames(scores.h)
    scores <- bind_rows(scores.t, scores.h) %>%
      relocate(Pedigree.Env) %>%
      arrange(match(Pedigree.Env, ped.env.order))
    # Save scores in list
    VI_scores_list[[vi_env_i]] <- scores
  }
}

##### Weather FPCA #####
w_FVE_list <- list()
w_scores_list <- list()

for (w_i in w_names) {
  # Filter and sort using Env order
  W_i <- W %>%
    filter(Weather_Variable == w_i) %>%
    pivot_wider(
      names_from  = DAP,
      values_from = Value,
      names_glue  = "{w_i}.{DAP}",
      values_fill = NA_real_
    ) %>%
    as.data.frame()
  
  # Ensure proper ordering of DAPs for the MakeFPCAInputs function
  W_i.dap <- paste(w_i, w.dap.sorted, sep = '.')
  W_i <- W_i[,c(w_org_cols, W_i.dap)]
  rownames(W_i) <- W_i$Env
  
  for (env_i in envs) {
    # Index for lists
    w_env_i <- paste(w_i, env_i, sep = '.')
    message(paste('Current held-out env and weather parameter:', w_i, env_i))
    # FPCA input preparation
    W_i_fpca <- MakeFPCAInputs(IDs = rep(rownames(W_i), each = length(w.dap.sorted)),
                                tVec = rep(w.dap.sorted, length(rownames(W_i))),
                                t(W_i[,!colnames(W_i) %in% w_org_cols]))
    # Env values in training and held-out
    env.t <- envs[envs != env_i]
    env.h <- envs[envs == env_i]
    # Split the MakeFPCAInputs object
    idx.t <- W_i_fpca$Lid %in% env.t
    idx.h <- W_i_fpca$Lid %in% env.h
    # FPCA with only the training data
    fpca_obj <- tryCatch({
      FPCA(W_i_fpca$Ly[idx.t], W_i_fpca$Lt[idx.t],
           list(dataType = "Sparse",
                methodMuCovEst = "smooth",
                methodBwCov = "GCV",
                methodBwMu = "GCV",
                plot = FALSE))
    }, error = function(e) {
      message(sprintf("FPCA failed for %s and %s: %s", w_i, env_i, e$message))
      return(NULL)
    })
    if (is.null(fpca_obj)) next
    # Extract scores from training data
    scores.t <- as.data.frame(fpca_obj$xiEst)
    rownames(scores.t) <- env.t
    colnames(scores.t) <- paste0('FPC', 1:ncol(scores.t), '.', w_i)
    # Extract FVE from training data
    nFPCs <- ncol(scores.t)
    fve.t <- fpca_obj$cumFVE
    nFVE <- min(length(fve.t), nFPCs)
    # Store FVE in a vector
    fpca_variation <- numeric(nFVE)
    if(nFVE >= 1) fpca_variation[1] <- fve.t[1]
    if(nFVE > 1){for(i in 2:nFVE){fpca_variation[i] <- fve.t[i] - fve.t[i-1]}}
    # Save FVE in list
    w_FVE_list[[w_env_i]] <- fpca_variation
    # Project held-out curves onto training FPCA bases
    lo <- min(fpca_obj$workGrid)
    hi <- max(fpca_obj$workGrid)
    held <- clip_fpca_lists(W_i_fpca$Ly[idx.h], W_i_fpca$Lt[idx.h], lo, hi)
    pred <- predict(fpca_obj, newLy = held$Ly, newLt = held$Lt)
    scores.h <- as.data.frame(pred$scores)
    rownames(scores.h) <- env.h
    colnames(scores.h) <- paste0('FPC', 1:ncol(scores.h), '.', w_i)
    # Join the training and predicted scores and sort the data frame
    scores.t$Env <- rownames(scores.t)
    scores.h$Env <- rownames(scores.h)
    scores <- bind_rows(scores.t, scores.h) %>%
      relocate(Env) %>%
      arrange(match(Env, envs))
    # Save scores in list
    w_scores_list[[w_env_i]] <- scores
  }
}

# Save VI FPC scores
VI_scores_list_revised <- list()
for (name in names(VI_scores_list)) {
  VI_scores_i <- VI_scores_list[[name]]
  vi_name_i <- strsplit(name, split = '\\.')[[1]][1]
  heldout_env_i <- paste(strsplit(name, split = '\\.')[[1]][2], strsplit(name, split = '\\.')[[1]][3], sep = '.')
  VI_scores_i$Vegetation.Index <- vi_name_i
  VI_scores_i$Heldout.Environment <- heldout_env_i
  VI_scores_list_revised[[name]] <- VI_scores_i %>% relocate(Pedigree.Env, Vegetation.Index, Heldout.Environment)
}

VI_scores_df <- rbindlist(VI_scores_list_revised, fill = TRUE)
fwrite(VI_scores_df, file.path(out_path, 'FPC_Scores_BLUEs_DAP_LOEO_Projected.csv'))

# Save Weather FPC scores
w_scores_list_revised <- list()
for (name in names(w_scores_list)) {
  w_scores_i <- w_scores_list[[name]]
  w_name_i <- strsplit(name, split = '\\.')[[1]][1]
  heldout_env_i <- paste(strsplit(name, split = '\\.')[[1]][2], strsplit(name, split = '\\.')[[1]][3], sep = '.')
  w_scores_i$Weather.Variable <- w_name_i
  w_scores_i$Heldout.Environment <- heldout_env_i
  w_scores_list_revised[[name]] <- w_scores_i %>% relocate(Env, Weather.Variable, Heldout.Environment)
}

w_scores_df <- rbindlist(w_scores_list_revised, fill = TRUE)
fwrite(w_scores_df, file.path(out_path, 'Weather_FPC_Scores_DAP_LOEO_Projected.csv'))
