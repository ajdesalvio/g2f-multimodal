library(BGLR)
library(dplyr)
library(data.table)

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "06_prediction_loeo", "path_helpers.R"))
loeo_paths <- g2f_loeo_paths("agdd")

assert_single_predictor_bglr_patch <- function() {
  helper <- tryCatch(
    getFromNamespace("setLT.RKHS", "BGLR"),
    error = function(error) NULL
  )
  # Inspect the affected assignment, not unrelated drop=FALSE expressions.
  target_subsets <- list()
  inspect <- function(expr) {
    if (!is.call(expr)) return(invisible(NULL))
    if (
      is.symbol(expr[[1L]]) &&
      as.character(expr[[1L]]) %in% c("=", "<-") &&
      length(expr) == 3L && identical(expr[[2L]], quote(LT$V))
    ) {
      rhs <- expr[[3L]]
      if (
        is.call(rhs) && identical(rhs[[1L]], as.name("[")) &&
        length(rhs) >= 4L && identical(rhs[[2L]], quote(LT$V)) &&
        identical(rhs[[4L]], as.name("tmp"))
      ) {
        target_subsets[[length(target_subsets) + 1L]] <<- rhs
      }
    }
    for (index in seq_along(expr)[-1L]) {
      if (is.call(expr[[index]])) inspect(expr[[index]])
    }
    invisible(NULL)
  }
  if (is.function(helper)) inspect(body(helper))
  expected <- quote(LT$V[, tmp, drop = FALSE])
  patched <- length(target_subsets) > 0L &&
    all(vapply(target_subsets, identical, logical(1), y = expected))
  if (!patched) {
    stop(
      "The installed BGLR setLT.RKHS() does not contain the verified ",
      "LT$V[, tmp, drop = FALSE] patch. See patches/README.md; ",
      "use the patched package from the final HPRC environment."
    )
  }
  invisible(TRUE)
}

# Argument parsing
args <- commandArgs(trailingOnly = TRUE)
sanitize_arg <- function(x) {
  trimws(gsub("[\r\n]+", "", x, perl = TRUE))
}
args <- vapply(args, sanitize_arg, FUN.VALUE = character(1))
if (length(args) != 2L || !nzchar(args[1]) || !nzchar(args[2])) {
  stop("Expected exactly 2 arguments: model_stub heldout_env")
}

# Paths and manuscript defaults
data_path <- loeo_paths$data_path
kernel_path <- Sys.getenv(
  "G2F_MODEL_PATH",
  unset = file.path(loeo_paths$kernel_path, "noZe", "models")
)
out_path <- loeo_paths$results_path
wd_root <- loeo_paths$work_path
dir.create(out_path, recursive = TRUE, showWarnings = FALSE)
dir.create(wd_root, recursive = TRUE, showWarnings = FALSE)

if (g2f_bool(Sys.getenv("G2F_INCLUDE_ZE", unset = "FALSE"))) {
  stop("This manuscript launcher is restricted to the final no-Ze analysis.")
}
assert_single_predictor_bglr_patch()

# Leave-one-environment-out model testing - load required files
pheno_file <- file.path(data_path, "Pheno_Data.rds")
if (file.exists(pheno_file)) {
  pheno <- readRDS(pheno_file)
} else {
  pheno <- fread(file.path(data_path, "Yield_DTA_DTS_BLUEs_G2F_2020_2021.csv")) %>% as.data.frame()
  pheno$Pedigree.Env <- paste(pheno$Pedigree, pheno$Env, sep = ".")
}
model_stub <- args[1]
heldout.env <- args[2]
model_file <- file.path(kernel_path, paste0(model_stub, ".", heldout.env, ".rds"))

if (!file.exists(model_file)) {
  stop("Model file does not exist: ", model_file)
}

ETAX <- readRDS(model_file)

print(paste0('Model ', model_stub, ', Held-out env: ', heldout.env))

# Create an exclusive scratch directory for this array job
array_id <- Sys.getenv("SLURM_ARRAY_TASK_ID", unset = "local")  # defaults to "local" if run interactively
wd_job <- file.path(wd_root, paste(model_stub, heldout.env, array_id, Sys.getpid(), sep = "_"))
dir.create(wd_job, recursive = TRUE, showWarnings = FALSE)
message("Temporary working dir: ", wd_job)

# Set untested location as missing data
df <- pheno
df$Y2 <- df$Yield.t.ha.BLUE
df$Y2[pheno$Env == heldout.env] <- NA

# Ensure response is numeric
y_t <- as.numeric(df$Y2)

# Evaluate model
seed <- suppressWarnings(as.integer(Sys.getenv("G2F_SEED", unset = "2026")))
if (is.na(seed)) stop("G2F_SEED must be an integer.")
set.seed(seed + sum(utf8ToInt(paste(model_stub, heldout.env, sep = "."))))
n_iter <- suppressWarnings(as.integer(Sys.getenv("G2F_N_ITER", unset = "10000")))
burn_in <- suppressWarnings(as.integer(Sys.getenv("G2F_BURN_IN", unset = "1000")))
fit <- BGLR(y = y_t, ETA = ETAX, nIter = n_iter, burnIn = burn_in, saveAt = file.path(wd_job, 'BGLR_'))

# Save y-hat values
out <- data.frame(Pedigree.Env = df$Pedigree.Env,
                  Pedigree = df$Pedigree,
                  Env = df$Env,
                  Actual = df$Yield.t.ha.BLUE,
                  Predicted = fit$yHat)
out$Untested.Env <- heldout.env
out.cor <- out %>% filter(Env == heldout.env) %>%
  summarize(Cor = cor(Actual, Predicted, use = 'complete.obs')) %>%
  as.data.frame()
out.cor$Untested.Env <- heldout.env

# Save CSV files
save.name <- paste(model_stub, heldout.env, sep = '.')
fwrite(out, file.path(out_path, paste0(save.name, '.PredVals.csv')))
fwrite(out.cor, file.path(out_path, paste0(save.name, '.Cor.csv')))
