# Optional, unaggregated export of finalized CV0/CV00 correlations and RMSE.
# The primary summary reads the task-level inputs directly; this export is
# not a required intermediate step.
suppressPackageStartupMessages(library(data.table))
project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
source(file.path(project_dir, "R", "utils", "prediction_results.R"))
paths <- g2f_paths()
inputs <- g2f_prediction_result_paths(paths, g2f_analysis_config())

read_domain <- function(directory, domain) {
  x <- g2f_read_csv_directory(directory,
    pattern = "CV_0_00.*[.]noZe[.]metrics[.]csv$", expected_count = 950L)
  g2f_assert_no_ze(x)
  x <- x[Metric %chin% c("CV0", "CV00")]
  stopifnot(nrow(x) == 26600L, all(x$Status == "ok"))
  x[, `:=`(Time_Domain = domain, Model = unname(g2f_model_labels()[Model_Name]))]
  stopifnot(!anyNA(x$Model), !anyNA(x$N), !anyNA(x$RMSE),
    !anyDuplicated(x[, .(Seed_Num, Fold_Num, Heldout_Env, Model_Name, Metric)]))
  x[, .(Time_Domain, Model, CV = Metric, Seed = Seed_Num, Fold = Fold_Num,
        Environment = Heldout_Env, N, Correlation = Cor, RMSE)]
}
x <- rbindlist(list(read_domain(inputs$cv0_dap, "DAP"),
                    read_domain(inputs$cv0_agdd, "AGDD")))
setorder(x, Time_Domain, Model, CV, Seed, Fold, Environment)
out <- g2f_results_dir(paths, "07_results_figures", "prediction_inputs")
fwrite(x, file.path(out, "G2F_CV_Environment_Metrics.csv"))
