# Configurable integration check for the AGDD CV preparation stages. This test
# writes only to the configured AGDD CV output directory.

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "05_prediction_cv", "path_helpers.R"))
cv_paths <- g2f_cv_paths("agdd")
script_dir <- file.path(g2f_prediction_root_dir, "scripts", "05_prediction_cv", "agdd")

Sys.setenv(
  G2F_DATA_PATH = cv_paths$data_path,
  G2F_CV_OUT_PATH = cv_paths$out_root,
  G2F_DERIVED_DATA_PATH = cv_paths$derived_path,
  G2F_INCLUDE_ZE = "FALSE",
  G2F_VI_NAMES = "NGRDI",
  G2F_WEATHER_TRAITS = "PTR"
)

source(file.path(script_dir, "preprocess_agdd_inputs.R"))
source(file.path(script_dir, "build_metadata.R"))
source(file.path(script_dir, "build_constant_kernels.R"))
source(file.path(script_dir, "build_weather_inputs.R"))

agdd <- run_agdd_preprocess(return_objects = TRUE, write_outputs = TRUE)
metadata <- run_cv_metadata(return_objects = TRUE, write_outputs = TRUE)
constant <- run_cv_constant_kernels(return_objects = TRUE, write_outputs = TRUE)
weather <- run_cv_weather_setup(return_objects = TRUE, write_outputs = TRUE)

stopifnot(
  nrow(metadata$female_folds) > 0L,
  length(constant$constant_bundle$order) > 0L,
  identical(weather$all_bundle$weather_traits, "PTR"),
  identical(toupper(Sys.getenv("G2F_INCLUDE_ZE")), "FALSE")
)

message("AGDD CV preparation smoke test passed.")
