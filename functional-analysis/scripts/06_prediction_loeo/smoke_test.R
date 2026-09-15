# Lightweight DAP LOEO relationship-matrix integration check. Run from the
# functional-analysis after configuring paths, and generate projected scores first.

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "06_prediction_loeo", "path_helpers.R"))
defaults <- g2f_loeo_paths("dap")
source(file.path(g2f_prediction_root_dir, "scripts", "06_prediction_loeo", "relationship_matrix_helpers.R"))

environment_order <- read.csv(
  file.path(defaults$data_path, "Env_Names_G2F_2020_2021.csv")
)$Env
heldout_env <- Sys.getenv("G2F_SMOKE_HELDOUT_ENV", unset = environment_order[[1]])

bundle <- build_loeo_m8_m10_matrices(
  time_domain = "DAP",
  heldout_env = heldout_env,
  data_path = defaults$data_path,
  score_path = defaults$score_path,
  out_path = defaults$matrix_path,
  vi_names = "NGRDI",
  n_vi_fpcs = 5L,
  n_weather_fpcs = 1L,
  weather_traits = "PTR",
  write_outputs = FALSE,
  return_objects = TRUE,
  save_matrix_csv = FALSE,
  save_model_matrix_rds = FALSE,
  verbose = TRUE
)

expected_order <- read.csv(
  file.path(defaults$data_path, "G2F.2020.2021.Pedigrees.csv")
)$Pedigree.Env

stopifnot(
  identical(bundle$metadata$vi_names_requested, "NGRDI"),
  identical(bundle$metadata$weather_traits, "PTR"),
  all(vapply(
    bundle$relationship_matrices,
    function(x) identical(rownames(x), expected_order) && identical(colnames(x), expected_order),
    logical(1)
  ))
)

message("DAP LOEO smoke test passed.")
