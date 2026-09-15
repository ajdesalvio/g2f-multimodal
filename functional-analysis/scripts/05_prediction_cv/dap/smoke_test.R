# Lightweight integration check for the DAP CV relationship-matrix workflow.
# Run from functional-analysis after configuring config/paths.yml, or set
# G2F_PROJECT_DIR, G2F_DATA_PATH, and G2F_CV_OUT_PATH explicitly.

source(file.path(Sys.getenv("G2F_PROJECT_DIR", unset = getwd()), "scripts", "05_prediction_cv", "path_helpers.R"))
cv_paths <- g2f_cv_paths("dap")
source(file.path(g2f_prediction_root_dir, "scripts", "05_prediction_cv", "dap", "relationship_matrix_helpers.R"))

bundle <- build_dap_cv_m8_m10_matrices(
  seed_num = 1L,
  fold_num = 1L,
  split_group = "CV_2_1",
  heldout_env = "None",
  data_path = cv_paths$data_path,
  out_root = cv_paths$out_root,
  pipeline_dir = file.path(g2f_prediction_root_dir, "scripts", "05_prediction_cv", "dap"),
  vi_names = "NGRDI",
  n_vi_fpcs = 5L,
  compute_eigs = FALSE,
  write_outputs = FALSE,
  return_objects = TRUE
)

expected_order <- read.csv(
  file.path(cv_paths$data_path, "G2F.2020.2021.Pedigrees.csv")
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

message("DAP CV smoke test passed.")
