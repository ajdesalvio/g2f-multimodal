# Build the environment/DAP/flight-date source table for the drone-data summary.

suppressPackageStartupMessages(library(data.table))

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))
source(file.path(project_dir, "scripts", "01_blue_variance", "vi_analysis_helpers.R"))
paths <- g2f_paths()

input_file <- g2f_resolve_input(
  paths,
  "2020_2021_G2F_Data_JPGs_VIs_ALLDATA_V1.csv",
  result_subdirs = "00_data_prep"
)
output_dir <- g2f_results_dir(paths, "07_results_figures", "drone")

prepared <- g2f_prepare_vi_data(as.data.frame(fread(input_file)))
flight_dates <- as.data.table(prepared$data)[
  !is.na(Env) & !is.na(DAP) & !is.na(Flight.Date),
  .(Env, DAP, Flight.Date)
]
flight_dates <- unique(flight_dates)
setorder(flight_dates, Env, DAP, Flight.Date)

if (nrow(flight_dates) == 0L) stop("No flight-date/DAP combinations remain after filtering.")
if (flight_dates[, anyDuplicated(paste(Env, DAP, sep = "|"))] > 0L) {
  stop("At least one environment/DAP combination maps to multiple flight dates.")
}

fwrite(
  flight_dates,
  file.path(output_dir, "Flight.Date.DAP.Conversion.csv")
)
