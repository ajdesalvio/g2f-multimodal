# Convert the MNH1.2020 monthly station workbook to the common weather schema.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(readxl)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))

paths <- g2f_paths()
input_file <- g2f_data_file(paths, "2020_g2f_waseca_raw_weather_V3.xls")
g2f_require_file(input_file, "MNH1.2020 raw station workbook")
output_dir <- g2f_results_dir(paths, "02_fpca_weather", "station")

sheet_plan <- data.frame(
  sheet = 2:13,
  skip = c(5L, 3L, rep(5L, 10L))
)

monthly <- Map(
  function(sheet, skip) {
    readxl::read_xls(input_file, sheet = sheet, skip = skip) |>
      as.data.frame() |>
      dplyr::select(dplyr::all_of(c(
        "Date_key",
        "Air T. Deg F Avg",
        "RH % Min",
        "RH % Max",
        "ST 2\" Deg F Avg"
      )))
  },
  sheet_plan$sheet,
  sheet_plan$skip
)

# The January sheet contains trailing non-observation rows in the source file.
monthly[[1L]] <- monthly[[1L]][seq_len(min(744L, nrow(monthly[[1L]]))), ]
weather <- dplyr::bind_rows(monthly)

output <- weather |>
  dplyr::transmute(
    Date_key = .data$Date_key,
    `Temperature [C]` = (.data[["Air T. Deg F Avg"]] - 32) * (5 / 9),
    `Relative Humidity [%]` = rowMeans(
      dplyr::pick(dplyr::all_of(c("RH % Min", "RH % Max"))),
      na.rm = TRUE
    ),
    `Soil Temperature [C]` = (.data[["ST 2\" Deg F Avg"]] - 32) * (5 / 9),
    `Field Location` = "MNH1",
    Year = 2020L
  )

data.table::fwrite(
  output,
  file.path(output_dir, "2020_g2f_waseca_raw_weather_stacked_FPCA.csv")
)
