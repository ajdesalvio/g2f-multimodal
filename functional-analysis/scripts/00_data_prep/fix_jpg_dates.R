# Correct the documented College Station flight-date mismatch.

suppressPackageStartupMessages({
  library(data.table)
})

source(file.path("R", "utils", "paths.R"))
paths <- g2f_paths()

input_file <- g2f_data_file(
  paths,
  "2020_2021_G2F_Data_Combined_Filtered_with_split_V6.csv"
)
g2f_require_file(input_file, "Version 6 image-date table")

output_dir <- g2f_results_file(paths, "00_data_prep")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_file <- file.path(
  output_dir,
  "2020_2021_G2F_Data_Combined_Filtered_with_split_V7.csv"
)

df <- fread(input_file)
df[, Flight.Date := as.character(Flight.Date)]

affected <- df$Flight.Date == "2020-06-20"
affected[is.na(affected)] <- FALSE
if (!any(affected)) {
  stop("No rows were found for the documented 2020-06-20 correction.")
}

df[affected, Flight.Date := "2020-06-21"]
for (column_name in c(
  "File.Name.JPG",
  "Unique.Location.ID.JPG",
  "File.Name.JPG.Unified"
)) {
  if (!column_name %in% names(df)) {
    stop("Expected column is missing: ", column_name)
  }
  set(
    df,
    i = which(affected),
    j = column_name,
    value = sub("20200620", "20200621", df[[column_name]][affected], fixed = TRUE)
  )
}
df[affected, DAP := DAP + 1]
setorder(df, Field.Location.Year, Pedigree, DAP)

fwrite(df, output_file)
message("Corrected ", sum(affected), " rows and wrote ", output_file)
