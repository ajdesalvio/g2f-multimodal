# Combine location-level vegetation-index and plant-height files.
#
# By default, inputs are expected in <data_dir>/vi_pht_location_files. Set
# G2F_VI_PHT_INPUT_DIR to use an existing location-level data directory.

suppressPackageStartupMessages({
  library(data.table)
})

source(file.path("R", "utils", "paths.R"))
paths <- g2f_paths()

input_dir <- Sys.getenv(
  "G2F_VI_PHT_INPUT_DIR",
  unset = g2f_data_file(paths, "vi_pht_location_files")
)
output_dir <- g2f_results_file(paths, "00_data_prep")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Explicit names replace the historical dependence on list.files() ordering.
pht_files <- c(
  "College_Station_2020_VIs_PHTs_G2F.csv",
  "Delaware_2020_VIs_PHTs_G2F.csv",
  "Michigan_2020_VIs_PHTs_G2F.csv",
  "Minnesota_2020_VIs_PHTs_G2F.csv",
  "Minnesota_2021_VIs_PHTs_G2F.csv",
  "Missouri_C5a_2020_VIs_PHTs_G2F.csv",
  "Missouri_C5b_2020_VIs_PHTs_G2F.csv",
  "Nebraska_2021_VIs_PHTs_G2F.csv"
)

vi_only_files <- c(
  "Arlington_2020_VIs_G2F.csv",
  "Arlington_2021_VIs_G2F.csv",
  "College_Station_2021_VIs_G2F.csv",
  "Hancock_2020_VIs_G2F.csv",
  "Hancock_2021_VIs_G2F.csv",
  "Iowa_2020_VIs_G2F.csv",
  "Iowa_2021_VIs_G2F.csv",
  "Madison_2020_VIs_G2F.csv",
  "Madison_2021_VIs_G2F.csv"
)

input_files <- file.path(input_dir, c(pht_files, vi_only_files))
missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing location-level VI/PHT input files:\n",
    paste0("- ", missing_files, collapse = "\n")
  )
}

read_input <- function(filename) {
  as.data.frame(fread(file.path(input_dir, filename)))
}

pht_list <- lapply(pht_files, read_input)
vi_only_list <- lapply(vi_only_files, read_input)

# The Michigan file historically contained rows without plot identifiers.
mi_index <- match("Michigan_2020_VIs_PHTs_G2F.csv", pht_files)
pht_list[[mi_index]] <- pht_list[[mi_index]][
  !is.na(pht_list[[mi_index]]$Plot_ID),
  ,
  drop = FALSE
]

common_pht_cols <- Reduce(intersect, lapply(pht_list, names))
common_vi_cols <- Reduce(intersect, lapply(vi_only_list, names))
unexpected_vi_cols <- setdiff(common_vi_cols, common_pht_cols)
if (length(unexpected_vi_cols) > 0L) {
  stop(
    "VI-only files contain columns absent from the common PHT schema: ",
    paste(unexpected_vi_cols, collapse = ", ")
  )
}

align_columns <- function(df) {
  missing_cols <- setdiff(common_pht_cols, names(df))
  df[missing_cols] <- NA
  df <- df[, common_pht_cols, drop = FALSE]

  # fwrite/fread can return IDate columns; use a portable character form.
  df[] <- lapply(df, function(column) {
    if (inherits(column, "IDate")) as.character(column) else column
  })
  df
}

aligned <- lapply(c(pht_list, vi_only_list), align_columns)

# Match each column's storage mode to the first PHT file, as in the source
# analysis. Integer and logical columns were treated as numeric.
target_classes <- vapply(aligned[[1L]], function(x) class(x)[1L], character(1L))
target_classes[target_classes %in% c("integer", "logical")] <- "numeric"

coerce_columns <- function(df) {
  for (column_name in names(df)) {
    if (target_classes[[column_name]] == "character") {
      df[[column_name]] <- as.character(df[[column_name]])
    } else if (target_classes[[column_name]] == "numeric") {
      df[[column_name]] <- as.numeric(df[[column_name]])
    }
  }
  df
}

final_df <- rbindlist(lapply(aligned, coerce_columns), use.names = TRUE)

output_file <- file.path(
  output_dir,
  "2020_2021_G2F_VIs_PHTs_COMBINED.csv"
)
fwrite(final_df, output_file, na = "NA")
message("Wrote ", nrow(final_df), " rows to ", output_file)
