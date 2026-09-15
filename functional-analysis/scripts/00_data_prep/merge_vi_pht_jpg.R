# Merge the VI/PHT table with plot-image filenames and flight dates.

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

source(file.path("R", "utils", "paths.R"))
paths <- g2f_paths()

output_dir <- g2f_results_file(paths, "00_data_prep")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

prefer_generated <- function(filename) {
  generated <- file.path(output_dir, filename)
  archived <- g2f_data_file(paths, filename)
  if (file.exists(generated)) generated else archived
}

vi_file <- prefer_generated("2020_2021_G2F_VIs_PHTs_COMBINED.csv")
jpg_file <- prefer_generated(
  "2020_2021_G2F_Data_Combined_Filtered_with_split_V7.csv"
)
g2f_require_file(vi_file, "Combined VI/PHT table")
g2f_require_file(jpg_file, "Corrected image-date table")

v <- as.data.frame(fread(vi_file))
jpg <- as.data.frame(fread(jpg_file))

v$Env <- paste(v$Field.Location, v$Year, sep = ".")
jpg$Env <- paste(jpg$Field.Location, jpg$Year, sep = ".")
v$Pedigree.Env <- paste(v$Pedigree, v$Env, sep = ".")
jpg$Pedigree.Env <- paste(jpg$Pedigree, jpg$Env, sep = ".")

# Confirmed analysis filters: no yield was collected for IAH4.2020, missing
# pedigrees are excluded, and only the 25 m Texas flights are retained.
v <- v %>%
  filter(Env != "IAH4.2020", !is.na(Pedigree)) %>%
  filter(!Flight.Altitude %in% c("60m", "80m"))

# These early DAPs were excluded when the VI BLUEs were calculated.
excluded_daps <- c(-33, -9, -3, 0, 1, 5)
v <- v %>% filter(!DAP %in% excluded_daps)
jpg <- jpg %>% filter(!DAP %in% excluded_daps)

v$Plot_ID.Env.DAP <- paste(v$Plot_ID, v$Env, v$DAP, sep = ".")
jpg$Plot_ID.Env.DAP <- paste(jpg$Plot_ID, jpg$Env, jpg$DAP, sep = ".")
v$Plot_ID.Pedigree.Env.DAP <- paste(
  v$Plot_ID,
  v$Pedigree.Env,
  v$DAP,
  sep = "."
)
jpg$Plot_ID.Pedigree.Env.DAP <- paste(
  jpg$Plot_ID,
  jpg$Pedigree.Env,
  jpg$DAP,
  sep = "."
)

# Michigan contains duplicated plot IDs, so its stable unique-location ID is
# the historical and confirmed join key. All other environments use plot ID.
v_mi <- v %>% filter(Env == "MIH1.2020")
jpg_mi <- jpg %>% filter(Env == "MIH1.2020")
v_18 <- v %>% filter(Env != "MIH1.2020")
jpg_18 <- jpg %>% filter(Env != "MIH1.2020")

v_mi$Unique.Location.ID.Pedigree.Env.DAP <- paste(
  v_mi$Unique.Location.ID,
  v_mi$Pedigree.Env,
  v_mi$DAP,
  sep = "."
)
jpg_mi$Unique.Location.ID.Pedigree.Env.DAP <- paste(
  jpg_mi$Unique.Location.ID,
  jpg_mi$Pedigree.Env,
  jpg_mi$DAP,
  sep = "."
)
v_18$Unique.Location.ID.Pedigree.Env.DAP <- paste(
  v_18$Unique.Location.ID,
  v_18$Pedigree.Env,
  v_18$DAP,
  sep = "."
)
jpg_18$Unique.Location.ID.Pedigree.Env.DAP <- paste(
  jpg_18$Unique.Location.ID,
  jpg_18$Pedigree.Env,
  jpg_18$DAP,
  sep = "."
)

v_jpg_mi <- left_join(
  v_mi,
  jpg_mi,
  by = "Unique.Location.ID.Pedigree.Env.DAP"
) %>%
  select(-Plot_ID.Pedigree.Env.DAP.y) %>%
  rename(Plot_ID.Pedigree.Env.DAP = Plot_ID.Pedigree.Env.DAP.x)

v_jpg_18 <- left_join(
  v_18,
  jpg_18,
  by = "Plot_ID.Pedigree.Env.DAP"
) %>%
  select(-Unique.Location.ID.Pedigree.Env.DAP.y) %>%
  rename(
    Unique.Location.ID.Pedigree.Env.DAP =
      Unique.Location.ID.Pedigree.Env.DAP.x
  )

if (!setequal(names(v_jpg_mi), names(v_jpg_18))) {
  stop("Michigan and non-Michigan joins produced incompatible schemas.")
}
v_jpg_18 <- v_jpg_18[, names(v_jpg_mi), drop = FALSE]
all_data <- rbind(v_jpg_mi, v_jpg_18)
names(all_data) <- sub("\\.x$", "", names(all_data))

output_file <- file.path(
  output_dir,
  "2020_2021_G2F_Data_JPGs_VIs_ALLDATA_V1.csv"
)
fwrite(all_data, output_file)
message("Wrote ", nrow(all_data), " merged rows to ", output_file)
