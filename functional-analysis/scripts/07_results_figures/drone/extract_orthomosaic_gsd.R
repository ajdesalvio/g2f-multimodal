# Extract ground sampling distance from georeferenced orthomosaic metadata.
#
# The local manifest must contain Env, Flight.Date, and Orthomosaic.Path. Paths
# may be absolute or relative to the manifest. For a projected CRS whose unit
# is not metres, also provide Metres.Per.CRS.Unit (for example, 0.3048 for ft).

suppressPackageStartupMessages({
  library(data.table)
  library(terra)
})

project_dir <- Sys.getenv("G2F_PROJECT_DIR", unset = getwd())
source(file.path(project_dir, "R", "utils", "paths.R"))

paths <- g2f_paths()
manifest_file <- Sys.getenv(
  "G2F_ORTHOMOSAIC_MANIFEST",
  unset = g2f_data_file(paths, "orthomosaic_file_manifest.csv")
)
g2f_require_file(manifest_file, "Orthomosaic file manifest")
output_dir <- g2f_results_dir(paths, "07_results_figures", "drone")

manifest <- data.table::fread(manifest_file)
required_columns <- c("Env", "Flight.Date", "Orthomosaic.Path")
missing_columns <- setdiff(required_columns, names(manifest))
if (length(missing_columns) > 0L) {
  stop("Orthomosaic manifest is missing: ", paste(missing_columns, collapse = ", "))
}
if (!"Metres.Per.CRS.Unit" %in% names(manifest)) {
  manifest$Metres.Per.CRS.Unit <- NA_real_
}

manifest_directory <- dirname(normalizePath(manifest_file, winslash = "/"))
resolve_orthomosaic <- function(path) {
  if (grepl("^(?:[A-Za-z]:|//|/)", path)) return(path)
  file.path(manifest_directory, path)
}

extract_one <- function(index) {
  record <- manifest[index]
  orthomosaic_file <- resolve_orthomosaic(record$Orthomosaic.Path)
  g2f_require_file(orthomosaic_file, "Orthomosaic")
  orthomosaic <- terra::rast(orthomosaic_file)

  if (terra::is.lonlat(orthomosaic)) {
    stop(
      "GSD cannot be read directly in metres from a longitude/latitude raster: ",
      orthomosaic_file
    )
  }

  wkt <- terra::crs(orthomosaic)
  metres_per_unit <- as.numeric(record$Metres.Per.CRS.Unit)
  if (!is.finite(metres_per_unit)) {
    metre_crs <- grepl(
      "(?:LENGTHUNIT|UNIT)\\[\\\"metre\\\",1(?:[.,]0*)?\\]",
      wkt,
      ignore.case = TRUE
    )
    if (!metre_crs) {
      stop(
        "The CRS is projected but is not explicitly metre-based. Supply ",
        "Metres.Per.CRS.Unit in the manifest for: ", orthomosaic_file
      )
    }
    metres_per_unit <- 1
  }

  pixel_resolution <- terra::res(orthomosaic)
  width_mm <- abs(pixel_resolution[[1L]]) * metres_per_unit * 1000
  height_mm <- abs(pixel_resolution[[2L]]) * metres_per_unit * 1000

  data.table::data.table(
    Env = record$Env,
    Flight.Date = record$Flight.Date,
    Orthomosaic.File = record$Orthomosaic.Path,
    Pixel.Width.mm = width_mm,
    Pixel.Height.mm = height_mm,
    Ground.Sampling.Distance.mm.per.pix = sqrt(width_mm * height_mm),
    CRS = wkt
  )
}

gsd <- data.table::rbindlist(lapply(seq_len(nrow(manifest)), extract_one))
data.table::setorder(gsd, Env, Flight.Date)
data.table::fwrite(
  gsd,
  file.path(output_dir, "Orthomosaic_Ground_Sampling_Distance.csv")
)
