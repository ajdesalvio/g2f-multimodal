# Static checks that do not require study data or analysis packages.

root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!file.exists(file.path(root, "config", "paths.example.yml"))) {
  stop("Run validation from the functional-analysis directory.")
}

r_files <- list.files(
  c(file.path(root, "R"), file.path(root, "scripts")),
  pattern = "[.]R$",
  recursive = TRUE,
  full.names = TRUE
)
parse_errors <- character()
for (file in r_files) {
  tryCatch(
    parse(file),
    error = function(error) {
      parse_errors <<- c(
        parse_errors,
        paste0(sub(paste0("^", root, "/"), "", file), ": ", conditionMessage(error))
      )
    }
  )
}
if (length(parse_errors) > 0L) {
  stop("R parse failures:\n", paste(parse_errors, collapse = "\n"))
}

csv_files <- list.files(root, pattern = "[.]csv$", recursive = TRUE, full.names = TRUE)
csv_errors <- character()
for (file in csv_files) {
  tryCatch(
    utils::read.csv(file, nrows = 5L, check.names = FALSE),
    error = function(error) {
      csv_errors <<- c(
        csv_errors,
        paste0(sub(paste0("^", root, "/"), "", file), ": ", conditionMessage(error))
      )
    }
  )
}
if (length(csv_errors) > 0L) {
  stop("CSV parse failures:\n", paste(csv_errors, collapse = "\n"))
}

path_scan_files <- setdiff(r_files, file.path(root, "scripts", "validate_repository.R"))
script_text <- unlist(lapply(path_scan_files, readLines, warn = FALSE), use.names = FALSE)
personal_path_patterns <- c(
  "[A-Za-z]:/Users/",
  "[A-Za-z]:\\\\Users\\\\",
  "My Drive/",
  "//murray_lab/",
  "/scratch/user/"
)
for (pattern in personal_path_patterns) {
  if (any(grepl(pattern, script_text, ignore.case = TRUE))) {
    stop("An active R script contains a personal or site-specific path matching: ", pattern)
  }
}

cat(
  "Validated ", length(r_files), " R files and ", length(csv_files),
  " CSV files.\n",
  sep = ""
)
