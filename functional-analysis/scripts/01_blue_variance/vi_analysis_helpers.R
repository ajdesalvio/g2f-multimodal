# Shared filters for VI BLUE and variance-component analyses.

g2f_drone_environments <- sort(c(
  "TXH1.2020", "TXH2.2020", "TXH3.2020", "DEH1.2020", "MIH1.2020",
  "MNH1.2020", "MNH1.2021", "MOH1.2020", "NEH1.2021", "WIH2.2020",
  "WIH2.2021", "WIH3.2021", "TXH1.2021", "TXH2.2021", "TXH3.2021",
  "WIH3.2020", "IAH4.2021", "WIH1.2020", "WIH1.2021"
))

# The historical analysis used columns BI through TNDGR and excluded VARI_new
# because of its documented data-quality problems.
g2f_vi_names <- c(
  "BI", "GLI", "NGRDI", "VARI", "BGI", "BI_new", "GLI_new", "NGRDI_new",
  "BGI_new", "BCC", "CIVE", "COM1", "COM2", "ExG", "ExG2", "ExGR", "EXR",
  "GminusB", "GminusR", "GdivB", "GdivR", "GCC", "MExG", "MGVRI", "NDI",
  "NDRBI", "NGBDI", "RminusB", "RdivB", "RCC", "MRCC", "RGBVI", "TGI",
  "VEG", "NRMBI", "MSRGR", "TNDGR"
)

g2f_prepare_vi_data <- function(df) {
  df$Env <- paste(df$Field.Location, df$Year, sep = ".")
  df$Env.Experiment <- paste(
    df$Field.Location,
    df$Year,
    df$Experiment,
    sep = "."
  )
  df <- df[df$Env %in% g2f_drone_environments, , drop = FALSE]

  early_dap_environments <- c(
    "TXH3.2020", "MNH1.2020", "TXH1.2021", "TXH2.2021", "TXH3.2021"
  )
  early_daps <- c(-33, -9, -3, 0, 1, 5)
  remove_early <-
    df$Env %in% early_dap_environments & df$DAP %in% early_daps
  df <- df[!remove_early, , drop = FALSE]

  # Confirmed exclusions: non-25 m Texas flights and globally sparse DAPs.
  df <- df[!df$Flight.Altitude %in% c("60m", "80m"), , drop = FALSE]
  df <- df[!df$DAP %in% c(121, 136, 151, 156, 163, 171), , drop = FALSE]
  df$Env.DAP <- paste(df$Env, df$DAP, sep = ".")

  # Preserve the confirmed historical rule: remove the 13 least-populated
  # environment/DAP combinations. Alphabetical tie-breaking makes it stable.
  counts <- as.data.frame(table(df$Env.DAP), stringsAsFactors = FALSE)
  names(counts) <- c("Env.DAP", "Frequency")
  counts <- counts[order(counts$Frequency, counts$Env.DAP), , drop = FALSE]
  if (nrow(counts) < 13L) stop("Fewer than 13 environment/DAP groups remain.")
  sparse_env_daps <- c(
    "WIH3.2021.51", "WIH3.2021.59", "WIH3.2021.65", "WIH3.2021.72",
    "WIH3.2021.79", "WIH3.2021.86", "WIH3.2021.88", "WIH3.2021.93",
    "WIH3.2021.95", "WIH3.2021.109", "WIH3.2021.115", "WIH3.2021.129",
    "WIH3.2021.142"
  )
  computed_sparse_env_daps <- counts$Env.DAP[seq_len(13L)]
  if (!setequal(sparse_env_daps, computed_sparse_env_daps)) {
    stop(
      "The 13 least-populated environment/DAP groups differ from the ",
      "confirmed final exclusions. Check the input data version."
    )
  }
  df <- df[!df$Env.DAP %in% sparse_env_daps, , drop = FALSE]

  # MOH1.2020.13 was excluded from model fitting in the final analysis.
  env_daps <- sort(setdiff(unique(df$Env.DAP), "MOH1.2020.13"))

  missing_vis <- setdiff(g2f_vi_names, names(df))
  if (length(missing_vis) > 0L) {
    stop("Missing VI columns: ", paste(missing_vis, collapse = ", "))
  }
  df[g2f_vi_names] <- lapply(df[g2f_vi_names], as.numeric)
  df$Range <- factor(df$Range)
  df$Pass <- factor(df$Pass)
  df$Replicate <- factor(df$Replicate)
  df$Pedigree <- factor(df$Pedigree)

  list(data = df, env_daps = env_daps, sparse_env_daps = sparse_env_daps)
}
