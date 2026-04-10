# --------------------------
# VALIDATION: TIMELINE SETUP
# --------------------------
# Shared setup for timeline validation scripts.
# Source this at the top of each validation script.

library(here)
library(arrow)
library(tidyverse)
library(scales)
library(patchwork)

# --------------------------
# FILE PATHS
# --------------------------

# Regex candidate caches — baseline (pre-refactor)
regex_ce_path  <- here("data", "analysis", "regex_candidates.parquet")
regex_ea_path  <- here("data", "analysis", "regex_candidates_ea.parquet")
regex_eis_path <- here("data", "analysis", "regex_candidates_eis.parquet")

# Regex candidate caches — refactored (post-DuckDB rewrite)
regex_ce_new_path  <- here("data", "analysis", "regex_candidates_ce_refactored.parquet")
regex_ea_new_path  <- here("data", "analysis", "regex_candidates_ea_refactored.parquet")
regex_eis_new_path <- here("data", "analysis", "regex_candidates_eis_refactored.parquet")

# Convenience lookup: source name -> list(baseline, refactored)
regex_paths <- list(
  CE  = list(baseline = regex_ce_path,  refactored = regex_ce_new_path),
  EA  = list(baseline = regex_ea_path,  refactored = regex_ea_new_path),
  EIS = list(baseline = regex_eis_path, refactored = regex_eis_new_path)
)

# BERT timeline output
bert_path <- here("data", "analysis", "projects_timeline_bert.parquet")

# --------------------------
# HELPERS
# --------------------------

#' Load a regex candidates parquet and tag with a run label
load_candidates <- function(path, label, source = NULL) {
  df <- read_parquet(path)
  df$run_label <- label
  if (!is.null(source)) df$source <- source
  # Normalise types
  if ("date" %in% names(df)) df$date <- as.Date(df$date)
  if ("main_document_imputed" %in% names(df)) {
    df$main_document_imputed <- as.logical(df$main_document_imputed)
  } else {
    df$main_document_imputed <- FALSE
  }
  if (!"doc_type" %in% names(df)) df$doc_type <- NA_character_
  as_tibble(df)
}

#' CATF palette (navy / blue / accent grey)
catf_colors <- c(
  navy  = "#012169",
  blue  = "#0047BB",
  grey  = "#6B7280",
  green = "#16A34A",
  red   = "#DC2626",
  amber = "#D97706"
)

source_palette <- c(CE = "#0047BB", EA = "#16A34A", EIS = "#D97706")

theme_nepa <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title    = element_text(colour = "#012169", face = "bold"),
      plot.subtitle = element_text(colour = "#6B7280"),
      axis.title    = element_text(colour = "#1A1A2E"),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
}
