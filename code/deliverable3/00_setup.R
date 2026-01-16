# --------------------------
# DELIVERABLE 3: SETUP
# --------------------------
# Shared setup for all deliverable 3 scripts
# Load libraries, data, and define helper functions

# --------------------------
# LIBRARIES
# --------------------------

library(here)
library(arrow)
library(tidyverse)
library(jsonlite)
library(scales)

# --------------------------
# FILE PATHS
# --------------------------

data_path <- here("data", "analysis", "projects_combined.parquet")
output_dir <- here("output", "deliverable3")
tables_dir <- here("output", "deliverable3", "tables")
figures_dir <- here("output", "deliverable3", "figures")

# Create output directories if they don't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------
# LOAD DATA
# --------------------------

cat("Loading data from:", data_path, "\n")
projects <- read_parquet(data_path)

cat("Total projects loaded:", nrow(projects), "\n\n")

# --------------------------
# CONSTANTS
# --------------------------

clean_energy_tags <- c(
  "Carbon Capture and Sequestration",
  "Conventional Energy Production - Nuclear",
  "Conventional Energy Production - Other",
  "Renewable Energy Production - Biomass",
  "Renewable Energy Production - Energy Storage",
  "Renewable Energy Production - Geothermal",
  "Renewable Energy Production - Hydrokinetic",
  "Renewable Energy Production - Hydropower",
  "Renewable Energy Production - Other",
  "Renewable Energy Production - Solar",
  "Renewable Energy Production - Wind, Offshore",
  "Renewable Energy Production - Wind, Onshore",
  "Nuclear Technology",
  "Electricity Transmission",
  "Utilities (electricity, gas, telecommunications)"
)

# --------------------------
# HELPER FUNCTIONS
# --------------------------

#' Explode JSON-encoded column to multiple rows
#' Handles: JSON arrays like '["value1", "value2"]', plain strings, lists, NULL
explode_column <- function(df, col_name) {
  df %>%
    mutate(!!col_name := sapply(.data[[col_name]], function(x) {
      if (is.null(x) || length(x) == 0 || (is.character(x) && x == "")) {
        return(NA_character_)
      }
      # Handle JSON-encoded arrays
      if (is.character(x) && grepl("^\\[", x)) {
        parsed <- tryCatch(
          jsonlite::fromJSON(x),
          error = function(e) x
        )
        if (is.character(parsed) && length(parsed) > 1) {
          return(paste(parsed, collapse = "|"))
        }
        return(as.character(parsed))
      }
      if (is.list(x)) return(paste(unlist(x), collapse = "|"))
      return(as.character(x))
    })) %>%
    separate_rows(!!col_name, sep = "\\|")
}

#' Create cross-tabulation table by process type
create_crosstab <- function(df, group_col, process_col = "process_type") {
  df %>%
    group_by(.data[[group_col]], .data[[process_col]]) %>%
    summarise(n = n(), .groups = "drop") %>%
    pivot_wider(
      names_from = all_of(process_col),
      values_from = n,
      values_fill = 0
    ) %>%
    mutate(Total = rowSums(select(., -1), na.rm = TRUE)) %>%
    arrange(desc(Total))
}

#' Add totals row to a crosstab table
add_totals_row <- function(df, group_col_name) {
  totals <- df %>%
    summarise(
      across(where(is.numeric), sum, na.rm = TRUE)
    ) %>%
    mutate(!!group_col_name := "Total") %>%
    select(!!group_col_name, everything())

  bind_rows(df, totals)
}

# --------------------------
# SETUP COMPLETE
# --------------------------

cat("=== Setup Complete ===\n")
cat("Output directories:\n")
cat("  Tables:", tables_dir, "\n")
cat("  Figures:", figures_dir, "\n")
