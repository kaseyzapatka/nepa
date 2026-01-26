# --------------------------
# DELIVERABLE 1: SETUP
# --------------------------
# Shared setup for all deliverable 1 scripts
# Load libraries, data, and define helper functions

# --------------------------
# LIBRARIES
# --------------------------

library(here)
library(arrow)
library(tidyverse)
library(jsonlite)
library(scales)
library(zoo)
library(googlesheets4)

# --------------------------
# FILE PATHS
# --------------------------

data_path <- here("data", "analysis", "projects_combined.parquet")


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
# LOAD DATA
# --------------------------

# Load project data
cat("Loading data from:", data_path, "\n")
projects <- read_parquet(data_path)

# Filter to clean energy only
clean_energy <- projects %>%
  filter(project_energy_type == "Clean") %>%
  glimpse()

# Explode lead_agency (handles rare multi-agency cases)
agency_data <- clean_energy %>%
  explode_column("lead_agency") %>%
  filter(!is.na(lead_agency) & lead_agency != "")

# Use pre-computed department column (renamed for consistency with this script)
agency_data <- agency_data %>%
  mutate(department = project_department)

