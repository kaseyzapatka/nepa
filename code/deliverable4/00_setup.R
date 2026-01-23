# --------------------------
# DELIVERABLE 4: SETUP
# --------------------------
# Shared setup for all deliverable 4 scripts
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
library(sf)
library(tidycensus)
library(tigris)

# options
options(tigris_use_cache = TRUE)

# --------------------------
# FILE PATHS
# --------------------------

data_path <- here("data", "analysis", "projects_combined.parquet")
output_dir <- here("output", "deliverable4")
tables_dir <- here("output", "deliverable4", "tables")
figures_dir <- here("output", "deliverable4", "figures")
maps_dir <- here("output", "deliverable4", "maps")

# Create output directories if they don't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(maps_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------
# LOAD DATA
# --------------------------

cat("Loading data from:", data_path, "\n")
projects <- read_parquet(data_path)

cat("Total projects loaded:", nrow(projects), "\n")
cat("Clean energy projects:", sum(projects$project_energy_type == "Clean"), "\n\n")

# Filter to clean energy only
clean_energy <- projects %>%
  filter(project_energy_type == "Clean") |> 
  # remove Utilities + Broadband, Waste Management, or Land Development tags
  filter(!project_utilities_to_filter_out)

cat("Clean energy dataset ready:", nrow(clean_energy), "projects\n")

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
#'
#' @param df Data frame to process
#' @param group_col Column name to group by (quoted string)
#' @param process_col Column for process type (default "process_type")
#' @param keep_cols Optional character vector of column names to keep (takes first value per group)
#' @return A tibble with counts by process type and optional extra columns
create_crosstab <- function(df, group_col, process_col = "process_type", keep_cols = NULL) {
  # Create the basic crosstab
  crosstab <- df %>%
    group_by(.data[[group_col]], .data[[process_col]]) %>%
    summarise(n = n(), .groups = "drop") %>%
    pivot_wider(
      names_from = all_of(process_col),
      values_from = n,
      values_fill = 0
    ) %>%
    mutate(Total = rowSums(select(., -1), na.rm = TRUE)) %>%
    arrange(desc(Total))

 # If keep_cols specified, get first values per group and join
  if (!is.null(keep_cols)) {
    extra_info <- df %>%
      group_by(.data[[group_col]]) %>%
      summarise(across(all_of(keep_cols), ~ first(na.omit(.))), .groups = "drop")

    crosstab <- crosstab %>%
      left_join(extra_info, by = group_col) %>%
      select(all_of(group_col), all_of(keep_cols), everything())
  }

  crosstab
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
# CATF BRAND THEME
# --------------------------
# Clean Air Task Force Brand Guide (November 2018)
# Colors, typography, and ggplot2 theme for consistent figure styling

# Primary Colors
catf_dark_blue <- "#0047BB"
catf_blue <- "#00B5E2"

# Secondary Colors
catf_magenta <- "#C22A90"
catf_purple <- "#75246C"
catf_lime <- "#93D500"
catf_teal <- "#00AE8D"
catf_light_blue <- "#8AB7E9"
catf_navy <- "#002169"

# Named color palette for easy access
catf_colors <- c(
  "dark_blue" = "#0047BB",
  "blue" = "#00B5E2",

  "magenta" = "#C22A90",
  "purple" = "#75246C",
  "lime" = "#93D500",
  "teal" = "#00AE8D",
  "light_blue" = "#8AB7E9",
  "navy" = "#002169"
)

# Categorical color palette (ordered for visual distinction)
catf_palette <- c(
  "#0047BB",
  "#00B5E2",

"#00AE8D",
  "#93D500",
  "#C22A90",
  "#75246C",
  "#8AB7E9",
  "#002169"
)

# Sequential palette (blue gradient)
catf_sequential <- c("#8AB7E9", "#00B5E2", "#0047BB", "#002169")

# Diverging palette (teal to magenta through blue)
catf_diverging <- c("#00AE8D", "#00B5E2", "#0047BB", "#75246C", "#C22A90")

#' CATF ggplot2 theme
#'
#' A minimal theme based on CATF brand guidelines.
#' Primary font: Circular Std (fallback: Helvetica, Arial, sans-serif)
#'
#' @param base_size Base font size (default 11)
#' @param base_family Base font family (default "Helvetica")
#' @return A ggplot2 theme object
theme_catf <- function(base_size = 11, base_family = "Helvetica") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # Text elements
      plot.title = element_text(
        face = "bold",
        size = rel(1.2),
        color = catf_navy,
        margin = margin(b = 10)
      ),
      plot.subtitle = element_text(
        size = rel(0.9),
        color = catf_dark_blue,
        margin = margin(b = 10)
      ),
      plot.caption = element_text(
        size = rel(0.8),
        color = "gray50",
        hjust = 1
      ),

      # Axis elements
      axis.title = element_text(
        size = rel(0.9),
        color = catf_navy
      ),
      axis.text = element_text(
        size = rel(0.85),
        color = "gray30"
      ),
      axis.line = element_line(color = "gray70", linewidth = 0.3),

      # Legend
      legend.title = element_text(
        face = "bold",
        size = rel(0.9),
        color = catf_navy
      ),
      legend.text = element_text(
        size = rel(0.85),
        color = "gray30"
      ),
      legend.position = "bottom",
      legend.key.size = unit(0.8, "lines"),

      # Panel
      panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),

      # Facets
      strip.text = element_text(
        face = "bold",
        size = rel(0.9),
        color = catf_navy
      ),
      strip.background = element_rect(fill = "gray95", color = NA),

      # Margins
      plot.margin = margin(15, 15, 10, 10)
    )
}

#' CATF discrete color scale
#'
#' @param ... Additional arguments passed to scale_color_manual
#' @return A ggplot2 color scale
scale_color_catf <- function(...) {
  scale_color_manual(values = catf_palette, ...)
}

#' CATF discrete fill scale
#'
#' @param ... Additional arguments passed to scale_fill_manual
#' @return A ggplot2 fill scale
scale_fill_catf <- function(...) {
  scale_fill_manual(values = catf_palette, ...)
}

#' CATF sequential color scale (continuous)
#'
#' @param ... Additional arguments passed to scale_color_gradient
#' @return A ggplot2 color scale
scale_color_catf_seq <- function(...) {
  scale_color_gradientn(colors = catf_sequential, ...)
}

#' CATF sequential fill scale (continuous)
#'
#' @param ... Additional arguments passed to scale_fill_gradient
#' @return A ggplot2 fill scale
scale_fill_catf_seq <- function(...) {
  scale_fill_gradientn(colors = catf_sequential, ...)
}

# Set default theme for session
theme_set(theme_catf())

# --------------------------
# SETUP COMPLETE
# --------------------------

cat("\n=== Setup Complete ===\n")
cat("Output directories:\n")
cat("  Tables:", tables_dir, "\n")
cat("  Figures:", figures_dir, "\n")
cat("CATF brand theme loaded and set as default\n")
