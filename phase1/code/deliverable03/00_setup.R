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
library(zoo)
library(googlesheets4)

source(here::here("phase1", "code", "utils", "utils.R"))

# --------------------------
# FILE PATHS
# --------------------------

data_path <- here("phase1", "data", "analysis", "projects_combined.parquet")
output_dir <- here("phase1", "output", "deliverable3")
tables_dir <- here("phase1", "output", "deliverable3", "tables")
figures_dir <- here("phase1", "output", "deliverable3", "figures")

# Create output directories if they don't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------
# LOAD DATA
# --------------------------

cat("Loading data from:", data_path, "\n")
projects <- read_parquet(data_path)

cat("Total projects loaded:", nrow(projects), "\n")
cat("Clean energy projects:", sum(projects$project_energy_type == "Clean"), "\n\n")

# Filter to clean energy only
# Note: project_energy_type already reflects final classification after all exclusions
# (utilities, military nuclear, nuclear waste) are applied in the Python extraction pipeline
clean_energy <- projects %>%
  filter(project_energy_type == "Clean") %>%
  glimpse()

cat("Clean energy dataset ready:", nrow(clean_energy), "projects\n")

# --------------------------
# TIMELINE INPUTS (DELIVERABLE 3)
# --------------------------

# Current production inputs:
# - CE uses BERT final dates (no LLM adjudication file yet)
# - EA uses LLM-adjudicated dates
# - EIS line is ready but optional until file exists
timeline_ce_path <- here("phase1", "data", "analysis", "projects_timeline_bert.parquet")
timeline_ea_path <- here("phase1", "data", "analysis", "projects_timeline_bert_ea_llm.parquet")
timeline_eis_path <- here("phase1", "data", "analysis", "projects_timeline_bert_eis_llm.parquet")

#' Load and harmonize timeline files for Deliverable 3
#'
#' Harmonization rule:
#' - CE: use BERT final initiation/decision dates
#' - EA/EIS: use LLM initiation/decision dates
#'
#' @return A tibble with harmonized timeline columns compatible with existing analysis scripts
load_timeline_for_deliverable3 <- function() {
  if (!file.exists(timeline_ce_path)) {
    stop("Missing required CE timeline file: ", timeline_ce_path)
  }
  if (!file.exists(timeline_ea_path)) {
    stop("Missing required EA timeline file: ", timeline_ea_path)
  }
  if (!file.exists(timeline_eis_path)) {
    stop("Missing required EIS timeline file: ", timeline_eis_path)
  }

  ce_df <- read_parquet(timeline_ce_path) %>%
    mutate(timeline_input_file = basename(timeline_ce_path))

  ea_df <- read_parquet(timeline_ea_path) %>%
    mutate(timeline_input_file = basename(timeline_ea_path))

  eis_df <- read_parquet(timeline_eis_path) %>%
    mutate(timeline_input_file = basename(timeline_eis_path))

  timeline_raw <- bind_rows(ce_df, ea_df, eis_df)

  # Ensure source exists even if a file is missing this field.
  if (!"dataset_source" %in% names(timeline_raw)) {
    timeline_raw <- timeline_raw %>% mutate(dataset_source = NA_character_)
  }

  # Harmonize final dates used in downstream analysis.
  # CE: BERT final dates
  # EA/EIS: LLM dates
  timeline_harmonized <- timeline_raw %>%
    mutate(
      dataset_source = toupper(as.character(dataset_source)),
      timeline_initiation_date_final = as.Date(case_when(
        dataset_source %in% c("EA", "EIS") ~ llm_initiation_date,
        TRUE ~ bert_initiation_date_final
      )),
      timeline_decision_date_final = as.Date(case_when(
        dataset_source %in% c("EA", "EIS") ~ llm_decision_date,
        TRUE ~ bert_decision_date_final
      )),
      timeline_method = case_when(
        dataset_source %in% c("EA", "EIS") ~ "llm",
        TRUE ~ "bert"
      ),
      # Keep legacy column names used in existing scripts, now harmonized.
      bert_initiation_date_final = timeline_initiation_date_final,
      bert_decision_date_final = timeline_decision_date_final,
      bert_decision_date = timeline_decision_date_final,
      bert_application_date = if_else(
        dataset_source %in% c("EA", "EIS"),
        timeline_initiation_date_final,
        as.Date(bert_application_date)
      ),
      bert_inferred_application_date = if_else(
        dataset_source %in% c("EA", "EIS"),
        as.Date(NA),
        as.Date(bert_inferred_application_date)
      ),
      bert_timeline_status = case_when(
        !is.na(timeline_decision_date_final) & !is.na(timeline_initiation_date_final) ~ "complete",
        !is.na(timeline_decision_date_final) & is.na(timeline_initiation_date_final) ~ "missing_initiation",
        is.na(timeline_decision_date_final) & !is.na(timeline_initiation_date_final) ~ "missing_decision",
        TRUE ~ "no_dates"
      ),
      # Explicit harmonized decision year: BERT for CE, LLM for EA/EIS.
      # Derived directly from timeline_decision_date_final (not its legacy alias)
      # so the source is unambiguous.
      decision_year = as.integer(format(timeline_decision_date_final, "%Y"))
    )

  timeline_harmonized
}

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
