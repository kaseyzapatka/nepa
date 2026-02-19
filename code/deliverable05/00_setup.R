# --------------------------
# DELIVERABLE 5: SETUP
# --------------------------
# Shared setup for deliverable 5 (Pages Over Time / FRA Analysis)
# Merges projects, timeline, and documents for EA and EIS clean energy projects
#
# Key output objects:
#   pages_data     - full merged dataset (projects + timeline + final document pages)
#   pages_analysis - analysis subset: complete timelines only, with FRA classification
#   coverage       - project counts at each filter step (for coverage figure)
#   fra_date       - FRA enactment date (2023-06-03)

# --------------------------
# LIBRARIES
# --------------------------

library(here)
library(arrow)
library(tidyverse)
library(jsonlite)
library(scales)
library(zoo)

# --------------------------
# FILE PATHS
# --------------------------

data_path <- here("data", "analysis", "projects_combined.parquet")
documents_path <- here("data", "analysis", "documents_combined.parquet")
timeline_ea_path <- here("data", "analysis", "projects_timeline_bert_ea_llm.parquet")
timeline_eis_path <- here("data", "analysis", "projects_timeline_bert_eis_llm.parquet")

output_dir <- here("output", "deliverable5")
tables_dir <- here("output", "deliverable5", "tables")
figures_dir <- here("output", "deliverable5", "figures")

# Create output directories
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------
# CONSTANTS
# --------------------------

# Fiscal Responsibility Act of 2023: signed June 3, 2023
# Sets page limits for NEPA reviews (EAs and EISs)
fra_date <- as.Date("2023-06-03")

# --------------------------
# LOAD AND FILTER PROJECTS
# --------------------------

cat("Loading projects from:", data_path, "\n")
projects_all <- read_parquet(data_path)

# Deliverable 5 scope: clean energy EA and EIS projects only
# (CE excluded because FRA page limits apply only to EA/EIS)
projects <- projects_all %>%
  filter(project_energy_type == "Clean", process_type %in% c("EA", "EIS"))

cat("Clean energy EA+EIS projects:", nrow(projects), "\n")
cat("  EA:", sum(projects$process_type == "EA"), "\n")
cat("  EIS:", sum(projects$process_type == "EIS"), "\n\n")

# --------------------------
# LOAD TIMELINE DATA
# --------------------------

cat("Loading timeline data...\n")
if (!file.exists(timeline_ea_path)) stop("Missing EA timeline file: ", timeline_ea_path)
if (!file.exists(timeline_eis_path)) stop("Missing EIS timeline file: ", timeline_eis_path)

ea_timeline <- read_parquet(timeline_ea_path)
eis_timeline <- read_parquet(timeline_eis_path)

# Combine EA + EIS timelines, harmonize to LLM-adjudicated dates
timeline <- bind_rows(
  ea_timeline %>% mutate(timeline_source = "ea_llm"),
  eis_timeline %>% mutate(timeline_source = "eis_llm")
) %>%
  mutate(
    dataset_source = toupper(as.character(dataset_source)),
    timeline_initiation_date = as.Date(llm_initiation_date),
    timeline_decision_date = as.Date(llm_decision_date)
  ) %>%
  select(project_id, dataset_source, timeline_source,
         timeline_initiation_date, timeline_decision_date) %>%
  distinct(project_id, .keep_all = TRUE)

cat("  Timeline records:", nrow(timeline), "\n")
cat("    EA:", sum(timeline$dataset_source == "EA"), "\n")
cat("    EIS:", sum(timeline$dataset_source == "EIS"), "\n\n")

# --------------------------
# LOAD AND FILTER DOCUMENTS
# --------------------------

cat("Loading documents from:", documents_path, "\n")
documents_all <- read_parquet(documents_path)

# Keep only FINAL documents:
#   EIS projects -> FEIS (Final Environmental Impact Statement)
#   EA projects  -> EA   (Final Environmental Assessment; DEA is draft)
final_docs <- documents_all %>%
  filter(
    (dataset_source == "EIS" & document_type_clean == "FEIS") |
    (dataset_source == "EA" & document_type_clean == "EA")
  ) %>%
  mutate(total_pages = as.numeric(total_pages))

cat("  Final documents found:", nrow(final_docs), "\n")
cat("    FEIS (EIS projects):", sum(final_docs$document_type_clean == "FEIS"), "\n")
cat("    EA (EA projects):", sum(final_docs$document_type_clean == "EA"), "\n")

# Deduplicate: one document per project
# Priority: main_document == "YES" first, then highest total_pages
final_docs_dedup <- final_docs %>%
  mutate(is_main = (main_document == "YES")) %>%
  group_by(project_id) %>%
  arrange(desc(is_main), desc(total_pages), .by_group = TRUE) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  select(-is_main)

n_dupes_removed <- nrow(final_docs) - nrow(final_docs_dedup)
cat("  After deduplication:", nrow(final_docs_dedup),
    "projects (", n_dupes_removed, "duplicates removed)\n\n")

# --------------------------
# COVERAGE TRACKING
# --------------------------
# Track how many projects survive each filter step (for coverage figure)

coverage_steps <- list()

# Step 1: Total clean energy EA/EIS with timeline data
# (All clean energy EA/EIS projects have timeline data, so these are equivalent)
coverage_steps[["Total clean energy\nwith timeline data"]] <- projects %>%
  semi_join(timeline, by = "project_id") %>%
  count(process_type, name = "n")

# Step 2: With final document available
coverage_steps[["With final document"]] <- projects %>%
  semi_join(final_docs_dedup, by = "project_id") %>%
  count(process_type, name = "n")

# --------------------------
# MERGE DATASETS
# --------------------------

cat("Merging projects + timeline + documents...\n")

# Inner join: keep only projects that have BOTH timeline data AND a final document
pages_data <- projects %>%
  inner_join(timeline, by = "project_id") %>%
  inner_join(
    final_docs_dedup %>%
      select(project_id, total_pages, document_type_clean, document_id, main_document),
    by = "project_id"
  )

cat("  Merged dataset:", nrow(pages_data), "projects\n")

# Step 4: With timeline + document
coverage_steps[["With timeline + document"]] <- pages_data %>%
  count(process_type, name = "n")

# --------------------------
# ANALYSIS SUBSET
# --------------------------

# Filter to projects with COMPLETE timelines (non-missing initiation AND decision)
# Add FRA period classification and time-derived variables
pages_analysis <- pages_data %>%
  filter(
    !is.na(timeline_initiation_date),
    !is.na(timeline_decision_date)
  ) %>%
  mutate(
    # FRA classification: if decision date >= FRA date, project should comply
    fra_period = if_else(timeline_decision_date >= fra_date, "Post-FRA", "Pre-FRA"),
    fra_period = factor(fra_period, levels = c("Pre-FRA", "Post-FRA")),
    # Time variables for plotting
    decision_year = year(timeline_decision_date),
    decision_month = floor_date(timeline_decision_date, "month"),
    duration_days = as.numeric(timeline_decision_date - timeline_initiation_date),
    duration_months = duration_days / 30.44
  )

# Step 5: Complete timeline (final analysis sample)
coverage_steps[["Complete timeline (analysis)"]] <- pages_analysis %>%
  count(process_type, name = "n")

# Assemble coverage tibble
coverage <- bind_rows(
  lapply(names(coverage_steps), function(step) {
    coverage_steps[[step]] %>% mutate(step = step)
  })
) %>%
  mutate(step = factor(step, levels = names(coverage_steps)))

cat("\n--- Coverage Summary ---\n")
print(coverage %>% pivot_wider(names_from = process_type, values_from = n, values_fill = 0))

cat("\n--- Analysis Sample ---\n")
cat("Total:", nrow(pages_analysis), "projects\n")
cat("  EA:", sum(pages_analysis$process_type == "EA"), "\n")
cat("  EIS:", sum(pages_analysis$process_type == "EIS"), "\n")
cat("  Pre-FRA:", sum(pages_analysis$fra_period == "Pre-FRA"), "\n")
cat("  Post-FRA:", sum(pages_analysis$fra_period == "Post-FRA"), "\n")
cat("  Median pages (EA):",
    median(pages_analysis$total_pages[pages_analysis$process_type == "EA"]), "\n")
cat("  Median pages (EIS):",
    median(pages_analysis$total_pages[pages_analysis$process_type == "EIS"]), "\n\n")

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
theme_catf <- function(base_size = 11, base_family = "Helvetica") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      plot.title = element_text(
        face = "bold", size = rel(1.2), color = catf_navy,
        margin = margin(b = 10)
      ),
      plot.subtitle = element_text(
        size = rel(0.9), color = catf_dark_blue,
        margin = margin(b = 10)
      ),
      plot.caption = element_text(
        size = rel(0.8), color = "gray50", hjust = 1
      ),
      axis.title = element_text(size = rel(0.9), color = catf_navy),
      axis.text = element_text(size = rel(0.85), color = "gray30"),
      axis.line = element_line(color = "gray70", linewidth = 0.3),
      legend.title = element_text(face = "bold", size = rel(0.9), color = catf_navy),
      legend.text = element_text(size = rel(0.85), color = "gray30"),
      legend.position = "bottom",
      legend.key.size = unit(0.8, "lines"),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      strip.text = element_text(face = "bold", size = rel(0.9), color = catf_navy),
      strip.background = element_rect(fill = "gray95", color = NA),
      plot.margin = margin(15, 15, 10, 10)
    )
}

scale_color_catf <- function(...) scale_color_manual(values = catf_palette, ...)
scale_fill_catf <- function(...) scale_fill_manual(values = catf_palette, ...)
scale_color_catf_seq <- function(...) scale_color_gradientn(colors = catf_sequential, ...)
scale_fill_catf_seq <- function(...) scale_fill_gradientn(colors = catf_sequential, ...)

# Set default theme for session
theme_set(theme_catf())

# --------------------------
# HELPER FUNCTIONS
# --------------------------

#' Explode JSON-encoded column to multiple rows
explode_column <- function(df, col_name) {
  df %>%
    mutate(!!col_name := sapply(.data[[col_name]], function(x) {
      if (is.null(x) || length(x) == 0 || (is.character(x) && x == "")) {
        return(NA_character_)
      }
      if (is.character(x) && grepl("^\\[", x)) {
        parsed <- tryCatch(jsonlite::fromJSON(x), error = function(e) x)
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

# --------------------------
# SETUP COMPLETE
# --------------------------

cat("\n=== Deliverable 5 Setup Complete ===\n")
cat("Key objects:\n")
cat("  pages_data     - full merged dataset (", nrow(pages_data), " projects)\n")
cat("  pages_analysis - analysis subset with complete timelines (", nrow(pages_analysis), " projects)\n")
cat("  coverage       - coverage tracking at each filter step\n")
cat("  fra_date       - FRA enactment date (", as.character(fra_date), ")\n")
cat("Output directories:\n")
cat("  Tables:", tables_dir, "\n")
cat("  Figures:", figures_dir, "\n")
cat("CATF brand theme loaded and set as default\n")
