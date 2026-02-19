# --------------------------
# DELIVERABLE 6: SETUP
# --------------------------
# Shared setup for technology-specific analyses.

library(here)
library(arrow)
library(tidyverse)
library(jsonlite)
library(scales)
library(googlesheets4)

# --------------------------
# FILE PATHS
# --------------------------

timeline_ce_path <- here("data", "analysis", "projects_timeline_bert.parquet")
timeline_ea_path <- here("data", "analysis", "projects_timeline_bert_ea_llm.parquet")
timeline_eis_path <- here("data", "analysis", "projects_timeline_bert_eis_llm.parquet")
projects_combined_path <- here("data", "analysis", "projects_combined.parquet")
output_dir <- here("output", "deliverable6")
tables_dir <- here("output", "deliverable6", "tables")
figures_dir <- here("output", "deliverable6", "figures")

# Create output directories if they don't exist
for (d in c(output_dir, tables_dir, figures_dir)) {
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
}

load_timeline_for_deliverable6 <- function() {
  required_paths <- c(
    CE = timeline_ce_path,
    EA = timeline_ea_path,
    EIS = timeline_eis_path
  )

  missing_files <- required_paths[!file.exists(required_paths)]
  if (length(missing_files) > 0) {
    stop(
      "Missing required timeline file(s):\n",
      paste0(" - ", names(missing_files), ": ", unname(missing_files), collapse = "\n")
    )
  }

  ce_df <- read_parquet(timeline_ce_path) %>%
    mutate(timeline_input_file = basename(timeline_ce_path))
  ea_df <- read_parquet(timeline_ea_path) %>%
    mutate(timeline_input_file = basename(timeline_ea_path))
  eis_df <- read_parquet(timeline_eis_path) %>%
    mutate(timeline_input_file = basename(timeline_eis_path))

  timeline_raw <- bind_rows(ce_df, ea_df, eis_df)

  if (!"dataset_source" %in% names(timeline_raw)) {
    timeline_raw <- timeline_raw %>% mutate(dataset_source = NA_character_)
  }
  if (!"llm_initiation_date" %in% names(timeline_raw)) {
    timeline_raw <- timeline_raw %>% mutate(llm_initiation_date = as.Date(NA))
  }
  if (!"llm_decision_date" %in% names(timeline_raw)) {
    timeline_raw <- timeline_raw %>% mutate(llm_decision_date = as.Date(NA))
  }

  timeline_raw %>%
    mutate(
      dataset_source = toupper(as.character(coalesce(dataset_source, process_type))),
      process_type = toupper(as.character(coalesce(process_type, dataset_source))),
      llm_initiation_date = as.Date(llm_initiation_date),
      llm_decision_date = as.Date(llm_decision_date),
      bert_initiation_date_final = as.Date(bert_initiation_date_final),
      bert_decision_date_final = as.Date(bert_decision_date_final),
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
      # Keep legacy fields used by downstream scripts, now harmonized.
      bert_initiation_date_final = timeline_initiation_date_final,
      bert_decision_date_final = timeline_decision_date_final
    )
}

cat("Loading timeline data from:\n",
    " -", timeline_ce_path, "\n",
    " -", timeline_ea_path, "\n",
    " -", timeline_eis_path, "\n")
timeline <- load_timeline_for_deliverable6()
cat("Timeline rows loaded:", nrow(timeline), "\n")
cat("Process types:", paste(sort(unique(as.character(na.omit(timeline$process_type)))), collapse = ", "), "\n")

# Merge technology-specific extraction fields from projects_combined (Python output).
tech_cols <- c(
  "project_id",
  "project_is_transmission",
  "project_is_transmission_broad",
  "project_is_transmission_strict",
  "project_is_geothermal",
  "project_is_pipeline",
  "project_is_carbon_pipeline",
  "project_is_hydrogen_pipeline",
  "project_is_natural_gas_pipeline",
  "project_has_transmission_type_tag",
  "project_has_transmission_build_text",
  "project_transmission_length_candidates_json",
  "project_transmission_length_candidate_count",
  "project_transmission_length_distinct_candidate_count",
  "project_transmission_length_taxonomy",
  "project_transmission_length_selection_method",
  "project_transmission_length_selected_candidate_ids",
  "project_transmission_length_llm_trigger",
  "project_transmission_length_llm_used",
  "project_transmission_length_llm_status",
  "project_geothermal_phase",
  "project_pipeline_group",
  "project_transmission_length_miles",
  "project_transmission_length_confidence",
  "project_transmission_length_source_text",
  "project_pipeline_length_miles",
  "project_pipeline_length_confidence",
  "project_pipeline_length_source_text",
  "project_pipeline_length_candidates_json",
  "project_pipeline_length_candidate_count",
  "project_pipeline_length_distinct_candidate_count",
  "project_energy_type"
)

projects <- timeline
if (file.exists(projects_combined_path)) {
  cat("Loading extraction dataset from:", projects_combined_path, "\n")
  combined <- read_parquet(projects_combined_path)
  available_tech_cols <- intersect(tech_cols, names(combined))
  if (!"project_id" %in% available_tech_cols) {
    stop("projects_combined.parquet is missing project_id; cannot merge technology columns.")
  }
  projects <- timeline %>%
    select(-any_of(setdiff(available_tech_cols, "project_id"))) %>%
    left_join(combined %>% select(all_of(available_tech_cols)), by = "project_id")
  cat("Merged technology columns:", paste(setdiff(available_tech_cols, "project_id"), collapse = ", "), "\n")
} else {
  cat("projects_combined.parquet not found; technology fields may be missing.\n")
}

# --------------------------
# HELPERS
# --------------------------

safe_fromJSON <- function(x) {
  if (is.null(x) || is.na(x) || !nzchar(as.character(x))) return(character(0))
  if (is.list(x)) return(unlist(x))
  if (is.character(x) && grepl("^\\[", x)) {
    out <- tryCatch(jsonlite::fromJSON(x), error = function(e) character(0))
    return(as.character(out))
  }
  as.character(x)
}

textify <- function(x) {
  vals <- safe_fromJSON(x)
  vals <- vals[!is.na(vals) & nzchar(trimws(vals))]
  paste(vals, collapse = " ")
}

extract_primary_state <- function(x) {
  vals <- safe_fromJSON(x)
  if (length(vals) == 0) return(NA_character_)
  vals[[1]]
}

state_region_map <- tibble(
  state = c(state.name, "District of Columbia"),
  region = c(as.character(state.region), "South")
)

add_timeline_metrics <- function(df) {
  df %>%
    mutate(
      bert_decision_date_final = as.Date(bert_decision_date_final),
      bert_initiation_date_final = as.Date(bert_initiation_date_final),
      bert_duration_days_final = as.numeric(bert_decision_date_final - bert_initiation_date_final),
      bert_duration_months_final = bert_duration_days_final / 30.44,
      project_state_primary = map_chr(project_state, extract_primary_state),
      project_region = as.character(state_region_map$region[match(project_state_primary, state_region_map$state)]),
      project_region = coalesce(project_region, "Unknown"),
      process_group = toupper(as.character(coalesce(dataset_source, process_type)))
    )
}

add_deliv6_fallback_features <- function(df) {
  df2 <- df %>%
    mutate(
      project_title_txt = map_chr(project_title, textify),
      project_description_txt = map_chr(project_description, textify),
      project_type_txt = map_chr(project_type, textify),
      project_text_full = str_squish(str_c(project_title_txt, project_description_txt, project_type_txt, sep = " "))
    )

  # Do not derive technology fields in R. Keep missing extraction fields as NA.
  ensure_missing_col <- function(d, col, default) {
    if (!col %in% names(d)) d[[col]] <- default
    d
  }

  logical_cols <- c(
    "project_is_transmission",
    "project_is_transmission_broad",
    "project_is_transmission_strict",
    "project_is_geothermal",
    "project_is_pipeline",
    "project_is_carbon_pipeline",
    "project_is_hydrogen_pipeline",
    "project_is_natural_gas_pipeline",
    "project_has_transmission_type_tag",
    "project_has_transmission_build_text",
    "project_transmission_length_llm_trigger",
    "project_transmission_length_llm_used"
  )
  numeric_cols <- c(
    "project_transmission_length_miles",
    "project_pipeline_length_miles",
    "project_transmission_length_candidate_count",
    "project_transmission_length_distinct_candidate_count",
    "project_pipeline_length_candidate_count",
    "project_pipeline_length_distinct_candidate_count"
  )
  character_cols <- c(
    "project_geothermal_phase",
    "project_pipeline_group",
    "project_transmission_length_confidence",
    "project_transmission_length_source_text",
    "project_pipeline_length_confidence",
    "project_pipeline_length_source_text",
    "project_transmission_length_candidates_json",
    "project_transmission_length_taxonomy",
    "project_transmission_length_selection_method",
    "project_transmission_length_selected_candidate_ids",
    "project_transmission_length_llm_status",
    "project_pipeline_length_candidates_json"
  )

  for (col in logical_cols) df2 <- ensure_missing_col(df2, col, NA)
  for (col in numeric_cols) df2 <- ensure_missing_col(df2, col, NA_real_)
  for (col in character_cols) df2 <- ensure_missing_col(df2, col, NA_character_)

  df2
}

prepare_deliverable6_data <- function(df = projects, clean_only = TRUE) {
  out <- df %>%
    add_deliv6_fallback_features() %>%
    add_timeline_metrics()

  if (clean_only && "project_energy_type" %in% names(out)) {
    out <- out %>% filter(project_energy_type == "Clean")
  }

  out
}

# Simple CATF-inspired colors used across scripts
catf_dark_blue <- "#0047BB"
catf_teal <- "#00AE8D"
catf_magenta <- "#C22A90"
catf_light_blue <- "#8AB7E9"
catf_navy <- "#002169"
