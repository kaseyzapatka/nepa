# --------------------------
# DELIVERABLE 2: SETUP
# --------------------------
# Shared setup for deliverable 2 (Programmatic & Tiered Reviews)
# Loads reviews data and merges with timeline for duration analysis
#
# Key output objects:
#   reviews            - full reviews dataset (clean energy EA/EIS projects)
#   reviews_tl         - reviews merged with timeline data (includes duration_days)
#   duration_data      - subset with valid duration (positive, non-missing)
#   non_standard       - only programmatic + tiered reviews
#   reviews_long_agency - unnested by lead agency (for agency-level analyses)
#   reviews_long_state  - unnested by state (for geographic analyses)

# --------------------------
# LIBRARIES
# --------------------------

library(here)
library(arrow)
library(tidyverse)
library(jsonlite)
library(scales)
library(gt)

source(here::here("code", "utils", "utils.R"))

# --------------------------
# FILE PATHS
# --------------------------

reviews_path      <- here("data", "analysis", "projects_reviews.parquet")
timeline_ea_path  <- here("data", "analysis", "projects_timeline_bert_ea_llm.parquet")
timeline_eis_path <- here("data", "analysis", "projects_timeline_bert_eis_llm.parquet")

output_dir  <- here("output", "deliverable2")
tables_dir  <- here("output", "deliverable2", "tables")
figures_dir <- here("output", "deliverable2", "figures")

dir.create(tables_dir,  showWarnings = FALSE, recursive = TRUE)
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------
# CONSTANTS
# --------------------------

review_type_levels <- c("Standard", "Programmatic", "Tiered")

review_type_colors <- c(
  "Standard"     = "gray75",
  "Programmatic" = "#0047BB",
  "Tiered"       = "#00B5E2"
)

# --------------------------
# HELPER: UNNEST LIST / JSON COLUMN
# --------------------------
# Handles both native Arrow list columns and JSON-string columns.
# Returns a long-form data frame with one value per row.

explode_col <- function(df, col_name) {
  col <- df[[col_name]]
  if (is.list(col)) {
    # Arrow native list column
    df %>%
      mutate(!!col_name := lapply(.data[[col_name]], function(x) {
        if (is.null(x) || length(x) == 0) NA_character_
        else as.character(x)
      })) %>%
      unnest(cols = !!sym(col_name), keep_empty = TRUE)
  } else {
    # JSON-string column
    df %>%
      mutate(!!col_name := lapply(as.character(.data[[col_name]]), function(x) {
        if (is.na(x) || x == "" || x == "[]") return(NA_character_)
        tryCatch(
          as.character(jsonlite::fromJSON(x)),
          error = function(e) NA_character_
        )
      })) %>%
      unnest(cols = !!sym(col_name), keep_empty = TRUE)
  }
}

# function to review samples 
sample_review <- function(review_type_filter, process_type_filter = NULL) {
  reviews |>
    filter(
      review_type == review_type_filter,
      if (!is.null(process_type_filter)) process_type == process_type_filter else TRUE
    ) |>
    select(project_id, project_type, process_type, project_review_type:review_type) |>
    slice_sample(n = 1) |> 
    glimpse()
}

# --------------------------
# LOAD REVIEWS
# --------------------------

cat("Loading reviews from:", reviews_path, "\n")
reviews_raw <- read_parquet(reviews_path) %>% as_tibble()
cat("  Total projects:", nrow(reviews_raw), "\n")

reviews <- reviews_raw %>%
  mutate(
    review_type  = factor(
      str_to_sentence(project_review_type),
      levels = review_type_levels
    ),
    process_type = factor(dataset_source, levels = c("EA", "EIS"))
  )

cat("Review type breakdown:\n")
print(count(reviews, review_type))

# --------------------------
# UNNEST AGENCY AND STATE
# --------------------------

reviews_long_agency <- reviews %>%
  select(project_id, project_review_type, review_type, process_type,
         lead_agency_harmonized) %>%
  explode_col("lead_agency_harmonized") %>%
  rename(agency = lead_agency_harmonized) %>%
  filter(!is.na(agency), agency != "")

reviews_long_state <- reviews %>%
  select(project_id, project_review_type, review_type, process_type,
         project_state) %>%
  explode_col("project_state") %>%
  rename(state = project_state) %>%
  filter(!is.na(state), state != "")

cat("  Agency records (unnested):", nrow(reviews_long_agency), "\n")
cat("  State records  (unnested):", nrow(reviews_long_state), "\n\n")

# --------------------------
# LOAD AND MERGE TIMELINE
# --------------------------

cat("Loading timeline data...\n")
if (!file.exists(timeline_ea_path))  stop("Missing EA timeline file")
if (!file.exists(timeline_eis_path)) stop("Missing EIS timeline file")

timeline <- bind_rows(
  read_parquet(timeline_ea_path)  %>% select(project_id, llm_initiation_date, llm_decision_date),
  read_parquet(timeline_eis_path) %>% select(project_id, llm_initiation_date, llm_decision_date)
) %>%
  distinct(project_id, .keep_all = TRUE) %>%
  mutate(
    initiation_date = as.Date(llm_initiation_date),
    decision_date   = as.Date(llm_decision_date)
  ) %>%
  select(project_id, initiation_date, decision_date)

cat("  Timeline records:", nrow(timeline), "\n")

# Patch in targeted re-adjudication results for incomplete non-standard projects
targeted_path <- here("data", "analysis", "projects_timeline_targeted_llm.parquet")
if (file.exists(targeted_path)) {
  targeted <- read_parquet(targeted_path) %>%
    select(project_id,
           targeted_initiation_date = llm_initiation_date,
           targeted_decision_date   = llm_decision_date)
  timeline <- timeline %>%
    left_join(targeted, by = "project_id") %>%
    mutate(
      initiation_date = coalesce(as.Date(targeted_initiation_date), initiation_date),
      decision_date   = coalesce(as.Date(targeted_decision_date),   decision_date)
    ) %>%
    select(-targeted_initiation_date, -targeted_decision_date)
  cat("  Targeted re-adjudication applied:", nrow(targeted), "projects patched\n")
}

# Full timeline with bert_dates_json (for candidate inspection)
tl_full <- bind_rows(
  read_parquet(timeline_ea_path),
  read_parquet(timeline_eis_path)
) |> distinct(project_id, .keep_all = TRUE)

if (file.exists(targeted_path)) {
  tl_full <- tl_full |>
    left_join(
      targeted |> select(project_id,
                          t_init = targeted_initiation_date,
                          t_dec  = targeted_decision_date),
      by = "project_id"
    ) |>
    mutate(
      llm_initiation_date = coalesce(as.Date(t_init), as.Date(llm_initiation_date)),
      llm_decision_date   = coalesce(as.Date(t_dec),  as.Date(llm_decision_date))
    ) |>
    select(-t_init, -t_dec)
}

# --------------------------
# MANUAL DATE OVERRIDES (TEMPORARY — presentation 2026-03-06)
# TODO: Integrate into pipeline after Thursday. See notes/status/reviews_status.md.
# Dates verified manually by inspecting document text, filenames, and NOI records.
# --------------------------

noi_cf2 <- as.Date(tl_full |>
  filter(project_id == "cf2fbe90d43ac57a9460fa857f34af6c") |>
  pull(noi_publication_date) |> first())

manual_overrides <- tibble(
  project_id           = c(
    "cf2fbe90d43ac57a9460fa857f34af6c",  # initiation <- NOI publication date
    "f95ec9530b352e3dd46e6473cb80dccf",  # decision  <- April 2019 (date in EA filename)
    "49cdaa3ff2e6c505c6822e8e9803eb9b",  # decision  <- May 2023 (date in draft filename)
    "4af8ad4f47941e4ccb53fe4349c258c3",  # decision  <- September 1995 (p.3 of FEIS)
                                         # initiation <- ~Jan 1993 (est.; BPA RFP Sept 1992; DEIS March 1995)
    "00d09887554d7ab68e49e9ab628583bf",  # decision  <- June 2025 (p.1 of DEIS)
    "8d13822f3d8b469efcdb2706caa463c7",  # decision  <- March 2022 (TVA Final EIS target)
    "6890cacf404f0068be5c1e94470e6c58",  # decision  <- Feb 2022 (est.; companion project a4b76252 decided 2022-02-25)
    "5445a80334ce78493711d6bc3d24fd81"   # decision  <- Sept 2012 (est.; FEIS era; 2009 ROD in doc belongs to prior EIS)
  ),
  override_initiation  = as.Date(c(noi_cf2, NA, NA, "1993-01-01", NA, NA, NA, NA)),
  override_decision    = as.Date(c(NA, "2019-04-01", "2023-05-01",
                                   "1995-09-01", "2025-06-01", "2022-03-01",
                                   "2022-02-25", "2012-09-01"))
)

# Patch timeline (-> reviews_tl, duration_data)
timeline <- timeline |>
  left_join(manual_overrides, by = "project_id") |>
  mutate(
    initiation_date = coalesce(override_initiation, initiation_date),
    decision_date   = coalesce(override_decision,   decision_date)
  ) |>
  select(-override_initiation, -override_decision)

# Patch tl_full (-> browse_ns, inspect_candidates)
tl_full <- tl_full |>
  left_join(manual_overrides, by = "project_id") |>
  mutate(
    llm_initiation_date = coalesce(override_initiation, as.Date(llm_initiation_date)),
    llm_decision_date   = coalesce(override_decision,   as.Date(llm_decision_date))
  ) |>
  select(-override_initiation, -override_decision)

cat("  Manual overrides applied:", nrow(manual_overrides), "projects\n")

# Coverage browse table: 161 non-standard projects with timeline status
browse_ns <- reviews |>
  filter(review_type %in% c("Programmatic", "Tiered")) |>
  select(project_id, process_type, review_type) |>
  left_join(
    tl_full |> select(project_id, llm_initiation_date, llm_decision_date, llm_decision_mode),
    by = "project_id"
  ) |>
  mutate(
    has_initiation = !is.na(llm_initiation_date),
    has_decision   = !is.na(llm_decision_date),
    complete       = has_initiation & has_decision
  ) |>
  arrange(complete, review_type, process_type)

# Inspect BERT date candidates for a specific project
# Shows all dates BERT found (bert_dates_json). LLM may have seen fewer after filtering.
# Use inspect_llm_prompt(pid) to see exactly what the LLM received.
inspect_candidates <- function(pid) {
  row <- tl_full |> filter(project_id == pid)
  if (nrow(row) == 0) { message("Project not found"); return(invisible(NULL)) }
  json_str <- row$bert_dates_json[[1]]
  if (is.null(json_str) || is.na(json_str)) { message("No candidates"); return(invisible(NULL)) }
  dates <- jsonlite::fromJSON(json_str, simplifyDataFrame = TRUE) |> as_tibble()
  # Normalize column names (field names vary slightly across pipeline versions)
  if ("type" %in% names(dates) && !"dtype" %in% names(dates))
    dates <- rename(dates, dtype = type)
  if ("context_cleaned" %in% names(dates) && !"context" %in% names(dates))
    dates <- rename(dates, context = context_cleaned)
  cat("Total BERT candidates:", nrow(dates), "| LLM saw:", row$llm_adj_n_candidates[[1]], "| mode:", row$llm_decision_mode[[1]], "\n\n")
  dates |>
    select(any_of(c("date", "dtype", "doc_type", "confidence", "bert_confidence", "context", "source"))) |>
    arrange(date) |>
    print(n = 500)
}

# Show the raw prompt that was sent to the LLM for a project
inspect_llm_prompt <- function(pid) {
  row <- tl_full |> filter(project_id == pid)
  if (nrow(row) == 0) { message("Project not found"); return(invisible(NULL)) }
  prompt <- row$llm_adj_prompt[[1]]
  if (is.null(prompt) || is.na(prompt)) { message("No LLM prompt stored"); return(invisible(NULL)) }
  cat(prompt)
}

reviews_tl <- reviews %>%
  left_join(timeline, by = "project_id") %>%
  mutate(
    duration_days   = as.numeric(decision_date - initiation_date),
    duration_months = duration_days / 30.44,
    has_duration    = !is.na(duration_days) & duration_days > 0
  )

duration_data <- reviews_tl %>% filter(has_duration)

cat("Projects with valid duration by review type:\n")
print(duration_data %>% count(review_type, process_type))

# --------------------------
# NON-STANDARD SUBSET
# --------------------------

non_standard <- reviews %>%
  filter(project_review_type %in% c("programmatic", "tiered")) %>%
  mutate(review_type = droplevels(review_type))

cat("\nNon-standard reviews:", nrow(non_standard), "\n")

# --------------------------
# CATF BRAND THEME
# --------------------------

catf_dark_blue  <- "#0047BB"
catf_blue       <- "#00B5E2"
catf_magenta    <- "#C22A90"
catf_purple     <- "#75246C"
catf_lime       <- "#93D500"
catf_teal       <- "#00AE8D"
catf_light_blue <- "#8AB7E9"
catf_navy       <- "#002169"

catf_palette <- c(
  "#0047BB", "#00B5E2", "#00AE8D", "#93D500",
  "#C22A90", "#75246C", "#8AB7E9", "#002169"
)

theme_catf <- function(base_size = 11, base_family = "Helvetica") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      plot.title       = element_text(face = "bold", size = rel(1.2), color = catf_navy,
                                      margin = margin(b = 10)),
      plot.subtitle    = element_text(size = rel(0.9), color = catf_dark_blue,
                                      margin = margin(b = 10)),
      plot.caption     = element_text(size = rel(0.8), color = "gray50", hjust = 1),
      axis.title       = element_text(size = rel(0.9), color = catf_navy),
      axis.text        = element_text(size = rel(0.85), color = "gray30"),
      axis.line        = element_line(color = "gray70", linewidth = 0.3),
      legend.title     = element_text(face = "bold", size = rel(0.9), color = catf_navy),
      legend.text      = element_text(size = rel(0.85), color = "gray30"),
      legend.position  = "bottom",
      legend.key.size  = unit(0.8, "lines"),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      strip.text       = element_text(face = "bold", size = rel(0.9), color = catf_navy),
      strip.background = element_rect(fill = "gray95", color = NA),
      plot.margin      = margin(15, 15, 10, 10)
    )
}

scale_color_catf <- function(...) scale_color_manual(values = catf_palette, ...)
scale_fill_catf  <- function(...) scale_fill_manual(values = catf_palette, ...)

theme_set(theme_catf())

# --------------------------
# SETUP COMPLETE
# --------------------------

cat("\n=== Deliverable 2 Setup Complete ===\n")
cat("Key objects:\n")
cat("  reviews            -", nrow(reviews), "projects\n")
cat("  reviews_tl         -", nrow(reviews_tl), "projects (with timeline joined)\n")
cat("  duration_data      -", nrow(duration_data), "projects with valid duration\n")
cat("  non_standard       -", nrow(non_standard), "programmatic + tiered projects\n")
cat("  reviews_long_agency -", nrow(reviews_long_agency), "rows (agency-unnested)\n")
cat("  reviews_long_state  -", nrow(reviews_long_state), "rows (state-unnested)\n")
cat("  tl_full            -", nrow(tl_full), "projects (full timeline with candidates)\n")
cat("  browse_ns          -", nrow(browse_ns), "non-standard projects; complete:", sum(browse_ns$complete), "\n")
cat("Output directories:\n")
cat("  Tables: ", tables_dir, "\n")
cat("  Figures:", figures_dir, "\n")

# --------------------------
# MANUAL OVERRIDE VERIFICATION
# --------------------------
# Confirm the 6 manually-overridden projects reached reviews_tl / duration_data.
# Expected: 5 of 6 in duration_data (Columbia Wind Farm missing initiation date).
cat("\nManual override verification (6 projects):\n")
reviews_tl %>%
  filter(project_id %in% manual_overrides$project_id) %>%
  select(project_id, review_type, process_type, initiation_date, decision_date, duration_days, has_duration) %>%
  print(n = 10, width = Inf)
