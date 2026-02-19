# --------------------------
# DELIVERABLE 6: IDENTIFICATION QA
# --------------------------

source(here::here("code", "deliverable06", "00_setup.R"))

set.seed(606)

analysis <- prepare_deliverable6_data() %>%
  mutate(
    process_group = toupper(as.character(coalesce(process_group, process_type, dataset_source))),
    transmission_strict = coalesce(project_is_transmission, FALSE),
    transmission_broad = coalesce(project_is_transmission_broad, FALSE),
    geothermal_flag = coalesce(project_is_geothermal, FALSE),
    pipeline_flag = coalesce(project_is_pipeline, FALSE),
    text_has_geothermal_keyword = str_detect(str_to_lower(project_text_full), "\\b(geothermal|enhanced geothermal|egs)\\b")
  )

cat("Rows in D6 clean-energy analysis set:", nrow(analysis), "\n")

# --------------------------
# IDENTIFICATION OVERVIEW
# --------------------------

n_total <- nrow(analysis)
n_transmission_strict <- sum(analysis$transmission_strict, na.rm = TRUE)
n_transmission_broad <- sum(analysis$transmission_broad, na.rm = TRUE)
n_transmission_broad_only <- sum(analysis$transmission_broad & !analysis$transmission_strict, na.rm = TRUE)
n_transmission_strict_not_broad <- sum(analysis$transmission_strict & !analysis$transmission_broad, na.rm = TRUE)
n_transmission_with_length <- sum(analysis$transmission_strict & !is.na(analysis$project_transmission_length_miles), na.rm = TRUE)

n_geothermal <- sum(analysis$geothermal_flag, na.rm = TRUE)
n_geothermal_unknown_phase <- sum(
  analysis$geothermal_flag & coalesce(analysis$project_geothermal_phase, "") == "unknown",
  na.rm = TRUE
)
n_geothermal_none_phase <- sum(
  analysis$geothermal_flag & coalesce(analysis$project_geothermal_phase, "") == "none",
  na.rm = TRUE
)
n_geothermal_keyword_not_flagged <- sum(!analysis$geothermal_flag & analysis$text_has_geothermal_keyword, na.rm = TRUE)

n_pipeline <- sum(analysis$pipeline_flag, na.rm = TRUE)

overview <- tibble(
  metric = c(
    "Total clean-energy projects in D6 analysis",
    "Transmission strict count",
    "Transmission broad count",
    "Transmission broad-only count (possible strict misses)",
    "Transmission strict-not-broad count (rule inconsistency check)",
    "Transmission strict with extracted length",
    "Transmission strict with extracted length (%)",
    "Geothermal flagged count",
    "Geothermal flagged with unknown phase",
    "Geothermal flagged with unknown phase (%)",
    "Geothermal flagged with none phase",
    "Projects with geothermal keyword but not geothermal-flagged",
    "Pipeline flagged count"
  ),
  value = c(
    n_total,
    n_transmission_strict,
    n_transmission_broad,
    n_transmission_broad_only,
    n_transmission_strict_not_broad,
    n_transmission_with_length,
    ifelse(n_transmission_strict > 0, 100 * n_transmission_with_length / n_transmission_strict, NA_real_),
    n_geothermal,
    n_geothermal_unknown_phase,
    ifelse(n_geothermal > 0, 100 * n_geothermal_unknown_phase / n_geothermal, NA_real_),
    n_geothermal_none_phase,
    n_geothermal_keyword_not_flagged,
    n_pipeline
  )
)

write_csv(overview, here(tables_dir, "table_identification_overview.csv"))

by_process <- analysis %>%
  group_by(process_group) %>%
  summarise(
    n_projects = n(),
    n_transmission_strict = sum(transmission_strict, na.rm = TRUE),
    n_transmission_broad = sum(transmission_broad, na.rm = TRUE),
    n_geothermal = sum(geothermal_flag, na.rm = TRUE),
    n_pipeline = sum(pipeline_flag, na.rm = TRUE),
    pct_transmission_strict = 100 * n_transmission_strict / n_projects,
    pct_geothermal = 100 * n_geothermal / n_projects,
    pct_pipeline = 100 * n_pipeline / n_projects,
    .groups = "drop"
  ) %>%
  arrange(process_group)

write_csv(by_process, here(tables_dir, "table_identification_by_process.csv"))

# --------------------------
# MANUAL AUDIT SAMPLES
# --------------------------

prep_audit_cols <- function(df, category) {
  df %>%
    transmute(
      audit_category = category,
      project_id,
      process_group,
      project_state_primary,
      project_title = str_squish(project_title_txt),
      project_type = str_squish(project_type_txt),
      project_description_snippet = str_trunc(str_squish(project_description_txt), 280),
      project_transmission_length_miles,
      project_transmission_length_confidence,
      project_transmission_length_source_text,
      project_geothermal_phase,
      manual_review_label = "",
      manual_review_notes = ""
    )
}

safe_sample_n <- function(df, n_target) {
  if (nrow(df) == 0 || n_target <= 0) return(df[0, ])
  df %>% slice_sample(n = min(n_target, nrow(df)))
}

n_per_category <- 60

transmission_strict_sample <- analysis %>%
  filter(transmission_strict) %>%
  safe_sample_n(n_per_category) %>%
  prep_audit_cols("transmission_strict")

transmission_broad_only_sample <- analysis %>%
  filter(transmission_broad, !transmission_strict) %>%
  safe_sample_n(n_per_category) %>%
  prep_audit_cols("transmission_broad_only")

geothermal_flagged_sample <- analysis %>%
  filter(geothermal_flag) %>%
  safe_sample_n(n_per_category) %>%
  prep_audit_cols("geothermal_flagged")

geothermal_keyword_not_flagged_sample <- analysis %>%
  filter(!geothermal_flag, text_has_geothermal_keyword) %>%
  safe_sample_n(n_per_category) %>%
  prep_audit_cols("geothermal_keyword_not_flagged")

transmission_audit <- bind_rows(transmission_strict_sample, transmission_broad_only_sample)
geothermal_audit <- bind_rows(geothermal_flagged_sample, geothermal_keyword_not_flagged_sample)

write_csv(transmission_audit, here(tables_dir, "table_transmission_identification_audit_sample.csv"))
write_csv(geothermal_audit, here(tables_dir, "table_geothermal_identification_audit_sample.csv"))

cat("Saved outputs to:\n", tables_dir, "\n")
cat("Transmission strict:", n_transmission_strict, "| broad:", n_transmission_broad, "| broad-only:", n_transmission_broad_only, "\n")
cat("Geothermal flagged:", n_geothermal, "| keyword-not-flagged:", n_geothermal_keyword_not_flagged, "\n")
