# --------------------------
# EXPLORATORY: TRANSMISSION LENGTHS
# --------------------------
# Candidate-level audit, multi-candidate review, adjudication QA, and
# length outlier flagging. Run interactively after rebuilding
# projects_transmission.parquet to validate extraction quality before
# re-running the main 01_transmission.R analysis script.

rm(list = ls())
library(here)
source(here::here("phase1", "code", "deliverable06", "00_setup.R"))

analysis <- prepare_deliverable6_data() %>%
  filter(project_is_transmission)

# --------------------------
# MANUAL LENGTH RECODES (QA overrides)
# --------------------------
# Keep in sync with code/deliverable06/01_transmission.R
analysis <- analysis %>%
  mutate(
    project_transmission_length_final = case_when(
      project_id == "ba2da0d34550f2a77a14b8a5a2c1c384"           ~ 26.6,
      project_id == "d65372a8-13b7-3afe-2126-341201c2dce4"        ~ NA_real_,
      project_id == "35677250-c914-f32c-b6b1-4022b39066ed"        ~ 1.34,
      project_id == "f2f52b2c-2327-c14f-73a9-57914bb18e12"        ~ 1.34,
      TRUE ~ project_transmission_length_final
    )
  )

# --------------------------
# EXPLORATORY
# --------------------------

#
# Transmissions
# ----------------------------------------
# Unpack every extracted length candidate to one row per candidate.
# Use this to audit taxonomies, spot width artifacts, and identify
# which projects need LLM adjudication before re-running Python extraction.
transmissions <-
  analysis |>
  select(
    project_id,
    project_transmission_length_miles,   # rule-based only
    project_transmission_length_final,   # LLM if used, else rule-based
    project_transmission_length_taxonomy,
    project_transmission_length_llm_trigger,
    project_transmission_length_llm_status,
    project_transmission_length_llm_reasoning,
    project_transmission_length_candidate_count,
    project_transmission_length_distinct_candidate_count,
    project_transmission_length_selected_candidate_ids,
    project_transmission_length_candidates_json,
    project_transmission_action,
  ) |>
  filter(nchar(coalesce(project_transmission_length_candidates_json, "")) > 2) |>
  mutate(
    parsed = map(
      project_transmission_length_candidates_json,
      ~tryCatch(
        jsonlite::fromJSON(.x, simplifyDataFrame = TRUE),
        error = function(e) NULL
      )
    )
  ) |>
  filter(map_lgl(parsed, ~!is.null(.x) && is.data.frame(.x) && nrow(.x) > 0)) |>
  unnest(parsed) |>
  select(-project_transmission_length_candidates_json) |>
  mutate(
    selected_ids        = map(project_transmission_length_selected_candidate_ids, safe_fromJSON),
    is_selected         = map2_lgl(candidate_id, selected_ids, ~.x %in% .y),
    hint_terms_txt      = map_chr(hint_terms, ~paste(unlist(.x), collapse = ", ")),
    is_width_artifact   = unit_normalized == "miles_from_feet" & value_miles < 0.25
  ) |>
  select(-hint_terms, -selected_ids) |>
  glimpse()


# write
sheet_write(
  data = transmissions,
  ss = "https://docs.google.com/spreadsheets/d/1KicEYrTlXJSk-fzQ2s30S6l8bpPNBlV75pPfWy0NTeI/edit?usp=sharing",
  sheet = "tx"
)

# Taxonomy breakdown across projects
transmissions |>
  distinct(project_id, project_transmission_length_taxonomy, project_transmission_length_llm_trigger) |>
  count(project_transmission_length_taxonomy, project_transmission_length_llm_trigger, name = "n_projects") |>
  arrange(project_transmission_length_taxonomy) |>
  print()

# Action type distribution across all candidates
transmissions |>
  count(project_transmission_action, name = "n_candidates") |>
  arrange(desc(n_candidates)) |>
  print()


#
# Multi-candidate rows: one row per candidate for projects with 2+ distinct values
# ----------------------------------------
transmissions_multiple <-
  transmissions |>
  filter(project_transmission_length_distinct_candidate_count >= 2) |>
  select(
    project_id,
    project_transmission_length_taxonomy,
    project_transmission_length_llm_trigger,
    project_transmission_length_llm_status,
    rule_based_miles  = project_transmission_length_miles,
    final_miles        = project_transmission_length_final,
    candidate_id,
    candidate_value_miles = value_miles,
    candidate_action_type,
    unit_normalized,
    hint_score,
    sentence_has_build_verb,
    is_selected,
    is_width_artifact,
    hint_terms_txt,
    source_text
  ) |>
  arrange(project_id, desc(is_selected), desc(hint_score)) |>
  glimpse()

# write
sheet_write(
  data = transmissions_multiple,
  ss = "https://docs.google.com/spreadsheets/d/1KicEYrTlXJSk-fzQ2s30S6l8bpPNBlV75pPfWy0NTeI/edit?usp=sharing",
  sheet = "tx_multiple"
)


#
# Adjudication review: one row per project with 2+ distinct nontrivial candidates
# ----------------------------------------
# Each row is one project. Candidate values are collapsed to a single string so
# you can review all ambiguous projects at a glance and verify LLM choices.
tx_adjudication <-
  transmissions |>
  filter(project_transmission_length_distinct_candidate_count >= 2) |>
  group_by(
    project_id,
    project_transmission_action,
    project_transmission_length_taxonomy,
    rule_based_miles  = project_transmission_length_miles,
    final_miles       = project_transmission_length_final,
    project_transmission_length_llm_trigger,
    project_transmission_length_llm_status,
    project_transmission_length_llm_reasoning
  ) |>
  summarise(
    n_candidates      = n(),
    candidate_values  = paste(sort(unique(round(value_miles, 3))), collapse = " | "),
    selected_texts    = paste(source_text[is_selected], collapse = " // "),
    .groups = "drop"
  ) |>
  left_join(
    analysis |>
      select(project_id, project_title_txt, project_description_txt),
    by = "project_id"
  ) |>
  arrange(project_id, project_transmission_length_llm_status) |>
  glimpse()

sheet_write(
  data = tx_adjudication,
  ss = "https://docs.google.com/spreadsheets/d/1KicEYrTlXJSk-fzQ2s30S6l8bpPNBlV75pPfWy0NTeI/edit?usp=sharing",
  sheet = "tx_adjudication"
)


#
# Length outlier QA: projects > 200 miles flagged for manual review
# ----------------------------------------
# Single-candidate rows (llm_trigger = FALSE) are never sent to the LLM —
# the length was selected by rule from a single regex hit. These are the most
# likely false positives (system totals, map scale bars, context references).
# Multi-candidate rows (llm_trigger = TRUE) should have been adjudicated by
# the LLM if --run llm was executed; if llm_status = "not_requested" the LLM
# run has not been applied to this parquet yet.

tx_length_outliers <- analysis %>%
  filter(project_transmission_length_final > 200) %>%
  arrange(desc(project_transmission_length_final)) %>%
  select(
    project_id,
    title               = project_title_txt,
    source              = dataset_source,
    length_miles        = project_transmission_length_final,
    confidence          = project_transmission_length_confidence,
    n_distinct_cands    = project_transmission_length_distinct_candidate_count,
    llm_trigger         = project_transmission_length_llm_trigger,
    llm_status          = project_transmission_length_llm_status,
    llm_reasoning       = project_transmission_length_llm_reasoning,
    from_pages          = project_transmission_length_from_pages,
    source_text         = project_transmission_length_source_text
  )

cat("\n=== LENGTH OUTLIERS (> 200 miles) ===\n")
cat("Projects to review:", nrow(tx_length_outliers), "\n")
cat("  Single-candidate — LLM cannot help (check source_text manually):",
    sum(!coalesce(tx_length_outliers$llm_trigger, FALSE)), "\n")
cat("  Multi-candidate — LLM should adjudicate:",
    sum(coalesce(tx_length_outliers$llm_trigger, FALSE)), "\n")
cat("  Multi-candidate still not_requested (LLM not yet run):",
    sum(coalesce(tx_length_outliers$llm_status, "") == "not_requested"), "\n\n")

print(tx_length_outliers, width = Inf, n = Inf)

sheet_write(
  data = tx_length_outliers,
  ss = "https://docs.google.com/spreadsheets/d/1KicEYrTlXJSk-fzQ2s30S6l8bpPNBlV75pPfWy0NTeI/edit?usp=sharing",
  sheet = "tx_length_outliers"
)
