# --------------------------
# DELIVERABLE 6: TRANSMISSION LINES
# --------------------------

source(here::here("code", "deliverable06", "00_setup.R"))

analysis <- prepare_deliverable6_data() %>%
  filter(project_is_transmission)

cat("Transmission projects:", nrow(analysis), "\n")
max_year <- as.integer(format(Sys.Date(), "%Y"))

# --------------------------
# EXPLORATORY
# --------------------------

#
# Create candidates
# ----------------------------------------
# Unpack every extracted length candidate to one row per candidate.
# Use this to audit taxonomies, spot width artifacts, and identify
# which projects need LLM adjudication before re-running Python extraction.
tx_candidates <- analysis |>
  select(
    project_id,
    project_transmission_length_miles,
    project_transmission_length_taxonomy,
    project_transmission_length_llm_trigger,
    project_transmission_length_llm_status,
    project_transmission_length_candidate_count,
    project_transmission_length_distinct_candidate_count,
    project_transmission_length_selected_candidate_ids,
    project_transmission_length_candidates_json
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

cat("Total candidates:", nrow(tx_candidates), "\n")
cat("Width artifact candidates:", sum(tx_candidates$is_width_artifact), "\n")
cat("Projects with llm_trigger=TRUE:", sum(analysis$project_transmission_length_llm_trigger, na.rm = TRUE), "\n")

# Taxonomy breakdown across projects
tx_candidates |>
  distinct(project_id, project_transmission_length_taxonomy, project_transmission_length_llm_trigger) |>
  count(project_transmission_length_taxonomy, project_transmission_length_llm_trigger, name = "n_projects") |>
  arrange(project_transmission_length_taxonomy) |>
  print()

# Action type distribution across all candidates
tx_candidates |>
  count(candidate_action_type, name = "n_candidates") |>
  arrange(desc(n_candidates)) |>
  print()

# Spot-check a specific project (Southline: expect new_build=240mi, upgrade=120mi)
tx_candidates |>
  filter(project_id == "c87a153c-f0c6-bd71-17e1-7e01ea9816a5") |>
  select(candidate_id, value_miles, unit_normalized, hint_score,
         candidate_action_type, sentence_has_build_verb, is_selected, is_width_artifact, source_text) |>
  print()

#
# Multi-candidate rows: one row per candidate for projects with 2+ distinct values
# ----------------------------------------
# Multi-candidate rows: one row per candidate for projects with 2+ distinct values
tx_multi_candidates <- tx_candidates |>
  filter(project_transmission_length_distinct_candidate_count >= 2) |>
  select(
    project_id,
    project_transmission_length_taxonomy,
    project_transmission_length_llm_trigger,
    project_transmission_length_llm_status,
    selected_length_miles = project_transmission_length_miles,
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
  arrange(project_id, desc(is_selected), desc(hint_score))

# write
sheet_write(
  data = tx_multi_candidates,
  ss = "https://docs.google.com/spreadsheets/d/1KicEYrTlXJSk-fzQ2s30S6l8bpPNBlV75pPfWy0NTeI/edit?usp=sharing",
  sheet = "tx_multi_candidates"
)


#
# UNKNOWN: Sample of 'unknown' action type cases for review — used to decide whether
# ----------------------------------------
# to add more regex patterns or accept the unknowns as a residual category.
tx_unknown_sample <- tx_candidates |>
  filter(candidate_action_type == "unknown", value_miles >= 0.25) |>
  #slice_sample(n = min(40, n())) |>
  slice_sample(n = 40) |>
  select(
    project_id,
    candidate_value_miles = value_miles,
    hint_score,
    sentence_has_build_verb,
    source_text
  ) |> 
  glimpse()

cat("Unknown action type candidates (>= 0.25 mi):",
    sum(tx_candidates$candidate_action_type == "unknown" & tx_candidates$value_miles >= 0.25), "\n")

sheet_write(
  data = tx_unknown_sample,
  ss = "https://docs.google.com/spreadsheets/d/1KicEYrTlXJSk-fzQ2s30S6l8bpPNBlV75pPfWy0NTeI/edit?usp=sharing",
  sheet = "tx_action_unknown_sample"
)

cat("Multi-candidate rows (for review):", nrow(tx_multi_candidates), "\n")






#
# example 2 
# ----------------------------------------
# the LLM gets this right
example2 <- 
tx_candidates |> 
  filter(project_id == "3e996a98-e88f-3af5-3b72-c4c4cc6b1152") |> 
  glimpse()

sheet_write(
  data = example2,
  ss = "https://docs.google.com/spreadsheets/d/1KicEYrTlXJSk-fzQ2s30S6l8bpPNBlV75pPfWy0NTeI/edit?usp=sharing",
  sheet = "example2"
)

#
# example 3
# ----------------------------------------

# the LLM gets this right
sample <- 
  tx_candidates |> 
  select(project_id) |> 
  slice_sample(n = 1) |> 
  pull() |> 
  print()

# 4d5e6399-df32-1bf0-9c39-73d09ba1df01
example3 <- 
  tx_candidates |> 
  filter(project_id %in% sample) |> 
  select(project_id, project_transmission_length_miles, project_transmission_length_taxonomy, value_miles, matched_text,source_text) |> 
  View()

#
# example 4
# ----------------------------------------

# the LLM gets this right
sample <- 
  tx_candidates |> 
  select(project_id) |> 
  slice_sample(n = 1) |> 
  pull() |> 
  print()

# 4d5e6399-df32-1bf0-9c39-73d09ba1df01
example4 <- 
  tx_candidates |> 
  filter(project_id %in% sample) |> 
  select(project_id, project_transmission_length_miles,project_transmission_length_taxonomy, value_miles, matched_text,source_text) |> 
  glimpse()

analysis |> glimpse()

analysis |> 
  filter(project_transmission_length_llm_used == TRUE) |>
  select(
    #project_id,
    lng  = project_transmission_length_miles,
    #project_transmission_length_llm_status,
    conf=project_transmission_length_confidence,
    txt =project_transmission_length_source_text
  ) |>
  print(n = Inf)

analysis |>
  count(project_transmission_length_taxonomy, project_transmission_length_llm_status) |>
  arrange(desc(n))

analysis |> 
  filter(project_transmission_length_taxonomy  == "do_not_sum") |> 
  select(id = project_id,lgth =  project_transmission_length_miles, txt =  project_transmission_length_source_text) |> 
  arrange(id) |> 
  select(lgth, txt) |> 
  print(n = 100)


analysis |> 
  #filter(project_id == "06ee24b6-e7bd-10d4-4924-31154372b4a3") |> 
  #filter(project_id == "29402d2a-61cf-25dc-5050-bb2f2d62ff48") |> 
  #filter(project_id == "88637c69-789b-99df-4593-7fb2601ea8d9") |> 
  #filter(project_id == "8d4f94cf-0cab-3ccf-00a0-c7c18dbfb2b9") |> 
  filter(project_id == "3fbe2462-7af6-4c8f-5613-90e8dc9bcc7c") |> 
  glimpse()

# LLM-trigger projects only (the ones that need adjudication)
tx_llm_trigger <- tx_candidates |>
  filter(project_transmission_length_llm_trigger == TRUE) |>
  arrange(project_id, desc(is_selected), desc(hint_score))

cat("LLM-trigger candidate rows:", nrow(tx_llm_trigger), "\n")


#
# Transmissions
# ----------------------------------------

transmissions <-
  analysis |>
  select(
    project_id, project_type, project_description, dataset_source, project_state,
    bert_initiation_date_final, bert_decision_date_final,
    project_is_transmission, project_transmission_length_miles,
    project_transmission_length_confidence, project_transmission_length_taxonomy,
    project_transmission_length_llm_trigger, project_transmission_length_source_text,
    bert_duration_days_final, bert_duration_months_final
  ) |>
  #filter(!is.na(bert_duration_months_final)) |> 
  glimpse()

sheet_write(
  data = transmissions,
  ss = "https://docs.google.com/spreadsheets/d/1KicEYrTlXJSk-fzQ2s30S6l8bpPNBlV75pPfWy0NTeI/edit?usp=sharing",
  sheet = "01_transmission"
)


# Project-level count summary
projects |> count(project_is_transmission)
projects |> select(project_id, project_title) |>  filter(project_id == "1aff267e-235b-abb2-347a-92d3ff989575") |> glimpse() # why is this still being pulled in?>
projects |> select(project_id, project_title) |>  filter(project_id == "284f25aa-e022-7781-51c0-d338390aa866") |> pull(project_title)
projects |> select(project_id, project_title, project_type) |>  filter(project_id == "284f25aa-e022-7781-51c0-d338390aa866") |> glimpse()
projects |> select(project_id, project_title, project_type) |>  filter(project_id == "29402d2a-61cf-25dc-5050-bb2f2d62ff48") |> glimpse()


# --------------------------
# TABLES
# --------------------------

tbl_transmission_summary <- tibble(
  metric = c(
    "Transmission projects",
    "With extracted length (miles)",
    "With calculable duration (days)",
    "Multi-state projects"
  ),
  value = c(
    nrow(analysis),
    sum(!is.na(analysis$project_transmission_length_miles)),
    sum(!is.na(analysis$bert_duration_days_final) & analysis$bert_duration_days_final >= 0),
    sum(coalesce(analysis$project_multi_state, FALSE))
  )
)

write_csv(tbl_transmission_summary, here(tables_dir, "table_transmission_summary.csv"))

analysis_len <- analysis %>%
  mutate(
    length_bin = case_when(
      is.na(project_transmission_length_miles) ~ "Missing",
      project_transmission_length_miles < 1 ~ "<1 mile",
      project_transmission_length_miles < 10 ~ "1-10 miles",
      project_transmission_length_miles < 50 ~ "10-50 miles",
      project_transmission_length_miles < 100 ~ "50-100 miles",
      TRUE ~ "100+ miles"
    ),
    length_bin = factor(length_bin, levels = c("<1 mile", "1-10 miles", "10-50 miles", "50-100 miles", "100+ miles", "Missing"))
  )

tbl_length_bins <- analysis_len %>%
  group_by(length_bin) %>%
  summarise(
    n_projects = n(),
    median_duration_days = median(bert_duration_days_final, na.rm = TRUE),
    p90_duration_days = quantile(bert_duration_days_final, 0.9, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(across(c(median_duration_days, p90_duration_days), ~ ifelse(is.finite(.x), round(.x, 1), NA_real_)))

write_csv(tbl_length_bins, here(tables_dir, "table_transmission_length_bins.csv"))

tbl_state_region <- analysis %>%
  group_by(project_region, project_state_primary) %>%
  summarise(
    n_projects = n(),
    median_length_miles = median(project_transmission_length_miles, na.rm = TRUE),
    median_duration_days = median(bert_duration_days_final, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n_projects))

write_csv(tbl_state_region, here(tables_dir, "table_transmission_state_region.csv"))

corr_data <- analysis %>%
  transmute(
    length_miles = project_transmission_length_miles,
    duration_days = bert_duration_days_final,
    n_dates = bert_n_dates_found,
    doc_count = project_doc_count,
    multi_state = as.numeric(coalesce(project_multi_state, FALSE))
  )

corr_pairs <- tribble(
  ~x, ~y,
  "length_miles", "duration_days",
  "length_miles", "n_dates",
  "length_miles", "doc_count",
  "length_miles", "multi_state"
)

tbl_corr <- corr_pairs %>%
  rowwise() %>%
  mutate(
    n_complete = sum(complete.cases(corr_data[[x]], corr_data[[y]])),
    correlation = ifelse(n_complete > 2, cor(corr_data[[x]], corr_data[[y]], use = "complete.obs"), NA_real_)
  ) %>%
  ungroup()

write_csv(tbl_corr, here(tables_dir, "table_transmission_correlations.csv"))

# --------------------------
# FIGURES
# --------------------------

fig_scatter <- analysis %>%
  filter(!is.na(project_transmission_length_miles), !is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
  ggplot(aes(x = project_transmission_length_miles, y = bert_duration_days_final)) +
  geom_point(alpha = 0.35, color = catf_dark_blue) +
  geom_smooth(method = "lm", se = TRUE, color = catf_teal, linewidth = 1) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Transmission Length vs Timeline Duration",
    subtitle = "Projects with extracted transmission length",
    x = "Transmission length (miles)",
    y = "Duration (days)"
  ) +
  theme_minimal(base_size = 11)

print(fig_scatter)

ggsave(
  filename = here(figures_dir, "fig_transmission_length_vs_duration.png"),
  plot = fig_scatter,
  width = 9,
  height = 6,
  dpi = 300
)

fig_region <- analysis %>%
  filter(!is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
  mutate(project_region = fct_infreq(project_region)) %>%
  ggplot(aes(x = project_region, y = bert_duration_days_final, fill = project_region)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.2, show.legend = FALSE) +
  scale_fill_manual(values = rep(c(catf_dark_blue, catf_light_blue, catf_teal, catf_magenta), 10)) +
  labs(
    title = "Transmission Project Duration by Region",
    subtitle = "Clean energy projects; duration = initiation to decision",
    x = "Region",
    y = "Duration (days)"
  ) +
  theme_minimal(base_size = 11)

print(fig_region)

ggsave(
  filename = here(figures_dir, "fig_transmission_duration_by_region.png"),
  plot = fig_region,
  width = 9,
  height = 6,
  dpi = 300
)

# Start vs decision lollipop (transmission-only)
start_counts <- analysis %>%
  mutate(start_year = as.integer(format(bert_initiation_date_final, "%Y"))) %>%
  filter(!is.na(start_year), start_year >= 2000, start_year <= max_year) %>%
  count(year = start_year, name = "n") %>%
  mutate(type = "Start")

decision_counts <- analysis %>%
  mutate(decision_year = as.integer(format(bert_decision_date_final, "%Y"))) %>%
  filter(!is.na(decision_year), decision_year >= 2000, decision_year <= max_year) %>%
  count(year = decision_year, name = "n") %>%
  mutate(type = "Decision")

start_end_long <- bind_rows(start_counts, decision_counts) %>%
  mutate(type = factor(type, levels = c("Start", "Decision")))

fig_start_end <- ggplot(start_end_long, aes(x = year, y = n, color = type)) +
  geom_segment(
    aes(x = year, xend = year, y = 0, yend = n),
    position = position_dodge(width = 0.6),
    linewidth = 0.7
  ) +
  geom_point(
    position = position_dodge(width = 0.6),
    size = 2.3
  ) +
  scale_color_manual(values = c(catf_teal, catf_magenta)) +
  scale_x_continuous(breaks = seq(2000, max_year, by = 2)) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.05))) +
  labs(
    title = "Transmission Projects: Start vs Decision Year",
    subtitle = "Transmission-only clean energy projects (strict definition)",
    x = "Year",
    y = "Number of Projects",
    color = NULL
  ) +
  theme_minimal(base_size = 11)

fig_start_end

ggsave(
  filename = here(figures_dir, "fig_transmission_start_vs_decision_lollipop.png"),
  plot = fig_start_end,
  width = 10,
  height = 6,
  dpi = 300
)

cat("Saved outputs to:\n", tables_dir, "\n", figures_dir, "\n")
