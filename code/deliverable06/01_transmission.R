# --------------------------
# DELIVERABLE 6: TRANSMISSION LINES
# --------------------------

# 4.92-4.83 about .09 cents to use

rm(list = ls())
source(here::here("code", "deliverable06", "00_setup.R"))

analysis <- prepare_deliverable6_data() %>%
  filter(project_is_transmission) |> 
  glimpse()

cat("Transmission projects (strict, clean energy):", nrow(analysis), "\n")
cat("  - Unambiguous (rule-based only):            ", sum(!analysis$project_transmission_length_llm_trigger, na.rm = TRUE), "\n")
cat("  - Flagged for LLM adjudication:             ", sum(analysis$project_transmission_length_llm_trigger, na.rm = TRUE), "\n")
max_year <- as.integer(format(Sys.Date(), "%Y"))

# check to see if LLM ran
if (any(!is.na(analysis$project_transmission_length_llm_reasoning))) {print("LLM ran")}

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
# Multi-candidate rows: one row per candidate for projects with 2+ distinct values
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
# you can review all 39 ambiguous projects at a glance and verify LLM choices.
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
  #arrange(project_transmission_length_llm_status, project_id) |> 
  arrange(project_id, project_transmission_length_llm_status) |> 
  glimpse()

sheet_write(
  data = tx_adjudication,
  ss = "https://docs.google.com/spreadsheets/d/1KicEYrTlXJSk-fzQ2s30S6l8bpPNBlV75pPfWy0NTeI/edit?usp=sharing",
  sheet = "tx_adjudication"
)

# --------------------------
# TABLES
# --------------------------

# Base working dataset: use _final (LLM-adjudicated when available, else rule-based)
analysis_len <- analysis %>%
  mutate(
    length_miles  = project_transmission_length_final,
    duration_days = bert_duration_days_final,
    length_bin = cut(
      project_transmission_length_final,
      breaks = c(0, 10, 50, 100, Inf),
      labels = c("<10 mi", "10–50 mi", "50–100 mi", "100+ mi"),
      right  = FALSE
    )
  )

# Summary table
tbl_transmission_summary <- tibble(
  Metric = c(
    "Transmission projects (strict, clean energy)",
    "With extracted length",
    "With calculable duration",
    "Multi-state projects",
    "Median length (miles)",
    "Median NEPA duration (days)"
  ),
  Value = c(
    nrow(analysis),
    sum(!is.na(analysis_len$length_miles)),
    sum(!is.na(analysis_len$duration_days) & analysis_len$duration_days >= 0),
    sum(coalesce(analysis$project_multi_state, FALSE)),
    round(median(analysis_len$length_miles, na.rm = TRUE), 1),
    round(median(analysis_len$duration_days, na.rm = TRUE), 1)
  )
)
tbl_transmission_summary
write_csv(tbl_transmission_summary, here(tables_dir, "table_transmission_summary.csv"))

# Length bins: n projects + median/p90 duration per band
tbl_length_bins <- analysis_len %>%
  filter(!is.na(length_bin)) %>%
  group_by(length_bin) %>%
  summarise(
    n_projects           = n(),
    median_length_miles  = round(median(length_miles, na.rm = TRUE), 1),
    median_duration_days = round(median(duration_days, na.rm = TRUE), 1),
    p90_duration_days    = round(quantile(duration_days, 0.9, na.rm = TRUE), 1),
    .groups = "drop"
  )
tbl_length_bins
write_csv(tbl_length_bins, here(tables_dir, "table_transmission_length_bins.csv"))

# State / region breakdown
tbl_state_region <- analysis_len %>%
  group_by(project_region, project_state_primary) %>%
  summarise(
    n_projects           = n(),
    median_length_miles  = round(median(length_miles, na.rm = TRUE), 1),
    median_duration_days = round(median(duration_days, na.rm = TRUE), 1),
    .groups = "drop"
  ) %>%
  arrange(project_region, desc(n_projects))
tbl_state_region |> print(n = 100)
write_csv(tbl_state_region, here(tables_dir, "table_transmission_state_region.csv"))

# Action type breakdown
tbl_action <- analysis_len %>%
  filter(!project_transmission_action %in% c("none", "unknown")) %>%
  group_by(action = project_transmission_action) %>%
  summarise(
    n_projects           = n(),
    median_length_miles  = round(median(length_miles, na.rm = TRUE), 1),
    median_duration_days = round(median(duration_days, na.rm = TRUE), 1),
    .groups = "drop"
  ) %>%
  arrange(desc(n_projects))
tbl_action
write_csv(tbl_action, here(tables_dir, "table_transmission_action.csv"))

# --------------------------
# FIGURES
# --------------------------

# Shared action-type color palette
action_colors <- c(
  "New Build"   = catf_dark_blue,
  "Upgrade"     = catf_teal,
  "Renewal"     = catf_magenta,
  "Fiber Optic" = catf_light_blue,
  "Acquisition" = catf_navy,
  "Mixed"       = "grey55",
  "Unknown"     = "grey75"
)

action_label <- function(x) str_to_title(str_replace_all(x, "_", " "))


# -- Fig 1: Simple length distribution histogram --
med_len <- median(analysis_len$length_miles, na.rm = TRUE)

fig_length_dist <- analysis_len %>%
  filter(!is.na(length_miles)) %>%
  ggplot(aes(x = length_miles)) +
  geom_histogram(fill = catf_dark_blue, color = "white", binwidth = 10, boundary = 0) +
  geom_vline(xintercept = med_len, linetype = "dashed", color = catf_teal, linewidth = 0.9) +
  annotate(
    "text", x = med_len + 3, y = Inf,
    label = paste0("Median: ", round(med_len, 1), " mi"),
    hjust = 0, vjust = 1.5, color = catf_teal, size = 3.5
  ) +
  scale_x_continuous(labels = scales::comma, breaks = seq(0, 300, 25)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(
    title = "Distribution of Transmission Line Lengths",
    subtitle = paste0(sum(!is.na(analysis_len$length_miles)), " strict clean energy transmission projects"),
    x = "Transmission length (miles)",
    y = "Number of projects"
  ) +
  theme_minimal(base_size = 11)

print(fig_length_dist)
ggsave(here(figures_dir, "fig_transmission_length_distribution.png"),
       fig_length_dist, width = 8, height = 5, dpi = 300)


# -- Fig 2: Length by action type (boxplot + jitter) --
fig_length_by_action <- analysis_len %>%
  filter(!is.na(length_miles),
         !project_transmission_action %in% c("none", "unknown", "mixed")) %>%
  mutate(
    action_label = fct_reorder(action_label(project_transmission_action), length_miles, median)
  ) %>%
  ggplot(aes(x = action_label, y = length_miles, fill = action_label, color = action_label)) +
  geom_boxplot(alpha = 0.75, outlier.shape = NA, show.legend = FALSE) +
  geom_jitter(width = 0.2, alpha = 0.55, size = 2, show.legend = FALSE) +
  coord_flip() +
  scale_fill_manual(values = action_colors) +
  scale_color_manual(values = action_colors) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Transmission Length by Project Action Type",
    subtitle = "Ordered by median; points = individual projects",
    x = NULL,
    y = "Transmission length (miles)"
  ) +
  theme_minimal(base_size = 11)

print(fig_length_by_action)
ggsave(here(figures_dir, "fig_transmission_length_by_action.png"),
       fig_length_by_action, width = 8, height = 5, dpi = 300)


# -- Fig 3: Length bins — projects and median duration side-by-side --
p_bin_n <- tbl_length_bins %>%
  ggplot(aes(x = length_bin, y = n_projects)) +
  geom_col(fill = catf_dark_blue, width = 0.6) +
  geom_text(aes(label = n_projects), vjust = -0.4, size = 3.5, fontface = "bold") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(x = NULL, y = "Projects", title = "Projects per length") +
  theme_minimal(base_size = 11)

p_bin_dur <- tbl_length_bins %>%
  filter(!is.na(median_duration_days)) %>%
  ggplot(aes(x = length_bin, y = median_duration_days)) +
  geom_col(fill = catf_teal, width = 0.6) +
  geom_text(aes(label = round(median_duration_days)), vjust = -0.4, size = 3.5, fontface = "bold") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(x = NULL, y = "Days", title = "Median NEPA duration") +
  theme_minimal(base_size = 11)

fig_length_bins_chart <- p_bin_n + p_bin_dur +
  plot_annotation(
    title    = "Longer Transmission Lines, Longer NEPA Reviews",
    subtitle = "Clean energy transmission projects by length band"
  )

print(fig_length_bins_chart)
ggsave(here(figures_dir, "fig_transmission_length_bins.png"),
       fig_length_bins_chart, width = 10, height = 5, dpi = 300)


# -- Fig 4a: State lollipop — number of projects --
region_colors <- c(
  "Northeast"     = catf_navy,
  "South"         = catf_magenta,
  "North Central" = catf_light_blue,
  "West"          = catf_dark_blue
)

tbl_state_region_clean <- tbl_state_region %>%
  filter(!is.na(project_state_primary), project_state_primary != "",
         !project_region %in% c("Unknown", NA))

fig_state_n <- tbl_state_region_clean %>%
  mutate(project_state_primary = fct_reorder(project_state_primary, n_projects)) %>%
  ggplot(aes(x = n_projects, y = project_state_primary, color = project_region)) +
  geom_segment(aes(x = 0, xend = n_projects, yend = project_state_primary),
               color = "grey85", linewidth = 0.8) +
  geom_point(size = 3.5, alpha = 0.85) +
  scale_color_manual(values = region_colors) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 5)) +
  labs(
    title    = "Transmission Projects by State",
    subtitle = "Number of strict clean energy projects per state; color = census region",
    x        = "Number of projects",
    y        = NULL,
    color    = "Region"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "right")

print(fig_state_n)
ggsave(here(figures_dir, "fig_transmission_state_n.png"),
       fig_state_n, width = 7, height = 8, dpi = 300)


# -- Fig 4b: State lollipop — median line length --
fig_state_length <- tbl_state_region_clean %>%
  filter(!is.na(median_length_miles)) %>%
  mutate(project_state_primary = fct_reorder(project_state_primary, median_length_miles)) %>%
  ggplot(aes(x = median_length_miles, y = project_state_primary, color = project_region)) +
  geom_segment(aes(x = 0, xend = median_length_miles, yend = project_state_primary),
               color = "grey85", linewidth = 0.8) +
  geom_point(size = 3.5, alpha = 0.85) +
  geom_text(aes(label = paste0("n=", n_projects)), hjust = -0.35, size = 2.8,
            color = "grey40", show.legend = FALSE) +
  scale_color_manual(values = region_colors) +
  scale_x_continuous(labels = scales::comma, breaks = scales::pretty_breaks(n = 5),
                     expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Median Transmission Length by State",
    subtitle = "Median extracted line length (miles) per state; color = census region",
    x        = "Median length (miles)",
    y        = NULL,
    color    = "Region"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "right")

print(fig_state_length)
ggsave(here(figures_dir, "fig_transmission_state_length.png"),
       fig_state_length, width = 7, height = 8, dpi = 300)


# -- Pearson r: length vs duration (used in fig_scatter subtitle) --
r_len_dur <- round(
  cor(analysis_len$length_miles, analysis_len$duration_days, use = "complete.obs"),
  2
)


# -- Fig 5: Length vs duration, colored by action type --
fig_scatter <- analysis_len %>%
  filter(!is.na(length_miles), !is.na(duration_days), duration_days >= 0) %>%
  mutate(
    action_label = case_when(
      project_transmission_action %in% c("none", "unknown", "mixed") ~ "Unknown / Mixed",
      TRUE ~ action_label(project_transmission_action)
    )
  ) %>%
  ggplot(aes(x = length_miles, y = duration_days, color = action_label)) +
  geom_point(alpha = 0.65, size = 2.2) +
  geom_smooth(
    aes(x = length_miles, y = duration_days),
    method = "lm", se = TRUE, color = "grey40", linewidth = 0.9,
    inherit.aes = FALSE
  ) +
  scale_color_manual(
    values = c(action_colors, "Unknown / Mixed" = "grey70"),
    name = "Action type"
  ) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title    = "Transmission Length vs. NEPA Duration",
    subtitle = paste0("Points colored by action type; grey line = overall trend  |  Pearson r = ", r_len_dur),
    x        = "Transmission length (miles)",
    y        = "Duration (days)"
  ) +
  theme_minimal(base_size = 11)

print(fig_scatter)
ggsave(here(figures_dir, "fig_transmission_length_vs_duration.png"),
       fig_scatter, width = 9, height = 6, dpi = 300)


# -- Fig 6: Duration by region (boxplot + jitter) --
fig_region <- analysis_len %>%
  filter(!is.na(duration_days), duration_days >= 0,
         !project_region %in% c("Unknown", NA)) %>%
  mutate(project_region = fct_reorder(project_region, duration_days, median)) %>%
  ggplot(aes(x = project_region, y = duration_days, fill = project_region)) +
  geom_boxplot(alpha = 0.75, outlier.shape = NA, show.legend = FALSE) +
  geom_jitter(width = 0.18, alpha = 0.45, size = 1.8, color = "grey30") +
  coord_cartesian(ylim = c(0, 1000)) +
  scale_fill_manual(values = region_colors) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title    = "NEPA Duration by Census Region",
    subtitle = "Strict clean energy transmission projects; ordered by median | capped at 1,000 days",
    x        = NULL,
    y        = "Duration (days)"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")

print(fig_region)
ggsave(here(figures_dir, "fig_transmission_duration_by_region.png"),
       fig_region, width = 8, height = 5, dpi = 300)


# -- Fig 7: Duration by action type --
fig_duration_by_action <- analysis_len %>%
  filter(!is.na(duration_days), duration_days >= 0,
         !project_transmission_action %in% c("none", "unknown", "mixed")) %>%
  mutate(
    action_label = fct_reorder(action_label(project_transmission_action), duration_days, median)
  ) %>%
  ggplot(aes(x = action_label, y = duration_days, fill = action_label, color = action_label)) +
  geom_boxplot(alpha = 0.75, outlier.shape = NA, show.legend = FALSE) +
  geom_jitter(width = 0.2, alpha = 0.55, size = 2, show.legend = FALSE) +
  coord_flip() +
  scale_fill_manual(values = action_colors) +
  scale_color_manual(values = action_colors) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title    = "NEPA Duration by Project Action Type",
    subtitle = "Ordered by median duration; points = individual projects",
    x        = NULL,
    y        = "Duration (days)"
  ) +
  theme_minimal(base_size = 11)

print(fig_duration_by_action)
ggsave(here(figures_dir, "fig_transmission_duration_by_action.png"),
       fig_duration_by_action, width = 8, height = 5, dpi = 300)


cat("Saved outputs to:\n", tables_dir, "\n", figures_dir, "\n")
