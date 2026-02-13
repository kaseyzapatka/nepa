# --------------------------
# DELIVERABLE 6: GEOTHERMAL PHASES
# --------------------------

source(here::here("code", "deliverable06", "00_setup.R"))

normalize_geothermal_key <- function(x) {
  x %>%
    str_to_lower() %>%
    str_replace_all("[[:punct:]]", " ") %>%
    str_replace_all("\\b(geothermal|exploration|exploratory|drilling|drill|well|wells|plant|power|project|phase|program|facility|unit|site|field)\\b", " ") %>%
    str_squish()
}

analysis <- prepare_deliverable6_data() %>%
  filter(project_is_geothermal) %>%
  mutate(
    geothermal_project_key = normalize_geothermal_key(project_title_txt),
    geothermal_project_key = if_else(nchar(geothermal_project_key) < 8, project_id, geothermal_project_key),
    geothermal_phase = factor(project_geothermal_phase, levels = c("exploration", "drilling", "plant", "multi_phase", "unknown", "none"))
  )

cat("Geothermal projects:", nrow(analysis), "\n")

# --------------------------
# TABLES
# --------------------------

tbl_phase_distribution <- analysis %>%
  count(geothermal_phase, name = "n_projects") %>%
  mutate(share = n_projects / sum(n_projects))

write_csv(tbl_phase_distribution, here(tables_dir, "table_geothermal_phase_distribution.csv"))

within_project_phase <- analysis %>%
  filter(!is.na(geothermal_phase), geothermal_phase != "none") %>%
  group_by(geothermal_project_key) %>%
  summarise(
    n_actions = n(),
    n_distinct_phases = n_distinct(geothermal_phase),
    phases = paste(sort(unique(as.character(geothermal_phase))), collapse = " | "),
    first_start = if (all(is.na(bert_initiation_date_final))) as.Date(NA) else min(bert_initiation_date_final, na.rm = TRUE),
    last_decision = if (all(is.na(bert_decision_date_final))) as.Date(NA) else max(bert_decision_date_final, na.rm = TRUE),
    span_days = as.numeric(last_decision - first_start),
    example_title = first(project_title_txt),
    .groups = "drop"
  ) %>%
  mutate(
    first_start = as.character(first_start),
    last_decision = as.character(last_decision),
    span_days = ifelse(is.na(first_start) | is.na(last_decision), NA_real_, span_days)
  ) %>%
  arrange(desc(n_distinct_phases), desc(n_actions))

write_csv(within_project_phase, here(tables_dir, "table_geothermal_within_project_phases.csv"))

tbl_phase_timeline <- analysis %>%
  filter(!is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
  group_by(geothermal_phase) %>%
  summarise(
    n_projects = n(),
    median_duration_days = median(bert_duration_days_final, na.rm = TRUE),
    p25_duration_days = quantile(bert_duration_days_final, 0.25, na.rm = TRUE),
    p75_duration_days = quantile(bert_duration_days_final, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(tbl_phase_timeline, here(tables_dir, "table_geothermal_phase_timeline.csv"))

# --------------------------
# FIGURES
# --------------------------

fig_phase_box <- analysis %>%
  filter(!is.na(bert_duration_days_final), bert_duration_days_final >= 0, geothermal_phase != "none") %>%
  ggplot(aes(x = geothermal_phase, y = bert_duration_days_final, fill = geothermal_phase)) +
  geom_boxplot(alpha = 0.85, outlier.alpha = 0.2, show.legend = FALSE) +
  scale_fill_manual(values = c(catf_dark_blue, catf_teal, catf_magenta, catf_light_blue, "gray60", "gray80")) +
  labs(
    title = "Geothermal Timelines by Project Phase",
    subtitle = "CE projects classified from project text",
    x = "Geothermal phase",
    y = "Duration (days)"
  ) +
  theme_minimal(base_size = 11)

ggsave(
  filename = here(figures_dir, "fig_geothermal_phase_duration_boxplot.png"),
  plot = fig_phase_box,
  width = 9,
  height = 6,
  dpi = 300
)

sequence_data <- analysis %>%
  filter(!is.na(bert_initiation_date_final), !is.na(bert_decision_date_final)) %>%
  group_by(geothermal_project_key) %>%
  mutate(n_actions = n()) %>%
  ungroup() %>%
  filter(n_actions > 1) %>%
  group_by(geothermal_project_key) %>%
  mutate(project_rank = dense_rank(desc(n_actions))) %>%
  ungroup() %>%
  arrange(desc(n_actions), geothermal_project_key) %>%
  slice_head(n = 250)

fig_sequence <- ggplot(sequence_data) +
  geom_segment(
    aes(
      x = bert_initiation_date_final,
      xend = bert_decision_date_final,
      y = fct_reorder(geothermal_project_key, n_actions),
      yend = fct_reorder(geothermal_project_key, n_actions),
      color = geothermal_phase
    ),
    linewidth = 1.1,
    alpha = 0.85
  ) +
  scale_color_manual(values = c(
    exploration = catf_dark_blue,
    drilling = catf_teal,
    plant = catf_magenta,
    multi_phase = catf_light_blue,
    unknown = "gray55",
    none = "gray80"
  )) +
  labs(
    title = "Within-Project Geothermal Sequencing",
    subtitle = "Each segment shows initiation-to-decision for one NEPA action",
    x = "Date",
    y = "Inferred project key",
    color = "Phase"
  ) +
  theme_minimal(base_size = 10)

ggsave(
  filename = here(figures_dir, "fig_geothermal_within_project_sequence.png"),
  plot = fig_sequence,
  width = 11,
  height = 8,
  dpi = 300
)

cat("Saved outputs to:\n", tables_dir, "\n", figures_dir, "\n")
