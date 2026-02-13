# --------------------------
# DELIVERABLE 6: TRANSMISSION LINES
# --------------------------

source(here::here("code", "deliverable06", "00_setup.R"))

analysis <- prepare_deliverable6_data() %>%
  filter(project_is_transmission)

cat("Transmission projects:", nrow(analysis), "\n")

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
    title = "Transmission Length vs CE Timeline Duration",
    subtitle = "Projects with extracted transmission length",
    x = "Transmission length (miles)",
    y = "Duration (days)"
  ) +
  theme_minimal(base_size = 11)

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
    subtitle = "CE projects; duration = initiation to decision",
    x = "Region",
    y = "Duration (days)"
  ) +
  theme_minimal(base_size = 11)

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
  filter(!is.na(start_year), start_year >= 2000, start_year <= 2025) %>%
  count(year = start_year, name = "n") %>%
  mutate(type = "Start")

decision_counts <- analysis %>%
  mutate(decision_year = as.integer(format(bert_decision_date_final, "%Y"))) %>%
  filter(!is.na(decision_year), decision_year >= 2000, decision_year <= 2025) %>%
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
  scale_x_continuous(breaks = seq(2000, 2025, by = 2)) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.05))) +
  labs(
    title = "Transmission Projects: Start vs Decision Year",
    subtitle = "Transmission-only CE projects (strict definition)",
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
