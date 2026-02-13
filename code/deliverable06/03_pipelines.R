# --------------------------
# DELIVERABLE 6: CARBON + HYDROGEN PIPELINES
# --------------------------

source(here::here("code", "deliverable06", "00_setup.R"))

analysis <- prepare_deliverable6_data() %>%
  filter(project_is_pipeline) %>%
  mutate(
    pipeline_group = case_when(
      project_is_carbon_pipeline ~ "Carbon pipeline",
      project_is_hydrogen_pipeline ~ "Hydrogen pipeline",
      project_is_natural_gas_pipeline ~ "Natural gas pipeline",
      TRUE ~ "Other pipeline"
    ),
    pipeline_group = factor(
      pipeline_group,
      levels = c("Carbon pipeline", "Hydrogen pipeline", "Natural gas pipeline", "Other pipeline")
    )
  )

cat("Pipeline projects:", nrow(analysis), "\n")

# --------------------------
# TABLES
# --------------------------

tbl_pipeline_summary <- analysis %>%
  group_by(pipeline_group) %>%
  summarise(
    n_projects = n(),
    n_with_length = sum(!is.na(project_pipeline_length_miles)),
    pct_with_length = n_with_length / n_projects,
    n_with_duration = sum(!is.na(bert_duration_days_final) & bert_duration_days_final >= 0),
    median_length_miles = median(project_pipeline_length_miles, na.rm = TRUE),
    median_duration_days = median(bert_duration_days_final, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(tbl_pipeline_summary, here(tables_dir, "table_pipeline_group_summary.csv"))

tbl_pipeline_state <- analysis %>%
  group_by(project_region, project_state_primary, pipeline_group) %>%
  summarise(
    n_projects = n(),
    median_length_miles = median(project_pipeline_length_miles, na.rm = TRUE),
    median_duration_days = median(bert_duration_days_final, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n_projects))

write_csv(tbl_pipeline_state, here(tables_dir, "table_pipeline_state_region.csv"))

# Comparison table: carbon/hydrogen vs natural gas baseline
baseline <- analysis %>%
  filter(pipeline_group %in% c("Carbon pipeline", "Hydrogen pipeline", "Natural gas pipeline")) %>%
  group_by(pipeline_group) %>%
  summarise(
    n = n(),
    median_duration_days = median(bert_duration_days_final, na.rm = TRUE),
    median_length_miles = median(project_pipeline_length_miles, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(baseline, here(tables_dir, "table_pipeline_baseline_compare.csv"))

corr_data <- analysis %>%
  transmute(
    length_miles = project_pipeline_length_miles,
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

write_csv(tbl_corr, here(tables_dir, "table_pipeline_correlations.csv"))

# --------------------------
# FIGURES
# --------------------------

plot_df <- analysis %>%
  filter(!is.na(bert_duration_days_final), bert_duration_days_final >= 0)

fig_duration <- ggplot(plot_df, aes(x = pipeline_group, y = bert_duration_days_final, fill = pipeline_group)) +
  geom_boxplot(alpha = 0.85, outlier.alpha = 0.2, show.legend = FALSE) +
  scale_fill_manual(values = c(catf_dark_blue, catf_teal, catf_magenta, catf_light_blue)) +
  labs(
    title = "Pipeline Timeline Durations by Technology Group",
    subtitle = "Includes natural gas reference group when present",
    x = "Pipeline group",
    y = "Duration (days)"
  ) +
  theme_minimal(base_size = 11)

ggsave(
  filename = here(figures_dir, "fig_pipeline_duration_by_group.png"),
  plot = fig_duration,
  width = 9,
  height = 6,
  dpi = 300
)

fig_scatter <- analysis %>%
  filter(!is.na(project_pipeline_length_miles), !is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
  ggplot(aes(x = project_pipeline_length_miles, y = bert_duration_days_final, color = pipeline_group)) +
  geom_point(alpha = 0.55) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.9) +
  scale_color_manual(values = c(catf_dark_blue, catf_teal, catf_magenta, catf_light_blue)) +
  labs(
    title = "Pipeline Length vs Timeline Duration",
    subtitle = "Carbon/hydrogen compared against natural gas where available",
    x = "Pipeline length (miles)",
    y = "Duration (days)",
    color = "Pipeline group"
  ) +
  theme_minimal(base_size = 11)

ggsave(
  filename = here(figures_dir, "fig_pipeline_length_vs_duration.png"),
  plot = fig_scatter,
  width = 9,
  height = 6,
  dpi = 300
)

cat("Saved outputs to:\n", tables_dir, "\n", figures_dir, "\n")
