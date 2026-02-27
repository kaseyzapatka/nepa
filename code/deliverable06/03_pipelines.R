# --------------------------
# DELIVERABLE 6: CARBON + HYDROGEN PIPELINES
# --------------------------

source(here::here("code", "deliverable06", "00_setup.R"))

pipeline_group_levels <- c(
  "Carbon pipeline",
  "Hydrogen pipeline",
  "Natural gas pipeline",
  "Oil/petroleum pipeline",
  "Water/irrigation pipeline",
  "Other pipeline"
)

pipeline_group_colors <- c(
  "Carbon pipeline"          = catf_dark_blue,
  "Hydrogen pipeline"        = catf_teal,
  "Natural gas pipeline"     = catf_magenta,
  "Oil/petroleum pipeline"   = catf_light_blue,
  "Water/irrigation pipeline" = "#E8A838",
  "Other pipeline"           = "gray55"
)

add_pipeline_group <- function(df) {
  # project_type_txt is a flattened string of the JSON project_type array,
  # produced by add_deliv6_fallback_features(). Using it here gives clean
  # controlled-vocabulary splits without relying on free text.
  df %>%
    mutate(
      is_oil_type   = str_detect(project_type_txt, regex("oil & gas|oil and gas|petroleum", ignore_case = TRUE)),
      is_water_type = str_detect(project_type_txt, regex("water resources|irrigation", ignore_case = TRUE)) & !is_oil_type,
      pipeline_group = case_when(
        project_is_carbon_pipeline      ~ "Carbon pipeline",
        project_is_hydrogen_pipeline    ~ "Hydrogen pipeline",
        project_is_natural_gas_pipeline ~ "Natural gas pipeline",
        is_oil_type                     ~ "Oil/petroleum pipeline",
        is_water_type                   ~ "Water/irrigation pipeline",
        TRUE                            ~ "Other pipeline"
      ),
      pipeline_group = factor(pipeline_group, levels = pipeline_group_levels)
    ) %>%
    select(-is_oil_type, -is_water_type)
}

# All pipeline projects from projects_combined (no clean-energy or timeline filter).
# Used for count, geography, and length tables.
analysis_all <- read_parquet(projects_combined_path) %>%
  filter(project_is_pipeline == TRUE) %>%
  add_deliv6_fallback_features() %>%
  mutate(
    project_state_primary = map_chr(project_state, extract_primary_state),
    project_region = as.character(state_region_map$region[match(project_state_primary, state_region_map$state)]),
    project_region = coalesce(project_region, "Unknown")
  ) %>%
  add_pipeline_group()

# Pipeline projects with timeline data (clean energy only, ~228 projects).
# Used for duration-based analyses and figures.
analysis_timeline <- prepare_deliverable6_data(clean_only = FALSE) %>%
  filter(project_is_pipeline) %>%
  add_pipeline_group()

cat("Pipeline projects (all):", nrow(analysis_all), "\n")
cat("Pipeline projects (with timeline):", nrow(analysis_timeline), "\n")


# --------------------------
# TABLES
# --------------------------

# Count and length stats from all pipelines; duration stats from timeline subset.
tbl_count_length <- analysis_all %>%
  group_by(pipeline_group) %>%
  summarise(
    n_projects = n(),
    n_with_length = sum(!is.na(project_pipeline_length_miles)),
    pct_with_length = n_with_length / n_projects,
    median_length_miles = median(project_pipeline_length_miles, na.rm = TRUE),
    .groups = "drop"
  )

tbl_duration <- analysis_timeline %>%
  group_by(pipeline_group) %>%
  summarise(
    n_with_duration = sum(!is.na(bert_duration_days_final) & bert_duration_days_final >= 0),
    median_duration_days = median(bert_duration_days_final, na.rm = TRUE),
    .groups = "drop"
  )

tbl_pipeline_summary <- tbl_count_length %>%
  left_join(tbl_duration, by = "pipeline_group")

write_csv(tbl_pipeline_summary, here(tables_dir, "table_pipeline_group_summary.csv"))

# State/region counts and lengths from all pipelines (duration too sparse to report by state)
tbl_pipeline_state <- analysis_all %>%
  group_by(project_region, project_state_primary, pipeline_group) %>%
  summarise(
    n_projects = n(),
    median_length_miles = median(project_pipeline_length_miles, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n_projects))

write_csv(tbl_pipeline_state, here(tables_dir, "table_pipeline_state_region.csv"))

# Baseline comparison: duration from timeline (clean energy pipelines only)
baseline <- analysis_timeline %>%
  filter(pipeline_group %in% c("Carbon pipeline", "Hydrogen pipeline", "Natural gas pipeline")) %>%
  group_by(pipeline_group) %>%
  summarise(
    n = n(),
    median_duration_days = median(bert_duration_days_final, na.rm = TRUE),
    median_length_miles = median(project_pipeline_length_miles, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(baseline, here(tables_dir, "table_pipeline_baseline_compare.csv"))

corr_data <- analysis_timeline %>%
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

duration_cap <- 500

duration_overall <- analysis_timeline %>%
  filter(!is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
  summarise(
    n_total    = n(),
    n_topcoded = sum(bert_duration_days_final > duration_cap),
    pct_top    = round(100 * n_topcoded / n_total, 1)
  )

duration_caption <- paste0(
  "Notes: Values above ", duration_cap, " days topcoded to cap ",
  "(", duration_overall$n_topcoded, " of ", duration_overall$n_total,
  " projects, ", duration_overall$pct_top, "%). ",
  "Duration analysis limited to clean energy projects with calculable timelines."
)

duration_n_labels <- analysis_timeline %>%
  filter(!is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
  count(pipeline_group) %>%
  mutate(label = paste0("n = ", n))

fig_duration <- analysis_timeline %>%
  filter(!is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
  mutate(duration_plot = pmin(bert_duration_days_final, duration_cap)) %>%
  ggplot(aes(x = pipeline_group, y = duration_plot, fill = pipeline_group)) +
  geom_violin(alpha = 0.5, trim = TRUE, color = NA) +
  geom_jitter(width = 0.15, alpha = 0.25, size = 1.2, color = "gray75", show.legend = FALSE) +
  geom_boxplot(
    width         = 0.2,
    outlier.shape = NA,
    fill          = NA,
    color         = catf_navy,
    linewidth     = 0.55
  ) +
  geom_text(
    data = duration_n_labels,
    aes(x = pipeline_group, y = duration_cap + 15, label = label),
    size = 3.2, color = "grey40", fontface = "italic", inherit.aes = FALSE
  ) +
  coord_cartesian(ylim = c(0, duration_cap + 28)) +
  scale_fill_manual(values = pipeline_group_colors) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title    = "Pipeline NEPA Duration by Technology Group",
    subtitle = paste0("Clean energy projects; values above ", duration_cap, " days topcoded to cap"),
    caption  = duration_caption,
    x        = NULL,
    y        = "Duration (days)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    plot.caption    = element_text(size = 8, color = "gray40", hjust = 0)
  )

print(fig_duration)
ggsave(
  filename = here(figures_dir, "fig_pipeline_duration_by_group.png"),
  plot     = fig_duration,
  width    = 10,
  height   = 6,
  dpi      = 300
)

fig_scatter <- analysis_timeline %>%
  filter(!is.na(project_pipeline_length_miles), !is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
  ggplot(aes(x = project_pipeline_length_miles, y = bert_duration_days_final, color = pipeline_group)) +
  geom_point(alpha = 0.55) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.9) +
  scale_color_manual(values = pipeline_group_colors) +
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

# -- Pipeline Length by Group (violin + box, topcoded at 100 miles) --
length_cap <- 25

length_stats <- analysis_all %>%
  filter(!is.na(project_pipeline_length_miles)) %>%
  summarise(
    n_total    = n(),
    n_topcoded = sum(project_pipeline_length_miles > length_cap),
    pct_top    = round(100 * n_topcoded / n_total, 1)
  )

length_n_labels <- analysis_all %>%
  filter(!is.na(project_pipeline_length_miles)) %>%
  count(pipeline_group) %>%
  mutate(label = paste0("n = ", n))

length_caption <- paste0(
  "Notes: Values above ", length_cap, " miles topcoded to cap ",
  "(", length_stats$n_topcoded, " of ", length_stats$n_total,
  " projects, ", length_stats$pct_top, "%)."
)

fig_length <- analysis_all %>%
  filter(!is.na(project_pipeline_length_miles)) %>%
  mutate(length_plot = pmin(project_pipeline_length_miles, length_cap)) %>%
  ggplot(aes(x = pipeline_group, y = length_plot, fill = pipeline_group)) +
  geom_violin(alpha = 0.5, trim = TRUE, color = NA) +
  geom_jitter(width = 0.15, alpha = 0.2, size = 1.0, color = "gray75", show.legend = FALSE) +
  geom_boxplot(
    width         = 0.2,
    outlier.shape = NA,
    fill          = NA,
    color         = catf_navy,
    linewidth     = 0.55
  ) +
  geom_text(
    data = length_n_labels,
    aes(x = pipeline_group, y = length_cap + 1, label = label),
    size = 3.2, color = "grey40", fontface = "italic", inherit.aes = FALSE
  ) +
  coord_cartesian(ylim = c(0, length_cap + 2.5)) +
  scale_fill_manual(values = pipeline_group_colors) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title   = "Pipeline Length by Group",
    subtitle = paste0("All pipeline NEPA actions; values above ", length_cap, " miles topcoded to cap"),
    caption = length_caption,
    x       = NULL,
    y       = "Pipeline length (miles)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    plot.caption    = element_text(size = 8, color = "gray40", hjust = 0)
  )

print(fig_length)
ggsave(
  filename = here(figures_dir, "fig_pipeline_length_by_group.png"),
  plot     = fig_length,
  width    = 10,
  height   = 6,
  dpi      = 300
)

cat("Saved outputs to:\n", tables_dir, "\n", figures_dir, "\n")

