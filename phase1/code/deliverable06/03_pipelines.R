# --------------------------
# DELIVERABLE 6: CARBON + HYDROGEN PIPELINES
# --------------------------

source(here::here("phase1", "code", "deliverable06", "00_setup.R"))
source(here::here("phase1", "code", "utils", "utils.R"))

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

# Pipeline projects with timeline data, restricted to energy pipeline groups for duration analysis.
# clean_only = FALSE so carbon/hydrogen (NEPATEC-tagged "Fossil") are not excluded.
# Water/irrigation and Other pipeline groups are excluded: they are outside the scope of this
# section ("Carbon and Hydrogen Pipelines") and would swamp the energy groups in duration figures.
energy_pipeline_groups <- c(
  "Carbon pipeline", "Hydrogen pipeline", "Natural gas pipeline", "Oil/petroleum pipeline"
)
# NOTE: The timeline parquet only covers clean energy projects (n=20,725).
# All pipeline-type-tagged projects are classified as "Fossil" by NEPATEC, so
# no pipeline projects appear in the timeline data. analysis_timeline will be empty.
# Duration figures are skipped when this is the case.
analysis_timeline <- prepare_deliverable6_data(clean_only = FALSE) %>%
  filter(project_is_pipeline) %>%
  add_pipeline_group() %>%
  filter(pipeline_group %in% energy_pipeline_groups)

has_timeline_data <- nrow(analysis_timeline) > 0

cat("Pipeline projects (all):", nrow(analysis_all), "\n")
cat("Pipeline projects (with timeline):", nrow(analysis_timeline), "\n")
if (!has_timeline_data) {
  cat("NOTE: No pipeline timeline data available. Timeline extraction covers clean-energy projects only;",
      "all pipeline-type-tagged projects are NEPATEC 'Fossil'. Duration figures will be skipped.\n")
}


text_pipelines <- 
  analysis_all |> 
  filter(!str_detect(project_type_txt, regex("Pipeline"))) |> 
  glimpse()

text_pipelines |> 
  count(pipeline_group)

text_pipelines |> 
  select(project_text_full) |> 
  slice_sample(n = 5) |> 
  pull()

# --------------------------
# FUNNEL COUNTS + DECARB CATEGORY
# --------------------------

n_all_nepa        <- read_parquet(projects_combined_path) %>% nrow()
n_pipeline        <- nrow(analysis_all)
n_pipeline_decarb <- sum(analysis_all$pipeline_group %in% c("Carbon pipeline", "Hydrogen pipeline"), na.rm = TRUE)
n_with_length        <- sum(!is.na(analysis_all$project_pipeline_length_miles))
n_pipeline_new_build <- sum(analysis_all$project_is_pipeline_new_build == TRUE, na.rm = TRUE)

cat("Total NEPA projects (NEPATEC 2.0):", n_all_nepa, "\n")
cat("Pipeline type-tagged:", n_pipeline, "\n")
cat("Pipeline new-build:", n_pipeline_new_build, "\n")
cat("Decarbonization (carbon + hydrogen):", n_pipeline_decarb, "\n")
cat("With extractable length:", n_with_length, "\n")

decarb_category_levels <- c("Decarbonization", "Fossil fuel", "Other")
decarb_category_colors <- c(
  "Decarbonization" = catf_dark_blue,
  "Fossil fuel"     = catf_magenta,
  "Other"           = "gray55"
)

analysis_all <- analysis_all %>%
  mutate(
    decarb_category = case_when(
      pipeline_group %in% c("Carbon pipeline", "Hydrogen pipeline")           ~ "Decarbonization",
      pipeline_group %in% c("Natural gas pipeline", "Oil/petroleum pipeline") ~ "Fossil fuel",
      TRUE                                                                     ~ "Other"
    ),
    decarb_category = factor(decarb_category, levels = decarb_category_levels)
  )

n_pipeline_other <- sum(analysis_all$pipeline_group == "Other pipeline", na.rm = TRUE)

# New-build subsets: pipeline-tagged projects that pass the construction filter.
# Used for duration analyses (scatter, bins, duration violin).
# Length distribution figures keep analysis_all (broad universe).
analysis_all_new_build      <- analysis_all      %>% filter(project_is_pipeline_new_build == TRUE)
analysis_timeline_new_build <- analysis_timeline %>% filter(project_is_pipeline_new_build == TRUE)

n_pipeline_decarb_new_build <- sum(
  analysis_all_new_build$pipeline_group %in% c("Carbon pipeline", "Hydrogen pipeline"),
  na.rm = TRUE
)

cat("Pipeline new-build:", nrow(analysis_all_new_build), "\n")
cat("  of which decarbonization (carbon + hydrogen):", n_pipeline_decarb_new_build, "\n")

# Cross-tab: decarb_category vs energy_type (NEPATEC tag vs our override)
cat("Decarb category vs NEPATEC energy type:\n")
print(count(analysis_all, decarb_category, project_energy_type))

tbl_pipeline_summary <- tibble(
  metric = c(
    "Total NEPA projects (NEPATEC 2.0)",
    "Pipeline type-tagged",
    "Pipeline new-build",
    "Decarbonization technologies (carbon + hydrogen)",
    "Decarbonization new-build",
    "With extractable length",
    "Other pipeline"
  ),
  value = c(
    n_all_nepa, n_pipeline,
    n_pipeline_new_build, n_pipeline_decarb, n_pipeline_decarb_new_build,
    n_with_length, n_pipeline_other
  )
)
write_csv(tbl_pipeline_summary, here(tables_dir, "table_pipeline_summary.csv"))

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

if (has_timeline_data) {
  tbl_duration <- analysis_timeline %>%
    group_by(pipeline_group) %>%
    summarise(
      n_with_duration = sum(!is.na(bert_duration_days_final) & bert_duration_days_final >= 0),
      median_duration_days = median(bert_duration_days_final, na.rm = TRUE),
      .groups = "drop"
    )
  tbl_pipeline_summary_groups <- tbl_count_length %>%
    left_join(tbl_duration, by = "pipeline_group")
} else {
  tbl_pipeline_summary_groups <- tbl_count_length %>%
    mutate(n_with_duration = NA_integer_, median_duration_days = NA_real_)
}

write_csv(tbl_pipeline_summary_groups, here(tables_dir, "table_pipeline_group_summary.csv"))

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

if (has_timeline_data) {
  # Baseline comparison: duration from timeline (decarbonization technology pipelines only)
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
} else {
  cat("Skipping baseline comparison and correlation tables (no timeline data).\n")
}

# --------------------------
# FIGURES
# --------------------------

# -- Pipeline Identification Funnel --
pipeline_funnel_stages <- c(
  "All NEPA projects\n(NEPATEC 2.0)",
  "1. Pipeline type tag\n(project_type field contains \"Pipeline\")",
  "2. New-build filter\n(construction language in title/description,\nno maintenance flag in title)",
  "3. Decarbonization technologies\n(carbon & hydrogen pipelines, new-build)"
)

pipeline_funnel_df <- tibble(
  stage  = factor(pipeline_funnel_stages, levels = rev(pipeline_funnel_stages)),
  n_keep = c(n_all_nepa, n_pipeline, n_pipeline_new_build, n_pipeline_decarb_new_build),
  n_total = n_all_nepa
) %>%
  mutate(n_drop = n_total - n_keep)

pipeline_funnel_long <- pipeline_funnel_df %>%
  pivot_longer(c(n_keep, n_drop), names_to = "status", values_to = "n") %>%
  mutate(status = factor(status, levels = c("n_drop", "n_keep")))

fig_pipeline_funnel <- ggplot(pipeline_funnel_long, aes(x = n, y = stage, fill = status)) +
  geom_col(width = 0.55, color = "white", linewidth = 0.25) +
  geom_text(
    data = filter(pipeline_funnel_df, n_keep >= 1000),
    aes(x = n_keep / 2, y = stage, label = scales::comma(n_keep)),
    inherit.aes = FALSE,
    color = "white", fontface = "bold", size = 3.6
  ) +
  geom_text(
    data = filter(pipeline_funnel_df, n_keep < 1000),
    aes(x = n_keep, y = stage, label = scales::comma(n_keep)),
    inherit.aes = FALSE,
    hjust = -0.35, fontface = "bold", color = catf_navy, size = 3.6
  ) +
  scale_fill_manual(
    values = c(n_keep = catf_dark_blue, n_drop = "#D8DCE8"),
    labels = c(n_keep = "Included", n_drop = "Excluded at this stage"),
    guide  = guide_legend(reverse = TRUE)
  ) +
  scale_x_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.18))
  ) +
  labs(x = "Projects (n)", y = NULL, fill = NULL) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position    = "bottom",
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text.y        = element_text(size = 9.5, lineheight = 1.1)
  )

print(fig_pipeline_funnel)
ggsave(here(figures_dir, "fig_pipeline_funnel.png"),
       fig_pipeline_funnel, width = 8, height = 5.5, dpi = 300)


# -- Decarbonization vs Fossil Fuel Breakdown --
decarb_counts <- analysis_all %>%
  count(decarb_category) %>%
  mutate(pct = round(100 * n / sum(n), 1))

group_decarb_df <- analysis_all %>%
  count(pipeline_group, decarb_category) %>%
  group_by(pipeline_group) %>%
  mutate(n_group_total = sum(n)) %>%
  ungroup() %>%
  mutate(pipeline_group = fct_reorder(pipeline_group, n_group_total))

fig_pipeline_decarb <- group_decarb_df %>%
  ggplot(aes(x = n, y = pipeline_group, fill = decarb_category)) +
  geom_col(width = 0.65, color = "white", linewidth = 0.3) +
  geom_text(
    data = distinct(group_decarb_df, pipeline_group, n_group_total),
    aes(x = n_group_total, y = pipeline_group, label = scales::comma(n_group_total)),
    inherit.aes = FALSE,
    hjust = -0.15, fontface = "bold", size = 3.4, color = catf_navy
  ) +
  scale_fill_manual(values = decarb_category_colors, name = NULL) +
  scale_x_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "Pipeline Projects by Technology Group and Category",
    subtitle = paste0(
      scales::comma(n_pipeline), " total pipeline NEPA actions  \u2014  ",
      "Decarbonization: ", decarb_counts$n[decarb_counts$decarb_category == "Decarbonization"],
      " (", decarb_counts$pct[decarb_counts$decarb_category == "Decarbonization"], "%)",
      "  |  Fossil fuel: ", decarb_counts$n[decarb_counts$decarb_category == "Fossil fuel"],
      " (", decarb_counts$pct[decarb_counts$decarb_category == "Fossil fuel"], "%)"
    ),
    x       = "Number of projects",
    y       = NULL,
    caption = paste0(
      "Note: Carbon and hydrogen pipelines are classified as Decarbonization for this analysis,\n",
      "overriding NEPATEC's energy_type tag (which classifies both as Fossil fuel in the source data)."
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.caption    = element_text(hjust = 0, size = 8, color = "gray40")
  )

print(fig_pipeline_decarb)
ggsave(here(figures_dir, "fig_pipeline_decarb_vs_fossil.png"),
       fig_pipeline_decarb, width = 9, height = 5, dpi = 300)


# -- Pipeline Length Sample Breakdown --
# Shows how many of the pipeline-tagged projects have extractable length,
# and breaks that down by pipeline group (like fig_transmission_sample_breakdown).

len_by_group <- analysis_all %>%
  count(pipeline_group, decarb_category, has_length = !is.na(project_pipeline_length_miles)) %>%
  pivot_wider(names_from = has_length, values_from = n, values_fill = 0) %>%
  rename(n_with_length = `TRUE`, n_no_length = `FALSE`) %>%
  mutate(n_total = n_with_length + n_no_length)

len_sample_df <- tibble(
  stage  = factor(
    c("Pipeline-tagged\n(all groups)", "With extractable\nlength"),
    levels = c("With extractable\nlength", "Pipeline-tagged\n(all groups)")
  ),
  n_keep = c(n_pipeline, n_with_length),
  n_total = n_pipeline
) %>%
  mutate(n_drop = n_total - n_keep)

len_sample_long <- len_sample_df %>%
  pivot_longer(c(n_keep, n_drop), names_to = "status", values_to = "n") %>%
  mutate(status = factor(status, levels = c("n_drop", "n_keep")))

fig_pipeline_length_coverage <- ggplot(len_sample_long, aes(x = n, y = stage, fill = status)) +
  geom_col(width = 0.5, color = "white", linewidth = 0.25) +
  geom_text(
    data = len_sample_df,
    aes(x = n_keep / 2, y = stage, label = paste0(scales::comma(n_keep), "\n(",
        round(100 * n_keep / n_pipeline, 1), "%)")),
    inherit.aes = FALSE,
    color = "white", fontface = "bold", size = 3.5, lineheight = 1.1
  ) +
  scale_fill_manual(
    values = c(n_keep = catf_dark_blue, n_drop = "#D8DCE8"),
    labels = c(n_keep = "With extractable length", n_drop = "No length extracted"),
    guide  = guide_legend(reverse = TRUE)
  ) +
  scale_x_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.05))) +
  labs(
    title    = "Pipeline Length Coverage",
    subtitle = paste0(
      scales::comma(n_with_length), " of ", scales::comma(n_pipeline),
      " pipeline-tagged projects have an extractable length (",
      round(100 * n_with_length / n_pipeline, 1), "%)"
    ),
    x       = "Projects (n)",
    y       = NULL,
    fill    = NULL,
    caption = paste0(
      "Note: Length is extracted from project title, description, and project type metadata only ",
      "(not NEPA document pages).\nNo LLM adjudication is applied; missing values reflect ",
      "projects with no numeric mileage in their metadata."
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position    = "bottom",
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text.y        = element_text(size = 10),
    plot.caption       = element_text(hjust = 0, size = 8, color = "gray40")
  )

print(fig_pipeline_length_coverage)
ggsave(here(figures_dir, "fig_pipeline_length_coverage.png"),
       fig_pipeline_length_coverage, width = 8, height = 3.5, dpi = 300)


# -- Pipeline Length Distribution Histogram --
len_hist_cap   <- 50
len_hist_data  <- analysis_all %>% filter(!is.na(project_pipeline_length_miles))
med_len_all    <- median(len_hist_data$project_pipeline_length_miles, na.rm = TRUE)
n_capped_hist  <- sum(len_hist_data$project_pipeline_length_miles > len_hist_cap, na.rm = TRUE)

fig_pipeline_length_hist <- len_hist_data %>%
  mutate(length_plot = pmin(project_pipeline_length_miles, len_hist_cap)) %>%
  ggplot(aes(x = length_plot)) +
  geom_histogram(fill = catf_dark_blue, color = "white", binwidth = 2, boundary = 0) +
  geom_vline(xintercept = med_len_all, linetype = "dashed", color = catf_teal, linewidth = 0.9) +
  annotate(
    "text", x = med_len_all + 1, y = Inf,
    label = paste0("Median: ", round(med_len_all, 1), " mi"),
    hjust = 0, vjust = 1.5, color = catf_teal, size = 3.5
  ) +
  scale_x_continuous(
    breaks = seq(0, len_hist_cap, 10),
    labels = scales::comma,
    limits = c(0, len_hist_cap)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(
    title    = "Distribution of Pipeline Lengths (All Groups)",
    subtitle = paste0(
      scales::comma(nrow(len_hist_data)), " pipeline projects with extractable length",
      if (n_capped_hist > 0) paste0("  |  ", n_capped_hist,
          " project(s) >", len_hist_cap, " mi top-coded") else ""
    ),
    x       = "Pipeline length (miles)",
    y       = "Number of projects",
    caption = paste0("Notes: Values above ", len_hist_cap, " miles top-coded to cap. All six technology groups combined.")
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.caption = element_text(hjust = 0, size = 8, color = "gray40"))

print(fig_pipeline_length_hist)
ggsave(here(figures_dir, "fig_pipeline_length_distribution.png"),
       fig_pipeline_length_hist, width = 8, height = 5, dpi = 300)


# -- Pipeline Length Bins: count + duration (two-panel, like transmission) --
analysis_all <- analysis_all %>%
  mutate(
    length_bin = cut(
      project_pipeline_length_miles,
      breaks = c(0, 1, 5, 25, Inf),
      labels = c("<1 mi", "1–5 mi", "5–25 mi", "25+ mi"),
      right  = FALSE
    )
  )

tbl_pipe_length_bins <- analysis_all %>%
  filter(!is.na(length_bin)) %>%
  group_by(length_bin) %>%
  summarise(
    n_projects          = n(),
    median_length_miles = round(median(project_pipeline_length_miles, na.rm = TRUE), 1),
    .groups = "drop"
  )
write_csv(tbl_pipe_length_bins, here(tables_dir, "table_pipeline_length_bins.csv"))

bin_colors_pipe <- c(
  "<1 mi"   = catf_teal,
  "1–5 mi"  = catf_light_blue,
  "5–25 mi" = catf_dark_blue,
  "25+ mi"  = catf_navy
)

p_pipe_bin_n <- tbl_pipe_length_bins %>%
  ggplot(aes(x = length_bin, y = n_projects)) +
  geom_col(fill = catf_dark_blue, width = 0.6) +
  geom_text(aes(label = n_projects), vjust = -0.4, size = 3.5, fontface = "bold") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(x = NULL, y = "Projects",
       title = paste0("Projects per length band  (n = ", sum(tbl_pipe_length_bins$n_projects), ")")) +
  theme_minimal(base_size = 11)

if (has_timeline_data) {
  # Length bins for duration analysis — use new-build subset
  analysis_timeline_new_build <- analysis_timeline_new_build %>%
    mutate(
      length_bin = cut(
        project_pipeline_length_miles,
        breaks = c(0, 1, 5, 25, Inf),
        labels = c("<1 mi", "1\u20135 mi", "5\u201325 mi", "25+ mi"),
        right  = FALSE
      )
    )

  pipe_bins_dur_n <- analysis_timeline_new_build %>%
    filter(!is.na(length_bin), !is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
    count(length_bin, name = "n_dur")

  dur_bin_cap <- 500

  p_pipe_bin_dur <- analysis_timeline_new_build %>%
    filter(!is.na(length_bin), !is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
    ggplot(aes(x = length_bin, y = bert_duration_days_final, fill = length_bin)) +
    geom_violin(alpha = 0.5, trim = TRUE, color = NA) +
    geom_boxplot(
      width = 0.2, outlier.alpha = 0.25, outlier.size = 0.8,
      fill = NA, color = catf_navy, linewidth = 0.55
    ) +
    geom_text(
      data = pipe_bins_dur_n,
      aes(x = length_bin, y = dur_bin_cap * 0.94, label = paste0("n = ", n_dur)),
      inherit.aes = FALSE, size = 3.2, color = "grey30", fontface = "italic"
    ) +
    coord_cartesian(ylim = c(0, dur_bin_cap)) +
    scale_fill_manual(values = bin_colors_pipe) +
    scale_y_continuous(labels = scales::comma) +
    labs(x = NULL, y = "Duration (days)",
         title = paste0("NEPA Duration by Length Band  (n = ",
                        sum(pipe_bins_dur_n$n_dur, na.rm = TRUE), " with duration data)")) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none")

  fig_pipeline_length_bins <- p_pipe_bin_n + p_pipe_bin_dur +
    plot_annotation(
      title    = "Longer Pipelines, Longer NEPA Reviews?",
      subtitle = "All pipeline projects by length band (left); duration analysis limited to timeline subset (right)",
      caption  = paste0(
        "Notes: Left panel (n = ", sum(tbl_pipe_length_bins$n_projects), "): all pipeline projects with extractable length.\n",
        "Right panel (n = ", sum(pipe_bins_dur_n$n_dur, na.rm = TRUE), "): pipeline projects with both length and duration data. ",
        "Duration capped at ", dur_bin_cap, " days."
      ),
      theme = theme(plot.caption = element_text(hjust = 0, size = 8))
    )
  ggsave(here(figures_dir, "fig_pipeline_length_bins.png"),
         fig_pipeline_length_bins, width = 10, height = 5, dpi = 300)

  duration_cap <- 500
  duration_overall <- analysis_timeline_new_build %>%
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
    "Duration analysis limited to new-build pipeline projects with calculable timelines."
  )
  duration_n_labels <- analysis_timeline_new_build %>%
    filter(!is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
    count(pipeline_group) %>%
    mutate(label = paste0("n = ", n))

  fig_duration <- analysis_timeline_new_build %>%
    filter(!is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
    mutate(duration_plot = pmin(bert_duration_days_final, duration_cap)) %>%
    ggplot(aes(x = pipeline_group, y = duration_plot, fill = pipeline_group)) +
    geom_violin(alpha = 0.5, trim = TRUE, color = NA) +
    geom_jitter(width = 0.15, alpha = 0.25, size = 1.2, color = "gray75", show.legend = FALSE) +
    geom_boxplot(
      width = 0.2, outlier.shape = NA, fill = NA, color = catf_navy, linewidth = 0.55
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
      subtitle = paste0("New-build projects; values above ", duration_cap, " days topcoded to cap"),
      caption  = duration_caption,
      x = NULL, y = "Duration (days)"
    ) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none", plot.caption = element_text(size = 8, color = "gray40", hjust = 0))
  ggsave(here(figures_dir, "fig_pipeline_duration_by_group.png"),
         fig_duration, width = 10, height = 6, dpi = 300)

  fig_scatter <- analysis_timeline_new_build %>%
    filter(!is.na(project_pipeline_length_miles), !is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
    ggplot(aes(x = project_pipeline_length_miles, y = bert_duration_days_final, color = pipeline_group)) +
    geom_point(alpha = 0.55) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.9) +
    scale_color_manual(values = pipeline_group_colors) +
    labs(
      title = "Pipeline Length vs Timeline Duration",
      subtitle = "Carbon/hydrogen compared against natural gas where available",
      x = "Pipeline length (miles)", y = "Duration (days)", color = "Pipeline group"
    ) +
    theme_minimal(base_size = 11)
  ggsave(here(figures_dir, "fig_pipeline_length_vs_duration.png"),
         fig_scatter, width = 9, height = 6, dpi = 300)
} else {
  # No timeline data: length bins figure shows count panel only
  fig_pipeline_length_bins <- p_pipe_bin_n +
    plot_annotation(
      title   = "Pipeline Projects by Length Band",
      caption = paste0("n = ", sum(tbl_pipe_length_bins$n_projects),
                       " pipeline projects with extractable length. Duration data unavailable (see report)."),
      theme = theme(plot.caption = element_text(hjust = 0, size = 8))
    )
  ggsave(here(figures_dir, "fig_pipeline_length_bins.png"),
         fig_pipeline_length_bins, width = 6, height = 4, dpi = 300)
  cat("Skipping duration figures (fig_pipeline_duration_by_group, fig_pipeline_length_vs_duration).\n")
}

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

# --------------------------
# VALIDATION OF TEXT VS TYPE
# --------------------------
text_vs_type <- analysis_all %>%
  mutate(
    has_pipeline_type = str_detect(project_type_txt, regex("pipeline", ignore_case = TRUE)),
    detection_source  = if_else(has_pipeline_type,
                                "project_type tag",
                                "title / description / NOI / doc titles only")
  )

# 1. Counts
text_vs_type %>% count(detection_source)

# 2. What project_type taxonomy are the text-only projects under?
text_vs_type %>%
  filter(!has_pipeline_type) %>%
  mutate(top_type = str_extract(project_type_txt, "[A-Za-z &/]+") %>% str_squish()) %>%
  count(top_type, sort = TRUE) %>%
  head(20) %>%
  print()

# 3. Length extraction rate: type-tagged vs. text-only
#    (high rate → likely real infrastructure projects; low → incidental mentions)
text_vs_type %>%
  group_by(detection_source) %>%
  summarise(
    n               = n(),
    n_with_length   = sum(!is.na(project_pipeline_length_miles)),
    pct_with_length = round(100 * n_with_length / n, 1)
  )

# 4. Pipeline group breakdown for text-only
text_vs_type %>%
  filter(!has_pipeline_type) %>%
  count(pipeline_group, sort = TRUE)

# 5. Sample titles to eyeball
text_vs_type %>%
  filter(!has_pipeline_type) %>%
  select(project_id, dataset_source, project_title_txt, project_type_txt, 
      #pipeline_group,project_pipeline_length_miles
  ) %>%
  slice_sample(n = 25) %>%
  print(width = Inf)


# --------------------------
# VALIDATION PIPELINE + TRANSMISSION 
# --------------------------
  
analysis_all |> 
  filter(pipeline_group %in% energy_pipeline_groups) |> 
  filter(project_is_pipeline_new_build == TRUE) |> 
  select(project_id, project_title, project_description_txt,, contains("project_type")) |> 
    filter(
    str_detect(project_type_txt, regex("pipeline", ignore_case = TRUE)),
    str_detect(project_type_txt, regex("transmission", ignore_case = TRUE))
  ) |> 
  slice_sample(n = 1) |> 
  pull(project_title, project_description_txt) |> 
  print()

analysis_all |> 
    filter(
    str_detect(project_type_txt, regex("pipeline", ignore_case = TRUE)),
    str_detect(project_type_txt, regex("transmission", ignore_case = TRUE))
  ) |> 
  filter(project_title == "Suchan Fed 1 POD") |> 
  #select(project_type) |> 
  glimpse()


