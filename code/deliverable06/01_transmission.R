# --------------------------
# DELIVERABLE 6: TRANSMISSION LINES
# --------------------------

# 4.92-4.83 about .09 cents to use

rm(list = ls())
source(here::here("code", "deliverable06", "00_setup.R"))
# Data sources:
#   Timeline:      projects_timeline_bert.parquet + _ea_llm + _eis_llm
#   Shared fields: projects_combined.parquet         (energy type, geothermal, pipeline)
#   Transmission:  projects_transmission.parquet      (all project_transmission_* columns)

analysis <- prepare_deliverable6_data() %>%
  filter(project_is_transmission) |>
  glimpse()

# --------------------------
# MANUAL LENGTH RECODES (QA overrides)
# --------------------------
# False positives identified during outlier review. Corrections applied to
# project_transmission_length_final before any downstream analysis.
#   ba2da0d3... Davis-Kingman Tap: regex grabbed DSWR system total (3,100 mi); correct is 26.6 mi
#   d65372a8... SDG&E Helicopter Landing Pad: "2022" is a year on map, not miles; set to NA
#   35677250... Idaho Power Maintenance (1): scale bar artifact; correct is 1.34 mi (BLM lands)
#   f2f52b2c... Idaho Power Maintenance (2): same document, same fix
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

cat("Electricity transmission projects:", nrow(analysis), "\n")
cat("  - Unambiguous (rule-based only):            ", sum(!analysis$project_transmission_length_llm_trigger, na.rm = TRUE), "\n")
cat("  - Flagged for LLM adjudication:             ", sum(analysis$project_transmission_length_llm_trigger, na.rm = TRUE), "\n")
max_year <- as.integer(format(Sys.Date(), "%Y"))

# Audit: confirm extraction build and LLM run timestamps
if ("project_tx_extraction_run_at" %in% names(analysis)) {
  cat("  Extraction built at:", analysis$project_tx_extraction_run_at[1], "\n")
}
if ("project_tx_llm_run_at" %in% names(analysis)) {
  llm_rows <- analysis$project_tx_llm_run_at[nzchar(coalesce(analysis$project_tx_llm_run_at, ""))]
  if (length(llm_rows) > 0) {
    cat("  LLM (Claude API) ran on", length(llm_rows), "rows; most recent:", max(llm_rows), "\n")
  } else {
    cat("  LLM not run on any rows (rerun with --run llm to adjudicate)\n")
  }
}

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
    "Electricity transmission projects",
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

# --------------------------
# SAMPLE COMPOSITION FIGURE
# --------------------------

n_total     <- nrow(analysis_len)
n_full_dur  <- sum(!is.na(analysis_len$duration_days) & analysis_len$duration_days >= 0, na.rm = TRUE)
n_dec_only  <- sum(!is.na(analysis_len$bert_decision_date_final) &
                     is.na(analysis_len$bert_initiation_date_final), na.rm = TRUE)
n_init_only <- sum(is.na(analysis_len$bert_decision_date_final) &
                     !is.na(analysis_len$bert_initiation_date_final), na.rm = TRUE)
n_no_dates  <- sum(is.na(analysis_len$bert_decision_date_final) &
                     is.na(analysis_len$bert_initiation_date_final), na.rm = TRUE)

avail_df <- tibble(
  status = factor(
    c("Full duration\n(both dates)", "Decision date only", "Initiation date only", "No dates"),
    levels = c("No dates", "Initiation date only", "Decision date only", "Full duration\n(both dates)")
  ),
  n     = c(n_full_dur, n_dec_only, n_init_only, n_no_dates),
  group = c("Complete", "Partial", "Partial", "Missing")
) %>%
  mutate(pct = round(n / n_total * 100))

fig_sample_breakdown <- avail_df %>%
  ggplot(aes(x = status, y = n, fill = group)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(n, "  (", pct, "%)")),
            hjust = -0.08, size = 3.5, color = "grey20") +
  coord_flip(clip = "off") +
  scale_fill_manual(
    values = c("Complete" = catf_teal, "Partial" = catf_light_blue, "Missing" = "grey75"),
    name   = "Date availability"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.3))) +
  labs(
    title    = paste0("Transmission Sample: N = ", n_total, " Electricity Transmission Projects"),
    subtitle = "Breakdown by date availability for NEPA duration calculations",
    x        = NULL,
    y        = "Number of projects"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

print(fig_sample_breakdown)
ggsave(here(figures_dir, "fig_transmission_sample_breakdown.png"),
       fig_sample_breakdown, width = 8, height = 4, dpi = 300)

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


# -- Fig 1: Length distribution histogram (top-coded at 300 mi) --
len_cap     <- 300
med_len     <- median(analysis_len$length_miles, na.rm = TRUE)
n_capped    <- sum(analysis_len$length_miles > len_cap, na.rm = TRUE)

fig_length_dist <- analysis_len %>%
  filter(!is.na(length_miles)) %>%
  mutate(length_miles_plot = pmin(length_miles, len_cap)) %>%
  ggplot(aes(x = length_miles_plot)) +
  geom_histogram(fill = catf_dark_blue, color = "white", binwidth = 10, boundary = 0) +
  geom_vline(xintercept = med_len, linetype = "dashed", color = catf_teal, linewidth = 0.9) +
  annotate(
    "text", x = med_len + 3, y = Inf,
    label = paste0("Median: ", round(med_len, 1), " mi"),
    hjust = 0, vjust = 1.5, color = catf_teal, size = 3.5
  ) +
  scale_x_continuous(
    labels = scales::comma,
    breaks = seq(0, len_cap, 50),
    limits = c(0, len_cap)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(
    title    = "Distribution of Transmission Line Lengths",
    subtitle = paste0(
      sum(!is.na(analysis_len$length_miles)), " electricity transmission projects",
      if (n_capped > 0) paste0("  |  ", n_capped, " project(s) >", len_cap, " mi top-coded") else ""
    ),
    x = "Transmission length (miles)",
    y = "Number of projects"
  ) +
  theme_minimal(base_size = 11)

print(fig_length_dist)
ggsave(here(figures_dir, "fig_transmission_length_distribution.png"),
       fig_length_dist, width = 8, height = 5, dpi = 300)


# -- Fig: Action type count bar chart (replaces table in report) --
fig_action_count <- tbl_action %>%
  mutate(action_lbl = fct_reorder(action_label(action), n_projects)) %>%
  ggplot(aes(x = n_projects, y = action_lbl, fill = action_label(action))) +
  geom_col(width = 0.6) +
  geom_text(aes(label = n_projects), hjust = -0.2, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = action_colors) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(
    title    = "Transmission Projects by Action Type",
    subtitle = "Count of electricity transmission projects per action category",
    x        = "Number of projects",
    y        = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")

print(fig_action_count)
ggsave(here(figures_dir, "fig_transmission_action_count.png"),
       fig_action_count, width = 8, height = 4, dpi = 300)


# -- Fig 2 & 7: Shared action-type ordering (by median duration) --
# Both length and duration figures use this same factor level order so that
# a given action type appears at the same vertical position in both charts.
action_order_levels <- analysis_len %>%
  filter(
    !is.na(duration_days), duration_days >= 0,
    !project_transmission_action %in% c("none", "unknown", "mixed")
  ) %>%
  group_by(lbl = action_label(project_transmission_action)) %>%
  summarise(med = median(duration_days, na.rm = TRUE), .groups = "drop") %>%
  arrange(med) %>%
  pull(lbl)

# -- Fig 2: Length by action type (boxplot + jitter) --
length_action_data <- analysis_len %>%
  filter(!is.na(length_miles),
         !project_transmission_action %in% c("none", "unknown", "mixed")) %>%
  mutate(action_label = factor(action_label(project_transmission_action),
                               levels = action_order_levels))

n_length_action <- length_action_data %>%
  count(action_label, name = "n")

fig_length_by_action <- length_action_data %>%
  ggplot(aes(x = action_label, y = length_miles, fill = action_label, color = action_label)) +
  geom_boxplot(alpha = 0.75, outlier.shape = NA, show.legend = FALSE, width = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.55, size = 2, show.legend = FALSE) +
  geom_text(
    data = n_length_action,
    aes(x = action_label, y = Inf, label = paste0("n=", n)),
    hjust = 1.15, size = 2.7, color = "gray30", inherit.aes = FALSE
  ) +
  coord_flip() +
  scale_fill_manual(values = action_colors) +
  scale_color_manual(values = action_colors) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Transmission Length by Project Action Type",
    subtitle = "Ordered by median duration; points = individual projects",
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

bin_colors <- c(
  "<10 mi"    = catf_teal,
  "10–50 mi"  = catf_light_blue,
  "50–100 mi" = catf_dark_blue,
  "100+ mi"   = catf_navy
)

p_bin_dur <- analysis_len %>%
  filter(!is.na(length_bin), !is.na(duration_days), duration_days >= 0) %>%
  ggplot(aes(x = length_bin, y = duration_days, fill = length_bin)) +
  geom_violin(alpha = 0.5, trim = TRUE, color = NA) +
  geom_boxplot(
    width = 0.2,
    outlier.alpha = 0.25,
    outlier.size  = 0.8,
    fill          = NA,
    color         = catf_navy,
    linewidth     = 0.55
  ) +
  geom_text(
    data = tbl_length_bins,
    aes(x = length_bin, y = 780, label = paste0("n = ", n_projects)),
    inherit.aes = FALSE, size = 3.2, color = "grey30", fontface = "italic"
  ) +
  coord_cartesian(ylim = c(0, 800)) +
  scale_fill_manual(values = bin_colors) +
  scale_y_continuous(labels = scales::comma) +
  labs(x = NULL, y = "Duration (days)", title = "NEPA Duration by Length Band") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")

fig_length_bins_chart <- p_bin_n + p_bin_dur +
  plot_annotation(
    title    = "Longer Transmission Lines, Longer NEPA Reviews",
    subtitle = "Electricity transmission projects by length band"
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
    subtitle = "Number of electricity transmission projects per state; color = census region",
    x        = "Number of projects",
    y        = NULL,
    color    = "Region"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

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
  theme(legend.position = "bottom")

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
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom", legend.text = element_text(size = 9)) +
  guides(color = guide_legend(nrow = 2))

print(fig_scatter)
ggsave(here(figures_dir, "fig_transmission_length_vs_duration.png"),
       fig_scatter, width = 9, height = 6, dpi = 300)


# -- Fig 5b: Scatter with outliers removed (top 5% of length and duration) --
scatter_base <- analysis_len %>%
  filter(!is.na(length_miles), !is.na(duration_days), duration_days >= 0)

dur_p95 <- quantile(scatter_base$duration_days, 0.95, na.rm = TRUE)
len_p95 <- quantile(scatter_base$length_miles,  0.95, na.rm = TRUE)

scatter_trim <- scatter_base %>%
  filter(duration_days <= dur_p95, length_miles <= len_p95) %>%
  mutate(
    action_label = case_when(
      project_transmission_action %in% c("none", "unknown", "mixed") ~ "Unknown / Mixed",
      TRUE ~ action_label(project_transmission_action)
    )
  )

n_excluded   <- nrow(scatter_base) - nrow(scatter_trim)
r_len_dur_trim <- round(cor(scatter_trim$length_miles, scatter_trim$duration_days,
                            use = "complete.obs"), 2)

fig_scatter_trim <- scatter_trim %>%
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
    title    = "Transmission Length vs. NEPA Duration (Outliers Removed)",
    subtitle = paste0(
      "Top 5% of length (>", round(len_p95), " mi) and duration (>", round(dur_p95),
      " days) excluded (n = ", n_excluded, " removed) | Pearson r = ", r_len_dur_trim
    ),
    x = "Transmission length (miles)",
    y = "Duration (days)"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom", legend.text = element_text(size = 9)) +
  guides(color = guide_legend(nrow = 2))

print(fig_scatter_trim)
ggsave(here(figures_dir, "fig_transmission_length_vs_duration_trim.png"),
       fig_scatter_trim, width = 9, height = 6, dpi = 300)


# -- Fig 6: Duration by region (boxplot + jitter) --
region_plot_data <- analysis_len %>%
  filter(!is.na(duration_days), duration_days >= 0,
         !project_region %in% c("Unknown", NA)) %>%
  mutate(project_region = fct_reorder(project_region, duration_days, median))

region_n_labels <- region_plot_data %>%
  count(project_region) %>%
  mutate(label = paste0("n = ", n))

fig_region <- region_plot_data %>%
  ggplot(aes(x = project_region, y = duration_days, fill = project_region)) +
  geom_boxplot(alpha = 0.75, outlier.shape = NA, show.legend = FALSE) +
  geom_jitter(width = 0.18, alpha = 0.45, size = 1.8, color = "grey30") +
  geom_text(
    data = region_n_labels,
    aes(x = project_region, y = 950, label = label),
    size = 3.2, color = "grey40", fontface = "italic", inherit.aes = FALSE
  ) +
  coord_cartesian(ylim = c(0, 1000)) +
  scale_fill_manual(values = region_colors) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title    = "NEPA Duration by Census Region",
    subtitle = "Electricity transmission projects; ordered by median | capped at 1,000 days",
    x        = NULL,
    y        = "Duration (days)"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")

print(fig_region)
ggsave(here(figures_dir, "fig_transmission_duration_by_region.png"),
       fig_region, width = 8, height = 5, dpi = 300)


# -- Fig 7: Duration by action type --
duration_action_data <- analysis_len %>%
  filter(!is.na(duration_days), duration_days >= 0,
         !project_transmission_action %in% c("none", "unknown", "mixed")) %>%
  mutate(action_label = factor(action_label(project_transmission_action),
                               levels = action_order_levels))

n_duration_action <- duration_action_data %>%
  count(action_label, name = "n")

fig_duration_by_action <- ggplot(duration_action_data,
    aes(x = action_label, y = duration_days, fill = action_label, color = action_label)) +
  geom_boxplot(alpha = 0.75, outlier.shape = NA, show.legend = FALSE, width = 0.4) +
  geom_jitter(width = 0.2, alpha = 0.55, size = 2, show.legend = FALSE) +
  geom_text(
    data = n_duration_action,
    aes(x = action_label, y = Inf, label = paste0("n=", n)),
    hjust = 1.15, size = 2.7, color = "gray30", inherit.aes = FALSE
  ) +
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
