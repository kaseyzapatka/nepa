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
    geothermal_phase = factor(project_geothermal_phase, levels = c("exploration", "drilling", "plant", "operations", "multi_phase", "unknown", "none"))
  )

cat("Geothermal projects:", nrow(analysis), "\n")


# --------------------------
# EXPLORATORY
# --------------------------
analysis |> 
  select(dataset_source) |> 
  group_by(dataset_source) |> 
  count() 

analysis |> 
  count(geothermal_phase)

# first 
#1 exploration         63
#2 drilling           197
#3 plant               63
#4 multi_phase        130
#5 unknown            461

# second 
#1 exploration         95
#2 drilling           264
#3 plant               72
#4 multi_phase        406
#5 NA                  77

# now 
#1 exploration        120
#2 drilling           406
#3 plant              112
#4 operations           4
#5 multi_phase        272

# --------------------------
# ML CLASSIFIER QC
# (only runs if the classify step has been run and the column exists)
# --------------------------
if ("project_geothermal_phase_ml_classified" %in% names(analysis)) {

  cat("\n--- ML classifier QC ---\n")
  n_ml <- sum(analysis$project_geothermal_phase_ml_classified, na.rm = TRUE)
  cat("Rows re-classified by ML:", n_ml, "/", nrow(analysis), "\n")

  # Phase distribution split by classification source
  cat("\nPhase counts by source (regex vs ML):\n")
  analysis %>%
    mutate(source = if_else(
      coalesce(project_geothermal_phase_ml_classified, FALSE),
      "ml", "regex"
    )) %>%
    count(geothermal_phase, source) %>%
    tidyr::pivot_wider(names_from = source, values_from = n, values_fill = 0L) %>%
    arrange(geothermal_phase) %>%
    print()

  # Confidence score summary
  if ("project_geothermal_phase_ml_confidence" %in% names(analysis)) {
    ml_rows <- analysis %>%
      filter(coalesce(project_geothermal_phase_ml_classified, FALSE))

    cat("\nML confidence summary (all ML-classified rows):\n")
    ml_rows %>%
      summarise(
        n       = n(),
        min     = round(min(project_geothermal_phase_ml_confidence, na.rm = TRUE), 3),
        p25     = round(quantile(project_geothermal_phase_ml_confidence, 0.25, na.rm = TRUE), 3),
        median  = round(median(project_geothermal_phase_ml_confidence, na.rm = TRUE), 3),
        p75     = round(quantile(project_geothermal_phase_ml_confidence, 0.75, na.rm = TRUE), 3),
        pct_low = round(mean(project_geothermal_phase_ml_confidence < 0.6, na.rm = TRUE), 3)
      ) %>%
      print()

    # Low-confidence rows to spot-check (confidence < 0.6)
    low_conf <- ml_rows %>%
      filter(project_geothermal_phase_ml_confidence < 0.6) %>%
      arrange(project_geothermal_phase_ml_confidence) %>%
      select(
        project_id, geothermal_phase,
        conf = project_geothermal_phase_ml_confidence,
        project_title_txt
      )
    cat("\nLow-confidence predictions (< 0.60):", nrow(low_conf), "rows\n")
    if (nrow(low_conf) > 0) print(low_conf)

    # Confidence distribution figure
    fig_ml_confidence <- ggplot(ml_rows, aes(x = project_geothermal_phase_ml_confidence,
                                              fill = geothermal_phase)) +
      geom_histogram(binwidth = 0.05, color = "white", linewidth = 0.2) +
      geom_vline(xintercept = 0.6, linetype = "dashed", color = "gray30") +
      annotate("text", x = 0.59, y = Inf, label = "0.60 threshold",
               hjust = 1, vjust = 1.5, size = 3, color = "gray30") +
      scale_fill_manual(values = phase_colors) +
      scale_x_continuous(limits = c(0, 1), labels = scales::percent) +
      labs(
        title    = "ML Classifier Confidence — Geothermal Phase",
        subtitle = paste0(comma(n_ml), " rows re-classified from 'unknown'"),
        x        = "Softmax confidence", y = "Count", fill = "Predicted phase"
      ) +
      theme_minimal(base_size = 11) +
      theme(legend.position = "right")

    print(fig_ml_confidence)
    ggsave(
      here(figures_dir, "fig_geothermal_ml_confidence.png"),
      fig_ml_confidence, width = 9, height = 5, dpi = 300
    )
  }
} else {
  cat("ML classify step not yet run — skipping ML QC block.\n")
  cat("  Run: python code/extract/extract_technology.py --geothermal-phase-classify\n")
}


# --------------------------
# IDENTIFICATION FUNNEL FIGURE
# --------------------------
# Geothermal uses a single type-tag gate (no build-text, length, or maintenance filters),
# so the funnel has only two stages. Update n_type_tagged after re-running:
#   python code/extract/extract_technology.py --run geothermal
# n_type_tagged is the count of clean energy projects with project_is_geothermal == TRUE
# before the prepare_deliverable6_data() clean energy filter is applied. Since the R
# analysis already filters to clean energy, n_type_tagged == nrow(analysis) unless there
# are non-clean geothermal projects. Confirm with:
#   projects_combined %>% filter(project_is_geothermal) %>% count(project_energy_type)

n_clean_energy  <- 20725L   # total decarbonization technology projects in NEPATEC 2.0
n_type_tagged   <- nrow(analysis)  # geothermal project_type tag + clean energy filter
# NOTE: if non-clean geothermal projects exist, set n_type_tagged manually to the
# count BEFORE the clean energy filter and add a third stage for the clean filter.

geo_stage_labels <- c(
  "Decarbonization technology\nprojects (NEPATEC 2.0)",
  "Geothermal project\ntype tag"
)

geo_funnel_df <- tibble(
  stage  = factor(geo_stage_labels, levels = rev(geo_stage_labels)),
  n_keep = c(n_clean_energy, n_type_tagged),
  n_total = n_clean_energy
) %>%
  mutate(n_drop = n_total - n_keep)

geo_funnel_long <- geo_funnel_df %>%
  pivot_longer(c(n_keep, n_drop), names_to = "status", values_to = "n") %>%
  mutate(status = factor(status, levels = c("n_drop", "n_keep")))

fig_geo_funnel <- ggplot(geo_funnel_long, aes(x = n, y = stage, fill = status)) +
  geom_col(width = 0.55, color = "white", linewidth = 0.25) +
  geom_text(
    data = filter(geo_funnel_df, n_keep >= 1000),
    aes(x = n_keep / 2, y = stage, label = scales::comma(n_keep)),
    inherit.aes = FALSE,
    color = "white", fontface = "bold", size = 3.6
  ) +
  geom_text(
    data = filter(geo_funnel_df, n_keep < 1000),
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
    expand = expansion(mult = c(0, 0.14))
  ) +
  labs(x = "Projects (n)", y = NULL, fill = NULL) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position    = "bottom",
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text.y        = element_text(size = 9.5, lineheight = 1.1)
  )

print(fig_geo_funnel)
ggsave(here(figures_dir, "fig_geothermal_funnel.png"),
       fig_geo_funnel, width = 8, height = 3.2, dpi = 300)


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

# --------------------------
# FIGURES
# --------------------------

phase_colors <- c(
  exploration = catf_dark_blue,
  drilling    = catf_teal,
  plant       = catf_magenta,
  operations  = catf_light_blue,
  multi_phase = "#E8A838",
  unknown     = "gray55",
  none        = "gray80"
)

# -- Fig 1: Phase distribution bar chart --
fig_phase_bar <- analysis %>%
  count(geothermal_phase, name = "n_projects") %>%
  filter(!is.na(geothermal_phase), geothermal_phase != "none") %>%
  mutate(
    phase_label = str_to_title(str_replace_all(as.character(geothermal_phase), "_", " ")),
    phase_label = fct_reorder(phase_label, n_projects)
  ) %>%
  ggplot(aes(x = phase_label, y = n_projects, fill = as.character(geothermal_phase))) +
  geom_col(show.legend = FALSE, width = 0.65) +
  geom_text(aes(label = n_projects), hjust = -0.2, size = 3.5, fontface = "bold") +
  coord_flip() +
  scale_fill_manual(values = phase_colors) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Geothermal NEPA Actions by Development Phase",
    subtitle = paste0(comma(nrow(analysis)), " clean geothermal projects identified in NEPATEC 2.0"),
    x = NULL,
    y = "Number of projects"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major.y = element_blank())

print(fig_phase_bar)
ggsave(
  filename = here(figures_dir, "fig_geothermal_phase_distribution.png"),
  plot = fig_phase_bar,
  width = 8,
  height = 5,
  dpi = 300
)

# -- Fig 2: NEPA Duration by Phase (violin + box, capped at 1,000 days) --
# Color scheme mirrors NEPA Duration by Length Band in the transmission section:
# sequential teal → light blue → dark blue → navy, with gray for unknown
phase_colors_box <- c(
  drilling    = catf_teal,
  plant       = catf_light_blue,
  operations  = catf_dark_blue,
  multi_phase = catf_navy,
  unknown     = "gray55"
)

# Compute exploration stats for the footnote
exploration_stats <- analysis %>%
  filter(geothermal_phase == "exploration", !is.na(bert_duration_days_final), bert_duration_days_final >= 0) %>%
  summarise(
    n      = n(),
    med    = round(median(bert_duration_days_final)),
    p25    = round(quantile(bert_duration_days_final, 0.25)),
    p75    = round(quantile(bert_duration_days_final, 0.75))
  )

exploration_caption <- paste0(
  "Notes: Exploration phase excluded from plot (n = ", exploration_stats$n,
  "; median = ", exploration_stats$med, " days",
  ", IQR: ", exploration_stats$p25, "\u2013", exploration_stats$p75, " days). ",
  "Duration values above 250 days are topcoded to 250 and shown at the cap."
)

phase_n_labels <- analysis %>%
  filter(
    !is.na(bert_duration_days_final),
    bert_duration_days_final >= 0,
    !geothermal_phase %in% c("none", "exploration")
  ) %>%
  count(geothermal_phase) %>%
  mutate(label = paste0("n = ", n))

fig_phase_box <- analysis %>%
  filter(
    !is.na(bert_duration_days_final),
    bert_duration_days_final >= 0,
    !geothermal_phase %in% c("none", "exploration")
  ) %>%
  mutate(duration_plot = pmin(bert_duration_days_final, 250)) %>%
  ggplot(aes(x = geothermal_phase, y = duration_plot, fill = geothermal_phase)) +
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
    data = phase_n_labels,
    aes(x = geothermal_phase, y = 255, label = label),
    size = 3.2, color = "grey40", fontface = "italic", inherit.aes = FALSE
  ) +
  coord_cartesian(ylim = c(0, 260)) +
  scale_fill_manual(values = phase_colors_box) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title    = "NEPA Duration by Geothermal Phase",
    subtitle = "Projects with calculable timelines; values above 250 days topcoded to cap",
    caption  = exploration_caption,
    x        = NULL,
    y        = "Duration (days)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position  = "none",
    plot.caption     = element_text(size = 8, color = "gray40", hjust = 0)
  )

print(fig_phase_box)
ggsave(
  filename = here(figures_dir, "fig_geothermal_phase_duration_boxplot.png"),
  plot = fig_phase_box,
  width = 9,
  height = 6,
  dpi = 300
)

# -- Fig 3: Within-Project Sequencing --
# Diagnostic: check date coverage before building the figure
cat("Geothermal projects with both dates:",
    sum(!is.na(analysis$bert_initiation_date_final) & !is.na(analysis$bert_decision_date_final)),
    "/", nrow(analysis), "\n")

# Projects with >=2 dated actions AND >=2 distinct phases (true sequencing)
multi_phase_keys <- analysis %>%
  filter(!is.na(bert_initiation_date_final), !is.na(bert_decision_date_final)) %>%
  group_by(geothermal_project_key) %>%
  summarise(
    n_dated_actions   = n(),
    n_distinct_phases = n_distinct(geothermal_phase),
    .groups = "drop"
  ) %>%
  filter(n_dated_actions >= 2, n_distinct_phases >= 2)

cat("Projects with >=2 dated actions and >=2 distinct phases:", nrow(multi_phase_keys), "\n")

set.seed(606)
sampled_keys <- multi_phase_keys %>%
  slice_sample(n = min(25, nrow(multi_phase_keys))) %>%
  pull(geothermal_project_key)

sequence_data <- analysis %>%
  filter(
    !is.na(bert_initiation_date_final),
    !is.na(bert_decision_date_final),
    geothermal_project_key %in% sampled_keys
  ) %>%
  group_by(geothermal_project_key) %>%
  mutate(
    n_actions     = n(),
    project_start = min(bert_initiation_date_final)
  ) %>%
  ungroup() %>%
  arrange(project_start, geothermal_project_key)

fig_sequence <- ggplot(sequence_data) +
  geom_segment(
    aes(
      x     = bert_initiation_date_final,
      xend  = bert_decision_date_final,
      y     = fct_reorder(geothermal_project_key, project_start, .fun = min),
      yend  = fct_reorder(geothermal_project_key, project_start, .fun = min),
      color = geothermal_phase
    ),
    linewidth = 1.5,
    alpha = 0.85
  ) +
  scale_color_manual(values = phase_colors, drop = FALSE) +
  labs(
    title    = "Within-Project Geothermal Sequencing",
    subtitle = "Random sample of projects with ≥2 dated actions spanning ≥2 phases",
    x        = "Date",
    y        = "Inferred project key",
    color    = "Phase"
  ) +
  theme_minimal(base_size = 10) +
  theme(axis.text.y = element_text(size = 7))

print(fig_sequence)

ggsave(
  filename = here(figures_dir, "fig_geothermal_within_project_sequence.png"),
  plot = fig_sequence,
  width = 11,
  height = 8,
  dpi = 300
)

cat("Saved outputs to:\n", tables_dir, "\n", figures_dir, "\n")
