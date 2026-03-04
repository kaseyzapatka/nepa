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
    subtitle = "914 decarbonization technology projects identified in NEPATEC 2.0",
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
