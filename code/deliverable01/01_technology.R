# --------------------------
# DELIVERABLE 1: DECARBONIZATION TECHNOLOGY BY TECHNOLOGY TYPE
# --------------------------
# Table 1: Decarbonization Technology by Technology (project_type)
# Includes co-occurrence analysis and related figures

# --------------------------
# SETUP
# --------------------------
rm(list = ls())
source(here::here("code", "deliverable01", "00_setup.R"))
# --------------------------
# PROCESS
# --------------------------

# Parse project types and create working datasets
clean_energy_parsed <- clean_energy %>%
  mutate(project_type_list = map(project_type, fromJSON))

# Explode project_type for table creation
tech_data <- clean_energy %>%
  explode_column("project_type") %>%
  filter(!is.na(project_type) & project_type != "")

tech_data |> glimpse()

# --------------------------
# FIGURES
# --------------------------

# Note: universe/definition figures (projects by review, energy type breakdown)
# are produced by code/_clean_energy/01_clean_energy.R

#
# Deliverable: Decarbonization Technology Bar Chart (by technology)
# ----------------------------------------
clean_energy_summary <- clean_energy_parsed %>%
  select(project_title, project_type_list) %>%
  unnest(project_type_list) %>%
  rename(technology = project_type_list) %>%
  filter(technology %in% clean_energy_tags) %>%
  distinct(project_title, technology) %>%
  count(technology, name = "n") %>%
  mutate(
    percent_projects = 100 * n / n_distinct(clean_energy$project_title),
    # Clean up labels: remove "Renewable Energy Production - " except for "Other"
    technology_label = case_when(
      technology == "Renewable Energy Production - Other" ~ technology,
      TRUE ~ str_remove(technology, "^Renewable Energy Production - ")
    )
  ) %>%
  arrange(desc(percent_projects))

fig_clean_energy_bar <- clean_energy_summary %>%
  ggplot(aes(x = percent_projects,
             y = reorder(technology_label, percent_projects))) +
  geom_col(fill = catf_dark_blue) +
  geom_text(aes(label = scales::comma(n)), hjust = -0.1, size = 3) +
  labs(
    x = "Percent of Projects with Decarbonization Technologies Tag",
    y = NULL,
    title = "Projects with Decarbonization Technologies Tag by Technology Type"
  ) +
  scale_x_continuous(
    labels = function(x) paste0(x, "%"),
    breaks = seq(0, 100, by = 5),
    expand = expansion(mult = c(0, 0.12))
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 9))

fig_clean_energy_bar

# save
ggsave(
  filename = here(figures_dir, "01_clean_energy_bar_chart.png"),
  plot = fig_clean_energy_bar,
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)


#
# Decarbonization Technology Bar Chart by Technology and Process Type (100% stacked)
# --------------------------------------------------------------------
clean_energy_summary_by_process <- clean_energy_parsed %>%
  select(project_title, project_type_list, process_type) %>%
  unnest(project_type_list) %>%
  rename(technology = project_type_list) %>%
  filter(technology %in% clean_energy_tags) %>%
  distinct(project_title, technology, process_type) %>%
  count(technology, process_type, name = "n") %>%
  group_by(technology) %>%
  mutate(
    share = n / sum(n),
    # Clean up labels: remove "Renewable Energy Production - " except for "Other"
    technology_label = case_when(
      technology == "Renewable Energy Production - Other" ~ technology,
      TRUE ~ str_remove(technology, "^Renewable Energy Production - ")
    )
  ) %>%
  ungroup()

# Use same order as fig_clean_energy_bar (by percent_projects)
tech_label_order <- clean_energy_summary %>%
  arrange(percent_projects) %>%
  pull(technology_label)

clean_energy_summary_by_process <- clean_energy_summary_by_process %>%
  mutate(technology_label = factor(technology_label, levels = tech_label_order))

# Totals for label layer (project counts per technology)
tech_totals <- clean_energy_summary_by_process %>%
  group_by(technology_label) %>%
  summarise(total = sum(n), .groups = "drop")

# Plot
fig_clean_energy_bar_by_process <- ggplot(
  clean_energy_summary_by_process,
  aes(
    x = share,
    y = technology_label,
    fill = process_type
  )
) +
  geom_col(width = 0.8) +
  geom_text(
    aes(label = ifelse(share >= 0.03, scales::percent(share, accuracy = 1), "")),
    position = position_stack(vjust = 0.5),
    size = 3,
    color = "white"
  ) +
  geom_text(
    data = tech_totals,
    aes(x = 1.02, y = technology_label, label = scales::comma(total)),
    inherit.aes = FALSE,
    hjust = 0,
    size = 3,
    color = "gray30"
  ) +
  scale_x_continuous(
    labels = scales::percent,
    breaks = seq(0, 1, by = 0.1),
    expand = expansion(mult = c(0, 0.08))
  ) +
  scale_fill_manual(
    values = c("CE" = catf_dark_blue, "EA" = catf_teal, "EIS" = catf_magenta)
  ) +
  labs(
    x = "Share of Projects",
    y = NULL,
    fill = "Process Type",
    title = "Projects with Decarbonization Tag by Technology and Process Type",
    caption = "Note: Percentage labels below 3% are excluded for readability."
  ) +
  theme(
    axis.text.y = element_text(size = 9),
    legend.position = "bottom"
  )

fig_clean_energy_bar_by_process

# save
ggsave(
  filename = here(figures_dir, "02_clean_energy_bar_by_process.png"),
  plot = fig_clean_energy_bar_by_process,
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)




# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Technology Script Complete ===\n")
cat("Tables saved to:", tables_dir, "\n")
cat("Figures saved to:", figures_dir, "\n")
