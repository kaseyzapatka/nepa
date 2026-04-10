# --------------------------
# DELIVERABLE 3: PROJECT STATUS BY ENERGY TYPE
# --------------------------
# Table 1: Project Status by Energy Type (Clean/Fossil/Other)
# Includes detailed breakdown of decarbonization technologies by technology

# --------------------------
# SETUP
# --------------------------

source(here::here("phase1", "code", "deliverable03", "00_setup.R"))

# --------------------------
# RECLASSIFY UTILITIES TO OTHER
# --------------------------
# Projects tagged as Clean but with Utilities/Broadband, Waste Management,
# or Land Development should be classified as "Other" for reporting

#projects <- projects %>%
#  mutate(
#    project_energy_type = if_else(
#      project_energy_type == "Clean" & project_utilities_to_filter_out,
#      "Other",
#      project_energy_type
#    )
#  )

cat("After reclassifying utilities to Other:\n")
cat("Decarbonization technologies projects:", sum(projects$project_energy_type == "Clean"), "\n")
cat("Other projects:", sum(projects$project_energy_type == "Other"), "\n\n")

# --------------------------
# FIGURES
# --------------------------

cat("\nCreating Figure 1: Project Status by Energy Type...\n")

# Prepare data for plotting (exclude Total row, long format)
fig_data <- projects %>%
  group_by(project_energy_type, process_type) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(project_energy_type) %>%
  mutate(
    total_energy_type = sum(n),
    pct = 100 * n / total_energy_type
  ) %>%
  ungroup()

# Figure 1: Grouped bar chart comparing process types

fig1 <- fig_data %>%
  mutate(project_energy_type = if_else(project_energy_type == "Clean", "Decarbonized", project_energy_type)) %>%
  ggplot(aes(
    x = n,
    y = reorder(project_energy_type, total_energy_type),
    fill = process_type
  )) +
  geom_col(position = position_dodge(width = 0.9)) +
  geom_text(
    aes(label = comma(n)),               # <-- format with commas
    position = position_dodge(width = 0.9),
    hjust = -0.1,                        # slightly outside the bar
    size = 3
  ) +
  labs(
    title = "Project Counts by Energy Type and Process Type",
    x = "Number of Projects",
    y = NULL,
    fill = "Process",
    caption = "NEPA review processes: CE (Categorical Exclusion), EA (Environmental Assessment), EIS (Environmental Impact Statement)"
  ) +
  scale_x_continuous(labels = comma, expand = expansion(mult = c(0, 0.05))) +
  scale_fill_catf() +
  theme_catf()

fig1

ggsave(
  filename = here("phase1", "output", "deliverable3", "figures", "01_energy_type_grouped.png"),
  plot = fig1,
  width = 10,
  height = 5,
  units = "in",
  dpi = 300
)

# Figure 2: Stacked bar chart showing composition
fig2 <- fig_data %>%
  mutate(project_energy_type = if_else(project_energy_type == "Clean", "Decarbonized", project_energy_type)) %>%
  ggplot(aes(x = reorder(project_energy_type, total_energy_type), y = pct, fill = process_type)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = ifelse(pct > 5, paste0(round(pct, 0), "%"), "")),
    position = position_stack(vjust = 0.5),
    color = "white",
    size = 3.5,
    fontface = "bold"
  ) +
  geom_text(
    data = fig_data %>%
      mutate(project_energy_type = if_else(project_energy_type == "Clean", "Decarbonized", project_energy_type)) %>%
      distinct(project_energy_type, total_energy_type),
    aes(x = reorder(project_energy_type, total_energy_type), y = 101,
        label = scales::comma(total_energy_type)),
    inherit.aes = FALSE,
    hjust = 0,
    size = 3,
    color = "gray30"
  ) +
  coord_flip() +
  labs(
    title = "Process Type Composition Within Energy Types",
    x = NULL,
    y = "Percent of Projects",
    fill = "Process",
    caption = "NEPA review processes: CE (Categorical Exclusion), EA (Environmental Assessment), EIS (Environmental Impact Statement). \nPercentages calculated within each energy type category."
  ) +
  scale_y_continuous(labels = percent_format(scale = 1),
                     expand = expansion(mult = c(0, 0.08))) +
  #scale_fill_brewer(palette = "Set2") +
  scale_fill_catf() +
  theme_catf()
  theme(
    legend.position = "top",
    plot.subtitle = element_text(size = 9, color = "gray40"),
    plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
  )

fig2

ggsave(
  filename = here("phase1", "output", "deliverable3", "figures", "02_energy_type_composition.png"),
  plot = fig2,
  width = 10,
  height = 5,
  units = "in",
  dpi = 300
)


# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Process Script Complete ===\n")
cat("Tables saved to:", tables_dir, "\n")
cat("Figures saved to:", figures_dir, "\n")
