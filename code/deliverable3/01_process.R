# --------------------------
# DELIVERABLE 3: PROJECT STATUS BY ENERGY TYPE
# --------------------------
# Table 1: Project Status by Energy Type (Clean/Fossil/Other)
# Includes detailed breakdown of clean energy by technology

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable3", "00_setup.R"))

# --------------------------
# RECLASSIFY UTILITIES TO OTHER
# --------------------------
# Projects tagged as Clean but with Utilities/Broadband, Waste Management,
# or Land Development should be classified as "Other" for reporting

projects <- projects %>%
  mutate(
    project_energy_type = if_else(
      project_energy_type == "Clean" & project_utilities_to_filter_out,
      "Other",
      project_energy_type
    )
  )

cat("After reclassifying utilities to Other:\n")
cat("  Clean energy projects:", sum(projects$project_energy_type == "Clean"), "\n")
cat("  Other projects:", sum(projects$project_energy_type == "Other"), "\n\n")

# --------------------------
# TABLE 1: PROJECT STATUS BY ENERGY TYPE
# --------------------------

cat("Creating Table 1: Project Status by Energy Type...\n")

table1 <- projects %>%
  group_by(project_energy_type, process_type) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(
    names_from = process_type,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(Total = rowSums(select(., -1), na.rm = TRUE)) %>%
  arrange(desc(Total))

# Add column totals
totals_row <- table1 %>%
  summarise(
    project_energy_type = "Total",
    EA = sum(EA, na.rm = TRUE),
    EIS = sum(EIS, na.rm = TRUE),
    CE = sum(CE, na.rm = TRUE),
    Total = sum(Total, na.rm = TRUE)
  )

table1 <- bind_rows(table1, totals_row)

# Rename for clarity
table1 <- table1 %>%
  rename(
    `Energy Type` = project_energy_type,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )

table1 %>% print()

# Save
write_csv(table1, here("output", "deliverable3", "tables", "table1_by_energy_type.csv"))
cat("  Saved: table1_by_energy_type.csv\n")


# --------------------------
# DETAILED BREAKDOWN: CLEAN ENERGY BY TECHNOLOGY x PROCESS TYPE
# --------------------------

cat("\nCreating supplementary table: Clean Energy Detail...\n")

# Use clean_energy from setup (already filtered to exclude utilities)
# Or filter here since we've reclassified utilities to "Other" above
clean_energy_detail <- projects %>%
  filter(project_energy_type == "Clean")

cat("Clean energy projects for detail table:", nrow(clean_energy_detail), "\n")

clean_by_tech <- clean_energy_detail %>%
  explode_column("project_type") %>%
  filter(!is.na(project_type) & project_type != "") %>%
  group_by(project_type, process_type) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(
    names_from = process_type,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(Total = rowSums(select(., -1), na.rm = TRUE)) %>%
  arrange(desc(Total))

# Add totals
totals_row <- clean_by_tech %>%
  summarise(
    project_type = "Total",
    EA = sum(EA, na.rm = TRUE),
    EIS = sum(EIS, na.rm = TRUE),
    CE = sum(CE, na.rm = TRUE),
    Total = sum(Total, na.rm = TRUE)
  )

clean_by_tech <- bind_rows(clean_by_tech, totals_row)

clean_by_tech <- clean_by_tech %>%
  rename(
    Technology = project_type,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )

clean_by_tech %>% print(n = 20)

write_csv(clean_by_tech, here(tables_dir, "clean_energy_by_technology_detail.csv"))
cat("  Saved: clean_energy_by_technology_detail.csv\n")


# --------------------------
# ENERGY TYPE COUNTS SUMMARY
# --------------------------

cat("\n=== Energy Type Summary ===\n")
summary_stats <- projects %>%
  group_by(project_energy_type) %>%
  summarise(
    count = n(),
    pct = n() / nrow(projects) * 100
  ) %>%
  arrange(desc(count)) %>%
  rename(
    `Energy Type` = project_energy_type,
    `Count` = count,
    `Percent` = pct
  )

print(summary_stats)

write_csv(summary_stats, here(tables_dir, "energy_type_summary.csv"))
cat("  Saved: energy_type_summary.csv\n")

cat("\nProjects flagged for review:", sum(projects$project_energy_type_questions, na.rm = TRUE), "\n")


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
  filename = here("output", "deliverable3", "figures", "01_energy_type_grouped.png"),
  plot = fig1,
  width = 10,
  height = 5,
  units = "in",
  dpi = 300
)

# Figure 2: Stacked bar chart showing composition
fig2 <- fig_data %>%
  ggplot(aes(x = reorder(project_energy_type, total_energy_type), y = pct, fill = process_type)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = ifelse(pct > 5, paste0(round(pct, 0), "%"), "")),
    position = position_stack(vjust = 0.5),
    color = "white",
    size = 3.5,
    fontface = "bold"
  ) +
  coord_flip() +
  labs(
    title = "Process Type Composition Within Energy Types",
    x = NULL,
    y = "Percent of Projects",
    fill = "Process",
    caption = "NEPA review processes: CE (Categorical Exclusion), EA (Environmental Assessment), EIS (Environmental Impact Statement). \nPercentages calculated within each energy type category."
  ) +
  scale_y_continuous(labels = percent_format(scale = 1), expand = expansion(mult = c(0, 0.02))) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal() +
  theme(
    legend.position = "top",
    plot.subtitle = element_text(size = 9, color = "gray40"),
    plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
  )

fig2

ggsave(
  filename = here("output", "deliverable3", "figures", "02_energy_type_composition.png"),
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
