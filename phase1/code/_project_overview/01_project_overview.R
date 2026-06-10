# --------------------------
# DECARBONIZATION TECHNOLOGY DEFINITIONS: TABLES AND FIGURES
# --------------------------
# Produces tables and figures for project_overview.qmd:
#   - Clean and fossil project type tag counts
#   - Projects by NEPA review process (all projects)
#   - Decarbonization technology projects by NEPA review process
#   - Energy type breakdown (Clean, Fossil, Other)

# --------------------------
# SETUP
# --------------------------

source(here::here("phase1", "code", "_project_overview", "00_setup.R"))

# --------------------------
# TABLES
# --------------------------

summarise_energy_tag_counts <- function(df, energy_type, tag_map, category_label) {
  tag_reference <- tag_map %>%
    distinct(`Project Type Tag`, tag_order) %>%
    mutate(`Energy Category` = category_label)

  tag_counts <- df %>%
    filter(project_energy_type == energy_type) %>%
    mutate(project_type_list = map(project_type, fromJSON)) %>%
    select(project_id, project_type_list) %>%
    unnest(project_type_list) %>%
    rename(raw_tag = project_type_list) %>%
    inner_join(tag_map, by = "raw_tag") %>%
    distinct(project_id, raw_tag, `Project Type Tag`) %>%
    count(`Project Type Tag`, name = "Projects") %>%
    mutate(`Energy Category` = category_label)

  tag_reference %>%
    left_join(tag_counts, by = c("Energy Category", "Project Type Tag")) %>%
    mutate(Projects = replace_na(Projects, 0L))
}

clean_energy_table_tags <- tibble(raw_tag = clean_energy_tags) %>%
  mutate(
    `Project Type Tag` = if_else(
      str_starts(raw_tag, "Renewable Energy Production -"),
      "Renewable Energy Production",
      raw_tag
    ),
    tag_order = dense_rank(match(`Project Type Tag`, unique(`Project Type Tag`)))
  )

fossil_energy_table_tags <- tibble(
  raw_tag = fossil_energy_tags,
  `Project Type Tag` = fossil_energy_tags,
  tag_order = seq_along(fossil_energy_tags)
)

# Mirrors the Deliverable 01 technology counting architecture:
# parse project_type JSON, unnest tags, keep one project-id/tag pair,
# then sum raw tags by table row within the final energy classification.
energy_tag_counts <- bind_rows(
  summarise_energy_tag_counts(
    projects,
    energy_type = "Clean",
    tag_map = clean_energy_table_tags,
    category_label = "Decarbonization Technology"
  ),
  summarise_energy_tag_counts(
    projects,
    energy_type = "Fossil",
    tag_map = fossil_energy_table_tags,
    category_label = "Fossil Fuel"
  )
) %>%
  arrange(
    factor(`Energy Category`,
           levels = c("Decarbonization Technology", "Fossil Fuel")),
    desc(Projects),
    tag_order
  ) %>%
  select(`Energy Category`, `Project Type Tag`, Projects)

write_csv(energy_tag_counts, file.path(tables_dir, "table1_energy_tag_counts.csv"))
cat("Saved table: table1_energy_tag_counts.csv\n")

# --------------------------
# FIGURES
# --------------------------

#
# All Projects by Review Process
# ----------------------------------------
projects_by_review <- projects %>%
  count(process_type, name = "projects") %>%
  mutate(
    process_type = factor(process_type,
                          levels = c("EIS", "EA", "CE"),
                          labels = c("Environmental Impact Statement (EIS)",
                                     "Environmental Assessment (EA)",
                                     "Categorical Exclusion (CE)"))
  )

fig_projects_by_review <- projects_by_review %>%
  ggplot(aes(x = process_type, y = projects)) +
  geom_col(fill = catf_dark_blue) +
  geom_text(aes(label = scales::comma(projects)), vjust = -0.3, size = 3.5) +
  labs(
    x = NULL,
    y = "Number of Projects",
    title = "Projects by NEPA Process Type"
  ) +
  scale_y_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.12))
  )

fig_projects_by_review

ggsave(
  filename = here(figures_dir, "00_fig_projects_by_review.png"),
  plot = fig_projects_by_review,
  width = 8,
  height = 5,
  dpi = 300
)

#
# Decarbonization Technology Projects by Review Process
# ----------------------------------------
clean_energy_by_review <- clean_energy %>%
  count(process_type, name = "projects") %>%
  mutate(
    process_type = factor(process_type,
                          levels = c("EIS", "EA", "CE"),
                          labels = c("Environmental Impact Statement (EIS)",
                                     "Environmental Assessment (EA)",
                                     "Categorical Exclusion (CE)"))
  )

fig_clean_energy_by_review <- clean_energy_by_review %>%
  ggplot(aes(x = process_type, y = projects)) +
  geom_col(fill = catf_dark_blue) +
  geom_text(aes(label = scales::comma(projects)), vjust = -0.3, size = 3.5) +
  labs(
    x = NULL,
    y = "Number of Projects",
    title = "Decarbonization Technology Projects by NEPA Process Type"
  ) +
  scale_y_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.12))
  )

fig_clean_energy_by_review

ggsave(
  filename = here(figures_dir, "00_fig_clean_energy_by_review.png"),
  plot = fig_clean_energy_by_review,
  width = 8,
  height = 5,
  dpi = 300
)

#
# Energy Type Breakdown (Clean, Fossil, Other)
# ----------------------------------------
# Note: project_energy_type already has final classification from Python pipeline
energy_type_summary <- projects %>%
  count(project_energy_type, name = "projects") %>%
  mutate(
    share = projects / sum(projects),
    project_energy_type = factor(
      if_else(project_energy_type == "Clean", "Decarbonized", project_energy_type),
      levels = c("Decarbonized", "Fossil", "Other")
    )
  )

fig_energy_type <- energy_type_summary %>%
  ggplot(aes(x = reorder(project_energy_type, -projects), y = projects,
             fill = project_energy_type)) +
  geom_col() +
  geom_text(aes(label = paste0(scales::comma(projects), "\n(",
                                scales::percent(share, accuracy = 0.1), ")")),
            vjust = -0.2, size = 3.5) +
  labs(
    x = NULL,
    y = "Number of Projects",
    title = "NEPA Projects by Energy Type"
  ) +
  scale_y_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.15))
  ) +
  scale_fill_manual(values = c("Decarbonized" = catf_teal,
                                "Fossil" = catf_navy,
                                "Other" = catf_light_blue)) +
  theme(legend.position = "none")

fig_energy_type

ggsave(
  filename = here(figures_dir, "00_energy_type_breakdown.png"),
  plot = fig_energy_type,
  width = 8,
  height = 5,
  dpi = 300
)

# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Decarbonization Technology Definitions Script Complete ===\n")
cat("Tables saved to:", tables_dir, "\n")
cat("Figures saved to:", figures_dir, "\n")
