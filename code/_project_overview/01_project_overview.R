# --------------------------
# CLEAN ENERGY DEFINITIONS: FIGURES
# --------------------------
# Produces figures for 00_project_overview.qmd:
#   - Projects by NEPA review process (all projects)
#   - Clean energy projects by NEPA review process
#   - Energy type breakdown (Clean, Fossil, Other)

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "_project_overview", "00_setup.R"))

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
# Clean Energy Projects by Review Process
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
    title = "Clean Energy Projects by NEPA Process Type"
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
    project_energy_type = factor(project_energy_type,
                                  levels = c("Clean", "Fossil", "Other"))
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
  scale_fill_manual(values = c("Clean" = catf_teal,
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

cat("\n=== Clean Energy Definitions Script Complete ===\n")
cat("Figures saved to:", figures_dir, "\n")
