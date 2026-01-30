# --------------------------
# DELIVERABLE 1: CLEAN ENERGY BY LEAD AGENCY
# --------------------------
# Table 2: Clean Energy by Lead Agency
# Analysis of which agencies handle clean energy projects

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable1", "00_setup.R"))

# --------------------------
# PROCESS
# --------------------------

# Note: project_department pre-computed in the Python extract pipeline.
# Only 40 of 61,881 projects (0.06%) have multiple lead agencies.
# We keep explode_column for lead_agency detail analysis, but use the
# pre-computed project_department for department-level grouping.

# Explode lead_agency (handles rare multi-agency cases)
agency_data <- clean_energy %>%
  explode_column("lead_agency") %>%
  filter(!is.na(lead_agency) & lead_agency != "")

# Use pre-computed department column (renamed for consistency with this script)
agency_data <- agency_data %>%
  mutate(department = project_department)

# Count projects per agency (detailed)
agency_counts <- agency_data %>%
  count(lead_agency, name = "n_projects") %>%
  arrange(desc(n_projects))

# Count projects per department (collapsed)
department_counts <- agency_data %>%
  count(department, name = "n_projects") %>%
  arrange(desc(n_projects))

# --------------------------
# EXPLORATORY
# --------------------------



# --------------------------
# TABLE: PROJECTS BY DEPARTMENT 
# --------------------------
# This table collapses lead agency into department for parsimony

table2 <- create_crosstab(agency_data, "department")

# Add totals row
table2 <- add_totals_row(table2, "department")

# Rename for clarity
table2 <- table2 %>%
  rename(
    Department = department,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )

table2 %>% print(n = 25)

# Save
write_csv(table2, here(tables_dir, "table2_by_department.csv"))



# --------------------------
# FIGURES
# --------------------------

#
# Deliverable: Department Bar Chart 
# ----------------------------------------
fig_departments <- department_counts %>%
  filter(department != "Other / Unclassified") %>%
  ggplot(aes(x = n_projects, y = reorder(department, n_projects))) +
  geom_col(fill = catf_dark_blue) +
  geom_text(aes(label = scales::comma(n_projects)), hjust = -0.1, size = 3) +
  labs(
    x = "Number of Clean Energy Projects",
    y = NULL,
    title = "Clean Energy Projects by Federal Department"
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 10)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15)), labels = scales::comma)

fig_departments

ggsave(
  filename = here(figures_dir, "02_departments.png"),
  plot = fig_departments,
  width = 10,
  height = 7,
  units = "in",
  dpi = 300
)


#
# Deliverable: Departments by review process 
# ----------------------------------------
dept_process <- agency_data %>%
  count(department, process_type) %>%
  group_by(department) %>%
  mutate(
    total = sum(n),
    percent = 100 * n / total
  ) %>%
  ungroup() %>%
  filter(department != "Other / Unclassified")

fig_dept_process <- dept_process %>%
  filter(total >= 5) |> 
  ggplot(aes(x = reorder(department, total), y = percent, fill = process_type)) +
  geom_col() +
  coord_flip() +
  labs(
    x = NULL,
    y = "Percent of Projects",
    fill = "Process Type",
    title = "Process Type Distribution by Federal Department",
    caption = "Note that Departments with fewer than 5 projects were removed for parsimony."
  ) +
  #scale_fill_brewer(palette = "Set1") +
  scale_fill_manual(
    values = c("CE" = catf_dark_blue, "EA" = catf_teal, "EIS" = catf_magenta)
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 9))

fig_dept_process

ggsave(
  filename = here(figures_dir, "02_department_process.png"),
  plot = fig_dept_process,
  width = 10,
  height = 7,
  units = "in",
  dpi = 300
)


# --------------------------
# SUMMARY ANALYSIS
# --------------------------

cat("\n=== Agency Analysis ===\n")

# Multi-agency projects (rare but worth noting)
cat("\nMulti-agency projects in dataset:\n")
multi_agency <- clean_energy %>%
  filter(str_detect(lead_agency, ","))
cat("  Count:", nrow(multi_agency), "(these have >1 lead agency)\n")

# Agencies with highest CE ratio (streamlined projects)
ce_ratio <- agency_data %>%
  count(lead_agency, process_type) %>%
  pivot_wider(names_from = process_type, values_from = n, values_fill = 0) %>%
  mutate(
    total = EA + EIS + CE,
    ce_ratio = CE / total
  ) %>%
  filter(total >= 50) %>%
  arrange(desc(ce_ratio))

cat("\nAgencies with highest CE ratio (min 50 projects):\n")
ce_ratio %>% slice_head(n = 10) %>% print()
