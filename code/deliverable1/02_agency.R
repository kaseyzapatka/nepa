# --------------------------
# DELIVERABLE 1: CLEAN ENERGY BY LEAD AGENCY
# --------------------------
# Table 2: Clean Energy by Lead Agency
# Analysis of which agencies handle clean energy projects

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable1", "00_setup.R"))

#install.packages("googledrive")
library(googlesheets4)


# --------------------------
# PROCESS
# --------------------------

# Note: Only 40 of 61,881 projects (0.06%) have multiple lead agencies.
# We keep explode_column for completeness, but this is rare.

# Explode lead_agency (handles rare multi-agency cases)
agency_data <- clean_energy %>%
  explode_column("lead_agency") %>%
  filter(!is.na(lead_agency) & lead_agency != "")

# Create collapsed department-level grouping
agency_data <- agency_data %>%
  mutate(
    department = case_when(
      # Department of Energy
      str_detect(lead_agency, "^Department of Energy") ~ "Department of Energy",

      # Department of the Interior
      str_detect(lead_agency, "^Department of the Interior") ~ "Department of the Interior",

      # Department of Agriculture
      str_detect(lead_agency, "^Department of Agriculture") ~ "Department of Agriculture",

      # Department of Defense
      str_detect(lead_agency, "^Department of Defense") ~ "Department of Defense",

      # Department of Homeland Security
      str_detect(lead_agency, "^Department of Homeland Security") ~ "Department of Homeland Security",

      # Department of Transportation
      str_detect(lead_agency, "^Department of Transportation") ~ "Department of Transportation",

      # Health and Human Services
      str_detect(lead_agency, "^Department of Health and Human Services") ~
        "Department of Health and Human Services",

      # Housing and Urban Development
      str_detect(lead_agency, "^Department of Housing and Urban Development") ~
        "Department of Housing and Urban Development",

      # Commerce
      str_detect(lead_agency, "^Department of Commerce") ~ "Department of Commerce",

      # State
      str_detect(lead_agency, "^Department of State") ~ "Department of State",

      # Justice
      str_detect(lead_agency, "^Department of Justice") ~ "Department of Justice",

      # Veterans Affairs
      str_detect(lead_agency, "^Department of Veterans Affairs") ~
        "Department of Veterans Affairs",

      # Treasury
      str_detect(lead_agency, "^Department of the Treasury") ~
        "Department of the Treasury",

      # Major Independent Agencies
      str_detect(lead_agency, "^Major Independent Agencies") ~
        "Major Independent Agencies",

      # Other Independent Agencies
      str_detect(lead_agency, "^Other Independent Agencies") ~
        "Other Independent Agencies",

      # GSA
      str_detect(lead_agency, "^General Services Administration") ~
        "General Services Administration",

      # Legislative
      lead_agency == "Legislative Branch" ~ "Legislative Branch",

      # International
      lead_agency == "International Assistance Programs" ~
        "International Assistance Programs",

      # Fallback
      TRUE ~ "Other / Unclassified"
    )
  )

# Count projects per agency (detailed)
agency_counts <- agency_data %>%
  count(lead_agency, name = "n_projects") %>%
  arrange(desc(n_projects))

# Count projects per department (collapsed)
department_counts <- agency_data %>%
  count(department, name = "n_projects") %>%
  arrange(desc(n_projects))

cat("Unique agencies (detailed):", nrow(agency_counts), "\n")
cat("Unique departments (collapsed):", nrow(department_counts), "\n\n")

cat("Top 10 agencies by project count:\n")
agency_counts %>% slice_head(n = 10) %>% print()

cat("\nDepartment-level counts:\n")
department_counts %>% print()

# --------------------------
# EXPLORATORY
# --------------------------

#
# Military
# ----------------------------------------
# Frank wanted to know the agency mix for  defense related nuclear projects --  DOE or DOD ? 
# nearly all of 481 were DOE
military_projects <- 
  agency_data |> 
  filter(str_detect(project_type, "Military and Defense") & str_detect(project_type, "Nuclear")) |> 
  select(project_title, department, project_type) |> 
  arrange(department) |> 
  glimpse()


# save
sheet_write(
  data = military_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "military_projects"
)

#
# Nuclear Waste
# ----------------------------------------
# Frank wanted to know if there were ways to disaggregate the nuclear waste reviews?

nuclear_waste_projects <- 
  agency_data |> 
  filter(str_detect(project_type, "Waste Management") & str_detect(project_type, "Nuclear")) |> 
  select(project_title, department, project_sponsor, project_type) |> 
  arrange(department) |> 
  glimpse()

# save
sheet_write(
  data = nuclear_waste_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "nuclear_waste_projects"
)


# --------------------------
# TABLE 2: BY DEPARTMENT (AGENCY COLLAPSED)
# --------------------------
# This table collapses lead agency into department for parsimony

cat("\nCreating Table 2b: Clean Energy by Department (Collapsed)...\n")

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
cat("  Saved: table2_by_department.csv\n")


# --------------------------
# FIGURES
# --------------------------

# Figure 2: Department-Level Bar Chart (Collapsed)
fig_departments <- department_counts %>%
  filter(department != "Other / Unclassified") %>%
  ggplot(aes(x = n_projects, y = reorder(department, n_projects))) +
  geom_col(fill = "steelblue") +
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
  filename = here(figures_dir, "06_departments.png"),
  plot = fig_departments,
  width = 10,
  height = 7,
  units = "in",
  dpi = 300
)
cat("  Saved: 06_departments.png\n")


# Figure 3: Department by Process Type
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
  ggplot(aes(x = reorder(department, total), y = percent, fill = process_type)) +
  geom_col() +
  coord_flip() +
  labs(
    x = NULL,
    y = "Percent of Projects",
    fill = "Process Type",
    title = "Process Type Distribution by Federal Department"
  ) +
  scale_fill_brewer(palette = "Set1") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 9))

fig_dept_process

ggsave(
  filename = here(figures_dir, "07_dept_process_type.png"),
  plot = fig_dept_process,
  width = 10,
  height = 7,
  units = "in",
  dpi = 300
)
cat("  Saved: 07_dept_process_type.png\n")


# --------------------------
# ANALYSIS
# --------------------------

cat("\n=== Agency Analysis ===\n")

# Multi-agency projects (rare but worth noting)
cat("\nMulti-agency projects in dataset:\n")
multi_agency <- clean_energy %>%
  filter(str_detect(lead_agency, ","))
cat("  Count:", nrow(multi_agency), "(these have >1 lead agency)\n")

# Agencies with most EIS (complex projects)
eis_agencies <- agency_data %>%
  filter(process_type == "EIS") %>%
  count(lead_agency, name = "n_eis") %>%
  arrange(desc(n_eis))

cat("\nTop 10 agencies by EIS count (most complex projects):\n")
eis_agencies %>% slice_head(n = 10) %>% print()

# Departments with most EIS
eis_depts <- agency_data %>%
  filter(process_type == "EIS") %>%
  count(department, name = "n_eis") %>%
  arrange(desc(n_eis))

cat("\nDepartments by EIS count:\n")
eis_depts %>% print()

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


# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Agency Script Complete ===\n")
cat("Tables saved to:", tables_dir, "\n")
cat("Figures saved to:", figures_dir, "\n")
