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

# Note: project_department is now pre-computed in the Python extract pipeline.
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
  select(project_id, project_title, department, project_type) |> 
  arrange(department) |> 
  glimpse()

# save
sheet_write(
  data = military_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "military_projects"
)

# pull all project IDs so we can filter out
military_project_ids_to_filter <- 
  military_projects |> 
  select(project_id) |> 
  glimpse()

# Save
write_csv(military_project_ids_to_filter, here("notes",  "military_project_ids_to_filter.csv"))


#
# Nuclear Waste
# ----------------------------------------
# Frank wanted to know if there were ways to disaggregate the nuclear waste reviews?
nuclear_waste_projects <- 
  agency_data |> 
  filter(str_detect(project_type, "Waste Management") & str_detect(project_type, "Nuclear")) |> 
  select(project_id, project_title, department, lead_agency, project_sponsor, project_type) |> 
  arrange(department) |> 
  glimpse()

# save
sheet_write(
  data = nuclear_waste_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "nuclear_waste_projects"
)

# remove NNSA, LM, and EM
nuclear_waste_projects_refined <- 
  nuclear_waste_projects |> 
  filter(
    # Filter for project_sponsor
    str_detect(project_sponsor, regex(
      paste(
        # NNSA patterns
        "\\bNNSA\\b",
        "National Nuclear Security Administration",
        "National Nuclear(?!\\s+Laboratory)",  # Excludes "National Nuclear Laboratory" 
        "Kansas City.{0,20}(Field Office|National Security Campus)",
        "Livermore.{0,20}(Field Office|Site Office)",
        "Lawrence Livermore National Laboratory",
        "Los Alamos.{0,20}(Field Office|Site Office|Area Office|National Laboratory)",
        "Office of Los Alamos Site Operations",  # NEW
        "Naval Nuclear.{0,20}(Propulsion Program|Laboratory)",
        "Nevada.{0,20}(Field Office|National Security Site)",
        "Pantex.{0,20}(Field Office|Plant)?",
        "\\bPantex\\b",
        "Sandia.{0,20}(Field Office|National Laboratories)",
        "Sandia Site Office",  # NEW
        "Savannah River.{0,20}(Mission Completion|Operations Office|Field Office|Site)?",
        "\\bSavannah River\\b",
        "Y-12.{0,20}(Site Office|Field Office|National Security Complex)",
        
        # Legacy Management patterns
        "Office of Legacy Management",
        "Legacy Management",
        
        # Environmental Management patterns
        "Office of Environmental Management",
        "Environmental Management",
        "Hanford Site",
        "Hanford Mission Integration Solutions",  # NEW
        "\\bRichland\\b",
        "Richland Operations.{0,20}Office?",
        "Office of River Protection",
        "\\bPaducah\\b",
        "Paducah Site",
        "\\bPortsmouth\\b",
        "Portsmouth Site",
        "Portsmouth and Paducah Field Office",
        "\\bOak Ridge\\b",
        "Oak Ridge Office of Environmental Management",
        "Waste Isolation Pilot Plant",
        "\\bCarlsbad\\b",
        "Carlsbad Field Office",
        
        sep = "|"
      ),
      ignore_case = TRUE
    )) |
    
    # OR filter for lead_agency (three major agencies only)
    str_detect(lead_agency, regex(
      paste(
        "National Nuclear Security Administration",
        "Office of Legacy Management",
        "Office of Environmental Management",
        sep = "|"
      ),
      ignore_case = TRUE
    ))
  ) |> 
  select(project_id, project_title, department, lead_agency, project_sponsor, project_type) |> 
  arrange(department) |> 
  glimpse()


# save
sheet_write(
  data = nuclear_waste_projects_refined,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "nuclear_waste_projects_refined"
)

# pull ids to filter
nuclear_waste_projects_ids_to_filter <- 
  nuclear_waste_projects_refined |> 
  select(project_id) |> 
  glimpse()

# Save
write_csv(nuclear_waste_projects_ids_to_filter, here("notes",  "nuclear_waste_projects_ids_to_filter.csv"))


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

