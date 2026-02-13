# --------------------------
# PROJECT REVIEW: NUCLEAR WASTE
# --------------------------

source(here::here("code", "deliverable01", "00_setup.R"))

# Explode lead_agency (handles rare multi-agency cases)
agency_data <- clean_energy %>%
  explode_column("lead_agency") %>%
  filter(!is.na(lead_agency) & lead_agency != "")

# Use pre-computed department column (renamed for consistency with this script)
agency_data <- agency_data %>%
  mutate(department = project_department)

# Frank wanted to know if there were ways to disaggregate the nuclear waste reviews?
# Over several iterations, we removed a number of agencies and sites that were not
# deemed to be "clean energy" by CAFT. Check notes/agencies_to_be_excluded.txt for a
# full list of agencies/sites that were removed from both the lead_agency column. We
# also removed

# 4,068
nuclear_waste_projects <-
  projects |>
  filter(str_detect(project_type, "Waste Management") & str_detect(project_type, "Nuclear")) |>
  filter(project_nuclear_waste_to_exclude) |> 
  select(project_id, project_title, project_department,lead_agency_harmonized, project_sponsor,project_type) |> 
  glimpse()

# save
sheet_write(
  data = nuclear_waste_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "nuclear_waste_projects")

# 34 to keep
nuclear_waste_projects_to_keep <-
  agency_data |>
  filter(str_detect(project_type, "Waste Management") & str_detect(project_type, "Nuclear")) |>
  select(project_id, project_title, department, lead_agency, project_sponsor, project_type) |>
  arrange(department) |>
  glimpse()

# save
sheet_write(
  data = nuclear_waste_projects_to_keep,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "nuclear_waste_projects_to_keep")
