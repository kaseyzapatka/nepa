# --------------------------
# DELIVERABLE 1: EXPLORATORY SCRIPT
# --------------------------
# Exploratory analysis for anything associated with deliverable 1

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable1", "00_setup.R"))

# --------------------------
# LOAD SPECIFIC DATA
# --------------------------
# Explode lead_agency (handles rare multi-agency cases)
agency_data <- clean_energy %>%
  explode_column("lead_agency") %>%
  filter(!is.na(lead_agency) & lead_agency != "")

# Use pre-computed department column (renamed for consistency with this script)
agency_data <- agency_data %>%
  mutate(department = project_department)

# --------------------------
# EXPLORATORY
# --------------------------

#
# Utilities only
# ----------------------------------------
# There were about 1,623 projects that had some combination of Utilities we didn't want 
# to count as clean energy
utilties_only_projects <-
  projects |> 
  filter(project_energy_type == "Clean") |> 
  # remove Utilities + Broadband, Waste Management, or Land Development tags
  filter(project_utilities_to_filter_out) |> 
  select(project_title, project_type) |> 
  glimpse()

# save
sheet_write(
  data = utilties_only_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "utilties_only_projects"
)

#
# Military
# ----------------------------------------
# Frank wanted to know the agency mix for  defense related nuclear projects 
# nearly all of 481 were DOE
# THESE WERE ALL REMOVED FROM CLEAN ENERGY

military_projects <- 
  projects |> 
  mutate(department = project_department) |> 
  filter(str_detect(project_type, "Military and Defense") & str_detect(project_type, "Nuclear")) |> 
  select(project_id, project_title, department, project_type) |> 
  arrange(department) |> 
  glimpse()

# save
sheet_write(
  data = military_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "military_projects_to_exclude")


military_project_ids_to_filter <- 
  military_projects |> 
  select(project_id) |> 
  glimpse()

write_csv(military_project_ids_to_filter, here("notes", "military_project_ids_to_filter.csv"))


#
# Nuclear Waste
# ----------------------------------------
# Frank wanted to know if there were ways to disaggregate the nuclear waste reviews?
# Over several iterations, we removed a number of agencies and sites that were not 
# deemed to be "clean energy" by CAFT. Check notes/agencies_to_be_excluded.txt for a 
# full list of agencies/sites that were removed from both the lead_agency column. We 
# also removed    

# 1,659 now 1,666
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
  sheet = "nuclear_waste_projects")


