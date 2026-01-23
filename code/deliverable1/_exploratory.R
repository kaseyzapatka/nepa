# --------------------------
# DELIVERABLE 1: EXPLORATORY SCRIPT
# --------------------------
# Exploratory analysis for anything associated with deliverable 1

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable1", "00_setup.R"))

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



