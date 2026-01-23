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
# PROJECTS
# --------------------------

projects %>%
  glimpse()



projects %>%
  distinct(lead_agency) |> 
  print(n=100)

projects %>%
  select(contains("department")) |> 
  glimpse()
  distinct(project_department) |> 
  print(n=100)

# --------------------------
# TIMELINE
# --------------------------

timeline <- read_parquet(here("data", "analysis", "projects_timeline.parquet")) |> glimpse()


# --------------------------
# DOCUMENTS
# --------------------------
documents <- read_parquet(here("data", "analysis", "documents_combined.parquet")) |> glimpse()

documents |> 
  filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  print()

