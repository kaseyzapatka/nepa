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
  #select(process_type) |> 
  #glimpse()
  #mutate(process_type = map_chr(process_type, ~ paste(fromJSON(.x), collapse = ", "))) |> 
  #glimpse()
  #filter(str_detect(project_sector, regex("geothermal", ignore_case = TRUE))) |> 
  filter(str_detect(project_type, regex("geothermal", ignore_case = TRUE))) |> 
  #filter(str_detect(project_title, regex("geothermal", ignore_case = TRUE))) |> 
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

#
# check to see if prepared_by column in documents provides any new information -- not really 
# ------------------------------

# sample generatro
documents |> 
  filter(dataset_source == "EIS") |> 
  filter(!is.na(prepared_by)) |> 
  slice_sample(n = 1) |> 
  select(project_id) |> 
  print()

documents |> 
  filter(project_id == "832c45f82b287d606cb23b5e898342fe") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()

documents |> 
  filter(project_id == "4f737e61f906e379aaf5b0670a50ae12") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()

documents |> 
  filter(project_id == "a7c79217cc119261606c44a0b85e7042") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()

documents |> 
  filter(project_id == "c8ef7f23d1dde9411d38fa045d364197") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()

documents |> 
  filter(project_id == "30ceb76704ebe7a9ea177c3ef2b2e84e") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()


documents |> 
  filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()


documents |> 
  filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()
