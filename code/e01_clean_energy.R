# --------------------------
# DEFINING CLEAN ENERGY
# --------------------------
# This file is an exploratory analyses to help us define clean energy for the 
# the project.

# --------------------------
# SETUP
# --------------------------

# clear environment
rm(list=ls()) 

# source setup files
source(here::here("code", "00_setup.R"))

# --------------------------
# DEFINING CLEAN ENERGY
# --------------------------

#
# Utilities only
# ----------------------------------------
# There were about 1,623 projects that had some combination of Utilities we didn't want 
# to count as clean energy
# ALL 1,623 WERE REMOVED FROM "CLEAN ENERGY" AND RECODED AS "OTHER"

# Identify utility only projects to exclude
utilty_projects_to_exclude <-
  projects |> 
  filter(project_energy_type == "Clean") |> 
  # remove Utilities + Broadband, Waste Management, or Land Development tags
  filter(project_utilities_to_filter_out) |> 
  select(project_id, project_title, project_department, project_type) |> 
  glimpse()

# save to google sheets so team can view
sheet_write(
  data = utilty_projects_to_exclude,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "utilty_projects_to_exclude"
)


#
# Military
# ----------------------------------------
# Frank wanted to know the agency mix for  defense related nuclear projects 
# nearly all of 482  were DOE
# ALL 482 WERE REMOVED FROM "CLEAN ENERGY" AND RECODED AS "OTHER"

military_projects_to_exclude <- 
  projects |> 
  filter(str_detect(project_type, "Military and Defense") & str_detect(project_type, "Nuclear")) |> 
  select(project_id, project_title, project_department, project_type) |> 
  arrange(project_department) |> 
  glimpse()

# save to google sheets so team can view
sheet_write(
  data = military_projects_to_exclude,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "military_projects_to_exclude")

# identify project IDs
military_project_ids_to_filter <- 
  military_projects_to_exclude |> 
  select(project_id) |> 
  glimpse()

# save project IDs so they can be filtered out in next run
write_csv(military_project_ids_to_filter, here("notes", "military_project_ids_to_filter.csv"))


#
# Nuclear Waste
# ----------------------------------------
# Frank wanted to know if there were ways to disaggregate the nuclear waste reviews?
# Over several iterations, we removed a number of agencies and sites that were not 
# deemed to be "clean energy" by CAFT. Check notes/agencies_to_be_excluded.txt for a 
# full list of agencies/sites that were removed from both the lead_agency column. We 
# also removed   

# Identify nuclear waste projects to keep 
nuclear_waste_projects <- 
  agency_data |> 
  filter(str_detect(project_type, "Waste Management") & str_detect(project_type, "Nuclear")) |> 
  select(project_id, project_title, department, lead_agency, project_sponsor, project_type) |> 
  arrange(department) |> 
  glimpse() # 1,590

# save to google sheets so team can view
sheet_write(
  data = nuclear_waste_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "nuclear_waste_projects")



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
