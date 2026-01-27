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
# LOAD DATA
# --------------------------
documents <- read_parquet(here("data", "analysis", "documents_combined.parquet")) |> glimpse()
ea_pages <- read_parquet(here("data", "processed", "ea", "pages.parquet")) |> glimpse()
projects_timeline <- read_parquet(here("data", "analysis", "projects_timeline.parquet")) |> glimpse()


# --------------------------
# TIMELINE
# --------------------------
# Need to update
timeline %>%
  filter(project_energy_type == "Clean") |> 
  count(project_timeline_doc_source) |> 
  print()

timeline %>%
  filter(project_energy_type == "Clean") |> 
  count(project_timeline_doc_source) |> 
  print()


# --------------------------
# DOCUMENTS
# --------------------------


document_sample <- 
  documents |> 
  left_join(projects) |> 
  filter(main_document == "YES") |> 
  slice_sample( n = 100) |> 
  select(project_title,project_id, document_id, document_type, file_name, main_document) |> 
  glimpse()


# save to google sheets so team can view
sheet_write(
  data = document_sample,
  ss = "https://docs.google.com/spreadsheets/d/1Khd_aLhw7ArxO9zMl1C6R7HlFPRr8Nl_S7pIxPvM1tc/edit?usp=sharing",
  sheet = "document_sample"
)


  documents |> 
  left_join(projects) |> 
  filter(project_id == "890d5651-4f06-9c9b-3567-7e90e7adf7c9") |> 
  glimpse()

documents |> glimpse()

#
# check to see if prepared_by column in documents provides any new information -- not really 
# ------------------------------

documents |> 
  filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  glimpse()


doc_title_sample <- 
  documents |> 
  slice_head(n = 100) |> 
  select(project_id, document_title, file_name, main_document, document_type_category) |> 
  glimpse()

# save to google sheets so team can view
sheet_write(
  data = doc_title_sample,
  ss = "https://docs.google.com/spreadsheets/d/1Khd_aLhw7ArxO9zMl1C6R7HlFPRr8Nl_S7pIxPvM1tc/edit?usp=sharing",
  sheet = "doc_title_sample"
)

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



# --------------------------
# TEST CASES
# --------------------------

#
# test 1: 39d059404366825eda2052316c81c34a: Chevron USA Inc. Three Lost Hills Well Abandonments
# ------------------------------
test <- 
  documents |> 
  filter(project_id == "39d059404366825eda2052316c81c34a") |> 
  glimpse()

# save to google sheets so team can view
sheet_write(
  data = test,
  ss = "https://docs.google.com/spreadsheets/d/1Khd_aLhw7ArxO9zMl1C6R7HlFPRr8Nl_S7pIxPvM1tc/edit?usp=sharing",
  sheet = "test"
)

projects_timeline |> 
  glimpse()
   filter(project_id == "39d059404366825eda2052316c81c34a") |> 
  glimpse()

#
# test 2: 6e9fc6608e30977c74305d2a98628a13: Coldfoot Cell tower
# ------------------------------

test2 <- 
  documents |> 
  filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  glimpse()

# save to google sheets so team can view
sheet_write(
  data = test2,
  ss = "https://docs.google.com/spreadsheets/d/1Khd_aLhw7ArxO9zMl1C6R7HlFPRr8Nl_S7pIxPvM1tc/edit?usp=sharing",
  sheet = "test2"
)

projects_timeline |> 
  filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  pull(project_dates) |> 
  #select(project_dates, project_date_contexts, project_year) |> 
  print()
