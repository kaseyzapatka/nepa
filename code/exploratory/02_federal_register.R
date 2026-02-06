# --------------------------
# EXPLORATORY: FEDERAL REGISTER
# --------------------------
# Exploratory analysis of public comments


# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable01", "00_setup.R"))


# --------------------------
# LOAD SPECIFIC DATA
# --------------------------

register_path <- here("data", "analysis", "noi_federal_register.parquet")
register <- read_parquet(register_path)


# --------------------------
# ANALYSIS
# --------------------------

register |> 
  filter(!is.na(noi_publication_date)) |> 
  #select(noi_publication_date:noi_document_number) |> 
  glimpse()

projects |> 
  filter(project_id == "8fa2e4a0ce54588261fa6730fcf58c03") |> 
  glimpse()
  filter(str_detect(project_title, regex("Coastal Virginia Offshore Wind Commercial Project", ignore_case = TRUE))) |> 
  glimpse()

clean_energy |> 
  select(project_id, project_title, project_state) |> 
  left_join(register) |> 
  filter(!is.na(noi_publication_date)) |> 
  glimpse()
