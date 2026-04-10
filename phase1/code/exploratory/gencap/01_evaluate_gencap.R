# Evaluate generation capacity extraction quality
# Load EA or EIS separately without needing the merge script


# --------------------------
# SETUP
# --------------------------

# remove lists 
rm(list = ls())

# source 
source(here::here("phase1", "code", "00_setup.R"))

# libraries 
library(dplyr)
library(stringr)
library(arrow)
library(googlesheets4)

# --------------------------
# LOAD DATA
# --------------------------
gencap <- read_parquet(here::here("phase1", "data", "analysis", "projects_gencap.parquet")) %>% 
  select(project_id, project_title, process_type, project_is_transmission_broad:project_gencap_candidates_json) |>  
  glimpse()


llm <- read_parquet(here::here("phase1", "data", "analysis", "projects_gencap.parquet")) %>% 
  filter(llm_merge_decision == "llm_override_regex") |> 
  select(project_id, project_title, process_type, project_is_transmission_broad:project_gencap_candidates_json, 
    project_gencap_llm_triggered:llm_merge_decision) |>  
  glimpse()

llm |> 
  #count(project_gencap_llm_selection_logic) |> 
  count(llm_merge_decision) |> 
  print()


# --------------------------
# COUNT VALIDATIONS
# --------------------------
# validate total count
gencap |> 
  count(process_type) |> 
  print()

# validate energy and power counts
gencap |> 
  filter(!is.na(project_gencap_value) | !is.na(project_gencap_energy_value)) |> 
  count(process_type) |> 
  print()

# validate power counts
gencap |> 
  filter(!is.na(project_gencap_value)) |> 
  count(process_type) |> 
  print()


# --------------------------
# RANDOM SAMPLE VALIDATION
# --------------------------
sample <- 
  gencap |> 
  filter(!is.na(project_gencap_value)) |> 
  select(project_id) |> 
  slice_sample(n = 1) |> 
  print()

gencap |> 
  filter(project_id %in% sample) |> 
  select(project_id, project_title, process_type, project_gencap_value, project_gencap_candidate_count, project_gencap_energy_candidate_count, project_gencap_source, project_gencap_context ) |> 
  glimpse()

gencap |> 
  filter(project_id %in% sample) |> 
  pull(project_gencap_context ) |> 
  print()


gencap |> 
  #select(project_gencap_energy_candidate_count) |> 
  #count(project_gencap_energy_candidate_count)
  count(project_gencap_candidate_count) |> 
  filter(project_gencap_candidate_count > 5) |> 
  summarize(total = sum(n))
  print(n = 100)


gencap |> 
  filter(!is.na(project_gencap_candidate_count)) |> 
  unnest(project_gencap_candidates_json ) |> 
  glimpse()



# --------------------------
# MULTIPLE CANDIDATE VALIDATION
# --------------------------
sample <- 
  gencap |> 
  filter(!is.na(project_gencap_value)) |> 
  filter(project_gencap_candidate_count >= 2) |> 
  select(project_id) |> 
  slice_sample(n = 1) |> 
  print()

gencap |> 
  filter(project_id %in% sample) |> 
  select(project_id, project_title, process_type, project_gencap_value, project_gencap_candidate_count, project_gencap_energy_candidate_count, project_gencap_source, project_gencap_context,project_gencap_candidates_json) |> 
  unnest(project_gencap_candidates_json ) |> 
  glimpse()

gencap |> 
  filter(project_id %in% sample) |> 
  unnest(project_gencap_candidates_json ) |> 
  pull(context ) |> 
  print()



# --------------------------
# MULTIPLE CANDIDATE VALIDATION -- LLM
# --------------------------
sample <- 
  llm |> 
  filter(llm_merge_decision == "llm_override_regex") |> 
  select(project_id) |> 
  slice_sample(n = 1) |> 
  print()

llm |> 
  filter(project_id %in% sample) |> 
  select(project_id, project_title, process_type, project_gencap_value, project_gencap_final_value, 
    project_gencap_candidate_count, project_gencap_energy_candidate_count, project_gencap_context, 
    project_gencap_final_quote, project_gencap_candidates_json, llm_merge_decision, project_gencap_llm_reasoning) |> 
  unnest(project_gencap_candidates_json ) |> 
  glimpse()

llm |> 
  filter(project_id %in% sample) |> 
  #pull(project_gencap_llm_reasoning,project_gencap_final_quote ) |> 
  pull(project_gencap_final_quote ) |> 
  print()

# 
# One off check
# ------------------------------
llm |> 
  filter(project_id  == "3c5c295fe8f72f86dbfda31b8a7b4348") |> 
  select(project_id, project_title, process_type, project_gencap_value, project_gencap_candidate_count, project_gencap_energy_candidate_count, project_gencap_source, project_gencap_context,project_gencap_candidates_json, project_gencap_final_value) |> 
  unnest(project_gencap_candidates_json ) |> 
  glimpse()

llm |> 
  filter(project_id  == "3c5c295fe8f72f86dbfda31b8a7b4348") |> 
  unnest(project_gencap_candidates_json ) |> 
  pull(context ) |> 
  print()

#sample # 3c5c295fe8f72f86dbfda31b8a7b4348 -- good example to check 




# --------------------------
# CREATE SAMPLE FOR CANDIDATE VALIDATION TO GOOGLE SHEET
# --------------------------

set.seed(123)
sample <- 
  llm |> 
  filter(llm_merge_decision == "llm_override_regex") |> 
  select(project_id) |> 
  slice_sample(n = 10) |> 
  pull()

validation <- 
  llm |> 
  filter(project_id %in% sample) |>
  select(project_id, project_title, process_type, project_gencap_value, project_gencap_final_value, 
    project_gencap_candidate_count, project_gencap_energy_candidate_count, project_gencap_context, 
    project_gencap_final_quote, project_gencap_candidates_json, llm_merge_decision, project_gencap_llm_reasoning) |> 
  unnest(project_gencap_candidates_json ) |> 
  glimpse()

# Write to google sheets for review
sheet_write(
  data = validation,
  ss = "https://docs.google.com/spreadsheets/d/15vqxrFe72cIKBVTviT2nn1LdcUvoC6siGnw4-7SUVWI/edit?usp=sharing",
  sheet = "validation"
)


gencap |> 
  filter(project_title == "Granite Reliable Power Wind Park") |> 
  unnest(project_gencap_candidates_json) |> 
  glimpse()


# Granite Reliable Power Wind Park - should be 99 bc need to multiple 33 wind generators by 3.0KW
llm |> 
  filter(project_title == "Granite Reliable Power Wind Park") |> 
  unnest(project_gencap_candidates_json) |> 
  glimpse()

# Kotzebue Wind Installation Project - Should be 425kW bc needs to sum all turbine compacities
llm |> 
  filter(project_title == "Kotzebue Wind Installation Project") |> 
  unnest(project_gencap_candidates_json) |> 
  glimpse()
