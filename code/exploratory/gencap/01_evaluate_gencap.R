# Evaluate generation capacity extraction quality
# Load EA or EIS separately without needing the merge script

rm(list = ls())

source(here::here("code", "00_setup.R"))

library(dplyr)
library(stringr)
library(arrow)
library(googlesheets4)

# ---- Load EA data ----

# Regex results (filtered to EA)
regex_ea <- read_parquet(here::here("data", "analysis", "projects_gencap.parquet")) %>%
  filter(project_energy_type == "Clean", dataset_source == "EA")

# LLM results
llm_ea <- read_parquet(here::here("data", "analysis", "gencap_ea_llm.parquet"))

# Join them
ea <- regex_ea %>%
  left_join(
    llm_ea %>% select(
      project_id,
      llm_value = capacity_value,
      llm_unit = capacity_unit,
      llm_confidence = confidence,
      llm_quote = source_quote,
      llm_method = extraction_method,
      llm_candidates = candidates_found
    ),
    by = "project_id"
  )

# ---- Summary ----

cat("========== EA Summary ==========\n")
cat(paste0("Total projects: ", nrow(ea), "\n"))
cat(paste0("Regex extracted: ", sum(!is.na(ea$project_gencap_value)), "\n"))
cat(paste0("LLM extracted: ", sum(!is.na(ea$llm_value)), "\n"))

cat("\nLLM extraction method:\n")
ea %>% filter(!is.na(llm_value)) %>% count(llm_method) %>% print()

cat("\nLLM confidence:\n")
ea %>% filter(!is.na(llm_value)) %>% count(llm_confidence) %>% print()

# ---- Sample for review ----

# Projects with LLM capacity - compare value and quote
ea_with_llm <- ea %>%
  filter(!is.na(llm_value)) %>%
  select(
    project_id,
    project_title,
    lead_agency,
    # Regex
    regex_value = project_gencap_value,
    regex_unit = project_gencap_unit,
    regex_context = project_gencap_context,
    # LLM
    llm_value,
    llm_unit,
    llm_method,
    llm_quote,
    llm_candidates
  )

cat(paste0("\n\nEA projects with LLM extraction: ", nrow(ea_with_llm), "\n"))

# View sample
ea_with_llm %>%
  select(project_title, project_id, regex_value, regex_unit, llm_value, llm_unit, llm_method) %>%
  filter(project_id == "d9c3d975f3e8c38c549f8182bec4181b") |> 
  glimpse(n = 30)

# View sample
ea_with_llm %>%
  select(project_title, project_id, regex_value, regex_unit, llm_value, llm_unit, llm_method) %>%
  filter(project_id == "d9c3d975f3e8c38c549f8182bec4181b") |> 
  glimpse(n = 30)

# ---- View a project in detail ----

view_project <- function(pid) {
  proj <- ea %>% filter(project_id == pid)
  if (nrow(proj) == 0) { cat("Not found\n"); return(invisible(NULL)) }

  cat("\n========================================\n")
  cat(paste0("Title: ", proj$project_title, "\n"))
  cat(paste0("Agency: ", proj$lead_agency, "\n"))
  cat("\n--- REGEX ---\n")
  cat(paste0("Value: ", proj$project_gencap_value, " ", proj$project_gencap_unit, "\n"))
  cat(paste0("Context: ", proj$project_gencap_context, "\n"))
  cat("\n--- LLM ---\n")
  cat(paste0("Value: ", proj$llm_value, " ", proj$llm_unit, "\n"))
  cat(paste0("Method: ", proj$llm_method, "\n"))
  cat(paste0("Quote: ", proj$llm_quote, "\n"))
  cat(paste0("Candidates: ", proj$llm_candidates, "\n"))

  invisible(proj)
}
view_project("d9c3d975f3e8c38c549f8182bec4181b")

# Example: view_project("some-id")

# ---- Export for manual review ----

# Uncomment to export:
# readr::write_csv(ea_with_llm, here::here("output", "deliverable3", "gencap_eval_ea.csv"))

# Write to google sheets for review
sheet_write(
  data = ea_with_llm,
  ss = "https://docs.google.com/spreadsheets/d/15vqxrFe72cIKBVTviT2nn1LdcUvoC6siGnw4-7SUVWI/edit?usp=sharing",
  sheet = "ea"
)
