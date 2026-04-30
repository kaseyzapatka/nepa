# --------------------------
# PHASE 2, DELIVERABLE 1: NEPA TRIGGERED — PROOF OF CONCEPT
# --------------------------
# Goal: Can we identify WHY NEPA was triggered for each project?
# Approach: Two layers —
#   (1) Metadata heuristic: lead agency → likely trigger type
#   (2) Regex on project_description: explicit trigger language
# This is a feasibility check, not a final pipeline.

library(here)
library(arrow)
library(tidyverse)

# --------------------------
# LOAD DATA
# --------------------------

projects <- read_parquet(here("data", "analysis", "projects_combined.parquet"))
clean_energy <- projects %>% filter(project_energy_type == "Clean")

cat("Clean energy projects:", nrow(clean_energy), "\n")
cat("Have description:", sum(!is.na(clean_energy$project_description) &
                              clean_energy$project_description != ""), "\n\n")

# ==========================================================
# LAYER 1: METADATA HEURISTIC (agency → likely trigger)
# ==========================================================
# The lead agency often implies the federal nexus. This won't be
# perfect but gives a baseline for the ~60-70% of "obvious" cases.

agency_trigger_map <- tribble(
  ~pattern,                          ~trigger_heuristic,
  "Bureau of Land Management",       "Federal Land",
  "Forest Service",                  "Federal Land",
  "Bureau of Reclamation",           "Federal Land",
  "Bureau of Indian Affairs",        "Federal Land / Trust",
  "National Park Service",           "Federal Land",
  "Fish and Wildlife Service",       "Federal Permit",
  "Bureau of Ocean Energy",          "Federal Permit / Lease",
  "Corps of Engineers",              "Federal Permit",
  "Federal Energy Regulatory",       "Federal Permit",
  "Nuclear Regulatory Commission",   "Federal Permit",
  "Department of Energy",            "Federal Funding / Action",
  "Bonneville Power",                "Federal Action",
  "Western Area Power",              "Federal Action",
  "Southwestern Power",              "Federal Action",
  "Southeastern Power",              "Federal Action",
  "Tennessee Valley Authority",      "Federal Action",
  "Rural Utilities Service",         "Federal Funding"
)

# Apply the heuristic: match on lead_agency text
assign_trigger_heuristic <- function(agency, map = agency_trigger_map) {
  if (is.na(agency) || agency == "") return(NA_character_)
  for (i in seq_len(nrow(map))) {
    if (str_detect(agency, fixed(map$pattern[i]))) {
      return(map$trigger_heuristic[i])
    }
  }
  NA_character_
}

clean_energy <- clean_energy %>%
  mutate(trigger_heuristic = map_chr(lead_agency, assign_trigger_heuristic))

cat("=== LAYER 1: Agency Heuristic ===\n")
cat("Coverage:", sum(!is.na(clean_energy$trigger_heuristic)), "/",
    nrow(clean_energy),
    sprintf("(%.1f%%)\n\n", mean(!is.na(clean_energy$trigger_heuristic)) * 100))

clean_energy %>%
  count(trigger_heuristic, sort = TRUE) %>%
  mutate(pct = sprintf("%.1f%%", n / sum(n) * 100)) %>%
  print(n = 20)

# ==========================================================
# LAYER 2: REGEX ON PROJECT DESCRIPTION
# ==========================================================
# Search for explicit trigger language in the project description.
# These patterns are intentionally broad for the POC.

trigger_patterns <- tribble(
  ~trigger_type,     ~regex,
  "Federal Land",    "(?i)federal land|public land|blm.?managed|national forest|federal property|government.?owned land|federal.?owned|agency.?administered land",
  "Federal Funding", "(?i)federal fund|federal grant|doe fund|federal financ|loan guarantee|federal loan|recovery act|arra|grant from|funded by",
  "Federal Permit",  "(?i)right.?of.?way|section 404|wetland permit|eagle take|incidental take|special.?use permit|easement|federal permit|federal authorization|federal license",
  "Federal Lease",   "(?i)federal lease|blm lease|offshore lease|wind energy lease|solar lease|geothermal lease",
  "Federal Trust",   "(?i)tribal land|trust land|indian land|reservation|allotment"
)

# Apply all patterns to project_description
for (i in seq_len(nrow(trigger_patterns))) {
  col_name <- paste0("trigger_regex_", str_replace_all(
    tolower(trigger_patterns$trigger_type[i]), " ", "_"))
  clean_energy <- clean_energy %>%
    mutate(!!col_name := str_detect(project_description,
                                     trigger_patterns$regex[i]) %in% TRUE)
}

# Combine regex flags into a single summary column
regex_cols <- names(clean_energy)[str_starts(names(clean_energy), "trigger_regex_")]

clean_energy <- clean_energy %>%
  mutate(
    trigger_regex_any = rowSums(across(all_of(regex_cols))) > 0,
    trigger_regex_types = pmap_chr(
      select(., all_of(regex_cols)),
      function(...) {
        vals <- c(...)
        types <- str_remove(regex_cols[vals], "trigger_regex_")
        if (length(types) == 0) return(NA_character_)
        paste(str_to_title(str_replace_all(types, "_", " ")), collapse = " + ")
      }
    )
  )

cat("\n=== LAYER 2: Regex on Description ===\n")
cat("Any trigger found:", sum(clean_energy$trigger_regex_any), "/",
    nrow(clean_energy),
    sprintf("(%.1f%%)\n\n", mean(clean_energy$trigger_regex_any) * 100))

cat("By trigger type:\n")
clean_energy %>%
  summarise(across(all_of(regex_cols), sum)) %>%
  pivot_longer(everything(), names_to = "trigger", values_to = "n") %>%
  mutate(
    trigger = str_remove(trigger, "trigger_regex_"),
    trigger = str_to_title(str_replace_all(trigger, "_", " ")),
    pct = sprintf("%.1f%%", n / nrow(clean_energy) * 100)
  ) %>%
  arrange(desc(n)) %>%
  print()

# ==========================================================
# COMBINED: How much do we cover with both layers?
# ==========================================================

clean_energy <- clean_energy %>%
  mutate(
    trigger_any = !is.na(trigger_heuristic) | trigger_regex_any,
    trigger_combined = case_when(
      trigger_regex_any ~ trigger_regex_types,           # prefer regex (more specific)
      !is.na(trigger_heuristic) ~ trigger_heuristic,     # fallback to heuristic
      TRUE ~ NA_character_
    )
  )

cat("\n=== COMBINED COVERAGE ===\n")
cat("Heuristic only:", sum(!is.na(clean_energy$trigger_heuristic) & !clean_energy$trigger_regex_any), "\n")
cat("Regex only:    ", sum(is.na(clean_energy$trigger_heuristic) & clean_energy$trigger_regex_any), "\n")
cat("Both:          ", sum(!is.na(clean_energy$trigger_heuristic) & clean_energy$trigger_regex_any), "\n")
cat("Neither:       ", sum(is.na(clean_energy$trigger_heuristic) & !clean_energy$trigger_regex_any), "\n")
cat("Total covered: ", sum(clean_energy$trigger_any), "/", nrow(clean_energy),
    sprintf("(%.1f%%)\n", mean(clean_energy$trigger_any) * 100))

# ==========================================================
# SPOT CHECK: Show examples by trigger type
# ==========================================================

cat("\n=== SPOT CHECK: Examples with regex trigger ===\n")
clean_energy %>%
  filter(trigger_regex_any) %>%
  slice_sample(n = 10) %>%
  #slice_sample(n = 1) %>%
  #  select(trigger_heuristic, project_description) |> 
  #slice_sample(n = 1) |> 
  #pull() |> 
  #print()
  select(project_id, lead_agency, project_type, trigger_heuristic,
         trigger_regex_types, project_description) %>%
  
  mutate(project_description = str_trunc(project_description, 150)) %>%
  as.data.frame() %>%
  print()

cat("\n=== SPOT CHECK: Projects with NO trigger signal ===\n")
clean_energy %>%
  filter(!trigger_any) %>%
  slice_sample(n = min(10, sum(!clean_energy$trigger_any))) %>%
  select(project_id, lead_agency, project_type, project_description) %>%
  mutate(project_description = str_trunc(project_description, 150)) %>%
  as.data.frame() %>%
  print()

# ==========================================================
# VERDICT
# ==========================================================

cat("\n")
cat("============================\n")
cat("  POC VERDICT\n")
cat("============================\n")
cat(sprintf("Agency heuristic alone:    %.1f%% coverage\n",
            mean(!is.na(clean_energy$trigger_heuristic)) * 100))
cat(sprintf("Description regex alone:   %.1f%% coverage\n",
            mean(clean_energy$trigger_regex_any) * 100))
cat(sprintf("Combined (either layer):   %.1f%% coverage\n",
            mean(clean_energy$trigger_any) * 100))
cat(sprintf("Gap (no signal):           %.1f%% (%d projects)\n",
            mean(!clean_energy$trigger_any) * 100,
            sum(!clean_energy$trigger_any)))
cat("\nThe gap projects will need LLM classification on document text.\n")
cat("If the gap is <30-40%%, this deliverable is feasible.\n")
cat("============================\n")


# view sample project description
clean_energy |> 
  filter(project_description != "") |> 
  select(project_description) |> 
  slice_sample(n = 1) |> 
  pull() |> 
  print()

