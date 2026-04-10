# --------------------------
# PROJECT REVIEW: DELIVERABLE 4 EXTRAS
# --------------------------
# Exploratory analyses not referenced in reports/deliverable04.qmd.

source(here::here("phase1", "code", "deliverable4", "00_setup.R"))

# --------------------------
# PROCESS (MATCH 01_geography.R)
# --------------------------

multi_state_data <- 
  clean_energy |> 
  filter(project_multi_state)

multi_department_data <- 
  clean_energy |> 
  filter(project_multi_department)

# --------------------------
# EXPLORATORY
# --------------------------

clean_energy |> 
  filter(project_multi_state) |> 
  select(project_multi_state) |> 
  glimpse() 

# --------------------------
# MULTI-DEPARTMENT EXPLORATORY
# --------------------------

clean_energy |> 
  select(project_sponsor, lead_agency) |> 
  slice_sample(n = 5) |> 
  print()

multi_department_data |>
  select(project_department, lead_agency, project_sponsor, project_multi_department) |>
  print(n = 50)

gmulti_agency_data <- 
  clean_energy |> 
  filter(dataset_source != "CE") |> 
  select(lead_agency, project_sponsor) |> 
  slice_sample(n = 100) |> 
  print(n = 100)

# Duplicate crosstab for inspection
department_links <- create_crosstab(
  multi_department_data,
  "lead_agency",
  keep_cols = c("project_title", "project_type")
) |>
  print()

# --------------------------
# LEAD AGENCY / PROJECT SPONSOR OVERLAP
# --------------------------

parse_agencies <- function(x) {
 if (is.na(x) || x == "") return(character(0))
 if (str_detect(x, "^\\[")) {
   tryCatch(fromJSON(x), error = function(e) x)
 } else {
   x
 }
}

agency_sponsor_overlap <- clean_energy |>
  filter(!is.na(lead_agency) & !is.na(project_sponsor)) |>
  mutate(
    agencies = map(lead_agency, parse_agencies),
    has_overlap = map2_lgl(agencies, project_sponsor, ~ {
      if (length(.x) == 0 || is.na(.y) || .y == "") return(FALSE)
      any(sapply(.x, function(agency) {
        str_detect(.y, regex(agency, ignore_case = TRUE))
      }))
    })
  )

cat("\n=== Lead Agency / Project Sponsor Overlap Analysis ===\n")
cat("Projects with lead_agency name appearing in project_sponsor:\n")
cat("  Count:", sum(agency_sponsor_overlap$has_overlap), "\n")
cat("  Percent:", round(mean(agency_sponsor_overlap$has_overlap) * 100, 2), "%\n")

overlap_examples <- agency_sponsor_overlap |>
  filter(has_overlap) |>
  select(lead_agency, project_sponsor, process_type) |>
  slice_head(n = 20) |> 
  print()

cat("\nExamples of projects with overlap:\n")
print(overlap_examples, n = 20)

# --------------------------
# BUREAU OF LAND MANAGEMENT QUESTION (PLACEHOLDER)
# --------------------------

projects %>%
  filter(str_detect(project_type, regex("geothermal", ignore_case = TRUE))) |> 
  filter(str_detect(lead_agency, regex("land management", ignore_case = TRUE))) |> 
  select(dataset_source, project_department, project_sponsor, lead_agency, project_energy_type) |> 
  slice_sample(n = 20) |> 
  print(n = 20 )
  glimpse()

cat("\n=== Deliverable 4 Extras Complete ===\n")
