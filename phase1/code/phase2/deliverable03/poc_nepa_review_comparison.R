# --------------------------
# PHASE 2, DELIVERABLE 3: NEPA REVIEW PROCESS APPLICATION — POC
# --------------------------
# Goal: Sketch a simple, feasible comparison of NEPA reviews
#       for fossil fuel vs. clean energy projects.
# Approach: Use existing project-level tables to compare
#           (1) CE/EA/EIS rates
#           (2) linear vs. non-linear mix
#           (3) timeline durations (where available)
# This is a feasibility check, not a final pipeline.

library(here)
library(arrow)
library(tidyverse)

# --------------------------
# LOAD DATA
# --------------------------
reviews <- read_parquet(here("phase1", "data", "analysis", "projects_reviews.parquet"))
timelines <- read_parquet(here("phase1", "data", "analysis", "projects_timeline.parquet"))

# --------------------------
# BASIC FILTERS
# --------------------------
# Keep only clean + fossil for the POC
reviews <- reviews %>%
  filter(project_energy_type %in% c("Clean", "Fossil"))

timelines <- timelines %>%
  filter(project_energy_type %in% c("Clean", "Fossil"))

# Optional exclusions (uncomment if you want tighter scope)
# reviews <- reviews %>%
#   filter(!project_utilities_to_exclude,
#          !project_military_to_exclude,
#          !project_nuclear_waste_to_exclude)
# timelines <- timelines %>%
#   filter(!project_utilities_to_exclude,
#          !project_military_to_exclude,
#          !project_nuclear_waste_to_exclude)

# --------------------------
# LINEAR VS NON-LINEAR HEURISTIC
# --------------------------
# Very simple, purely keyword-based on project_type + sector.
# This should be refined later, but it's enough for a POC.

linear_keywords <- c(
  "transmission", "pipeline", "line", "corridor",
  "right-of-way", "row", "rail", "road", "highway",
  "trail", "utility line", "cable", "intertie"
)

is_linear <- function(type, sector) {
  txt <- str_to_lower(paste(type, sector, sep = " "))
  any(str_detect(txt, str_c(linear_keywords, collapse = "|")))
}

reviews <- reviews %>%
  mutate(project_is_linear = map2_lgl(project_type, project_sector, is_linear))

timelines <- timelines %>%
  mutate(project_is_linear = map2_lgl(project_type, project_sector, is_linear))

# --------------------------
# (1) CE / EA / EIS RATES
# --------------------------
# Use project_review_type from the reviews table.
# This field is already derived in Phase 1.

cat("\n=== REVIEW TYPE RATES (CE/EA/EIS) ===\n")
review_rates <- reviews %>%
  filter(!is.na(project_review_type)) %>%
  count(project_energy_type, project_review_type) %>%
  group_by(project_energy_type) %>%
  mutate(pct = n / sum(n)) %>%
  arrange(project_energy_type, desc(n))

print(review_rates, n = 50)

# --------------------------
# (2) LINEAR VS NON-LINEAR MIX
# --------------------------
cat("\n=== LINEAR VS NON-LINEAR MIX ===\n")
linear_mix <- reviews %>%
  count(project_energy_type, project_is_linear) %>%
  group_by(project_energy_type) %>%
  mutate(pct = n / sum(n))

print(linear_mix, n = 50)

# --------------------------
# (3) TIMELINE DURATION COMPARISON
# --------------------------
# This is only feasible for projects with durations.

cat("\n=== TIMELINE DURATION (DAYS) ===\n")
duration_summary <- timelines %>%
  filter(!is.na(project_duration_days), project_duration_days > 0) %>%
  group_by(project_energy_type) %>%
  summarise(
    n = n(),
    median_days = median(project_duration_days, na.rm = TRUE),
    mean_days = mean(project_duration_days, na.rm = TRUE),
    p25 = quantile(project_duration_days, 0.25, na.rm = TRUE),
    p75 = quantile(project_duration_days, 0.75, na.rm = TRUE)
  )

print(duration_summary)

# --------------------------
# (4) OPTIONAL: REVIEW TYPE x LINEAR
# --------------------------
cat("\n=== REVIEW TYPE BY LINEAR STATUS ===\n")
review_by_linear <- reviews %>%
  filter(!is.na(project_review_type)) %>%
  count(project_energy_type, project_is_linear, project_review_type) %>%
  group_by(project_energy_type, project_is_linear) %>%
  mutate(pct = n / sum(n))

print(review_by_linear, n = 50)

cat("\nPOC complete. Next steps: refine linear classification,\nvalidate review types, and integrate geography + timelines.\n")
