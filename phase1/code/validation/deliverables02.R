# =============================================================================
# Reviews Exploratory Data Analysis & Validation Sheets
# =============================================================================
# Analysis of programmatic/tiered reviews + Google Sheets for client validation

library(tidyverse)
library(arrow)
library(here)
library(scales)
library(googlesheets4)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

reviews <- read_parquet(here("phase1", "data", "analysis", "projects_reviews.parquet"))

cat("Loaded", nrow(reviews), "projects\n")

# -----------------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------------

review_type_summary <- reviews |>

count(project_review_type, name = "n") |>
  mutate(
    pct = n / sum(n),
    pct_label = percent(pct, accuracy = 0.1)
  ) |>
  arrange(desc(n))

cat("\n=== Review Type Distribution ===\n")
print(review_type_summary)

confidence_summary <- reviews |>
  count(project_review_confidence, name = "n") |>
  mutate(pct = percent(n / sum(n), accuracy = 0.1))

cat("\n=== Confidence Distribution ===\n")
print(confidence_summary)

source_summary <- reviews |>
  count(project_review_source, name = "n") |>
  mutate(pct = percent(n / sum(n), accuracy = 0.1))

cat("\n=== Detection Source Distribution ===\n")
print(source_summary)

# -----------------------------------------------------------------------------
# Cross-tabulations
# -----------------------------------------------------------------------------

review_by_process <- reviews |>
  count(dataset_source, project_review_type) |>
  pivot_wider(names_from = project_review_type, values_from = n, values_fill = 0)

cat("\n=== Review Type by Process (EA/EIS) ===\n")
print(review_by_process)

prog_tiered_by_agency <- reviews |>
  filter(project_review_type %in% c("programmatic", "tiered")) |>
  count(lead_agency, project_review_type) |>
  pivot_wider(names_from = project_review_type, values_from = n, values_fill = 0) |>
  mutate(total = coalesce(programmatic, 0L) + coalesce(tiered, 0L)) |>
  arrange(desc(total))

cat("\n=== Programmatic/Tiered by Agency ===\n")
print(prog_tiered_by_agency)

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

fig_review_type <- reviews |>
  count(project_review_type) |>
  mutate(project_review_type = fct_reorder(project_review_type, n)) |>
  ggplot(aes(x = project_review_type, y = n, fill = project_review_type)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = comma(n)), hjust = -0.1, size = 3.5) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  scale_fill_manual(values = c(
    "standard" = "gray70",
    "programmatic" = "#2171b5",
    "tiered" = "#6baed6"
  )) +
  labs(
    title = "Review Type Distribution",
    subtitle = "Clean energy EA/EIS projects (n = 1,416)",
    x = NULL,
    y = "Number of projects"
  ) +
  theme_minimal() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  )

print(fig_review_type)

fig_review_by_process <- reviews |>
  count(dataset_source, project_review_type) |>
  ggplot(aes(x = dataset_source, y = n, fill = project_review_type)) +
  geom_col(position = "dodge") +
  geom_text(
    aes(label = n),
    position = position_dodge(width = 0.9),
    vjust = -0.3,
    size = 3
  ) +
  scale_fill_manual(
    values = c(
      "standard" = "gray70",
      "programmatic" = "#2171b5",
      "tiered" = "#6baed6"
    ),
    name = "Review Type"
  ) +
  labs(
    title = "Review Type by NEPA Process",
    subtitle = "Clean energy projects",
    x = "Process Type",
    y = "Number of projects"
  ) +
  theme_minimal()

print(fig_review_by_process)

# -----------------------------------------------------------------------------
# Validation datasets (separate for programmatic and tiered)
# -----------------------------------------------------------------------------

# Programmatic reviews - columns specific to programmatic
programmatic_validation <- reviews |>
  filter(project_review_type == "programmatic") |>
  select(
    project_id,
    project_title,
    confidence = project_review_confidence,
    detection_source = project_review_source,
    evidence_text = project_review_match_text,
    lead_agency,
    process_type = dataset_source
  ) |>
  mutate(
    correct = NA_character_,
    notes = NA_character_
  ) |>
  arrange(desc(confidence), project_title)

# Tiered reviews - includes tiers_from columns
tiered_validation <- reviews |>
  filter(project_review_type == "tiered") |>
  select(
    project_id,
    project_title,
    confidence = project_review_confidence,
    detection_source = project_review_source,
    tiers_from = project_review_tiers_from,
    evidence_text = project_review_match_text,
    evidence_context = project_review_tiers_from_context,
    lead_agency,
    process_type = dataset_source
  ) |>
  mutate(
    correct = NA_character_,
    notes = NA_character_
  ) |>
  arrange(desc(confidence), project_title)

cat("\n=== Validation Data Prepared ===\n")
cat("Programmatic:", nrow(programmatic_validation), "\n")
cat("Tiered:", nrow(tiered_validation), "\n")

# -----------------------------------------------------------------------------
# Write to Google Sheet
# -----------------------------------------------------------------------------

SHEET_URL <- "https://docs.google.com/spreadsheets/d/1d25Arj2IFR3SLcgv8B6tPdLbs2cDZtC_IjMhZUcueBA/edit?usp=sharing"

sheet_write(programmatic_validation, ss = SHEET_URL, sheet = "Programmatic")
cat("Wrote Programmatic sheet\n")

sheet_write(tiered_validation, ss = SHEET_URL, sheet = "Tiered")
cat("Wrote Tiered sheet\n")

# -----------------------------------------------------------------------------
# Local CSV backup
# -----------------------------------------------------------------------------

output_dir <- here("phase1", "output", "deliverable2")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

write_csv(programmatic_validation, here(output_dir, "programmatic_validation.csv"))
write_csv(tiered_validation, here(output_dir, "tiered_validation.csv"))

cat("\nLocal backups saved to:", output_dir, "\n")
cat("Done!\n")
