# --------------------------
# DELIVERABLE 1: NEPA TRIGGERED — Analysis and Figures
# --------------------------
# Produces six figures and supporting CSVs from the trigger classification output.
#
# Primary figure: trigger type × process type (CE/EA/EIS) — 100% stacked bar
# Supporting figures: agency heatmap, technology breakdown, state choropleth,
#                     multi-label combination bar, timeline placeholder
# Table: representative evidence text excerpts (quotable for report)
#
# Input:
#   data/analysis/nepa_trigger/projects_nepa_trigger.parquet
#   data/analysis/projects_combined.parquet
#
# Output (all in output/deliverable01/):
#   fig1_trigger_by_process.png
#   fig2_agency_trigger_heatmap.png
#   fig3_trigger_combinations.png        (multi-label; on request)
#   fig4_trigger_by_technology.png
#   fig5_state_choropleth.png
#   trigger_evidence_excerpts.csv
#   trigger_source_distribution.csv      (pipeline diagnostics)
#   trigger_rule_distribution.csv        (top rules by volume)
#
# Usage:
#   Rscript phase2/code/deliverable01/02_analyze_triggers.R

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(forcats)
  library(purrr)
  library(scales)
  library(usmap)   # install.packages("usmap") if missing
})

# --------------------------
# PATHS
# --------------------------

BASE_DIR   <- here::here()  # repo root; install.packages("here") if missing
OUTPUT_DIR <- file.path(BASE_DIR, "output", "deliverable01")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

TRIGGERS_PATH <- file.path(BASE_DIR, "data", "analysis", "nepa_trigger",
                            "projects_nepa_trigger.parquet")
PROJECTS_PATH <- file.path(BASE_DIR, "data", "analysis", "projects_combined.parquet")

# --------------------------
# LOAD AND PREPARE
# --------------------------

triggers <- read_parquet(TRIGGERS_PATH)
projects <- read_parquet(PROJECTS_PATH)

df <- left_join(triggers, projects, by = "project_id") |>
  filter(project_energy_type == "Clean")  # enforce scope; n should be 20,725

# nepa_trigger_secondary is an Arrow list<string> column.
# R reads it as a list-column. purrr::map* and length() work correctly on list-columns.
# To unnest for secondary trigger frequency counts: tidyr::unnest(df, nepa_trigger_secondary)

# --- Column normalization ---
# Actual column names in projects_combined.parquet (confirmed from db.py):
#   lead_agency_harmonized  (NOT agency_name)
#   project_state           (NOT state)
#   project_type            (NOT project_technology)
# Rename once here; all figure code uses the short aliases below.
df <- df |>
  rename(
    agency_name        = lead_agency_harmonized,
    state              = project_state,
    project_technology = project_type
  )

cat(sprintf("Loaded %d clean energy projects\n", nrow(df)))
cat("Trigger distribution:\n")
print(table(df$nepa_trigger_primary, useNA = "ifany"))

# --------------------------
# FIGURE 1 — Primary trigger × process type
# --------------------------
# Main deliverable figure. 100% stacked bar; primary trigger only.
# Do not display secondary labels here.

fig1 <- df |>
  count(nepa_trigger_primary, process_type) |>
  group_by(process_type) |>
  mutate(pct = n / sum(n)) |>
  ungroup() |>
  ggplot(aes(x = process_type, y = pct, fill = nepa_trigger_primary)) +
  geom_col(position = "fill", width = 0.7) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_fill_brewer(palette = "Set2", name = "Primary Trigger") +
  labs(
    title = "NEPA Trigger Type by Review Process",
    subtitle = "Clean energy projects only (n = 20,725)",
    x = "Review Process", y = "Share of Projects"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right")

ggsave(file.path(OUTPUT_DIR, "fig1_trigger_by_process.png"),
       fig1, width = 8, height = 5, dpi = 150)
cat("Saved fig1_trigger_by_process.png\n")

# --------------------------
# FIGURE 2 — Agency-trigger heatmap
# --------------------------
# Top 18 agencies by project count. Cells = % of each agency's projects per trigger class.

top_agencies <- df |>
  count(agency_name, sort = TRUE) |>
  filter(!is.na(agency_name)) |>
  slice_head(n = 18) |>
  pull(agency_name)

fig2 <- df |>
  filter(agency_name %in% top_agencies) |>
  count(agency_name, nepa_trigger_primary) |>
  group_by(agency_name) |>
  mutate(pct = n / sum(n), total = sum(n)) |>
  ungroup() |>
  ggplot(aes(
    x = nepa_trigger_primary,
    y = fct_reorder(agency_name, total),
    fill = pct
  )) +
  geom_tile(color = "white") +
  geom_text(aes(label = percent(pct, accuracy = 1)), size = 3, color = "white") +
  scale_fill_viridis_c(
    option = "plasma", direction = -1,
    labels = percent_format(accuracy = 1),
    name = "% of agency\nprojects"
  ) +
  labs(
    title = "Trigger Type Distribution by Lead Agency",
    subtitle = "Top 18 agencies by project count",
    x = "Primary Trigger", y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))

ggsave(file.path(OUTPUT_DIR, "fig2_agency_trigger_heatmap.png"),
       fig2, width = 11, height = 7, dpi = 150)
cat("Saved fig2_agency_trigger_heatmap.png\n")

# --------------------------
# FIGURE 3 — Multi-trigger combination bar  (secondary analysis; on request)
# --------------------------
# Top 10 primary + secondary combinations. nepa_trigger_secondary is a list-column.

fig3 <- df |>
  mutate(trigger_combo = map2_chr(
    nepa_trigger_primary,
    nepa_trigger_secondary,
    ~ if (length(.y) == 0 || all(is.na(.y))) {
        .x
      } else {
        paste(c(.x, sort(.y)), collapse = " + ")
      }
  )) |>
  count(trigger_combo, sort = TRUE) |>
  slice_head(n = 10) |>
  ggplot(aes(x = n, y = fct_reorder(trigger_combo, n))) +
  geom_col(fill = "#012169") +
  geom_text(aes(label = comma(n)), hjust = -0.15, size = 3.5) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15)), labels = comma) +
  labs(
    title = "Top 10 NEPA Trigger Combinations",
    subtitle = "Primary + secondary triggers combined",
    x = "Projects", y = NULL
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(OUTPUT_DIR, "fig3_trigger_combinations.png"),
       fig3, width = 9, height = 5, dpi = 150)
cat("Saved fig3_trigger_combinations.png\n")

# --------------------------
# FIGURE 4 — Trigger × energy technology
# --------------------------
# project_technology = project_type from projects_combined.parquet.
# Filters to technologies with >= 50 projects to avoid thin bars.

tech_min_n <- 50
tech_counts <- df |> count(project_technology) |> filter(n >= tech_min_n)

fig4 <- df |>
  filter(project_technology %in% tech_counts$project_technology,
         !is.na(project_technology)) |>
  count(project_technology, nepa_trigger_primary) |>
  group_by(project_technology) |>
  mutate(pct = n / sum(n), total = sum(n)) |>
  ungroup() |>
  ggplot(aes(
    x = pct,
    y = fct_reorder(project_technology, total),
    fill = nepa_trigger_primary
  )) +
  geom_col(position = "fill") +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  scale_fill_brewer(palette = "Set2", name = "Primary Trigger") +
  labs(
    title = "Primary NEPA Trigger by Energy Technology",
    subtitle = sprintf("Technologies with >= %d projects", tech_min_n),
    x = "Share of Projects", y = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right")

ggsave(file.path(OUTPUT_DIR, "fig4_trigger_by_technology.png"),
       fig4, width = 10, height = 6, dpi = 150)
cat("Saved fig4_trigger_by_technology.png\n")

# --------------------------
# FIGURE 5 — State choropleth (dominant trigger per state)
# --------------------------
# Dominant = trigger class with most projects per state.
# usmap expects state abbreviations in a column named 'state'.

state_dominant <- df |>
  filter(!is.na(state)) |>
  count(state, nepa_trigger_primary) |>
  group_by(state) |>
  slice_max(n, n = 1, with_ties = FALSE) |>
  ungroup() |>
  rename(dominant_trigger = nepa_trigger_primary)

fig5 <- usmap::plot_usmap(data = state_dominant, values = "dominant_trigger", regions = "states") +
  scale_fill_brewer(palette = "Set2", name = "Dominant\nTrigger", na.value = "grey90") +
  labs(
    title = "Dominant NEPA Trigger Type by State",
    subtitle = "Most common primary trigger among clean energy projects in each state"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "right")

ggsave(file.path(OUTPUT_DIR, "fig5_state_choropleth.png"),
       fig5, width = 11, height = 6, dpi = 150)
cat("Saved fig5_state_choropleth.png\n")

# --------------------------
# FIGURE 6 — Trigger × timeline duration  (placeholder; requires D4)
# --------------------------
# Uncomment after projects_timeline_bert.parquet is produced by D4.

# timeline_path <- file.path(BASE_DIR, "data", "analysis", "projects_timeline_bert.parquet")
# if (file.exists(timeline_path)) {
#   timelines <- read_parquet(timeline_path)
#   df_t <- left_join(df, timelines, by = "project_id") |>
#     filter(!is.na(review_duration_days), review_duration_days > 0)
#
#   fig6 <- ggplot(
#     df_t,
#     aes(x = fct_reorder(nepa_trigger_primary, review_duration_days, median),
#         y = review_duration_days / 365)
#   ) +
#     geom_boxplot(fill = "#0047BB", alpha = 0.5, outlier.alpha = 0.3) +
#     coord_flip() +
#     labs(title = "Review Duration by NEPA Trigger Type",
#          x = "Primary Trigger", y = "Duration (years)") +
#     theme_minimal(base_size = 13)
#
#   ggsave(file.path(OUTPUT_DIR, "fig6_trigger_by_duration.png"),
#          fig6, width = 9, height = 5, dpi = 150)
#   cat("Saved fig6_trigger_by_duration.png\n")
# } else {
#   cat("Skipping Fig 6: projects_timeline_bert.parquet not found (run D4 first)\n")
# }

# --------------------------
# TABLE — Representative evidence text excerpts
# --------------------------
# 2 high-confidence examples per class with quotable document text for the report.
# Sources: purpose_and_need, description, or doc_title (not embedding or agency_metadata).

set.seed(42)
excerpts <- df |>
  filter(
    nepa_trigger_confidence == "high",
    nepa_trigger_evidence_source %in% c("purpose_and_need", "description", "doc_title"),
    !is.na(nepa_trigger_evidence_text),
    nchar(nepa_trigger_evidence_text) > 30
  ) |>
  group_by(nepa_trigger_primary) |>
  slice_sample(n = 2) |>
  select(
    nepa_trigger_primary, process_type, agency_name, project_title,
    nepa_trigger_evidence_text, nepa_trigger_evidence_source, nepa_trigger_rule_id
  ) |>
  ungroup() |>
  arrange(nepa_trigger_primary)

write.csv(excerpts, file.path(OUTPUT_DIR, "trigger_evidence_excerpts.csv"), row.names = FALSE)
cat(sprintf("Saved trigger_evidence_excerpts.csv (%d rows)\n", nrow(excerpts)))

# --------------------------
# DIAGNOSTICS
# --------------------------

source_dist <- df |>
  count(nepa_trigger_evidence_source, nepa_trigger_confidence) |>
  arrange(nepa_trigger_evidence_source, nepa_trigger_confidence)
write.csv(source_dist, file.path(OUTPUT_DIR, "trigger_source_distribution.csv"),
          row.names = FALSE)

rule_dist <- df |>
  count(nepa_trigger_rule_id, nepa_trigger_primary, sort = TRUE) |>
  slice_head(n = 25)
write.csv(rule_dist, file.path(OUTPUT_DIR, "trigger_rule_distribution.csv"),
          row.names = FALSE)

flag_rate <- mean(df$nepa_trigger_manual_review, na.rm = TRUE)
cat(sprintf("\nManual review flag rate: %.1f%%  (target: < 5%%)\n", flag_rate * 100))
if (flag_rate > 0.05) {
  cat("WARNING: flag rate exceeds 5%% target. Return to pipeline and tighten thresholds.\n")
}

cat(sprintf("Dual-nexus projects (federal_land + federal_permit): %d (%.1f%%)\n",
            sum(df$is_dual_nexus, na.rm = TRUE),
            100 * mean(df$is_dual_nexus, na.rm = TRUE)))

cat("\nDone. All outputs written to:", OUTPUT_DIR, "\n")
