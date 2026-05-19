# --------------------------
# DELIVERABLE 1: NEPA TRIGGERED — Analysis and Figures
# --------------------------
# Produces seven figures and supporting CSVs from the trigger classification output.
#
# Figure list:
#   fig1  — Primary trigger counts (horizontal bar, sorted)
#   fig2  — Trigger × review process (100% stacked bar)
#   fig3  — Review process mix within each trigger (inverse of fig2), sorted by CE share
#   fig4  — Federal department × trigger heatmap, sorted by Unknown share
#   fig5  — Trigger × energy technology, sorted by Funding share
#   fig6  — State choropleth (dominant trigger per state)
#   fig7  — County choropleth (dominant trigger per county)
#   fig8  — Federal funding mechanism counts (requires funding sidecar)
#   fig9  — Federal funding program/source counts (requires funding sidecar)
#   fig10 — Federal funding amount extraction coverage (requires funding sidecar)
#   fig11 — Federal funding amounts by mechanism (median + IQR; requires funding sidecar)
#
# Input:
#   phase2/data/analysis/deliverable01/projects_nepa_trigger.parquet
#   phase2/data/analysis/deliverable01/projects_funding_details.parquet (optional sidecar)
#   phase2/data/analysis/projects_combined.parquet
#
# Output (all in phase2/output/deliverable01/):
#   fig1_trigger_counts.png
#   fig2_trigger_by_process.png
#   fig3_process_by_trigger.png
#   fig4_department_trigger_heatmap.png
#   fig5_trigger_by_technology.png
#   fig6_state_choropleth.png
#   fig7_county_choropleth.png
#   fig8_funding_mechanism_counts.png
#   fig9_funding_program_counts.png
#   fig10_funding_amount_coverage.png
#   federal_funding_detail_summary.csv
#   trigger_evidence_excerpts.csv
#   trigger_source_distribution.csv
#   trigger_rule_distribution.csv
#
# Usage:
#   Rscript phase2/code/deliverable01/02_analyze_triggers.R


# --------------------------
# LIBRARIES AND SETTINGS
# --------------------------

# clear environment
rm(list = ls())

suppressPackageStartupMessages({
  library(here)
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(forcats)
  library(purrr)
  library(scales)
  library(tigris)
  library(sf)
  library(jsonlite)
  library(stringr)
})

# CATF brand theme, colors, and scale helpers (phase2 canonical copy)
source(here::here("phase2", "code", "utils", "utils.R"))

# --------------------------
# PATHS
# --------------------------

BASE_DIR   <- here::here()
OUTPUT_DIR <- file.path(BASE_DIR, "phase2", "output", "deliverable01")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

TRIGGERS_PATH <- file.path(BASE_DIR, "phase2", "data", "analysis", "deliverable01",
                            "projects_nepa_trigger.parquet")
FUNDING_DETAILS_PATH <- file.path(BASE_DIR, "phase2", "data", "analysis", "deliverable01",
                                  "projects_funding_details.parquet")
PROJECTS_PATH <- file.path(BASE_DIR, "phase2", "data", "analysis", "projects_combined.parquet")

# --------------------------
# LABEL LOOKUPS  ← root of all display labels used in figures AND exported CSVs
# --------------------------
# "Federal" prefix stripped throughout; labels are unambiguous in context.

trigger_labels <- c(
  federal_direct_action        = "Direct Action",
  federal_funding              = "Funding",
  federal_land                 = "Land",
  federal_permit               = "Permit",
  federal_program              = "Program",
  federal_property_transaction = "Property Transaction",
  pma                          = "PMA/TVA",
  unknown                      = "Unknown"
)

# Named color vector for trigger labels — defined once, used in all trigger fill scales.
# Unknown uses neutral grey (matching map NA fill); light_blue moved from Unknown → Permit.
# PMA/TVA uses amber (#F2A900) — distinct from the existing seven-color palette.
trigger_colors <- c(
  "Funding"              = "#0047BB",  # catf_dark_blue
  "Direct Action"        = "#00AE8D",  # catf_teal
  "Land"                 = "#C22A90",  # catf_magenta
  "Permit"               = "#8AB7E9",  # catf_light_blue
  "Program"              = "#00B5E2",  # catf_blue
  "Property Transaction" = "#75246C",  # catf_purple
  "PMA/TVA"              = "#F2A900",  # amber
  "Unknown"              = "grey70"    # neutral grey — matches map NA aesthetic
)

process_labels <- c(
  CE  = "Categorical Exclusion",
  EA  = "Environmental Assessment",
  EIS = "Environmental Impact Statement"
)

funding_type_labels <- c(
  pmc_nd_form           = "EERE Grant (PMC-ND Form)",
  arpa_e                = "ARPA-E Award",
  grant_or_award        = "Grant/Award",
  formula_grant         = "Formula Grant",
  cooperative_agreement = "Cooperative Agreement",
  loan_guarantee        = "Loan Guarantee",
  federal_loan          = "Federal Loan",
  revolving_loan        = "Revolving Loan",
  cost_share            = "Cost Share",
  financial_assistance  = "Financial Assistance",
  generic_funding       = "Generic Funding",
  unknown_funding       = "Unknown Funding Type"
)

# Maps first matching clean-energy NEPATEC tag → short display label
tech_labels <- c(
  "Renewable Energy Production - Solar"              = "Solar",
  "Renewable Energy Production - Wind, Onshore"      = "Wind (Onshore)",
  "Renewable Energy Production - Wind, Offshore"     = "Wind (Offshore)",
  "Electricity Transmission"                         = "Transmission",
  "Carbon Capture and Sequestration"                 = "Carbon Capture & Storage",
  "Renewable Energy Production - Hydropower"         = "Hydropower",
  "Renewable Energy Production - Geothermal"         = "Geothermal",
  "Renewable Energy Production - Biomass"            = "Biomass",
  "Renewable Energy Production - Energy Storage"     = "Energy Storage",
  "Conventional Energy Production - Nuclear"         = "Nuclear",
  "Nuclear Technology"                               = "Nuclear Technology",
  "Renewable Energy Production - Hydrokinetic"       = "Hydrokinetic",
  "Renewable Energy Production - Other"              = "Renewable (Other)",
  "Utilities (electricity, gas, telecommunications)" = "Utilities"
)

# --------------------------
# JSON PARSING HELPERS
# --------------------------
# project_state, project_county, project_type, and lead_agency_harmonized are
# stored as JSON arrays (e.g. '["Alaska", "Montana"]').

parse_json_first <- function(x) {
  if (is.na(x) || nchar(trimws(x)) == 0 || x == "[]") return(NA_character_)
  if (grepl("^\\[", x)) {
    tryCatch({
      parsed <- fromJSON(x)
      if (length(parsed) == 0) return(NA_character_)
      return(as.character(parsed[[1]]))
    }, error = function(e) NA_character_)
  }
  as.character(x)
}

parse_json_all <- function(x) {
  if (is.na(x) || nchar(trimws(x)) == 0 || x == "[]") return(character(0))
  if (grepl("^\\[", x)) {
    tryCatch({
      parsed <- fromJSON(x)
      if (length(parsed) == 0) return(character(0))
      return(as.character(parsed))
    }, error = function(e) character(0))
  }
  as.character(x)
}

# --------------------------
# LOAD AND PREPARE
# --------------------------

options(arrow.skip_nul = TRUE)

triggers <- read_parquet(TRIGGERS_PATH)
projects <- read_parquet(PROJECTS_PATH)
funding_details <- NULL

df_raw <- left_join(triggers, projects, by = "project_id") |>
  filter(project_energy_type == "Clean")

# State + DC abbreviation lookup
state_abbrev_lookup <- setNames(c(state.abb, "DC"), c(state.name, "District of Columbia"))

df <- df_raw |>
  mutate(
    agency_name  = map_chr(lead_agency_harmonized, parse_json_first),
    state_full   = map_chr(project_state,  parse_json_first),
    county       = map_chr(project_county, parse_json_first),
    state        = state_abbrev_lookup[state_full],
    department   = project_department,
    project_technology = map_chr(project_type, function(x) {
      tags    <- parse_json_all(x)
      ce_tags <- tags[tags %in% names(tech_labels)]
      if (length(ce_tags) > 0) tech_labels[[ce_tags[[1]]]] else NA_character_
    }),
    trigger_label = recode(nepa_trigger_primary, !!!trigger_labels),
    process_label = recode(process_type, !!!process_labels)
  )

# Trigger factor: frequency order, Unknown last
trigger_freq_order <- df |>
  filter(trigger_label != "Unknown") |>
  count(trigger_label, sort = TRUE) |>
  pull(trigger_label)
trigger_order <- c(trigger_freq_order, "Unknown")

df <- df |>
  mutate(
    trigger_label = factor(trigger_label, levels = trigger_order),
    process_type  = factor(process_type,  levels = c("CE", "EA", "EIS"))
  )

cat(sprintf("Loaded %d decarbonization projects\n", nrow(df)))
cat("Primary trigger distribution:\n")
print(table(df$trigger_label, useNA = "ifany"))

if (file.exists(FUNDING_DETAILS_PATH)) {
  funding_details <- read_parquet(FUNDING_DETAILS_PATH) |>
    semi_join(df |> filter(nepa_trigger_primary == "federal_funding") |> select(project_id),
              by = "project_id") |>
    mutate(
      funding_type_label = recode(federal_funding_type_primary,
                                  !!!funding_type_labels,
                                  .default = federal_funding_type_primary)
    )

  cat(sprintf("Loaded funding details sidecar (%d funding-primary projects)\n",
              nrow(funding_details)))
} else {
  cat("Funding details sidecar not found; skipping funding detail figures.\n")
}

# --------------------------
# FIGURE 1 — Primary trigger counts (horizontal bar)
# --------------------------

fig1_data <- df |>
  count(trigger_label) |>
  mutate(trigger_label = fct_reorder(trigger_label, n))

fig1 <- ggplot(fig1_data, aes(x = n, y = trigger_label, fill = trigger_label)) +
  geom_col(show.legend = TRUE) +
  geom_text(aes(label = comma(n)), hjust = -0.15, size = 3.5, color = "gray20") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18)), labels = comma) +
  scale_fill_manual(values = trigger_colors, name = NULL, drop = FALSE) +
  labs(
    title    = "NEPA Trigger Type: Project Counts",
    subtitle = "Decarbonization projects by primary trigger classification",
    x = "Number of Projects", y = NULL
  ) +
  theme_catf(base_size = 13) +
  theme(
    legend.position = "bottom",
    axis.text.y     = element_text(color = "gray30")
  ) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE, reverse = TRUE))

ggsave(file.path(OUTPUT_DIR, "fig1_trigger_counts.png"),
       fig1, width = 9, height = 5.5, dpi = 150)
cat("Saved fig1_trigger_counts.png\n")

# --------------------------
# FIGURE 2 — Trigger × review process (100% stacked bar)
# --------------------------

fig2_data <- df |>
  count(trigger_label, process_type) |>
  group_by(process_type) |>
  mutate(pct = n / sum(n)) |>
  ungroup() |>
  mutate(trigger_label = factor(trigger_label, levels = rev(trigger_order)))

fig2 <- ggplot(fig2_data,
               aes(x = process_type, y = pct, fill = trigger_label)) +
  geom_col(position = position_fill(), width = 0.65) +
  geom_text(
    data    = filter(fig2_data, pct > 0.05),
    aes(label = percent(pct, accuracy = 1)),
    position = position_fill(vjust = 0.5),
    color = "white", size = 3.5, fontface = "bold"
  ) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_x_discrete(labels = c(
    CE  = "Categorical\nExclusion",
    EA  = "Environmental\nAssessment",
    EIS = "Environmental\nImpact Statement"
  )) +
  scale_fill_manual(values = trigger_colors, name = NULL, drop = FALSE) +
  labs(
    title    = "NEPA Trigger Type by Review Process",
    subtitle = sprintf("Decarbonization projects only (n = %s)", comma(nrow(df))),
    x = NULL, y = "Share of Projects"
  ) +
  theme_catf(base_size = 13) +
  theme(legend.position = "bottom") +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE, reverse = TRUE))

ggsave(file.path(OUTPUT_DIR, "fig2_trigger_by_process.png"),
       fig2, width = 8, height = 6.5, dpi = 150)
cat("Saved fig2_trigger_by_process.png\n")

# --------------------------
# FIGURE 3 — Review process mix within each trigger, sorted by CE share
# --------------------------
# Sorted top-to-bottom by CE share (highest CE at top = last factor level).

fig3_data <- df |>
  count(trigger_label, process_type) |>
  group_by(trigger_label) |>
  mutate(pct = n / sum(n), total = sum(n)) |>
  ungroup()

# Compute CE share per trigger; fill 0 for triggers with no CE projects
ce_share <- fig3_data |>
  filter(as.character(process_type) == "CE") |>
  select(trigger_label, ce_pct = pct) |>
  mutate(trigger_label = as.character(trigger_label))

# Factor levels: lowest CE at bottom, highest CE at top (ascending → top = last)
ce_order <- data.frame(trigger_label = trigger_order, stringsAsFactors = FALSE) |>
  left_join(ce_share, by = "trigger_label") |>
  mutate(ce_pct = coalesce(ce_pct, 0)) |>
  arrange(ce_pct) |>
  pull(trigger_label)

fig3_data <- fig3_data |>
  mutate(trigger_label = factor(trigger_label, levels = ce_order))

fig3_totals <- fig3_data |>
  distinct(trigger_label, total)

fig3 <- ggplot(fig3_data,
               aes(x = pct, y = trigger_label, fill = process_type)) +
  geom_col(position = position_fill(), width = 0.65) +
  geom_text(
    data    = filter(fig3_data, pct > 0.05),
    aes(label = percent(pct, accuracy = 1)),
    position = position_fill(vjust = 0.5),
    color = "white", size = 3.5, fontface = "bold"
  ) +
  geom_text(
    data = fig3_totals,
    aes(x = 1.09, y = trigger_label, label = comma(total)),
    inherit.aes = FALSE,
    hjust = 1,
    color = catf_navy, size = 3.5, fontface = "bold"
  ) +
  scale_x_continuous(
    labels = percent_format(accuracy = 1),
    breaks = seq(0, 1, by = 0.25),
    limits = c(0, 1.1),
    expand = expansion(mult = c(0, 0.01))
  ) +
  scale_fill_manual(
    values = c(CE = catf_teal, EA = catf_dark_blue, EIS = catf_navy),
    labels = process_labels,
    name   = NULL
  ) +
  labs(
    title    = "Review Process Mix by NEPA Trigger Type",
    subtitle = "Sorted by CE share; right labels show total projects",
    x = "Share of Projects", y = NULL
  ) +
  theme_catf(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(file.path(OUTPUT_DIR, "fig3_process_by_trigger.png"),
       fig3, width = 10, height = 5.5, dpi = 150)
cat("Saved fig3_process_by_trigger.png\n")

# --------------------------
# FIGURE 4 — Federal department × trigger heatmap
# --------------------------
# Sorted by total N descending (largest department at top).
# Includes a "Total N" column on the right; legend below figure.

dept_trigger <- df |>
  filter(!is.na(department)) |>
  count(department, trigger_label) |>
  group_by(department) |>
  mutate(pct = n / sum(n), total = sum(n)) |>
  ungroup()

# Sort departments: largest total N at top (ggplot y-axis: bottom = first level)
dept_order <- dept_trigger |>
  distinct(department, total) |>
  arrange(total) |>   # ascending so largest is rendered at top
  pull(department)

dept_trigger <- dept_trigger |>
  mutate(department = factor(department, levels = dept_order))

# Totals data for the right-hand "N" column
dept_totals <- dept_trigger |>
  distinct(department, total) |>
  mutate(trigger_label = factor("N", levels = c(levels(dept_trigger$trigger_label), "N")))

# Extend trigger_label factor to include the "N" sentinel column
dept_trigger <- dept_trigger |>
  mutate(trigger_label = factor(trigger_label, levels = c(levels(trigger_label), "N")))

fig4 <- ggplot(dept_trigger,
               aes(x = trigger_label, y = department, fill = pct)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(
    aes(label = percent(pct, accuracy = 1),
        color  = if_else(pct > 0.25, "white", catf_navy)),
    size = 3, fontface = "bold"
  ) +
  # Total N column — no fill tile, just right-aligned count label
  geom_text(
    data = dept_totals,
    aes(x = trigger_label, y = department, label = scales::comma(total)),
    inherit.aes = FALSE,
    color = catf_navy, size = 3, fontface = "bold"
  ) +
  scale_color_identity() +
  scale_fill_gradientn(
    colors = c(catf_light_blue, catf_dark_blue, catf_navy),
    labels = percent_format(accuracy = 1),
    name   = "Share of department projects",
    na.value = "white"
  ) +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  labs(
    title    = "NEPA Trigger Distribution by Federal Department",
    subtitle = "Share of each department's decarbonization projects per trigger class; N = total projects",
    x = NULL, y = NULL
  ) +
  theme_catf(base_size = 12) +
  theme(
    axis.text.x      = element_text(lineheight = 0.85),
    legend.position  = "bottom",
    legend.key.width = unit(2, "cm")
  )

ggsave(file.path(OUTPUT_DIR, "fig4_department_trigger_heatmap.png"),
       fig4, width = 13, height = 7, dpi = 150)
cat("Saved fig4_department_trigger_heatmap.png\n")

# --------------------------
# FIGURE 5 — Trigger × energy technology, sorted by Funding share
# --------------------------

tech_min_n <- 50
tech_counts <- df |>
  filter(!is.na(project_technology)) |>
  count(project_technology) |>
  filter(n >= tech_min_n)

fig5_data <- df |>
  filter(project_technology %in% tech_counts$project_technology,
         !is.na(project_technology)) |>
  count(project_technology, trigger_label) |>
  group_by(project_technology) |>
  mutate(pct = n / sum(n), total = sum(n)) |>
  ungroup()

# Sort technologies: highest Funding share at bottom (first factor level)
funding_tech_order <- fig5_data |>
  filter(as.character(trigger_label) == "Funding") |>
  arrange(desc(pct)) |>
  pull(project_technology)
no_funding_techs <- setdiff(unique(fig5_data$project_technology), funding_tech_order)
fig5_data <- fig5_data |>
  mutate(project_technology = factor(project_technology,
                                     levels = c(funding_tech_order, no_funding_techs)))

fig5_totals <- fig5_data |>
  distinct(project_technology, total)

fig5 <- ggplot(fig5_data,
               aes(x = pct, y = project_technology, fill = trigger_label)) +
  geom_col(position = position_fill(), width = 0.7) +
  geom_text(
    aes(label = if_else(pct > 0.10, percent(pct, accuracy = 1), "")),
    position = position_fill(vjust = 0.5),
    color = "white", size = 3, fontface = "bold"
  ) +
  geom_text(
    data = fig5_totals,
    aes(x = 1.09, y = project_technology, label = comma(total)),
    inherit.aes = FALSE,
    hjust = 1,
    color = catf_navy, size = 3.5, fontface = "bold"
  ) +
  scale_x_continuous(
    labels = percent_format(accuracy = 1),
    breaks = seq(0, 1, by = 0.25),
    limits = c(0, 1.1),
    expand = expansion(mult = c(0, 0.01))
  ) +
  scale_fill_manual(values = trigger_colors, name = NULL, drop = FALSE) +
  labs(
    title    = "Primary NEPA Trigger by Energy Technology",
    subtitle = sprintf("Technologies with >= %d projects, sorted by Funding share; right labels show total projects", tech_min_n),
    x = "Share of Projects", y = NULL
  ) +
  theme_catf(base_size = 13) +
  theme(legend.position = "bottom") +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE, reverse = TRUE))

ggsave(file.path(OUTPUT_DIR, "fig5_trigger_by_technology.png"),
       fig5, width = 11, height = 6.5, dpi = 150)
cat("Saved fig5_trigger_by_technology.png\n")

# --------------------------
# FIGURE 6 — State choropleth (dominant trigger per state)
# --------------------------

state_dominant <- df |>
  filter(!is.na(state)) |>
  count(state, trigger_label) |>
  group_by(state) |>
  slice_max(n, n = 1, with_ties = FALSE) |>
  ungroup() |>
  rename(dominant_trigger = trigger_label)

tryCatch({
  states_sf <- states(cb = TRUE, resolution = "20m", year = 2020,
                      progress_bar = FALSE) |>
    filter(!STUSPS %in% c("PR", "VI", "GU", "MP", "AS")) |>
    shift_geometry() |>
    left_join(state_dominant, by = c("STUSPS" = "state"))

  fig6 <- ggplot(states_sf) +
    geom_sf(aes(fill = dominant_trigger), color = "white", linewidth = 0.3) +
    scale_fill_catf(name = "Dominant\nTrigger", na.value = "grey85", drop = FALSE) +
    labs(
      title    = "Dominant NEPA Trigger Type by State",
      subtitle = "Most common primary trigger among decarbonization projects in each state"
    ) +
    theme_void(base_size = 12) +
    theme(
      plot.title      = element_text(face = "bold", color = catf_navy,      size = 14),
      plot.subtitle   = element_text(color = catf_dark_blue, size = 10),
      legend.position = "right"
    )

  ggsave(file.path(OUTPUT_DIR, "fig6_state_choropleth.png"),
         fig6, width = 11, height = 6, dpi = 150)
  cat("Saved fig6_state_choropleth.png\n")
}, error = function(e) {
  cat(sprintf("Skipped fig6_state_choropleth.png: %s\n", conditionMessage(e)))
})

# --------------------------
# FIGURE 7 — County choropleth (dominant trigger per county)
# --------------------------
# project_county is a JSON array of county names; parse_json_first() extracts
# the first listed county. Matching to tigris county geometries uses
# case-insensitive NAME comparison (tigris stores names without "County" suffix).

county_dominant <- df |>
  filter(!is.na(state), !is.na(county), nchar(county) > 0) |>
  mutate(county_lower = str_to_lower(str_remove(county, "(?i) county$"))) |>
  count(state, county_lower, trigger_label) |>
  group_by(state, county_lower) |>
  slice_max(n, n = 1, with_ties = FALSE) |>
  ungroup() |>
  rename(dominant_trigger = trigger_label)

tryCatch({
  counties_sf <- counties(cb = TRUE, resolution = "20m", year = 2020,
                          progress_bar = FALSE) |>
    filter(!STUSPS %in% c("PR", "VI", "GU", "MP", "AS")) |>
    shift_geometry() |>
    mutate(county_lower = str_to_lower(NAME)) |>
    left_join(county_dominant, by = c("STUSPS" = "state", "county_lower"))

  fig7 <- ggplot(counties_sf) +
    geom_sf(aes(fill = dominant_trigger), color = "white", linewidth = 0.05) +
    scale_fill_catf(name = "Dominant\nTrigger", na.value = "grey90", drop = FALSE) +
    labs(
      title    = "Dominant NEPA Trigger Type by County",
      subtitle = "Most common primary trigger among decarbonization projects in each county"
    ) +
    theme_void(base_size = 12) +
    theme(
      plot.title      = element_text(face = "bold", color = catf_navy,      size = 14),
      plot.subtitle   = element_text(color = catf_dark_blue, size = 10),
      legend.position = "right"
    )

  ggsave(file.path(OUTPUT_DIR, "fig7_county_choropleth.png"),
         fig7, width = 11, height = 6, dpi = 150)
  cat("Saved fig7_county_choropleth.png\n")
}, error = function(e) {
  cat(sprintf("Skipped fig7_county_choropleth.png: %s\n", conditionMessage(e)))
})

# ---------------------------------------------------------------------------
# FUNDING DETAIL FIGURES — fig8, fig9, fig10, fig11
# Each block is independent; all are skipped (with a message) if the
# projects_funding_details.parquet sidecar does not exist.
# ---------------------------------------------------------------------------

funding_ready <- !is.null(funding_details) && nrow(funding_details) > 0

# Shared preamble (computed once if sidecar is present)
if (funding_ready) {
  funding_n <- nrow(funding_details)

  funding_program_long <- funding_details |>
    select(project_id, federal_funding_program_multi) |>
    unnest_longer(federal_funding_program_multi,
                  values_to = "funding_program",
                  keep_empty = FALSE) |>
    filter(!is.na(funding_program), nchar(funding_program) > 0)
}

# --------------------------
# FIGURE 8 — Funding mechanism type counts
# --------------------------

if (funding_ready) {
  fig8_data <- funding_details |>
    count(funding_type_label, sort = TRUE) |>
    mutate(
      pct = n / funding_n,
      funding_type_label = fct_reorder(funding_type_label, n)
    )

  fig8 <- ggplot(fig8_data, aes(x = n, y = funding_type_label)) +
    geom_col(fill = catf_teal, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%s (%s)", comma(n), percent(pct, accuracy = 1))),
              hjust = -0.08, size = 3.3, color = "gray20") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.22)), labels = comma) +
    labs(
      title    = "Federal Funding Mechanism Details",
      subtitle = sprintf("Funding-primary decarbonization projects (n = %s)", comma(funding_n)),
      x = "Number of Projects", y = NULL
    ) +
    theme_catf(base_size = 13)

  ggsave(file.path(OUTPUT_DIR, "fig8_funding_mechanism_counts.png"),
         fig8, width = 10, height = 5.8, dpi = 150)
  cat("Saved fig8_funding_mechanism_counts.png\n")
} else {
  cat("Funding sidecar absent; skipped fig8_funding_mechanism_counts.png\n")
}

# --------------------------
# FIGURE 9 — Funding program/source label counts
# --------------------------

if (funding_ready && nrow(funding_program_long) > 0) {
  fig9_data <- funding_program_long |>
    distinct(project_id, funding_program) |>
    count(funding_program, sort = TRUE) |>
    mutate(
      pct = n / funding_n,
      funding_program = fct_reorder(funding_program, n)
    )

  fig9 <- ggplot(fig9_data, aes(x = n, y = funding_program)) +
    geom_col(fill = catf_dark_blue, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%s (%s)", comma(n), percent(pct, accuracy = 1))),
              hjust = -0.08, size = 3.3, color = "gray20") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.22)), labels = comma) +
    scale_y_discrete(labels = function(x) str_wrap(x, width = 38)) +
    labs(
      title    = "Federal Funding Program and Source Labels",
      subtitle = "Multi-label — a project may appear under multiple programs; % denominated by all funding-primary projects",
      x = "Number of Projects", y = NULL,
      caption  = paste0(
        "Acronyms:\n", 
        "FOA = Funding Opportunity Announcement (DOE competitive grant vehicle)\n",
        "ARRA = American Recovery and Reinvestment Act\n",
        "EECG = Energy Efficiency & Conservation Block Grant\n",
        "SEP = American Recovery and Reinvestment Act\n",
        "Title XVII = Energy Policy Act of 2005 §XVII (DOE loan guarantee authority)\n",
        "WAP = Weatherization Assistance Program\n",
        "BIL = Bipartisan Infrastructure Law\n",
        "IRA = Inflation Reduction Act\n"
      )
    ) +
    theme_catf(base_size = 13) +
    theme(
      plot.caption  = element_text(size = rel(0.75), hjust = 0, color = "gray40",
                                   margin = margin(t = 8)),
      axis.text.y   = element_text(lineheight = 0.85)
    )

  ggsave(file.path(OUTPUT_DIR, "fig9_funding_program_counts.png"),
         fig9, width = 10, height = 6.5, dpi = 150)
  cat("Saved fig9_funding_program_counts.png\n")
} else if (funding_ready) {
  cat("No funding program labels found; skipped fig9_funding_program_counts.png\n")
} else {
  cat("Funding sidecar absent; skipped fig9_funding_program_counts.png\n")
}

# --------------------------
# FIGURE 10 — Funding amount extraction coverage
# --------------------------

if (funding_ready) {
  fig10_data <- tibble::tibble(
    metric = c("Federal Amount", "Total Project Cost", "Recipient Cost Share", "Funding Share"),
    n = c(
      sum(!is.na(funding_details$federal_funding_amount_usd)),
      sum(!is.na(funding_details$federal_funding_total_project_cost_usd)),
      sum(!is.na(funding_details$federal_funding_recipient_cost_share_usd)),
      sum(!is.na(funding_details$federal_funding_share_pct))
    )
  ) |>
    mutate(pct = n / funding_n, metric = factor(metric, levels = metric))

  fig10 <- ggplot(fig10_data, aes(x = metric, y = pct, fill = metric)) +
    geom_col(show.legend = FALSE, width = 0.65) +
    geom_text(aes(label = sprintf("%s\n%s", comma(n), percent(pct, accuracy = 1))),
              vjust = -0.25, size = 3.4, color = "gray20") +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       expand = expansion(mult = c(0, 0.16))) +
    scale_fill_catf(drop = FALSE) +
    labs(
      title    = "Federal Funding Amount Extraction Coverage",
      subtitle = "Evidence-backed fields only; missing = no reliable dollar amount was extracted",
      x = NULL, y = "Share of Funding-Primary Projects"
    ) +
    theme_catf(base_size = 13) +
    theme(axis.text.x = element_text(lineheight = 0.9))

  ggsave(file.path(OUTPUT_DIR, "fig10_funding_amount_coverage.png"),
         fig10, width = 9, height = 5.5, dpi = 150)
  cat("Saved fig10_funding_amount_coverage.png\n")
} else {
  cat("Funding sidecar absent; skipped fig10_funding_amount_coverage.png\n")
}

# --------------------------
# FIGURE 11 — Federal funding amounts by mechanism (median + IQR)
# --------------------------

if (funding_ready) {
  n_with_amount <- sum(!is.na(funding_details$federal_funding_amount_usd))
  TOPCAP_USD    <- 1e9   # display cap; stats computed on full data

  # Raw rows for box plots — exclude NA amounts; include ALL mechanism types
  # (no minimum-n filter so every mechanism with at least one amount is shown)
  fig11_raw <- funding_details |>
    filter(!is.na(federal_funding_amount_usd)) |>
    mutate(funding_type_label = fct_reorder(
      funding_type_label, federal_funding_amount_usd, .fun = median, .na_rm = TRUE
    ))

  # Count values above the display cap for the caption
  n_above_cap <- sum(fig11_raw$federal_funding_amount_usd > TOPCAP_USD, na.rm = TRUE)

  if (nrow(fig11_raw) > 0) {
    fig11 <- ggplot(fig11_raw,
                    aes(x = federal_funding_amount_usd, y = funding_type_label)) +
      geom_boxplot(
        fill         = catf_teal,
        color        = catf_navy,
        alpha        = 0.75,
        outlier.size  = 0.7,
        outlier.alpha = 0.25,
        outlier.color = catf_dark_blue
      ) +
      scale_x_continuous(
        labels = label_dollar(scale_cut = cut_short_scale()),
        expand = expansion(mult = c(0, 0.05))
      ) +
      coord_cartesian(xlim = c(0, TOPCAP_USD)) +   # clip display; stats on full data
      labs(
        title    = "Distribution of Federal Funding Amounts by Mechanism Type",
        subtitle = "Box = Q1–Q3, center = median, whiskers = 1.5×IQR, dots = outliers; display capped at $1B",
        x = "Federal Funding Amount (USD)", y = NULL,
        caption  = sprintf(
          paste0(
            "Dollar amounts extracted for %s of %s funding-primary projects (%s%%).\n",
            "Most documents do not include an explicit award figure in extractable text.\n",
            "%s award(s) above $1B not shown; all values used when computing statistics."
          ),
          comma(n_with_amount), comma(funding_n),
          round(100 * n_with_amount / funding_n, 1),
          comma(n_above_cap)
        )
      ) +
      theme_catf(base_size = 13) +
      theme(plot.caption = element_text(size = rel(0.78), hjust = 0, color = "gray40",
                                        margin = margin(t = 6)))

    ggsave(file.path(OUTPUT_DIR, "fig11_funding_amount_distribution.png"),
           fig11, width = 10, height = 5.8, dpi = 150)
    cat("Saved fig11_funding_amount_distribution.png\n")
  } else {
    cat("Fewer than 5 projects with positive amounts per mechanism; skipped fig11.\n")
  }
} else {
  cat("Funding sidecar absent; skipped fig11_funding_amount_distribution.png\n")
}

# --------------------------
# FUNDING DETAIL SUMMARY CSV
# --------------------------

if (funding_ready) {
  mechanism_summary <- if (exists("fig8_data")) {
    fig8_data |>
      transmute(section = "mechanism", item = as.character(funding_type_label),
                n = n, pct = pct, value = NA_real_)
  } else tibble::tibble(section=character(), item=character(), n=integer(),
                        pct=numeric(), value=numeric())

  program_summary <- if (nrow(funding_program_long) > 0) {
    funding_program_long |>
      distinct(project_id, funding_program) |>
      count(funding_program, sort = TRUE) |>
      transmute(section = "program", item = funding_program,
                n = n, pct = n / funding_n, value = NA_real_)
  } else tibble::tibble(section=character(), item=character(), n=integer(),
                        pct=numeric(), value=numeric())

  amount_summary <- bind_rows(
    if (exists("fig10_data")) {
      fig10_data |>
        transmute(section = "amount_coverage", item = as.character(metric),
                  n = n, pct = pct, value = NA_real_)
    } else tibble::tibble(section=character(), item=character(), n=integer(),
                          pct=numeric(), value=numeric()),
    tibble::tibble(
      section = "amount_distribution",
      item    = c("Median Federal Amount", "Mean Federal Amount"),
      n       = sum(!is.na(funding_details$federal_funding_amount_usd)),
      pct     = mean(!is.na(funding_details$federal_funding_amount_usd)),
      value   = c(
        median(funding_details$federal_funding_amount_usd, na.rm = TRUE),
        mean(funding_details$federal_funding_amount_usd,   na.rm = TRUE)
      )
    )
  )

  funding_summary <- bind_rows(mechanism_summary, program_summary, amount_summary)
  write.csv(funding_summary,
            file.path(OUTPUT_DIR, "federal_funding_detail_summary.csv"),
            row.names = FALSE)
  cat(sprintf("Saved federal_funding_detail_summary.csv (%d rows)\n", nrow(funding_summary)))
}

# --------------------------
# TABLE — Representative evidence text excerpts
# --------------------------

set.seed(42)
excerpts <- df |>
  filter(
    nepa_trigger_confidence == "high",
    nepa_trigger_evidence_source %in% c("purpose_and_need", "description", "doc_title"),
    !is.na(nepa_trigger_evidence_text),
    nchar(nepa_trigger_evidence_text) > 30
  ) |>
  group_by(trigger_label) |>
  slice_sample(n = 2) |>
  select(
    `Trigger Type`   = trigger_label,
    `Review Process` = process_type,
    `Lead Agency`    = agency_name,
    `Project Title`  = project_title,
    `Evidence Text`  = nepa_trigger_evidence_text,
    `Evidence Source`= nepa_trigger_evidence_source
  ) |>
  ungroup() |>
  arrange(`Trigger Type`)

write.csv(excerpts, file.path(OUTPUT_DIR, "trigger_evidence_excerpts.csv"),
          row.names = FALSE)
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

cat(sprintf("Dual-nexus projects (Land + Permit): %d (%.1f%%)\n",
            sum(df$is_dual_nexus, na.rm = TRUE),
            100 * mean(df$is_dual_nexus, na.rm = TRUE)))

cat("\nDone. All outputs written to:", OUTPUT_DIR, "\n")
