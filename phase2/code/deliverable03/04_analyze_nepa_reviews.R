# --------------------------
# DELIVERABLE 3: NEPA Review Patterns — Figures and Tables
# --------------------------
# Produces figures and CSVs comparing NEPA review patterns for fossil fuel
# vs. decarbonization projects (technology, agency, CE citations, geography,
# visual impacts, geothermal vs. oil/gas, and timelines).
#
# Figure list:
#   fig1  — CE/EA/EIS rates: Clean vs. Fossil (100% stacked bar)
#   fig2  — CE/EA/EIS rates by tech_group (sorted by CE share)
#   fig3  — Linear vs. Non-linear × energy group (faceted)
#   fig4  — Top 15 CE codes overall (bar)
#   fig5  — Top CE codes by energy type (faceted)
#   fig6  — CE citation heatmap by agency (tile)
#   fig7  — State choropleth: Decarbonization (sqrt scale)
#   fig8  — State choropleth: Fossil Fuel (sqrt scale)
#   fig9  — County choropleth: Decarbonization (Jenks breaks)
#   fig10 — County choropleth: Fossil Fuel (Jenks breaks)
#   fig11 — State facet: energy × process type (2×3 grid)
#   fig12 — Visual analysis universe: project counts by tech_group × energy_group
#   fig13 — Word cloud 2×2 grid: EA/EIS × Decarb/Fossil (TF-IDF top terms)
#   fig14 — Topic prevalence by group (top NMF/BERTopic topics)
#   fig15 — CE/EA/EIS rates: Geothermal vs. Oil & Gas
#   fig16 — Geothermal vs. Oil & Gas share by state (100% stacked bar, all states)
#   fig17 — State choropleth: Geothermal share (diverging blue-purple-red)
#   fig18 — Visual framing comparison (CEQ-axis ratios by energy × process)
#   fig19 — Section length boxplots by tech_group × process_type
#   fig20 — Duration by period × process type × energy (conditional on timeline.parquet)
#
# Input:
#   phase2/data/analysis/deliverable03/projects_nepa_reviews.parquet
#   phase2/data/analysis/deliverable03/ce_citations.parquet
#   phase2/data/analysis/deliverable03/projects_visual_impacts.parquet
#   phase2/data/analysis/deliverable03/projects_geothermal_og.parquet
#   phase2/data/analysis/timeline.parquet  (optional — section 6 skipped if missing)
#
# Output (all in phase2/output/deliverable03/):
#   fig1_review_rates_by_energy.png ... fig17_geo_og_state_map.png
#   review_rates_within_blm.csv, review_rates_within_doe.csv
#   ce_by_trigger.csv, ce_by_geometry.csv
#   geo_state_counts.csv
#   visual_prevalence_table.csv
#   geothermal_comparison_table.csv
#   timeline_coverage.csv, duration_summary.csv  (conditional)
#
# Usage:
#   Rscript phase2/code/deliverable03/02_analyze_nepa_reviews.R

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
  library(stringr)
  library(tigris)
  library(sf)
  library(classInt)
  library(jsonlite)
})

# CATF brand theme, colors, and scale helpers (phase2 canonical copy)
source(here::here("phase2", "code", "utils", "utils.R"))

# Left-justify all captions globally (applies to every theme_catf() call in this script)
theme_update(plot.caption = element_text(hjust = 0))

# Fossil red palette (red hues used for all fossil tech categories throughout)
FOSSIL_RED       <- "#C0392B"   # dark red  — Land-based Oil & Gas, Coal
FOSSIL_RED_MED   <- "#E74C3C"   # medium red — Offshore Oil & Gas
FOSSIL_ORANGE    <- "#CA6F1E"   # terracotta — Pipeline
FOSSIL_TAN       <- "#BA4A00"   # burnt orange — Rural Energy
FOSSIL_PALE      <- "#F5B7B1"   # pale red   — Other Fossil

CE_CODE_FOOTNOTE <- paste0(
  "CE code key:\n",
  "B1.3 (BLM H-1790-1): Activities in previously disturbed areas with minimal soil disturbance\n",
  "B3.1 (BLM H-1790-1): Well operations at locations where prior NEPA analysis was completed within five years\n",
  "B3.6 (BLM H-1790-1): Individual drilling permit (APD) in a developed field\n",
  "B5.1 (BLM H-1790-1): Installation of small-scale fluid mineral facilities\n",
  "516 DM 6 (DOI Departmental Manual): Routine operations with limited environmental impact\n",
  "516 DM 11.9 (DOI Departmental Manual): Non-significant amendments to resource management plans\n",
  "EPAct 2005 §390: Energy Policy Act of 2005, Section 390 — five statutory CEs for oil & gas operations on federal land\n",
  "A9 (BLM H-1790-1): Action covered by a statutory categorical exclusion"
)

# --------------------------
# PATHS
# --------------------------

BASE_DIR   <- here::here()
OUTPUT_DIR <- file.path(BASE_DIR, "phase2", "output", "deliverable03")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

D03_DIR       <- file.path(BASE_DIR, "phase2", "data", "analysis", "deliverable03")
REVIEWS_PATH  <- file.path(D03_DIR, "projects_nepa_reviews.parquet")
CE_PATH       <- file.path(D03_DIR, "ce_citations.parquet")
VISUAL_PATH   <- file.path(D03_DIR, "projects_visual_impacts.parquet")
GEO_OG_PATH   <- file.path(D03_DIR, "projects_geothermal_og.parquet")
TIMELINE_PATH <- file.path(BASE_DIR, "phase2", "data", "analysis", "timeline.parquet")

# Human-readable interpretive labels for NMF topics.
# Keys are the auto-generated term labels from the Python pipeline.
# If a run produces different auto-labels (e.g., after vocabulary changes),
# add the new label string here — unmatched labels fall through to their
# auto-generated form unchanged.
TOPIC_INTERP <- c(
  "shadow / flicker"                        = "Wind Turbine Shadow Flicker",
  "contrast / visual contrast"              = "VRM Contrast Rating & Solar Glare",
  "glare / transmission / light"            = "Industrial & Infrastructure Corridors",
  "contrast / visual contrast / objectives" = "BLM VRM Objectives & Landscape Management"
)

# New visual-impact pipeline parquets (Stage 1–4 outputs)
VISUAL_SECTIONS_PATH      <- file.path(D03_DIR, "visual_sections.parquet")
VISUAL_TEXT_PATH          <- file.path(D03_DIR, "projects_visual_text.parquet")
VISUAL_TOPICS_PATH        <- file.path(D03_DIR, "visual_topics.parquet")
VISUAL_TOPIC_SUMMARY_PATH <- file.path(D03_DIR, "visual_topic_summary.parquet")
VISUAL_FRAMING_PATH       <- file.path(D03_DIR, "visual_framing.parquet")
VISUAL_EXAMPLES_PATH      <- file.path(D03_DIR, "visual_examples.parquet")
VISUAL_TOPIC_TERMS_PATH   <- file.path(OUTPUT_DIR, "visual_topic_terms_detail.csv")
VISUAL_TOPIC_EXCERPTS_PATH <- file.path(OUTPUT_DIR, "visual_topic_excerpts.csv")
VRM_ELEMENTS_PATH          <- file.path(D03_DIR, "vrm_elements.parquet")

# --------------------------
# BRAND CONSTANTS
# --------------------------

process_colors <- c(
  EIS = catf_navy,        # #012169 — darkest
  EA  = catf_dark_blue,   # #0047BB — medium
  CE  = catf_light_blue   # #8AB7E9 — lightest
)

process_labels <- c(
  CE  = "Categorical Exclusion",
  EA  = "Environmental Assessment",
  EIS = "Environmental Impact Statement"
)

energy_colors <- c(
  "Clean"          = catf_dark_blue,
  "Fossil"         = catf_navy,
  "Decarbonization" = catf_dark_blue,
  "Fossil Fuel"    = catf_navy
)

tech_colors <- c(
  # Clean (blue/teal/green palette)
  "Wind"                = catf_dark_blue,
  "Solar"               = "#F5A623",
  "Transmission"        = catf_teal,
  "Geothermal"          = "#4A90D9",
  "Hydropower"          = "#5DADE2",
  "Biomass"             = catf_lime,
  "Energy Storage"      = catf_magenta,
  "CCS"                 = catf_purple,
  "Nuclear"             = "#8B4513",
  "Other Clean"         = catf_blue,
  # Fossil (red palette — distinct from the blue/process-type colors)
  "Land-based Oil & Gas" = FOSSIL_RED,
  "Offshore Oil & Gas"   = FOSSIL_RED_MED,
  "Coal"                 = "#7B241C",
  "Pipeline"             = FOSSIL_ORANGE,
  "Rural Energy"         = FOSSIL_TAN,
  "Other Fossil"         = FOSSIL_PALE
)

DATA_CAPTION <- "Data source: NEPATEC 2.0 / NEPAccess"
FIG_W <- 10
FIG_H <- 7

save_fig <- function(name, width = FIG_W, height = FIG_H) {
  ggsave(file.path(OUTPUT_DIR, name), width = width, height = height, dpi = 300)
  invisible(NULL)
}

# --------------------------
# INLINE HELPERS
# --------------------------
# (not in utils.R — defined here to avoid sourcing 00_setup.R wholesale)

parse_json_first <- function(x) {
  if (is.null(x) || is.na(x) || x == "" || x == "null" || x == "[]") return(NA_character_)
  result <- tryCatch(jsonlite::fromJSON(x), error = function(e) NULL)
  if (is.null(result)) return(as.character(x))
  if (is.character(result) && length(result) >= 1) return(result[[1]])
  if (is.list(result) && length(result) >= 1)      return(as.character(result[[1]]))
  return(as.character(result))
}

parse_json_all <- function(x) {
  if (is.null(x) || is.na(x) || x == "" || x == "null" || x == "[]") return(character(0))
  result <- tryCatch(jsonlite::fromJSON(x), error = function(e) NULL)
  if (is.null(result)) return(as.character(x))
  as.character(unlist(result))
}

explode_column <- function(df, col_name) {
  df |>
    dplyr::mutate(!!col_name := purrr::map(.data[[col_name]], function(x) {
      vals <- parse_json_all(x)
      if (length(vals) == 0) NA_character_ else vals
    })) |>
    tidyr::unnest(!!rlang::sym(col_name), keep_empty = TRUE)
}

safe_agency_match <- function(x, pattern) {
  val <- parse_json_first(x)
  if (is.null(val) || is.na(val)) return(FALSE)
  str_detect(val, regex(pattern, ignore_case = TRUE))
}

# --------------------------
# LOAD DATA
# --------------------------

if (!file.exists(REVIEWS_PATH)) {
  stop(paste(
    "projects_nepa_reviews.parquet not found.\n",
    "Run: python phase2/code/deliverable03/01_build_nepa_reviews.py --reviews"
  ))
}

cat("Loading review data...\n")
df <- read_parquet(REVIEWS_PATH)
cat(sprintf("  projects_nepa_reviews: %s rows\n", scales::comma(nrow(df))))

CE_AVAILABLE     <- file.exists(CE_PATH)
VISUAL_AVAILABLE <- file.exists(VISUAL_PATH)
GEO_OG_AVAILABLE <- file.exists(GEO_OG_PATH)

# New visual-impact pipeline availability flags (each section gracefully skipped
# if its input parquet is missing)
VISUAL_SECTIONS_AVAILABLE <- file.exists(VISUAL_SECTIONS_PATH)
VISUAL_TEXT_AVAILABLE     <- file.exists(VISUAL_TEXT_PATH)
VISUAL_TOPICS_AVAILABLE   <- file.exists(VISUAL_TOPICS_PATH) ||
                             file.exists(VISUAL_TOPIC_SUMMARY_PATH)
VISUAL_FRAMING_AVAILABLE  <- file.exists(VISUAL_FRAMING_PATH)
VISUAL_EXAMPLES_AVAILABLE <- file.exists(VISUAL_EXAMPLES_PATH)
VRM_ELEMENTS_AVAILABLE    <- file.exists(VRM_ELEMENTS_PATH)

ce_cits_raw <- if (CE_AVAILABLE) read_parquet(CE_PATH) else NULL
# Consolidate all Section 390 variants (Energy Policy Act / National Energy Policy Act /
# bare "Section 390") into a single canonical code. All are the same statute;
# "National Energy Policy Act" is a citation error that appears in BLM documents.
ce_cits <- if (!is.null(ce_cits_raw)) {
  ce_cits_raw |>
    mutate(ce_code = if_else(
      str_detect(ce_code, regex("section\\s*390", ignore_case = TRUE)),
      "EPAct 2005 §390",
      ce_code
    ))
} else NULL
vis     <- if (VISUAL_AVAILABLE)  read_parquet(VISUAL_PATH) else NULL
geo_og  <- if (GEO_OG_AVAILABLE) read_parquet(GEO_OG_PATH) else NULL

if (!CE_AVAILABLE)
  message("ce_citations.parquet not found — Section 2 (CE Citations) will be skipped.")
if (!VISUAL_AVAILABLE)
  message("projects_visual_impacts.parquet not found — Section 4 (Visual Impacts) will be skipped.")
if (!GEO_OG_AVAILABLE)
  message("projects_geothermal_og.parquet not found — Section 5 (Geothermal/OG) will be skipped.")

if (CE_AVAILABLE)
  cat(sprintf("  ce_citations: %s rows\n", scales::comma(nrow(ce_cits))))
if (VISUAL_AVAILABLE)
  cat(sprintf("  visual_impacts: %s rows\n", scales::comma(nrow(vis))))
if (GEO_OG_AVAILABLE)
  cat(sprintf("  geothermal_og: %s rows\n", scales::comma(nrow(geo_og))))


# ===========================================================================
# SECTION 1: REVIEW RATES
# ===========================================================================
cat("\n--- Section 1: Review Rates ---\n")

# Fig 1 — CE/EA/EIS rates: Decarbonization vs. Fossil Fuel (100% stacked bar) ----
rate_by_energy <- df |>
  filter(!is.na(process_type), energy_group %in% c("Decarbonization", "Fossil Fuel")) |>
  count(energy_group, process_type) |>
  group_by(energy_group) |>
  mutate(pct = n / sum(n)) |>
  ungroup() |>
  mutate(
    energy_group = factor(energy_group, levels = c("Fossil Fuel", "Decarbonization")),
    process_type = factor(process_type, levels = c("EIS", "EA", "CE"))
  )

ggplot(rate_by_energy, aes(x = energy_group, y = pct, fill = process_type)) +
  geom_col(position = position_stack(reverse = TRUE)) +
  geom_text(aes(label = scales::percent(pct, accuracy = 1)),
            position = position_stack(reverse = TRUE, vjust = 0.5),
            color = "white", size = 3.5, fontface = "bold") +
  scale_fill_manual(values = process_colors, labels = process_labels) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = NULL, y = "Share of Projects", fill = "Review Type",
       title = "NEPA Review Type by Energy Category",
       caption = paste0(
         DATA_CAPTION, "\n",
         "Decarbonization includes wind, solar, electricity transmission, geothermal, hydropower,\n",
         "biomass, energy storage, carbon capture and sequestration (CCS), and nuclear\n",
         "(20,725 projects). Fossil Fuel includes land-based oil & gas, offshore oil & gas,\n",
         "coal, pipelines, and rural energy projects (10,783 projects)."
       )) +
  theme_catf() +
  theme(legend.position = "bottom", plot.caption = element_text(hjust = 0))
save_fig("fig1_review_rates_by_energy.png")

# Fig 2 — CE/EA/EIS rates by tech_group (sorted by CE share) ----
rate_by_tech <- df |>
  filter(!is.na(process_type), !is.na(tech_group), !tech_group %in% c("Other")) |>
  count(tech_group, process_type) |>
  group_by(tech_group) |>
  mutate(pct = n / sum(n)) |>
  ungroup()

# Build CE-share ordering: tech groups with CE sorted ascending; those without CE first (0 share)
ce_order <- rate_by_tech |>
  filter(process_type == "CE") |>
  arrange(pct) |>
  pull(tech_group)

no_ce_techs <- setdiff(unique(rate_by_tech$tech_group), ce_order)
ce_order_complete <- c(no_ce_techs, ce_order)

rate_by_tech <- rate_by_tech |>
  mutate(
    tech_group   = factor(tech_group, levels = ce_order_complete),
    process_type = factor(process_type, levels = c("EIS", "EA", "CE"))
  )

# Dual color scheme: clean = blue shades, fossil = red shades (mirroring CE/EA/EIS hierarchy)
fossil_tech_groups <- c("Land-based Oil & Gas", "Offshore Oil & Gas", "Coal",
                        "Pipeline", "Rural Energy", "Other Fossil")

dual_process_colors <- c(
  "Decarb EIS"  = catf_navy,
  "Decarb EA"   = catf_dark_blue,
  "Decarb CE"   = catf_light_blue,
  "Fossil EIS"  = "#7B241C",
  "Fossil EA"   = "#E74C3C",
  "Fossil CE"   = "#F1948A"
)
dual_process_labels <- c(
  "Decarb EIS" = "EIS",   "Decarb EA"  = "EA",    "Decarb CE"  = "CE",
  "Fossil EIS" = "EIS",   "Fossil EA"  = "EA",    "Fossil CE"  = "CE"
)

rate_by_tech <- rate_by_tech |>
  mutate(
    is_fossil = tech_group %in% fossil_tech_groups,
    fill_key  = factor(
      paste0(if_else(is_fossil, "Fossil", "Decarb"), " ", process_type),
      levels = names(dual_process_colors)
    )
  )

axis_label_colors <- ifelse(
  levels(rate_by_tech$tech_group) %in% fossil_tech_groups,
  FOSSIL_RED, catf_navy
)

ggplot(rate_by_tech, aes(x = tech_group, y = pct, fill = fill_key)) +
  geom_col(position = position_stack(reverse = TRUE)) +
  geom_text(aes(label = ifelse(pct >= 0.04, scales::percent(pct, accuracy = 1), "")),
            position = position_stack(reverse = TRUE, vjust = 0.5),
            color = "white", size = 3) +
  scale_fill_manual(
    values = dual_process_colors,
    labels = dual_process_labels,
    breaks = names(dual_process_colors),
    guide  = guide_legend(
      title  = "Review Type",
      nrow   = 2,
      byrow  = TRUE,
      override.aes = list(
        fill = c(catf_navy, catf_dark_blue, catf_light_blue, "#7B241C", "#E74C3C", "#F1948A")
      )
    )
  ) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = NULL, y = "Share of Projects", fill = "Review Type",
       title = "NEPA Review Type by Technology",
       caption = paste0(
         DATA_CAPTION, "\n",
         "Blue bars = Decarbonization projects; red bars = Fossil Fuel projects. ",
         "CCS = Carbon Capture and Sequestration/Storage."
       )) +
  theme_catf() +
  theme(
    legend.position = "bottom",
    axis.text.y     = element_text(color = axis_label_colors),
    plot.caption    = element_text(hjust = 0)
  )
save_fig("fig2_review_rates_by_tech.png", height = 8)

# Fig 3 — Linear vs. Non-linear × energy group ----
rate_linear <- df |>
  filter(!is.na(process_type), !is.na(is_linear),
         project_energy_type %in% c("Clean", "Fossil")) |>
  mutate(
    project_class = ifelse(is_linear, "Linear", "Non-linear"),
    process_type  = factor(process_type, levels = c("EIS", "EA", "CE"))
  ) |>
  count(project_energy_type, project_class, process_type) |>
  group_by(project_energy_type, project_class) |>
  mutate(pct = n / sum(n))

if (nrow(rate_linear) == 0) {
  message("Skipping fig3: is_linear not yet derived (all NULL)")
} else {
  ggplot(rate_linear, aes(x = project_class, y = pct, fill = process_type)) +
    geom_col(position = position_stack(reverse = TRUE)) +
    scale_fill_manual(values = process_colors, labels = process_labels) +
    scale_y_continuous(labels = percent_format()) +
    facet_wrap(~ project_energy_type) +
    coord_flip() +
    labs(x = NULL, y = "Share of Projects", fill = "Review Type",
         title = "Review Type by Project Geometry and Energy Category",
         caption = DATA_CAPTION) +
    theme_catf() +
    theme(legend.position = "bottom")
  save_fig("fig3_review_rates_linear.png")
}

# Statistics: chi-square + Cramér's V ----
contingency <- df |>
  filter(project_energy_type %in% c("Clean", "Fossil"), !is.na(process_type)) |>
  count(project_energy_type, process_type) |>
  pivot_wider(names_from = process_type, values_from = n, values_fill = 0)

mat       <- as.matrix(contingency[, -1])
chi_res   <- chisq.test(mat)
n_total   <- sum(mat)
cramers_v <- sqrt(chi_res$statistic / (n_total * (min(dim(mat)) - 1)))
cat(sprintf(
  "Chi-sq = %.1f, df = %d, p = %.3e, Cramér's V = %.3f\n",
  chi_res$statistic, chi_res$parameter, chi_res$p.value, cramers_v
))

# Within-agency controlled comparisons ----
within_blm <- df |>
  filter(
    map_lgl(lead_agency_harmonized,
            ~ safe_agency_match(.x, "BLM|Bureau of Land Management")),
    project_energy_type %in% c("Clean", "Fossil"),
    !is.na(process_type)
  ) |>
  count(project_energy_type, process_type) |>
  group_by(project_energy_type) |>
  mutate(pct = n / sum(n))

within_doe <- df |>
  filter(
    map_lgl(lead_agency_harmonized,
            ~ safe_agency_match(.x, "Department of Energy|DOE")),
    project_energy_type %in% c("Clean", "Fossil"),
    !is.na(process_type)
  ) |>
  count(project_energy_type, process_type) |>
  group_by(project_energy_type) |>
  mutate(pct = n / sum(n))

write.csv(within_blm, file.path(OUTPUT_DIR, "review_rates_within_blm.csv"), row.names = FALSE)
write.csv(within_doe, file.path(OUTPUT_DIR, "review_rates_within_doe.csv"), row.names = FALSE)
cat("  Section 1 done.\n")


# ===========================================================================
# SECTION 2: CATEGORICAL EXCLUSIONS
# ===========================================================================
cat("\n--- Section 2: Categorical Exclusions ---\n")
if (!CE_AVAILABLE) {
  message("Section 2 skipped: ce_citations.parquet not found.")
} else {

# Fig 4 — Top 15 CE codes overall ----
top_codes <- ce_cits |>
  count(ce_code, sort = TRUE) |>
  slice_head(n = 15) |>
  mutate(ce_code = reorder(ce_code, n))

ggplot(top_codes, aes(x = n, y = ce_code)) +
  geom_col(fill = catf_dark_blue) +
  geom_text(aes(label = scales::comma(n)), hjust = -0.1, size = 3, color = catf_navy) +
  scale_x_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.15))) +
  labs(x = "Number of Documents", y = NULL,
       title = "Most-Cited Categorical Exclusions",
       caption = paste0(DATA_CAPTION, "\n", CE_CODE_FOOTNOTE)) +
  theme_catf() +
  theme(plot.caption = element_text(size = rel(0.75), hjust = 0,
                                    color = "gray40", margin = margin(t = 8),
                                    lineheight = 1.3))
save_fig("fig4_top_ce_codes.png", height = 9)

# Fig 5 — CE codes side by side: top codes with Decarbonization vs. Fossil Fuel bars ----
ce_by_energy_all <- ce_cits |>
  left_join(df |> select(project_id, energy_group), by = "project_id") |>
  filter(energy_group %in% c("Decarbonization", "Fossil Fuel")) |>
  count(energy_group, ce_code)

# Top 10 codes by total citations; y-axis sorted by Decarbonization N (largest at top)
top10_codes <- ce_by_energy_all |>
  group_by(ce_code) |>
  summarise(total = sum(n), .groups = "drop") |>
  slice_max(total, n = 10) |>
  pull(ce_code)

decarb_order <- ce_by_energy_all |>
  filter(energy_group == "Decarbonization", ce_code %in% top10_codes) |>
  arrange(n) |>   # ascending so factor levels give largest at top
  pull(ce_code)

not_in_decarb  <- setdiff(top10_codes, decarb_order)
top_fig5_codes <- c(not_in_decarb, decarb_order)

ce_by_energy <- ce_by_energy_all |>
  filter(ce_code %in% top_fig5_codes) |>
  mutate(ce_code = factor(ce_code, levels = top_fig5_codes))

fig5_fill <- c("Decarbonization" = catf_dark_blue, "Fossil Fuel" = FOSSIL_RED)

ggplot(ce_by_energy, aes(x = n, y = ce_code, fill = energy_group)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = fig5_fill) +
  scale_x_continuous(labels = scales::comma) +
  labs(x = "Documents", y = NULL, fill = NULL,
       title = "Top CE Citations by Energy Category",
       caption = paste0(DATA_CAPTION, "\n", CE_CODE_FOOTNOTE)) +
  theme_catf() +
  theme(
    legend.position = "bottom",
    plot.caption    = element_text(size = rel(0.75), hjust = 0,
                                   color = "gray40", margin = margin(t = 8),
                                   lineheight = 1.3)
  )
save_fig("fig5_ce_by_energy.png", height = 9)

# Fig 6 — CE heatmap by agency ----
top5_agencies <- ce_cits |>
  left_join(df |> select(project_id, lead_agency_harmonized), by = "project_id") |>
  mutate(agency = map_chr(lead_agency_harmonized, parse_json_first)) |>
  filter(!is.na(agency)) |>
  count(agency, sort = TRUE) |>
  slice_head(n = 5) |>
  pull(agency)

ce_heatmap <- ce_cits |>
  left_join(df |> select(project_id, lead_agency_harmonized), by = "project_id") |>
  mutate(agency = map_chr(lead_agency_harmonized, parse_json_first)) |>
  filter(agency %in% top5_agencies, ce_code %in% levels(top_codes$ce_code)) |>
  count(agency, ce_code) |>
  group_by(agency) |>
  mutate(pct = n / sum(n))

ggplot(ce_heatmap, aes(x = ce_code, y = agency, fill = pct)) +
  geom_tile(color = "white") +
  scale_fill_gradient(
    low    = "#deebf7",
    high   = catf_navy,
    labels = percent_format(),
    guide  = guide_colorbar(barwidth = unit(12, "cm"), barheight = unit(0.4, "cm"),
                            title.position = "top", title.hjust = 0.5)
  ) +
  labs(x = "CE Code", y = NULL, fill = "% of Agency CEs",
       title = "CE Citation Heatmap by Agency",
       caption = paste0(DATA_CAPTION, "\n", CE_CODE_FOOTNOTE)) +
  theme_catf() +
  theme(
    axis.text.x     = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    plot.caption    = element_text(size = rel(0.75), hjust = 0,
                                   color = "gray40", margin = margin(t = 8),
                                   lineheight = 1.3)
  )
save_fig("fig6_ce_by_agency.png", width = 12, height = 7)

# CE cross-tab: by trigger and geometry ----
ce_by_trigger <- ce_cits |>
  left_join(df |> select(project_id, nepa_trigger_primary), by = "project_id") |>
  filter(!is.na(nepa_trigger_primary)) |>
  count(nepa_trigger_primary, ce_code, sort = TRUE) |>
  group_by(nepa_trigger_primary) |>
  slice_head(n = 5)

ce_by_geometry <- ce_cits |>
  left_join(df |> select(project_id, is_linear, project_energy_type), by = "project_id") |>
  filter(!is.na(is_linear), project_energy_type %in% c("Clean", "Fossil")) |>
  mutate(geometry = ifelse(is_linear, "Linear", "Non-linear")) |>
  count(geometry, project_energy_type, ce_code, sort = TRUE) |>
  group_by(geometry, project_energy_type) |>
  slice_head(n = 5)

if (nrow(ce_by_geometry) == 0) {
  message("Skipping ce_by_geometry.csv: is_linear not yet derived")
  ce_by_geometry <- NULL
}

write.csv(ce_by_trigger,  file.path(OUTPUT_DIR, "ce_by_trigger.csv"),  row.names = FALSE)
if (!is.null(ce_by_geometry)) write.csv(ce_by_geometry, file.path(OUTPUT_DIR, "ce_by_geometry.csv"), row.names = FALSE)
cat("  Section 2 done.\n")
} # end if (CE_AVAILABLE)


# ===========================================================================
# SECTION 3: GEOGRAPHY
# ===========================================================================
cat("\n--- Section 3: Geography ---\n")

options(tigris_use_cache = TRUE)

us_states <- states(cb = TRUE, resolution = "20m") |>
  filter(!STUSPS %in% c("PR", "VI", "GU", "AS", "MP")) |>
  shift_geometry()

us_counties <- counties(cb = TRUE, resolution = "20m") |>
  filter(!STATEFP %in% c("72", "78", "66", "69", "60")) |>
  shift_geometry()

# Normalize state name column — tigris returns STATE_NAME (uppercase) in cb=TRUE downloads
if ("STATE_NAME" %in% names(us_counties) && !"state_name" %in% names(us_counties)) {
  us_counties <- us_counties |> rename(state_name = STATE_NAME)
} else if (!"state_name" %in% names(us_counties)) {
  # Fallback: build from states sf object (STUSPS → full NAME)
  state_lu <- us_states |> sf::st_drop_geometry() |> select(STUSPS, state_name = NAME)
  us_counties <- us_counties |> left_join(state_lu, by = "STUSPS")
}

# Explode project_state (JSON array) to one row per state per project.
# energy_group is pre-computed in the Python builder; "Other" projects are already
# excluded (energy_group == "Other"), so no misclassification risk here.
location_data <- df |>
  filter(!is.na(process_type), energy_group %in% c("Decarbonization", "Fossil Fuel")) |>
  explode_column("project_state") |>
  filter(!is.na(project_state), project_state != "", project_state != "[]")

state_counts <- location_data |>
  count(energy_group, project_state, name = "n_projects")

write.csv(state_counts, file.path(OUTPUT_DIR, "geo_state_counts.csv"), row.names = FALSE)

# Fig 7 & 8 — State choropleths (sqrt scale) ----
make_state_map <- function(energy, title_suffix) {
  data <- state_counts |>
    filter(energy_group == energy) |>
    right_join(us_states, by = c("project_state" = "NAME")) |>
    st_as_sf() |>
    mutate(n_projects = replace_na(n_projects, 0))

  ggplot(data) +
    geom_sf(aes(fill = n_projects), color = "white", linewidth = 0.2) +
    scale_fill_gradient(
      low    = "#deebf7",
      high   = catf_navy,
      labels = scales::comma,
      name   = "Count of projects",
      guide  = guide_colorbar(barwidth = unit(8, "cm"), barheight = unit(0.5, "cm"),
                              title.position = "top", title.hjust = 0.5)
    ) +
    coord_sf(datum = NA) +
    labs(title   = paste("Projects by State —", title_suffix),
         caption = DATA_CAPTION) +
    theme_void() +
    theme_catf() +
    theme(legend.position = "bottom")
}

make_state_map("Decarbonization", "Decarbonization Technologies")
save_fig("fig7_state_decarb.png", width = 12, height = 7)

make_state_map("Fossil Fuel", "Fossil Fuel Technologies")
save_fig("fig8_state_fossil.png", width = 12, height = 7)

# Fig 9 & 10 — County choropleths (Jenks breaks) ----
county_data <- df |>
  filter(!is.na(process_type), energy_group %in% c("Decarbonization", "Fossil Fuel")) |>
  mutate(first_state = map_chr(project_state, parse_json_first)) |>
  explode_column("project_county") |>
  filter(!is.na(project_county), project_county != "", project_county != "[]")

county_counts <- county_data |>
  count(energy_group, project_county, first_state, name = "n_projects")

# Compute shared Jenks breaks across both energy types so maps are comparable
all_county_n <- county_counts$n_projects[county_counts$n_projects > 0]
shared_breaks <- tryCatch(
  classIntervals(all_county_n, n = 4, style = "jenks")$brks,
  error = function(e) quantile(all_county_n, seq(0, 1, 0.25), na.rm = TRUE)
)
shared_breaks <- unique(shared_breaks)
if (length(shared_breaks) < 2)
  shared_breaks <- c(0, max(all_county_n, na.rm = TRUE) + 1)

make_county_map <- function(energy, title_suffix) {
  data <- county_counts |>
    filter(energy_group == energy) |>
    mutate(jenks = cut(n_projects, shared_breaks, include.lowest = TRUE))

  county_sf <- us_counties |>
    left_join(data, by = c("NAME" = "project_county", "state_name" = "first_state"))

  ggplot() +
    geom_sf(data = us_counties, fill = "grey95", color = "white", linewidth = 0.1) +
    geom_sf(data = county_sf, aes(fill = jenks), color = NA) +
    geom_sf(data = us_states, fill = NA, color = "grey40", linewidth = 0.3) +
    scale_fill_brewer(
      palette  = "Blues",
      name     = "Count of projects",
      na.value = "grey95",
      drop     = FALSE
    ) +
    coord_sf(datum = NA) +
    labs(
      title    = paste("Projects by County —", title_suffix),
      subtitle = "Jenks natural breaks (shared scale); grey = no projects",
      caption  = DATA_CAPTION
    ) +
    theme_void() +
    theme_catf() +
    theme(legend.position = "bottom")
}

make_county_map("Decarbonization", "Decarbonization Technologies")
save_fig("fig9_county_decarb.png", width = 14, height = 8)

make_county_map("Fossil Fuel", "Fossil Fuel Technologies")
save_fig("fig10_county_fossil.png", width = 14, height = 8)

# Fig 11 — State facet: energy × process type (2×3 grid) ----
# Build a complete grid (energy × process × state) so no NA facet panels appear.
state_pct_raw <- location_data |>
  filter(!is.na(energy_group), !is.na(process_type)) |>
  count(energy_group, project_state, process_type) |>
  group_by(energy_group, project_state) |>
  mutate(pct = n / sum(n)) |>
  ungroup()

all_state_names <- us_states |> sf::st_drop_geometry() |> pull(NAME)

state_process <- crossing(
  energy_group  = c("Decarbonization", "Fossil Fuel"),
  process_type  = factor(c("EIS", "EA", "CE"), levels = c("EIS", "EA", "CE")),
  project_state = all_state_names
) |>
  left_join(state_pct_raw, by = c("energy_group", "process_type", "project_state")) |>
  mutate(pct = replace_na(pct, 0)) |>
  left_join(us_states, by = c("project_state" = "NAME")) |>
  st_as_sf()

ggplot(state_process) +
  geom_sf(aes(fill = pct), color = "white", linewidth = 0.1) +
  scale_fill_gradient(
    low    = "#deebf7",
    high   = catf_navy,
    labels = percent_format(),
    name   = "Share of\nProjects",
    guide  = guide_colorbar(barwidth = unit(5, "cm"), barheight = unit(0.4, "cm"),
                            title.position = "top", title.hjust = 0.5)
  ) +
  facet_grid(energy_group ~ process_type,
             labeller = labeller(process_type = as_labeller(process_labels))) +
  coord_sf(datum = NA) +
  labs(title    = "Process Type Share by State and Energy Category",
       subtitle = "Share within each energy category × state: of all projects in a state, what fraction went through each review type",
       caption  = DATA_CAPTION) +
  theme_void() +
  theme_catf() +
  theme(legend.position = "bottom")
save_fig("fig11_state_process_facet.png", width = 14, height = 9)

cat("  Section 3 done.\n")


# ===========================================================================
# SECTION 4: VISUAL IMPACTS (rewritten — new pipeline)
# ===========================================================================
# Replaces the old similarity-score figures with linguistic analyses driven by
# the Stage 1–4 pipeline outputs (visual_sections.parquet, projects_visual_text,
# visual_topics, visual_topic_summary, visual_framing, visual_examples). Each
# figure is independently guarded so missing inputs only skip their own figure.
cat("\n--- Section 4: Visual Impacts ---\n")

# energy_group factor levels used across all Section 4 figures
ENERGY_LEVELS <- c("Decarbonization", "Fossil Fuel")

# ---------------------------------------------------------------------------
# fig12 — Visual analysis universe: project counts by tech_group × energy_group
# ---------------------------------------------------------------------------
if (!VISUAL_TEXT_AVAILABLE) {
  message("fig12 skipped: projects_visual_text.parquet not found.")
} else {
  tryCatch({
    vtext <- read_parquet(VISUAL_TEXT_PATH)

    universe <- vtext |>
      filter(!is.na(tech_group),
             !tech_group %in% c("Other", "Other Clean", "Other Fossil"),
             energy_group %in% ENERGY_LEVELS,
             process_type %in% c("EA", "EIS")) |>
      count(tech_group, energy_group, name = "n_projects") |>
      mutate(
        energy_group = factor(energy_group, levels = ENERGY_LEVELS),
        tech_group   = fct_reorder(tech_group, n_projects, .fun = sum)
      )

    if (nrow(universe) == 0) {
      message("fig12 skipped: no EA/EIS rows after filtering.")
    } else {
      ggplot(universe, aes(x = tech_group, y = n_projects, fill = energy_group)) +
        geom_col(width = 0.8, alpha = 0.7) +
        geom_text(aes(label = scales::comma(n_projects)),
                  hjust = -0.15, size = 3, color = catf_navy) +
        scale_fill_manual(values = c("Decarbonization" = catf_navy,
                                     "Fossil Fuel"     = "#7B241C"),
                          name = NULL) +
        scale_y_continuous(labels = scales::comma,
                           expand = expansion(mult = c(0, 0.18))) +
        coord_flip() +
        labs(x = NULL, y = "Projects in Visual Analysis Universe",
             title = "Visual Impact Analysis Universe by Technology",
             subtitle = "EA and EIS only; sorted by total project count",
             caption = DATA_CAPTION) +
        theme_catf() +
        theme(legend.position = "bottom")
      save_fig("fig12_visual_project_counts.png", height = 8)
    }
  }, error = function(e) {
    message(sprintf("fig12 failed: %s", conditionMessage(e)))
  })
}

# ---------------------------------------------------------------------------
# fig13 — Word cloud 2×2 grid (EA/EIS × Decarb/Fossil)
# ---------------------------------------------------------------------------
# Per-cell TF-IDF vs. all-other-cells-combined. Wraps the whole block in
# tryCatch so a missing ggwordcloud / tidytext / patchwork install doesn't
# prevent the rest of the script from loading.
if (!VISUAL_TEXT_AVAILABLE) {
  message("fig13 skipped: projects_visual_text.parquet not found.")
} else {
  tryCatch({
    suppressPackageStartupMessages({
      library(tidytext)
      library(ggwordcloud)
      library(patchwork)
    })

    if (!exists("vtext")) vtext <- read_parquet(VISUAL_TEXT_PATH)

    # NEPA stopword list — covers NEPA boilerplate, visual-section universal
    # terms, agency jargon, and geographic/measurement fillers. Terms that
    # appear equally in all four cells have near-zero TF-IDF and just add noise.
    nepa_stop <- c(
      # NEPA process boilerplate
      "project", "alternative", "alternatives", "action", "actions",
      "proposed", "would", "may", "shall", "must", "could", "might",
      "area", "areas", "site", "sites", "section", "sections",
      "appendix", "table", "figure", "page", "pages", "see", "also",
      "et", "al", "e.g", "i.e", "etc", "u.s", "us",
      "impact", "impacts", "effect", "effects", "affect", "affects",
      "resource", "resources", "environmental", "environment",
      "analysis", "analyses", "review", "assessment",
      "agency", "agencies", "federal", "department", "bureau", "office",
      "blm", "doe", "nepa", "ea", "eis", "bor", "fws", "usfs", "army",
      "page_break", "pagebreak", "draft", "final", "deis", "feis",
      "applicant", "operator", "lessee", "permittee",
      # visual-section universal terms (appear in all cells equally)
      "visual", "scenic", "landscape", "aesthetics", "aesthetic",
      "viewshed", "view", "views", "scenery", "character",
      # BLM VRM framework jargon (splits cells by admin framework, not impact type)
      "vrm", "kop", "class", "lands", "land", "management",
      "plan", "plans", "planning",
      # measurement and geographic fillers
      "acres", "miles", "mile", "feet", "foot", "percent",
      "road", "roads", "surface", "adjacent", "within", "along",
      "near", "around", "north", "south", "east", "west",
      "northern", "southern", "eastern", "western",
      "county", "state", "national", "public",
      "located", "location", "locations",
      # project lifecycle boilerplate
      "construction", "operation", "operations", "facilities", "facility",
      "development", "activities", "activity", "during", "after", "before",
      "long", "term", "short", "existing", "new", "proposed",
      # common qualifiers that span all groups
      "significant", "less", "greater", "high", "low", "level",
      "potential", "potentially", "associated", "result", "results",
      "expected", "anticipated", "approximately", "generally", "typically",
      "water", "vegetation", "habitat", "species", "soil",
      # process verbs
      "considered", "evaluated", "identified", "determined", "described",
      "include", "includes", "including", "require", "requires",
      # Roman numerals -- BLM VRM class designations cause ii/iii/iv to dominate
      "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii"
    )

    wc_text <- vtext |>
      filter(energy_group %in% ENERGY_LEVELS,
             process_type %in% c("EA", "EIS"),
             !is.na(visual_analysis_text),
             nchar(visual_analysis_text) > 100) |>
      mutate(cell = energy_group) |>
      select(project_id, cell, text = visual_analysis_text)

    if (nrow(wc_text) == 0) {
      message("fig13 skipped: no rows after filtering visual_text.")
    } else {
      # Project-level bigram TF-IDF aggregated to cell level.
      # Keeps one TF-IDF score per (project, bigram), then sums across projects
      # in a cell and requires n_projects >= 10. This prevents any single large
      # project from dominating the cell vocabulary with proper nouns / location names.
      wc_tokens_raw <- wc_text |>
        tidytext::unnest_ngrams(bigram, text, n = 2) |>
        tidyr::separate(bigram, c("w1", "w2"), sep = " ", remove = TRUE) |>
        filter(
          !w1 %in% tidytext::stop_words$word, !w2 %in% tidytext::stop_words$word,
          !w1 %in% nepa_stop, !w2 %in% nepa_stop,
          str_detect(w1, "^[a-z]{3,}$"), str_detect(w2, "^[a-z]{3,}$"),
          !str_detect(w1, "^[bcdfghjklmnpqrstvwxyz]{3,}$"),
          !str_detect(w2, "^[bcdfghjklmnpqrstvwxyz]{3,}$")
        ) |>
        tidyr::unite(bigram, w1, w2, sep = " ") |>
        count(project_id, cell, bigram)

      wc_tfidf <- wc_tokens_raw |>
        tidytext::bind_tf_idf(bigram, project_id, n) |>
        group_by(cell, bigram) |>
        summarise(
          tf_idf     = sum(tf_idf),
          n_projects = n_distinct(project_id),
          .groups    = "drop"
        ) |>
        filter(n_projects >= 10) |>
        rename(word = bigram) |>
        group_by(cell) |>
        slice_max(tf_idf, n = 45, with_ties = FALSE) |>
        ungroup()



      make_wc_panel <- function(cell_name, panel_color) {
        d <- wc_tfidf |> filter(cell == cell_name)
        if (nrow(d) == 0) {
          return(ggplot() +
                   labs(title = cell_name, subtitle = "(no terms)") +
                   theme_void() +
                   theme(plot.title = element_text(face = "bold",
                                                   color = panel_color,
                                                   hjust = 0.5)))
        }
        ggplot(d, aes(label = word, size = tf_idf)) +
          ggwordcloud::geom_text_wordcloud(color = panel_color,
                                           rm_outside = TRUE) +
          scale_size_area(max_size = 14) +
          labs(title = cell_name) +
          theme_minimal(base_size = 11) +
          theme(plot.title = element_text(face = "bold", color = panel_color,
                                          hjust = 0.5, size = rel(1.1)),
                panel.grid  = element_blank(),
                axis.text   = element_blank(),
                axis.title  = element_blank(),
                axis.ticks  = element_blank())
      }

      p_d <- make_wc_panel("Decarbonization", catf_navy)
      p_f <- make_wc_panel("Fossil Fuel",     "#7B241C")

      wc_grid <- (p_d | p_f) +
        patchwork::plot_annotation(
          title    = "Distinguishing Visual-Impact Vocabulary: Decarbonization vs. Fossil Fuel",
          subtitle = "Top 45 TF-IDF bigrams per portfolio (each portfolio vs. the other)",
          caption  = DATA_CAPTION,
          theme    = theme_catf()
        )

      ggsave(file.path(OUTPUT_DIR, "fig13_wordcloud_grid.png"),
             wc_grid, width = 12, height = 6, dpi = 300)
    }
  }, error = function(e) {
    message(sprintf("fig13 skipped: %s", conditionMessage(e)))
  })
}

# ---------------------------------------------------------------------------
# fig14 — Topic prevalence by group
# ---------------------------------------------------------------------------
# Top 10 topics from visual_topic_summary.parquet; side-by-side decarb/fossil
# bars using the chosen model rows.
if (!file.exists(VISUAL_TOPIC_SUMMARY_PATH)) {
  message("fig14 skipped: visual_topic_summary.parquet not found.")
} else {
  tryCatch({
    topic_summary <- read_parquet(VISUAL_TOPIC_SUMMARY_PATH)

    # Pick the "chosen" model rows (prefer NMF; fall back to whatever single
    # model is present). The Python pipeline writes both NMF and (optionally)
    # BERTopic rows distinguished by `model`.
    chosen_model <- if ("model" %in% names(topic_summary)) {
      models_present <- unique(topic_summary$model)
      if ("nmf" %in% models_present) "nmf" else models_present[[1]]
    } else NA_character_

    topic_chosen <- if (!is.na(chosen_model)) {
      topic_summary |> filter(model == chosen_model)
    } else topic_summary

    # Pick top 10 by total project count; apply interpretive labels
    top10 <- topic_chosen |>
      arrange(desc(n_total)) |>
      slice_head(n = 10) |>
      mutate(
        auto_label = ifelse(is.na(label) | label == "", paste0("topic_", topic_id), label),
        label      = dplyr::coalesce(TOPIC_INTERP[auto_label], auto_label)
      )

    if (nrow(top10) == 0 ||
        !all(c("n_decarb", "n_fossil") %in% names(top10))) {
      message("fig14 skipped: topic_summary missing n_decarb/n_fossil columns.")
    } else {
      topic_long <- top10 |>
        select(topic_id, label, n_decarb, n_fossil) |>
        pivot_longer(c(n_decarb, n_fossil),
                     names_to = "energy_group", values_to = "n") |>
        mutate(
          energy_group = recode(energy_group,
                                n_decarb = "Decarbonization",
                                n_fossil = "Fossil Fuel"),
          energy_group = factor(energy_group, levels = ENERGY_LEVELS),
          label        = fct_reorder(label, n, .fun = sum)
        )

      topic_fill <- c("Decarbonization" = catf_navy,
                      "Fossil Fuel"     = "#7B241C")

      ggplot(topic_long, aes(x = label, y = n, fill = energy_group)) +
        geom_col(position = position_dodge(width = 0.8), width = 0.75, alpha = 0.7) +
        geom_text(aes(label = scales::comma(n)),
                  position = position_dodge(width = 0.8),
                  hjust = -0.15, size = 2.8, color = catf_navy) +
        scale_fill_manual(values = topic_fill) +
        scale_y_continuous(labels = scales::comma,
                           expand = expansion(mult = c(0, 0.18))) +
        coord_flip() +
        labs(x = NULL, y = "Projects with Topic", fill = NULL,
             title = "Top Visual-Impact Topics by Energy Category",
             subtitle = "NMF topic model; interpretive labels assigned from top discriminating terms",
             caption = DATA_CAPTION) +
        theme_catf() +
        theme(legend.position = "bottom")
      save_fig("fig14_topic_prevalence.png", height = 8)
    }

    # Also export the topic summary as CSV for reporting
    write.csv(topic_chosen,
              file.path(OUTPUT_DIR, "visual_topic_summary_table.csv"),
              row.names = FALSE)
  }, error = function(e) {
    message(sprintf("fig14 skipped: %s", conditionMessage(e)))
  })
}

# ---------------------------------------------------------------------------
# fig18 — Framing comparison (CEQ-axis ratios)
# ---------------------------------------------------------------------------
if (!VISUAL_FRAMING_AVAILABLE) {
  message("fig18 skipped: visual_framing.parquet not found.")
} else {
  tryCatch({
    framing <- read_parquet(VISUAL_FRAMING_PATH)

    # Need energy_group + process_type — join from projects_visual_text if not
    # already present, otherwise from df.
    framing_join_cols <- intersect(
      c("energy_group", "process_type"), names(framing)
    )
    if (length(framing_join_cols) < 2) {
      base_lookup <- if (VISUAL_TEXT_AVAILABLE) {
        if (!exists("vtext")) vtext <- read_parquet(VISUAL_TEXT_PATH)
        vtext |> select(project_id, energy_group, process_type)
      } else {
        df |> select(project_id, energy_group, process_type)
      }
      framing <- framing |> left_join(base_lookup, by = "project_id")
    }

    # Compute project-level adversity ratio if not already provided
    if (!"adversity_ratio" %in% names(framing)) {
      # Column may be named adv_neg/adv_pos (short) or adv_neg_count/adv_pos_count (long)
      neg_col <- if ("adv_neg" %in% names(framing)) "adv_neg" else "adv_neg_count"
      pos_col <- if ("adv_pos" %in% names(framing)) "adv_pos" else "adv_pos_count"
      framing <- framing |>
        mutate(
          adversity_ratio = ifelse(
            (.data[[neg_col]] + .data[[pos_col]]) > 0,
            .data[[neg_col]] / (.data[[neg_col]] + .data[[pos_col]]),
            NA_real_
          )
        )
    }

    framing_axes <- framing |>
      filter(energy_group %in% ENERGY_LEVELS,
             process_type %in% c("EA", "EIS")) |>
      select(project_id, energy_group,
             any_of(c("significance_ratio", "adversity_ratio", "mitigation_ratio"))) |>
      pivot_longer(any_of(c("significance_ratio", "adversity_ratio",
                            "mitigation_ratio")),
                   names_to = "axis", values_to = "value") |>
      filter(!is.na(value)) |>
      group_by(axis, energy_group) |>
      summarise(mean_ratio = mean(value, na.rm = TRUE),
                n_projects = n(), .groups = "drop") |>
      mutate(
        axis = factor(recode(axis,
                             significance_ratio = "Significance (high / total)",
                             adversity_ratio    = "Adversity (negative / total)",
                             mitigation_ratio   = "Mitigation strength (strong / total)"),
                      levels = c("Significance (high / total)",
                                 "Adversity (negative / total)",
                                 "Mitigation strength (strong / total)")),
        energy_group = factor(energy_group, levels = ENERGY_LEVELS)
      )

    if (nrow(framing_axes) == 0) {
      message("fig18 skipped: no framing rows after filtering.")
    } else {
      fram_fill <- c(
        "Decarbonization" = catf_navy,
        "Fossil Fuel"     = "#7B241C"
      )

      ggplot(framing_axes,
             aes(x = energy_group, y = mean_ratio, fill = energy_group)) +
        geom_col(width = 0.6, alpha = 0.7) +
        geom_text(aes(label = scales::percent(mean_ratio, accuracy = 1)),
                  vjust = -0.4, size = 3.5, color = catf_navy) +
        scale_fill_manual(values = fram_fill, guide = "none") +
        scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                           expand = expansion(mult = c(0, 0.18))) +
        facet_wrap(~ axis, ncol = 1, scales = "free_y") +
        labs(x = NULL, y = "Mean project-level ratio",
             title = "Visual-Impact Framing: Decarbonization vs. Fossil Fuel",
             subtitle = "CEQ-axis lexicon ratios averaged across all EA/EIS projects",
             caption = paste0(
               DATA_CAPTION, "\n",
               "Significance = high / (high + low); Adversity = negative / (negative + positive); ",
               "Mitigation = strong / (strong + weak)."
             )) +
        theme_catf()
      save_fig("fig18_visual_framing.png", height = 10)
    }
  }, error = function(e) {
    message(sprintf("fig18 skipped: %s", conditionMessage(e)))
  })
}


# ---------------------------------------------------------------------------
# Framing examples table -- one high/low sentence per measure
# ---------------------------------------------------------------------------
tryCatch({
  if (!exists("framing")) framing <- read_parquet(VISUAL_FRAMING_PATH)
  if (!exists("vtext"))   vtext   <- read_parquet(VISUAL_TEXT_PATH)

  fd <- framing |>
    left_join(vtext |> select(project_id, visual_text_clean), by = "project_id") |>
    filter(!is.na(visual_text_clean), nchar(visual_text_clean) > 200)

  if (!"adversity_ratio" %in% names(fd)) {
    neg_col <- if ("adv_neg" %in% names(fd)) "adv_neg" else "adv_neg_count"
    pos_col <- if ("adv_pos" %in% names(fd)) "adv_pos" else "adv_pos_count"
    fd <- fd |> mutate(
      adversity_ratio = ifelse(
        (.data[[neg_col]] + .data[[pos_col]]) > 0,
        .data[[neg_col]] / (.data[[neg_col]] + .data[[pos_col]]),
        NA_real_
      )
    )
  }

  pick_sent <- function(text, pattern, max_chars = 700) {
    if (is.na(text)) return(NA_character_)
    # Split on sentence boundaries AND newlines to handle paragraph-style text
    raw <- unlist(str_split(text, "(?<=[.!?])\\s+|\\n+"))
    sents <- str_squish(raw)
    sents <- sents[nchar(sents) >= 50]
    for (s in sents) {
      if (str_detect(tolower(s), pattern) && nchar(s) <= max_chars)
        return(str_trunc(s, 350))
    }
    NA_character_
  }

  # Pre-filter fd by energy_group before picking examples; avoids scoping
  # issues with closures and tibble subsetting inside get_ex.
  get_ex <- function(df, ratio_col, high, kw) {
    d <- as.data.frame(df)
    d <- if (high) d[order(-d[[ratio_col]], na.last = TRUE), ] else d[order(d[[ratio_col]], na.last = TRUE), ]
    d <- d[!is.na(d[[ratio_col]]), ]
    for (i in seq_len(min(nrow(d), 80))) {
      s <- pick_sent(d$visual_text_clean[i], kw)
      if (!is.na(s)) return(s)
    }
    NA_character_
  }

  mk_rows <- function(eg) {
    fd_eg <- fd[which(fd$energy_group == eg), ]
    tibble::tribble(
      ~energy_group, ~Measure, ~Framing, ~`Sample text`,
      eg, "Significance", "High",
      get_ex(fd_eg, "significance_ratio", TRUE,
             "\\bsignificant\\b|\\bmajor\\b|\\bsubstantial\\b|\\bsevere\\b"),
      eg, "Significance", "Low",
      get_ex(fd_eg, "significance_ratio", FALSE,
             "less than significant|not significant|\\bnegligible\\b|\\bminor\\b"),
      eg, "Adversity",    "Negative",
      get_ex(fd_eg, "adversity_ratio", TRUE,
             "adverse|detrimental|degrad|harm"),
      eg, "Adversity",    "Positive",
      get_ex(fd_eg, "adversity_ratio", FALSE,
             "beneficial|enhance|improve|no adverse|no effect"),
      eg, "Mitigation",   "Strong",
      get_ex(fd_eg, "mitigation_ratio", TRUE,
             "shall|will install|required to|committed to|painted"),
      eg, "Mitigation",   "Weak",
      get_ex(fd_eg, "mitigation_ratio", FALSE,
             "residual|unavoidable|cannot be fully|cannot fully|remain")
    )
  }

  framing_ex <- dplyr::bind_rows(
    mk_rows("Decarbonization"),
    mk_rows("Fossil Fuel")
  )

  write.csv(framing_ex, file.path(OUTPUT_DIR, "framing_examples.csv"), row.names = FALSE)
  message("framing_examples.csv written")
}, error = function(e) message(sprintf("framing examples skipped: %s", conditionMessage(e))))
# ---------------------------------------------------------------------------
# fig19 — Section length boxplot (heading-anchored sections only)
# ---------------------------------------------------------------------------
if (!VISUAL_SECTIONS_AVAILABLE) {
  message("fig19 skipped: visual_sections.parquet not found.")
} else {
  tryCatch({
    sections <- read_parquet(VISUAL_SECTIONS_PATH)

    sec_box <- sections |>
      filter(extraction_method == "heading_anchored",
             energy_group %in% ENERGY_LEVELS,
             !is.na(tech_group),
             !tech_group %in% c("Other", "Other Clean", "Other Fossil"),
             process_type %in% c("EA", "EIS"),
             !is.na(n_words),
             n_words > 0) |>
      mutate(
        energy_group = factor(energy_group, levels = ENERGY_LEVELS),
        tech_group   = fct_reorder(tech_group, n_words, .fun = median)
      )

    if (nrow(sec_box) == 0) {
      message("fig19 skipped: no heading-anchored sections after filtering.")
    } else {
      ggplot(sec_box, aes(x = tech_group, y = n_words, fill = energy_group)) +
        geom_boxplot(outlier.size = 0.4, alpha = 0.7) +
        scale_fill_manual(values = c("Decarbonization" = catf_navy,
                                     "Fossil Fuel"     = "#7B241C"),
                          name = NULL) +
        scale_y_log10(labels = scales::comma,
                      breaks = c(100, 500, 1000, 5000, 10000)) +
        coord_flip() +
        labs(x = NULL, y = "Section length (words, log scale)",
             title = "Visual Section Length by Technology: Decarbonization vs. Fossil Fuel",
             subtitle = "Heading-anchored sections only; sorted by median; blue = Decarbonization, red = Fossil Fuel",
             caption = DATA_CAPTION) +
        theme_catf() +
        theme(legend.position = "bottom")
      save_fig("fig19_visual_section_length.png", height = 9)
    }
  }, error = function(e) {
    message(sprintf("fig19 skipped: %s", conditionMessage(e)))
  })
}

# ---------------------------------------------------------------------------
# fig21 -- VRM element-level contrast rating distribution
# ---------------------------------------------------------------------------
if (!VRM_ELEMENTS_AVAILABLE) {
  message("fig21 skipped: vrm_elements.parquet not found.")
} else {
  tryCatch({
    vrm_el <- read_parquet(VRM_ELEMENTS_PATH)
    vrm_el <- vrm_el |>
      dplyr::filter(
        !is.na(rating), !is.na(element),
        energy_group %in% c("Decarbonization", "Fossil Fuel")
      ) |>
      dplyr::mutate(
        element      = stringr::str_to_title(element),
        rating       = factor(rating, levels = c("None", "Weak", "Moderate", "Strong")),
        energy_group = factor(energy_group, levels = c("Decarbonization", "Fossil Fuel"))
      )

    if (nrow(vrm_el) < 10) {
      message("fig21 skipped: too few VRM element rows.")
    } else {
      vrm_pct <- vrm_el |>
        dplyr::group_by(energy_group, element, rating) |>
        dplyr::summarise(n_projects = dplyr::n_distinct(project_id), .groups = "drop") |>
        dplyr::group_by(energy_group, element) |>
        dplyr::mutate(
          n_total = sum(n_projects),
          pct     = 100 * n_projects / n_total
        ) |>
        dplyr::ungroup()

      rating_colors <- c(
        "None"     = "#D3D3D3",
        "Weak"     = "#8AB7E9",
        "Moderate" = "#0047BB",
        "Strong"   = "#012169"
      )

      p21 <- ggplot(vrm_pct, aes(x = element, y = pct, fill = rating)) +
        geom_col(position = "stack", width = 0.7) +
        facet_wrap(~ energy_group, ncol = 2) +
        scale_fill_manual(values = rating_colors, name = "Contrast Rating") +
        scale_y_continuous(
          labels = scales::label_percent(scale = 1),
          expand = expansion(mult = c(0, 0.05))
        ) +
        labs(
          title    = "VRM Element-Level Contrast Ratings by Energy Category",
          subtitle = "Share of projects with each contrast rating per BLM VRM element",
          x = "VRM Element", y = "% of Projects"
        ) +
        theme_bw(base_size = 11) +
        theme(
          axis.text.x        = element_text(angle = 30, hjust = 1),
          legend.position    = "right",
          panel.grid.major.x = element_blank(),
          strip.background   = element_rect(fill = catf_navy),
          strip.text         = element_text(color = "white", face = "bold")
        )

      print(p21)
      save_fig("fig21_vrm_elements.png", height = 6, width = 10)
      message("fig21: VRM element-level ratings chart written.")
    }
  }, error = function(e) {
    message(sprintf("fig21 skipped: %s", conditionMessage(e)))
  })
}

# ---------------------------------------------------------------------------
# Examples table — gt-rendered + CSV export
# ---------------------------------------------------------------------------
if (!VISUAL_EXAMPLES_AVAILABLE) {
  message("Examples table skipped: visual_examples.parquet not found.")
} else {
  tryCatch({
    suppressPackageStartupMessages({
      library(gt)
    })

    examples <- read_parquet(VISUAL_EXAMPLES_PATH)

    # Defensive: coerce list-valued columns (e.g. agency JSON arrays) to scalar
    # strings before passing to gt.
    flatten_to_chr <- function(x) {
      if (is.list(x)) {
        vapply(x, function(v) {
          if (is.null(v) || length(v) == 0) NA_character_
          else paste(as.character(v), collapse = "; ")
        }, character(1))
      } else if (is.character(x)) {
        # In case lead_agency is a JSON-encoded array string
        vapply(x, function(v) {
          parsed <- parse_json_first(v)
          if (is.na(parsed)) v else parsed
        }, character(1))
      } else {
        as.character(x)
      }
    }

    if ("lead_agency" %in% names(examples)) {
      examples$lead_agency <- flatten_to_chr(examples$lead_agency)
    }

    examples_tbl <- examples |>
      filter(energy_group %in% ENERGY_LEVELS) |>
      mutate(
        energy_group   = factor(energy_group, levels = ENERGY_LEVELS),
        project_title  = stringr::str_trunc(as.character(project_title), 60),
        excerpt        = paste0("“",
                                stringr::str_trunc(as.character(excerpt), 300),
                                "”"),
        framing_summary = ifelse(is.na(framing_summary) | framing_summary == "",
                                  "-", framing_summary)
      ) |>
      arrange(energy_group, tech_group) |>
      select(any_of(c("energy_group", "process_type", "tech_group",
                      "lead_agency", "project_title", "excerpt",
                      "framing_summary")))

    # Write raw CSV regardless of whether gt succeeds
    write.csv(examples_tbl,
              file.path(OUTPUT_DIR, "visual_examples_table.csv"),
              row.names = FALSE)

    if (nrow(examples_tbl) == 0) {
      message("Examples table HTML skipped: no rows after filtering.")
    } else {
      gt_tbl <- examples_tbl |>
        gt::gt(groupname_col = "energy_group") |>
        gt::cols_label(
          process_type    = "Process",
          tech_group      = "Technology",
          lead_agency     = "Lead Agency",
          project_title   = "Project",
          excerpt         = "Excerpt",
          framing_summary = "Framing"
        ) |>
        gt::cols_align(align = "left") |>
        gt::tab_style(
          style    = gt::cell_text(weight = "bold"),
          locations = gt::cells_column_labels()
        ) |>
        gt::tab_style(
          style = list(
            gt::cell_fill(color = catf_navy),
            gt::cell_text(color = "white", weight = "bold", size = gt::px(13))
          ),
          locations = gt::cells_row_groups()
        ) |>
        gt::tab_style(
          style    = gt::cell_fill(color = "#f8f9fa"),
          locations = gt::cells_body(rows = seq(2, nrow(examples_tbl), by = 2))
        ) |>
        gt::tab_options(
          table.font.size   = gt::px(12),
          data_row.padding  = gt::px(8),
          row_group.padding = gt::px(6),
          table.font.names  = "Helvetica"
        ) |>
        gt::tab_header(
          title    = "Illustrative Visual-Impact Excerpts",
          subtitle = "Heading-anchored sections, 150–500 words; grouped by energy category"
        )

      gt::gtsave(gt_tbl,
                 filename = file.path(OUTPUT_DIR, "visual_examples_table.html"))
    }
  }, error = function(e) {
    message(sprintf("Examples table skipped: %s", conditionMessage(e)))
  })
}

# ---------------------------------------------------------------------------
# fig14b — Topic term weights (companion to fig14)
# Faceted lollipop chart: top 10 NMF terms per topic, sized by component weight.
# ---------------------------------------------------------------------------
if (!file.exists(VISUAL_TOPIC_TERMS_PATH)) {
  message("fig14b skipped: visual_topic_terms_detail.csv not found.")
} else {
  tryCatch({
    topic_terms <- read.csv(VISUAL_TOPIC_TERMS_PATH, stringsAsFactors = FALSE) |>
      filter(model == "nmf", rank <= 10)

    # Join in human-readable label from topic summary; apply interpretive labels
    if (file.exists(VISUAL_TOPIC_SUMMARY_PATH)) {
      ts_labels <- read_parquet(VISUAL_TOPIC_SUMMARY_PATH) |>
        filter(model == "nmf") |>
        select(topic_id, label, n_total) |>
        mutate(
          auto_label = label,
          label      = dplyr::coalesce(TOPIC_INTERP[auto_label], auto_label)
        )
      topic_terms <- topic_terms |>
        left_join(ts_labels, by = "topic_id") |>
        mutate(panel_label = paste0("Topic ", topic_id, ": ", label,
                                    " (n=", scales::comma(n_total), ")"))
    } else {
      topic_terms <- topic_terms |>
        mutate(panel_label = paste0("Topic ", topic_id, ": ", topic_label))
    }

    topic_terms <- topic_terms |>
      mutate(
        term = fct_reorder(term, weight),
        panel_label = factor(panel_label)
      )

    ggplot(topic_terms, aes(x = weight, y = term)) +
      geom_segment(aes(xend = 0, yend = term), color = "grey70", linewidth = 0.5) +
      geom_point(aes(size = weight), color = catf_dark_blue, alpha = 0.85) +
      scale_size_continuous(range = c(1.5, 5), guide = "none") +
      facet_wrap(~ panel_label, scales = "free_y", ncol = 2) +
      labs(
        x = "NMF component weight (higher = more characteristic of topic)",
        y = NULL,
        title = "Top 10 Terms per Visual-Impact Topic (NMF)",
        subtitle = "Term weight reflects how strongly each word characterises its topic relative to others",
        caption = DATA_CAPTION
      ) +
      theme_catf() +
      theme(
        strip.text     = element_text(size = 8, face = "bold", hjust = 0),
        axis.text.y    = element_text(size = 8),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(color = "grey90")
      )

    save_fig("fig14b_topic_terms.png", width = 13, height = 10)
  }, error = function(e) {
    message(sprintf("fig14b skipped: %s", conditionMessage(e)))
  })
}

# ---------------------------------------------------------------------------
# fig14d — NMF elbow / k-selection validation
# ---------------------------------------------------------------------------
VISUAL_TOPIC_ELBOW_PATH <- file.path(OUTPUT_DIR, "nmf_elbow_data.csv")
if (!file.exists(VISUAL_TOPIC_ELBOW_PATH)) {
  message("fig14d skipped: nmf_elbow_data.csv not found.")
} else {
  tryCatch({
    elbow <- read.csv(VISUAL_TOPIC_ELBOW_PATH) |>
      mutate(k = as.integer(k))

    chosen_k <- 4L

    # Normalise reconstruction error to 0–1 range for dual-axis clarity
    err_min <- min(elbow$reconstruction_error)
    err_max <- max(elbow$reconstruction_error)
    elbow <- elbow |>
      mutate(
        recon_norm     = (reconstruction_error - err_min) / (err_max - err_min),
        sharpness_norm = (mean_sharpness - min(mean_sharpness)) /
                         (max(mean_sharpness) - min(mean_sharpness))
      )

    ggplot(elbow, aes(x = k)) +
      geom_line(aes(y = recon_norm, colour = "Reconstruction error"), linewidth = 1) +
      geom_point(aes(y = recon_norm, colour = "Reconstruction error"), size = 3) +
      geom_line(aes(y = sharpness_norm, colour = "Topic sharpness"), linewidth = 1,
                linetype = "dashed") +
      geom_point(aes(y = sharpness_norm, colour = "Topic sharpness"), size = 3) +
      geom_vline(xintercept = chosen_k, linetype = "dotted",
                 colour = catf_navy, linewidth = 0.8) +
      annotate("text", x = chosen_k + 0.15, y = 0.85,
               label = sprintf("Chosen k = %d", chosen_k),
               hjust = 0, size = 3.2, colour = catf_navy) +
      scale_colour_manual(
        values = c("Reconstruction error" = catf_dark_blue,
                   "Topic sharpness"      = "#C0392B"),
        name = NULL
      ) +
      scale_x_continuous(breaks = elbow$k) +
      scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
      labs(
        x       = "Number of topics (k)",
        y       = "Normalised score",
        title   = "NMF Topic-Count Validation (Elbow Analysis)",
        subtitle = paste0(
          "Reconstruction error drops sharply k=2→3, flattens k=4+. ",
          "Topic sharpness (term-weight std) falls monotonically — ",
          "more topics dilute coherence without improving fit."
        ),
        caption = DATA_CAPTION
      ) +
      theme_catf() +
      theme(legend.position = "bottom")

    save_fig("fig14d_nmf_elbow.png", height = 5, width = 8)
    message("fig14d: NMF elbow figure written.")
  }, error = function(e) {
    message(sprintf("fig14d skipped: %s", conditionMessage(e)))
  })
}

# ---------------------------------------------------------------------------
# fig14c — Topic excerpt table (companion text examples)
# ---------------------------------------------------------------------------
if (!file.exists(VISUAL_TOPIC_EXCERPTS_PATH)) {
  message("fig14c skipped: visual_topic_excerpts.csv not found.")
} else {
  tryCatch({
    excerpts <- read.csv(VISUAL_TOPIC_EXCERPTS_PATH, stringsAsFactors = FALSE)

    # Truncate excerpts to ~250 chars for display
    excerpts <- excerpts |>
      mutate(
        excerpt_short = ifelse(nchar(excerpt) > 250,
                               paste0(substr(excerpt, 1, 247), "..."),
                               excerpt),
        cell = paste0(energy_group, " | ", tech_group)
      ) |>
      select(topic_id, topic_label, cell, excerpt_short)

    write.csv(excerpts,
              file.path(OUTPUT_DIR, "visual_topic_excerpts_table.csv"),
              row.names = FALSE)

    tryCatch({
      suppressPackageStartupMessages(library(gt))
      gt_exc <- excerpts |>
        rename(
          `Topic` = topic_label,
          `Energy · Technology` = cell,
          `Example text` = excerpt_short
        ) |>
        select(-topic_id) |>
        gt::gt(groupname_col = "Topic") |>
        gt::cols_align(align = "left") |>
        gt::tab_style(
          style    = gt::cell_text(weight = "bold"),
          locations = gt::cells_row_groups()
        ) |>
        gt::tab_options(
          table.font.size   = gt::px(11),
          data_row.padding  = gt::px(6),
          row_group.padding = gt::px(5)
        ) |>
        gt::tab_header(
          title    = "Visual-Impact Topic Examples",
          subtitle = "Three representative excerpts per NMF topic"
        )
      gt::gtsave(gt_exc, file.path(OUTPUT_DIR, "visual_topic_excerpts_table.html"))
      message("fig14c: wrote visual_topic_excerpts_table.html")
    }, error = function(e) {
      message(sprintf("fig14c HTML skipped (gt error): %s", conditionMessage(e)))
    })
  }, error = function(e) {
    message(sprintf("fig14c skipped: %s", conditionMessage(e)))
  })
}

cat("  Section 4 done.\n")


# ===========================================================================
# SECTION 5: GEOTHERMAL VS. OIL & GAS
# ===========================================================================
cat("\n--- Section 5: Geothermal vs. Oil & Gas ---\n")
if (!GEO_OG_AVAILABLE) {
  message("Section 5 skipped: projects_geothermal_og.parquet not found.")
} else {

# Fig 15 — CE/EA/EIS rates: Geothermal vs. Oil & Gas ----
# Collapse land-based and offshore into "Oil & Gas" (only 5 offshore, all in Alaska)
rate_compare <- geo_og |>
  mutate(tech_group = if_else(
    tech_group %in% c("Land-based Oil & Gas", "Offshore Oil & Gas"),
    "Oil & Gas", tech_group
  )) |>
  filter(!is.na(process_type)) |>
  count(tech_group, process_type) |>
  group_by(tech_group) |>
  mutate(pct = n / sum(n)) |>
  ungroup() |>
  mutate(
    fill_key = factor(
      paste0(tech_group, " ", process_type),
      levels = c("Geothermal EIS", "Geothermal EA", "Geothermal CE",
                 "Oil & Gas EIS",  "Oil & Gas EA",  "Oil & Gas CE")
    ),
    process_type = factor(process_type, levels = c("EIS", "EA", "CE"))
  )

rate_compare_totals <- rate_compare |>
  group_by(tech_group) |>
  summarise(total_n = sum(n), .groups = "drop")

fig15_colors <- c(
  "Geothermal EIS" = catf_navy,
  "Geothermal EA"  = catf_dark_blue,
  "Geothermal CE"  = catf_light_blue,
  "Oil & Gas EIS"  = "#7B241C",
  "Oil & Gas EA"   = "#E74C3C",
  "Oil & Gas CE"   = "#F1948A"
)

fig15_labels <- c(
  "Geothermal EIS" = "EIS", "Geothermal EA" = "EA", "Geothermal CE" = "CE",
  "Oil & Gas EIS"  = "EIS", "Oil & Gas EA"  = "EA", "Oil & Gas CE"  = "CE"
)

ggplot(rate_compare, aes(x = tech_group, y = pct, fill = fill_key)) +
  geom_col(position = position_stack(reverse = TRUE)) +
  geom_text(aes(label = ifelse(pct >= 0.05, scales::percent(pct, accuracy = 1), "")),
            position = position_stack(reverse = TRUE, vjust = 0.5),
            color = "white", size = 3.5, fontface = "bold") +
  scale_fill_manual(
    values = fig15_colors,
    labels = fig15_labels,
    guide  = guide_legend(
      title = "Review Type",
      nrow  = 2,
      byrow = TRUE,
      override.aes = list(
        fill = c(catf_navy, catf_dark_blue, catf_light_blue, "#7B241C", "#E74C3C", "#F1948A")
      )
    )
  ) +
  geom_text(data = rate_compare_totals,
            aes(x = tech_group, y = 1.01,
                label = paste0("n = ", scales::comma(total_n))),
            hjust = 0, size = 3.5, color = catf_navy,
            fontface = "plain", inherit.aes = FALSE) +
  scale_y_continuous(labels = percent_format(),
                     expand = expansion(mult = c(0, 0.15))) +
  coord_flip() +
  labs(x = NULL, y = "Share of Projects", fill = "Review Type",
       title = "NEPA Review Type: Geothermal vs. Oil & Gas",
       caption = DATA_CAPTION) +
  theme_catf() +
  theme(legend.position = "bottom")
save_fig("fig15_geo_og_rates.png")

# Fig 16 — Geothermal vs. Oil & Gas share by state (all states) ----
# Count all states for both tech groups (after collapsing oil & gas subtypes)
geo_og_all_states <- geo_og |>
  mutate(tech_group = if_else(
    tech_group %in% c("Land-based Oil & Gas", "Offshore Oil & Gas"),
    "Oil & Gas", tech_group
  )) |>
  explode_column("project_state") |>
  filter(!is.na(project_state), project_state != "") |>
  count(tech_group, project_state) |>
  complete(tech_group, project_state, fill = list(n = 0))

# Within-state share; drop states with no projects in either category
geo_og_state <- geo_og_all_states |>
  group_by(project_state) |>
  mutate(total = sum(n), pct = if_else(total > 0, n / total, 0)) |>
  ungroup() |>
  filter(total > 0)

# Order by geothermal share ascending → highest-geothermal state at top after coord_flip
geo_share_order <- geo_og_state |>
  filter(tech_group == "Geothermal") |>
  arrange(pct) |>
  pull(project_state)

geo_og_state <- geo_og_state |>
  mutate(
    project_state = factor(project_state, levels = geo_share_order),
    tech_group    = factor(tech_group, levels = c("Geothermal", "Oil & Gas"))
  )

fig16_colors <- c(
  "Geothermal" = catf_dark_blue,
  "Oil & Gas"  = FOSSIL_RED
)

ggplot(geo_og_state, aes(x = project_state, y = pct, fill = tech_group)) +
  geom_col(position = position_stack(reverse = TRUE)) +
  geom_text(aes(label = ifelse(pct >= 0.05, scales::percent(pct, accuracy = 1), "")),
            position = position_stack(reverse = TRUE, vjust = 0.5),
            color = "white", size = 3) +
  scale_fill_manual(values = fig16_colors) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = NULL, y = "Share of Projects", fill = NULL,
       title = "Geothermal vs. Oil & Gas Share by State",
       subtitle = "All states with geothermal or oil & gas projects; ordered by geothermal share",
       caption = paste0(
         DATA_CAPTION, "\n",
         "Oil & Gas includes both land-based and offshore projects."
       )) +
  theme_catf() +
  theme(legend.position = "bottom")
save_fig("fig16_geo_og_states.png", width = FIG_W, height = 16)

# Fig 17 — State choropleth: Geothermal share (diverging blue-purple-red) ----
# Each state colored by what fraction of its geothermal+oil&gas projects are geothermal.
# Blue = geothermal-dominant, red = oil&gas-dominant, purple = balanced (~50/50).
geo_share_map <- geo_og_state |>
  filter(tech_group == "Geothermal") |>
  select(project_state, geo_pct = pct)

# Join to us_states sf (already loaded in Section 3 geography block)
geo_map_sf <- us_states |>
  left_join(geo_share_map, by = c("NAME" = "project_state")) |>
  st_as_sf()

ggplot(geo_map_sf) +
  geom_sf(aes(fill = geo_pct), color = "white", linewidth = 0.2) +
  scale_fill_gradient2(
    low      = FOSSIL_RED,       # red  — oil & gas dominant
    mid      = "#7B68EE",        # purple — 50/50 split
    high     = catf_dark_blue,   # blue — geothermal dominant
    midpoint = 0.5,
    limits   = c(0, 1),
    labels   = percent_format(accuracy = 1),
    na.value = "grey90",
    name     = "Geothermal share\n(of geo + O&G projects)",
    guide    = guide_colorbar(
      barwidth      = unit(12, "cm"),
      barheight     = unit(0.4, "cm"),
      title.position = "top",
      title.hjust    = 0.5
    )
  ) +
  labs(
    title    = "Geothermal vs. Oil & Gas Dominance by State",
    subtitle = "Blue = geothermal-dominant  |  Red = oil & gas-dominant  |  Grey = no projects in either",
    caption  = paste0(DATA_CAPTION, "\nShare = geothermal projects ÷ (geothermal + oil & gas) within each state.")
  ) +
  theme_catf() +
  theme_void() +
  theme(
    legend.position  = "bottom",
    plot.title       = element_text(face = "bold", color = catf_navy, margin = margin(b = 6)),
    plot.subtitle    = element_text(color = catf_dark_blue, margin = margin(b = 6)),
    plot.caption     = element_text(size = rel(0.8), color = "gray50", hjust = 0),
    legend.title     = element_text(face = "bold", color = catf_navy),
    legend.text      = element_text(color = "gray30"),
    plot.margin      = margin(15, 15, 10, 10)
  )
save_fig("fig17_geo_og_state_map.png", width = 14, height = 8)

# Geothermal comparison table ----
# "Does geothermal look more like clean energy or oil/gas?"

visual_section_pct_for <- function(ids) {
  mean(vis$visual_section_found[vis$project_id %in% ids], na.rm = TRUE)
}

clean_ids  <- df$project_id[df$project_energy_type == "Clean" & !is.na(df$process_type)]
fossil_ids <- df$project_id[df$project_energy_type == "Fossil" & !is.na(df$process_type)]
geo_ids    <- geo_og$project_id[geo_og$tech_group == "Geothermal" & !is.na(geo_og$process_type)]

clean_avg <- df |>
  filter(project_energy_type == "Clean", !is.na(process_type)) |>
  summarise(
    group                      = "Clean Energy Average",
    n                          = n(),
    ce_share                   = mean(process_type == "CE"),
    blm_share                  = mean(
      map_lgl(lead_agency_harmonized,
              ~ safe_agency_match(.x, "BLM|Bureau of Land Management")),
      na.rm = TRUE
    ),
    federal_land_trigger_share = mean(nepa_trigger_primary == "federal_land", na.rm = TRUE),
    visual_section_pct         = visual_section_pct_for(clean_ids)
  )

fossil_avg <- df |>
  filter(project_energy_type == "Fossil", !is.na(process_type)) |>
  summarise(
    group                      = "Fossil Fuel Average",
    n                          = n(),
    ce_share                   = mean(process_type == "CE"),
    blm_share                  = mean(
      map_lgl(lead_agency_harmonized,
              ~ safe_agency_match(.x, "BLM|Bureau of Land Management")),
      na.rm = TRUE
    ),
    federal_land_trigger_share = mean(nepa_trigger_primary == "federal_land", na.rm = TRUE),
    visual_section_pct         = NA_real_   # visual extractor runs EA/EIS only
  )

geo_avg <- geo_og |>
  filter(tech_group == "Geothermal", !is.na(process_type)) |>
  summarise(
    group                      = "Geothermal (All Agencies)",
    n                          = n(),
    ce_share                   = mean(process_type == "CE"),
    blm_share                  = mean(
      map_lgl(lead_agency_harmonized,
              ~ safe_agency_match(.x, "BLM|Bureau of Land Management")),
      na.rm = TRUE
    ),
    federal_land_trigger_share = mean(nepa_trigger_primary == "federal_land", na.rm = TRUE),
    visual_section_pct         = visual_section_pct_for(geo_ids)
  )

comparison_table <- bind_rows(clean_avg, fossil_avg, geo_avg) |>
  select(group, n, ce_share, blm_share, federal_land_trigger_share, visual_section_pct)

write.csv(comparison_table,
          file.path(OUTPUT_DIR, "geothermal_comparison_table.csv"),
          row.names = FALSE)

write.csv(geo_og_state,
          file.path(OUTPUT_DIR, "geo_og_comparison.csv"),
          row.names = FALSE)

cat("  Section 5 done.\n")
} # end if (GEO_OG_AVAILABLE)


# ===========================================================================
# SECTION 6: TIMELINES (CONDITIONAL)
# ===========================================================================
cat("\n--- Section 6: Timelines ---\n")

if (!file.exists(TIMELINE_PATH)) {
  message("Skipping timeline section: data/analysis/timeline.parquet not found.")
  message("Expected columns: project_id, initiation_date, decision_date, process_type")
} else {
  timeline <- read_parquet(TIMELINE_PATH)

  # Coverage table — always first; durations are meaningless without this denominator
  # Only pull project_energy_type from df; timeline already has process_type
  coverage <- timeline |>
    left_join(df |> select(project_id, project_energy_type), by = "project_id") |>
    group_by(project_energy_type, process_type) |>
    summarise(
      n_total          = n(),
      n_has_initiation = sum(!is.na(initiation_date)),
      n_has_decision   = sum(!is.na(decision_date)),
      n_both           = sum(!is.na(initiation_date) & !is.na(decision_date)),
      pct_both         = mean(!is.na(initiation_date) & !is.na(decision_date)),
      .groups          = "drop"
    )

  write.csv(coverage, file.path(OUTPUT_DIR, "timeline_coverage.csv"), row.names = FALSE)
  cat("Timeline coverage:\n"); print(as.data.frame(coverage))

  # Durations
  durations <- timeline |>
    filter(!is.na(initiation_date), !is.na(decision_date)) |>
    mutate(
      duration_days = as.numeric(as.Date(decision_date) - as.Date(initiation_date)),
      period = case_when(
        as.Date(decision_date) >= as.Date("2023-08-16") ~ "Post-FRA",
        as.Date(decision_date) >= as.Date("2020-09-14") ~ "Post-2020 CEQ",
        TRUE                                             ~ "Pre-2020"
      ),
      period = factor(period, levels = c("Pre-2020", "Post-2020 CEQ", "Post-FRA"))
    ) |>
    filter(duration_days > 0, duration_days < 365 * 20) |>
    left_join(df |> select(project_id, project_energy_type, tech_group),
              by = "project_id")

  summary_table <- durations |>
    filter(!is.na(project_energy_type), !is.na(process_type)) |>
    group_by(project_energy_type, process_type, period) |>
    summarise(
      n           = n(),
      median_days = median(duration_days),
      p25         = quantile(duration_days, 0.25),
      p75         = quantile(duration_days, 0.75),
      .groups     = "drop"
    )

  # Fig 20 — Median duration by period × process type × energy ----
  ggplot(summary_table |> filter(!is.na(project_energy_type)),
         aes(x = period, y = median_days, fill = project_energy_type)) +
    geom_col(position = "dodge") +
    scale_fill_manual(values = energy_colors) +
    facet_wrap(~ process_type) +
    labs(x = NULL, y = "Median Review Duration (days)",
         fill = "Energy Category",
         title = "Review Duration by Period, Process Type, and Energy Category",
         caption = DATA_CAPTION) +
    theme_catf() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          legend.position = "bottom")
  save_fig("fig20_duration_by_energy_process.png")

  write.csv(summary_table,
            file.path(OUTPUT_DIR, "duration_summary.csv"),
            row.names = FALSE)

  cat("  Section 6 done.\n")
}


# ===========================================================================
# DONE
# ===========================================================================
cat(sprintf(
  "\nAll outputs written to: %s\n",
  OUTPUT_DIR
))
