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
#   fig12 — Visual impact prevalence by tech_group × process type
#   fig13 — Visual similarity score distribution (boxplot)
#   fig14 — Visual section detection rate by tech_group
#   fig15 — CE/EA/EIS rates: Geothermal vs. Oil & Gas
#   fig16 — Geothermal vs. Oil & Gas share by state (100% stacked bar, all states)
#   fig17 — State choropleth: Geothermal share (diverging blue-purple-red)
#   fig18 — Duration by period × process type × energy (conditional on timeline.parquet)
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
# SECTION 4: VISUAL IMPACTS
# ===========================================================================
cat("\n--- Section 4: Visual Impacts ---\n")
if (!VISUAL_AVAILABLE) {
  message("Section 4 skipped: projects_visual_impacts.parquet not found.")
} else {

VISUAL_THRESHOLD <- 0.4

vis_joined <- vis |>
  left_join(df |> select(project_id, tech_group, process_type,
                         project_energy_type), by = "project_id") |>
  mutate(
    has_visual   = visual_impacts_max_similarity >= VISUAL_THRESHOLD,
    process_type = factor(process_type, levels = c("EIS", "EA", "CE"))
  )

# Calibration diagnostics ----
cat("Visual similarity distribution (quantiles):\n")
print(quantile(vis$visual_impacts_max_similarity,
               c(0.25, 0.5, 0.75, 0.9, 0.95), na.rm = TRUE))
cat(sprintf(
  "Projects above threshold (%.1f): %.1f%%\n",
  VISUAL_THRESHOLD,
  100 * mean(vis$visual_impacts_max_similarity >= VISUAL_THRESHOLD, na.rm = TRUE)
))

# Fig 12 — Prevalence by tech_group × process type ----
prevalence <- vis_joined |>
  filter(!is.na(tech_group), !tech_group %in% c("Other", "Other Clean", "Other Fossil"),
         !is.na(process_type)) |>
  count(tech_group, process_type, has_visual) |>
  group_by(tech_group, process_type) |>
  mutate(pct = n / sum(n)) |>
  filter(has_visual) |>
  ungroup()

ggplot(prevalence, aes(x = reorder(tech_group, pct), y = pct, fill = process_type)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = process_colors, labels = process_labels) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = NULL, y = "% with Substantive Visual Analysis", fill = "Review Type",
       title = "Visual Impact Discussion Prevalence by Technology",
       subtitle = paste0("Lexical prefilter + all-MiniLM-L6-v2, threshold = ", VISUAL_THRESHOLD),
       caption = DATA_CAPTION) +
  theme_catf() +
  theme(legend.position = "bottom")
save_fig("fig12_visual_prevalence_by_tech.png")

# Fig 13 — Similarity score distribution (boxplot) ----
vis_joined |>
  filter(!is.na(tech_group), !tech_group %in% c("Other", "Other Clean", "Other Fossil"),
         !is.na(visual_impacts_max_similarity)) |>
  ggplot(aes(
    x    = reorder(tech_group, visual_impacts_max_similarity, median),
    y    = visual_impacts_max_similarity,
    fill = tech_group
  )) +
  geom_boxplot(outlier.size = 0.5, alpha = 0.85) +
  geom_hline(yintercept = VISUAL_THRESHOLD, linetype = "dashed", color = "grey40") +
  scale_fill_manual(values = tech_colors, guide = "none") +
  coord_flip() +
  labs(x = NULL, y = "Max Cosine Similarity",
       title = "Visual Impact Similarity Distribution by Technology",
       subtitle = paste0("Dashed line = threshold (", VISUAL_THRESHOLD, ")"),
       caption = DATA_CAPTION) +
  theme_catf()
save_fig("fig13_visual_similarity_dist.png")

# Fig 14 — Dedicated visual section detection rate ----
section_rates <- vis_joined |>
  filter(!is.na(tech_group), !tech_group %in% c("Other", "Other Clean", "Other Fossil")) |>
  group_by(tech_group) |>
  summarise(
    pct_section_found    = mean(visual_section_found, na.rm = TRUE),
    median_mention_count = median(visual_mention_count, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(section_rates,
       aes(x = reorder(tech_group, pct_section_found), y = pct_section_found)) +
  geom_col(fill = catf_dark_blue) +
  scale_y_continuous(labels = percent_format()) +
  coord_flip() +
  labs(x = NULL, y = "% of Projects with Dedicated Visual Section",
       title = "Visual Resource Section Detection Rate by Technology",
       caption = DATA_CAPTION) +
  theme_catf()
save_fig("fig14_visual_section_detection.png")

# Export prevalence table ----
visual_prevalence_table <- vis_joined |>
  filter(!is.na(tech_group), !is.na(process_type)) |>
  group_by(tech_group, process_type) |>
  summarise(
    n_projects       = n(),
    n_has_visual     = sum(has_visual, na.rm = TRUE),
    pct_has_visual   = mean(has_visual, na.rm = TRUE),
    median_similarity = median(visual_impacts_max_similarity, na.rm = TRUE),
    pct_section_found = mean(visual_section_found, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(visual_prevalence_table,
          file.path(OUTPUT_DIR, "visual_prevalence_table.csv"),
          row.names = FALSE)

cat("  Section 4 done.\n")
} # end if (VISUAL_AVAILABLE)


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

  # Fig 18 — Median duration by period × process type × energy ----
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
  save_fig("fig18_duration_by_energy_process.png")

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
