# --------------------------
# DELIVERABLE 1: NEPA TRIGGERED — Secondary-trigger & review-status cross-tabulations
# --------------------------
# Extends the D1 analysis with two deferred integrations:
#
#   (A) Secondary-trigger multi-label cross-tabulations (todo #55)
#       - primary x secondary co-occurrence matrix (multi-label projects only)
#       - dominant trigger combinations (share of multi-label and of all projects)
#       - multi-label rate by review process (CE/EA/EIS) and by energy technology
#
#   (B) Trigger class x review-status integration (todo #56)
#       - trigger class x programmatic/tiered status, sourced from D6's FONSI
#         enrichment `is_tiered`/`tiers_from` fields (the ONLY committed Phase 2
#         programmatic/tiered flag; D2's outputs are significance determinations
#         and regulatory regimes, NOT programmatic/tiered flags -- see the report's
#         Coverage & Limitations note). FONSI-enriched subset only.
#       - trigger class x regulatory review regime (pre-/post-FRA), sourced from
#         D2's project_regime.parquet, for the EA/EIS decision universe.
#
# All numbers regenerable from committed parquets; no API calls.
#
# Input:
#   phase2/data/analysis/deliverable01/projects_nepa_trigger.parquet
#   phase2/data/analysis/projects_combined.parquet
#   phase2/data/analysis/deliverable06/fonsi_enrichment.parquet   (D6 tiering flag)
#   phase2/data/analysis/deliverable02/project_regime.parquet      (D2 regulatory regime)
#
# Output (all in phase2/output/deliverable01/):
#   secondary_cooccurrence_matrix.csv
#   secondary_top_combos.csv
#   secondary_multilabel_by_process.csv
#   secondary_multilabel_by_technology.csv
#   trigger_tiering_crosstab.csv
#   trigger_regime_crosstab.csv
#   fig13_secondary_cooccurrence.png
#   fig14_secondary_multilabel_rates.png
#   fig15_trigger_tiering.png
#
# Usage:
#   Rscript phase2/code/deliverable01/04_secondary_review_crosstabs.R

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
  library(jsonlite)
  library(stringr)
  library(tibble)
})

source(here::here("phase2", "code", "utils", "utils.R"))

# --------------------------
# PATHS
# --------------------------
BASE_DIR   <- here::here()
OUTPUT_DIR <- file.path(BASE_DIR, "phase2", "output", "deliverable01")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

TRIGGERS_PATH  <- file.path(BASE_DIR, "phase2", "data", "analysis", "deliverable01",
                            "projects_nepa_trigger.parquet")
PROJECTS_PATH  <- file.path(BASE_DIR, "phase2", "data", "analysis", "projects_combined.parquet")
FONSI_ENRICH_PATH <- file.path(BASE_DIR, "phase2", "data", "analysis", "deliverable06",
                               "fonsi_enrichment.parquet")
REGIME_PATH    <- file.path(BASE_DIR, "phase2", "data", "analysis", "deliverable02",
                            "project_regime.parquet")

# --------------------------
# LABEL LOOKUPS (mirrors 02_create_figures.R)
# --------------------------
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

trigger_colors <- c(
  "Funding"              = "#0047BB",
  "Direct Action"        = "#00AE8D",
  "Land"                 = "#C22A90",
  "Permit"               = "#8AB7E9",
  "Program"              = "#00B5E2",
  "Property Transaction" = "#75246C",
  "PMA/TVA"              = "#012169",
  "Unknown"              = "grey70"
)

process_labels <- c(
  CE  = "Categorical Exclusion",
  EA  = "Environmental Assessment",
  EIS = "Environmental Impact Statement"
)

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
# LOAD AND PREPARE (Clean universe)
# --------------------------
options(arrow.skip_nul = TRUE)

triggers <- read_parquet(TRIGGERS_PATH, col_select = all_of(c(
  "project_id", "nepa_trigger_primary", "nepa_trigger_count", "nepa_trigger_combo"
)))
projects <- read_parquet(PROJECTS_PATH, col_select = all_of(c(
  "project_id", "project_energy_type", "process_type", "project_type"
)))

df <- left_join(triggers, projects, by = "project_id") |>
  filter(project_energy_type == "Clean") |>
  collect() |>
  mutate(
    trigger_label = recode(nepa_trigger_primary, !!!trigger_labels),
    project_technology = map_chr(project_type, function(x) {
      tags    <- parse_json_all(x)
      ce_tags <- tags[tags %in% names(tech_labels)]
      if (length(ce_tags) > 0) tech_labels[[ce_tags[[1]]]] else NA_character_
    })
  )

n_total    <- nrow(df)
trigger_order <- c(
  df |> filter(trigger_label != "Unknown") |> count(trigger_label, sort = TRUE) |>
    pull(trigger_label),
  "Unknown"
)

cat(sprintf("Loaded %s Clean-universe projects.\n", comma(n_total)))

# ============================================================================
# (A) SECONDARY-TRIGGER MULTI-LABEL CROSS-TABULATIONS  (todo #55)
# ============================================================================

multi <- df |> filter(nepa_trigger_count >= 2)
n_multi <- nrow(multi)
cat(sprintf("Multi-label (2+ triggers): %s (%.1f%%)\n", comma(n_multi), 100 * n_multi / n_total))

# --- Primary x secondary co-occurrence (long) --------------------------------
# combo is the sorted, pipe-joined set of all detected classes; the secondary set
# is combo minus the primary. One row per (primary, secondary) pair.
cooc <- multi |>
  mutate(
    combo_parts = strsplit(nepa_trigger_combo, "\\|"),
    secondary   = map2(combo_parts, nepa_trigger_primary, ~ setdiff(.x, .y))
  ) |>
  select(project_id, nepa_trigger_primary, secondary) |>
  unnest_longer(secondary) |>
  filter(!is.na(secondary), secondary != "") |>
  transmute(
    primary_label   = recode(nepa_trigger_primary, !!!trigger_labels),
    secondary_label = recode(secondary,            !!!trigger_labels)
  ) |>
  count(primary_label, secondary_label, name = "n") |>
  arrange(desc(n))

write.csv(cooc, file.path(OUTPUT_DIR, "secondary_cooccurrence_matrix.csv"), row.names = FALSE)
cat("Saved secondary_cooccurrence_matrix.csv\n")

# --- Dominant combinations ---------------------------------------------------
combo_tab <- multi |>
  mutate(
    combo_label = map_chr(strsplit(nepa_trigger_combo, "\\|"), function(parts) {
      paste(unname(trigger_labels[parts]), collapse = " + ")
    })
  ) |>
  count(combo_label, name = "n") |>
  mutate(
    pct_of_multi = n / n_multi,
    pct_of_all   = n / n_total
  ) |>
  arrange(desc(n))

write.csv(combo_tab, file.path(OUTPUT_DIR, "secondary_top_combos.csv"), row.names = FALSE)
cat("Saved secondary_top_combos.csv\n")

# --- Multi-label rate by review process --------------------------------------
ml_by_process <- df |>
  filter(!is.na(process_type)) |>
  group_by(process_type) |>
  summarise(
    n_total = n(),
    n_multi = sum(nepa_trigger_count >= 2, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    pct_multi     = n_multi / n_total,
    process_label = recode(process_type, !!!process_labels)
  ) |>
  arrange(desc(pct_multi))

write.csv(ml_by_process, file.path(OUTPUT_DIR, "secondary_multilabel_by_process.csv"),
          row.names = FALSE)
cat("Saved secondary_multilabel_by_process.csv\n")

# --- Multi-label rate by energy technology -----------------------------------
tech_min_n <- 50
ml_by_tech <- df |>
  filter(!is.na(project_technology)) |>
  group_by(project_technology) |>
  summarise(
    n_total = n(),
    n_multi = sum(nepa_trigger_count >= 2, na.rm = TRUE),
    .groups = "drop"
  ) |>
  filter(n_total >= tech_min_n) |>
  mutate(pct_multi = n_multi / n_total) |>
  arrange(desc(pct_multi))

write.csv(ml_by_tech, file.path(OUTPUT_DIR, "secondary_multilabel_by_technology.csv"),
          row.names = FALSE)
cat("Saved secondary_multilabel_by_technology.csv\n")

# --- FIGURE 13: primary x secondary co-occurrence heatmap --------------------
present_labels <- intersect(trigger_order, union(cooc$primary_label, cooc$secondary_label))
heat <- cooc |>
  mutate(
    primary_label   = factor(primary_label,   levels = rev(present_labels)),
    secondary_label = factor(secondary_label, levels = present_labels)
  )

fig13 <- ggplot(heat, aes(x = secondary_label, y = primary_label, fill = n)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = comma(n)), size = 3.6, fontface = "bold",
            color = ifelse(heat$n > max(heat$n) * 0.55, "white", "gray20")) +
  scale_fill_gradientn(colors = catf_sequential, name = "Projects",
                       labels = comma) +
  scale_x_discrete(position = "top") +
  labs(
    title    = "Primary x Secondary Trigger Co-occurrence",
    subtitle = sprintf("Multi-label projects only (n = %s with 2+ trigger classes)", comma(n_multi)),
    x = "Secondary trigger", y = "Primary trigger"
  ) +
  theme_catf(base_size = 12) +
  theme(
    legend.position = "right",
    panel.grid.major = element_blank(),
    axis.text.x = element_text(angle = 30, hjust = 0)
  )

ggsave(file.path(OUTPUT_DIR, "fig13_secondary_cooccurrence.png"),
       fig13, width = 8.5, height = 6, dpi = 150)
cat("Saved fig13_secondary_cooccurrence.png\n")

# --- FIGURE 14: multi-label rate by process and by technology ----------------
panel_process <- ml_by_process |>
  transmute(panel = "By review process",
            group = recode(process_type, !!!process_labels),
            pct_multi, n_total, n_multi)
panel_tech <- ml_by_tech |>
  transmute(panel = "By energy technology",
            group = project_technology,
            pct_multi, n_total, n_multi)

ml_plot <- bind_rows(panel_process, panel_tech) |>
  mutate(panel = factor(panel, levels = c("By review process", "By energy technology"))) |>
  group_by(panel) |>
  mutate(group = fct_reorder(group, pct_multi)) |>
  ungroup()

fig14 <- ggplot(ml_plot, aes(x = group, y = pct_multi)) +
  geom_col(fill = catf_dark_blue, width = 0.68) +
  geom_text(aes(label = percent(pct_multi, accuracy = 0.1)),
            hjust = -0.15, size = 3.2, color = "gray25") +
  coord_flip() +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.18))) +
  facet_wrap(~ panel, scales = "free_y", ncol = 1) +
  labs(
    title    = "Share of Projects with 2+ Trigger Classes",
    subtitle = "Multi-label rate by review process and by energy technology (technologies with n >= 50)",
    x = NULL, y = "Share with a secondary trigger"
  ) +
  theme_catf(base_size = 12)

ggsave(file.path(OUTPUT_DIR, "fig14_secondary_multilabel_rates.png"),
       fig14, width = 8.5, height = 7, dpi = 150)
cat("Saved fig14_secondary_multilabel_rates.png\n")

# ============================================================================
# (B) TRIGGER CLASS x REVIEW-STATUS INTEGRATION  (todo #56)
# ============================================================================
# Sourcing note: a portfolio-wide programmatic/tiered classifier does NOT exist
# in Phase 2. Verified: D2's outputs (significance_determinations, project_regime,
# project_cohorts) carry significance determinations and regulatory regimes, not
# programmatic/tiered flags. The ONLY committed programmatic/tiered signal is D6's
# FONSI-enrichment `is_tiered`/`tiers_from`, available for the FONSI-enriched
# subset. We use that (correct) source here and cross-tab against trigger class;
# we add a portfolio-scale trigger x regulatory-regime table from D2 for context.

trig_primary <- df |> select(project_id, trigger_label)

# --- Trigger x tiering status (D6, correct source) ---------------------------
fonsi <- read_parquet(FONSI_ENRICH_PATH,
                      col_select = all_of(c("project_id", "is_tiered"))) |>
  collect()

tier_join <- fonsi |>
  inner_join(trig_primary, by = "project_id") |>
  mutate(tiering = case_when(
    is.na(is_tiered) ~ "Undetermined",
    is_tiered        ~ "Tiered",
    TRUE             ~ "Standalone"
  ))

n_fonsi_enriched <- nrow(tier_join)
n_determined     <- sum(tier_join$tiering != "Undetermined")
n_tiered         <- sum(tier_join$tiering == "Tiered")

tiering_tab <- tier_join |>
  count(trigger_label, tiering, name = "n") |>
  pivot_wider(names_from = tiering, values_from = n, values_fill = 0)
for (col in c("Tiered", "Standalone", "Undetermined")) {
  if (!col %in% names(tiering_tab)) tiering_tab[[col]] <- 0L
}
tiering_tab <- tiering_tab |>
  mutate(
    determined = Tiered + Standalone,
    total      = Tiered + Standalone + Undetermined,
    pct_tiered_of_determined = ifelse(determined > 0, Tiered / determined, NA_real_)
  ) |>
  select(trigger_label, Tiered, Standalone, Undetermined, determined, total,
         pct_tiered_of_determined) |>
  arrange(desc(Tiered), desc(determined))

write.csv(tiering_tab, file.path(OUTPUT_DIR, "trigger_tiering_crosstab.csv"), row.names = FALSE)
cat(sprintf("Saved trigger_tiering_crosstab.csv (%s FONSI-enriched; %s determined; %s tiered)\n",
            comma(n_fonsi_enriched), comma(n_determined), comma(n_tiered)))

# --- FIGURE 15: trigger x tiering (determined FONSIs only) -------------------
tier_fig_data <- tier_join |>
  filter(tiering != "Undetermined") |>
  count(trigger_label, tiering, name = "n") |>
  group_by(trigger_label) |>
  mutate(grp_total = sum(n)) |>
  ungroup() |>
  filter(grp_total >= 3) |>
  mutate(
    trigger_label = fct_reorder(trigger_label, grp_total),
    tiering = factor(tiering, levels = c("Standalone", "Tiered"))
  )

fig15 <- ggplot(tier_fig_data, aes(x = trigger_label, y = n, fill = tiering)) +
  geom_col(width = 0.68) +
  geom_text(aes(label = n), position = position_stack(vjust = 0.5),
            color = "white", size = 3.3, fontface = "bold") +
  coord_flip() +
  scale_fill_manual(values = c("Standalone" = "#8AB7E9", "Tiered" = "#012169"),
                    name = NULL) +
  labs(
    title    = "Trigger Class x Programmatic Tiering (FONSI subset)",
    subtitle = sprintf("D6 FONSI-enrichment tiering flag; %s of %s enriched FONSIs have a determination",
                       comma(n_determined), comma(n_fonsi_enriched)),
    x = NULL, y = "FONSI projects"
  ) +
  theme_catf(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(OUTPUT_DIR, "fig15_trigger_tiering.png"),
       fig15, width = 8, height = 5.5, dpi = 150)
cat("Saved fig15_trigger_tiering.png\n")

# --- Trigger x regulatory regime (D2, portfolio-scale context) ---------------
regime <- read_parquet(REGIME_PATH,
                       col_select = all_of(c("project_id", "process_type",
                                             "decision_period", "fra_overlay"))) |>
  collect()

regime_tab <- regime |>
  inner_join(trig_primary, by = "project_id") |>
  group_by(trigger_label) |>
  summarise(
    n_regime      = n(),
    n_pre_2020ceq = sum(decision_period == "pre_2020_ceq", na.rm = TRUE),
    n_ceq_2020_25 = sum(decision_period %in% c("ceq_2020_rule", "ceq_2022_phase1",
                                               "ceq_2024_phase2", "ceq_2025_interim_removal"),
                        na.rm = TRUE),
    n_period_unknown = sum(decision_period == "unknown" | is.na(decision_period)),
    n_fra_overlay = sum(fra_overlay, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(desc(n_regime))

write.csv(regime_tab, file.path(OUTPUT_DIR, "trigger_regime_crosstab.csv"), row.names = FALSE)
cat(sprintf("Saved trigger_regime_crosstab.csv (%s EA/EIS projects in the D2 regime universe)\n",
            comma(nrow(regime))))

cat("\nDone: secondary + review-status cross-tabulations.\n")
