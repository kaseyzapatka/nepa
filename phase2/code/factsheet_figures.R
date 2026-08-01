#!/usr/bin/env Rscript
# factsheet_figures.R  (Phase 2)
# Builds the figures used by the Phase 2 factsheets (phase2/factsheets/*.qmd).
#
# Three kinds of figures:
#   1. RETITLED — readRDS the ggplot object saved next to the deliverable's .png
#                 (every NN_create_figures.R now writes a .rds sidecar), add a
#                 client-facing headline title with labs(), and re-save. No
#                 re-computation — the upstream figure is the single source of truth.
#   2. FROM SCRATCH — built here only when there is no upstream original to reuse
#                 (currently just fs1_duration_by_technology; see its TODO).
#   3. COPIED   — passed through unchanged from phase2/output/deliverableXX/figures/
#                 so every factsheet reads from ONE directory. To retitle one of
#                 these later, promote it to a RETITLED section.
#
# Run with: Rscript phase2/code/factsheet_figures.R
# Output:   phase2/output/factsheet/figures/  (+ summary tables in ../tables/)

library(here)
library(arrow)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(forcats)
library(ggplot2)
library(scales)

fig_dir <- here("phase2", "output", "factsheet", "figures")
tbl_dir <- here("phase2", "output", "factsheet", "tables")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tbl_dir, recursive = TRUE, showWarnings = FALSE)

D01_OUT <- here("phase2", "output", "deliverable01")
D03_OUT <- here("phase2", "output", "deliverable03")
D04_FIG <- here("phase2", "output", "deliverable04", "figures")
D06_FIG <- here("phase2", "output", "deliverable06", "figures")

# ---------------------------------------------------------------------------
# CATF brand colors + theme (same as phase1/code/factsheet_figures.R)
# ---------------------------------------------------------------------------
catf_dark_blue  <- "#0047BB"
catf_blue       <- "#00B5E2"
catf_magenta    <- "#C22A90"
catf_purple     <- "#75246C"
catf_lime       <- "#93D500"
catf_teal       <- "#00AE8D"
catf_light_blue <- "#8AB7E9"
catf_navy       <- "#002169"
catf_red        <- "#7B241C"
catf_palette    <- c("#0047BB","#00B5E2","#00AE8D","#93D500","#C22A90","#75246C","#8AB7E9","#002169")

theme_catf <- function(base_size = 11, base_family = "Helvetica") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      plot.title       = element_text(face = "bold", size = rel(1.2), color = catf_navy,
                                      margin = margin(b = 10)),
      plot.subtitle    = element_text(size = rel(0.9), color = catf_dark_blue,
                                      margin = margin(b = 10)),
      plot.caption     = element_text(size = rel(0.8), color = "gray50", hjust = 0),
      axis.title       = element_text(size = rel(0.9), color = catf_navy),
      axis.text        = element_text(size = rel(0.85), color = "gray30"),
      axis.line        = element_line(color = "gray70", linewidth = 0.3),
      legend.title     = element_text(face = "bold", size = rel(0.9), color = catf_navy),
      legend.text      = element_text(size = rel(0.85), color = "gray30"),
      legend.position  = "bottom",
      legend.key.size  = unit(0.8, "lines"),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      strip.text       = element_text(face = "bold", size = rel(0.9), color = catf_navy),
      strip.background = element_rect(fill = "gray95", color = NA),
      plot.margin      = margin(15, 15, 10, 10)
    )
}
scale_color_catf <- function(...) scale_color_manual(values = catf_palette, ...)
scale_fill_catf  <- function(...) scale_fill_manual(values = catf_palette, ...)
theme_set(theme_catf())

process_levels <- c("CE", "EA", "EIS")

# ===========================================================================
# COPIED FIGURES — passthrough from the deliverable output directories
# ===========================================================================
message("--- Copying passthrough figures ---")

passthrough <- c(
  # --- Fact Sheet 1: Timelines (D4) ---
  file.path(D04_FIG, "fig_d4_complete_timeline_share_boxplot.png"),
  file.path(D04_FIG, "fig_d4_coverage_by_process.png"),
  file.path(D04_FIG, "fig_d4_duration_histogram.png"),
  file.path(D04_FIG, "fig_d4_duration_summary_intervals_by_energy.png"),
  file.path(D04_FIG, "fig_d4_solar_duration.png"),
  file.path(D04_FIG, "fig_d4_projects_by_decision_year.png"),
  file.path(D04_FIG, "fig_d4_projects_by_decision_year_doe.png"),
  file.path(D04_FIG, "fig_d4_fra_comparison.png"),
  file.path(D04_FIG, "fig_d4_duration_by_ceq_regime.png"),
  file.path(D04_FIG, "fig_d4_duration_trend_ceq_regime.png"),
  file.path(D04_FIG, "fig_d4_pages_over_time.png"),
  file.path(D04_FIG, "fig_d4_pages_pre_post_fra.png"),
  file.path(D04_FIG, "fig_d4_pages_compliance.png"),
  # --- Fact Sheet 2: Triggers (D1) ---
  file.path(D01_OUT, "fig2_trigger_by_process.png"),
  file.path(D01_OUT, "fig4_department_trigger_heatmap.png"),
  file.path(D01_OUT, "fig5_trigger_by_technology.png"),
  file.path(D01_OUT, "fig10_funding_amount_coverage.png"),
  file.path(D01_OUT, "fig11_funding_amount_distribution.png"),
  # --- Fact Sheet 3: Categorical Exclusions (D6 + D3) ---
  file.path(D06_FIG, "fig_d6_coverage_grid.png"),
  file.path(D06_FIG, "fig_d6_sizes.png"),
  file.path(D06_FIG, "fig_d6_ce_split.png"),
  file.path(D06_FIG, "fig_d6_ce_scatter.png"),
  file.path(D06_FIG, "fig_d6_ce_numlimit.png"),
  file.path(D06_FIG, "fig_d6_adoption_gap.png"),
  file.path(D06_FIG, "fig_d6_mitigated_overall.png"),
  file.path(D06_FIG, "fig_d6_mitigation_roles.png"),
  file.path(D06_FIG, "fig_d6_timeline.png"),
  file.path(D03_OUT, "fig4_top_ce_codes.png"),
  # --- Fact Sheet 4: Visual Impacts (D3) ---
  file.path(D03_OUT, "fig12_visual_project_counts.png"),
  file.path(D03_OUT, "fig13_wordcloud_grid.png"),
  file.path(D03_OUT, "fig18_visual_framing.png"),
  file.path(D03_OUT, "fig19a_section_length_energy.png"),
  file.path(D03_OUT, "fig14_topic_prevalence.png"),
  file.path(D03_OUT, "fig14d_nmf_elbow.png")
)

missing <- passthrough[!file.exists(passthrough)]
if (length(missing)) {
  warning("Missing source figures (skipped):\n  ", paste(missing, collapse = "\n  "))
}
invisible(file.copy(passthrough[file.exists(passthrough)], fig_dir, overwrite = TRUE))
message("  Copied ", sum(file.exists(passthrough)), " figures to ", fig_dir)

# ===========================================================================
# FS1 Fig — Review duration by process, headline title  [RETITLED]
# Source: readRDS D4's fig_d4_duration_summary_intervals.rds (the .rds sidecar
# written by phase2/code/deliverable04/08_create_figures.R). labs() overrides the
# upstream title with the factsheet headline.
# ===========================================================================
message("\n--- FS1: duration by process (retitled from D4 .rds) ---")

# Retitle helper: .rds sidecars are gitignored (regenerable via the deliverables'
# NN_create_figures.R). If one is absent on this machine, keep the committed PNG
# and say so instead of aborting the whole run.
retitle <- function(rds_path, out_name, new_title, width, height) {
  tryCatch({
    p <- readRDS(rds_path) + labs(title = new_title)
    ggsave(file.path(fig_dir, out_name), p, width = width, height = height, dpi = 300)
    message("  Saved: ", out_name)
  }, error = function(e) {
    message("  SKIPPED ", out_name, " (missing ", basename(rds_path),
            " — re-run the deliverable's figure script to regenerate; committed PNG kept)")
  })
}

retitle(file.path(D04_FIG, "fig_d4_duration_summary_intervals.rds"),
        "fs1_duration_by_process.png",
        "NEPA Review Duration Climbs From Days (CE) to Months (EA) to Years (EIS)",
        width = 10, height = 5.5)

# ===========================================================================
# FS1 Fig — Review duration by technology  [FROM SCRATCH — STOPGAP]
# TODO: This is a from-scratch build only because D4 has no upstream figure for
#   duration-by-technology yet. Once the user adds that figure to
#   phase2/code/deliverable04/08_create_figures.R (which will write a
#   fig_d4_*_by_technology.rds sidecar), replace this whole section with a
#   readRDS + labs() RETITLED block like the others above and delete the build.
# Timeline: timeline_project_dates.parquet (complete timelines, headline frame)
# Technology: deliverable03 projects_nepa_reviews.parquet (tech_group, energy_group)
# ===========================================================================
message("\n--- FS1: duration by technology (from-scratch stopgap) ---")

# Headline duration frame — mirrors phase2/code/deliverable04/08_create_figures.R
# exactly: complete statuses; year-only dates excluded; month-only dates imputed
# to mid-month; non-negative spans. Durations are recomputed here because the
# parquet's precomputed duration_days column is populated ONLY for day/day-
# precision pairs — filtering on it silently drops every month-dated complete
# timeline (undercounting EA/EIS technology cells by 15-75%).
tl <- read_parquet(
  here("phase2", "data", "analysis", "timeline", "timeline_project_dates.parquet"),
  col_select = c("project_id", "process_type", "timeline_status",
                 "initiation_date", "decision_date",
                 "initiation_date_granularity", "decision_date_granularity")
) |>
  filter(timeline_status %in% c("complete_clear", "complete_with_proxy"),
         !is.na(initiation_date), !is.na(decision_date),
         initiation_date_granularity != "year", decision_date_granularity != "year") |>
  mutate(
    initiation_date = as.Date(initiation_date),
    decision_date   = as.Date(decision_date),
    .init_mid = if_else(initiation_date_granularity == "month",
                        lubridate::floor_date(initiation_date, "month") + 14, initiation_date),
    .dec_mid  = if_else(decision_date_granularity == "month",
                        lubridate::floor_date(decision_date, "month") + 14, decision_date),
    duration_days = as.integer(.dec_mid - .init_mid)
  ) |>
  filter(!is.na(duration_days), duration_days >= 0) |>
  select(project_id, process_type, timeline_status, duration_days)

tech <- read_parquet(
  here("phase2", "data", "analysis", "deliverable03", "projects_nepa_reviews.parquet"),
  col_select = c("project_id", "tech_group", "energy_group")
) |>
  filter(energy_group %in% c("Decarbonization", "Fossil Fuel"), !is.na(tech_group))

tl_tech <- tl |>
  inner_join(tech, by = "project_id") |>
  mutate(duration_months = duration_days / 30.44,
         process_group = factor(process_type, levels = process_levels))

tech_summary_all <- tl_tech |>
  group_by(energy_group, tech_group, process_group) |>
  summarise(
    n      = n(),
    p10    = quantile(duration_months, 0.10),
    p25    = quantile(duration_months, 0.25),
    median = median(duration_months),
    p75    = quantile(duration_months, 0.75),
    p90    = quantile(duration_months, 0.90),
    .groups = "drop"
  )

# Display rule: NO suppression — every technology cell is shown so the panels
# account for every complete energy timeline. Each bar carries an n label, and
# the factsheet caption warns that very small cells are unstable; percentile
# bars for tiny n collapse toward the observed values.
tech_summary <- tech_summary_all

write_csv(tech_summary, file.path(tbl_dir, "fs1_duration_by_technology.csv"))

# EA + EIS only: the CE panel is uniformly ~<1 month and adds no contrast
fs1_tech <- tech_summary |>
  filter(process_group %in% c("EA", "EIS")) |>
  ggplot(aes(y = fct_reorder(tech_group, median), color = energy_group)) +
  geom_segment(aes(x = p10, xend = p90,
                   yend = fct_reorder(tech_group, median)),
               linewidth = 1.6, alpha = 0.35) +
  geom_segment(aes(x = p25, xend = p75,
                   yend = fct_reorder(tech_group, median)),
               linewidth = 4.5, alpha = 0.6) +
  geom_point(aes(x = median), size = 2.6) +
  geom_text(aes(x = p90, label = paste0("n=", n)),
            nudge_x = 2, hjust = 0, size = 2.7, color = "gray40") +
  facet_wrap(~process_group, ncol = 2, scales = "free_x") +
  scale_color_manual(values = c("Decarbonization" = catf_dark_blue,
                                "Fossil Fuel" = catf_red)) +
  scale_x_continuous(labels = label_number(accuracy = 1),
                     expand = expansion(mult = c(0.02, 0.18))) +
  labs(
    title = "Within Each Review Process, Decarbonization and Fossil Technologies\nFace Broadly Similar Review Times",
    subtitle = "Thin bar = 10th–90th percentile; thick bar = interquartile range; point = median. All complete timelines; every technology cell shown — n labels flag small cells.",
    x = "Duration (months)", y = NULL, color = NULL
  ) +
  theme_catf() +
  theme(legend.position = "bottom")

ggsave(file.path(fig_dir, "fs1_duration_by_technology.png"), fs1_tech,
       width = 11, height = 7, dpi = 300)
message("  Saved: fs1_duration_by_technology.png")

# Duration by energy group summary table (used for inline numbers in FS1)
energy_summary <- tl |>
  inner_join(
    read_parquet(here("phase2", "data", "analysis", "projects_combined.parquet"),
                 col_select = c("project_id", "project_energy_type")),
    by = "project_id"
  ) |>
  mutate(duration_months = duration_days / 30.44) |>
  group_by(project_energy_type, process_type) |>
  summarise(
    n = n(),
    median_days   = median(duration_days),
    median_months = median(duration_months),
    p25_months    = quantile(duration_months, 0.25),
    p75_months    = quantile(duration_months, 0.75),
    .groups = "drop"
  )
write_csv(energy_summary, file.path(tbl_dir, "fs1_duration_by_energy.csv"))
message("  Saved table: fs1_duration_by_energy.csv")

# ===========================================================================
# FS2 Fig — Trigger counts, headline title
# Source: deliverable01 projects_nepa_trigger.parquet x projects_combined.parquet
# ===========================================================================
message("\n--- FS2: trigger counts (retitled from D1 .rds) ---")

trigger_labels <- c(
  federal_funding              = "Funding",
  federal_land                 = "Land",
  pma                          = "PMA/TVA",
  federal_direct_action        = "Direct Action",
  federal_permit               = "Permit",
  federal_program              = "Program",
  federal_property_transaction = "Property Transaction",
  unknown                      = "Unknown"
)

triggers <- read_parquet(
  here("phase2", "data", "analysis", "deliverable01", "projects_nepa_trigger.parquet"),
  col_select = c("project_id", "nepa_trigger_primary")
) |>
  inner_join(
    read_parquet(here("phase2", "data", "analysis", "projects_combined.parquet"),
                 col_select = c("project_id", "project_energy_type", "process_type")),
    by = "project_id"
  ) |>
  filter(project_energy_type == "Clean") |>
  mutate(trigger = recode(nepa_trigger_primary, !!!trigger_labels))

n_clean_total <- nrow(triggers)

trig_counts <- triggers |>
  count(trigger, name = "n") |>
  mutate(share = n / n_clean_total)

write_csv(trig_counts, file.path(tbl_dir, "fs2_trigger_counts.csv"))

# <X>% for the headline title comes from the summary table just written.
funding_share <- trig_counts$share[trig_counts$trigger == "Funding"]

# [RETITLED + RELABELED] readRDS D1's fig1_trigger_counts.rds, override the
# title, and swap the count-only bar labels for count + share of total — the
# factsheet narrative describes triggers as percents of the portfolio, so the
# figure shows both.
tryCatch({
  p <- readRDS(file.path(D01_OUT, "fig1_trigger_counts.rds"))
  p$layers <- Filter(function(l) !inherits(l$geom, "GeomText"), p$layers)
  p <- p +
    geom_text(aes(label = sprintf("%s  (%s)", comma(n), percent(n / sum(n), accuracy = 0.1))),
              hjust = -0.12, size = 3.5, color = "gray20") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.28)), labels = comma) +
    labs(title = sprintf("Federal Funding Triggers %s of All Decarbonization NEPA Reviews —\nMore Than Any Other Federal Nexus",
                         percent(funding_share, accuracy = 1)))
  ggsave(file.path(fig_dir, "fs2_trigger_counts.png"), p, width = 10, height = 5.5, dpi = 300)
  message("  Saved: fs2_trigger_counts.png")
}, error = function(e) {
  message("  SKIPPED fs2_trigger_counts.png (missing fig1_trigger_counts.rds",
          " — re-run the deliverable's figure script to regenerate; committed PNG kept)")
})

# ===========================================================================
# FS2 Fig — Review process within each trigger class, headline title
# ===========================================================================
message("\n--- FS2: process by trigger (retitled from D1 .rds) ---")

proc_by_trig <- triggers |>
  filter(!is.na(process_type), trigger != "Unknown") |>
  count(trigger, process_type, name = "n") |>
  group_by(trigger) |>
  mutate(total = sum(n), share = n / total) |>
  ungroup() |>
  mutate(process_type = factor(process_type, levels = rev(process_levels)))

write_csv(proc_by_trig, file.path(tbl_dir, "fs2_process_by_trigger.csv"))

# <Y>% (Funding -> CE share) for the headline title comes from the summary table.
funding_ce_share <- proc_by_trig |>
  filter(trigger == "Funding", process_type == "CE") |>
  pull(share)

# [RETITLED] readRDS D1's fig3_process_by_trigger.rds and override the title.
retitle(file.path(D01_OUT, "fig3_process_by_trigger.rds"),
        "fs2_process_by_trigger.png",
        sprintf("Funding-Triggered Reviews Are Almost Entirely Categorically Excluded (%s)",
                percent(funding_ce_share, accuracy = 1)),
        width = 10, height = 5.5)

# ===========================================================================
# FS2 Tbl — Trigger shares by technology
# Read from D1's fig5 .rds sidecar (the same frame the copied
# fig5_trigger_by_technology.png plots) so factsheet prose and figure always agree.
# ===========================================================================
message("\n--- FS2: trigger shares by technology (table from D1's fig5 .rds) ---")

tryCatch({
  fig5 <- readRDS(file.path(D01_OUT, "fig5_trigger_by_technology.rds"))
  fig5$data |>
    transmute(technology = as.character(project_technology),
              trigger    = as.character(trigger_label),
              n, share = pct, total) |>
    write_csv(file.path(tbl_dir, "fs2_trigger_by_technology.csv"))
  message("  Saved table: fs2_trigger_by_technology.csv")
}, error = function(e) {
  message("  SKIPPED fs2_trigger_by_technology.csv (missing fig5_trigger_by_technology.rds",
          " — re-run the deliverable's figure script to regenerate; committed CSV kept)")
})

# ===========================================================================
# FS5: Determinations of Significance Across Resource Areas (D2)
# Sources: phase2/output/deliverable02/analysis/ (figures + summary CSVs) and
#          phase2/data/analysis/deliverable02/ (validation parquets).
# Figure numbers below refer to the D2 report's rendered figure order.
# ===========================================================================
message("\n--- FS5: significance figures (D2) ---")

D02_ANALYSIS <- here("phase2", "output", "deliverable02", "analysis")
D02_DATA     <- here("phase2", "data", "analysis", "deliverable02")

# --- COPIED passthrough figures (kept at their D2 basenames) ---
fs5_passthrough <- file.path(D02_ANALYSIS, paste0(c(
  "fig_corpus_overview",         # Fig 1  — scale of analysis (methodology)
  "fig_outcomes_by_resource",    # Fig 5  — how agencies stay below the line
  "fig_mitigation_by_resource",  # Fig 8  — mitigation share by resource
  "fig_mitigation_landscape",    # Fig 9  — volume x mitigation-intensity
  "fig_dept_by_resource",        # Fig 11 — mitigation by agency (BLM vs DOE)
  "fig_fonsi_technology",        # Fig 13 — mitigation by technology
  "fig_fonsi_enforceability",    # Fig 14 — enforceability of mitigation
  "fig_eis_unavoidable",         # Fig 18 — significant & unavoidable counts
  "fig_eis_breadth",             # Fig 19 — how broadly an EIS crosses the line
  "fig_eis_factors",             # Fig 21 — why an impact is significant
  "fig_eis_by_agency",           # Fig 23 — significance by agency
  "fig_eis_technology",          # Fig 24 — significance by technology
  "fig_eis_mitigable"            # Fig 26 — significant but reducible
), ".png"))
missing5 <- fs5_passthrough[!file.exists(fs5_passthrough)]
if (length(missing5)) {
  warning("FS5 missing source figures (skipped):\n  ", paste(missing5, collapse = "\n  "))
}
invisible(file.copy(fs5_passthrough[file.exists(fs5_passthrough)], fig_dir, overwrite = TRUE))
message("  Copied ", sum(file.exists(fs5_passthrough)), " FS5 passthrough figures")

# --- RETITLED headline figures (readRDS the D2 .rds sidecar + labs()) ---
# Fig 7 — mitigated share
retitle(file.path(D02_ANALYSIS, "fig_mitigated_share.rds"),
        "fs5_mitigated_share.png",
        "Most Decarbonization FONSIs Reach \"No Significant Impact\"\nOnly With Committed Mitigation",
        width = 8, height = 4.8)

# Fig 17 — which resources cross the line
retitle(file.path(D02_ANALYSIS, "fig_eis_above_line.rds"),
        "fs5_eis_above_line.png",
        "Visual Impacts Cross the Significance Line\nMore Than Any Other Resource",
        width = 9.5, height = 6.5)

# Fig 20 — two ways a resource can be a problem
retitle(file.path(D02_ANALYSIS, "fig_fonsi_vs_eis.rds"),
        "fs5_fonsi_vs_eis.png",
        "Some Resources Get Mitigated Below the Line;\nOthers Cross It",
        width = 9, height = 6.5)

# --- Staged summary CSVs for inline numbers + example tables ---
fs5_tables <- c(
  "mitigation_document_level.csv",        # FONSI docs/projects + mitigated share
  "mitigation_by_resource.csv",           # top mitigation-driving resources
  "mitigation_resource_match_overall.csv",# same-resource match share
  "mitigation_examples.csv",              # Table 2 (subset in QMD)
  "eis_coverage_funnel.csv",              # EIS funnel counts
  "eis_class_distribution.csv",           # EIS significant / unavoidable counts
  "eis_mitigation_document_level.csv",    # EIS projects/documents
  "eis_resource_significance.csv",        # resources most likely to cross
  "eis_significance_factors.csv",         # leading significance factors
  "eis_agency.csv",                       # significance spread by agency
  "eis_technology.csv",                   # significance by technology
  "eis_unavoidable_examples.csv",         # Table 4 (subset in QMD)
  "eis_factor_examples.csv"               # Table 5 (subset in QMD)
)
missing5t <- fs5_tables[!file.exists(file.path(D02_ANALYSIS, fs5_tables))]
if (length(missing5t)) {
  warning("FS5 missing source tables (skipped):\n  ", paste(missing5t, collapse = "\n  "))
}
invisible(file.copy(file.path(D02_ANALYSIS, fs5_tables), tbl_dir, overwrite = TRUE))
message("  Staged ", sum(file.exists(file.path(D02_ANALYSIS, fs5_tables))), " FS5 summary tables")

# --- Derived validation summary (held-out F1 from the two D2 validation parquets;
#     the held-out test is the honest score the D2 report headlines) ---
val_metric_keep <- c(candidate_is_determination       = "finds",
                     resource_determination_detection  = "resource",
                     determination_class_macro_f1       = "class")
read_val <- function(path, track) {
  read_parquet(path) |>
    mutate(score = coalesce(f1, precision)) |>
    filter(scope == "holdout", metric %in% names(val_metric_keep)) |>
    transmute(track = track, metric = recode(metric, !!!val_metric_keep), score)
}
fs5_val <- bind_rows(
  read_val(file.path(D02_DATA, "validation_metrics.parquet"),     "fonsi"),
  read_val(file.path(D02_DATA, "validation_metrics_eis.parquet"), "eis")
)
write_csv(fs5_val, file.path(tbl_dir, "fs5_validation.csv"))
message("  Saved table: fs5_validation.csv")

message("\nDone. Figures: ", fig_dir, "\nTables:  ", tbl_dir)
