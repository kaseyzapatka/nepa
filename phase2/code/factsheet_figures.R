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

fs1_dur <- readRDS(file.path(D04_FIG, "fig_d4_duration_summary_intervals.rds")) +
  labs(title = "NEPA Review Duration Climbs From Days (CE) to Months (EA) to Years (EIS)")

ggsave(file.path(fig_dir, "fs1_duration_by_process.png"), fs1_dur,
       width = 10, height = 5.5, dpi = 300)
message("  Saved: fs1_duration_by_process.png")

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

tl <- read_parquet(
  here("phase2", "data", "analysis", "timeline", "timeline_project_dates.parquet"),
  col_select = c("project_id", "process_type", "timeline_status", "duration_days")
) |>
  filter(timeline_status %in% c("complete_clear", "complete_with_proxy"),
         !is.na(duration_days), duration_days >= 0)

tech <- read_parquet(
  here("phase2", "data", "analysis", "deliverable03", "projects_nepa_reviews.parquet"),
  col_select = c("project_id", "tech_group", "energy_group")
) |>
  filter(energy_group %in% c("Decarbonization", "Fossil Fuel"), !is.na(tech_group))

tl_tech <- tl |>
  inner_join(tech, by = "project_id") |>
  mutate(duration_months = duration_days / 30.44,
         process_group = factor(process_type, levels = process_levels))

tech_summary <- tl_tech |>
  group_by(energy_group, tech_group, process_group) |>
  summarise(
    n      = n(),
    p10    = quantile(duration_months, 0.10),
    p25    = quantile(duration_months, 0.25),
    median = median(duration_months),
    p75    = quantile(duration_months, 0.75),
    p90    = quantile(duration_months, 0.90),
    .groups = "drop"
  ) |>
  filter(n >= 15)   # suppress unstable cells

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
    subtitle = "Thin bar = 10th–90th percentile; thick bar = interquartile range; point = median. Complete timelines; cells with n < 15 suppressed.",
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

# [RETITLED] readRDS D1's fig1_trigger_counts.rds and override the title.
fs2_counts <- readRDS(file.path(D01_OUT, "fig1_trigger_counts.rds")) +
  labs(title = sprintf(
    "Federal Funding Triggers %s of All Decarbonization NEPA Reviews —\nMore Than Any Other Federal Nexus",
    percent(funding_share, accuracy = 1)))

ggsave(file.path(fig_dir, "fs2_trigger_counts.png"), fs2_counts,
       width = 10, height = 5.5, dpi = 300)
message("  Saved: fs2_trigger_counts.png")

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
fs2_proc <- readRDS(file.path(D01_OUT, "fig3_process_by_trigger.rds")) +
  labs(title = sprintf(
    "Funding-Triggered Reviews Are Almost Entirely Categorically Excluded (%s)",
    percent(funding_ce_share, accuracy = 1)))

ggsave(file.path(fig_dir, "fs2_process_by_trigger.png"), fs2_proc,
       width = 10, height = 5.5, dpi = 300)
message("  Saved: fs2_process_by_trigger.png")

message("\nDone. Figures: ", fig_dir, "\nTables:  ", tbl_dir)
