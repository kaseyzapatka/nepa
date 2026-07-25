# D4 solar duration analysis: solar-tagged projects vs the all-decarbonization reference
#
# Phase 2 re-creation of the Phase 1 solar timeline analysis
# (phase1/code/deliverable03/04_solar_figures.R Figure A and the factsheet's solar
# duration chart), sourced entirely from the Phase 2 timeline —
# timeline_project_dates.parquet — with the solar tag and decarb scope taken from
# the Phase 2 projects_combined.parquet. No Phase 1 file is read.
#
# Duration frame: identical to 08_create_figures.R's headline frame — complete_clear +
# complete_with_proxy, YEAR-granularity endpoints excluded, month-granularity
# imputed to the mid-month 15th, non-negative durations. NOTE: the plan file
# (plans/deliverable04_solar.md) specified the parquet's raw `duration_days`
# column instead. As of the 2026-07-24 finalize fix that column is current for
# day/day pairs (05/06 run finalize_duration_days), but this recomputed frame is
# still used because it additionally applies the month->15th midpoints the raw
# column intentionally leaves null — matching 08_create_figures.R's headline frame.
#
# Outputs:
#   phase2/output/deliverable04/figures/fig_d4_solar_duration.png
#   phase2/output/deliverable04/diagnostics/d4_solar_duration.csv
#
# Usage: Rscript phase2/code/deliverable04/08_create_figures_solar.R

suppressPackageStartupMessages({
  library(here); library(arrow); library(dplyr); library(tidyr)
  library(readr); library(stringr); library(lubridate); library(ggplot2); library(scales)
})

PHASE2 <- here::here("phase2")
TL     <- file.path(PHASE2, "data", "analysis", "timeline")
FIG    <- file.path(PHASE2, "output", "deliverable04", "figures")
DIAG   <- file.path(PHASE2, "output", "deliverable04", "diagnostics")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
dir.create(DIAG, recursive = TRUE, showWarnings = FALSE)

SOLAR_TAG      <- "Renewable Energy Production - Solar"
PROCESS_LEVELS <- c("CE", "EA", "EIS")

catf_navy <- "#002169"; catf_dark_blue <- "#0047BB"; catf_light_blue <- "#8AB7E9"
catf_lime <- "#93D500"
PROCESS_COLORS <- c("CE" = catf_lime, "EA" = catf_dark_blue, "EIS" = catf_navy)
theme_catf <- function(base = 11) {
  theme_minimal(base_size = base) +
    theme(plot.title = element_text(face = "bold", color = catf_navy, size = rel(1.2)),
          plot.subtitle = element_text(color = catf_dark_blue, size = rel(0.9)),
          plot.caption = element_text(color = "gray50", size = rel(0.8), hjust = 0),
          panel.grid.minor = element_blank())
}

# ---------------------------------------------------------------------------
# Assemble: headline duration frame (mirrors 08_create_figures.R) + project tags
# ---------------------------------------------------------------------------
dates <- read_parquet(
  file.path(TL, "timeline_project_dates.parquet"),
  col_select = c("project_id", "process_type", "timeline_status",
                 "initiation_date", "decision_date",
                 "initiation_date_granularity", "decision_date_granularity")
) |>
  mutate(initiation_date = as.Date(initiation_date),
         decision_date   = as.Date(decision_date))

tags <- read_parquet(
  file.path(PHASE2, "data", "analysis", "projects_combined.parquet"),
  col_select = c("project_id", "project_type", "project_energy_type")
) |>
  mutate(is_solar  = str_detect(as.character(project_type), fixed(SOLAR_TAG)),
         is_decarb = coalesce(project_energy_type == "Clean", FALSE))

frame <- dates |>
  inner_join(tags, by = "project_id") |>
  filter(timeline_status %in% c("complete_clear", "complete_with_proxy"),
         !is.na(initiation_date), !is.na(decision_date),
         initiation_date_granularity != "year", decision_date_granularity != "year") |>
  mutate(
    .init_mid = if_else(initiation_date_granularity == "month",
                        lubridate::floor_date(initiation_date, "month") + 14, initiation_date),
    .dec_mid  = if_else(decision_date_granularity == "month",
                        lubridate::floor_date(decision_date, "month") + 14, decision_date),
    duration_days   = as.integer(.dec_mid - .init_mid),
    duration_months = duration_days / 30.44,
    process_group   = factor(process_type, levels = PROCESS_LEVELS)
  ) |>
  filter(!is.na(duration_months), duration_months >= 0, !is.na(process_group))

interval_stats <- function(df, group_label) {
  df |>
    group_by(process_group) |>
    summarise(
      n             = n(),
      p10           = quantile(duration_months, 0.10, na.rm = TRUE),
      p25           = quantile(duration_months, 0.25, na.rm = TRUE),
      median_months = median(duration_months, na.rm = TRUE),
      p75           = quantile(duration_months, 0.75, na.rm = TRUE),
      p90           = quantile(duration_months, 0.90, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(group = group_label)
}

solar_summary  <- interval_stats(filter(frame, is_solar),  "solar")
decarb_summary <- interval_stats(filter(frame, is_decarb), "decarb_reference")

message("Solar complete timelines by process:")
print(as.data.frame(solar_summary[, c("process_group", "n", "median_months")]))
message("All-decarbonization reference medians:")
print(as.data.frame(decarb_summary[, c("process_group", "n", "median_months")]))

write_csv(bind_rows(solar_summary, decarb_summary),
          file.path(DIAG, "d4_solar_duration.csv"))
message("Wrote d4_solar_duration.csv")

# ---------------------------------------------------------------------------
# Figure: solar interval chart (p10-p90, IQR, median) with decarb reference
# ticks on EA/EIS. Mirrors 08_create_figures.R Fig 6 geometry (Phase 1 ref:
# 03_duration_summary_intervals_by_process_solar.png + factsheet Fig 3).
# CE reference omitted: solar CE and decarb CE medians are both ~1 month.
# ---------------------------------------------------------------------------
solar_plot <- solar_summary |>
  mutate(
    median_label = case_when(
      median_months < 1  ~ sprintf("%s: ~1 month", process_group),
      median_months < 12 ~ sprintf("%s: ~%.0f months", process_group, median_months),
      TRUE               ~ sprintf("%s: ~%.0f months (%.1f years)", process_group,
                                   median_months, median_months / 12)
    ),
    label_hjust = if_else(median_months < 3, 0, 0.5)
  )

ref_plot <- decarb_summary |>
  filter(process_group %in% c("EA", "EIS")) |>
  mutate(y = as.numeric(factor(process_group, levels = PROCESS_LEVELS)),
         ref_label = sprintf("All decarb: ~%.0f months", median_months))

fig_solar <- ggplot(solar_plot, aes(y = process_group, color = process_group)) +
  geom_segment(aes(x = p10, xend = p90, yend = process_group), linewidth = 1.8, alpha = 0.35) +
  geom_segment(aes(x = p25, xend = p75, yend = process_group), linewidth = 5.5, alpha = 0.55) +
  geom_point(aes(x = median_months), size = 3.2) +
  geom_text(aes(x = median_months, label = median_label, hjust = label_hjust),
            nudge_y = 0.28, size = 3.2, fontface = "bold", color = "gray20") +
  geom_text(aes(x = p90, label = paste0("n=", comma(n))),
            nudge_x = 1.2, hjust = 0, size = 3, color = "gray30") +
  geom_segment(data = ref_plot, inherit.aes = FALSE, linetype = "dashed", color = "gray35",
               aes(x = median_months, xend = median_months, y = y - 0.35, yend = y + 0.35)) +
  geom_text(data = ref_plot, inherit.aes = FALSE,
            aes(x = median_months, y = y - 0.42, label = ref_label),
            size = 2.9, color = "gray35", hjust = 0.5) +
  scale_color_manual(values = PROCESS_COLORS, drop = FALSE) +
  scale_x_continuous(labels = label_number(accuracy = 1),
                     expand = expansion(mult = c(0.02, 0.12))) +
  labs(
    title    = "Timeline Duration by Review Process — Solar Projects",
    subtitle = "Thin bar = p10-p90, thick bar = IQR (p25-p75), point = median; dashed tick = all-decarbonization median",
    x = "Duration (months)", y = "Review Process", color = NULL
  ) +
  theme_catf() +
  theme(legend.position = "none")

ggsave(file.path(FIG, "fig_d4_solar_duration.png"), fig_solar,
       width = 10, height = 6, dpi = 300)
saveRDS(fig_solar, file.path(FIG, "fig_d4_solar_duration.rds"))
message("Wrote fig_d4_solar_duration.png")
