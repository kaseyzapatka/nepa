# D4: Timeline Duration Analysis
#
# Reads timeline_project_dates.parquet and produces headline duration tables,
# coverage diagnostics, regulatory-period comparisons, and all main figures
# (including recreations of the Phase 1 D3 timeline charts adapted for Phase 2 schema).
#
# FRA breakpoint: 2023-08-16 (CEQ final rule effective date)
# Legislative markers: ARRA 2009, BIL 2021, IRA 2022
#
# Output tables (phase2/output/deliverable04/diagnostics/):
#   d4_duration_summary.csv          — headline medians/percentiles by process
#   d4_duration_by_period.csv        — medians by process × regulatory period
#   d4_endpoint_coverage.csv         — ROD vs FEIS-fallback endpoint counts
#   d4_coverage_by_process.csv       — timeline_status counts by process
#   d4_coverage_diagnostics.csv      — initiation/decision/complete_clear rates
#   d4_proxy_sensitivity.csv         — complete_clear vs complete_with_proxy comparison
#   d4_duration_by_year.csv          — median duration per process × decision year
#   d4_fra_comparison.csv            — pre/post FRA median comparison
#   d4_flag_summary.csv              — quality flag counts by process
#
# Output figures (phase2/output/deliverable04/figures/):
#   fig_d4_coverage_by_process.png             — stacked bar: both/decision/initiation/none
#   fig_d4_duration_histogram.png              — duration histogram by process (complete_clear)
#   fig_d4_fra_comparison.png                  — median duration pre vs post FRA
#   fig_d4_duration_trend.png                  — median duration trend by year
#   fig_d4_complete_timeline_share_boxplot.png — binary completion rate per process
#   fig_d4_duration_summary_intervals.png      — p10–p90 interval chart by process
#   fig_d4_project_timeline_spans.png          — horizontal span chart, init→decision
#   fig_d4_projects_by_decision_year.png       — project counts by year, faceted by process
#
# Energy-type breakout figures (Clean / Fossil / Other):
#   fig_d4_duration_histogram_by_energy.png         — 3×3 grid: process × energy type
#   fig_d4_duration_summary_intervals_by_energy.png — interval bars faceted by process, colored by energy
#   fig_d4_fra_comparison_by_energy.png             — pre/post FRA grid: process × energy type
#   fig_d4_projects_by_decision_year_by_energy.png  — stacked year bars by energy type
#
# Usage:
#   Rscript phase2/code/deliverable04/08_analyze.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(lubridate)
  library(arrow)
  library(ggplot2)
  library(scales)
  library(stringr)
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PHASE2  <- here::here("phase2")
DATA    <- file.path(PHASE2, "data", "analysis", "timeline")
OUTPUT  <- file.path(PHASE2, "output", "deliverable04")
DIAG    <- file.path(OUTPUT, "diagnostics")
FIGS    <- file.path(OUTPUT, "figures")
dir.create(DIAG, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROCESS_LEVELS    <- c("CE", "EA", "EIS")
FRA_CUT_DATE      <- as.Date("2023-08-16")
ARRA_DATE         <- as.Date("2009-02-17")
BIL_DATE          <- as.Date("2021-11-15")
IRA_DATE          <- as.Date("2022-08-16")
PROXY_SENSITIVITY <- TRUE

# ---------------------------------------------------------------------------
# CATF brand colors and theme (from Phase 1 brand guide)
# ---------------------------------------------------------------------------

catf_dark_blue  <- "#0047BB"
catf_blue       <- "#00B5E2"
catf_magenta    <- "#C22A90"
catf_purple     <- "#75246C"
catf_lime       <- "#93D500"
catf_teal       <- "#00AE8D"
catf_light_blue <- "#8AB7E9"
catf_navy       <- "#012169"

PROCESS_COLORS <- c("CE" = catf_lime, "EA" = catf_dark_blue, "EIS" = catf_navy)
ENERGY_LEVELS  <- c("Decarb", "Fossil", "Other")
ENERGY_COLORS  <- c("Decarb" = catf_teal, "Fossil" = catf_magenta, "Other" = catf_light_blue)

theme_catf <- function(base_size = 11, base_family = "Helvetica") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      plot.title       = element_text(face = "bold", size = rel(1.2), color = catf_navy,
                                      margin = margin(b = 10)),
      plot.subtitle    = element_text(size = rel(0.9), color = catf_dark_blue,
                                      margin = margin(b = 10)),
      plot.caption     = element_text(size = rel(0.8), color = "gray50", hjust = 1),
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

theme_set(theme_catf())

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

message("Loading timeline_project_dates.parquet...")
dates_raw <- read_parquet(file.path(DATA, "timeline_project_dates.parquet"))
message("  ", nrow(dates_raw), " rows")

# Join energy type from Phase 1 combined project file
energy_meta <- read_parquet(
  here::here("phase1", "data", "analysis", "projects_combined.parquet"),
  col_select = c("project_id", "project_energy_type")
)
dates_raw <- dates_raw |>
  left_join(energy_meta, by = "project_id")
message("  Energy type joined (", sum(!is.na(dates_raw$project_energy_type)), " matched)")

burden <- tryCatch({
  idx <- read_parquet(file.path(DATA, "timeline_document_index.parquet"),
                      col_select = c("project_id", "process_type", "project_doc_count",
                                     "total_pages", "scan_priority")) |>
    distinct(project_id, .keep_all = TRUE)
  message("  Joined document burden from timeline_document_index.parquet")
  idx
}, error = function(e) NULL)

# ---------------------------------------------------------------------------
# Clean and derive fields
# ---------------------------------------------------------------------------

dates <- dates_raw |>
  mutate(
    initiation_date = as.Date(initiation_date),
    decision_date   = as.Date(decision_date),
    duration_days   = as.integer(duration_days),

    process_group = factor(process_type, levels = PROCESS_LEVELS),

    reg_period = case_when(
      decision_date >= FRA_CUT_DATE          ~ "post_FRA",
      decision_date >= IRA_DATE              ~ "IRA",
      decision_date >= BIL_DATE              ~ "BIL",
      decision_date >= ARRA_DATE             ~ "post_ARRA",
      decision_date >= as.Date("2000-01-01") ~ "pre_ARRA_2000s",
      TRUE                                   ~ "pre_2000"
    ),
    reg_period = factor(reg_period, levels = c(
      "pre_2000", "pre_ARRA_2000s", "post_ARRA", "BIL", "IRA", "post_FRA"
    )),

    decision_year = year(decision_date),

    has_proxy_flag = str_detect(coalesce(timeline_flags, ""), "proxy"),
    is_proxy_only  = str_detect(coalesce(timeline_flags, ""), "proxy_only"),

    final_eis_date = as.Date(final_eis_date),
    endpoint_date  = coalesce(decision_date, final_eis_date),
    endpoint_source_type = case_when(
      !is.na(decision_date)  ~ "decision",
      !is.na(final_eis_date) ~ "final_eis",
      TRUE                   ~ NA_character_
    ),
    endpoint_date_granularity = case_when(
      !is.na(decision_date)  ~ decision_date_granularity,
      !is.na(final_eis_date) ~ final_eis_date_granularity,
      TRUE                   ~ "unknown"
    ),
    endpoint_duration_days = if_else(
      !is.na(initiation_date) & !is.na(endpoint_date) &
        initiation_date_granularity == "day" & endpoint_date_granularity == "day" &
        endpoint_date >= initiation_date,
      as.integer(endpoint_date - initiation_date),
      NA_integer_
    ),

    # Complete = both dates present at any granularity (used in coverage charts)
    timeline_complete = !is.na(initiation_date) & !is.na(decision_date),

    energy_type = factor(
      dplyr::recode(coalesce(project_energy_type, "Other"), "Clean" = "Decarb"),
      levels = ENERGY_LEVELS
    ),
  )

# Headline: complete_clear = both dates at day granularity (used for duration analysis)
headline <- dates |>
  filter(timeline_status == "complete_clear", !is.na(duration_days))

message("complete_clear rows with duration: ", nrow(headline))

# ---------------------------------------------------------------------------
# Helper: duration summary stats
# ---------------------------------------------------------------------------

duration_summary_stats <- function(df, group_vars) {
  df |>
    group_by(across(all_of(group_vars))) |>
    summarise(
      n             = n(),
      median_days   = median(duration_days, na.rm = TRUE),
      p10_days      = quantile(duration_days, 0.10, na.rm = TRUE),
      p25_days      = quantile(duration_days, 0.25, na.rm = TRUE),
      p75_days      = quantile(duration_days, 0.75, na.rm = TRUE),
      p90_days      = quantile(duration_days, 0.90, na.rm = TRUE),
      mean_days     = mean(duration_days, na.rm = TRUE),
      pct_lt_1y     = mean(duration_days < 365, na.rm = TRUE),
      pct_gt_5y     = mean(duration_days > 5 * 365, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      median_months = round(median_days / 30.44, 1),
      p10_months    = round(p10_days    / 30.44, 1),
      p90_months    = round(p90_days    / 30.44, 1),
    )
}

# ---------------------------------------------------------------------------
# 1. Headline duration summary (complete_clear)
# ---------------------------------------------------------------------------

dur_process <- duration_summary_stats(headline, "process_type")
dur_period  <- duration_summary_stats(headline, c("process_type", "reg_period"))

write_csv(dur_process, file.path(DIAG, "d4_duration_summary.csv"))
message("Wrote d4_duration_summary.csv")

write_csv(dur_period, file.path(DIAG, "d4_duration_by_period.csv"))
message("Wrote d4_duration_by_period.csv")

# ---------------------------------------------------------------------------
# 1b. Endpoint coverage (ROD vs FEIS fallback)
# ---------------------------------------------------------------------------

endpoint_coverage <- dates |>
  group_by(process_type, endpoint_source_type) |>
  summarise(
    n_projects           = n(),
    n_with_endpoint      = sum(!is.na(endpoint_date)),
    n_feis_proxy         = sum(coalesce(final_eis_is_proxy, FALSE) &
                                 endpoint_source_type == "final_eis", na.rm = TRUE),
    n_day_duration       = sum(!is.na(endpoint_duration_days)),
    median_endpoint_days = median(endpoint_duration_days, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(endpoint_coverage, file.path(DIAG, "d4_endpoint_coverage.csv"))
message("Wrote d4_endpoint_coverage.csv")

# ---------------------------------------------------------------------------
# 2. Coverage diagnostics
# ---------------------------------------------------------------------------

coverage <- dates |>
  group_by(process_type, timeline_status) |>
  summarise(n = n(), .groups = "drop") |>
  group_by(process_type) |>
  mutate(
    total_process = sum(n),
    pct           = round(100 * n / total_process, 1)
  ) |>
  ungroup()

coverage_energy <- dates |>
  mutate(
    has_initiation    = !is.na(initiation_date),
    has_decision      = !is.na(decision_date),
    is_complete_clear = timeline_status == "complete_clear",
  ) |>
  group_by(process_type) |>
  summarise(
    n_total            = n(),
    n_initiation       = sum(has_initiation),
    n_decision         = sum(has_decision),
    n_complete_clear   = sum(is_complete_clear),
    pct_initiation     = round(100 * mean(has_initiation), 1),
    pct_decision       = round(100 * mean(has_decision), 1),
    pct_complete_clear = round(100 * mean(is_complete_clear), 1),
    .groups = "drop"
  )

write_csv(coverage,        file.path(DIAG, "d4_coverage_by_process.csv"))
write_csv(coverage_energy, file.path(DIAG, "d4_coverage_diagnostics.csv"))
message("Wrote d4_coverage_by_process.csv, d4_coverage_diagnostics.csv")

# ---------------------------------------------------------------------------
# 3. Proxy sensitivity
# ---------------------------------------------------------------------------

if (PROXY_SENSITIVITY) {
  proxy_dates <- dates |>
    filter(
      timeline_status %in% c("complete_clear", "complete_with_proxy"),
      !is.na(initiation_date), !is.na(decision_date)
    ) |>
    mutate(
      duration_days_approx = as.integer(decision_date - initiation_date),
      uses_proxy           = timeline_status == "complete_with_proxy"
    )

  sensitivity_summary <- proxy_dates |>
    group_by(process_type, uses_proxy) |>
    summarise(
      n           = n(),
      median_days = median(duration_days_approx, na.rm = TRUE),
      p25_days    = quantile(duration_days_approx, 0.25, na.rm = TRUE),
      p75_days    = quantile(duration_days_approx, 0.75, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(dataset = if_else(uses_proxy, "with_proxy", "clear_only"))

  write_csv(sensitivity_summary, file.path(DIAG, "d4_proxy_sensitivity.csv"))
  message("Wrote d4_proxy_sensitivity.csv")
}

# ---------------------------------------------------------------------------
# 4. Duration by decision year
# ---------------------------------------------------------------------------

dur_year <- headline |>
  filter(!is.na(decision_year), decision_year >= 1990, decision_year <= year(Sys.Date())) |>
  group_by(process_type, decision_year) |>
  summarise(
    n           = n(),
    median_days = median(duration_days, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(median_months = round(median_days / 30.44, 1))

write_csv(dur_year, file.path(DIAG, "d4_duration_by_year.csv"))
message("Wrote d4_duration_by_year.csv")

# ---------------------------------------------------------------------------
# 5. Pre/post FRA comparison
# ---------------------------------------------------------------------------

fra_comparison <- headline |>
  filter(!is.na(decision_date)) |>
  mutate(period = if_else(decision_date >= FRA_CUT_DATE, "post_FRA", "pre_FRA")) |>
  group_by(process_type, period) |>
  summarise(
    n           = n(),
    median_days = median(duration_days, na.rm = TRUE),
    p25_days    = quantile(duration_days, 0.25, na.rm = TRUE),
    p75_days    = quantile(duration_days, 0.75, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(median_months = round(median_days / 30.44, 1))

write_csv(fra_comparison, file.path(DIAG, "d4_fra_comparison.csv"))
message("Wrote d4_fra_comparison.csv")

# ---------------------------------------------------------------------------
# 6. Quality flags summary
# ---------------------------------------------------------------------------

flag_summary <- dates |>
  filter(!is.na(timeline_flags), timeline_flags != "") |>
  mutate(flag_list = str_split(timeline_flags, "\\|")) |>
  tidyr::unnest(flag_list) |>
  filter(flag_list != "") |>
  group_by(process_type, flag_list) |>
  summarise(n = n(), .groups = "drop") |>
  arrange(process_type, desc(n))

write_csv(flag_summary, file.path(DIAG, "d4_flag_summary.csv"))
message("Wrote d4_flag_summary.csv")

# ---------------------------------------------------------------------------
# 7. Console summary
# ---------------------------------------------------------------------------

cat("\n=== D4 TIMELINE ANALYSIS SUMMARY ===\n\n")

cat("Coverage by process:\n")
print(
  dates |>
    group_by(process_type, timeline_status) |>
    summarise(n = n(), .groups = "drop") |>
    pivot_wider(names_from = timeline_status, values_from = n, values_fill = 0)
)

cat("\nHeadline durations (complete_clear only):\n")
print(
  headline |>
    group_by(process_type) |>
    summarise(
      n             = n(),
      median_days   = median(duration_days),
      median_months = round(median(duration_days) / 30.44, 1),
      p10_days      = quantile(duration_days, 0.10),
      p90_days      = quantile(duration_days, 0.90),
      .groups = "drop"
    )
)

cat("\nFRA period comparison (post 2023-08-16 vs prior):\n")
print(fra_comparison |> select(process_type, period, n, median_months, p25_days, p75_days))

cat("\nAll output files written to:", OUTPUT, "\n")

# ===========================================================================
# FIGURES
# ===========================================================================
#
# Figs 1–4: D4-specific analyses (coverage breakdown, histogram, FRA, trend)
# Figs 5–9: Phase 1 D3 timeline charts recreated for Phase 2 schema

# ---------------------------------------------------------------------------
# Fig 1: Coverage stacked bar — both dates / decision only / initiation only / none
# ---------------------------------------------------------------------------

coverage_fig <- dates |>
  mutate(
    coverage_group = case_when(
      !is.na(decision_date) & !is.na(initiation_date) ~ "Both dates",
      !is.na(decision_date)                            ~ "Decision only",
      !is.na(initiation_date)                          ~ "Initiation only",
      TRUE                                             ~ "No date"
    ),
    coverage_group = factor(coverage_group,
      levels = c("Both dates", "Decision only", "Initiation only", "No date"))
  ) |>
  count(process_type, coverage_group) |>
  group_by(process_type) |>
  mutate(pct = n / sum(n)) |>
  ungroup()

p_coverage <- ggplot(coverage_fig, aes(x = process_type, y = pct, fill = coverage_group)) +
  geom_col(width = 0.6) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_fill_manual(values = c(
    "Both dates"      = catf_navy,
    "Decision only"   = catf_dark_blue,
    "Initiation only" = catf_light_blue,
    "No date"         = "#CCCCCC"
  )) +
  labs(
    title    = "D4 Timeline Coverage by Review Type",
    subtitle = "Share of projects by timeline completeness category",
    x = NULL, y = "Share of projects", fill = NULL
  )

ggsave(file.path(FIGS, "fig_d4_coverage_by_process.png"),
       p_coverage, width = 7, height = 5, dpi = 150)
message("Wrote fig_d4_coverage_by_process.png")

# ---------------------------------------------------------------------------
# Fig 2: Duration histogram by process (complete_clear, day granularity)
# ---------------------------------------------------------------------------

dur_plot <- headline |>
  filter(duration_days > 0, duration_days < 365 * 15) |>
  mutate(
    duration_years = duration_days / 365.25,
    process_group  = factor(process_type, levels = PROCESS_LEVELS)
  )

p_hist <- ggplot(dur_plot, aes(x = duration_years, fill = process_group)) +
  geom_histogram(bins = 40, color = "white", linewidth = 0.2) +
  facet_wrap(~process_group, scales = "free_y", ncol = 1, drop = FALSE) +
  scale_fill_manual(values = PROCESS_COLORS, guide = "none") +
  scale_x_continuous(breaks = 0:15, labels = function(x) paste0(x, "y")) +
  labs(
    title = "D4 Review Duration Distribution (complete_clear only)",
    x = "Duration (years)", y = "Reviews"
  )

ggsave(file.path(FIGS, "fig_d4_duration_histogram.png"),
       p_hist, width = 7, height = 8, dpi = 150)
message("Wrote fig_d4_duration_histogram.png")

# ---------------------------------------------------------------------------
# Fig 3: Median review duration pre vs post FRA (Aug 16, 2023)
# ---------------------------------------------------------------------------

fra_fig <- fra_comparison |>
  mutate(period = factor(period, levels = c("pre_FRA", "post_FRA"),
                         labels = c("Pre-FRA\n(before Aug 2023)", "Post-FRA\n(Aug 2023+)")))

p_fra <- ggplot(fra_fig, aes(x = period, y = median_months, fill = period)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(round(median_months, 1), " mo\n(n=", n, ")")),
            vjust = -0.3, size = 3.2) +
  facet_wrap(~process_type, ncol = 3) +
  scale_fill_manual(
    values = c("Pre-FRA\n(before Aug 2023)" = catf_light_blue,
               "Post-FRA\n(Aug 2023+)"      = catf_dark_blue),
    guide = "none"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(
    title = "D4 Median Review Duration: Pre vs Post FRA (Aug 16, 2023)",
    x = NULL, y = "Median duration (months)"
  )

ggsave(file.path(FIGS, "fig_d4_fra_comparison.png"),
       p_fra, width = 9, height = 5, dpi = 150)
message("Wrote fig_d4_fra_comparison.png")

# ---------------------------------------------------------------------------
# Fig 4: Median duration trend by decision year (complete_clear, n >= 5)
# ---------------------------------------------------------------------------

leg_vlines <- c(ARRA = 2009, BIL = 2021, IRA = 2022, FRA = 2023.6)

p_trend <- ggplot(dur_year |> filter(n >= 5),
                  aes(x = decision_year, y = median_months, color = process_type)) +
  geom_line(linewidth = 0.8) +
  geom_point(aes(size = n), alpha = 0.7) +
  geom_vline(xintercept = leg_vlines, linetype = "dashed",
             color = "grey50", linewidth = 0.5) +
  annotate("text", x = leg_vlines, y = Inf, label = names(leg_vlines),
           vjust = 1.5, hjust = -0.1, size = 3, color = "grey40") +
  scale_color_manual(values = PROCESS_COLORS) +
  scale_size_continuous(range = c(1, 4), guide = "none") +
  scale_x_continuous(breaks = seq(1990, 2026, 5)) +
  labs(
    title = "D4 Median Review Duration by Year (complete_clear, n≥5)",
    x = "Decision year", y = "Median duration (months)", color = NULL
  )

ggsave(file.path(FIGS, "fig_d4_duration_trend.png"),
       p_trend, width = 10, height = 5, dpi = 150)
message("Wrote fig_d4_duration_trend.png")

# ---------------------------------------------------------------------------
# Fig 5: Complete timeline share by process (boxplot + mean dot)
# Phase 1 ref: 03_complete_timeline_share_boxplot.png
# ---------------------------------------------------------------------------

process_summary_complete <- tibble(process_group = factor(PROCESS_LEVELS, levels = PROCESS_LEVELS)) |>
  left_join(
    dates |>
      filter(!is.na(process_group)) |>
      group_by(process_group) |>
      summarise(
        n_projects     = n(),
        n_complete     = sum(timeline_complete, na.rm = TRUE),
        share_complete = n_complete / n_projects,
        .groups = "drop"
      ),
    by = "process_group"
  ) |>
  mutate(
    n_projects     = replace_na(n_projects, 0L),
    n_complete     = replace_na(n_complete, 0L),
    share_complete = if_else(n_projects > 0, share_complete, NA_real_),
    label = case_when(
      n_projects == 0 ~ "Pending",
      TRUE ~ sprintf("%s/%s (%.0f%%)", comma(n_complete), comma(n_projects),
                     100 * share_complete)
    )
  )

complete_box <- dates |>
  filter(!is.na(process_group)) |>
  mutate(complete_num = as.numeric(timeline_complete))

fig_complete_share <- ggplot(complete_box,
                              aes(x = process_group, y = complete_num, fill = process_group)) +
  geom_boxplot(outlier.shape = NA, width = 0.55, alpha = 0.35, na.rm = TRUE) +
  stat_summary(fun = mean, geom = "point", size = 3, color = catf_navy) +
  geom_text(
    data = process_summary_complete,
    aes(x = process_group, y = 1.07, label = label),
    inherit.aes = FALSE, size = 3, color = "gray30"
  ) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, 1.12),
    breaks = seq(0, 1, by = 0.2)
  ) +
  scale_fill_manual(values = PROCESS_COLORS, drop = FALSE) +
  labs(
    title    = "Share of Reviews with Complete Timelines",
    subtitle = "Boxplot shows review-level completion (0/1); dot is mean share by process",
    x = "Review Process",
    y = "Completion Share"
  ) +
  theme(legend.position = "none")

ggsave(file.path(FIGS, "fig_d4_complete_timeline_share_boxplot.png"),
       fig_complete_share, width = 9, height = 6, dpi = 300)
message("Wrote fig_d4_complete_timeline_share_boxplot.png")

# ---------------------------------------------------------------------------
# Fig 6: Duration summary intervals by process (p10–p90, IQR, median)
# Phase 1 ref: 03_duration_summary_intervals_by_process.png
# ---------------------------------------------------------------------------

interval_df <- headline |>
  mutate(
    duration_months = duration_days / 30.44,
    process_group   = factor(process_type, levels = PROCESS_LEVELS)
  ) |>
  filter(!is.na(duration_months), duration_months >= 0)

interval_summary <- interval_df |>
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
  mutate(
    median_label = case_when(
      median_months < 1  ~ sprintf("%s: < 1 month", process_group),
      median_months < 12 ~ sprintf("%s: ~%.0f months", process_group, median_months),
      TRUE               ~ sprintf("%s: ~%.0f months (%.1f yr)", process_group,
                                   median_months, median_months / 12)
    ),
    label_hjust = if_else(median_months < 3, 0, 0.5)
  )

fig_duration_intervals <- ggplot(interval_summary, aes(y = process_group, color = process_group)) +
  geom_segment(aes(x = p10, xend = p90, yend = process_group), linewidth = 1.8, alpha = 0.35) +
  geom_segment(aes(x = p25, xend = p75, yend = process_group), linewidth = 5.5, alpha = 0.55) +
  geom_point(aes(x = median_months), size = 3.2) +
  geom_text(
    aes(x = median_months, label = median_label, hjust = label_hjust),
    nudge_y = 0.28, size = 3.2, fontface = "bold", color = "gray20"
  ) +
  geom_text(
    aes(x = p90, label = paste0("n=", comma(n))),
    nudge_x = 0.5, hjust = 0, size = 3, color = "gray30"
  ) +
  scale_color_manual(values = PROCESS_COLORS, drop = FALSE) +
  scale_x_continuous(
    labels = label_number(accuracy = 1),
    expand = expansion(mult = c(0.02, 0.12))
  ) +
  labs(
    title    = "Timeline Duration Summary by Review Process",
    subtitle = "Thin bar = p10–p90  |  Thick bar = IQR (p25–p75)  |  Point = median (complete_clear only)",
    x = "Duration (months)",
    y = "Review Process",
    color = NULL
  ) +
  theme(legend.position = "none")

ggsave(file.path(FIGS, "fig_d4_duration_summary_intervals.png"),
       fig_duration_intervals, width = 10, height = 6, dpi = 300)
message("Wrote fig_d4_duration_summary_intervals.png")

# ---------------------------------------------------------------------------
# Fig 7: Project initiation → decision timeline spans (faceted by process)
# Phase 1 ref: 03_project_timeline_spans_by_process.png
# ---------------------------------------------------------------------------

max_spans_per_process <- 300

span_colors <- c(
  "< 6 months"  = catf_lime,
  "6-12 months" = catf_teal,
  "1-2 years"   = catf_dark_blue,
  "2-5 years"   = catf_purple,
  ">= 5 years"  = catf_navy
)

spans_df <- dates |>
  filter(
    !is.na(process_group),
    timeline_status == "complete_clear",
    !is.na(initiation_date),
    !is.na(decision_date),
    decision_date >= initiation_date
  ) |>
  mutate(
    duration_months = as.numeric(decision_date - initiation_date) / 30.44,
    duration_bin = case_when(
      duration_months < 6  ~ "< 6 months",
      duration_months < 12 ~ "6-12 months",
      duration_months < 24 ~ "1-2 years",
      duration_months < 60 ~ "2-5 years",
      TRUE                 ~ ">= 5 years"
    ),
    duration_bin = factor(duration_bin, levels = names(span_colors))
  ) |>
  group_by(process_group) |>
  arrange(duration_months, .by_group = TRUE) |>
  mutate(
    row_id = row_number(),
    keep = if (n() <= max_spans_per_process) {
      rep(TRUE, n())
    } else {
      row_id %in% round(seq(1, n(), length.out = max_spans_per_process))
    }
  ) |>
  filter(keep) |>
  mutate(project_order = row_number()) |>
  ungroup()

fig_timeline_spans <- ggplot(spans_df) +
  geom_segment(
    aes(
      x = initiation_date, xend = decision_date,
      y = project_order,   yend = project_order,
      color = duration_bin
    ),
    alpha = 0.8, linewidth = 0.45
  ) +
  facet_wrap(~process_group, scales = "free_y", ncol = 1, drop = FALSE) +
  scale_color_manual(values = span_colors, drop = FALSE) +
  labs(
    title    = "Review Timelines by Process Type",
    subtitle = paste0(
      "Complete (clear) timelines only; sorted by duration (up to ",
      comma(max_spans_per_process), " per process)"
    ),
    x = "Date",
    y = "Reviews (sorted by duration)",
    color = "Duration"
  ) +
  theme(
    legend.position    = "top",
    axis.text.y        = element_blank(),
    axis.ticks.y       = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.spacing      = grid::unit(1.1, "lines")
  )

ggsave(file.path(FIGS, "fig_d4_project_timeline_spans.png"),
       fig_timeline_spans, width = 12, height = 10, dpi = 300)
message("Wrote fig_d4_project_timeline_spans.png")

# ---------------------------------------------------------------------------
# Fig 8: Projects by decision year (bar chart, faceted by process)
# Phase 1 ref: 03_projects_by_year.png
# ---------------------------------------------------------------------------

year_counts <- dates |>
  filter(!is.na(process_group), !is.na(decision_year)) |>
  filter(decision_year >= 2000, decision_year <= 2025) |>
  count(process_group, decision_year, name = "n_projects")

# Legislative event markers; labels only in CE (top) panel
# FRA (2023) is omitted — too close to IRA (2022) to label cleanly; see fig_d4_fra_comparison
year_events <- tibble(
  xintercept    = c(2009,           2021,          2022),
  label         = c("ARRA\nFeb 09", "BIL\nNov 21", "IRA\nAug 22"),
  hjust_val     = c(-0.08,           1.08,          -0.08),
  process_group = factor("CE", levels = PROCESS_LEVELS)
)

fig_by_year <- ggplot(year_counts, aes(x = decision_year, y = n_projects)) +
  geom_vline(xintercept = year_events$xintercept,
             linetype = "dashed", color = catf_teal, linewidth = 0.75, alpha = 0.9) +
  geom_col(aes(fill = process_group), alpha = 0.85) +
  geom_text(aes(label = comma(n_projects)), vjust = -0.3, size = 2.6, color = "gray30") +
  geom_text(
    data = year_events,
    aes(x = xintercept, y = Inf, label = label, hjust = hjust_val),
    vjust = 1.3, size = 2.3, color = catf_teal, lineheight = 0.85,
    inherit.aes = FALSE
  ) +
  facet_wrap(~process_group, scales = "free_y", ncol = 1, drop = FALSE) +
  scale_x_continuous(breaks = seq(2000, 2025, by = 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22)), labels = comma) +
  scale_fill_manual(values = PROCESS_COLORS, guide = "none") +
  labs(
    title    = "Decarbonization Reviews by Decision Year",
    subtitle = "Faceted by NEPA review process. Dashed lines mark major legislation.",
    x = "Decision Year",
    y = "Number of Reviews",
    caption = "Year derived from decision date."
  )

ggsave(file.path(FIGS, "fig_d4_projects_by_decision_year.png"),
       fig_by_year, width = 11, height = 9, dpi = 300)
message("Wrote fig_d4_projects_by_decision_year.png")

# ===========================================================================
# ENERGY-TYPE BREAKOUT FIGURES
# ===========================================================================

# ---------------------------------------------------------------------------
# Fig E1: Duration histogram — 3×3 grid (process × energy type)
# ---------------------------------------------------------------------------

dur_energy <- headline |>
  filter(duration_days > 0, duration_days < 365 * 15) |>
  mutate(duration_years = duration_days / 365.25)

p_hist_energy <- ggplot(dur_energy, aes(x = duration_years, fill = energy_type)) +
  geom_histogram(bins = 35, color = "white", linewidth = 0.15) +
  facet_grid(process_group ~ energy_type, scales = "free_y") +
  scale_fill_manual(values = ENERGY_COLORS, guide = "none") +
  scale_x_continuous(breaks = c(0, 5, 10, 15), labels = function(x) paste0(x, "y")) +
  labs(
    title    = "Review Duration Distribution by Process and Energy Type",
    subtitle = "complete_clear timelines only; rows = NEPA process, columns = energy type",
    x = "Duration (years)", y = "Reviews"
  )

ggsave(file.path(FIGS, "fig_d4_duration_histogram_by_energy.png"),
       p_hist_energy, width = 12, height = 8, dpi = 150)
message("Wrote fig_d4_duration_histogram_by_energy.png")

# ---------------------------------------------------------------------------
# Fig E2: Duration summary intervals — faceted by process, rows = energy type
# ---------------------------------------------------------------------------

interval_energy <- headline |>
  mutate(duration_months = duration_days / 30.44) |>
  filter(!is.na(duration_months), duration_months >= 0) |>
  group_by(process_group, energy_type) |>
  summarise(
    n             = n(),
    p10           = quantile(duration_months, 0.10, na.rm = TRUE),
    p25           = quantile(duration_months, 0.25, na.rm = TRUE),
    median_months = median(duration_months, na.rm = TRUE),
    p75           = quantile(duration_months, 0.75, na.rm = TRUE),
    p90           = quantile(duration_months, 0.90, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    median_label = case_when(
      median_months < 1  ~ sprintf("< 1 mo  (n=%s)", comma(n)),
      median_months < 12 ~ sprintf("%.0f mo  (n=%s)", median_months, comma(n)),
      TRUE               ~ sprintf("%.0f mo / %.1f yr  (n=%s)", median_months,
                                   median_months / 12, comma(n))
    )
  )

fig_intervals_energy <- ggplot(interval_energy, aes(y = energy_type, color = energy_type)) +
  geom_segment(aes(x = p10, xend = p90, yend = energy_type), linewidth = 2, alpha = 0.35) +
  geom_segment(aes(x = p25, xend = p75, yend = energy_type), linewidth = 6, alpha = 0.55) +
  geom_point(aes(x = median_months), size = 3.5) +
  geom_text(
    aes(x = median_months, label = median_label),
    nudge_y = 0.3, hjust = 0.5, size = 2.7, fontface = "bold", color = "gray20"
  ) +
  facet_wrap(~process_group, ncol = 1, scales = "free_x") +
  scale_color_manual(values = ENERGY_COLORS, drop = FALSE) +
  scale_x_continuous(
    labels = label_number(accuracy = 1),
    expand = expansion(mult = c(0.05, 0.05))
  ) +
  labs(
    title    = "Timeline Duration by Process and Energy Type",
    subtitle = "Thin bar = p10–p90  |  Thick bar = IQR  |  Point = median (complete_clear only)",
    x = "Duration (months)",
    y = NULL,
    color = "Energy type"
  ) +
  theme(legend.position = "top")

ggsave(file.path(FIGS, "fig_d4_duration_summary_intervals_by_energy.png"),
       fig_intervals_energy, width = 11, height = 9, dpi = 300)
message("Wrote fig_d4_duration_summary_intervals_by_energy.png")

# ---------------------------------------------------------------------------
# Fig E3: FRA comparison — 3×3 grid (rows = process, cols = energy type)
# ---------------------------------------------------------------------------

fra_energy <- headline |>
  filter(!is.na(decision_date)) |>
  mutate(period = if_else(decision_date >= FRA_CUT_DATE, "post_FRA", "pre_FRA")) |>
  group_by(process_type, energy_type, period) |>
  summarise(
    n           = n(),
    median_days = median(duration_days, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    median_months = round(median_days / 30.44, 1),
    period = factor(period, levels = c("pre_FRA", "post_FRA"),
                    labels = c("Pre-FRA", "Post-FRA"))
  )

p_fra_energy <- ggplot(fra_energy, aes(x = period, y = median_months, fill = period)) +
  geom_col(width = 0.55) +
  geom_text(
    aes(label = paste0(round(median_months, 1), " mo\n(n=", n, ")")),
    vjust = -0.25, size = 2.6
  ) +
  facet_grid(process_type ~ energy_type, scales = "free_y") +
  scale_fill_manual(
    values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue),
    guide = "none"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.28))) +
  labs(
    title    = "Median Review Duration Pre vs Post FRA by Energy Type",
    subtitle = "FRA cutoff: Aug 16, 2023  |  Rows = NEPA process, columns = energy type",
    x = NULL, y = "Median duration (months)"
  )

ggsave(file.path(FIGS, "fig_d4_fra_comparison_by_energy.png"),
       p_fra_energy, width = 11, height = 8, dpi = 150)
message("Wrote fig_d4_fra_comparison_by_energy.png")

# ---------------------------------------------------------------------------
# Fig E4: Projects by decision year — stacked by energy type, faceted by process
# ---------------------------------------------------------------------------

year_counts_energy <- dates |>
  filter(!is.na(process_group), !is.na(decision_year)) |>
  filter(decision_year >= 2000, decision_year <= 2025) |>
  count(process_group, energy_type, decision_year, name = "n_projects")

fig_by_year_energy <- ggplot(year_counts_energy,
                              aes(x = decision_year, y = n_projects, fill = energy_type)) +
  geom_vline(xintercept = year_events$xintercept,
             linetype = "dashed", color = "gray60", linewidth = 0.6, alpha = 0.8) +
  geom_col(alpha = 0.88, width = 0.85) +
  geom_text(
    data = year_events,
    aes(x = xintercept, y = Inf, label = label, hjust = hjust_val),
    vjust = 1.3, size = 2.1, color = "gray40", lineheight = 0.85,
    inherit.aes = FALSE
  ) +
  facet_wrap(~process_group, scales = "free_y", ncol = 1, drop = FALSE) +
  scale_x_continuous(breaks = seq(2000, 2025, by = 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22)), labels = comma) +
  scale_fill_manual(values = ENERGY_COLORS) +
  labs(
    title    = "Decarbonization Reviews by Decision Year and Energy Type",
    subtitle = "Stacked bars show energy type composition. Dashed lines mark major legislation.",
    x = "Decision Year",
    y = "Number of Reviews",
    fill = "Energy type",
    caption = "Year derived from decision date."
  ) +
  theme(legend.position = "top")

ggsave(file.path(FIGS, "fig_d4_projects_by_decision_year_by_energy.png"),
       fig_by_year_energy, width = 11, height = 9, dpi = 300)
message("Wrote fig_d4_projects_by_decision_year_by_energy.png")

message("\nAll figures written to: ", FIGS)
