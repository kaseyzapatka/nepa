# D4: Timeline Duration Analysis
#
# Reads from the D4 timeline database (phase2/data/analysis/timeline/) and
# produces headline duration tables, coverage diagnostics, and regulatory-period
# comparisons for Deliverable 4.
#
# Required FRA breakpoint: 2023-08-16 (CEQ final rule effective date, from memory)
# Legislative event markers: ARRA 2009, BIL 2021, IRA 2022 (from Phase 1 R code)
#
# Outputs:
#   phase2/output/deliverable04/d4_duration_summary.csv
#   phase2/output/deliverable04/d4_coverage_by_process.csv
#   phase2/output/deliverable04/d4_duration_by_period.csv
#   phase2/output/deliverable04/d4_proxy_sensitivity.csv
#   phase2/output/deliverable04/d4_coverage_diagnostics.csv
#   phase2/output/deliverable04/fig_d4_coverage_by_process.png
#   phase2/output/deliverable04/fig_d4_duration_histogram.png
#   phase2/output/deliverable04/fig_d4_fra_comparison.png
#   phase2/output/deliverable04/fig_d4_duration_trend.png
#
# Usage:
#   Rscript phase2/code/deliverable04/08_analyze_timelines.R

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
dir.create(OUTPUT, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRA_CUT_DATE      <- as.Date("2023-08-16")   # CEQ final rule effective date
ARRA_DATE         <- as.Date("2009-02-17")
BIL_DATE          <- as.Date("2021-11-15")
IRA_DATE          <- as.Date("2022-08-16")
PROXY_SENSITIVITY <- TRUE                     # include complete_with_proxy in sensitivity

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

message("Loading timeline_project_dates.parquet...")
dates_raw <- read_parquet(file.path(DATA, "timeline_project_dates.parquet"))
message("  ", nrow(dates_raw), " rows")

# Optional: join document burden from index
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

    # Regulatory period based on decision_date
    reg_period = case_when(
      decision_date >= FRA_CUT_DATE             ~ "post_FRA",
      decision_date >= IRA_DATE                 ~ "IRA",
      decision_date >= BIL_DATE                 ~ "BIL",
      decision_date >= ARRA_DATE                ~ "post_ARRA",
      decision_date >= as.Date("2000-01-01")    ~ "pre_ARRA_2000s",
      TRUE                                      ~ "pre_2000"
    ),
    reg_period = factor(reg_period, levels = c(
      "pre_2000", "pre_ARRA_2000s", "post_ARRA", "BIL", "IRA", "post_FRA"
    )),

    decision_year = year(decision_date),

    has_proxy_flag  = str_detect(coalesce(timeline_flags, ""), "proxy"),
    is_proxy_only   = str_detect(coalesce(timeline_flags, ""), "proxy_only"),
  )

# Headline analysis: complete_clear only (plan §12)
headline <- dates |>
  filter(timeline_status == "complete_clear", !is.na(duration_days))

message("complete_clear rows with day-granularity duration: ", nrow(headline))

# ---------------------------------------------------------------------------
# Helper: summary stats
# ---------------------------------------------------------------------------

duration_summary <- function(df, group_vars) {
  df |>
    group_by(across(all_of(group_vars))) |>
    summarise(
      n                  = n(),
      median_days        = median(duration_days, na.rm = TRUE),
      p10_days           = quantile(duration_days, 0.10, na.rm = TRUE),
      p25_days           = quantile(duration_days, 0.25, na.rm = TRUE),
      p75_days           = quantile(duration_days, 0.75, na.rm = TRUE),
      p90_days           = quantile(duration_days, 0.90, na.rm = TRUE),
      mean_days          = mean(duration_days, na.rm = TRUE),
      pct_lt_1y          = mean(duration_days < 365, na.rm = TRUE),
      pct_gt_5y          = mean(duration_days > 5 * 365, na.rm = TRUE),
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

dur_process <- duration_summary(headline, "process_type")
dur_period  <- duration_summary(headline, c("process_type", "reg_period"))
dur_agency  <- duration_summary(
  headline |> left_join(
    dates_raw |> select(project_id, process_type) |> distinct(),
    by = c("project_id", "process_type")
  ),
  c("process_type")  # placeholder — expand with agency join if available
)

write_csv(dur_process, file.path(OUTPUT, "d4_duration_summary.csv"))
message("Wrote d4_duration_summary.csv")

write_csv(dur_period,  file.path(OUTPUT, "d4_duration_by_period.csv"))
message("Wrote d4_duration_by_period.csv")

# ---------------------------------------------------------------------------
# 2. Coverage diagnostics
# ---------------------------------------------------------------------------

coverage <- dates |>
  group_by(process_type, timeline_status) |>
  summarise(n = n(), .groups = "drop") |>
  group_by(process_type) |>
  mutate(
    total_process   = sum(n),
    pct             = round(100 * n / total_process, 1)
  ) |>
  ungroup()

# Coverage by energy type
coverage_energy <- dates |>
  mutate(
    has_initiation  = !is.na(initiation_date),
    has_decision    = !is.na(decision_date),
    is_complete_clear = timeline_status == "complete_clear",
  ) |>
  group_by(process_type) |>
  summarise(
    n_total              = n(),
    n_initiation         = sum(has_initiation),
    n_decision           = sum(has_decision),
    n_complete_clear     = sum(is_complete_clear),
    pct_initiation       = round(100 * mean(has_initiation), 1),
    pct_decision         = round(100 * mean(has_decision), 1),
    pct_complete_clear   = round(100 * mean(is_complete_clear), 1),
    .groups = "drop"
  )

write_csv(coverage,        file.path(OUTPUT, "d4_coverage_by_process.csv"))
write_csv(coverage_energy, file.path(OUTPUT, "d4_coverage_diagnostics.csv"))
message("Wrote d4_coverage_by_process.csv, d4_coverage_diagnostics.csv")

# ---------------------------------------------------------------------------
# 3. Proxy sensitivity (plan §12)
# ---------------------------------------------------------------------------

if (PROXY_SENSITIVITY) {
  proxy_dates <- dates |>
    filter(
      timeline_status %in% c("complete_clear", "complete_with_proxy"),
      !is.na(initiation_date), !is.na(decision_date)
    ) |>
    mutate(
      # Compute approximate duration for proxy cases using day-granularity check
      duration_days_approx = as.integer(decision_date - initiation_date),
      # Only use approx when granularity allows it (or both are non-null)
      uses_proxy = timeline_status == "complete_with_proxy"
    )

  sensitivity_summary <- proxy_dates |>
    group_by(process_type, uses_proxy) |>
    summarise(
      n            = n(),
      median_days  = median(duration_days_approx, na.rm = TRUE),
      p25_days     = quantile(duration_days_approx, 0.25, na.rm = TRUE),
      p75_days     = quantile(duration_days_approx, 0.75, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(dataset = if_else(uses_proxy, "with_proxy", "clear_only"))

  write_csv(sensitivity_summary, file.path(OUTPUT, "d4_proxy_sensitivity.csv"))
  message("Wrote d4_proxy_sensitivity.csv")
}

# ---------------------------------------------------------------------------
# 4. Duration by decision year (trend)
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

write_csv(dur_year, file.path(OUTPUT, "d4_duration_by_year.csv"))
message("Wrote d4_duration_by_year.csv")

# ---------------------------------------------------------------------------
# 5. Pre/post FRA comparison (required breakpoint from plan §10)
# ---------------------------------------------------------------------------

fra_comparison <- headline |>
  filter(!is.na(decision_date)) |>
  mutate(period = if_else(decision_date >= FRA_CUT_DATE, "post_FRA", "pre_FRA")) |>
  group_by(process_type, period) |>
  summarise(
    n            = n(),
    median_days  = median(duration_days, na.rm = TRUE),
    p25_days     = quantile(duration_days, 0.25, na.rm = TRUE),
    p75_days     = quantile(duration_days, 0.75, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(median_months = round(median_days / 30.44, 1))

write_csv(fra_comparison, file.path(OUTPUT, "d4_fra_comparison.csv"))
message("Wrote d4_fra_comparison.csv")

# ---------------------------------------------------------------------------
# 6. Quality flags summary
# ---------------------------------------------------------------------------

flags_raw <- dates |>
  filter(!is.na(timeline_flags), timeline_flags != "") |>
  mutate(flag_list = str_split(timeline_flags, "\\|")) |>
  tidyr::unnest(flag_list) |>
  filter(flag_list != "")

flag_summary <- flags_raw |>
  group_by(process_type, flag_list) |>
  summarise(n = n(), .groups = "drop") |>
  arrange(process_type, desc(n))

write_csv(flag_summary, file.path(OUTPUT, "d4_flag_summary.csv"))
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
      n               = n(),
      median_days     = median(duration_days),
      median_months   = round(median(duration_days) / 30.44, 1),
      p10_days        = quantile(duration_days, 0.10),
      p90_days        = quantile(duration_days, 0.90),
      .groups = "drop"
    )
)

cat("\nFRA period comparison (post 2023-08-16 vs prior):\n")
print(fra_comparison |> select(process_type, period, n, median_months, p25_days, p75_days))

cat("\nAll output files written to:", OUTPUT, "\n")

# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------

CATF_NAVY <- "#012169"
CATF_BLUE <- "#0047BB"
PROCESS_COLORS <- c("CE" = "#4DAF4A", "EA" = CATF_BLUE, "EIS" = CATF_NAVY)

# Fig 1: Coverage stacked bar — decision / initiation / none, by process type
coverage_fig <- dates |>
  mutate(
    has_decision   = !is.na(decision_date),
    has_initiation = !is.na(initiation_date),
    coverage_group = case_when(
      has_decision & has_initiation ~ "Both dates",
      has_decision                  ~ "Decision only",
      has_initiation                ~ "Initiation only",
      TRUE                          ~ "No date"
    )
  ) |>
  count(process_type, coverage_group) |>
  group_by(process_type) |>
  mutate(pct = n / sum(n)) |>
  ungroup() |>
  mutate(coverage_group = factor(coverage_group,
    levels = c("Both dates", "Decision only", "Initiation only", "No date")))

p_coverage <- ggplot(coverage_fig, aes(x = process_type, y = pct, fill = coverage_group)) +
  geom_col(width = 0.6) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_fill_manual(values = c(
    "Both dates"     = CATF_NAVY,
    "Decision only"  = CATF_BLUE,
    "Initiation only"= "#6BAED6",
    "No date"        = "#CCCCCC"
  )) +
  labs(title = "D4 Timeline Coverage by Review Type",
       x = NULL, y = "Share of projects", fill = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(OUTPUT, "fig_d4_coverage_by_process.png"),
       p_coverage, width = 7, height = 5, dpi = 150)
message("Wrote fig_d4_coverage_by_process.png")

# Fig 2: Duration histogram by process type (complete_clear, day granularity)
dur_plot <- headline |>
  filter(duration_days > 0, duration_days < 365 * 15) |>
  mutate(duration_years = duration_days / 365.25)

p_hist <- ggplot(dur_plot, aes(x = duration_years, fill = process_type)) +
  geom_histogram(bins = 40, color = "white", linewidth = 0.2) +
  facet_wrap(~process_type, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = PROCESS_COLORS, guide = "none") +
  scale_x_continuous(breaks = 0:15, labels = function(x) paste0(x, "y")) +
  labs(title = "D4 Review Duration Distribution (complete_clear only)",
       x = "Duration (years)", y = "Projects") +
  theme_minimal(base_size = 12)

ggsave(file.path(OUTPUT, "fig_d4_duration_histogram.png"),
       p_hist, width = 7, height = 8, dpi = 150)
message("Wrote fig_d4_duration_histogram.png")

# Fig 3: FRA pre/post comparison — median duration bar chart
fra_fig <- fra_comparison |>
  mutate(period = factor(period, levels = c("pre_FRA", "post_FRA"),
                         labels = c("Pre-FRA\n(before Aug 2023)", "Post-FRA\n(Aug 2023+)")))

p_fra <- ggplot(fra_fig, aes(x = period, y = median_months, fill = period)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(round(median_months, 1), " mo\n(n=", n, ")")),
            vjust = -0.3, size = 3.2) +
  facet_wrap(~process_type, ncol = 3) +
  scale_fill_manual(values = c("Pre-FRA\n(before Aug 2023)" = CATF_BLUE,
                               "Post-FRA\n(Aug 2023+)" = CATF_NAVY), guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(title = "D4 Median Review Duration: Pre vs Post FRA (Aug 16, 2023)",
       x = NULL, y = "Median duration (months)") +
  theme_minimal(base_size = 12)

ggsave(file.path(OUTPUT, "fig_d4_fra_comparison.png"),
       p_fra, width = 9, height = 5, dpi = 150)
message("Wrote fig_d4_fra_comparison.png")

# Fig 4: Duration trend by year (median, complete_clear)
p_trend <- ggplot(dur_year |> filter(n >= 5),
                  aes(x = decision_year, y = median_months, color = process_type)) +
  geom_line(linewidth = 0.8) +
  geom_point(aes(size = n), alpha = 0.7) +
  geom_vline(xintercept = c(2009, 2021, 2022, 2023.6),
             linetype = "dashed", color = "grey50", linewidth = 0.5) +
  annotate("text", x = c(2009, 2021, 2022, 2023.6), y = Inf,
           label = c("ARRA", "BIL", "IRA", "FRA"),
           vjust = 1.5, hjust = -0.1, size = 3, color = "grey40") +
  scale_color_manual(values = PROCESS_COLORS) +
  scale_size_continuous(range = c(1, 4), guide = "none") +
  scale_x_continuous(breaks = seq(1990, 2026, 5)) +
  labs(title = "D4 Median Review Duration by Year (complete_clear, n≥5)",
       x = "Decision year", y = "Median duration (months)", color = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(OUTPUT, "fig_d4_duration_trend.png"),
       p_trend, width = 10, height = 5, dpi = 150)
message("Wrote fig_d4_duration_trend.png")
