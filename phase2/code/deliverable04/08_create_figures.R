# D4: Timeline Duration Analysis
#
# Reads timeline_project_dates.parquet and produces headline duration tables,
# coverage diagnostics, regulatory-period comparisons, and all main figures
# (including recreations of the Phase 1 D3 timeline charts adapted for Phase 2 schema).
#
# FRA breakpoint: 2023-06-03 (FRA enactment; matches Phase 1 D5 + the pages analysis)
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
#   d4_register_source_candidates.csv — register vs doc-text breakdown (candidate level)
#   d4_register_source_projects.csv   — register vs doc-text breakdown (project level)
#
# Output figures (phase2/output/deliverable04/figures/):
#   fig_d4_register_source_candidates.png      — stacked bar: source path of selected candidates
#   fig_d4_register_source_projects.png        — stacked bar: source path at project level
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
#   Rscript phase2/code/deliverable04/08_create_figures.R

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
FRA_CUT_DATE      <- as.Date("2023-06-03")  # FRA enactment (matches Phase 1 D5 + the pages analysis)
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
ENERGY_LEVELS         <- c("Decarb", "Fossil", "Other")
ENERGY_COLORS         <- c("Decarb" = catf_teal, "Fossil" = catf_magenta, "Other" = catf_light_blue)
ENERGY_PROCESS_COLORS <- c("Decarb" = catf_lime, "Fossil" = catf_dark_blue, "Other" = catf_navy)

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

message("Loading timeline_candidates.parquet (selected columns)...")
candidates_raw <- read_parquet(
  file.path(DATA, "timeline_candidates.parquet"),
  col_select = c("project_id", "process_type", "retrieval_tier",
                 "candidate_source_type", "selected_for_initiation", "selected_for_decision")
)
message("  ", nrow(candidates_raw), " candidate rows")

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
    # Compute duration directly from the (possibly LLM-recovered) dates rather than the
    # precomputed duration_days column, which is stale post-06 (it was only populated for the
    # pre-adjudication day-level subset, so it undercounts complete timelines — e.g. EIS 213
    # vs 425 complete_clear). decision_date >= initiation_date is guaranteed here because the
    # negative-duration rows are reclassified to invalid_order below.
    duration_days   = as.integer(decision_date - initiation_date),

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

# --- ORDER INVARIANT ASSERTION (source fix landed 2026-07-13) ---
# Negative-duration "complete" rows (decision before initiation) are now reclassified to
# invalid_order AT SOURCE: 05_select_dates.normalize_invalid_order (post-imputation) and
# 05c_inject_ground_truth._normalize_invalid_order (post-injection). The old runtime stopgap that
# silently patched them here was removed. Assert the invariant instead, so any regression fails
# loudly rather than being silently mis-counted.
.neg_mask <- dates$timeline_status %in% c("complete_clear", "complete_with_proxy") &
             !is.na(dates$initiation_date) & !is.na(dates$decision_date) &
             dates$decision_date < dates$initiation_date
if (any(.neg_mask)) {
  stop(sprintf(
    "Order invariant violated: %d complete rows have decision_date < initiation_date. This must be fixed at SOURCE (05/05c normalizer), not here.",
    as.integer(sum(.neg_mask))))
}

# Headline duration frame: ALL complete timelines (complete_clear + complete_with_proxy), so
# proxy / Final-EIS-publication decisions are included (this lifts EIS from ~425 to ~1,330).
# Month-granularity dates are imputed to the mid-month 15th (idempotent — dates already stored at
# the 15th are unchanged); YEAR-granularity dates are EXCLUDED from durations because a day cannot
# be responsibly imputed from a year alone (these are almost entirely a subset of CE initiations,
# ~1,100 of them — including them would add +/-6 months of noise to CE's ~3-week median).
headline <- dates |>
  filter(timeline_status %in% c("complete_clear", "complete_with_proxy"),
         !is.na(initiation_date), !is.na(decision_date),
         initiation_date_granularity != "year", decision_date_granularity != "year") |>
  mutate(
    .init_mid = if_else(initiation_date_granularity == "month",
                        lubridate::floor_date(initiation_date, "month") + 14, initiation_date),
    .dec_mid  = if_else(decision_date_granularity == "month",
                        lubridate::floor_date(decision_date, "month") + 14, decision_date),
    duration_days = as.integer(.dec_mid - .init_mid)
  ) |>
  filter(!is.na(duration_days), duration_days >= 0)

message("complete (clear+proxy) rows with duration: ", nrow(headline))

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
    # decision_is_feis_fallback is the live flag; final_eis_is_proxy is never set by 05/05c
    n_feis_proxy         = sum(coalesce(decision_is_feis_fallback, FALSE), na.rm = TRUE),
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
# 6b. Register date source — candidate-level and project-level
# ---------------------------------------------------------------------------

SOURCE_LEVELS <- c(
  "Register", "NOI register",
  "Doc text – full page", "Doc text – section",
  "Doc text – keyword", "Doc text – EA full read",
  "Ground truth verified", "Other"
)

SOURCE_COLORS <- c(
  "Register"                       = catf_navy,
  "NOI register"                   = catf_dark_blue,
  "Doc text – full page"      = catf_teal,
  "Doc text – section"        = catf_lime,
  "Doc text – keyword"        = catf_light_blue,
  "Doc text – EA full read"   = catf_magenta,
  "Ground truth verified"          = catf_purple,
  "Other"                          = "gray60",
  "No date"                        = "#DDDDDD",
  # 3-category project-level provenance (collapsed)
  "Register API"                   = catf_navy,
  "Doc Text"                       = catf_dark_blue,
  "No Date"                        = "#DDDDDD"
)

candidates <- candidates_raw |>
  mutate(
    source_path = case_when(
      retrieval_tier == "tier_a" & candidate_source_type == "noi_notice" ~ "NOI register",
      retrieval_tier == "tier_a"                 ~ "Register",
      retrieval_tier == "tier_b"                 ~ "Doc text – full page",
      retrieval_tier == "tier_c"                 ~ "Doc text – section",
      retrieval_tier == "tier_d"                 ~ "Doc text – keyword",
      retrieval_tier == "ea_decision_full_read"  ~ "Doc text – EA full read",
      TRUE                                       ~ "Other"
    ),
    source_path = factor(source_path, levels = SOURCE_LEVELS),
    is_register = retrieval_tier == "tier_a"
  )

# --- Candidate-level: total pulled and selected, by process × source × endpoint ---
reg_cand <- bind_rows(
  candidates |>
    group_by(process_type, source_path) |>
    summarise(
      n_candidates = n(),
      n_selected   = sum(coalesce(selected_for_initiation, FALSE)),
      .groups = "drop"
    ) |>
    mutate(endpoint = "Initiation"),
  candidates |>
    group_by(process_type, source_path) |>
    summarise(
      n_candidates = n(),
      n_selected   = sum(coalesce(selected_for_decision, FALSE)),
      .groups = "drop"
    ) |>
    mutate(endpoint = "Decision")
) |>
  mutate(process_type = factor(process_type, levels = PROCESS_LEVELS))

write_csv(reg_cand, file.path(DIAG, "d4_register_source_candidates.csv"))
message("Wrote d4_register_source_candidates.csv")

# --- Project-level: source of each project's winning selected candidate ---
# When a project has multiple selected candidates (rare), prefer register over doc text
source_rank <- c(
  "Register", "NOI register",
  "Doc text – full page", "Doc text – section",
  "Doc text – keyword", "Doc text – EA full read",
  "Other"
)

sel_init <- candidates |>
  filter(coalesce(selected_for_initiation, FALSE)) |>
  mutate(src_rank = match(as.character(source_path), source_rank)) |>
  group_by(project_id, process_type) |>
  slice_min(src_rank, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(project_id, process_type, init_source = source_path, init_is_register = is_register)

sel_dec <- candidates |>
  filter(coalesce(selected_for_decision, FALSE)) |>
  mutate(src_rank = match(as.character(source_path), source_rank)) |>
  group_by(project_id, process_type) |>
  slice_min(src_rank, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(project_id, process_type, dec_source = source_path, dec_is_register = is_register)

# Project-level provenance, collapsed to 3 client-facing categories:
#   Register API = BLM/DOE NEPA-register dates pulled via the agency metadata API (source_type
#                  "metadata"); Doc Text = any date parsed from the documents (document_text,
#                  LLM-adjudicated picks, Final-EIS publication, human-verified); No Date = missing.
src3 <- function(present, source_type) dplyr::case_when(
  !present                  ~ "No Date",
  source_type == "metadata" ~ "Register API",
  TRUE                      ~ "Doc Text"
)
proj_source <- dates |>
  select(project_id, process_type,
         initiation_source_type, decision_source_type,
         has_init = initiation_date, has_dec = decision_date) |>
  mutate(
    has_init = !is.na(has_init), has_dec = !is.na(has_dec),
    init_source_label = src3(has_init, initiation_source_type),
    dec_source_label  = src3(has_dec,  decision_source_type)
  )

SOURCE_PROJ_LEVELS <- c("Register API", "Doc Text", "No Date")

reg_proj <- bind_rows(
  proj_source |>
    count(process_type, source = init_source_label) |>
    mutate(endpoint = "Initiation"),
  proj_source |>
    count(process_type, source = dec_source_label) |>
    mutate(endpoint = "Decision")
) |>
  group_by(process_type, endpoint) |>
  mutate(total = sum(n), pct = round(100 * n / total, 1)) |>
  ungroup() |>
  mutate(
    process_type = factor(process_type, levels = PROCESS_LEVELS),
    source       = factor(source, levels = SOURCE_PROJ_LEVELS),
    endpoint     = factor(endpoint, levels = c("Initiation", "Decision"))
  )

write_csv(reg_proj, file.path(DIAG, "d4_register_source_projects.csv"))
message("Wrote d4_register_source_projects.csv")

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
  # Explicit label y-position (segment centre) so text always aligns with the
  # reverse-stacked bars: "Both dates" (first level) at the bottom, "No date" at the top.
  arrange(process_type, coverage_group) |>
  mutate(y_center = cumsum(pct) - pct / 2) |>
  ungroup() |>
  mutate(
    lab       = sprintf("%.0f%%", 100 * pct),
    # "No date" sits on a light-grey segment -> label it blue so it reads on white
    lab_color = if_else(coverage_group == "No date", catf_dark_blue, "white")
  )

p_coverage <- ggplot(coverage_fig, aes(x = process_type, y = pct, fill = coverage_group)) +
  # reverse = TRUE puts the first factor level ("Both dates") at the BOTTOM of the stack
  geom_col(width = 0.6, position = position_stack(reverse = TRUE)) +
  geom_text(
    aes(y = y_center, label = ifelse(pct >= 0.03, lab, ""), color = lab_color),
    size = 3, fontface = "bold"
  ) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_fill_manual(values = c(
    "Both dates"      = catf_navy,
    "Decision only"   = catf_dark_blue,
    "Initiation only" = catf_light_blue,
    "No date"         = "#CCCCCC"
  )) +
  scale_color_identity() +
  labs(
    title    = "Timeline Coverage by Review Type",
    subtitle = "Share of projects by timeline completeness category",
    x = NULL, y = "Share of projects", fill = NULL
  )

ggsave(file.path(FIGS, "fig_d4_coverage_by_process.png"),
       p_coverage, width = 9, height = 6, dpi = 300)
saveRDS(p_coverage, file.path(FIGS, "fig_d4_coverage_by_process.rds"))
message("Wrote fig_d4_coverage_by_process.png")

# ---------------------------------------------------------------------------
# Fig 1b: Coverage by process AND energy type — so Decarb (clean) coverage is
# directly visible and comparable to Phase 1 (which is clean-only by construction).
# ---------------------------------------------------------------------------

coverage_energy_fig <- dates |>
  filter(!is.na(process_group)) |>
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
  count(process_group, energy_type, coverage_group) |>
  group_by(process_group, energy_type) |>
  mutate(pct = n / sum(n), n_proc_energy = sum(n)) |>
  ungroup()

# "Both dates" share label per process x energy (the headline number to compare to Phase 1)
both_lab <- coverage_energy_fig |>
  filter(coverage_group == "Both dates") |>
  mutate(lab = sprintf("%.0f%%", 100 * pct))

p_coverage_energy <- ggplot(coverage_energy_fig,
                            aes(x = energy_type, y = pct, fill = coverage_group)) +
  # reverse = TRUE -> "Both dates" at the bottom, "No date" at the top
  geom_col(width = 0.7, position = position_stack(reverse = TRUE)) +
  # centre the "% with both dates" label in the navy bottom segment, white text
  geom_text(data = both_lab, aes(x = energy_type, y = pct / 2, label = lab),
            inherit.aes = FALSE, size = 2.8, color = "white", fontface = "bold") +
  facet_wrap(~process_group, nrow = 1) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_fill_manual(values = c(
    "Both dates"      = catf_navy,
    "Decision only"   = catf_dark_blue,
    "Initiation only" = catf_light_blue,
    "No date"         = "#CCCCCC"
  )) +
  labs(
    title    = "Timeline Coverage by Review Type and Energy Type",
    subtitle = "% on bars = share with BOTH dates (the Phase-1-comparable number; Decarb = clean energy)",
    x = NULL, y = "Share of projects", fill = NULL
  )

ggsave(file.path(FIGS, "fig_d4_coverage_by_process_and_energy.png"),
       p_coverage_energy, width = 11, height = 5, dpi = 300)
saveRDS(p_coverage_energy, file.path(FIGS, "fig_d4_coverage_by_process_and_energy.rds"))
message("Wrote fig_d4_coverage_by_process_and_energy.png")

# Also write the underlying table (so the Decarb numbers are exact for the deliverable)
coverage_energy_tbl <- coverage_energy_fig |>
  select(process_group, energy_type, coverage_group, n, pct) |>
  arrange(process_group, energy_type, coverage_group)
write_csv(coverage_energy_tbl, file.path(DIAG, "d4_coverage_by_process_and_energy.csv"))
message("Wrote d4_coverage_by_process_and_energy.csv")

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
    title = "Review Duration Distribution",
    x = "Duration (years)", y = "Reviews"
  )

ggsave(file.path(FIGS, "fig_d4_duration_histogram.png"),
       p_hist, width = 7, height = 8, dpi = 300)
saveRDS(p_hist, file.path(FIGS, "fig_d4_duration_histogram.rds"))
message("Wrote fig_d4_duration_histogram.png")

# ---------------------------------------------------------------------------
# Fig 3: Median review duration pre vs post FRA (June 3, 2023)
# ---------------------------------------------------------------------------

fra_fig <- fra_comparison |>
  mutate(period = factor(period, levels = c("pre_FRA", "post_FRA"),
                         labels = c("Pre-FRA\n(before Aug 2023)", "Post-FRA\n(Aug 2023+)")))

p_fra <- ggplot(fra_fig, aes(x = period, y = median_months, fill = process_type)) +
  geom_col(aes(alpha = period), width = 0.5) +
  geom_text(aes(label = paste0(round(median_months, 1), " mo\n(n=", n, ")")),
            vjust = -0.3, size = 3.2) +
  facet_wrap(~process_type, ncol = 3) +
  scale_fill_manual(values = PROCESS_COLORS, guide = "none") +
  scale_alpha_manual(
    values = c("Pre-FRA\n(before Aug 2023)" = 0.35, "Post-FRA\n(Aug 2023+)" = 1.0),
    guide = "none"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(
    title    = "D4 Median Review Duration: Pre vs Post FRA (June 3, 2023)",
    subtitle = "Lighter bar = Pre-FRA (before Aug 2023)  |  Solid bar = Post-FRA (Aug 2023+)",
    x = NULL, y = "Median duration (months)"
  )

ggsave(file.path(FIGS, "fig_d4_fra_comparison.png"),
       p_fra, width = 9, height = 5, dpi = 150)
saveRDS(p_fra, file.path(FIGS, "fig_d4_fra_comparison.rds"))
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
saveRDS(p_trend, file.path(FIGS, "fig_d4_duration_trend.rds"))
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

# Persist the exact per-process complete-timeline shares this figure prints (complete = BOTH dates
# present) so the report narrative can cite the SAME numbers (see d4_complete_share.csv). This keeps
# the "% complete" text aligned with fig-complete-share and fig-coverage by construction.
write_csv(
  process_summary_complete |>
    transmute(process_type = as.character(process_group), n_complete, n_projects, share_complete),
  file.path(DIAG, "d4_complete_share.csv")
)
message("Wrote d4_complete_share.csv")

# Each process gets an identical full 0-100% reference box; the navy dot marks that process's actual
# completion share (initiation + decision present). This keeps the uniform box-per-process look while
# letting EIS show clearly — the dot carries the value, the box is just a 0-100% frame. (The earlier
# 0/1 boxplot collapsed EIS onto the axis because its share, 24.5%, fell below the 25% quartile.)
fig_complete_share <- ggplot(process_summary_complete,
                             aes(x = process_group, fill = process_group)) +
  geom_crossbar(aes(y = 0.5, ymin = 0, ymax = 1),
                width = 0.55, alpha = 0.30, fatten = 0, na.rm = TRUE) +
  geom_point(aes(y = share_complete), size = 4, color = catf_navy, na.rm = TRUE) +
  geom_text(
    aes(y = 1.07, label = label),
    size = 3, color = "gray30"
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
    subtitle = "Dot = share with a complete timeline (initiation + decision); box = 0-100% frame",
    x = "Review Process",
    y = "Completion Share"
  ) +
  theme(legend.position = "none")

ggsave(file.path(FIGS, "fig_d4_complete_timeline_share_boxplot.png"),
       fig_complete_share, width = 9, height = 6, dpi = 300)
saveRDS(fig_complete_share, file.path(FIGS, "fig_d4_complete_timeline_share_boxplot.rds"))
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
      median_months < 1  ~ sprintf("%s: ~1 month", process_group),
      median_months < 12 ~ sprintf("%s: ~%.0f months", process_group, median_months),
      TRUE               ~ sprintf("%s: ~%.0f months (%.0f years)", process_group,
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
    breaks = scales::breaks_width(20),
    labels = label_number(accuracy = 1),
    expand = expansion(mult = c(0.02, 0.12))
  ) +
  labs(
    title    = "Timeline Duration by Review Process",
    x = "Duration (months)",
    y = "Review Process",
    color = NULL
  ) +
  theme(legend.position = "none")

ggsave(file.path(FIGS, "fig_d4_duration_summary_intervals.png"),
       fig_duration_intervals, width = 10, height = 6, dpi = 300)
saveRDS(fig_duration_intervals, file.path(FIGS, "fig_d4_duration_summary_intervals.rds"))
message("Wrote fig_d4_duration_summary_intervals.png")

# ---------------------------------------------------------------------------
# Fig 7: Project initiation → decision timeline spans (faceted by process)
# Phase 1 ref: 03_project_timeline_spans_by_process.png
# ---------------------------------------------------------------------------

max_spans_per_process <- 300

spans_df <- dates |>
  filter(
    !is.na(process_group),
    timeline_status == "complete_clear",
    !is.na(initiation_date),
    !is.na(decision_date),
    decision_date >= initiation_date
  ) |>
  mutate(
    duration_months = as.numeric(decision_date - initiation_date) / 30.44
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
      color = process_group
    ),
    alpha = 0.8, linewidth = 0.45
  ) +
  facet_wrap(~process_group, scales = "free_y", ncol = 1, drop = FALSE) +
  scale_color_manual(values = PROCESS_COLORS, guide = "none") +
  labs(
    title    = "Review Timelines by Process Type",
    subtitle = paste0(
      "Complete (clear) timelines only; sorted by duration (up to ",
      comma(max_spans_per_process), " per process)"
    ),
    x = "Date",
    y = "Reviews (sorted by duration)"
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
saveRDS(fig_timeline_spans, file.path(FIGS, "fig_d4_project_timeline_spans.rds"))
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
    title    = "Reviews by Decision Year",
    subtitle = "All projects, faceted by NEPA review process. Dashed lines mark major legislation.",
    x = "Decision Year",
    y = "Number of Reviews",
    caption = "Year derived from decision date."
  )

ggsave(file.path(FIGS, "fig_d4_projects_by_decision_year.png"),
       fig_by_year, width = 11, height = 9, dpi = 300)
saveRDS(fig_by_year, file.path(FIGS, "fig_d4_projects_by_decision_year.rds"))
message("Wrote fig_d4_projects_by_decision_year.png")

# ---------------------------------------------------------------------------
# Fig 8b: DOE projects by decision year (DOE is the largest single agency in
# NEPATEC; grant/loan programs under ARRA/BIL/IRA drive much of the volume).
# Phase 1 ref: 03_projects_by_year_doe.png
# ---------------------------------------------------------------------------

doe_agency <- read_parquet(file.path(DATA, "timeline_document_index.parquet"),
                           col_select = c("project_id", "lead_agency_harmonized")) |>
  group_by(project_id) |>
  summarise(agency = first(na.omit(lead_agency_harmonized)), .groups = "drop") |>
  filter(str_detect(coalesce(agency, ""), regex("Energy", ignore_case = TRUE)))

year_counts_doe <- dates |>
  semi_join(doe_agency, by = "project_id") |>
  filter(!is.na(process_group), !is.na(decision_year),
         decision_year >= 2000, decision_year <= 2025) |>
  count(process_group, decision_year, name = "n_projects")

fig_by_year_doe <- ggplot(year_counts_doe, aes(x = decision_year, y = n_projects)) +
  geom_vline(xintercept = year_events$xintercept,
             linetype = "dashed", color = catf_teal, linewidth = 0.75, alpha = 0.9) +
  geom_col(aes(fill = process_group), alpha = 0.85) +
  geom_text(aes(label = comma(n_projects)), vjust = -0.3, size = 2.6, color = "gray30") +
  geom_text(data = year_events,
            aes(x = xintercept, y = Inf, label = label, hjust = hjust_val),
            vjust = 1.3, size = 2.3, color = catf_teal, lineheight = 0.85, inherit.aes = FALSE) +
  facet_wrap(~process_group, scales = "free_y", ncol = 1, drop = FALSE) +
  scale_x_continuous(breaks = seq(2000, 2025, by = 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22)), labels = comma) +
  scale_fill_manual(values = PROCESS_COLORS, guide = "none") +
  labs(
    title    = "DOE Projects by Decision Year",
    subtitle = "Department of Energy projects only, faceted by review process. Dashed lines mark major legislation.",
    x = "Decision Year", y = "Number of Reviews",
    caption = "Year derived from decision date. Lead agency = Department of Energy."
  )

ggsave(file.path(FIGS, "fig_d4_projects_by_decision_year_doe.png"),
       fig_by_year_doe, width = 11, height = 9, dpi = 300)
saveRDS(fig_by_year_doe, file.path(FIGS, "fig_d4_projects_by_decision_year_doe.rds"))
message("Wrote fig_d4_projects_by_decision_year_doe.png")

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
  scale_fill_manual(values = ENERGY_PROCESS_COLORS, guide = "none") +
  scale_x_continuous(breaks = c(0, 5, 10, 15), labels = function(x) paste0(x, "y")) +
  labs(
    title    = "Review Duration Distribution by Process and Energy Type",
    subtitle = "complete_clear timelines only; rows = NEPA process, columns = energy type",
    x = "Duration (years)", y = "Reviews"
  )

ggsave(file.path(FIGS, "fig_d4_duration_histogram_by_energy.png"),
       p_hist_energy, width = 12, height = 8, dpi = 150)
saveRDS(p_hist_energy, file.path(FIGS, "fig_d4_duration_histogram_by_energy.rds"))
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
      median_months < 1  ~ sprintf("~1 mo  (n=%s)", comma(n)),
      median_months < 12 ~ sprintf("%.0f mo  (n=%s)", median_months, comma(n)),
      TRUE               ~ sprintf("%.0f mo / %.1f yr  (n=%s)", median_months,
                                   median_months / 12, comma(n))
    )
  )

# Second EA "Fossil" row: document-anchored fossil EAs (initiation_source_type != register). Their
# ~5-month median reflects true review length once the register administrative-start artifact is
# removed. Added ONLY to the EA facet (register-anchoring is the EA/fossil issue), as a companion to
# the register-anchored "Fossil" row so the two views sit side by side.
fossil_ea_doc <- headline |>
  filter(process_group == "EA", energy_type == "Fossil",
         initiation_source_type != "metadata") |>
  mutate(duration_months = duration_days / 30.44) |>
  filter(!is.na(duration_months), duration_months >= 0) |>
  summarise(
    process_group = factor("EA", levels = PROCESS_LEVELS),
    energy_type   = "Fossil (doc-anchored)",
    n             = n(),
    p10           = quantile(duration_months, 0.10, na.rm = TRUE),
    p25           = quantile(duration_months, 0.25, na.rm = TRUE),
    median_months = median(duration_months, na.rm = TRUE),
    p75           = quantile(duration_months, 0.75, na.rm = TRUE),
    p90           = quantile(duration_months, 0.90, na.rm = TRUE)
  ) |>
  mutate(median_label = case_when(
    median_months < 1  ~ sprintf("~1 mo  (n=%s)", comma(n)),
    median_months < 12 ~ sprintf("%.0f mo  (n=%s)", median_months, comma(n)),
    TRUE               ~ sprintf("%.0f mo / %.1f yr  (n=%s)", median_months,
                                 median_months / 12, comma(n))
  ))

ENERGY_LEVELS_EXT <- c("Decarb", "Fossil", "Fossil (doc-anchored)", "Other")
interval_energy <- bind_rows(interval_energy, fossil_ea_doc) |>
  mutate(energy_type = factor(as.character(energy_type), levels = ENERGY_LEVELS_EXT))
# Highlight the document-anchored sensitivity row with one bold standout colour (magenta) so it
# pops against the cool Decarb/Fossil/Other palette — simple, no extra mark clutter.
ENERGY_PROCESS_COLORS_EXT <- c(ENERGY_PROCESS_COLORS,
                               "Fossil (doc-anchored)" = catf_magenta)

fig_intervals_energy <- ggplot(interval_energy, aes(y = energy_type, color = energy_type)) +
  geom_segment(aes(x = p10, xend = p90, yend = energy_type), linewidth = 2, alpha = 0.35) +
  geom_segment(aes(x = p25, xend = p75, yend = energy_type), linewidth = 6, alpha = 0.55) +
  geom_point(aes(x = median_months), size = 3.5) +
  geom_text(
    aes(x = median_months, label = median_label),
    nudge_y = 0.3, hjust = 0.5, size = 2.7, fontface = "bold", color = "gray20"
  ) +
  facet_wrap(~process_group, ncol = 1, scales = "free") +
  scale_color_manual(values = ENERGY_PROCESS_COLORS_EXT, drop = FALSE) +
  scale_x_continuous(
    # Per-panel tick spacing for the free-x facets: CE spans only weeks-to-months (5-mo ticks),
    # EA runs to ~3 years (10-mo ticks), EIS to ~10 years (20-mo ticks).
    breaks = function(lims) {
      w <- if (diff(lims) <= 20) 5 else if (diff(lims) <= 60) 10 else 20
      seq(0, lims[2], by = w)
    },
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
saveRDS(fig_intervals_energy, file.path(FIGS, "fig_d4_duration_summary_intervals_by_energy.rds"))
message("Wrote fig_d4_duration_summary_intervals_by_energy.png")

# ---------------------------------------------------------------------------
# Fig E2b: Fossil-EA review length — register-anchored (current) vs document-anchored
# (conservative), with Other/Decarb EAs for reference. Exposes the register-anchoring artifact
# and the coverage-vs-plausibility trade-off (see report §fossil-EA).
# ---------------------------------------------------------------------------

.ea_head <- headline |> filter(process_type == "EA")
fossil_box_df <- dplyr::bind_rows(
  .ea_head |> filter(energy_type == "Fossil") |>
    mutate(grp = "Fossil — CURRENT\n(register)"),
  .ea_head |> filter(energy_type == "Fossil", initiation_source_type != "metadata") |>
    mutate(grp = "Fossil — CONSERVATIVE\n(doc-anchored)"),
  .ea_head |> filter(energy_type == "Other")  |> mutate(grp = "Other EA"),
  .ea_head |> filter(energy_type == "Decarb") |> mutate(grp = "Decarb EA")
)
.grp_levels <- c("Fossil — CURRENT\n(register)", "Fossil — CONSERVATIVE\n(doc-anchored)",
                 "Other EA", "Decarb EA")
fossil_box_df$grp <- factor(fossil_box_df$grp, levels = .grp_levels)
fossil_box_stats <- fossil_box_df |> group_by(grp) |>
  summarise(n = n(), med = median(duration_days), .groups = "drop")

fig_fossil_box <- ggplot(fossil_box_df, aes(x = grp, y = duration_days, fill = grp)) +
  geom_boxplot(outlier.shape = NA, width = 0.6, colour = "white", linewidth = 0.6) +
  geom_vline(xintercept = 2.5, linetype = "dashed", colour = "grey80") +
  geom_text(data = fossil_box_stats, aes(x = grp, y = med, label = paste0(round(med), "d")),
            inherit.aes = FALSE, vjust = -0.7, fontface = "bold", size = 3.3) +
  geom_text(data = fossil_box_stats, aes(x = grp, y = -28, label = paste0("n=", n)),
            inherit.aes = FALSE, size = 3, colour = "grey40") +
  scale_fill_manual(values = c(
    "Fossil — CURRENT\n(register)"          = catf_navy,
    "Fossil — CONSERVATIVE\n(doc-anchored)" = catf_lime,
    "Other EA"                              = catf_light_blue,
    "Decarb EA"                             = catf_light_blue)) +
  coord_cartesian(ylim = c(-35, 650)) +
  labs(
    title    = "Fossil-fuel EA review length: register-anchored vs document-anchored",
    subtitle = sprintf(paste0("The document-anchored view lands near Other/Decarb EAs — the ~40-day register median is an\n",
                              "administrative-entry artifact — but survives for only %d of %d complete fossil EAs."),
                       fossil_box_stats$n[fossil_box_stats$grp == "Fossil — CONSERVATIVE\n(doc-anchored)"],
                       fossil_box_stats$n[fossil_box_stats$grp == "Fossil — CURRENT\n(register)"]),
    x = NULL, y = "Init → decision duration (days)") +
  theme(legend.position = "none")

ggsave(file.path(FIGS, "fig_d4_fossil_ea_anchor_boxplot.png"),
       fig_fossil_box, width = 9, height = 5.5, dpi = 300)
saveRDS(fig_fossil_box, file.path(FIGS, "fig_d4_fossil_ea_anchor_boxplot.rds"))
message("Wrote fig_d4_fossil_ea_anchor_boxplot.png")

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

p_fra_energy <- ggplot(fra_energy, aes(x = period, y = median_months, fill = energy_type)) +
  geom_col(aes(alpha = period), width = 0.55) +
  geom_text(
    aes(label = paste0(round(median_months, 1), " mo\n(n=", n, ")")),
    vjust = -0.25, size = 2.6
  ) +
  facet_grid(process_type ~ energy_type, scales = "free_y") +
  scale_fill_manual(values = ENERGY_PROCESS_COLORS, guide = "none") +
  scale_alpha_manual(
    values = c("Pre-FRA" = 0.35, "Post-FRA" = 1.0),
    guide = "none"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.28))) +
  labs(
    title    = "Median Review Duration Pre vs Post FRA by Energy Type",
    subtitle = "Lighter bar = Pre-FRA  |  Solid bar = Post-FRA  |  Rows = NEPA process, columns = energy type",
    x = NULL, y = "Median duration (months)"
  )

ggsave(file.path(FIGS, "fig_d4_fra_comparison_by_energy.png"),
       p_fra_energy, width = 11, height = 8, dpi = 150)
saveRDS(p_fra_energy, file.path(FIGS, "fig_d4_fra_comparison_by_energy.rds"))
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
  scale_fill_manual(values = ENERGY_PROCESS_COLORS) +
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
saveRDS(fig_by_year_energy, file.path(FIGS, "fig_d4_projects_by_decision_year_by_energy.rds"))
message("Wrote fig_d4_projects_by_decision_year_by_energy.png")

message("\nAll figures written to: ", FIGS)

# ---------------------------------------------------------------------------
# Fig R1: Register vs doc-text — candidate level (selected candidates only)
# ---------------------------------------------------------------------------
# Shows, of the candidates that were ultimately chosen as the initiation or
# decision date, what retrieval path produced them.

cand_sel_plot <- reg_cand |>
  filter(n_selected > 0) |>
  group_by(process_type, endpoint) |>
  mutate(pct_selected = n_selected / sum(n_selected)) |>
  ungroup() |>
  mutate(endpoint = factor(endpoint, levels = c("Initiation", "Decision")))

p_cand_source <- ggplot(
  cand_sel_plot,
  aes(x = process_type, y = pct_selected, fill = source_path)
) +
  geom_col(width = 0.65) +
  geom_text(
    aes(label = ifelse(pct_selected >= 0.04,
                       paste0(round(pct_selected * 100), "%"), "")),
    position = position_stack(vjust = 0.5),
    size = 2.9, color = "white", fontface = "bold"
  ) +
  facet_wrap(~endpoint, ncol = 2) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_fill_manual(values = SOURCE_COLORS, drop = FALSE,
                    guide = guide_legend(reverse = TRUE)) +
  labs(
    title    = "D4 Timeline: Source of Selected Date Candidates by Review Type",
    subtitle = "Stacked bars show share of selected candidates from each retrieval path",
    x        = NULL,
    y        = "Share of selected candidates",
    fill     = "Source path"
  ) +
  theme(legend.position = "right")

ggsave(file.path(FIGS, "fig_d4_register_source_candidates.png"),
       p_cand_source, width = 10, height = 6, dpi = 300)
saveRDS(p_cand_source, file.path(FIGS, "fig_d4_register_source_candidates.rds"))
message("Wrote fig_d4_register_source_candidates.png")

# ---------------------------------------------------------------------------
# Fig R2: Register vs doc-text — project level (all projects, incl. no date)
# ---------------------------------------------------------------------------
# Shows, for every project in the corpus, where its selected initiation and
# decision date came from. "No date" = project never got a date at this endpoint.

p_proj_source <- ggplot(
  reg_proj |>
    mutate(
      # Stack: Register API bottom, Doc Text middle, No Date top
      source = factor(source, levels = rev(SOURCE_PROJ_LEVELS)),
      # No Date sits on a light-grey segment -> label it navy so it reads on white
      lab_color = if_else(as.character(source) == "No Date", catf_navy, "white")
    ),
  aes(x = endpoint, y = pct / 100, fill = source)
) +
  geom_col(width = 0.65) +
  geom_text(
    aes(label = ifelse(pct >= 3, paste0(round(pct), "%"), ""), color = lab_color),
    position = position_stack(vjust = 0.5),
    size = 2.8, fontface = "bold"
  ) +
  facet_wrap(~process_type, ncol = 3) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  # legend top->bottom = No Date / Doc Text / Register API, matching the stack
  scale_fill_manual(values = SOURCE_COLORS, drop = FALSE,
                    guide = guide_legend(reverse = FALSE)) +
  scale_color_identity() +
  labs(
    title    = "Date Source by Review Type",
    x        = NULL,
    y        = "Share of all projects",
    fill     = "Date source"
  ) +
  theme(legend.position = "right")

ggsave(file.path(FIGS, "fig_d4_register_source_projects.png"),
       p_proj_source, width = 12, height = 5, dpi = 300)
saveRDS(p_proj_source, file.path(FIGS, "fig_d4_register_source_projects.rds"))
message("Wrote fig_d4_register_source_projects.png")
