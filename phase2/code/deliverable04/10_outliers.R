# D4: Timeline Duration Outliers
#
# Surfaces projects with implausibly long (or negative) init->decision spans so we can
# separate GENUINELY long NEPA processes (client-investigable "where it went wrong" cases)
# from pipeline extraction errors (an "initiation" that is actually a historical citation,
# a facility-establishment date, or a prior-action anchor rather than the NEPA start).
#
# Reads timeline_project_dates.parquet (+ document_index for title/agency/state/energy) and
# writes, to phase2/output/deliverable04/diagnostics/:
#   d4_duration_outliers.csv        — ALL outliers (every process), full provenance + evidence
#   d4_duration_outliers_client.csv — EA/EIS only, likely-REAL, client-facing columns
#
# An outlier is duration_days > LONG_THRESHOLD_DAYS (default 5000 ≈ 13.7 yr) or duration < 0.
# `suspect_error` is a HEURISTIC triage flag, not a verdict — the evidence_text columns are
# included so each call can be audited by eye. Final real/error labels live in the report.
#
# Usage:
#   Rscript phase2/code/deliverable04/10_outliers.R [LONG_THRESHOLD_DAYS]

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(arrow)
  library(lubridate)
  library(stringr)
})

args <- commandArgs(trailingOnly = TRUE)
LONG_THRESHOLD_DAYS <- if (length(args) >= 1) as.numeric(args[1]) else 5000

PHASE2 <- here::here("phase2")
DATA   <- file.path(PHASE2, "data", "analysis", "timeline")
OUT    <- file.path(PHASE2, "output", "deliverable04", "diagnostics")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Load: dates + per-project metadata (title/agency/state/energy from doc index)
# ---------------------------------------------------------------------------
dates <- read_parquet(file.path(DATA, "timeline_project_dates.parquet"))

meta <- read_parquet(
  file.path(DATA, "timeline_document_index.parquet"),
  col_select = c("project_id", "project_title", "lead_agency_harmonized",
                 "project_state", "project_energy_type")
) |>
  group_by(project_id) |>
  summarise(across(everything(), ~ first(na.omit(.x))), .groups = "drop")

# ---------------------------------------------------------------------------
# Compute durations on resolved (both-date) projects; flag outliers
# ---------------------------------------------------------------------------
resolved <- dates |>
  filter(timeline_status %in% c("complete_clear", "complete_with_proxy"),
         !is.na(initiation_date), !is.na(decision_date)) |>
  mutate(
    init_d = as.Date(initiation_date),
    dec_d  = as.Date(decision_date),
    duration_days = as.integer(dec_d - init_d),
    duration_years = round(duration_days / 365.25, 1),
    init_year = year(init_d)
  )

outliers <- resolved |>
  filter(duration_days > LONG_THRESHOLD_DAYS | duration_days < 0) |>
  left_join(meta, by = "project_id") |>
  mutate(
    outlier_type = if_else(duration_days < 0, "negative", "long"),
    # HEURISTIC error-triage. Most genuine NEPATEC processes are post-1990 and day/month
    # precise on at least the initiation; an early or coarse initiation that an LLM picked
    # is the classic "historical-citation-as-initiation" failure mode.
    suspect_error = case_when(
      duration_days < 0                                              ~ TRUE,
      init_year < 1985                                               ~ TRUE,
      initiation_date_granularity == "year"                         ~ TRUE,
      init_year < 1995 & initiation_source_type == "api_adjudication" ~ TRUE,
      TRUE                                                          ~ FALSE
    ),
    suspect_reason = case_when(
      duration_days < 0                                              ~ "negative duration (bad ordering)",
      init_year < 1985                                               ~ "initiation pre-1985 (likely historical citation)",
      initiation_date_granularity == "year"                         ~ "year-granularity initiation (imprecise)",
      init_year < 1995 & initiation_source_type == "api_adjudication" ~ "early LLM-picked initiation (verify)",
      TRUE                                                          ~ "plausibly real long process"
    )
  ) |>
  arrange(desc(suspect_error), process_type, desc(duration_days))

# ---------------------------------------------------------------------------
# Write: full audit table + client-facing EA/EIS likely-real subset
# ---------------------------------------------------------------------------
audit_cols <- c("project_id", "process_type", "project_title", "lead_agency_harmonized",
                "project_state", "project_energy_type", "duration_days", "duration_years",
                "outlier_type", "suspect_error", "suspect_reason",
                "initiation_date", "initiation_date_granularity", "initiation_source_type",
                "decision_date", "decision_date_granularity", "decision_source_type",
                "initiation_evidence_text", "decision_evidence_text")

write_csv(outliers |> select(any_of(audit_cols)),
          file.path(OUT, "d4_duration_outliers.csv"))

client <- outliers |>
  filter(process_type %in% c("EA", "EIS"), !suspect_error, outlier_type == "long") |>
  select(project_id, process_type, project_title, lead_agency_harmonized, project_state,
         project_energy_type, duration_years, duration_days,
         initiation_date, decision_date)
write_csv(client, file.path(OUT, "d4_duration_outliers_client.csv"))

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
cat(sprintf("\nOutliers (>%.0f days or negative), by process and triage:\n", LONG_THRESHOLD_DAYS))
print(outliers |> count(process_type, suspect_error, outlier_type) |> arrange(process_type))
cat(sprintf("\nTotal outliers: %d  |  suspect_error: %d  |  likely-real: %d\n",
            nrow(outliers), sum(outliers$suspect_error), sum(!outliers$suspect_error)))
cat(sprintf("Client EA/EIS likely-real candidates: %d\n", nrow(client)))
cat(sprintf("\nWrote:\n  %s\n  %s\n",
            file.path(OUT, "d4_duration_outliers.csv"),
            file.path(OUT, "d4_duration_outliers_client.csv")))
