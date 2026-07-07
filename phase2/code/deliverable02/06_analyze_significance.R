#!/usr/bin/env Rscript
# D2 Phase 6 — significance analysis (plan v2.11 §8).
#
# Reads the determination dataset (+ threshold child + corpus) and produces the headline
# tables. HEADLINE-DENOMINATOR GATE: every primary table filters to
# agency_scope_status == 'primary_blm_doe_family' (plus in-scope time). context_other_agency /
# manual_scope_review rows are reported separately, never folded into A1 primary rates.
# Dual denominators (projects AND determinations); cells below MIN_CELL suppressed.
#
# NOTE: on a --dry-run determinations table every row is extraction_method='regex' &
# needs_human_review=TRUE, so these tables are ILLUSTRATIVE until the billable LLM pass + gold.
#
# Run:  Rscript phase2/code/deliverable02/06_analyze_significance.R
suppressMessages({library(arrow); library(dplyr); library(tidyr); library(readr); library(stringr)})

A <- "phase2/data/analysis/deliverable02"
OUT <- "phase2/output/deliverable02/analysis"; dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
MIN_CELL <- 5
NON_DET <- c("not_a_determination", "ambiguous")

# FONSI track is the default; pass --with-eis to also fold in the EIS track
# (04 writes *_eis.parquet so the two tracks never clobber each other).
WITH_EIS <- "--with-eis" %in% commandArgs(trailingOnly = TRUE)

det <- read_parquet(file.path(A, "significance_determinations.parquet"))
thr <- read_parquet(file.path(A, "determination_thresholds.parquet"))
eis_det_path <- file.path(A, "significance_determinations_eis.parquet")
if (WITH_EIS && file.exists(eis_det_path)) {
  det <- bind_rows(det, read_parquet(eis_det_path))
  eis_thr_path <- file.path(A, "determination_thresholds_eis.parquet")
  if (file.exists(eis_thr_path)) thr <- bind_rows(thr, read_parquet(eis_thr_path))
  cat("combined FONSI + EIS tracks. determinations by process_type:\n")
  print(table(det$process_type))
} else if (WITH_EIS) {
  cat("--with-eis passed but no", eis_det_path, "found — FONSI only.\n")
} else {
  cat("FONSI track only (pass --with-eis to combine the EIS track).\n")
}

# ---- headline gate ----
primary <- det %>%
  filter(agency_scope_status == "primary_blm_doe_family",
         analysis_scope == "primary",
         !determination_class %in% NON_DET)
cat(sprintf("determinations total=%d  primary-scope determinations=%d  (projects=%d)\n",
            nrow(det), nrow(primary), n_distinct(primary$project_id)))

suppress <- function(df, col = "n") { df[[paste0(col, "_suppressed")]] <- df[[col]] < MIN_CELL; df }
w <- function(df, name) { write_csv(df, file.path(OUT, name)); cat("  wrote", name, "\n") }

# 1. headline cross-resource significance map (resource x class)
w(primary %>% count(shared_resource_area, determination_class) %>%
    suppress() %>% arrange(desc(n)), "resource_by_class.csv")

# 2. class distribution + dual denominators
w(bind_rows(
    primary %>% count(determination_class, name = "n_determinations"),
    primary %>% distinct(project_id, determination_class) %>%
      count(determination_class, name = "n_projects") %>% rename(n_determinations = n_projects) %>%
      mutate(determination_class = paste0(determination_class, " [project-level]"))
  ), "class_distribution_dual_denominator.csv")

# 3. cross-agency (BLM vs DOE-family subagencies, within primary scope)
w(primary %>% count(agency, determination_class) %>% suppress(), "agency_by_class.csv")

# 4. cross-cohort (FRA label = 2023-06-03)
w(primary %>% count(cohort_by_date, determination_class) %>% suppress(), "cohort_by_class.csv")

# 5. threshold profile from the CHILD table (not the scalar summary)
thr_primary <- thr %>% semi_join(primary, by = "determination_instance_id") %>%
  left_join(primary %>% select(determination_instance_id, determination_class), by = "determination_instance_id")
w(thr_primary %>% count(threshold_type, determination_class) %>% suppress(), "threshold_by_class.csv")

# 6. mitigation read (would-be-significant -> committed mitigation)
w(primary %>% count(determination_class, mitigation_flag) %>% suppress(), "mitigation_by_class.csv")

# 7. context universe reported SEPARATELY (never in primary rates)
w(det %>% filter(!determination_class %in% NON_DET) %>%
    count(agency_scope_status, determination_class) %>% suppress(),
  "context_universe_by_scope.csv")

# 8. association layer (interpretable): significant(0/1) ~ threshold flags + agency + cohort.
#    Guarded — needs enough adjudicated (non-regex) rows; skipped on a dry-run table.
adjudicated <- primary %>% filter(extraction_method == "regex+llm")
if (nrow(adjudicated) >= 200 && n_distinct(adjudicated$determination_class) > 1) {
  d <- adjudicated %>%
    mutate(significant = as.integer(determination_class %in%
             c("significant_adverse", "significant_unavoidable", "eis_required")))
  thr_wide <- thr %>% semi_join(adjudicated, by = "determination_instance_id") %>%
    distinct(determination_instance_id, threshold_type) %>% mutate(v = 1) %>%
    pivot_wider(names_from = threshold_type, values_from = v, values_fill = 0,
                names_prefix = "thr_")
  d <- d %>% left_join(thr_wide, by = "determination_instance_id")
  thr_cols <- grep("^thr_", names(d), value = TRUE)
  form <- as.formula(paste("significant ~ agency + cohort_by_date +",
                           paste(thr_cols, collapse = " + ")))
  fit <- glm(form, data = d, family = binomial())
  or <- data.frame(term = names(coef(fit)), odds_ratio = exp(coef(fit)))
  w(or, "association_odds_ratios.csv")
  cat("  association layer fit on", nrow(d), "adjudicated determinations\n")
} else {
  cat("  [association layer skipped] needs >=200 adjudicated (regex+llm) determinations; ",
      "run the billable LLM pass first.\n")
}

cat("\nDone. Primary-scope tables in", OUT, "\n")
cat("Reminder: dry-run tables are illustrative; regenerate after the LLM pass + gold validation.\n")
