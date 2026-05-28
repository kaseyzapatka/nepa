# --------------------------
# PHASE 2, DELIVERABLE 4: TIMELINES — PROOF OF CONCEPT
# --------------------------
# Goal: Assess what we can deliver for Phase 2 timeline analysis.
# Questions:
#   (1) What does existing BERT timeline coverage look like?
#   (2) Can we segment by agency, year, state, project type?
#   (3) How big is the EA/EIS gap (no BERT data yet)?
#   (4) How useful is the Federal Register NOI data?
#   (5) What data quality issues exist (future dates, zero durations)?
# This is a feasibility check, not a final pipeline.

library(here)
library(arrow)
library(tidyverse)

# --------------------------
# LOAD DATA
# --------------------------

projects <- read_parquet(here("data", "analysis", "projects_combined.parquet"))
bert <- read_parquet(here("data", "analysis", "projects_timeline_bert.parquet"))
noi <- read_parquet(here("data", "analysis", "noi_federal_register.parquet"))

clean_energy <- projects %>% filter(project_energy_type == "Clean")

cat("Clean energy projects:", nrow(clean_energy), "\n")
cat("  CE:", sum(clean_energy$process_type == "CE"), "\n")
cat("  EIS:", sum(clean_energy$process_type == "EIS"), "\n")
cat("  EA:", sum(clean_energy$process_type == "EA"), "\n\n")

# ==========================================================
# 1. EXISTING BERT TIMELINE COVERAGE
# ==========================================================

cat("=" %>% strrep(60), "\n")
cat("  1. BERT TIMELINE COVERAGE (Phase 1 output)\n")
cat("=" %>% strrep(60), "\n\n")

# Merge BERT with current clean energy projects
merged <- clean_energy %>%
  select(project_id, process_type, lead_agency, lead_agency_harmonized,
         project_department, project_type, project_state, project_energy_type) %>%
  left_join(
    bert %>% select(project_id,
                    bert_decision_date_final, bert_initiation_date_final,
                    bert_decision_confidence, bert_n_dates_found,
                    bert_earliest_review_date, bert_latest_review_date),
    by = "project_id"
  ) %>%
  mutate(
    decision_dt = as.Date(bert_decision_date_final),
    initiation_dt = as.Date(bert_initiation_date_final),
    duration_days = as.numeric(decision_dt - initiation_dt),
    decision_year = year(decision_dt),
    has_decision = !is.na(decision_dt),
    has_initiation = !is.na(initiation_dt),
    has_both = has_decision & has_initiation & duration_days > 0
  )

# Overall coverage
cat("--- Overall Coverage ---\n")
coverage_by_type <- merged %>%
  group_by(process_type) %>%
  summarise(
    n = n(),
    has_decision = sum(has_decision, na.rm = TRUE),
    pct_decision = sprintf("%.1f%%", mean(has_decision, na.rm = TRUE) * 100),
    has_initiation = sum(has_initiation, na.rm = TRUE),
    pct_initiation = sprintf("%.1f%%", mean(has_initiation, na.rm = TRUE) * 100),
    has_both = sum(has_both, na.rm = TRUE),
    pct_both = sprintf("%.1f%%", mean(has_both, na.rm = TRUE) * 100),
    .groups = "drop"
  )
print(as.data.frame(coverage_by_type))

cat("\nKey gap: BERT was only run on CE projects. EA (573) and EIS (753)\n")
cat("have ZERO BERT timeline data. Extending BERT to EA/EIS is Step 1.\n\n")

# ==========================================================
# 2. CE DURATION ANALYSIS (where data exists)
# ==========================================================

cat("=" %>% strrep(60), "\n")
cat("  2. CE DURATION DISTRIBUTION\n")
cat("=" %>% strrep(60), "\n\n")

valid_dur <- merged %>% filter(has_both, duration_days > 0, duration_days < 7300)

cat("CE projects with valid duration:", nrow(valid_dur), "\n")
cat("(Excludes durations > 20 years as likely data quality issues)\n\n")

cat("Duration percentiles:\n")
percentiles <- quantile(valid_dur$duration_days, probs = c(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))
for (i in seq_along(percentiles)) {
  cat(sprintf("  %s: %d days (%.1f years)\n",
              names(percentiles)[i], percentiles[i], percentiles[i] / 365))
}

cat("\nDuration buckets:\n")
valid_dur <- valid_dur %>%
  mutate(duration_bucket = case_when(
    duration_days < 30 ~ "< 30 days",
    duration_days < 90 ~ "30-90 days",
    duration_days < 365 ~ "90 days - 1 year",
    duration_days < 730 ~ "1-2 years",
    duration_days < 1825 ~ "2-5 years",
    TRUE ~ "> 5 years"
  ))
valid_dur %>%
  count(duration_bucket) %>%
  mutate(pct = sprintf("%.1f%%", n / sum(n) * 100)) %>%
  print()

# ==========================================================
# 3. SEGMENTATION BY AGENCY
# ==========================================================

cat("\n")
cat("=" %>% strrep(60), "\n")
cat("  3. SEGMENTATION BY AGENCY\n")
cat("=" %>% strrep(60), "\n\n")

agency_stats <- merged %>%
  filter(process_type == "CE") %>%
  group_by(lead_agency_harmonized) %>%
  summarise(
    n = n(),
    has_decision = sum(has_decision, na.rm = TRUE),
    has_both = sum(has_both, na.rm = TRUE),
    median_days = median(duration_days[has_both], na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n >= 10) %>%
  mutate(
    pct_decision = sprintf("%.1f%%", has_decision / n * 100),
    pct_both = sprintf("%.1f%%", has_both / n * 100),
    median_yrs = ifelse(!is.na(median_days), sprintf("%.1f", median_days / 365), "N/A")
  ) %>%
  arrange(desc(n))

cat("Agencies with >= 10 CE clean energy projects:\n")
print(as.data.frame(agency_stats %>% head(10)))

cat("\nDOE has 16K projects but only 17% have both dates.\n")
cat("BLM has 3.2K projects with 57% having both dates.\n")
cat("Agency is a strong segmentation variable.\n")

# ==========================================================
# 4. SEGMENTATION BY YEAR
# ==========================================================

cat("\n")
cat("=" %>% strrep(60), "\n")
cat("  4. SEGMENTATION BY DECISION YEAR\n")
cat("=" %>% strrep(60), "\n\n")

year_stats <- valid_dur %>%
  filter(decision_year >= 2009, decision_year <= 2024) %>%
  group_by(decision_year) %>%
  summarise(
    n = n(),
    median_days = median(duration_days),
    mean_days = mean(duration_days),
    .groups = "drop"
  )
print(as.data.frame(year_stats))

cat("\nNote: Year-over-year trends visible but noisy.\n")
cat("Spike in 2009-2010 = Recovery Act / ARRA projects (fast DOE CEs).\n")

# ==========================================================
# 5. SEGMENTATION BY STATE
# ==========================================================

cat("\n")
cat("=" %>% strrep(60), "\n")
cat("  5. SEGMENTATION BY STATE (top 15)\n")
cat("=" %>% strrep(60), "\n\n")

state_stats <- valid_dur %>%
  group_by(project_state) %>%
  summarise(
    n = n(),
    median_days = median(duration_days),
    mean_days = mean(duration_days),
    .groups = "drop"
  ) %>%
  filter(n >= 20) %>%
  arrange(desc(n))
print(as.data.frame(state_stats %>% head(15)))

# ==========================================================
# 6. PRE/POST FAST-41 COMPARISON
# ==========================================================

cat("\n")
cat("=" %>% strrep(60), "\n")
cat("  6. PRE/POST FAST-41 (December 2015)\n")
cat("=" %>% strrep(60), "\n\n")

fast41 <- valid_dur %>%
  mutate(era = ifelse(decision_dt < as.Date("2016-01-01"), "Pre-FAST-41", "Post-FAST-41")) %>%
  group_by(era) %>%
  summarise(
    n = n(),
    median_days = median(duration_days),
    mean_days = mean(duration_days),
    pct_under_30 = sprintf("%.1f%%", mean(duration_days < 30) * 100),
    pct_under_90 = sprintf("%.1f%%", mean(duration_days < 90) * 100),
    .groups = "drop"
  )
print(as.data.frame(fast41))
cat("\nCaution: Post-FAST-41 appears LONGER, but this is likely a selection\n")
cat("effect — later years have better initiation coverage, so longer\n")
cat("projects are now included that were previously missed.\n")

# ==========================================================
# 7. FEDERAL REGISTER NOI CROSS-REFERENCE
# ==========================================================

cat("\n")
cat("=" %>% strrep(60), "\n")
cat("  7. FEDERAL REGISTER NOI DATA\n")
cat("=" %>% strrep(60), "\n\n")

noi_ce <- noi %>%
  inner_join(clean_energy %>% select(project_id), by = "project_id")

cat("NOI matches for clean energy:", nrow(noi_ce), "\n")
cat("  With publication date:", sum(!is.na(noi_ce$noi_publication_date)), "\n")
cat("  Without date:", sum(is.na(noi_ce$noi_publication_date)), "\n")

# What process types have NOI matches?
noi_process <- noi_ce %>%
  inner_join(clean_energy %>% select(project_id, process_type), by = "project_id") %>%
  count(process_type)
cat("\nNOI matches by process type:\n")
print(as.data.frame(noi_process))

# NOIs with dates — can they fill initiation gaps?
noi_with_date <- noi_ce %>%
  filter(!is.na(noi_publication_date)) %>%
  inner_join(merged %>% select(project_id, has_initiation, process_type), by = "project_id")

fills_gap <- noi_with_date %>%
  filter(!has_initiation)
cat("\nNOI dates that could fill initiation gaps:", nrow(fills_gap), "\n")
cat("(Projects with NOI date but no BERT initiation date)\n")

# ==========================================================
# 8. DATA QUALITY FLAGS
# ==========================================================

cat("\n")
cat("=" %>% strrep(60), "\n")
cat("  8. DATA QUALITY ISSUES\n")
cat("=" %>% strrep(60), "\n\n")

# Future dates
future <- merged %>% filter(decision_dt > as.Date("2025-12-31"))
cat("Decision dates after 2025:", nrow(future), "\n")

# Very short durations (same day)
same_day <- merged %>% filter(has_decision, has_initiation, duration_days == 0)
cat("Same-day initiation and decision:", nrow(same_day), "\n")

# Negative durations (initiation after decision)
negative <- merged %>% filter(has_decision, has_initiation, duration_days < 0)
cat("Negative duration (init after decision):", nrow(negative), "\n")

# Suspiciously long
long <- merged %>% filter(duration_days > 7300)
cat("Duration > 20 years:", nrow(long), "\n")

# 1-day durations
one_day <- merged %>% filter(duration_days == 1)
cat("Duration exactly 1 day:", nrow(one_day), "\n")

cat("\nTotal data quality concerns:", nrow(future) + nrow(same_day) + nrow(negative) + nrow(long), "\n")

# ==========================================================
# 9. EA/EIS GAP ASSESSMENT
# ==========================================================

cat("\n")
cat("=" %>% strrep(60), "\n")
cat("  9. EA/EIS GAP — WHAT NEEDS TO BE DONE\n")
cat("=" %>% strrep(60), "\n\n")

ea_eis <- clean_energy %>% filter(process_type %in% c("EA", "EIS"))
cat("Clean energy EA/EIS projects:", nrow(ea_eis), "\n")
cat("  EA:", sum(ea_eis$process_type == "EA"), "\n")
cat("  EIS:", sum(ea_eis$process_type == "EIS"), "\n\n")

# Check page counts for EA and EIS
ea_pages <- read_parquet(here("data", "processed", "ea", "documents.parquet"))
eis_pages <- read_parquet(here("data", "processed", "eis", "documents.parquet"))

cat("EA documents:", nrow(ea_pages), "\n")
if ("total_pages" %in% names(ea_pages)) {
  cat("EA median pages per doc:", median(ea_pages$total_pages, na.rm = TRUE), "\n")
  cat("EA total pages:", sum(ea_pages$total_pages, na.rm = TRUE), "\n")
}

cat("EIS documents:", nrow(eis_pages), "\n")
if ("total_pages" %in% names(eis_pages)) {
  cat("EIS median pages per doc:", median(eis_pages$total_pages, na.rm = TRUE), "\n")
  cat("EIS total pages:", sum(eis_pages$total_pages, na.rm = TRUE), "\n")
}

cat("\nTo extend timelines to EA/EIS, need to:\n")
cat("  1. Run regex extraction on EA and EIS pages (new regex_candidates)\n")
cat("  2. Apply BERT classifier (already trained, just needs new input)\n")
cat("  3. Validate on EA/EIS sample (different document structure)\n")
cat("  4. EA/EIS documents are MUCH longer than CEs — more dates, more noise\n")

# ==========================================================
# VERDICT
# ==========================================================

cat("\n")
cat("=" %>% strrep(60), "\n")
cat("  POC VERDICT\n")
cat("=" %>% strrep(60), "\n")
cat("\n")
cat("WHAT WE ALREADY HAVE:\n")
cat(sprintf("  - CE decision dates: %d / %d (%.1f%%)\n",
            sum(merged$has_decision & merged$process_type == "CE"),
            sum(merged$process_type == "CE"),
            mean(merged$has_decision[merged$process_type == "CE"]) * 100))
cat(sprintf("  - CE both dates (for duration): %d / %d (%.1f%%)\n",
            sum(merged$has_both & merged$process_type == "CE", na.rm = TRUE),
            sum(merged$process_type == "CE"),
            mean(merged$has_both[merged$process_type == "CE"], na.rm = TRUE) * 100))
cat(sprintf("  - Median CE duration: %d days (%.1f years)\n",
            median(valid_dur$duration_days),
            median(valid_dur$duration_days) / 365))
cat("\n")
cat("WHAT PHASE 2 NEEDS TO ADD:\n")
cat("  1. Extend BERT to EA/EIS (1,326 clean energy projects, 0% coverage now)\n")
cat("  2. Improve initiation coverage (currently 48.5% for CE)\n")
cat("  3. Cross-reference Federal Register NOIs (only 205 usable dates)\n")
cat("  4. Segmentation analysis (agency, year, state, pre/post-FAST-41)\n")
cat("  5. Data quality cleanup (future dates, negative durations)\n")
cat("  6. Outlier identification for case studies\n")
cat("\n")
cat("FEASIBILITY: HIGH. The BERT pipeline exists and works for CE.\n")
cat("The main work is extension (EA/EIS) and analysis, not new methods.\n")
cat("Segmentation data is rich — agency, year, state, project type all\n")
cat("show meaningful variation. Duration analysis is possible for ~4,600\n")
cat("CE projects today and will grow as initiation coverage improves.\n")
cat("=" %>% strrep(60), "\n")
