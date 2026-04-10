# --------------------------
# VALIDATION: BERT TIMELINE COVERAGE
# --------------------------
# Inspect the BERT timeline output to understand:
#   - Coverage of decision and initiation dates
#   - Confidence distribution
#   - Failure modes and gap patterns
#   - Year plausibility of selected dates

rm(list = ls())
source(here::here("code", "validation", "timeline", "00_setup.R"))

library(jsonlite)

safe_fromJSON <- function(x) {
  tryCatch(fromJSON(x, flatten = TRUE), error = function(e) NULL)
}

# --------------------------
# LOAD
# --------------------------
bert <- read_parquet(bert_path) |> as_tibble()
cat("Projects loaded:", nrow(bert), "\n")

# --------------------------
# 1. COVERAGE SUMMARY
# --------------------------
cat("\n── Coverage summary ─────────────────────────────────────\n")
bert |>
  summarise(
    n_total              = n(),
    has_decision         = sum(!is.na(bert_decision_date) & bert_decision_date != ""),
    has_decision_final   = sum(!is.na(bert_decision_date_final) & bert_decision_date_final != ""),
    has_initiation       = sum(!is.na(bert_initiation_date_final) & bert_initiation_date_final != ""),
    has_application      = sum(!is.na(bert_application_date) & bert_application_date != ""),
    has_both             = sum(
      (!is.na(bert_decision_date_final) & bert_decision_date_final != "") &
      (!is.na(bert_initiation_date_final) & bert_initiation_date_final != "")
    ),
    has_error            = sum(!is.na(bert_error) & bert_error != ""),
  ) |>
  pivot_longer(everything(), names_to = "metric", values_to = "n") |>
  mutate(pct = round(n / n[metric == "n_total"] * 100, 1)) |>
  print()

# --------------------------
# 2. CONFIDENCE DISTRIBUTION
# --------------------------
cat("\n── Decision confidence distribution ─────────────────────\n")
bert |>
  filter(!is.na(bert_decision_confidence)) |>
  count(bert_decision_confidence, sort = TRUE) |>
  mutate(pct = round(n / sum(n) * 100, 1)) |>
  print()

p_conf <- bert |>
  filter(!is.na(bert_decision_confidence)) |>
  ggplot(aes(x = factor(bert_decision_confidence))) +
  geom_bar(fill = catf_colors["blue"]) +
  labs(
    title = "BERT decision date confidence scores",
    x = "Confidence", y = "Projects"
  ) +
  theme_nepa()

print(p_conf)

# --------------------------
# 3. DECISION DATE YEAR DISTRIBUTION
# --------------------------
cat("\n── Decision date year distribution ──────────────────────\n")
decision_years <- bert |>
  filter(!is.na(bert_decision_date_final), bert_decision_date_final != "") |>
  mutate(year = as.integer(substr(bert_decision_date_final, 1, 4))) |>
  filter(!is.na(year))

decision_years |>
  count(year) |>
  print(n = 30)

p_decision_year <- decision_years |>
  count(year) |>
  filter(year >= 1990, year <= 2030) |>
  ggplot(aes(x = year, y = n)) +
  geom_col(fill = catf_colors["navy"]) +
  scale_x_continuous(breaks = seq(1990, 2030, 5)) +
  labs(
    title    = "BERT decision dates by year",
    subtitle = "CE clean energy projects",
    x = "Year", y = "Projects"
  ) +
  theme_nepa()

print(p_decision_year)

# --------------------------
# 4. INITIATION DATE YEAR DISTRIBUTION
# --------------------------
initiation_years <- bert |>
  filter(!is.na(bert_initiation_date_final), bert_initiation_date_final != "") |>
  mutate(year = as.integer(substr(bert_initiation_date_final, 1, 4))) |>
  filter(!is.na(year))

p_init_year <- initiation_years |>
  count(year) |>
  filter(year >= 1990, year <= 2030) |>
  ggplot(aes(x = year, y = n)) +
  geom_col(fill = catf_colors["green"]) +
  scale_x_continuous(breaks = seq(1990, 2030, 5)) +
  labs(
    title    = "BERT initiation dates by year",
    subtitle = "CE clean energy projects",
    x = "Year", y = "Projects"
  ) +
  theme_nepa()

print(p_init_year)

# --------------------------
# 5. DURATION PLAUSIBILITY
# --------------------------
cat("\n── Duration (decision − initiation) in days ─────────────\n")
duration <- bert |>
  filter(
    !is.na(bert_decision_date_final), bert_decision_date_final != "",
    !is.na(bert_initiation_date_final), bert_initiation_date_final != ""
  ) |>
  mutate(
    d_decision   = as.Date(bert_decision_date_final),
    d_initiation = as.Date(bert_initiation_date_final),
    duration_days = as.integer(d_decision - d_initiation)
  )

duration |>
  summarise(
    n             = n(),
    negative      = sum(duration_days < 0),
    under_30_days = sum(duration_days >= 0 & duration_days < 30),
    median_days   = median(duration_days),
    p25_days      = quantile(duration_days, 0.25),
    p75_days      = quantile(duration_days, 0.75),
    max_days      = max(duration_days)
  ) |>
  print()

p_duration <- duration |>
  filter(duration_days >= 0, duration_days <= 365 * 15) |>
  mutate(duration_years = duration_days / 365.25) |>
  ggplot(aes(x = duration_years)) +
  geom_histogram(binwidth = 0.5, fill = catf_colors["blue"], colour = "white") +
  labs(
    title    = "Review duration: initiation → decision (years)",
    subtitle = "Capped at 15 years for display; negative durations excluded",
    x = "Duration (years)", y = "Projects"
  ) +
  theme_nepa()

print(p_duration)

# --------------------------
# 6. FAILURE MODES
# --------------------------
cat("\n── Projects with no decision date — what do they look like? ──\n")
no_decision <- bert |>
  filter(is.na(bert_decision_date_final) | bert_decision_date_final == "")

cat("Count:", nrow(no_decision), "\n")

# How many candidates did they have in the regex stage?
ce_cands <- read_parquet(regex_ce_new_path) |> as_tibble()
no_decision |>
  left_join(
    ce_cands |> count(project_id, name = "n_candidates"),
    by = "project_id"
  ) |>
  summarise(
    zero_candidates   = sum(n_candidates == 0 | is.na(n_candidates)),
    some_candidates   = sum(!is.na(n_candidates) & n_candidates > 0),
    median_candidates = median(n_candidates, na.rm = TRUE)
  ) |>
  print()

cat("\n── Projects with no initiation date ─────────────────────\n")
no_init <- bert |>
  filter(is.na(bert_initiation_date_final) | bert_initiation_date_final == "")
cat("Count:", nrow(no_init), "of", nrow(bert), "\n")

# Candidate coverage for no-initiation projects
no_init |>
  left_join(
    ce_cands |> count(project_id, name = "n_candidates"),
    by = "project_id"
  ) |>
  summarise(
    zero_candidates   = sum(n_candidates == 0 | is.na(n_candidates)),
    some_candidates   = sum(!is.na(n_candidates) & n_candidates > 0),
    median_candidates = median(n_candidates, na.rm = TRUE)
  ) |>
  print()

# --------------------------
# 7. QUICK LOOKUP: inspect a specific project
# --------------------------
inspect_bert <- function(pid) {
  row <- bert |> filter(project_id == pid)
  if (nrow(row) == 0) { cat("Project not found:", pid, "\n"); return(invisible(NULL)) }
  cat("\n── Project:", pid, "──────────────────────────────────────\n")
  cat("Title:       ", row$project_title[1], "\n")
  cat("Agency:      ", row$lead_agency[1], "\n")
  cat("Decision:    ", row$bert_decision_date_final[1], "(confidence:", row$bert_decision_confidence[1], ")\n")
  cat("Initiation:  ", row$bert_initiation_date_final[1], "\n")
  cat("Application: ", row$bert_application_date[1], "\n")
  cat("n dates found:", row$bert_n_dates_found[1], "\n")
  if (!is.na(row$bert_error[1]) && row$bert_error[1] != "") {
    cat("Error:       ", row$bert_error[1], "\n")
  }
  # Show the raw BERT date candidates
  if (!is.na(row$bert_dates_json[1])) {
    parsed <- safe_fromJSON(row$bert_dates_json[1])
    if (!is.null(parsed)) {
      cat("\nBERT date candidates:\n")
      print(as_tibble(parsed))
    }
  }
  # Also show regex candidates for comparison
  cands <- ce_cands |>
    filter(project_id == pid) |>
    select(date, match, context, position_pct, doc_type)
  cat("\nRegex candidates (", nrow(cands), " total):\n", sep = "")
  print(cands, n = 20)
}

# Example: inspect a project with no initiation date
sample_no_init <- no_init |> slice_head(n = 1) |> pull(project_id)
inspect_bert(sample_no_init)

# To inspect any project, call:
#   inspect_bert("your-project-id-here")
