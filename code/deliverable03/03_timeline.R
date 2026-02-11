# --------------------------
# DELIVERABLE 3: TIMELINE ANALYSIS
# --------------------------
# BERT-based timeline extraction for clean energy CE projects
# Full run: data/analysis/projects_timeline_bert.parquet

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable03", "00_setup.R"))

# --------------------------
# LOAD BERT TIMELINE DATA
# --------------------------

bert_timeline_path <- here("data", "analysis", "projects_timeline_bert.parquet")
timeline <- read_parquet(bert_timeline_path)
cat("Projects loaded:", nrow(timeline), "\n\n")

# Derive year from decision date (or inferred application date as fallback)
timeline <- timeline %>%
  mutate(
    bert_decision_date = as.Date(bert_decision_date),
    bert_application_date = as.Date(bert_application_date),
    bert_inferred_application_date = as.Date(bert_inferred_application_date),
    bert_earliest_review_date = as.Date(bert_earliest_review_date),
    # Year from decision date
    bert_year = as.integer(format(bert_decision_date, "%Y")),
    # Duration: decision minus best available start date
    bert_start_date = coalesce(bert_application_date, bert_inferred_application_date),
    bert_duration_days = as.numeric(bert_decision_date - bert_start_date)
  ) |> 
  glimpse()

# --------------------------
# TABLE 1: EXTRACTION COVERAGE SUMMARY
# --------------------------

cat("=== BERT Extraction Coverage ===\n\n")

n_total <- nrow(timeline)
n_has_decision <- sum(!is.na(timeline$bert_decision_date))
n_has_app <- sum(!is.na(timeline$bert_application_date))
n_has_inferred_app <- sum(!is.na(timeline$bert_inferred_application_date))
n_has_review <- sum(timeline$bert_n_review_dates > 0)
n_has_any_start <- sum(!is.na(timeline$bert_start_date))
n_has_duration <- sum(!is.na(timeline$bert_duration_days) & timeline$bert_duration_days >= 0)
n_errors <- sum(!is.na(timeline$bert_error))

coverage_table <- tibble(
  Metric = c(
    "Total CE clean energy projects",
    "Decision date found",
    "Explicit initiation date found",
    "Inferred initiation (earliest review as proxy)",
    "Any start date (explicit or inferred)",
    "Review dates found (at least one)",
    "Duration calculable (decision + start, >= 0 days)",
    "Errors (no dates extracted)"
  ),
  Count = c(n_total, n_has_decision, n_has_app, n_has_inferred_app,
            n_has_any_start, n_has_review, n_has_duration, n_errors),
  Percent = sprintf("%.1f%%", 100 * Count / n_total)
)

# --------------------------
# FIGURE: DATE COUNT DISTRIBUTION PER PROJECT
# --------------------------

cat("\nCreating Figure: Date count distribution per project...\n")

date_dist <- timeline %>%
  mutate(
    n_dates_bin = case_when(
      bert_n_dates_found == 0 ~ "0",
      bert_n_dates_found == 1 ~ "1",
      bert_n_dates_found == 2 ~ "2",
      bert_n_dates_found == 3 ~ "3",
      bert_n_dates_found <= 5 ~ "4-5",
      bert_n_dates_found <= 10 ~ "6-10",
      TRUE ~ "11+"
    ),
    n_dates_bin = factor(n_dates_bin,
                         levels = c("0", "1", "2", "3", "4-5", "6-10", "11+"))
  ) %>%
  count(n_dates_bin, name = "n") %>%
  mutate(pct = 100 * n / sum(n))

fig_date_dist <- ggplot(date_dist, aes(x = n_dates_bin, y = n)) +
  geom_col(fill = catf_dark_blue, alpha = 0.8) +
  geom_text(aes(label = sprintf("%s\n(%.0f%%)", scales::comma(n), pct)),
            vjust = -0.3, size = 3, color = "gray30") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2)), labels = scales::comma) +
  labs(
    title = "Number of Dates Extracted per Project",
    subtitle = sprintf("%s CE clean energy projects | BERT classifier",
                       scales::comma(n_total)),
    x = "Dates extracted per project",
    y = "Number of projects"
  ) +
  theme_catf()

fig_date_dist

fig_date_dist_path <- here(figures_dir, "03_bert_date_distribution.png")
ggsave(fig_date_dist_path, fig_date_dist, width = 8, height = 6, dpi = 300)
cat("  Saved:", fig_date_dist_path, "\n")
print(fig_date_dist)

# --------------------------
# FIGURE: CE PROJECTS BY DECISION YEAR
# --------------------------

cat("\nCreating Figure: CE projects by decision year...\n")

year_counts <- timeline %>%
  filter(!is.na(bert_year)) %>%
  filter(bert_year >= 2000, bert_year <= 2025) %>%
  count(bert_year, name = "n_projects")

fig_by_year <- ggplot(year_counts, aes(x = bert_year, y = n_projects)) +
  geom_col(fill = catf_dark_blue, alpha = 0.8) +
  geom_text(aes(label = scales::comma(n_projects)), vjust = -0.5, size = 2.5, color = "gray30") +
  scale_x_continuous(breaks = seq(2000, 2025, by = 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)), labels = scales::comma) +
  labs(
    title = "CE Clean Energy Projects by Decision Year",
    subtitle = sprintf("%s projects with decision date | BERT classifier",
                       scales::comma(sum(year_counts$n_projects))),
    x = "Decision Year",
    y = "Number of Projects",
    caption = "Year derived from BERT-classified decision date (signature/approval)."
  ) +
  theme_catf()

fig_by_year

fig_by_year_path <- here(figures_dir, "03_projects_by_year.png")
ggsave(fig_by_year_path, fig_by_year, width = 10, height = 6, dpi = 300)
cat("  Saved:", fig_by_year_path, "\n")
print(fig_by_year)

# --------------------------
# FIGURE: EXTRACTION COVERAGE BREAKDOWN
# --------------------------

cat("\nCreating Figure: Extraction coverage breakdown...\n")

coverage_bars <- tibble(
  category = c("Decision date", "Explicit initiation", "Inferred initiation",
                "Any start date", "Review dates"),
  count = c(n_has_decision, n_has_app, n_has_inferred_app,
            n_has_any_start, n_has_review),
  pct = 100 * count / n_total
) %>%
  mutate(category = factor(category, levels = rev(category))) |> 
  filter(category != "Inferred initiation") |> 
  glimpse()

fig_coverage <- ggplot(coverage_bars, aes(x = category, y = pct)) +
  geom_col(fill = catf_blue, alpha = 0.8) +
  geom_text(aes(label = sprintf("%.0f%%\n(%s)", pct, scales::comma(count))),
            hjust = -0.1, size = 3, color = "gray30") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.3)), limits = c(0, 100)) +
  coord_flip() +
  labs(
    title = "Timeline Extraction Coverage",
    subtitle = sprintf("%s CE clean energy projects | BERT classifier",
                       scales::comma(n_total)),
    x = NULL,
    y = "Percent of total projects (20,863)"
  ) +
  theme_catf()

fig_coverage

fig_coverage_path <- here(figures_dir, "03_bert_coverage.png")
ggsave(fig_coverage_path, fig_coverage, width = 9, height = 5, dpi = 300)
cat("  Saved:", fig_coverage_path, "\n")
print(fig_coverage)

# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Timeline Analysis Summary ===\n\n")
cat("Decision date coverage:", sprintf("%.0f%%", 100 * n_has_decision / n_total), "\n")
cat("Explicit initiation coverage:", sprintf("%.0f%%", 100 * n_has_app / n_total), "\n")
cat("Inferred initiation coverage:", sprintf("%.0f%%", 100 * n_has_inferred_app / n_total), "\n")
cat("Duration calculable:", sprintf("%.0f%%", 100 * n_has_duration / n_total), "\n")
cat("Median dates per project:", median(timeline$bert_n_dates_found), "\n\n")

cat("Files saved to:", figures_dir, "\n")
cat("Tables saved to:", tables_dir, "\n")

# --------------------------
# BERT TIMELINE EXAMPLES (for client review)
# --------------------------
# Source: code/exploratory/timeline/01_compare_decisions.R
# Shows 6 curated project examples from BERT v8 classification

cat("\n=== BERT Timeline Examples ===\n\n")

# --- helpers (JSON parsing) ---

safe_fromJSON <- function(x) {
  tryCatch(fromJSON(x, flatten = TRUE), error = function(e) NULL)
}

normalize_parsed <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.data.frame(x)) return(as_tibble(x))
  if (is.list(x)) {
    if (!is.null(names(x)) && length(names(x)) > 0) {
      return(as_tibble(x))
    }
    return(bind_rows(lapply(x, as_tibble)))
  }
  NULL
}

extract_contexts <- function(df, json_col, model_label) {
  df %>%
    mutate(parsed = map(.data[[json_col]], safe_fromJSON)) %>%
    mutate(parsed = map(parsed, normalize_parsed)) %>%
    select(project_id, project_title, lead_agency, parsed) %>%
    unnest(parsed) %>%
    mutate(model = model_label) %>%
    select(project_id, project_title, lead_agency, model, type, date, source, confidence, everything())
}

# --- load BERT v8 results ---

bert_path <- here("data", "analysis", "test50_bert_v8.parquet")
bert <- read_parquet(bert_path)
bert_ctx <- extract_contexts(bert, "bert_dates_json", "bert")

cat("BERT v8 results loaded:", nrow(bert), "projects,",
    nrow(bert_ctx), "date contexts\n\n")

# --- curated project examples ---

example_ids <- c(
  "3e3bb9f5-f5ab-651d-b2d1-50ec99d99db0",
  "46f4da85-af1c-0e66-a706-9a7292dd9689",
  "824ba268-8ddf-a34f-f9a7-625e7727c242",
  "f2812da0-16c5-fbd1-9e16-10bf8e67c514",
  "dec68c6f-da24-f178-7bf9-30dcd886fb12",
  "5c512493-33a9-ff2c-5f13-3a8d55464b93"
)

examples_list <- list()

for (i in seq_along(example_ids)) {
  ex <- bert_ctx %>%
    filter(project_id == example_ids[i]) %>%
    select(project_title, type, date, source) %>%
    arrange(date) %>%
    mutate(example = i)

  examples_list[[i]] <- ex

  cat(sprintf("Example %d (%s): %d date contexts\n",
              i, example_ids[i], nrow(ex)))
}

# Combine all examples into one table
examples_all <- bind_rows(examples_list)

examples_all |> glimpse()

# Save combined CSV
examples_csv_path <- here(tables_dir, "03_bert_client_examples.csv")
write_csv(examples_all, examples_csv_path)
cat("\nSaved combined examples:", examples_csv_path, "\n")

# Save individual CSVs
for (i in seq_along(examples_list)) {
  ex_path <- here(tables_dir, sprintf("03_bert_example%d.csv", i))
  write_csv(examples_list[[i]], ex_path)
}
cat("Saved individual example CSVs to:", tables_dir, "\n")

# Write to Google Sheets
gs_url <- "https://docs.google.com/spreadsheets/d/1HuvVNDiPAG3WegTy58yn_LLUQ8RnSFwTg0BeabcyM08/edit?usp=sharing"

#for (i in seq_along(examples_list)) {
#  sheet_write(
#    data = examples_list[[i]],
#    ss = gs_url,
#    sheet = sprintf("example%d", i)
#  )
#}
cat("Written examples to Google Sheet\n")
