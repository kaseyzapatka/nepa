# --------------------------
# PROJECT REVIEW: DELIVERABLE 3 EXTRAS
# --------------------------
# Tables/diagnostics not referenced in reports/deliverable03.qmd.

source(here::here("code", "deliverable3", "00_setup.R"))

# --------------------------
# RECLASSIFY UTILITIES TO OTHER (MATCH 01_process.R)
# --------------------------

projects <- projects %>%
  mutate(
    project_energy_type = if_else(
      project_energy_type == "Clean" & project_utilities_to_filter_out,
      "Other",
      project_energy_type
    )
  )

# --------------------------
# TABLE 1: PROJECT STATUS BY ENERGY TYPE
# --------------------------

cat("Creating Table 1: Project Status by Energy Type...\n")

table1 <- projects %>%
  group_by(project_energy_type, process_type) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(
    names_from = process_type,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(Total = rowSums(select(., -1), na.rm = TRUE)) %>%
  arrange(desc(Total))

totals_row <- table1 %>%
  summarise(
    project_energy_type = "Total",
    EA = sum(EA, na.rm = TRUE),
    EIS = sum(EIS, na.rm = TRUE),
    CE = sum(CE, na.rm = TRUE),
    Total = sum(Total, na.rm = TRUE)
  )

table1 <- bind_rows(table1, totals_row)

table1 <- table1 %>%
  rename(
    `Energy Type` = project_energy_type,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )

table1 %>% print()

write_csv(table1, here("output", "deliverable3", "tables", "table1_by_energy_type.csv"))
cat("  Saved: table1_by_energy_type.csv\n")

# --------------------------
# DETAILED BREAKDOWN: CLEAN ENERGY BY TECHNOLOGY x PROCESS TYPE
# --------------------------

cat("\nCreating supplementary table: Clean Energy Detail...\n")

clean_energy_detail <- projects %>%
  filter(project_energy_type == "Clean")

clean_by_tech <- clean_energy_detail %>%
  explode_column("project_type") %>%
  filter(!is.na(project_type) & project_type != "") %>%
  group_by(project_type, process_type) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(
    names_from = process_type,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(Total = rowSums(select(., -1), na.rm = TRUE)) %>%
  arrange(desc(Total))

totals_row <- clean_by_tech %>%
  summarise(
    project_type = "Total",
    EA = sum(EA, na.rm = TRUE),
    EIS = sum(EIS, na.rm = TRUE),
    CE = sum(CE, na.rm = TRUE),
    Total = sum(Total, na.rm = TRUE)
  )

clean_by_tech <- bind_rows(clean_by_tech, totals_row)

clean_by_tech <- clean_by_tech %>%
  rename(
    Technology = project_type,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )

clean_by_tech %>% print(n = 20)
write_csv(clean_by_tech, here(tables_dir, "clean_energy_by_technology_detail.csv"))
cat("  Saved: clean_energy_by_technology_detail.csv\n")

# --------------------------
# ENERGY TYPE COUNTS SUMMARY
# --------------------------

cat("\n=== Energy Type Summary ===\n")
summary_stats <- projects %>%
  group_by(project_energy_type) %>%
  summarise(
    count = n(),
    pct = n() / nrow(projects) * 100
  ) %>%
  arrange(desc(count)) %>%
  rename(
    `Energy Type` = project_energy_type,
    `Count` = count,
    `Percent` = pct
  )

print(summary_stats)
write_csv(summary_stats, here(tables_dir, "energy_type_summary.csv"))
cat("  Saved: energy_type_summary.csv\n")

cat("\nProjects flagged for review:", sum(projects$project_energy_type_questions, na.rm = TRUE), "\n")

# --------------------------
# CAPACITY SUMMARY STATS
# --------------------------

gencap_path <- here("data", "analysis", "projects_gencap_merged.parquet")
if (file.exists(gencap_path)) {
  gencap_projects <- read_parquet(gencap_path)

  gencap_projects <- gencap_projects %>%
    mutate(
      capacity_mw = case_when(
        project_gencap_unit == "GW" ~ project_gencap_value * 1000,
        project_gencap_unit == "kW" ~ project_gencap_value / 1000,
        TRUE ~ project_gencap_value
      )
    )

  gencap_reasonable <- gencap_projects %>%
    filter(!is.na(capacity_mw) & capacity_mw > 0 & capacity_mw <= 5000)

  summary_stats <- gencap_reasonable %>%
    group_by(dataset_source) %>%
    summarise(
      n_projects = n(),
      median_mw = median(capacity_mw, na.rm = TRUE),
      mean_mw = mean(capacity_mw, na.rm = TRUE),
      min_mw = min(capacity_mw, na.rm = TRUE),
      max_mw = max(capacity_mw, na.rm = TRUE),
      .groups = "drop"
    )

  write_csv(summary_stats, here(tables_dir, "capacity_summary_stats.csv"))
  cat("  Saved: capacity_summary_stats.csv\n")
}

# --------------------------
# TIMELINE EXTRA TABLES
# --------------------------

bert_timeline_path <- here("data", "analysis", "projects_timeline_bert.parquet")
timeline <- read_parquet(bert_timeline_path)

timeline <- timeline %>%
  mutate(
    bert_decision_date = as.Date(bert_decision_date),
    bert_application_date = as.Date(bert_application_date),
    bert_inferred_application_date = as.Date(bert_inferred_application_date),
    bert_earliest_review_date = as.Date(bert_earliest_review_date),
    bert_year = as.integer(format(bert_decision_date, "%Y")),
    bert_start_date = coalesce(bert_application_date, bert_inferred_application_date),
    bert_duration_days = as.numeric(bert_decision_date - bert_start_date)
  )

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

coverage_path <- here(tables_dir, "03_bert_coverage.csv")
write_csv(coverage_table, coverage_path)
cat("\nSaved:", coverage_path, "\n")

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

write_csv(date_dist, here(tables_dir, "03_bert_date_distribution.csv"))

year_counts <- timeline %>%
  filter(!is.na(bert_year)) %>%
  filter(bert_year >= 2000, bert_year <= 2025) %>%
  count(bert_year, name = "n_projects")

write_csv(year_counts, here(tables_dir, "03_year_by_process_type.csv"))

cat("\n=== Deliverable 3 Extras Complete ===\n")
