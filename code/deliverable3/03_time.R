# --------------------------
# DELIVERABLE 3: TIMELINE ANALYSIS
# --------------------------
# Preliminary analysis of project timelines
# Note: Timeline data extraction is in development - results are preliminary

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable3", "00_setup.R"))

# --------------------------
# LOAD TIMELINE DATA
# --------------------------

timeline_path <- here("data", "analysis", "projects_timeline.parquet")

if (!file.exists(timeline_path)) {
  cat("WARNING: Timeline data not found at:", timeline_path, "\n")
  cat("Run: python code/extract/extract_timeline.py --run\n")
  cat("Creating placeholder outputs...\n\n")

  # Create placeholder outputs
  placeholder_msg <- "Timeline extraction pending"

} else {
  cat("Loading timeline data from:", timeline_path, "\n")
  timeline <- read_parquet(timeline_path)
  cat("Projects loaded:", nrow(timeline), "\n\n")
}

# --------------------------
# DATA QUALITY SUMMARY
# --------------------------

cat("=== Timeline Data Quality Summary ===\n\n")

if (exists("timeline")) {
  # Calculate data quality metrics
  n_total <- nrow(timeline)
  n_has_dates <- sum(!is.na(timeline$project_date_earliest))
  n_has_year <- sum(!is.na(timeline$project_year))
  n_has_duration <- sum(!is.na(timeline$project_duration_days))

  # Outlier detection
  n_duration_5yr <- sum(timeline$project_duration_days > 365 * 5, na.rm = TRUE)
  n_duration_10yr <- sum(timeline$project_duration_days > 365 * 10, na.rm = TRUE)
  n_needs_review <- sum(timeline$project_timeline_needs_review == TRUE, na.rm = TRUE)

  cat("Total projects:", n_total, "\n")
  cat("Projects with dates extracted:", n_has_dates,
      sprintf("(%.1f%%)", 100 * n_has_dates / n_total), "\n")
  cat("Projects with year assigned:", n_has_year,
      sprintf("(%.1f%%)", 100 * n_has_year / n_total), "\n")
  cat("Projects with duration calculated:", n_has_duration,
      sprintf("(%.1f%%)", 100 * n_has_duration / n_total), "\n\n")

  cat("Outlier analysis:\n")
  cat("  Duration > 5 years:", n_duration_5yr,
      sprintf("(%.1f%%)", 100 * n_duration_5yr / n_has_duration), "\n")
  cat("  Duration > 10 years:", n_duration_10yr,
      sprintf("(%.1f%%)", 100 * n_duration_10yr / n_has_duration), "\n")
  cat("  Flagged for review:", n_needs_review,
      sprintf("(%.1f%%)", 100 * n_needs_review / n_total), "\n\n")

  # Duration statistics
  if (n_has_duration > 0) {
    dur_stats <- timeline %>%
      filter(!is.na(project_duration_days)) %>%
      summarise(
        mean_days = mean(project_duration_days),
        median_days = median(project_duration_days),
        min_days = min(project_duration_days),
        max_days = max(project_duration_days),
        sd_days = sd(project_duration_days)
      )

    cat("Duration statistics (days):\n")
    cat("  Mean:", round(dur_stats$mean_days), "\n")
    cat("  Median:", round(dur_stats$median_days), "\n")
    cat("  Min:", round(dur_stats$min_days), "\n")
    cat("  Max:", round(dur_stats$max_days), "\n")
    cat("  Std Dev:", round(dur_stats$sd_days), "\n\n")
  }
}

# --------------------------
# FIGURE 1: PROJECT COUNTS BY YEAR (FACETED BY PROCESS TYPE)
# --------------------------

cat("Creating Figure 1: Project Counts by Year (faceted by process type)...\n")

if (exists("timeline") && n_has_year > 0) {

  # Prepare data - count by year AND process_type
  year_counts <- timeline %>%
    filter(!is.na(project_year)) %>%
    mutate(project_year = as.integer(project_year)) %>%
    filter(project_year >= 2000, project_year <= 2025) %>%
    count(project_year, process_type, name = "n_projects") %>%
    # Ensure all process types have factor levels for consistent faceting
    mutate(process_type = factor(process_type, levels = c("EA", "EIS", "CE")))

  # Create figure with facets stacked vertically
  fig1 <- ggplot(year_counts, aes(x = project_year, y = n_projects)) +
    geom_col(fill = catf_dark_blue, alpha = 0.8) +
    geom_text(aes(label = scales::comma(n_projects)), vjust = -0.5, size = 2.5, color = "gray30") +
    scale_x_continuous(breaks = seq(2000, 2025, by = 2)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.2)), labels = scales::comma) +
    facet_wrap(~ process_type, ncol = 1, scales = "free_y") +
    labs(
      title = "NEPA Projects by Year and Process Type",
      subtitle = sprintf("Preliminary data | %s projects with year assigned | %d%% flagged for review",
                         scales::comma(n_has_year), round(100 * n_needs_review / n_total)),
      x = "Project Year",
      y = "Number of Projects",
      caption = "Note: Year derived from decision date or latest date in documents.\nData quality under review - results are preliminary."
    ) +
    theme_catf() +
    theme(
      strip.text = element_text(size = rel(1.1), face = "bold"),
      panel.spacing = unit(1, "lines")
    )

  # Save figure - larger height for stacked facets
  fig1_path <- here(figures_dir, "03_projects_by_year.png")
  ggsave(fig1_path, fig1, width = 12, height = 10, dpi = 300)
  cat("  Saved:", fig1_path, "\n")

  print(fig1)

} else {
  cat("  Skipped: No year data available\n")
}

# --------------------------
# FIGURE 2: DURATION DISTRIBUTION
# --------------------------

cat("\nCreating Figure 2: Duration Distribution...\n")

if (exists("timeline") && n_has_duration > 0) {

  # Prepare data - convert to years for readability
  duration_data <- timeline %>%
    filter(!is.na(project_duration_days)) %>%
    mutate(
      duration_years = project_duration_days / 365,
      duration_category = case_when(
        duration_years <= 1 ~ "≤1 year",
        duration_years <= 2 ~ "1-2 years",
        duration_years <= 5 ~ "2-5 years",
        duration_years <= 10 ~ "5-10 years",
        TRUE ~ ">10 years"
      ),
      duration_category = factor(duration_category,
                                  levels = c("≤1 year", "1-2 years", "2-5 years",
                                            "5-10 years", ">10 years"))
    )

  # Summary by category
  duration_summary <- duration_data %>%
    count(duration_category, name = "n") %>%
    mutate(pct = 100 * n / sum(n))

  # Create figure - bar chart of duration categories
  fig2 <- ggplot(duration_summary, aes(x = duration_category, y = n)) +
    geom_col(fill = catf_blue, alpha = 0.8) +
    geom_text(aes(label = sprintf("%d\n(%.0f%%)", n, pct)),
              vjust = -0.3, size = 3, color = "gray30") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
    labs(
      title = "Distribution of Project Duration",
      subtitle = sprintf("Preliminary data | %d projects | High outlier rate indicates data quality issues",
                         n_has_duration),
      x = "Project Duration",
      y = "Number of Projects",
      caption = "Note: Duration = latest date - earliest date in documents.\nDurations >5 years likely include false positive dates (citations, references)."
    ) +
    theme_catf() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

  # Save figure
  fig2_path <- here(figures_dir, "03_duration_distribution.png")
  ggsave(fig2_path, fig2, width = 8, height = 6, dpi = 300)
  cat("  Saved:", fig2_path, "\n")

  print(fig2)

  # Print summary table
  cat("\nDuration category summary:\n")
  print(duration_summary)

} else {
  cat("  Skipped: No duration data available\n")
}

# --------------------------
# FIGURE 3: DURATION BOXPLOT BY PROCESS TYPE
# --------------------------

cat("\nCreating Figure 3: Duration Boxplot by Process Type...\n")

if (exists("timeline") && n_has_duration > 0) {

  # Prepare data - convert to years for readability
  boxplot_data <- timeline %>%
    filter(!is.na(project_duration_days)) %>%
    mutate(
      duration_years = project_duration_days / 365,
      process_type = factor(process_type, levels = c("EA", "EIS", "CE"))
    )

  # Calculate summary stats for annotation
  duration_stats <- boxplot_data %>%
    group_by(process_type) %>%
    summarise(
      n = n(),
      median_years = median(duration_years),
      mean_years = mean(duration_years),
      .groups = "drop"
    )

  # Create boxplot
  fig3 <- ggplot(boxplot_data, aes(x = process_type, y = duration_years, fill = process_type)) +
    geom_boxplot(alpha = 0.7, outlier.alpha = 0.3, outlier.size = 1) +
    geom_text(
      data = duration_stats,
      aes(x = process_type, y = -1, label = sprintf("n = %s", scales::comma(n))),
      size = 3, color = "gray40", vjust = 1
    ) +
    scale_fill_manual(values = c("EA" = catf_dark_blue, "EIS" = catf_blue, "CE" = catf_teal)) +
    scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0.1, 0.05))) +
    coord_cartesian(ylim = c(-2, max(boxplot_data$duration_years) * 1.05)) +
    labs(
      title = "Project Duration by Process Type",
      subtitle = sprintf("Preliminary data | %s projects | Outliers indicate data quality issues",
                         scales::comma(n_has_duration)),
      x = "Process Type",
      y = "Duration (Years)",
      caption = "Note: Duration = latest date - earliest date in documents.\nLong durations likely include false positive dates from citations/references."
    ) +
    theme_catf() +
    theme(legend.position = "none")

  # Save figure
  fig3_path <- here(figures_dir, "03_duration_boxplot.png")
  ggsave(fig3_path, fig3, width = 8, height = 7, dpi = 300)
  cat("  Saved:", fig3_path, "\n")

  print(fig3)

  # Print duration stats by process type
  cat("\nDuration statistics by process type (years):\n")
  print(duration_stats)

} else {
  cat("  Skipped: No duration data available\n")
}

# --------------------------
# TABLE: YEAR BY PROCESS TYPE
# --------------------------

cat("\nCreating Table: Projects by Year and Process Type...\n")

if (exists("timeline") && n_has_year > 0) {

  table_year_process <- timeline %>%
    filter(!is.na(project_year)) %>%
    mutate(project_year = as.integer(project_year)) %>%
    filter(project_year >= 2010, project_year <= 2025) %>%
    count(project_year, process_type) %>%
    pivot_wider(names_from = process_type, values_from = n, values_fill = 0) %>%
    mutate(Total = rowSums(select(., -project_year), na.rm = TRUE)) %>%
    arrange(desc(project_year))

  # Add totals row
  totals <- table_year_process %>%
    summarise(across(where(is.numeric), sum)) %>%
    mutate(project_year = NA_integer_)

  table_year_process <- bind_rows(table_year_process, totals) %>%
    mutate(project_year = ifelse(is.na(project_year), "Total", as.character(project_year)))

  cat("\nProjects by Year and Process Type:\n")
  print(table_year_process)

  # Save table
  table_path <- here(tables_dir, "03_year_by_process_type.csv")
  write_csv(table_year_process, table_path)
  cat("\nSaved:", table_path, "\n")

} else {
  cat("  Skipped: No year data available\n")
}

# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Timeline Analysis Summary ===\n\n")

cat("DATA QUALITY CONCERNS:\n")
cat("1. High outlier rate:", sprintf("%.0f%%", 100 * n_duration_5yr / n_has_duration),
    "of projects show duration >5 years\n")
cat("2. Review flags:", sprintf("%.0f%%", 100 * n_needs_review / n_total),
    "of projects flagged for manual review\n")
cat("3. Sample size: Current data is from test extraction, not full dataset\n\n")

cat("NEXT STEPS FOR IMPROVING TIMELINE DATA:\n")
cat("1. Run full timeline extraction on all projects (EA, EIS, CE)\n")
cat("2. Apply improved citation/reference filtering (already implemented)\n")
cat("3. Consider QA-based extraction for decision dates\n")
cat("4. Validate sample of flagged projects manually\n")
cat("5. Filter to projects with reliable dates for final analysis\n\n")

cat("Files saved to:", figures_dir, "\n")
cat("Tables saved to:", tables_dir, "\n")
