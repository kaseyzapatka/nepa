# --------------------------
# DELIVERABLE 3: GENERATION CAPACITY
# --------------------------
# Table 2: Generation Capacity by Process Type
# Analyzes clean energy projects by their generation capacity

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable3", "00_setup.R"))

# --------------------------
# FILE PATHS
# --------------------------

gencap_path <- here("data", "analysis", "projects_gencap.parquet")

# --------------------------
# TABLE 2: GENERATION CAPACITY
# --------------------------

cat("Creating Table 2: Generation Capacity...\n")

if (file.exists(gencap_path)) {
  gencap_projects <- read_parquet(gencap_path)

  # Filter to projects with capacity data
  has_cap <- gencap_projects %>%
    filter(!is.na(project_gencap_value))

  cat("  Projects with capacity data:", nrow(has_cap), "\n")

  if (nrow(has_cap) > 0) {

    # --------------------------
    # NORMALIZE TO MW
    # --------------------------

    gencap_projects <- gencap_projects %>%
      mutate(
        capacity_mw = case_when(
          project_gencap_unit == "GW" ~ project_gencap_value * 1000,
          project_gencap_unit == "kW" ~ project_gencap_value / 1000,
          project_gencap_unit == "MWh" ~ project_gencap_value,
          TRUE ~ project_gencap_value
        )
      )

    # Filter to reasonable range (remove outliers)
    gencap_reasonable <- gencap_projects %>%
      filter(!is.na(capacity_mw) & capacity_mw > 0 & capacity_mw <= 5000)

    cat("  Projects with reasonable capacity (<=5000 MW):", nrow(gencap_reasonable), "\n")

    # --------------------------
    # CREATE CAPACITY CATEGORIES
    # --------------------------

    gencap_reasonable <- gencap_reasonable %>%
      mutate(
        capacity_category = case_when(
          capacity_mw < 10 ~ "Small (<10 MW)",
          capacity_mw < 100 ~ "Medium (10-100 MW)",
          capacity_mw < 500 ~ "Large (100-500 MW)",
          TRUE ~ "Utility-scale (>500 MW)"
        ),
        capacity_category = factor(
          capacity_category,
          levels = c("Small (<10 MW)", "Medium (10-100 MW)", "Large (100-500 MW)", "Utility-scale (>500 MW)")
        )
      )

    # --------------------------
    # TABLE 2: CAPACITY BY PROCESS TYPE
    # --------------------------

    table2 <- gencap_reasonable %>%
      group_by(capacity_category, dataset_source) %>%
      summarise(n = n(), .groups = "drop") %>%
      pivot_wider(
        names_from = dataset_source,
        values_from = n,
        values_fill = 0
      ) %>%
      mutate(Total = rowSums(select(., -1), na.rm = TRUE))

    # Add totals row
    totals_row <- table2 %>%
      summarise(
        capacity_category = "Total",
        CE = sum(CE, na.rm = TRUE),
        EA = sum(EA, na.rm = TRUE),
        EIS = sum(EIS, na.rm = TRUE),
        Total = sum(Total, na.rm = TRUE)
      )

    table2 <- bind_rows(table2, totals_row)

    # Rename for output
    table2 <- table2 %>%
      rename(
        `Generation Capacity` = capacity_category,
        `Categorical Exclusion` = CE,
        `Environmental Assessment` = EA,
        `Environmental Impact Statement` = EIS
      )

    table2 %>% print()

    output_file2 <- here(tables_dir, "table2_by_generation_capacity.csv")
    write_csv(table2, output_file2)
    cat("  Saved:", output_file2, "\n")

    # --------------------------
    # FIGURE 1: EXTRACTION COVERAGE BY PROCESS TYPE
    # --------------------------

    cat("\nCreating Figure 1: Extraction Coverage...\n")

    coverage_data <- gencap_projects %>%
      group_by(dataset_source) %>%
      summarise(
        total = n(),
        with_capacity = sum(!is.na(project_gencap_value)),
        reasonable = sum(!is.na(capacity_mw) & capacity_mw > 0 & capacity_mw <= 5000, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        pct_extracted = 100 * with_capacity / total,
        pct_reasonable = 100 * reasonable / total,
        dataset_source = factor(dataset_source, levels = c("CE", "EA", "EIS"))
      )

    fig1 <- coverage_data %>%
      ggplot(aes(x = dataset_source, y = pct_extracted, fill = dataset_source)) +
      geom_col(width = 0.7) +
      geom_text(
        aes(label = paste0(round(pct_extracted, 1), "%")),
        vjust = -0.5,
        size = 4,
        fontface = "bold"
      ) +
      geom_text(
        aes(label = paste0("(", comma(with_capacity), " / ", comma(total), ")"), y = pct_extracted / 2),
        color = "white",
        size = 3.5
      ) +
      labs(
        title = "Generation Capacity Extraction Coverage by Process Type",
        subtitle = "Percentage of clean energy projects with capacity values extracted",
        x = "Process Type",
        y = "Percent with Capacity Extracted",
        caption = "CE = Categorical Exclusion, EA = Environmental Assessment, EIS = Environmental Impact Statement\nLower CE coverage reflects smaller projects that often lack explicit capacity values."
      ) +
      scale_y_continuous(
        limits = c(0, 100),
        labels = percent_format(scale = 1),
        expand = expansion(mult = c(0, 0.1))
      ) +
      scale_fill_brewer(palette = "Set2", guide = "none") +
      theme_minimal() +
      theme(
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"),
        plot.caption = element_text(size = 8, color = "gray50", hjust = 0),
        axis.text = element_text(size = 11)
      )

    fig1

    ggsave(
      filename = here(figures_dir, "04_capacity_coverage.png"),
      plot = fig1,
      width = 8,
      height = 6,
      units = "in",
      dpi = 300
    )
    cat("  Saved: 04_capacity_coverage.png\n")

    # --------------------------
    # FIGURE 2: CAPACITY CATEGORIES BY PROCESS TYPE
    # --------------------------

    cat("\nCreating Figure 2: Capacity Categories...\n")

    cap_category_data <- gencap_reasonable %>%
      group_by(dataset_source, capacity_category) %>%
      summarise(n = n(), .groups = "drop") %>%
      group_by(dataset_source) %>%
      mutate(
        total = sum(n),
        pct = 100 * n / total
      ) %>%
      ungroup() %>%
      mutate(dataset_source = factor(dataset_source, levels = c("CE", "EA", "EIS")))

    fig2 <- cap_category_data %>%
      ggplot(aes(x = dataset_source, y = n, fill = capacity_category)) +
      geom_col(width = 0.7) +
      geom_text(
        aes(label = ifelse(n > 30, comma(n), "")),
        position = position_stack(vjust = 0.5),
        color = "white",
        size = 3.5,
        fontface = "bold"
      ) +
      labs(
        title = "Project Capacity Distribution by Process Type",
        subtitle = "Clean energy projects with extracted generation capacity (reasonable range: 0-5000 MW)",
        x = "Process Type",
        y = "Number of Projects",
        fill = "Capacity Category",
        caption = "CE = Categorical Exclusion, EA = Environmental Assessment, EIS = Environmental Impact Statement\nCapacity normalized to MW. Projects with values >5000 MW excluded as likely extraction errors."
      ) +
      scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.05))) +
      scale_fill_brewer(palette = "YlOrRd", direction = 1) +
      theme_minimal() +
      theme(
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"),
        plot.caption = element_text(size = 8, color = "gray50", hjust = 0),
        legend.position = "right",
        axis.text = element_text(size = 11)
      )

    fig2

    ggsave(
      filename = here(figures_dir, "05_capacity_by_process.png"),
      plot = fig2,
      width = 10,
      height = 6,
      units = "in",
      dpi = 300
    )
    cat("  Saved: 05_capacity_by_process.png\n")

    # --------------------------
    # SAVE SUMMARY STATS
    # --------------------------

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
} else {
  cat("  Generation capacity data not found.\n")
  cat("  Run extract_gencap.py first to generate this data.\n")

  # Create placeholder table
  table2 <- data.frame(
    `Generation Capacity` = c("High", "Medium", "Low", "Total"),
    `Environmental Assessment` = c("TBD", "TBD", "TBD", "TBD"),
    `Environmental Impact Statement` = c("TBD", "TBD", "TBD", "TBD"),
    `Categorical Exclusion` = c("TBD", "TBD", "TBD", "TBD"),
    check.names = FALSE
  )

  output_file2 <- here(tables_dir, "table2_by_generation_capacity_placeholder.csv")
  write_csv(table2, output_file2)
  cat("  Saved placeholder:", output_file2, "\n")
}

# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Capacity Script Complete ===\n")
cat("Tables saved to:", tables_dir, "\n")
cat("Figures saved to:", figures_dir, "\n")
