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
    # Create capacity bins (customize thresholds as needed)
    gencap_projects <- gencap_projects %>%
      mutate(
        capacity_category = case_when(
          is.na(project_gencap_value) ~ "Unknown",
          project_gencap_unit == "kW" & project_gencap_value < 1000 ~ "Low (<1 MW)",
          project_gencap_unit == "kW" & project_gencap_value >= 1000 ~ "Medium (1-100 MW)",
          project_gencap_unit == "MW" & project_gencap_value < 100 ~ "Medium (1-100 MW)",
          project_gencap_unit == "MW" & project_gencap_value >= 100 ~ "High (100+ MW)",
          project_gencap_unit == "GW" ~ "High (100+ MW)",
          TRUE ~ "Unknown"
        )
      )

    table2 <- gencap_projects %>%
      group_by(capacity_category, process_type) %>%
      summarise(n = n(), .groups = "drop") %>%
      pivot_wider(
        names_from = process_type,
        values_from = n,
        values_fill = 0
      ) %>%
      mutate(Total = rowSums(select(., -1), na.rm = TRUE))

    # Reorder categories
    cat_order <- c("High (100+ MW)", "Medium (1-100 MW)", "Low (<1 MW)", "Unknown", "Total")

    # Add totals
    totals_row <- table2 %>%
      summarise(
        capacity_category = "Total",
        EA = sum(EA, na.rm = TRUE),
        EIS = sum(EIS, na.rm = TRUE),
        CE = sum(CE, na.rm = TRUE),
        Total = sum(Total, na.rm = TRUE)
      )

    table2 <- bind_rows(table2, totals_row) %>%
      mutate(capacity_category = factor(capacity_category, levels = cat_order)) %>%
      arrange(capacity_category)

    # Rename
    table2 <- table2 %>%
      rename(
        `Generation Capacity` = capacity_category,
        `Environmental Assessment` = EA,
        `Environmental Impact Statement` = EIS,
        `Categorical Exclusion` = CE
      )

    table2 %>% print()

    output_file2 <- here(tables_dir, "table2_by_generation_capacity.csv")
    write_csv(table2, output_file2)
    cat("  Saved:", output_file2, "\n")
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
