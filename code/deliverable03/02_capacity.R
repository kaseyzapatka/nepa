# --------------------------
# DELIVERABLE 3: GENERATION CAPACITY
# --------------------------
# Table 2: Generation Capacity by Process Type
# Analyzes clean energy projects by their generation capacity

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable03", "00_setup.R"))

# --------------------------
# FILE PATHS
# --------------------------

gencap_path <- here("data", "analysis", "projects_gencap_merged.parquet")

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


# --------------------------
# GENERATION CAPACITY EXAMPLES (for client review)
# --------------------------
# Three curated projects illustrating extraction quality:
#   A) Regex and LLM agree
#   B) Regex and LLM disagree (regex grabbed wrong number)
#   C) Multiple capacity dimensions (power + energy + LLM)

cat("\n=== Generation Capacity Examples ===\n\n")

gencap_merged <- read_parquet(here("data", "analysis", "projects_gencap_merged.parquet"))

example_ids <- c(
  "3689d8443cb2835804a5c9e61ccf1d30",    # EA:  Solana / Abengoa Solar (agree)
  "166574ea3c128fb3a46bd0dd1a3fcfc9",     # EIS: Fourmile Hill Geothermal (disagree)
  "6faf66e9757e4d865ceef6462911a854"      # EA:  Advanced Clean Energy Storage (multi)
)

# Helper: safely truncate or return NA
safe_trunc <- function(x, width = 250) {
  x <- as.character(x)
  ifelse(is.na(x) | x == "NA", NA_character_, str_trunc(x, width))
}

examples_list <- list()

for (i in seq_along(example_ids)) {
  row <- gencap_merged %>%
    filter(project_id == example_ids[i])

  if (nrow(row) == 0) {
    cat(sprintf("  Example %d: project_id %s not found, skipping\n", i, example_ids[i]))
    next
  }

  row <- row[1, ]

  # Build rows for each extraction method, showing both context fields
  ex_rows <- list()

  # Regex row (always present if value exists)
  if (!is.na(row$project_gencap_value)) {
    ex_rows[[length(ex_rows) + 1]] <- tibble(
      method = "Regex (power)",
      value = row$project_gencap_value,
      unit = as.character(row$project_gencap_unit),
      confidence = as.character(row$project_gencap_confidence),
      regex_context = safe_trunc(row$project_gencap_context),
      llm_quote = NA_character_
    )
  }

  # Regex energy row (if project has energy value too)
  if (!is.na(row$project_gencap_energy_value) && row$project_gencap_energy_value > 0) {
    ex_rows[[length(ex_rows) + 1]] <- tibble(
      method = "Regex (energy)",
      value = row$project_gencap_energy_value,
      unit = as.character(row$project_gencap_energy_unit),
      confidence = as.character(row$project_gencap_confidence),
      regex_context = safe_trunc(row$project_gencap_context),
      llm_quote = NA_character_
    )
  }

  # LLM row
  if (!is.na(row$llm_capacity_value) && row$llm_capacity_value > 0) {
    ex_rows[[length(ex_rows) + 1]] <- tibble(
      method = "LLM",
      value = row$llm_capacity_value,
      unit = as.character(row$llm_capacity_unit),
      confidence = as.character(row$llm_confidence),
      regex_context = NA_character_,
      llm_quote = safe_trunc(row$llm_source_quote)
    )
  }

  # Final merged row
  if (!is.na(row$project_gencap_final_value)) {
    ex_rows[[length(ex_rows) + 1]] <- tibble(
      method = "Final (merged)",
      value = row$project_gencap_final_value,
      unit = as.character(row$project_gencap_final_unit),
      confidence = as.character(row$project_gencap_final_confidence),
      regex_context = NA_character_,
      llm_quote = NA_character_
    )
  }

  ex <- bind_rows(ex_rows) %>%
    mutate(
      project_title = row$project_title,
      dataset_source = as.character(row$dataset_source),
      .before = 1
    )

  examples_list[[i]] <- ex
  cat(sprintf("Example %d (%s — %s): %s (%d rows)\n",
              i, row$dataset_source, example_ids[i], row$project_title, nrow(ex)))
}

# Save individual CSVs
for (i in seq_along(examples_list)) {
  if (!is.null(examples_list[[i]])) {
    ex_path <- here(tables_dir, sprintf("04_gencap_example%d.csv", i))
    write_csv(examples_list[[i]], ex_path)
    cat(sprintf("  Saved: %s\n", ex_path))
  }
}

# Save combined CSV
examples_all <- bind_rows(examples_list, .id = "example")
examples_csv_path <- here(tables_dir, "04_gencap_client_examples.csv")
write_csv(examples_all, examples_csv_path)
cat("Saved combined examples:", examples_csv_path, "\n")
