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

gencap_candidates <- c(
  here("data", "analysis", "projects_gencap.parquet"),
  here("data", "analysis", "projects_gencap_merged.parquet")
)
gencap_path <- gencap_candidates[file.exists(gencap_candidates)][1]

# --------------------------
# TABLE 2: GENERATION CAPACITY
# --------------------------

cat("Creating Table 2: Generation Capacity...\n")

if (!is.na(gencap_path) && file.exists(gencap_path)) {
  gencap_projects <- read_parquet(gencap_path)
  if (!"project_gencap_final_value" %in% names(gencap_projects)) {
    gencap_projects$project_gencap_final_value <- gencap_projects$project_gencap_value
  }
  if (!"project_gencap_final_unit" %in% names(gencap_projects)) {
    gencap_projects$project_gencap_final_unit <- gencap_projects$project_gencap_unit
  }
  if (!"project_gencap_final_source" %in% names(gencap_projects)) {
    gencap_projects$project_gencap_final_source <- gencap_projects$project_gencap_source
  }
  if (!"project_gencap_final_confidence" %in% names(gencap_projects)) {
    gencap_projects$project_gencap_final_confidence <- gencap_projects$project_gencap_confidence
  }

  gencap_projects <- gencap_projects %>%
    mutate(
      capacity_value_use = coalesce(project_gencap_final_value, project_gencap_value),
      capacity_unit_use = coalesce(project_gencap_final_unit, project_gencap_unit),
      capacity_source_use = coalesce(project_gencap_final_source, project_gencap_source),
      capacity_confidence_use = coalesce(project_gencap_final_confidence, project_gencap_confidence)
    ) %>%
    mutate(
      has_capacity = !is.na(capacity_value_use) & !is.na(capacity_unit_use)
    )

  # --------------------------
  # COVERAGE TABLE (SIMPLE)
  # --------------------------

  capacity_coverage_table <- gencap_projects %>%
    group_by(dataset_source) %>%
    summarise(
      total_projects = n(),
      projects_with_capacity = sum(has_capacity, na.rm = TRUE),
      coverage_percent = 100 * projects_with_capacity / total_projects,
      .groups = "drop"
    ) %>%
    arrange(factor(dataset_source, levels = c("CE", "EA", "EIS")))

  capacity_coverage_total <- capacity_coverage_table %>%
    summarise(
      dataset_source = "Total",
      across(where(is.numeric), \(x) sum(x, na.rm = TRUE))
    ) %>%
    mutate(
      coverage_percent = 100 * projects_with_capacity / total_projects
    ) %>%
    select(names(capacity_coverage_table))

  capacity_coverage_table <- bind_rows(capacity_coverage_table, capacity_coverage_total) %>%
    rename(
      `Process Type` = dataset_source,
      `Total Projects` = total_projects,
      `Projects with Capacity` = projects_with_capacity,
      `Coverage (%)` = coverage_percent
    )
  write_csv(capacity_coverage_table, here(tables_dir, "table2_capacity_coverage_summary.csv"))
  cat("  Saved: table2_capacity_coverage_summary.csv\n")

  # --------------------------
  # POWER VS ENERGY COVERAGE TABLE
  # --------------------------

  power_energy_rows <- gencap_projects %>%
    group_by(dataset_source) %>%
    summarise(
      total_projects = n(),
      has_power = sum(!is.na(project_gencap_value), na.rm = TRUE),
      has_energy = sum(!is.na(project_gencap_energy_value), na.rm = TRUE),
      has_both = sum(!is.na(project_gencap_value) & !is.na(project_gencap_energy_value), na.rm = TRUE),
      pct_power = 100 * has_power / total_projects,
      pct_energy = 100 * has_energy / total_projects,
      .groups = "drop"
    ) %>%
    arrange(factor(dataset_source, levels = c("CE", "EA", "EIS")))

  power_energy_total <- power_energy_rows %>%
    summarise(
      dataset_source = "Total",
      total_projects = sum(total_projects),
      has_power = sum(has_power),
      has_energy = sum(has_energy),
      has_both = sum(has_both),
      pct_power = 100 * sum(has_power) / sum(total_projects),
      pct_energy = 100 * sum(has_energy) / sum(total_projects)
    )

  power_energy_table <- bind_rows(power_energy_rows, power_energy_total) %>%
    rename(
      `Process Type` = dataset_source,
      `Total Projects` = total_projects,
      `Power (n)` = has_power,
      `Energy (n)` = has_energy,
      `Both (n)` = has_both,
      `Power (%)` = pct_power,
      `Energy (%)` = pct_energy
    )

  write_csv(power_energy_table, here(tables_dir, "table2_power_energy_coverage.csv"))
  cat("  Saved: table2_power_energy_coverage.csv\n")

  # --------------------------
  # POWER VS ENERGY FIGURE
  # --------------------------

  power_energy_plot_data <- power_energy_rows %>%
    select(dataset_source, pct_power, pct_energy) %>%
    pivot_longer(
      cols = c(pct_power, pct_energy),
      names_to = "metric",
      values_to = "pct"
    ) %>%
    mutate(
      metric = recode(metric,
        "pct_power"  = "Power (MW/GW/kW)",
        "pct_energy" = "Energy (MWh/GWh/kWh)"
      ),
      dataset_source = factor(dataset_source, levels = c("CE", "EA", "EIS"))
    )

  fig_power_energy <- power_energy_plot_data %>%
    ggplot(aes(x = dataset_source, y = pct, fill = metric)) +
    geom_col(position = "dodge", width = 0.65) +
    geom_text(
      aes(label = paste0(round(pct, 1), "%")),
      position = position_dodge(width = 0.65),
      vjust = -0.4,
      size = 3.5,
      fontface = "bold"
    ) +
    labs(
      title = "Power and Energy Extraction Coverage by Process Type",
      subtitle = "Percent of projects with at least one power (MW) or energy (MWh) value extracted",
      x = "Process Type",
      y = "Projects with Extraction (%)",
      fill = "Metric",
      caption = paste0(
        "CE = Categorical Exclusion, EA = Environmental Assessment, EIS = Environmental Impact Statement\n",
        "Energy values include both storage capacity and annual output projections; interpret with caution."
      )
    ) +
    scale_y_continuous(
      limits = c(0, 100),
      labels = percent_format(scale = 1),
      expand = expansion(mult = c(0, 0.12))
    ) +
    scale_fill_manual(values = c(
      "Power (MW/GW/kW)"     = catf_blue,
      "Energy (MWh/GWh/kWh)" = catf_teal
    )) +
    theme_catf() +
    theme(
      legend.position = "right",
      plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
    )

  ggsave(
    filename = here(figures_dir, "08_power_energy_coverage.png"),
    plot = fig_power_energy,
    width = 8,
    height = 5,
    units = "in",
    dpi = 300
  )
  cat("  Saved: 08_power_energy_coverage.png\n")

  # --------------------------
  # EXTRACTION SOURCE TABLE
  # --------------------------

  source_levels <- c("title", "description", "document", "none")

  source_rows <- gencap_projects %>%
    mutate(
      gencap_source_clean = case_when(
        project_gencap_source %in% source_levels ~ project_gencap_source,
        TRUE ~ "none"
      ),
      gencap_source_clean = factor(gencap_source_clean, levels = source_levels)
    ) %>%
    group_by(dataset_source, gencap_source_clean) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(dataset_source) %>%
    mutate(pct = 100 * n / sum(n)) %>%
    ungroup()

  source_wide <- source_rows %>%
    pivot_wider(
      id_cols = gencap_source_clean,
      names_from = dataset_source,
      values_from = c(n, pct),
      values_fill = 0,
      names_glue = "{dataset_source}_{.value}"
    )

  # Compute totals column
  source_totals <- gencap_projects %>%
    mutate(
      gencap_source_clean = case_when(
        project_gencap_source %in% source_levels ~ project_gencap_source,
        TRUE ~ "none"
      ),
      gencap_source_clean = factor(gencap_source_clean, levels = source_levels)
    ) %>%
    count(gencap_source_clean) %>%
    mutate(Total_pct = 100 * n / sum(n)) %>%
    rename(Total_n = n)

  source_table <- source_wide %>%
    left_join(source_totals, by = "gencap_source_clean") %>%
    arrange(gencap_source_clean) %>%
    rename(`Extraction Source` = gencap_source_clean)

  write_csv(source_table, here(tables_dir, "table2_gencap_source_breakdown.csv"))
  cat("  Saved: table2_gencap_source_breakdown.csv\n")

  # --------------------------
  # EXTRACTION SOURCE FIGURE
  # --------------------------

  source_fill <- c(
    "document"    = catf_blue,
    "description" = catf_light_blue,
    "title"       = catf_lime
  )

  source_labels <- c(
    "document"    = "Document",
    "description" = "Description",
    "title"       = "Title"
  )

  fig_source <- source_rows %>%
    filter(gencap_source_clean != "none") %>%
    group_by(dataset_source) %>%
    mutate(pct = 100 * n / sum(n)) %>%
    ungroup() %>%
    mutate(
      dataset_source = factor(dataset_source, levels = c("CE", "EA", "EIS")),
      gencap_source_clean = factor(
        gencap_source_clean,
        levels = c("document", "description", "title")
      )
    ) %>%
    ggplot(aes(x = dataset_source, y = pct, fill = gencap_source_clean)) +
    geom_col(width = 0.65) +
    geom_text(
      aes(
        label = ifelse(pct >= 4, paste0(round(pct, 0), "%"), ""),
        color = ifelse(gencap_source_clean == "none", "gray30", "white")
      ),
      position = position_stack(vjust = 0.5),
      size = 3.5,
      fontface = "bold"
    ) +
    labs(
      title = "Generation Capacity Extraction Source by Process Type",
      subtitle = "Where the pipeline found the capacity value for each project (100% stacked)",
      x = "Process Type",
      y = "Percent of Projects",
      fill = "Source",
      caption = paste0(
        "Title = project title contained a MW value; Description = project description field;\n",
        "Document = document text pages; None = no capacity value found"
      )
    ) +
    scale_y_continuous(
      labels = percent_format(scale = 1),
      expand = expansion(mult = c(0, 0.02))
    ) +
    scale_fill_manual(values = source_fill, labels = source_labels) +
    scale_color_identity(guide = "none") +
    theme_catf() +
    theme(
      legend.position = "right",
      plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
    )

  ggsave(
    filename = here(figures_dir, "09_gencap_source.png"),
    plot = fig_source,
    width = 8,
    height = 5,
    units = "in",
    dpi = 300
  )
  cat("  Saved: 09_gencap_source.png\n")

  # Filter to projects with capacity data
  has_cap <- gencap_projects %>%
    filter(has_capacity)

  cat("  Projects with capacity data:", nrow(has_cap), "\n")

  if (nrow(has_cap) > 0) {

    # --------------------------
    # NORMALIZE TO MW
    # --------------------------

    gencap_projects <- gencap_projects %>%
      mutate(
        capacity_mw = case_when(
          capacity_unit_use == "GW" ~ capacity_value_use * 1000,
          capacity_unit_use == "kW" ~ capacity_value_use / 1000,
          TRUE ~ capacity_value_use
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
        with_capacity = sum(has_capacity, na.rm = TRUE),
        reasonable = sum(!is.na(capacity_mw) & capacity_mw > 0 & capacity_mw <= 5000, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        pct_extracted = 100 * with_capacity / total,
        pct_reasonable = 100 * reasonable / total,
        dataset_source = factor(dataset_source, levels = c("CE", "EA", "EIS")),
        label_color = case_when(
          dataset_source == "CE" ~ "black",
          TRUE ~ "white"
        )
      )

    process_fill <- c(
      "CE" = catf_light_blue,
      "EA" = catf_blue,
      "EIS" = catf_dark_blue
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
        aes(
          label = paste0("(", comma(with_capacity), " / ", comma(total), ")"),
          y = pct_extracted / 2,
          color = label_color
        ),
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
      scale_fill_manual(values = process_fill, guide = "none") +
      scale_color_identity(guide = "none") +
      theme_catf() +
      theme(
        plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
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

    fig2_fill <- c(
      "Small (<10 MW)" = catf_lime,
      "Medium (10-100 MW)" = catf_light_blue,
      "Large (100-500 MW)" = catf_blue,
      "Utility-scale (>500 MW)" = catf_navy
    )

    fig2 <- cap_category_data %>%
      ggplot(aes(x = dataset_source, y = pct, fill = capacity_category)) +
      geom_col(width = 0.7) +
      geom_text(
        aes(
          label = ifelse(pct >= 5, paste0(round(pct, 1), "%"), ""),
          color = ifelse(
            capacity_category %in% c("Small (<10 MW)", "Medium (10-100 MW)"),
            "black",
            "white"
          )
        ),
        position = position_stack(vjust = 0.5),
        size = 3.5,
        fontface = "bold"
      ) +
      labs(
        title = "Project Capacity Distribution by Process Type",
        subtitle = "Stacked percent of projects with extracted generation capacity (reasonable range: 0-5000 MW)",
        x = "Process Type",
        y = "Percent of Extracted Projects",
        fill = "Capacity Category",
        caption = "CE = Categorical Exclusion, EA = Environmental Assessment, EIS = Environmental Impact Statement\nCapacity normalized to MW. Projects with values >5000 MW excluded as likely extraction errors."
      ) +
      scale_y_continuous(labels = percent_format(scale = 1), expand = expansion(mult = c(0, 0.05))) +
      scale_fill_manual(values = fig2_fill, drop = FALSE) +
      scale_color_identity(guide = "none") +
      theme_catf() +
      theme(
        plot.caption = element_text(size = 8, color = "gray50", hjust = 0),
        legend.position = "right"
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
    # FIGURE 3: CAPACITY DISTRIBUTION (VIOLIN + BOXPLOT)
    # --------------------------

    cat("\nCreating Figure 3: Capacity Distribution...\n")

    distribution_data <- gencap_reasonable %>%
      mutate(
        dataset_source = factor(dataset_source, levels = c("CE", "EA", "EIS"))
      )

    fig3 <- distribution_data %>%
      ggplot(aes(x = dataset_source, y = capacity_mw, fill = dataset_source)) +
      geom_violin(alpha = 0.5, color = NA, trim = FALSE) +
      geom_boxplot(
        width = 0.16,
        alpha = 0.9,
        outlier.alpha = 0.15,
        outlier.size = 0.8,
        color = "gray20"
      ) +
      stat_summary(
        fun = median,
        geom = "point",
        shape = 21,
        size = 2.8,
        fill = "white",
        color = "black"
      ) +
      labs(
        title = "Distribution of Extracted Generation Capacity by Process Type",
        subtitle = "Violin shows density; boxplot shows median and interquartile range (MW, log scale)",
        x = "Process Type",
        y = "Generation Capacity (MW, log scale)"
      ) +
      scale_y_log10(
        breaks = c(1, 5, 10, 50, 100, 500, 1000, 5000),
        labels = label_number(big.mark = ",")
      ) +
      scale_fill_manual(values = process_fill, guide = "none") +
      theme_catf() +
      theme(
        plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
      )

    fig3

    ggsave(
      filename = here(figures_dir, "06_capacity_distribution_violin_box.png"),
      plot = fig3,
      width = 10,
      height = 6,
      units = "in",
      dpi = 300
    )
    cat("  Saved: 06_capacity_distribution_violin_box.png\n")

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

gencap_examples_path <- gencap_candidates[file.exists(gencap_candidates)][1]
if (is.na(gencap_examples_path) || !file.exists(gencap_examples_path)) {
  stop("Generation capacity parquet not found (expected projects_gencap_merged.parquet or projects_gencap.parquet).")
}
gencap_merged <- read_parquet(gencap_examples_path)

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

