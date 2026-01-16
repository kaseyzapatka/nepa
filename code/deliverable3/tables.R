# --------------------------
# DELIVERABLE 3: CE vs EA vs EIS
# --------------------------
# Creates tables for:
# 1. Project Status by Energy Type (Clean/Fossil/Other)
# 2. Generation Capacity by Process Type (if gencap data available)
# 3. Change Over Time (placeholder - requires timeline extraction)

library(arrow)
library(dplyr)
library(tidyr)
library(jsonlite)

# --------------------------
# SETUP
# --------------------------

# File paths (relative to this script location)
script_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
base_dir <- file.path(script_dir, "../..")

# Or use this if not running in RStudio:
# base_dir <- normalizePath(file.path(getwd(), "../.."))

data_path <- file.path(base_dir, "data/analysis/projects_combined.parquet")
gencap_path <- file.path(base_dir, "data/analysis/projects_gencap.parquet")
output_dir <- file.path(base_dir, "output/deliverable3")

# Create output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)


# --------------------------
# LOAD DATA
# --------------------------

cat("Loading data from:", data_path, "\n")
projects <- read_parquet(data_path)
projects |> glimpse()
cat("Total projects loaded:", nrow(projects), "\n\n")


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

# Add column totals
totals_row <- table1 %>%
  summarise(
    project_energy_type = "Total",
    EA = sum(EA, na.rm = TRUE),
    EIS = sum(EIS, na.rm = TRUE),
    CE = sum(CE, na.rm = TRUE),
    Total = sum(Total, na.rm = TRUE)
  )

table1 <- bind_rows(table1, totals_row)

# Rename for clarity
table1 <- table1 %>%
  rename(
    `Energy Type` = project_energy_type,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )
table1 |> print()
# Save
output_file1 <- file.path(output_dir, "table1_by_energy_type.csv")
write.csv(table1, output_file1, row.names = FALSE)
cat("  Saved:", output_file1, "\n")


# --------------------------
# TABLE 2: GENERATION CAPACITY (if available)
# --------------------------

cat("\nCreating Table 2: Generation Capacity...\n")

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

    output_file2 <- file.path(output_dir, "table2_by_generation_capacity.csv")
    write.csv(table2, output_file2, row.names = FALSE)
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

  output_file2 <- file.path(output_dir, "table2_by_generation_capacity_placeholder.csv")
  write.csv(table2, output_file2, row.names = FALSE)
  cat("  Saved placeholder:", output_file2, "\n")
}


# --------------------------
# TABLE 3: CHANGE OVER TIME (placeholder)
# --------------------------

cat("\nCreating Table 3: Change Over Time...\n")
cat("  Note: Timeline extraction not yet implemented.\n")
cat("  This requires parsing dates from document text.\n")

# Create placeholder table
table3 <- data.frame(
  Year = c("2025", "2024", "2023", "2022", "2021", "Total"),
  `Environmental Assessment` = rep("TBD", 6),
  `Environmental Impact Statement` = rep("TBD", 6),
  `Categorical Exclusion` = rep("TBD", 6),
  check.names = FALSE
)

output_file3 <- file.path(output_dir, "table3_by_year_placeholder.csv")
write.csv(table3, output_file3, row.names = FALSE)
cat("  Saved placeholder:", output_file3, "\n")


# --------------------------
# DETAILED BREAKDOWN: CLEAN ENERGY BY TECHNOLOGY x PROCESS TYPE
# --------------------------

cat("\nCreating supplementary table: Clean Energy Detail...\n")

# Helper function to parse JSON-encoded columns and explode to multiple rows
explode_column <- function(df, col_name) {
  df %>%
    mutate(!!col_name := sapply(.data[[col_name]], function(x) {
      if (is.null(x) || length(x) == 0 || (is.character(x) && x == "")) {
        return(NA_character_)
      }
      # Handle JSON-encoded arrays
      if (is.character(x) && grepl("^\\[", x)) {
        parsed <- tryCatch(
          jsonlite::fromJSON(x),
          error = function(e) x
        )
        if (is.character(parsed) && length(parsed) > 1) {
          return(paste(parsed, collapse = "|"))
        }
        return(as.character(parsed))
      }
      if (is.list(x)) return(paste(unlist(x), collapse = "|"))
      return(as.character(x))
    })) %>%
    separate_rows(!!col_name, sep = "\\|")
}

clean_energy <- projects %>%
  filter(project_energy_type == "Clean")

clean_by_tech <- clean_energy %>%
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

# Add totals
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

output_file4 <- file.path(output_dir, "clean_energy_by_technology_detail.csv")
write.csv(clean_by_tech, output_file4, row.names = FALSE)
cat("  Saved:", output_file4, "\n")


# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Deliverable 3 Complete ===\n")
cat("Output files saved to:", output_dir, "\n")
cat("  - table1_by_energy_type.csv\n")
cat("  - table2_by_generation_capacity*.csv\n")
cat("  - table3_by_year_placeholder.csv\n")
cat("  - clean_energy_by_technology_detail.csv\n")


# --------------------------
# ENERGY TYPE COUNTS SUMMARY
# --------------------------

cat("\n=== Energy Type Summary ===\n")
summary_stats <- projects %>%
  group_by(project_energy_type) %>%
  summarise(
    count = n(),
    pct = n() / nrow(projects) * 100
  )

print(summary_stats)

cat("\nProjects flagged for review:", sum(projects$project_energy_type_questions, na.rm = TRUE), "\n")
