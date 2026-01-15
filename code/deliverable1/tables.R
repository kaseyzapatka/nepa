# --------------------------
# DELIVERABLE 1: CLEAN ENERGY PROJECTS
# --------------------------
# Creates three tables:
# 1. Clean Energy by Technology (project_type)
# 2. Clean Energy by Lead Agency
# 3. Clean Energy by Location (state)

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
output_dir <- file.path(base_dir, "output/deliverable1")

# Create output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)


# --------------------------
# LOAD DATA
# --------------------------

cat("Loading data from:", data_path, "\n")
projects <- read_parquet(data_path)

projects |> head()

cat("Total projects loaded:", nrow(projects), "\n")
cat("Clean energy projects:", sum(projects$project_energy_type == "Clean"), "\n\n")

# Filter to clean energy only
clean_energy <- projects %>%
  filter(project_energy_type == "Clean")

# test
projects |> 
  filter(project_title== "Install Oil Field Facilities and Equipment on Little 6 Tank Facility Sec 6D") |> 
  glimpse()

# --------------------------
# HELPER FUNCTIONS
# --------------------------

# Function to parse JSON-encoded columns and explode to multiple rows
# Handles: JSON arrays like '["value1", "value2"]', plain strings, lists, NULL
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

# Function to create cross-tabulation table
create_crosstab <- function(df, group_col, process_col = "process_type") {
  df %>%
    group_by(.data[[group_col]], .data[[process_col]]) %>%
    summarise(n = n(), .groups = "drop") %>%
    pivot_wider(
      names_from = all_of(process_col),
      values_from = n,
      values_fill = 0
    ) %>%
    mutate(Total = rowSums(select(., -1), na.rm = TRUE)) %>%
    arrange(desc(Total))
}


# --------------------------
# TABLE 1: BY TECHNOLOGY (project_type)
# --------------------------

cat("Creating Table 1: Clean Energy by Technology...\n")

# Explode project_type (can have multiple values per project)
tech_data <- clean_energy %>%
  explode_column("project_type") %>%
  filter(!is.na(project_type) & project_type != "")

# Create crosstab
table1 <- create_crosstab(tech_data, "project_type")
table1 |> print(n = 100)

# Add column totals
totals_row <- table1 %>%
  summarise(
    project_type = "Total",
    EA = sum(EA, na.rm = TRUE),
    EIS = sum(EIS, na.rm = TRUE),
    CE = sum(CE, na.rm = TRUE),
    Total = sum(Total, na.rm = TRUE)
  )

table1 <- bind_rows(table1, totals_row)
table1 |> print(n = 100)

# Rename for clarity
table1 <- table1 %>%
  rename(
    Technology = project_type,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )

# Save
output_file1 <- file.path(output_dir, "table1_by_technology.csv")
write.csv(table1, output_file1, row.names = FALSE)
cat("  Saved:", output_file1, "\n")


# --------------------------
# TABLE 2: BY LEAD AGENCY
# --------------------------

cat("Creating Table 2: Clean Energy by Lead Agency...\n")

# Explode lead_agency
agency_data <- clean_energy %>%
  explode_column("lead_agency") %>%
  filter(!is.na(lead_agency) & lead_agency != "")

# Create crosstab
table2 <- create_crosstab(agency_data, "lead_agency")
table2 |> print(n = 100)

# Add column totals
totals_row <- table2 %>%
  summarise(
    lead_agency = "Total",
    EA = sum(EA, na.rm = TRUE),
    EIS = sum(EIS, na.rm = TRUE),
    CE = sum(CE, na.rm = TRUE),
    Total = sum(Total, na.rm = TRUE)
  )

table2 <- bind_rows(table2, totals_row)


# Rename for clarity
table2 <- table2 %>%
  rename(
    Agency = lead_agency,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )

# Save
output_file2 <- file.path(output_dir, "table2_by_agency.csv")
write.csv(table2, output_file2, row.names = FALSE)
cat("  Saved:", output_file2, "\n")


# --------------------------
# TABLE 3: BY LOCATION (STATE)
# --------------------------

cat("Creating Table 3: Clean Energy by Location (State)...\n")

# Explode project_state
location_data <- clean_energy %>%
  explode_column("project_state") %>%
  filter(!is.na(project_state) & project_state != "")

# Create crosstab
table3 <- create_crosstab(location_data, "project_state")

# Add column totals
totals_row <- table3 %>%
  summarise(
    project_state = "Total",
    EA = sum(EA, na.rm = TRUE),
    EIS = sum(EIS, na.rm = TRUE),
    CE = sum(CE, na.rm = TRUE),
    Total = sum(Total, na.rm = TRUE)
  )

table3 <- bind_rows(table3, totals_row)

# Rename for clarity
table3 <- table3 %>%
  rename(
    State = project_state,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )
table3 |> print(n = 100)

# Save
output_file3 <- file.path(output_dir, "table3_by_state.csv")
write.csv(table3, output_file3, row.names = FALSE)
cat("  Saved:", output_file3, "\n")


# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Deliverable 1 Complete ===\n")
cat("Output files saved to:", output_dir, "\n")
cat("  - table1_by_technology.csv\n")
cat("  - table2_by_agency.csv\n")
cat("  - table3_by_state.csv\n")
cat("\nNote: Projects may appear in multiple rows if they have multiple\n")
cat("project types, agencies, or locations.\n")


# --------------------------
# FLAGGED PROJECTS REPORT
# --------------------------

cat("\n=== Flagged Projects for Review ===\n")
flagged <- clean_energy %>%
  filter(project_energy_type_questions == TRUE)

cat("Projects flagged for manual review:", nrow(flagged), "\n")

if (nrow(flagged) > 0) {
  flagged_output <- file.path(output_dir, "flagged_for_review.csv")
  flagged %>%
    select(project_id, project_title, project_type, project_energy_type_questions) %>%
    write.csv(flagged_output, row.names = FALSE)
  cat("  Saved flagged projects to:", flagged_output, "\n")
}

flagged |> glimpse()

projects |> glimpse()
