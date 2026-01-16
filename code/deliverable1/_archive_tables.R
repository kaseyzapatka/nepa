# --------------------------
# DELIVERABLE 1: CLEAN ENERGY PROJECTS
# --------------------------
# Creates three tables:
# 1. Clean Energy by Technology (project_type)
# 2. Clean Energy by Lead Agency
# 3. Clean Energy by Location (state)

library(arrow)
library(tidyverse)
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
projects |> glimpse()

cat("Total projects loaded:", nrow(projects), "\n")
cat("Clean energy projects:", sum(projects$project_energy_type == "Clean"), "\n\n")

# Filter to clean energy only
clean_energy <- projects %>%
  filter(project_energy_type == "Clean") |> 
  glimpse()

# test - should not be in clean energy database
projects |> 
  filter(project_title== "Install Oil Field Facilities and Equipment on Little 6 Tank Facility Sec 6D") |> 
  glimpse()

clean_energy |> glimpse()


#clean_energy |> 
#  select(project_id, project_title, project_type) |> 
#  slice_head(n = 100) |> 
#  View()


# project type by clean
# --------------------------------

type_by_clean <-
  clean_energy %>%
    count(project_type_count, project_type_count_clean) %>%
    group_by(project_type_count) %>%
    mutate(percent = n / sum(n)) %>%
    ungroup() %>%
    ggplot(
      aes(
        x = factor(project_type_count),
        y = percent,
        fill = factor(project_type_count_clean)
      )
    ) +
    geom_col() +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(
      x = "Total Number of Project Types",
      y = "Percent of Projects",
      fill = "Number of Clean Energy Types",
      title = "Total number of project types by clean energy"
    ) +
    theme_minimal()


# save
ggsave(
  filename = "output/deliverable1/figures/type_by_clean.png",
  plot = type_by_clean,
  width = 8,
  height = 5,
  units = "in",
  dpi = 300
)



# --------------------------
# OPTION 1: BINARY HAS CLEAN ENERGY TAG
# --------------------------

clean_energy_tags <- c(
  "Carbon Capture and Sequestration",
  "Conventional Energy Production - Nuclear",
  "Conventional Energy Production - Other",
  "Renewable Energy Production - Biomass",
  "Renewable Energy Production - Energy Storage",
  "Renewable Energy Production - Geothermal",
  "Renewable Energy Production - Hydrokinetic",
  "Renewable Energy Production - Hydropower",
  "Renewable Energy Production - Other",
  "Renewable Energy Production - Solar",
  "Renewable Energy Production - Wind, Offshore",
  "Renewable Energy Production - Wind, Onshore",
  "Nuclear Technology",
  "Electricity Transmission",
  "Utilities (electricity, gas, telecommunications)"
)

# relaxed 
# --------------------------------


clean_energy_summary <-
  clean_energy %>% 
  # parse JSON safely per row
  mutate(
    project_type = map(project_type, fromJSON)
  ) %>%
  # explode list-column
  unnest(project_type) %>%
  # keep only clean-energy tags
  filter(project_type %in% clean_energy_tags) %>%
  # ensure projects counted once per tag
  distinct(project_title, project_type) %>%
  count(project_type) %>%
  mutate(
    percent_projects = 100 * n / n_distinct(clean_energy$project_title)
  ) %>%
  arrange(desc(percent_projects)) |> 
  glimpse()


clean_energy_summary_strict <-
  clean_energy %>% 
  filter(project_energy_type_strict == "Clean") |> 
  # parse JSON safely per row
  mutate(
    project_type = map(project_type, fromJSON)
  ) %>%
  # explode list-column
  unnest(project_type) %>%
  # keep only clean-energy tags
  filter(project_type %in% clean_energy_tags) %>%
  # ensure projects counted once per tag
  distinct(project_title, project_type) %>%
  count(project_type) %>%
  mutate(
    percent_projects = 100 * n / n_distinct(clean_energy$project_title)
  ) %>%
  arrange(desc(percent_projects)) |> 
  glimpse()

clean_energy_bar_chart <-
  clean_energy_summary |> 
  ggplot(aes(x = percent_projects,
            y = reorder(project_type, percent_projects))) +
    geom_col() +
    labs(
      x = "Percent of projects",
      y = NULL,
      title = "Projects with Clean Energy Tags"
    ) +
    theme_minimal()

clean_energy_bar_chart



clean_energy_bar_chart_strict <-
  clean_energy_summary_strict |> 
  ggplot(aes(x = percent_projects,
            y = reorder(project_type, percent_projects))) +
    geom_col() +
    labs(
      x = "Percent of projects",
      y = NULL,
      title = "Projects with Clean Energy Tags"
    ) +
    theme_minimal()

clean_energy_bar_chart_strict

# save
ggsave(
  filename = "output/deliverable1/figures/01_clean_energy_bar_chart.png",
  plot = clean_energy_bar_chart,
  width = 8,
  height = 5,
  units = "in",
  dpi = 300
)

ggsave(
  filename = "output/deliverable1/figures/01_clean_energy_bar_chart_strict.png",
  plot = clean_energy_bar_chart_strict,
  width = 8,
  height = 5,
  units = "in",
  dpi = 300
)


# --------------------------
# OPTION 2: COUNT OF CLEAN ENERGY PROJECTS
# --------------------------

project_tag_counts <- clean_energy %>%
  # parse JSON strings into lists
  mutate(project_type_list = map(project_type, fromJSON)) %>%
  # count how many clean-energy tags are in each project
  mutate(num_clean_tags = map_int(project_type_list, ~ sum(.x %in% clean_energy_tags))) %>%
  # count total number of tags per project (all tags, including non-clean energy)
  mutate(num_total_tags = map_int(project_type_list, ~ length(.x))) %>%
  mutate(num_other_tags =  num_total_tags - num_clean_tags) |> 
  # categorize for visualization: 0, 1, 2, 3+
  mutate(tag_category = case_when(
    num_clean_tags == 0 ~ "0",
    num_clean_tags == 1 ~ "1",
    num_clean_tags == 2 ~ "2",
    num_clean_tags == 3 ~ "3",
    num_clean_tags == 4 ~ "4",
    num_clean_tags >= 5 ~ "5+"
  )) |> 
  glimpse()

# difference
project_tag_counts |> glimpse() # 26,621
project_tag_counts |> dim() # 26,621
project_tag_counts |> filter(project_energy_type_strict == "Clean") |> dim() #21,232

# 26621-21232 = 5389

# utilities and broadband
project_tag_counts |> filter(project_is_utilities_broadband_only == "TRUE") |> dim() # 408
# nuclear tech 
project_tag_counts |> filter(project_is_nuclear_tech_only == "TRUE") |> dim() # 4,981



# CLEAN ENERGY BY OTHER 
# --------------------------------------

# summarize for plotting
summary_tags <- project_tag_counts %>%
  count(tag_category) %>%
  mutate(percent = 100 * n / sum(n)) |> 
  glimpse()

clean_energy_by_tags <- 
  summary_tags |> 
  ggplot(aes(x = tag_category, y = percent, fill = tag_category)) +
    geom_col(show.legend = FALSE) +
    geom_text(aes(label = paste0(round(percent,1), "%")), vjust = -0.5) +
    labs(
      x = "Number of Clean-Energy Tags per Project",
      y = "Percent of Projects",
      title = "Single vs Multi-Tag Composition of Projects"
    ) +
    theme_minimal()

# save
ggsave(
  filename = "output/deliverable1/figures/01_clean_energy_by_tags.png",
  plot = clean_energy_by_tags,
  width = 8,
  height = 5,
  units = "in",
  dpi = 300
)



# CLEAN ENERGY BY OTHER 
# --------------------------------------

# add num_other_tags
project_tag_counts <- project_tag_counts %>%
  mutate(num_other_tags = num_total_tags - num_clean_tags,
         # categorize other tags for simplicity
         other_category = case_when(
           num_other_tags == 0 ~ "0",
           num_other_tags == 1 ~ "1",
           num_other_tags >= 2 ~ "2+"
         ))

# summarize by clean tag count and other tag category
summary_tags <- project_tag_counts %>%
  count(tag_category, other_category) %>%
  group_by(tag_category) %>%
  mutate(percent_within_clean = 100 * n / sum(n)) %>%
  ungroup() 

# stacked bar plot
stacked_bar <-
ggplot(summary_tags, aes(x = tag_category, y = percent_within_clean, fill = other_category)) +
  geom_col() +
  geom_text(aes(label = paste0(round(percent_within_clean,1), "%")), 
            position = position_stack(vjust = 0.5), color = "white") +
  labs(
    x = "Number of Clean-Energy Tags per Project",
    y = "Percent of Projects",
    fill = "Number of Other Tags",
    title = "Composition of Clean-Energy Tags by Other Project Tags"
  ) +
  theme_minimal()

stacked_bar


# save
ggsave(
  filename = "output/deliverable1/figures/01_stacked_bar.png",
  plot = stacked_bar,
  width = 8,
  height = 5,
  units = "in",
  dpi = 300
)


#  UTILITIES
# --------------------------------------

project_tag_counts |> glimpse()

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

table1 |> pull(project_type) |> print()

# Add column totals
totals_row <- table1 %>%
  summarise(
    project_type = "Total",
    EA = sum(EA, na.rm = TRUE),
    EIS = sum(EIS, na.rm = TRUE),
    CE = sum(CE, na.rm = TRUE),
    Total = sum(Total, na.rm = TRUE)
  ) |> 
  glimpse()

table1 <- bind_rows(table1, totals_row)
table1 |> print(n = 100)

# Rename for clarity
table1_totals <- table1 %>%
  rename(
    Technology = project_type,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  ) |> 
  print(n = 100)

# Save
output_file1 <- file.path(output_dir, "table1_by_technology.csv")
write.csv(table1_totals, output_file1, row.names = FALSE)
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
# TABLE 4: CO-OCCURRENCE SUMMARY BY CLEAN ENERGY TYPE
# --------------------------
# For each clean energy category, shows the top 3 most common co-occurring
# project types (with counts and percentages). Helps identify which "clean
# energy" projects are truly energy-focused vs. mixed-use.

cat("Creating Table 4: Co-occurrence Summary by Clean Energy Type...\n")

# Parse project_type JSON and create long-form data with all tags per project
projects_with_tags <- clean_energy %>%
  mutate(project_type_list = map(project_type, fromJSON)) %>%
  select(project_id, project_title, project_type_list)

# For each clean energy tag, find co-occurring tags
cooccurrence_summary <- map_dfr(clean_energy_tags, function(ce_tag) {

  # Find projects that have this clean energy tag
  projects_with_ce_tag <- projects_with_tags %>%
    filter(map_lgl(project_type_list, ~ ce_tag %in% .x))

  n_projects <- nrow(projects_with_ce_tag)

  if (n_projects == 0) {
    return(tibble(
      clean_energy_category = ce_tag,
      total_projects = 0,
      cooccur_rank = 1:3,
      cooccur_category = NA_character_,
      cooccur_count = NA_integer_,
      cooccur_percent = NA_real_
    ))
  }

  # Get all OTHER tags that appear with this clean energy tag
  cooccur_counts <- projects_with_ce_tag %>%
    unnest(project_type_list) %>%
    rename(other_tag = project_type_list) %>%
    # Exclude the clean energy tag itself
    filter(other_tag != ce_tag) %>%
    count(other_tag, name = "cooccur_count") %>%
    mutate(cooccur_percent = round(100 * cooccur_count / n_projects, 1)) %>%
    arrange(desc(cooccur_count)) %>%
    slice_head(n = 3) %>%
    mutate(cooccur_rank = row_number())

  # Pad to 3 rows if fewer than 3 co-occurring tags
  if (nrow(cooccur_counts) < 3) {
    cooccur_counts <- cooccur_counts %>%
      bind_rows(tibble(
        other_tag = rep(NA_character_, 3 - nrow(cooccur_counts)),
        cooccur_count = rep(NA_integer_, 3 - nrow(cooccur_counts)),
        cooccur_percent = rep(NA_real_, 3 - nrow(cooccur_counts)),
        cooccur_rank = (nrow(cooccur_counts) + 1):3
      ))
  }

  cooccur_counts %>%
    mutate(
      clean_energy_category = ce_tag,
      total_projects = n_projects
    ) %>%
    rename(cooccur_category = other_tag) %>%
    select(clean_energy_category, total_projects, cooccur_rank,
           cooccur_category, cooccur_count, cooccur_percent)
})

# Pivot to wide format for cleaner presentation
table4 <- cooccurrence_summary %>%
  pivot_wider(
    id_cols = c(clean_energy_category, total_projects),
    names_from = cooccur_rank,
    values_from = c(cooccur_category, cooccur_count, cooccur_percent),
    names_glue = "{.value}_{cooccur_rank}"
  ) %>%
  arrange(desc(total_projects)) %>%
  select(
    clean_energy_category, total_projects,
    cooccur_category_1, cooccur_count_1, cooccur_percent_1,
    cooccur_category_2, cooccur_count_2, cooccur_percent_2,
    cooccur_category_3, cooccur_count_3, cooccur_percent_3
  )

# Rename for clarity
table4 <- table4 %>%
  rename(
    `Clean Energy Category` = clean_energy_category,
    `Total Projects` = total_projects,
    `Top Co-occurring Category` = cooccur_category_1,
    `Count (1)` = cooccur_count_1,
    `% (1)` = cooccur_percent_1,
    `2nd Co-occurring Category` = cooccur_category_2,
    `Count (2)` = cooccur_count_2,
    `% (2)` = cooccur_percent_2,
    `3rd Co-occurring Category` = cooccur_category_3,
    `Count (3)` = cooccur_count_3,
    `% (3)` = cooccur_percent_3
  )

table4 |> print(n = 100)

# Save
output_file4 <- file.path(output_dir, "table4_cooccurrence_summary.csv")
write.csv(table4, output_file4, row.names = FALSE)
cat("  Saved:", output_file4, "\n")


# --------------------------
# TABLE 5: EXHAUSTIVE CO-OCCURRENCE BY CLEAN ENERGY TYPE (AGGREGATED)
# --------------------------
# For each clean energy category, lists ALL co-occurring project types
# with counts. Aggregated table for summary analysis.

cat("Creating Table 5: Exhaustive Co-occurrence by Clean Energy Type...\n")

# For each clean energy tag, get ALL co-occurring tags (not just top 3)
exhaustive_cooccurrence <- map_dfr(clean_energy_tags, function(ce_tag) {

  # Find projects that have this clean energy tag
  projects_with_ce_tag <- projects_with_tags %>%
    filter(map_lgl(project_type_list, ~ ce_tag %in% .x))

  n_projects <- nrow(projects_with_ce_tag)

  if (n_projects == 0) {
    return(tibble(
      clean_energy_category = ce_tag,
      total_projects_with_category = 0,
      cooccurring_type = NA_character_,
      cooccur_count = NA_integer_,
      cooccur_percent = NA_real_
    ))
  }

  # Get ALL other tags that appear with this clean energy tag
  projects_with_ce_tag %>%
    unnest(project_type_list) %>%
    rename(cooccurring_type = project_type_list) %>%
    # Exclude the clean energy tag itself
    filter(cooccurring_type != ce_tag) %>%
    count(cooccurring_type, name = "cooccur_count") %>%
    mutate(
      clean_energy_category = ce_tag,
      total_projects_with_category = n_projects,
      cooccur_percent = round(100 * cooccur_count / n_projects, 1)
    ) %>%
    arrange(desc(cooccur_count)) %>%
    select(clean_energy_category, total_projects_with_category,
           cooccurring_type, cooccur_count, cooccur_percent)
})

# Rename for clarity
table5 <- exhaustive_cooccurrence %>%
  # filter below 5 % for clarity
  filter(cooccur_percent > 5) |> 
  rename(
    `Clean Energy Category` = clean_energy_category,
    `Total Projects with Category` = total_projects_with_category,
    `Co-occurring Project Type` = cooccurring_type,
    `Co-occurrence Count` = cooccur_count,
    `Co-occurrence %` = cooccur_percent
  )

table5 |> print(n = 50)

# Save
output_file5 <- file.path(output_dir, "table5_cooccurrence_exhaustive.csv")
write.csv(table5, output_file5, row.names = FALSE)
cat("  Saved:", output_file5, "\n")


# --------------------------
# TABLE 6: PROJECT-LEVEL CO-OCCURRENCE DETAIL
# --------------------------
# For each clean energy category, lists individual projects with their
# co-occurring project types. Includes project_title for inspection.

cat("Creating Table 6: Project-level Co-occurrence Detail...\n")

# For each clean energy tag, get project-level detail
project_cooccurrence_detail <- map_dfr(clean_energy_tags, function(ce_tag) {

  # Find projects that have this clean energy tag
  projects_with_ce_tag <- projects_with_tags %>%
    filter(map_lgl(project_type_list, ~ ce_tag %in% .x))

  if (nrow(projects_with_ce_tag) == 0) {
    return(tibble(
      clean_energy_category = ce_tag,
      project_id = NA_character_,
      project_title = NA_character_,
      cooccurring_type = NA_character_
    ))
  }

  # Get project-level co-occurrences (one row per project-cooccurrence pair)
  projects_with_ce_tag %>%
    unnest(project_type_list) %>%
    rename(cooccurring_type = project_type_list) %>%
    # Exclude the clean energy tag itself
    filter(cooccurring_type != ce_tag) %>%
    mutate(clean_energy_category = ce_tag) %>%
    select(clean_energy_category, project_id, project_title, cooccurring_type)
})

# Rename for clarity
table6 <- project_cooccurrence_detail %>%
  rename(
    `Clean Energy Category` = clean_energy_category,
    `Project ID` = project_id,
    `Project Title` = project_title,
    `Co-occurring Project Type` = cooccurring_type
  )

table6 |> print(n = 50)

# Save
output_file6 <- file.path(output_dir, "table6_cooccurrence_projects.csv")
write.csv(table6, output_file6, row.names = FALSE)
cat("  Saved:", output_file6, "\n")


# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Deliverable 1 Complete ===\n")
cat("Output files saved to:", output_dir, "\n")
cat("  - table1_by_technology.csv\n")
cat("  - table2_by_agency.csv\n")
cat("  - table3_by_state.csv\n")
cat("  - table4_cooccurrence_summary.csv\n")
cat("  - table5_cooccurrence_exhaustive.csv\n")
cat("  - table6_cooccurrence_projects.csv\n")
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


# --------------------------
# UTILITIES + BROADBAND ONLY ANALYSIS
# --------------------------
# Count projects that have ONLY Utilities + Broadband tags (likely telecom, not energy)

cat("\n=== Utilities + Broadband Only Analysis ===\n")

utilities_broadband_only <- clean_energy %>%
  mutate(project_type_list = map(project_type, fromJSON)) %>%
  mutate(
    has_utilities = map_lgl(project_type_list,
      ~ "Utilities (electricity, gas, telecommunications)" %in% .x),
    has_broadband = map_lgl(project_type_list,
      ~ "Broadband" %in% .x),
    tag_count = map_int(project_type_list, length)
  ) %>%
  # Filter to projects with ONLY these two tags
 filter(has_utilities & has_broadband & tag_count == 2)

cat("Projects with ONLY Utilities + Broadband tags:", nrow(utilities_broadband_only), "\n")
cat("  (These are likely telecom projects, not clean energy)\n")

# Show breakdown by process type
cat("\nBy process type:\n")
utilities_broadband_only %>%
  count(process_type) %>%
  print()

# Save these for reference
if (nrow(utilities_broadband_only) > 0) {
  ub_output <- file.path(output_dir, "utilities_broadband_only.csv")
  utilities_broadband_only %>%
    select(project_id, project_title, project_type, process_type) %>%
    write.csv(ub_output, row.names = FALSE)
  cat("\n  Saved to:", ub_output, "\n")
}


# --------------------------
# NUCLEAR TECHNOLOGY ONLY ANALYSIS
# --------------------------
# Count projects with Nuclear Technology but NOT Nuclear Production

cat("\n=== Nuclear Technology Only Analysis ===\n")

nuclear_tech_only <- clean_energy %>%
  mutate(project_type_list = map(project_type, fromJSON)) %>%
  mutate(
    has_nuclear_tech = map_lgl(project_type_list,
      ~ "Nuclear Technology" %in% .x),
    has_nuclear_production = map_lgl(project_type_list,
      ~ "Conventional Energy Production - Nuclear" %in% .x)
  ) %>%
  # Has Nuclear Technology but NOT Nuclear Production
  filter(has_nuclear_tech & !has_nuclear_production)

cat("Projects with Nuclear Technology but NOT Nuclear Production:", nrow(nuclear_tech_only), "\n")
cat("  (These are likely waste/R&D projects, not power generation)\n")

# Show breakdown by process type
cat("\nBy process type:\n")
nuclear_tech_only %>%
  count(process_type) %>%
  print()


# --------------------------
# STRICT VS BROAD COMPARISON
# --------------------------

cat("\n=== Strict vs Broad Clean Energy Counts ===\n")
cat("Broad clean energy count:", nrow(clean_energy), "\n")

strict_exclusions <- nrow(utilities_broadband_only) + nrow(nuclear_tech_only)
strict_count <- nrow(clean_energy) - strict_exclusions

cat("Strict exclusions:\n")
cat("  - Utilities + Broadband only:", nrow(utilities_broadband_only), "\n")
cat("  - Nuclear Technology only:", nrow(nuclear_tech_only), "\n")
cat("  - Total excluded:", strict_exclusions, "\n")
cat("Strict clean energy count:", strict_count, "\n")
cat("Reduction:", round(100 * strict_exclusions / nrow(clean_energy), 1), "%\n")


projects |> glimpse()
