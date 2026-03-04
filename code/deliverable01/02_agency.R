# --------------------------
# DELIVERABLE 1: DECARBONIZATION TECHNOLOGY BY LEAD AGENCY
# --------------------------
# Table 2: Decarbonization Technology by Lead Agency
# Analysis of which agencies handle decarbonization technology projects

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable01", "00_setup.R"))

# --------------------------
# PROCESS
# --------------------------

# Note: project_department pre-computed in the Python extract pipeline.
# Only 40 of 61,881 projects (0.06%) have multiple lead agencies.
# We keep explode_column for lead_agency detail analysis, but use the
# pre-computed project_department for department-level grouping.

# Explode lead_agency (handles rare multi-agency cases)
agency_data <- clean_energy %>%
  explode_column("lead_agency") %>%
  filter(!is.na(lead_agency) & lead_agency != "")

# Use pre-computed department column (renamed for consistency with this script)
agency_data <- agency_data %>%
  mutate(department = project_department)

# Count projects per agency (detailed)
agency_counts <- agency_data %>%
  count(lead_agency, name = "n_projects") %>%
  arrange(desc(n_projects))

# Count projects per department (collapsed)
department_counts <- agency_data %>%
  count(department, name = "n_projects") %>%
  arrange(desc(n_projects))


# --------------------------
# TABLE: PROJECTS BY DEPARTMENT 
# --------------------------
# This table collapses lead agency into department for parsimony

table2 <- create_crosstab(agency_data, "department")

# Add totals row
table2 <- add_totals_row(table2, "department")

# Rename for clarity
table2 <- table2 %>%
  rename(
    Department = department,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )

table2 %>% print(n = 25)

# Save
write_csv(table2, here(tables_dir, "table2_by_department.csv"))



# --------------------------
# FIGURES
# --------------------------

#
# Deliverable: Department Bar Chart 
# ----------------------------------------
fig_departments <- department_counts %>%
  filter(department != "Other / Unclassified") %>%
  ggplot(aes(x = n_projects, y = reorder(department, n_projects))) +
  geom_col(fill = catf_dark_blue) +
  geom_text(aes(label = scales::comma(n_projects)), hjust = -0.1, size = 3) +
  labs(
    x = "Number of Projects Tagged with Decarbonization Technologies",
    y = NULL,
    title = "Projects Tagged with Decarbonization Technologies by Federal Department"
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 10)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15)), labels = scales::comma)

fig_departments

ggsave(
  filename = here(figures_dir, "02_departments.png"),
  plot = fig_departments,
  width = 10,
  height = 7,
  units = "in",
  dpi = 300
)


#
# Deliverable: Departments by review process 
# ----------------------------------------
dept_process <- agency_data %>%
  count(department, process_type) %>%
  group_by(department) %>%
  mutate(
    total = sum(n),
    percent = 100 * n / total
  ) %>%
  ungroup() %>%
  filter(department != "Other / Unclassified")

# Totals for label layer
dept_totals <- dept_process %>%
  distinct(department, total)

fig_dept_process <- dept_process %>%
  filter(total >= 5) |>
  ggplot(aes(x = reorder(department, total), y = percent, fill = process_type)) +
  geom_col() +
  geom_text(
    aes(label = ifelse(percent >= 3, scales::percent(percent / 100, accuracy = 1), "")),
    position = position_stack(vjust = 0.5),
    size = 3,
    color = "white"
  ) +
  geom_text(
    data = dept_totals %>% filter(total >= 5),
    aes(x = reorder(department, total), y = 101, label = scales::comma(total)),
    inherit.aes = FALSE,
    hjust = 0,
    size = 3,
    color = "gray30"
  ) +
  coord_flip() +
  labs(
    x = NULL,
    y = "Percent of Projects",
    fill = "Process Type",
    title = "Process Type Distribution by Federal Department",
    caption = "Note: Departments with fewer than 5 projects were removed for parsimony.\nPercentage labels below 3% are excluded for readability."
  ) +
  #scale_fill_brewer(palette = "Set1") +
  scale_fill_manual(
    values = c("CE" = catf_dark_blue, "EA" = catf_teal, "EIS" = catf_magenta)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 9))

fig_dept_process

ggsave(
  filename = here(figures_dir, "02_department_process.png"),
  plot = fig_dept_process,
  width = 10,
  height = 7,
  units = "in",
  dpi = 300
)


# --------------------------
# KEY AGENCIES ANALYSIS
# --------------------------
# Analysis of specific agencies requested for deliverable
#
# Note on data format:
# - lead_agency values are stored as JSON arrays (e.g., '["Department of Energy - Power Marketing Administration"]')
# - DOE Power Marketing Administrations (Bonneville, Southeastern, Southwestern, Western Area) are
#   combined under "Power Marketing Administration" in the source data
# - We use pattern matching to identify agencies within the JSON strings

cat("\n=== Key Agencies Analysis ===\n")

# Define key agencies with regex patterns to match actual data format
# Data format: "Department of X - Agency Name"
key_agencies <- tribble(
  ~department, ~display_name, ~pattern,
  # DOE - Power Marketing Administrations are combined in source data
  "Department of Energy", "Power Marketing Administration", "Power Marketing Administration",
  # Interior agencies
  "Department of the Interior", "Bureau of Land Management", "Bureau of Land Management",
  "Department of the Interior", "Fish and Wildlife Service", "Fish and Wildlife",
  "Department of the Interior", "Bureau of Indian Affairs", "Bureau of Indian Affairs",
  "Department of the Interior", "Bureau of Ocean Energy Management", "Bureau of Ocean Energy Management",
  "Department of the Interior", "National Park Service", "National Park Service",
  # USDA agencies
  "Department of Agriculture", "Forest Service", "Forest Service",
  "Department of Agriculture", "Natural Resources Conservation Service", "Natural Resources Conservation",
  "Department of Agriculture", "Rural Development", "Rural Development",
  "Department of Agriculture", "Rural Utilities Service", "Rural Utilities Service"
)

# Create function to match agency based on patterns
match_key_agency <- function(agency_str) {
  if (is.na(agency_str) || agency_str == "") return(NA_character_)

  for (i in 1:nrow(key_agencies)) {
    if (str_detect(agency_str, regex(key_agencies$pattern[i], ignore_case = TRUE))) {
      return(key_agencies$display_name[i])
    }
  }
  return(NA_character_)
}

# Match key agencies in the data
key_agency_data <- agency_data %>%
  mutate(
    matched_agency = sapply(lead_agency, match_key_agency)
  ) %>%
  filter(!is.na(matched_agency)) %>%
  # Remove original department column (based on first lead_agency, not accurate for exploded data)
  select(-department) %>%
  # Add correct department from key_agencies lookup
  left_join(
    key_agencies %>% select(display_name, department),
    by = c("matched_agency" = "display_name")
  )

cat("Key agencies found in data:\n")
key_agency_data %>%
  distinct(matched_agency, department) %>%
  arrange(department, matched_agency) %>%
  print(n = 20)

cat("\nTotal projects in key agencies:", nrow(key_agency_data), "\n")

# --------------------------
# TABLE: KEY AGENCIES BY REVIEW PROCESS
# --------------------------

# Create crosstab for key agencies (using matched_agency)
key_agency_crosstab <- key_agency_data %>%
  count(department, matched_agency, process_type) %>%
  pivot_wider(
    names_from = process_type,
    values_from = n,
    values_fill = 0
  ) %>%
  # Ensure all columns exist
  mutate(
    EA = if ("EA" %in% names(.)) EA else 0L,
    EIS = if ("EIS" %in% names(.)) EIS else 0L,
    CE = if ("CE" %in% names(.)) CE else 0L
  ) %>%
  mutate(Total = EA + EIS + CE) %>%
  # Reorder columns
  select(department, matched_agency, EIS, EA, CE, Total) %>%
  arrange(department, desc(Total))

# Create display table with department grouping
table_key_agencies <- key_agency_crosstab %>%
  rename(
    Department = department,
    Agency = matched_agency,
    `Environmental Impact Statement` = EIS,
    `Environmental Assessment` = EA,
    `Categorical Exclusion` = CE
  )

# Add subtotals by department
dept_subtotals <- key_agency_crosstab %>%
  group_by(department) %>%
  summarise(
    matched_agency = paste0("  Subtotal: ", first(department)),
    EIS = sum(EIS),
    EA = sum(EA),
    CE = sum(CE),
    Total = sum(Total),
    .groups = "drop"
  ) %>%
  rename(
    Department = department,
    Agency = matched_agency,
    `Environmental Impact Statement` = EIS,
    `Environmental Assessment` = EA,
    `Categorical Exclusion` = CE
  )

# Add grand total
grand_total <- tibble(
  Department = "TOTAL",
  Agency = "All Key Agencies",
  `Environmental Impact Statement` = sum(key_agency_crosstab$EIS),
  `Environmental Assessment` = sum(key_agency_crosstab$EA),
  `Categorical Exclusion` = sum(key_agency_crosstab$CE),
  Total = sum(key_agency_crosstab$Total)
)

cat("\nKey Agencies by Review Process:\n")
table_key_agencies %>% print(n = 20)

cat("\nDepartment Subtotals:\n")
dept_subtotals %>% print()

cat("\nGrand Total:\n")
grand_total %>% print()

# Save table
write_csv(table_key_agencies, here(tables_dir, "table_key_agencies.csv"))
write_csv(dept_subtotals, here(tables_dir, "table_key_agencies_subtotals.csv"))
cat("  Saved: table_key_agencies.csv\n")

# --------------------------
# FIGURE: KEY AGENCIES BY REVIEW PROCESS (SHARE)
# --------------------------

# Calculate share by process type for each agency
key_agency_process <- key_agency_data %>%
  count(department, matched_agency, process_type) %>%
  group_by(matched_agency) %>%
  mutate(
    total = sum(n),
    share = n / total
  ) %>%
  ungroup()

# Order departments for display with condensed labels (line breaks to save width)
dept_order <- c("Department of Energy", "Department of the Interior", "Department of Agriculture")
dept_labels <- c(
  "Department of Energy" = "Department\nof Energy",
  "Department of the Interior" = "Department\nof the Interior",
  "Department of Agriculture" = "Department\nof Agriculture"
)

# Create ordered factor for agencies (within department, by total projects)
agency_order <- key_agency_process %>%
  group_by(department, matched_agency) %>%
  summarise(total = sum(n), .groups = "drop") %>%
  mutate(department = factor(department, levels = dept_order)) %>%
  arrange(department, total) %>%
  pull(matched_agency) |>
  glimpse()

key_agency_process <- key_agency_process %>%
  mutate(
    #matched_agency = factor(matched_agency, levels = agency_order),
    department = factor(department, levels = dept_order, labels = dept_labels[dept_order])
  ) |>
  glimpse()

# Create summary with totals for label layer
agency_totals <- key_agency_process %>%
  group_by(department, matched_agency) %>%
  summarise(total = sum(n), .groups = "drop")

# Create stacked bar chart with proportional panel heights by department
fig_key_agency_process <- key_agency_process %>%
  ggplot(aes(x = matched_agency, y = share, fill = process_type)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = ifelse(share >= 0.03, scales::percent(share, accuracy = 1), "")),
    position = position_stack(vjust = 0.5),
    size = 3,
    color = "white"
  ) +
  # Add total counts on the right side
  geom_text(
    data = agency_totals,
    aes(x = matched_agency, y = 1.02, label = scales::comma(total)),
    inherit.aes = FALSE,
    hjust = 0,
    size = 3,
    color = "gray30"
  ) +
  coord_flip(clip = "off") +
  # Use facet_grid with space = "free_y" for proportional panel heights
  facet_grid(department ~ ., scales = "free_y", space = "free_y", switch = "y") +
  labs(
    x = NULL,
    y = "Share of Projects",
    fill = "Process Type",
    title = "NEPA Process Type Distribution by Key Federal Agency",
    caption = "Note: DOE Power Marketing Administration includes Bonneville, Southeastern, Southwestern, and Western Area.\nNumbers on right show total project count per agency."
  ) +
  scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.08))) +
  scale_fill_manual(
    values = c("CE" = catf_dark_blue, "EA" = catf_teal, "EIS" = catf_magenta)
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 9),
    strip.text.y.left = element_text(size = 9, face = "bold", angle = 0, hjust = 1, lineheight = 1.1),
    strip.background = element_rect(fill = "grey95", color = NA),
    strip.placement = "outside",
    panel.spacing = unit(0.8, "lines"),
    plot.margin = margin(10, 30, 10, 10)  # Extra right margin for counts
  )

fig_key_agency_process

ggsave(
  filename = here(figures_dir, "03_key_agency_process.png"),
  plot = fig_key_agency_process,
  width = 12,
  height = 8,
  units = "in",
  dpi = 300
)
cat("  Saved: 03_key_agency_process.png\n")


# --------------------------
# FIGURE: COVERAGE-VERIFIED AGENCIES ONLY (DOE, BLM, FOREST SERVICE)
# --------------------------
# These three agencies have comprehensive EA/CE data in NEPATEC.
# All other agencies appear primarily via the EPA EIS database (EIS only).

coverage_verified <- bind_rows(
  # DOE: use full department
  agency_data %>%
    filter(department == "Department of Energy") %>%
    mutate(agency_label = "Dept. of Energy (DOE)", dept_label = "Department of Energy"),
  # BLM and Forest Service from key agency matches
  key_agency_data %>%
    filter(matched_agency %in% c("Bureau of Land Management", "Forest Service")) %>%
    mutate(
      agency_label = case_when(
        matched_agency == "Bureau of Land Management" ~ "Bureau of Land Management (BLM)",
        matched_agency == "Forest Service" ~ "Forest Service (USFS)"
      ),
      dept_label = case_when(
        matched_agency == "Bureau of Land Management" ~ "Department of the Interior",
        matched_agency == "Forest Service" ~ "Department of Agriculture"
      )
    )
) %>%
  count(dept_label, agency_label, process_type) %>%
  group_by(agency_label) %>%
  mutate(
    total = sum(n),
    share = n / total
  ) %>%
  ungroup()

coverage_totals <- coverage_verified %>%
  distinct(dept_label, agency_label, total)

dept_order_cv <- c("Department of Energy", "Department of the Interior", "Department of Agriculture")
dept_labels_cv <- c(
  "Department of Energy" = "Department\nof Energy",
  "Department of the Interior" = "Department\nof the Interior",
  "Department of Agriculture" = "Department\nof Agriculture"
)

coverage_verified <- coverage_verified %>%
  mutate(dept_label = factor(dept_label, levels = dept_order_cv, labels = dept_labels_cv[dept_order_cv]))

coverage_totals <- coverage_totals %>%
  mutate(dept_label = factor(dept_label, levels = dept_order_cv, labels = dept_labels_cv[dept_order_cv]))

fig_coverage_verified <- coverage_verified %>%
  ggplot(aes(x = agency_label, y = share, fill = process_type)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = ifelse(share >= 0.03, scales::percent(share, accuracy = 1), "")),
    position = position_stack(vjust = 0.5),
    size = 3,
    color = "white"
  ) +
  geom_text(
    data = coverage_totals,
    aes(x = agency_label, y = 1.02, label = scales::comma(total)),
    inherit.aes = FALSE,
    hjust = 0,
    size = 3,
    color = "gray30"
  ) +
  coord_flip(clip = "off") +
  facet_grid(dept_label ~ ., scales = "free_y", space = "free_y", switch = "y") +
  labs(
    x = NULL,
    y = "Share of Projects",
    fill = "Process Type",
    title = "NEPA Process Type Distribution (Coverage-Verified Agencies)",
    caption = "Only agencies with comprehensive EA/CE data in NEPATEC: DOE (all sub-agencies), BLM, and Forest Service.\nAll other agencies appear primarily via EIS records only. Numbers on right show total project count."
  ) +
  scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.08))) +
  scale_fill_manual(
    values = c("CE" = catf_dark_blue, "EA" = catf_teal, "EIS" = catf_magenta)
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 9),
    strip.text.y.left = element_text(size = 9, face = "bold", angle = 0, hjust = 1, lineheight = 1.1),
    strip.background = element_rect(fill = "grey95", color = NA),
    strip.placement = "outside",
    panel.spacing = unit(0.8, "lines"),
    plot.margin = margin(10, 30, 10, 10)
  )

fig_coverage_verified

ggsave(
  filename = here(figures_dir, "04_coverage_verified_process.png"),
  plot = fig_coverage_verified,
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)
cat("  Saved: 04_coverage_verified_process.png\n")
