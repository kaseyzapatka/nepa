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
# AGENCY-LEVEL ANALYSIS (harmonized from raw lead_agency strings)
# --------------------------
# We explode lead_agency (which retains "Department of X - Agency" format) so
# that each exploded row correctly derives its own department from its own
# agency string. Using project_department instead would mis-assign the primary
# project department to all secondary agencies on multi-agency projects.

agency_harmonized <- clean_energy %>%
  explode_column("lead_agency") %>%
  filter(!is.na(lead_agency) & lead_agency != "") %>%
  mutate(
    # Expand common abbreviations so the split below works uniformly
    lead_agency_exp = lead_agency %>%
      str_replace("^DOE\\s*-\\s*",  "Department of Energy - ") %>%
      str_replace("^DOI\\s*-\\s*",  "Department of the Interior - ") %>%
      str_replace("^USDA\\s*-\\s*", "Department of Agriculture - ") %>%
      str_replace("^DOD\\s*-\\s*",  "Department of Defense - ") %>%
      str_replace("^DOT\\s*-\\s*",  "Department of Transportation - "),
    # Department: extract from the agency string, not from project_department.
    # project_department reflects only the primary agency on multi-agency projects,
    # so a secondary "Department of Energy" entry on an Interior-primary project
    # would be mis-assigned. We use the lead_agency string itself where possible.
    department = case_when(
      # "Department of X - Agency": use the prefix
      str_detect(lead_agency_exp, " - ") ~
        str_extract(lead_agency_exp, "^.+?(?= - )") %>% str_trim(),
      # Standalone department name: use as-is
      str_detect(lead_agency_exp, "^Department of ")          ~ lead_agency_exp,
      str_detect(lead_agency_exp, "^Major Independent ")      ~ "Major Independent Agencies",
      str_detect(lead_agency_exp, "^Other Independent ")      ~ "Other Independent Agencies",
      str_detect(lead_agency_exp, "^General Services Admin")  ~ "General Services Administration",
      # Bare sub-agency or unknown abbreviation: fall back to project-level department
      TRUE ~ project_department
    ),
    # Sub-agency display name: everything after " - "; full string if no split
    lead_agency_harmonized = if_else(
      str_detect(lead_agency_exp, " - "),
      str_extract(lead_agency_exp, "(?<= - ).+$") %>% str_trim(),
      lead_agency_exp
    )
  ) %>%
  select(-lead_agency_exp)

cat("\nUnique harmonized agencies (DOE/Interior/USDA):\n")
agency_harmonized %>%
  filter(department %in% c("Department of Energy", "Department of the Interior", "Department of Agriculture")) %>%
  count(department, lead_agency_harmonized, name = "n_projects") %>%
  arrange(department, desc(n_projects)) %>%
  print(n = 50)


# --------------------------
# TABLE: AGENCIES BY REVIEW PROCESS WITH COVERAGE FLAG
# --------------------------
# For meeting: shows all agencies grouped by department with coverage annotation.
# Only DOE, BLM, and Forest Service have complete EA/CE data in NEPATEC.
# All other agencies are represented only via the EPA EIS database (EIS-only).

meeting_table <- agency_harmonized %>%
  filter(department != "Other / Unclassified") %>%
  mutate(
    lead_agency_harmonized = case_when(
      # Generic USDA records (no sub-agency label) are Forest Service CEs/EAs stored
      # without the "Forest Service" identifier. Roll them into Forest Service so CE
      # counts appear on the correct row instead of a separate generic row.
      lead_agency_harmonized == department & department == "Department of Agriculture" ~ "Forest Service",
      # All other generic records (sub-agency name == department name): label as generic
      lead_agency_harmonized == department ~ paste0(department, " (generic)"),
      TRUE ~ lead_agency_harmonized
    )
  ) %>%
  count(department, lead_agency_harmonized, process_type) %>%
  pivot_wider(names_from = process_type, values_from = n, values_fill = 0) %>%
  mutate(
    EA  = if ("EA"  %in% names(.)) EA  else 0L,
    EIS = if ("EIS" %in% names(.)) EIS else 0L,
    CE  = if ("CE"  %in% names(.)) CE  else 0L,
    Total = EIS + EA + CE,
    Coverage = if_else(EIS > 0 & EA > 0 & CE > 0, "Full", "Not complete")
  ) %>%
  select(department, lead_agency_harmonized, EIS, EA, CE, Total, Coverage) %>%
  arrange(department, desc(Total))

write_csv(meeting_table, here(tables_dir, "table_meeting_coverage.csv"))
cat("  Saved: table_meeting_coverage.csv\n")

meeting_table %>%
  rename(Department = department, Agency = lead_agency_harmonized) %>%
  print(n = 50)


# --------------------------
# FIGURE: ALL AGENCIES BY REVIEW PROCESS (SHARE)
# --------------------------

agency_process_h <- agency_harmonized %>%
  filter(department %in% c("Department of Energy", "Department of the Interior", "Department of Agriculture")) %>%
  count(department, lead_agency_harmonized, process_type) %>%
  group_by(lead_agency_harmonized) %>%
  mutate(total = sum(n), share = n / total) %>%
  ungroup() %>%
  filter(total >= 10)

dept_order_h <- c("Department of Energy", "Department of the Interior", "Department of Agriculture")
dept_labels_h <- c(
  "Department of Energy"       = "Department\nof Energy",
  "Department of the Interior" = "Department\nof the Interior",
  "Department of Agriculture"  = "Department\nof Agriculture"
)

# Pre-compute agency factor ordering by total so both geom layers use the same levels.
# reorder() in aes() is global and character-based; a pre-computed factor is safer
# when geom_text draws from a separate data frame.
agency_levels_h <- agency_process_h %>%
  distinct(lead_agency_harmonized, total) %>%
  arrange(total) %>%
  pull(lead_agency_harmonized)

agency_totals_h <- agency_process_h %>%
  group_by(department, lead_agency_harmonized) %>%
  summarise(total = first(total), .groups = "drop") %>%
  mutate(
    agency_f   = factor(lead_agency_harmonized, levels = agency_levels_h),
    department = factor(department, levels = dept_order_h, labels = dept_labels_h[dept_order_h])
  )

agency_process_h <- agency_process_h %>%
  mutate(
    agency_f   = factor(lead_agency_harmonized, levels = agency_levels_h),
    department = factor(department, levels = dept_order_h, labels = dept_labels_h[dept_order_h])
  )

fig_agency_process <- agency_process_h %>%
  ggplot(aes(x = agency_f, y = share, fill = process_type)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = ifelse(share >= 0.03, scales::percent(share, accuracy = 1), "")),
    position = position_stack(vjust = 0.5),
    size = 3,
    color = "white"
  ) +
  geom_text(
    data = agency_totals_h,
    aes(x = agency_f, y = 1.02, label = scales::comma(total)),
    inherit.aes = FALSE,
    hjust = 0,
    size = 3,
    color = "gray30"
  ) +
  coord_flip(clip = "off") +
  facet_grid(department ~ ., scales = "free_y", space = "free_y", switch = "y") +
  labs(
    x = NULL,
    y = "Share of Projects",
    fill = "Process Type",
    title = "NEPA Process Type Distribution by Federal Agency",
    subtitle = "Only includes Departments of Energy, the Interior, and Agriculture (USDA)",
    caption = "Agencies with fewer than 10 projects excluded. Numbers on right show total project count.\nNote: EA/CE data is complete only for DOE, BLM, and Forest Service; all other agencies appear only via the EPA EIS database."
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

fig_agency_process

ggsave(
  filename = here(figures_dir, "03_agency_process.png"),
  plot = fig_agency_process,
  width = 12,
  height = 10,
  units = "in",
  dpi = 300
)
cat("  Saved: 03_agency_process.png\n")


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
  # BLM from harmonized agencies (case-insensitive match)
  agency_harmonized %>%
    filter(str_detect(lead_agency_harmonized, regex("bureau of land management", ignore_case = TRUE))) %>%
    mutate(
      agency_label = "Bureau of Land Management (BLM)",
      dept_label = "Department of the Interior"
    ),
  # Forest Service: explicit FS records + generic USDA CEs
  # Note: NEPATEC stores some Forest Service CEs under the generic "Department of Agriculture"
  # lead_agency rather than "Department of Agriculture - Forest Service". We include both.
  # If no USDA CEs exist in the clean energy dataset, the second arm returns 0 rows.
  bind_rows(
    agency_harmonized %>% filter(str_detect(lead_agency_harmonized, regex("forest service", ignore_case = TRUE))),
    agency_data %>% filter(department == "Department of Agriculture", process_type == "CE")
  ) %>%
    mutate(agency_label = "Forest Service (USFS)", dept_label = "Department of Agriculture")
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
    caption = "Only agencies with comprehensive EA/CE data in NEPATEC: DOE (all sub-agencies), BLM, and Forest Service."
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
