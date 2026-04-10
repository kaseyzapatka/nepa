# --------------------------
# PROJECT REVIEW: TECHNOLOGY EXTRAS
# --------------------------
# Extra tables/figures and diagnostics not used in deliverable01 report.

source(here::here("phase1", "code", "deliverable01", "00_setup.R"))

# --------------------------
# PREP
# --------------------------

clean_energy_parsed <- clean_energy %>%
  mutate(project_type_list = map(project_type, fromJSON))

projects_with_tags <- clean_energy_parsed %>%
  select(project_id, project_title, project_type_list)

# Explode lead_agency (handles rare multi-agency cases)
agency_data <- clean_energy %>%
  explode_column("lead_agency") %>%
  filter(!is.na(lead_agency) & lead_agency != "") %>%
  mutate(department = project_department)

# Explode project_state (can have multiple values per project)
location_data <- clean_energy %>% 
  explode_column("project_state") %>%
  filter(!is.na(project_state) & project_state != "")

# Helper data/functions for location extras
state_abbr <- c(
  "Alabama" = "AL", "Alaska" = "AK", "Arizona" = "AZ", "Arkansas" = "AR",
  "California" = "CA", "Colorado" = "CO", "Connecticut" = "CT", "Delaware" = "DE",
  "Florida" = "FL", "Georgia" = "GA", "Hawaii" = "HI", "Idaho" = "ID",
  "Illinois" = "IL", "Indiana" = "IN", "Iowa" = "IA", "Kansas" = "KS",
  "Kentucky" = "KY", "Louisiana" = "LA", "Maine" = "ME", "Maryland" = "MD",
  "Massachusetts" = "MA", "Michigan" = "MI", "Minnesota" = "MN", "Mississippi" = "MS",
  "Missouri" = "MO", "Montana" = "MT", "Nebraska" = "NE", "Nevada" = "NV",
  "New Hampshire" = "NH", "New Jersey" = "NJ", "New Mexico" = "NM", "New York" = "NY",
  "North Carolina" = "NC", "North Dakota" = "ND", "Ohio" = "OH", "Oklahoma" = "OK",
  "Oregon" = "OR", "Pennsylvania" = "PA", "Rhode Island" = "RI", "South Carolina" = "SC",
  "South Dakota" = "SD", "Tennessee" = "TN", "Texas" = "TX", "Utah" = "UT",
  "Vermont" = "VT", "Virginia" = "VA", "Washington" = "WA", "West Virginia" = "WV",
  "Wisconsin" = "WI", "Wyoming" = "WY", "District of Columbia" = "DC",
  "Puerto Rico" = "PR", "Guam" = "GU", "Virgin Islands" = "VI"
)

clean_project_type <- function(x) {
  if (is.character(x) && grepl("^\\[", x)) {
    types <- tryCatch(fromJSON(x), error = function(e) x)
  } else if (is.list(x)) {
    types <- unlist(x)
  } else {
    types <- x
  }

  cleaned <- sapply(types, function(t) {
    t <- str_replace(t, "^Renewable Energy Production - ", "")
    t <- str_replace(t, "Utilities \\(electricity, gas, telecommunications\\)", "Utilities")
    t <- str_replace(t, "Carbon Capture and Sequestration", "Carbon Capture")
    return(t)
  })

  paste(unique(cleaned), collapse = ", ")
}

county_with_state <- clean_energy %>%
  mutate(
    first_state = str_extract(project_state, '(?<=\\[")[^"]+'),
    first_state = ifelse(is.na(first_state), project_state, first_state)
  ) %>%
  explode_column("project_county") %>%
  filter(!is.na(project_county) & project_county != "" & project_county != "[]")

county_process_summary <- county_with_state %>%
  count(project_county, first_state, process_type, name = "n_projects") %>%
  arrange(process_type, desc(n_projects))

get_top_counties <- function(process, n = 10) {
  county_process_summary %>%
    filter(process_type == process) %>%
    slice_head(n = n) %>%
    select(project_county, first_state, n_projects)
}

create_county_project_table <- function(counties_df, process_type_filter) {
  projects_in_counties <- county_with_state %>%
    filter(process_type == process_type_filter) %>%
    inner_join(
      counties_df,
      by = c("project_county" = "project_county", "first_state" = "first_state")
    ) %>%
    select(project_id, project_title, project_type, project_county, first_state) %>%
    distinct() %>%
    mutate(
      project_type_clean = sapply(project_type, clean_project_type),
      state_short = ifelse(first_state %in% names(state_abbr),
                           state_abbr[first_state], first_state),
      location = paste0(project_county, ", ", state_short)
    ) %>%
    select(
      `Project Title` = project_title,
      `Technology` = project_type_clean,
      `Location` = location
    ) %>%
    arrange(Location, `Project Title`)

  return(projects_in_counties)
}

top_ce_counties <- get_top_counties("CE", 10)
top_ea_counties <- get_top_counties("EA", 10)
top_eis_counties <- get_top_counties("EIS", 10)

# --------------------------
# TABLE 1: BY TECHNOLOGY (project_type)
# --------------------------

tech_data <- clean_energy %>%
  explode_column("project_type") %>%
  filter(!is.na(project_type) & project_type != "")

table1 <- create_crosstab(tech_data, "project_type")
table1 <- add_totals_row(table1, "project_type")
table1 <- table1 %>%
  rename(
    Technology = project_type,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )

table1 %>% print(n = 100)
write_csv(table1, here(tables_dir, "table1_by_technology.csv"))

# --------------------------
# TABLE 4: CO-OCCURRENCE SUMMARY (TOP 3)
# --------------------------

cooccurrence_summary <- map_dfr(clean_energy_tags, function(ce_tag) {
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

  cooccur_counts <- projects_with_ce_tag %>%
    unnest(project_type_list) %>%
    rename(other_tag = project_type_list) %>%
    filter(other_tag != ce_tag) %>%
    count(other_tag, name = "cooccur_count") %>%
    mutate(cooccur_percent = round(100 * cooccur_count / n_projects, 1)) %>%
    arrange(desc(cooccur_count)) %>%
    slice_head(n = 3) %>%
    mutate(cooccur_rank = row_number())

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
  ) %>%
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

table4 |> print(n = 10)
write_csv(table4, here(tables_dir, "table4_cooccurrence_summary.csv"))

# --------------------------
# TABLE 5: EXHAUSTIVE CO-OCCURRENCE (>5%)
# --------------------------

exhaustive_cooccurrence <- map_dfr(clean_energy_tags, function(ce_tag) {
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

  projects_with_ce_tag %>%
    unnest(project_type_list) %>%
    rename(cooccurring_type = project_type_list) %>%
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

table5 <- exhaustive_cooccurrence %>%
  filter(cooccur_percent > 5) %>%
  rename(
    `Clean Energy Category` = clean_energy_category,
    `Total Projects with Category` = total_projects_with_category,
    `Co-occurring Project Type` = cooccurring_type,
    `Co-occurrence Count` = cooccur_count,
    `Co-occurrence %` = cooccur_percent
  )

table5 |> print(n = 10)
write_csv(table5, here(tables_dir, "table5_cooccurrence_exhaustive.csv"))

# --------------------------
# TABLE 6: PROJECT-LEVEL CO-OCCURRENCE DETAIL
# --------------------------

project_cooccurrence_detail <- map_dfr(clean_energy_tags, function(ce_tag) {
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

  projects_with_ce_tag %>%
    unnest(project_type_list) %>%
    rename(cooccurring_type = project_type_list) %>%
    filter(cooccurring_type != ce_tag) %>%
    mutate(clean_energy_category = ce_tag) %>%
    select(clean_energy_category, project_id, project_title, cooccurring_type)
})

table6 <- project_cooccurrence_detail %>%
  rename(
    `Clean Energy Category` = clean_energy_category,
    `Project ID` = project_id,
    `Project Title` = project_title,
    `Co-occurring Project Type` = cooccurring_type
  )

table6 |> print(n = 10)
write_csv(table6, here(tables_dir, "table6_cooccurrence_projects.csv"))

# --------------------------
# FIGURES: TAG COMPOSITION
# --------------------------

project_tag_counts <- clean_energy %>%
  mutate(project_type_list = map(project_type, fromJSON)) %>%
  mutate(num_clean_tags = map_int(project_type_list, ~ sum(.x %in% clean_energy_tags))) %>%
  mutate(num_total_tags = map_int(project_type_list, ~ length(.x))) %>%
  mutate(num_other_tags = num_total_tags - num_clean_tags) %>%
  mutate(tag_category = case_when(
    num_clean_tags == 0 ~ "0",
    num_clean_tags == 1 ~ "1",
    num_clean_tags == 2 ~ "2",
    num_clean_tags == 3 ~ "3",
    num_clean_tags == 4 ~ "4",
    num_clean_tags >= 5 ~ "5+"
  )) %>%
  mutate(other_category = case_when(
    num_other_tags == 0 ~ "0",
    num_other_tags == 1 ~ "1",
    num_other_tags >= 2 ~ "2+"
  ))

summary_tags <- project_tag_counts %>%
  count(tag_category) %>%
  mutate(percent = 100 * n / sum(n))

fig_tag_composition <- summary_tags %>%
  ggplot(aes(x = tag_category, y = percent, fill = tag_category)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = paste0(round(percent, 1), "%")), vjust = -0.5) +
  labs(
    x = "Number of Clean Energy Tags per Project",
    y = "Percent of Projects",
    title = "Single vs Multi-Tag Composition of Clean Energy Projects"
  ) +
  scale_fill_brewer(palette = "Blues") +
  theme_minimal()

ggsave(
  filename = here(figures_dir, "02_clean_energy_by_tags.png"),
  plot = fig_tag_composition,
  width = 8,
  height = 5,
  units = "in",
  dpi = 300
)

summary_stacked <- project_tag_counts %>%
  count(tag_category, other_category) %>%
  group_by(tag_category) %>%
  mutate(percent_within_clean = 100 * n / sum(n)) %>%
  ungroup()

fig_stacked_bar <- summary_stacked %>%
  ggplot(aes(x = tag_category, y = percent_within_clean, fill = other_category)) +
  geom_col() +
  geom_text(aes(label = paste0(round(percent_within_clean, 1), "%")),
            position = position_stack(vjust = 0.5), color = "white", size = 3) +
  labs(
    x = "Number of Clean Energy Tags per Project",
    y = "Percent of Projects",
    fill = "Number of\nOther Tags",
    title = "Clean Energy Tags by Co-occurring Non-Energy Tags"
  ) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal()

ggsave(
  filename = here(figures_dir, "03_stacked_bar.png"),
  plot = fig_stacked_bar,
  width = 8,
  height = 5,
  units = "in",
  dpi = 300
)

fig_type_by_clean <- clean_energy %>%
  count(project_type_count, project_type_count_clean) %>%
  group_by(project_type_count) %>%
  mutate(percent = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(
    x = factor(project_type_count),
    y = percent,
    fill = factor(project_type_count_clean)
  )) +
  geom_col() +
  scale_y_continuous(labels = percent_format()) +
  labs(
    x = "Total Number of Project Types",
    y = "Percent of Projects",
    fill = "Number of\nClean Energy Types",
    title = "Project Complexity: Total Tags vs Clean Energy Tags"
  ) +
  scale_fill_brewer(palette = "YlGnBu") +
  theme_minimal()

ggsave(
  filename = here(figures_dir, "04_type_by_clean.png"),
  plot = fig_type_by_clean,
  width = 10,
  height = 5,
  units = "in",
  dpi = 300
)

# --------------------------
# ANALYSIS: FLAGGED PROJECTS
# --------------------------

cat("\n=== Flagged Projects for Review ===\n")

flagged <- clean_energy %>%
  filter(project_energy_type_questions == TRUE)

cat("Projects flagged for manual review:", nrow(flagged), "\n")

if (nrow(flagged) > 0) {
  flagged %>%
    select(project_id, project_title, project_type, project_energy_type_questions) %>%
    write_csv(here(tables_dir, "flagged_for_review.csv"))
}

# --------------------------
# ANALYSIS: UTILITIES + BROADBAND ONLY
# --------------------------

cat("\n=== Utilities + Broadband Only Analysis ===\n")

utilities_broadband_only <- clean_energy_parsed %>%
  mutate(
    has_utilities = map_lgl(project_type_list,
                            ~ "Utilities (electricity, gas, telecommunications)" %in% .x),
    has_broadband = map_lgl(project_type_list, ~ "Broadband" %in% .x),
    tag_count = map_int(project_type_list, length)
  ) %>%
  filter(has_utilities & has_broadband & tag_count == 2)

cat("Projects with ONLY Utilities + Broadband tags:", nrow(utilities_broadband_only), "\n")

if (nrow(utilities_broadband_only) > 0) {
  utilities_broadband_only %>%
    select(project_id, project_title, project_type, process_type) %>%
    write_csv(here(tables_dir, "utilities_broadband_only.csv"))
}

# --------------------------
# ANALYSIS: NUCLEAR TECHNOLOGY ONLY
# --------------------------

cat("\n=== Nuclear Technology Only Analysis ===\n")

nuclear_tech_only <- clean_energy_parsed %>%
  mutate(
    has_nuclear_tech = map_lgl(project_type_list, ~ "Nuclear Technology" %in% .x),
    has_nuclear_production = map_lgl(project_type_list,
                                     ~ "Conventional Energy Production - Nuclear" %in% .x)
  ) %>%
  filter(has_nuclear_tech & !has_nuclear_production)

cat("Projects with Nuclear Technology but NOT Nuclear Production:", nrow(nuclear_tech_only), "\n")

# --------------------------
# ANALYSIS: STRICT VS BROAD COMPARISON
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

cat("\n=== Technology Extras Complete ===\n")

# --------------------------
# AGENCY EXTRAS (FROM 02_agency.R)
# --------------------------

cat("\n=== Agency Extras ===\n")

# Multi-agency projects (rare but worth noting)
cat("\nMulti-agency projects in dataset:\n")
multi_agency <- clean_energy %>%
  filter(str_detect(lead_agency, ","))
cat("  Count:", nrow(multi_agency), "(these have >1 lead agency)\n")

# Agencies with highest CE ratio (streamlined projects)
ce_ratio <- agency_data %>%
  count(lead_agency, process_type) %>%
  pivot_wider(names_from = process_type, values_from = n, values_fill = 0) %>%
  mutate(
    total = EA + EIS + CE,
    ce_ratio = CE / total
  ) %>%
  filter(total >= 50) %>%
  arrange(desc(ce_ratio))

cat("\nAgencies with highest CE ratio (min 50 projects):\n")
ce_ratio %>% slice_head(n = 10) %>% print()

# Exploratory: check for FERC
projects |> 
  filter(str_detect(project_sponsor, regex("Regulatory Commission", ignore_case = TRUE))) |> 
  select(lead_agency_harmonized) |> 
  distinct() |> 
  glimpse()

ferc_hits <- agency_data |>
  filter(
    str_detect(lead_agency, regex("Federal Energy Regulatory Commission|\\bFERC\\b", ignore_case = TRUE)) |
      str_detect(project_department, regex("Federal Energy Regulatory Commission|\\bFERC\\b", ignore_case = TRUE))
  ) |>
  distinct(project_department, lead_agency)

ferc_hits |> print(n = 50)
nrow(ferc_hits)

# --------------------------
# LOCATION EXTRAS (FROM 03_location.R)
# --------------------------

cat("\n=== Location Extras ===\n")

# share that needs geocoding
clean_energy |> 
  count(project_location_needs_geocoding) |> 
  glimpse() # 45/22305

# view locations that need to be geo-coded  
clean_energy |> 
  filter(project_location_needs_geocoding == TRUE) |> 
  select(project_location, project_state:project_lon) |> 
  print(n = 50) 

# view locations that need to be geo-coded  
clean_energy |> 
  filter(project_county == "[]") |> 
  select(project_location, project_state:project_lon) |> 
  print(n = 50) 

clean_energy |> 
  filter(process_type  == "CE") |> 
  select(process_type, project_county, project_state) |> 
  filter(project_county == "[]") |> 
  glimpse()

# South Carolina projects exploratory sheet
south_carolina_projects <- 
  location_data |> 
  filter(str_detect(project_state, "South Carolina")) |> 
  select(project_title, project_type, project_state) |> 
  glimpse()

sheet_write(
  data = south_carolina_projects,
  ss = "https://docs.google.com/spreadsheets/d/1FXIN41UEhh4GJERrv0bze70UQT9I0eeWOnfH3plYF3g/edit?usp=sharing",
  sheet = "south_carolina_projects"
)

# Table: State + County nested crosstab with totals at the top
table_county_state <- location_data %>%
  mutate(
    project_county = map(
      project_county,
      ~ if (.x == "[]" | is.na(.x)) NA_character_ else fromJSON(.x)
    )
  ) %>%
  unnest(project_county, keep_empty = TRUE) %>%
  mutate(
    geo_label = if_else(
      is.na(project_county),
      project_state,
      paste0("  \u2514\u2500 ", project_county)
    ),
    geo_state = project_state,
    geo_level = if_else(is.na(project_county), "State", "County")
  ) %>%
  count(geo_label, geo_state, geo_level, process_type) %>%
  pivot_wider(
    names_from = process_type,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(Total = rowSums(across(c(EA, EIS, CE)))) %>%
  arrange(
    geo_state,
    desc(geo_level == "State"),
    geo_label
  ) %>%
  select(
    Geography = geo_label,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE,
    Total
  )

grand_totals <- table_county_state %>%
  summarise(
    Geography = "TOTAL (All States & Counties)",
    across(where(is.numeric), sum)
  )

table_county_state <- bind_rows(grand_totals, table_county_state)
table_county_state %>% print(n = 80)

write_csv(
  table_county_state,
  here(tables_dir, "table3_by_state_and_county_totals.csv")
)

sheet_write(
  data = table_county_state,
  ss = "https://docs.google.com/spreadsheets/d/1FXIN41UEhh4GJERrv0bze70UQT9I0eeWOnfH3plYF3g/edit?usp=sharing",
  sheet = "table_county_state"
)

# Deep dive: top 10 counties tables (CSV backups)
table_ce_deep_dive <- create_county_project_table(top_ce_counties, "CE")
table_ea_deep_dive <- create_county_project_table(top_ea_counties, "EA")
table_eis_deep_dive <- create_county_project_table(top_eis_counties, "EIS")

write_csv(table_ce_deep_dive, here(tables_dir, "deep_dive_ce_top_counties.csv"))
write_csv(table_ea_deep_dive, here(tables_dir, "deep_dive_ea_top_counties.csv"))
write_csv(table_eis_deep_dive, here(tables_dir, "deep_dive_eis_top_counties.csv"))

# --------------------------
# ANALYSIS: COUNTY DATA COVERAGE BY PROCESS TYPE
# --------------------------

cat("\n=== County Data Coverage Analysis ===\n")

county_coverage <- clean_energy %>%
  mutate(
    has_county = !is.na(project_county) &
      project_county != "" &
      project_county != "[]"
  ) %>%
  group_by(process_type) %>%
  summarise(
    total_projects = n(),
    projects_with_county = sum(has_county),
    projects_missing_county = sum(!has_county),
    pct_with_county = round(100 * sum(has_county) / n(), 1),
    pct_missing_county = round(100 * sum(!has_county) / n(), 1)
  ) %>%
  arrange(desc(pct_missing_county))

cat("\nCounty data coverage by NEPA process type:\n")
print(county_coverage)

cat("\n=== Summary Statistics ===\n")
cat("Overall county data coverage:", round(100 * n_projects_with_county / nrow(clean_energy), 1), "%\n")
cat("CE projects with county data:",
    round(100 * sum(clean_energy$process_type == "CE" &
                    clean_energy$project_county != "[]" &
                    !is.na(clean_energy$project_county), na.rm = TRUE) /
            sum(clean_energy$process_type == "CE"), 1), "%\n")
cat("EA projects with county data:",
    round(100 * sum(clean_energy$process_type == "EA" &
                    clean_energy$project_county != "[]" &
                    !is.na(clean_energy$project_county), na.rm = TRUE) /
            sum(clean_energy$process_type == "EA"), 1), "%\n")
cat("EIS projects with county data:",
    round(100 * sum(clean_energy$process_type == "EIS" &
                    clean_energy$project_county != "[]" &
                    !is.na(clean_energy$project_county), na.rm = TRUE) /
            sum(clean_energy$process_type == "EIS"), 1), "%\n")

# Sample projects missing county data by process type
cat("\n=== Sample CE Projects Missing County Data ===\n")
ce_missing_county <- clean_energy %>%
  filter(process_type == "CE" & (is.na(project_county) | project_county == "[]")) %>%
  select(project_id, project_title, project_location, project_state, project_county, lead_agency) %>%
  slice_head(n = 10)

print(ce_missing_county)

cat("\n=== Sample CE Projects WITH County Data ===\n")
ce_with_county <- clean_energy %>%
  filter(process_type == "CE" & !is.na(project_county) & project_county != "[]") %>%
  select(project_id, project_title, project_location, project_state, project_county, lead_agency) %>%
  slice_head(n = 10)

print(ce_with_county)

# Agency patterns in missing county data
cat("\n=== County Data Coverage by Agency (for CE projects) ===\n")
ce_county_by_agency <- clean_energy %>%
  filter(process_type == "CE") %>%
  mutate(
    has_county = !is.na(project_county) &
      project_county != "" &
      project_county != "[]"
  ) %>%
  group_by(project_department) %>%
  summarise(
    total_ce_projects = n(),
    ce_with_county = sum(has_county),
    pct_with_county = round(100 * sum(has_county) / n(), 1)
  ) %>%
  arrange(desc(total_ce_projects)) %>%
  slice_head(n = 10)

print(ce_county_by_agency)

cat("\n=== EA Projects Missing County Data ===\n")
ea_missing_county <- clean_energy %>%
  filter(process_type == "EA" & (is.na(project_county) | project_county == "[]")) %>%
  select(project_id, project_title, project_location, project_state, project_county, lead_agency, project_department)

cat("Total EA projects missing county:", nrow(ea_missing_county), "out of",
    sum(clean_energy$process_type == "EA"), "EA projects\n")
cat("Sample of EA projects missing county data:\n")
print(ea_missing_county %>% slice_head(n = 15))

cat("\n=== Location Field Analysis for EA Missing County ===\n")
ea_location_patterns <- ea_missing_county %>%
  mutate(
    has_location = !is.na(project_location) & project_location != "" & project_location != "[]",
    location_length = nchar(project_location),
    mentions_county = str_detect(tolower(project_location), "county|counties"),
    has_coordinates = str_detect(project_location, "latitude|longitude|lat|long|coord"),
    has_legal_desc = str_detect(project_location, "T\\.|Township|Section|Range|Meridian"),
    location_type = case_when(
      !has_location ~ "No location",
      mentions_county ~ "Mentions county",
      has_coordinates ~ "Has coordinates",
      has_legal_desc ~ "Legal description",
      TRUE ~ "Other description"
    )
  )

cat("\nEA Location field patterns:\n")
print(ea_location_patterns %>% count(location_type, sort = TRUE))

cat("\n=== EIS Projects Missing County Data ===\n")
eis_missing_county <- clean_energy %>%
  filter(process_type == "EIS" & (is.na(project_county) | project_county == "[]")) %>%
  select(project_id, project_title, project_location, project_state, project_county, lead_agency, project_department)

cat("Total EIS projects missing county:", nrow(eis_missing_county), "out of",
    sum(clean_energy$process_type == "EIS"), "EIS projects\n")
cat("Sample of EIS projects missing county data:\n")
print(eis_missing_county %>% slice_head(n = 15))

cat("\n=== Location Field Analysis for EIS Missing County ===\n")
eis_location_patterns <- eis_missing_county %>%
  mutate(
    has_location = !is.na(project_location) & project_location != "" & project_location != "[]",
    location_length = nchar(project_location),
    mentions_county = str_detect(tolower(project_location), "county|counties"),
    has_coordinates = str_detect(project_location, "latitude|longitude|lat|long|coord"),
    has_legal_desc = str_detect(project_location, "T\\.|Township|Section|Range|Meridian"),
    mentions_multiple = str_detect(tolower(project_location), "multiple|various|several|region|area"),
    location_type = case_when(
      !has_location ~ "No location",
      mentions_county ~ "Mentions county",
      has_coordinates ~ "Has coordinates",
      mentions_multiple ~ "Multiple/regional",
      has_legal_desc ~ "Legal description",
      TRUE ~ "Other description"
    )
  )

cat("\nEIS Location field patterns:\n")
print(eis_location_patterns %>% count(location_type, sort = TRUE))

cat("\n=== Sample EIS Projects by Location Pattern ===\n")
for (loc_type in unique(eis_location_patterns$location_type)) {
  cat("\n", loc_type, ":\n")
  sample <- eis_location_patterns %>%
    filter(location_type == loc_type) %>%
    select(project_title, project_location, project_state) %>%
    slice_head(n = 3)
  print(sample)
}

cat("\n=== Geocoding Potential (projects with lat/long) ===\n")
projects_with_coords <- clean_energy %>%
  filter(process_type %in% c("EA", "EIS")) %>%
  filter(is.na(project_county) | project_county == "[]") %>%
  filter(!is.na(project_lat) & !is.na(project_lon) &
           project_lat != 0 & project_lon != 0) %>%
  select(process_type, project_id, project_title, project_lat, project_lon,
         project_location, project_state)

cat("EA/EIS projects missing county but WITH lat/long coordinates:\n")
print(projects_with_coords %>% count(process_type))

if (nrow(projects_with_coords) > 0) {
  cat("\nSample projects that could be reverse geocoded:\n")
  print(projects_with_coords %>% slice_head(n = 10))
}

cat("\n=== Agency Patterns for EA/EIS Missing County ===\n")
ea_eis_missing_by_agency <- clean_energy %>%
  filter(process_type %in% c("EA", "EIS")) %>%
  mutate(
    has_county = !is.na(project_county) & project_county != "" & project_county != "[]"
  ) %>%
  group_by(process_type, project_department) %>%
  summarise(
    total = n(),
    missing_county = sum(!has_county),
    pct_missing = round(100 * sum(!has_county) / n(), 1),
    .groups = "drop"
  ) %>%
  filter(total >= 5) %>%
  arrange(process_type, desc(pct_missing))

print(ea_eis_missing_by_agency)

cat("\n=== Location Analysis ===\n")

eis_states <- location_data %>%
  filter(process_type == "EIS") %>%
  count(project_state, name = "n_eis") %>%
  arrange(desc(n_eis))

cat("\nTop 10 states by EIS count (most complex projects):\n")
eis_states %>% slice_head(n = 10) %>% print()

eis_ratio <- location_data %>%
  count(project_state, process_type) %>%
  pivot_wider(names_from = process_type, values_from = n, values_fill = 0) %>%
  mutate(
    total = EA + EIS + CE,
    eis_ratio = EIS / total
  ) %>%
  filter(total >= 100) %>%
  arrange(desc(eis_ratio))

cat("\nStates with highest EIS ratio (min 100 projects):\n")
eis_ratio %>% slice_head(n = 10) %>% print()

multi_state <- clean_energy %>%
  mutate(n_states = str_count(project_state, "\\|") + 1) %>%
  filter(n_states > 1)

cat("\nMulti-state projects:", nrow(multi_state), "\n")
cat("(Projects spanning multiple states)\n")

# Aiken, SC exploratory filter
clean_energy |> 
  select(project_title, project_state, project_type, project_county) |> 
  filter(sapply(project_state, function(x) {
    states <- fromJSON(x)
    "South Carolina" %in% states
  })) |> 
  filter(sapply(project_county, function(x) {
    states <- fromJSON(x)
    "Aiken" %in% states
  })) |> 
  glimpse()
