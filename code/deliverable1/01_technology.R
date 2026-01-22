# --------------------------
# DELIVERABLE 1: CLEAN ENERGY BY TECHNOLOGY
# --------------------------
# Table 1: Clean Energy by Technology (project_type)
# Includes co-occurrence analysis and related figures

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable1", "00_setup.R"))

# --------------------------
# PROCESS
# --------------------------

# Parse project types and create working datasets
clean_energy_parsed <- clean_energy %>%
  mutate(project_type_list = map(project_type, fromJSON))

# Count clean energy tags per project
project_tag_counts <- clean_energy_parsed %>%
  mutate(
    num_clean_tags = map_int(project_type_list, ~ sum(.x %in% clean_energy_tags)),
    num_total_tags = map_int(project_type_list, ~ length(.x)),
    num_other_tags = num_total_tags - num_clean_tags,
    tag_category = case_when(
      num_clean_tags == 0 ~ "0",
      num_clean_tags == 1 ~ "1",
      num_clean_tags == 2 ~ "2",
      num_clean_tags == 3 ~ "3",
      num_clean_tags == 4 ~ "4",
      num_clean_tags >= 5 ~ "5+"
    ),
    other_category = case_when(
      num_other_tags == 0 ~ "0",
      num_other_tags == 1 ~ "1",
      num_other_tags >= 2 ~ "2+"
    )
  )

# Explode project_type for table creation
tech_data <- clean_energy %>%
  explode_column("project_type") %>%
  filter(!is.na(project_type) & project_type != "")

# Projects with tags for co-occurrence analysis
projects_with_tags <- clean_energy_parsed %>%
  #select(project_id, project_title, project_type_list)
  select(project_type_list)


# --------------------------
# EXPLORATORY
# --------------------------

#
# Utilities only
# ----------------------------------------
# There were about 1,623 projects that had some combination of Utilities we didn't want 
# to count as clean energy
utilties_only_projects <-
  projects |> 
  filter(project_energy_type == "Clean") |> 
  # remove Utilities + Broadband, Waste Management, or Land Development tags
  filter(project_utilities_to_filter_out) |> 
  select(project_title, project_type) |> 
  glimpse()

# save
sheet_write(
  data = utilties_only_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "utilties_only_projects"
)


# --------------------------
# TABLE 1: BY TECHNOLOGY (project_type)
# --------------------------

cat("Creating Table 1: Clean Energy by Technology...\n")

table1 <- create_crosstab(tech_data, "project_type")

# Add totals row
table1 <- add_totals_row(table1, "project_type")

# Rename for clarity
table1 <- table1 %>%
  rename(
    Technology = project_type,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )

table1 %>% print(n = 100)

# Save
write_csv(table1, here(tables_dir, "table1_by_technology.csv"))
cat("  Saved: table1_by_technology.csv\n")


# --------------------------
# TABLE 4: CO-OCCURRENCE SUMMARY (TOP 3)
# --------------------------

cat("Creating Table 4: Co-occurrence Summary by Clean Energy Type...\n")

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

  # Pad to 3 rows if needed
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

# Pivot to wide format
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
cat("  Saved: table4_cooccurrence_summary.csv\n")


# --------------------------
# TABLE 5: EXHAUSTIVE CO-OCCURRENCE (>5%)
# --------------------------

cat("Creating Table 5: Exhaustive Co-occurrence by Clean Energy Type...\n")

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
cat("  Saved: table5_cooccurrence_exhaustive.csv\n")


# --------------------------
# TABLE 6: PROJECT-LEVEL CO-OCCURRENCE DETAIL
# --------------------------

cat("Creating Table 6: Project-level Co-occurrence Detail...\n")

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
cat("  Saved: table6_cooccurrence_projects.csv\n")


# --------------------------
# FIGURES
# --------------------------

#
# Executive Summary: Clean Energy by review
# ----------------------------------------
projects_by_review <- projects %>%
  count(process_type, name = "projects") %>%
  mutate(
    process_type = factor(process_type, 
                          levels = c("CE", "EA", "EIS"),
                          labels = c("Categorical Exclusion (CE)", 
                                     "Environmental Assessment (EA)", 
                                     "Environmental Impact Statement (EIS)"))
  )

fig_projects_by_review <- projects_by_review %>%
  ggplot(aes(x = reorder(process_type, projects), y = projects)) +
  geom_col(fill = catf_dark_blue) +
  geom_text(aes(label = scales::comma(projects)), vjust = -0.3, size = 3.5) +
  labs(
    x = NULL,
    y = "Number of Projects",
    title = "Projects by NEPA Process Type"
  ) +
  scale_y_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.12))
  )

fig_projects_by_review

ggsave(
  filename = here(figures_dir, "00_fig_projects_by_review.png"),
  plot = fig_projects_by_review,
  width = 8,
  height = 5,
  dpi = 300
)

#
# Executive Summary: Clean Energy by review
# ----------------------------------------
clean_energy_by_review <- clean_energy %>%
  count(process_type, name = "projects") %>%
  mutate(
    process_type = factor(process_type, 
                          levels = c("CE", "EA", "EIS"),
                          labels = c("Categorical Exclusion (CE)", 
                                     "Environmental Assessment (EA)", 
                                     "Environmental Impact Statement (EIS)"))
  )

fig_clean_energy_by_review <- clean_energy_by_review %>%
  ggplot(aes(x = reorder(process_type, projects), y = projects)) +
  geom_col(fill = catf_dark_blue) +
  geom_text(aes(label = scales::comma(projects)), vjust = -0.3, size = 3.5) +
  labs(
    x = NULL,
    y = "Number of Projects",
    title = "Clean Energy Projects by NEPA Process Type"
  ) +
  scale_y_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.12))
  )

fig_clean_energy_by_review

ggsave(
  filename = here(figures_dir, "00_fig_clean_energy_by_review.png"),
  plot = fig_clean_energy_by_review,
  width = 8,
  height = 5,
  dpi = 300
)


#
# Executive Summary: Energy Type Breakdown (Clean, Fossil, Other)
# ----------------------------------------
energy_type_summary <- projects %>%
  count(project_energy_type, name = "projects") %>%
  mutate(
    share = projects / sum(projects),
    project_energy_type = factor(project_energy_type, 
                                  levels = c("Clean", "Fossil", "Other"))
  )

fig_energy_type <- energy_type_summary %>%
  ggplot(aes(x = reorder(project_energy_type, -projects), y = projects, 
             fill = project_energy_type)) +
  geom_col() +
  geom_text(aes(label = paste0(scales::comma(projects), "\n(", 
                                scales::percent(share, accuracy = 0.1), ")")), 
            vjust = -0.2, size = 3.5) +
  labs(
    x = NULL,
    y = "Number of Projects",
    title = "NEPA Projects by Energy Type"
  ) +
  scale_y_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.15))
  ) +
  scale_fill_manual(values = c("Clean" = catf_teal, 
                                "Fossil" = catf_navy, 
                                "Other" = catf_light_blue)) +
  theme(legend.position = "none")

fig_energy_type

ggsave(
  filename = here(figures_dir, "00_energy_type_breakdown.png"),
  plot = fig_energy_type,
  width = 8,
  height = 5,
  dpi = 300
)


#
# Deliverable: Clean Energy Bar Chart (by technology)
# ----------------------------------------
clean_energy_summary <- clean_energy_parsed %>%
  select(project_title, project_type_list) %>%
  unnest(project_type_list) %>%
  rename(technology = project_type_list) %>%
  filter(technology %in% clean_energy_tags) %>%
  distinct(project_title, technology) %>%
  count(technology, name = "n") %>%
  mutate(
    percent_projects = 100 * n / n_distinct(clean_energy$project_title)
  ) %>%
  arrange(desc(percent_projects))

fig_clean_energy_bar <- clean_energy_summary %>%
  ggplot(aes(x = percent_projects,
             y = reorder(technology, percent_projects))) +
  geom_col(fill = catf_dark_blue) +
  geom_text(aes(label = scales::comma(n)), hjust = -0.1, size = 3) +
  labs(
    x = "Percent of Clean Energy Projects",
    y = NULL,
    title = "Clean Energy Projects by Technology Type"
  ) +
  scale_x_continuous(
    labels = function(x) paste0(x, "%"),
    expand = expansion(mult = c(0, 0.12))
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 9))

fig_clean_energy_bar

# save
ggsave(
  filename = here(figures_dir, "01_clean_energy_bar_chart.png"),
  plot = fig_clean_energy_bar,
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)


#
# Clean Energy Bar Chart by Technology and Process Type (100% stacked)
# --------------------------------------------------------------------
clean_energy_summary_by_process <- clean_energy_parsed %>%
  select(project_title, project_type_list, process_type) %>%
  unnest(project_type_list) %>%
  rename(technology = project_type_list) %>%
  filter(technology %in% clean_energy_tags) %>%
  distinct(project_title, technology, process_type) %>%
  count(technology, process_type, name = "n") %>%
  group_by(technology) %>%
  mutate(
    share = n / sum(n)
  ) %>%
  ungroup()

# Use same order as fig_clean_energy_bar (by percent_projects)
tech_order <- clean_energy_summary %>%
  arrange(percent_projects) %>%
  pull(technology)

clean_energy_summary_by_process <- clean_energy_summary_by_process %>%
  mutate(technology = factor(technology, levels = tech_order))

# Plot
fig_clean_energy_bar_by_process <- ggplot(
  clean_energy_summary_by_process,
  aes(
    x = share,
    y = technology,
    fill = process_type
  )
) +
  geom_col(width = 0.8) +
  geom_text(
    aes(label = scales::percent(share, accuracy = 1)),
    position = position_stack(vjust = 0.5),
    size = 3,
    color = "white"
  ) +
  scale_x_continuous(
    labels = scales::percent,
    expand = expansion(mult = c(0, 0.02))
  ) +
  scale_fill_manual(
    values = c("CE" = catf_dark_blue, "EA" = catf_teal, "EIS" = catf_magenta)
  ) +
  labs(
    x = "Share of Projects",
    y = NULL,
    fill = "Process Type",
    title = "Clean Energy Projects by Technology and Process Type"
  ) +
  theme(
    axis.text.y = element_text(size = 9),
    legend.position = "bottom"
  )

fig_clean_energy_bar_by_process

# save
ggsave(
  filename = here(figures_dir, "02_clean_energy_bar_by_process.png"),
  plot = fig_clean_energy_bar_by_process,
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)




# --------------------------
# EXTRA FIGURES
# --------------------------


# Figure 2: Single vs Multi-Tag Composition
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

fig_tag_composition

ggsave(
  filename = here(figures_dir, "02_clean_energy_by_tags.png"),
  plot = fig_tag_composition,
  width = 8,
  height = 5,
  units = "in",
  dpi = 300
)
cat("  Saved: 02_clean_energy_by_tags.png\n")


# Figure 3: Clean vs Other Tags Stacked Bar
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

fig_stacked_bar

ggsave(
  filename = here(figures_dir, "03_stacked_bar.png"),
  plot = fig_stacked_bar,
  width = 8,
  height = 5,
  units = "in",
  dpi = 300
)
cat("  Saved: 03_stacked_bar.png\n")


# Figure 4: Project Type by Clean Energy Count
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

fig_type_by_clean

ggsave(
  filename = here(figures_dir, "04_type_by_clean.png"),
  plot = fig_type_by_clean,
  width = 10,
  height = 5,
  units = "in",
  dpi = 300
)
cat("  Saved: 04_type_by_clean.png\n")


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
  cat("  Saved: flagged_for_review.csv\n")
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
cat("  (These are likely telecom projects, not clean energy)\n")

cat("\nBy process type:\n")
utilities_broadband_only %>%
  count(process_type) %>%
  print()

if (nrow(utilities_broadband_only) > 0) {
  utilities_broadband_only %>%
    select(project_id, project_title, project_type, process_type) %>%
    write_csv(here(tables_dir, "utilities_broadband_only.csv"))
  cat("  Saved: utilities_broadband_only.csv\n")
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
cat("  (These are likely waste/R&D projects, not power generation)\n")

cat("\nBy process type:\n")
nuclear_tech_only %>%
  count(process_type) %>%
  print()


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


# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Technology Script Complete ===\n")
cat("Tables saved to:", tables_dir, "\n")
cat("Figures saved to:", figures_dir, "\n")
