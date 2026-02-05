# --------------------------
# DELIVERABLE 1: CLEAN ENERGY BY LOCATION
# --------------------------
# Table 3: Clean Energy by State
# Geographic analysis of clean energy projects

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable01", "00_setup.R"))


# --------------------------
# EXPLORATORY ANALYSIS
# --------------------------

# share that needs geocoding
clean_energy |> 
  count(project_location_needs_geocoding) |> 
  glimpse() # 45/22305

# view locations that need to be geo-coded  
clean_energy |> 
  filter(project_location_needs_geocoding == TRUE) |> 
  select(project_location, project_state:project_lon) |> 
  print(n = 50) 

  # most are in US Territories or have a default geolocation - geographic center of US


# view locations that need to be geo-coded  
clean_energy |> 
  #filter(is.na(project_location)) |> 
  #filter(is_empty(project_county)) |> 
  filter(project_county == "[]") |> 
  select(project_location, project_state:project_lon) |> 
  print(n = 50) 

clean_energy |> 
  filter(process_type  == "CE") |> 
  #select(process_type, project_lat, project_lon, project_county, project_state) |> 
  select(process_type, project_county, project_state) |> 
  filter(project_county == "[]") |> 
  glimpse()


# --------------------------
# PROCESS
# --------------------------

# Explode project_state (can have multiple values per project)
location_data <- clean_energy %>% 
  explode_column("project_state") %>%
  filter(!is.na(project_state) & project_state != "")

# Count projects per state
state_counts <- location_data %>%
  count(project_state, name = "n_projects") %>%
  arrange(desc(n_projects))

cat("Unique states/territories:", nrow(state_counts), "\n")
cat("Top 10 states by project count:\n")
state_counts %>% slice_head(n = 10) %>% print()



# --------------------------
# EXPLORATORY
# --------------------------

south_carolina_projects <- 
  location_data |> 
  filter(str_detect(project_state, "South Carolina")) |> 
  select(project_title, project_type, project_state) |> 
  glimpse()

# save
sheet_write(
  data = south_carolina_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "south_carolina_projects"
)

# --------------------------
# TABLE 3: BY STATE
# --------------------------

cat("\nCreating Table 3: Clean Energy by Location (State)...\n")

table3 <- create_crosstab(location_data, "project_state")

# Add totals row
table3 <- add_totals_row(table3, "project_state")

# Rename for clarity
table3 <- table3 %>%
  rename(
    State = project_state,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  )

table3 %>% print(n = 60)

# Save
write_csv(table3, here(tables_dir, "table3_by_state.csv"))

# --------------------------
# TABLE 4: BY STATE & BY COUNTY
# --------------------------

# A–G: State + County nested crosstab with totals at the top
table_county_state <- location_data %>%
  # ---- A. Parse + unnest county JSON ----
  mutate(
    project_county = map(
      project_county,
      ~ if (.x == "[]" | is.na(.x)) NA_character_ else fromJSON(.x)
    )
  ) %>%
  unnest(project_county, keep_empty = TRUE) %>%
  
  # ---- B. Create identifiers + display labels ----
  mutate(
    geo_label = if_else(
      is.na(project_county),
      project_state,
      paste0("  \u2514\u2500 ", project_county)   # indented county
    ),
    geo_state = project_state,
    geo_level = if_else(is.na(project_county), "State", "County")
  ) %>%
  
  # ---- C. Crosstab by geography + process type ----
  count(geo_label, geo_state, geo_level, process_type) %>%
  pivot_wider(
    names_from = process_type,
    values_from = n,
    values_fill = 0
  ) %>%
  
  # ---- D. Add totals ----
  mutate(
    Total = rowSums(across(c(EA, EIS, CE)))
  ) %>%
  
  # ---- E. Order: state first, then counties ----
  arrange(
    geo_state,
    desc(geo_level == "State"),
    geo_label
  ) %>%
  
  # ---- F. Clean up for output ----
  select(
    Geography = geo_label,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE,
    Total
  )

# ---- G. Add grand totals row at the top ----
grand_totals <- table_county_state %>%
  summarise(
    Geography = "TOTAL (All States & Counties)",
    across(where(is.numeric), sum)
  )

table_county_state <- bind_rows(grand_totals, table_county_state)

# ---- H. Print + save ----
table_county_state %>% print(n = 80)

write_csv(
  table_county_state,
  here(tables_dir, "table3_by_state_and_county_totals.csv")
)


# save
sheet_write(
  data = table_county_state,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "table_county_state"
)


# --------------------------
# FIGURES
# --------------------------

# Figure 1: Top 20 States Bar Chart
top_states <- state_counts %>%
  slice_head(n = 20)

fig_top_states <- top_states %>%
  ggplot(aes(x = n_projects, y = reorder(project_state, n_projects))) +
  geom_col(fill = catf_dark_blue) +
  geom_text(aes(label = n_projects), hjust = -0.2, size = 3) +
  labs(
    x = "Number of Clean Energy Projects",
    y = NULL,
    title = "Top 20 States for Clean Energy Projects"
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 9)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15)))

fig_top_states

ggsave(
  filename = here(figures_dir, "07_top_states.png"),
  plot = fig_top_states,
  width = 10,
  height = 8,
  units = "in",
  dpi = 300
)
cat("  Saved: 07_top_states.png\n")


# Figure 2: State by Process Type (Top 15) - reorder by share of CE
top15_state_process <- location_data %>%
  filter(project_state %in% top_states$project_state[1:15]) %>%
  count(project_state, process_type) %>%
  group_by(project_state) %>%
  mutate(
    total = sum(n),
    percent = 100 * n / total
  ) %>%
  ungroup()

fig_state_process <- top15_state_process %>%
  ggplot(aes(x = reorder(project_state, total), y = percent, fill = process_type)) +
  geom_col() +
  geom_text(
    aes(label = ifelse(percent >= 3, paste0(round(percent), "%"), "")),
    position = position_stack(vjust = 0.5),
    size = 3,
    color = "white"
  ) +
  coord_flip() +
  labs(
    x = NULL,
    y = "Percent of Projects",
    fill = "Process Type",
    title = "Process Type Distribution by State (Top 15)",
    caption = "Note: Percentage labels below 3% are excluded for readability."
  ) +
  scale_fill_manual(
    values = c("CE" = catf_dark_blue, "EA" = catf_teal, "EIS" = catf_magenta)
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 9))

fig_state_process

ggsave(
  filename = here(figures_dir, "08_state_process_type.png"),
  plot = fig_state_process,
  width = 10,
  height = 7,
  units = "in",
  dpi = 300
)
cat("  Saved: 08_state_process_type.png\n")


# --------------------------
# MAPS
# --------------------------

# Load additional packages for mapping
library(sf)
library(tigris)
options(tigris_use_cache = TRUE)

# Create maps output directory
maps_dir <- here("output", "deliverable1", "maps")
dir.create(maps_dir, showWarnings = FALSE, recursive = TRUE)

cat("\n=== Creating Maps ===\n")

# --- LOAD AND PREPARE BASE GEOMETRIES ---

# Get US states shapefile and shift AK/HI
us_states <- states(cb = TRUE, resolution = "20m") %>%
  filter(!STUSPS %in% c("PR", "VI", "GU", "AS", "MP")) %>%
  shift_geometry()  # Move AK and HI underneath continental US

# Join with project counts
state_map_data <- us_states %>%
  left_join(state_counts, by = c("NAME" = "project_state")) %>%
  mutate(n_projects = replace_na(n_projects, 0))

# Get US counties shapefile and shift AK/HI
us_counties <- counties(cb = TRUE, resolution = "20m") %>%
  filter(!STATEFP %in% c("72", "78", "66", "69", "60")) %>%
  shift_geometry()

# Get state names for joining
state_fips <- tigris::fips_codes %>%
  select(state, state_code, state_name) %>%
  distinct()

us_counties <- us_counties %>%
  left_join(state_fips, by = c("STATEFP" = "state_code"))

# --- PROCESS COUNTY DATA ---

# Extract first state for proper county matching (counties have same names across states)
county_with_state <- clean_energy %>%
  mutate(
    first_state = str_extract(project_state, '(?<=\\[")[^"]+'),
    first_state = ifelse(is.na(first_state), project_state, first_state)
  ) %>%
  explode_column("project_county") %>%
  filter(!is.na(project_county) & project_county != "" & project_county != "[]")

county_counts <- county_with_state %>%
  count(project_county, first_state, name = "n_projects") %>%
  arrange(desc(n_projects))

# Calculate data coverage stats for footnotes
n_projects_with_county <- n_distinct(county_with_state$project_id)
pct_with_county <- round(100 * n_projects_with_county / nrow(clean_energy), 1)
n_missing_county <- nrow(clean_energy) - n_projects_with_county

cat("  Projects with county data:", n_projects_with_county,
    "(", pct_with_county, "% of clean energy projects)\n")
cat("  Projects missing county:", n_missing_county, "\n")


# Join county counts with shapefile
county_map_data <- us_counties %>%
  left_join(
    county_counts,
    by = c("NAME" = "project_county", "state_name" = "first_state")
  ) %>%
  mutate(n_projects = replace_na(n_projects, 0))


# --- MAP 1: State Choropleth ---
cat("Creating state choropleth map...\n")

fig_state_choropleth <- ggplot(state_map_data) +
  geom_sf(aes(fill = n_projects), color = "white", size = 0.2) +
  scale_fill_gradient(
    low = "#deebf7",
    high = "#08519c",
    name = "Number of\nProjects",
    labels = scales::comma,
    trans = "sqrt"
  ) +
  labs(
    title = "Clean Energy Projects by State",
    subtitle = paste0("Total: ", scales::comma(sum(state_counts$n_projects)), " project-state pairs"),
    caption = paste0(
      "Note: Projects spanning multiple states are counted in each state.\n",
      "Data source: NEPAccess database. Includes EA, EIS, and CE documents."
    )
  ) +
  theme_void() +
  theme(
    legend.position = "right",
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 10, color = "gray40"),
    plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
  )

fig_state_choropleth

ggsave(
  filename = here(maps_dir, "09_state_choropleth.png"),
  plot = fig_state_choropleth,
  width = 12,
  height = 8,
  units = "in",
  dpi = 300
)
cat("  Saved: maps/09_state_choropleth.png\n")


# --- MAP 2: County Choropleth ---
cat("Creating county choropleth map...\n")

fig_county_choropleth <- ggplot(county_map_data) +
  geom_sf(aes(fill = n_projects), color = NA) +
  geom_sf(data = state_map_data, fill = NA, color = "white", size = 0.3) +
  scale_fill_gradient(
    low = "#deebf7",
    high = "#08519c",
    name = "Number of\nProjects",
    labels = scales::comma,
    trans = "sqrt"
  ) +
  labs(
    title = "Clean Energy Projects by County",
    subtitle = paste0(scales::comma(sum(county_counts$n_projects)), " project-county pairs shown"),
    caption = paste0(
      "Note: County data available for ", pct_with_county, "% of clean energy projects ",
      "(", scales::comma(n_missing_county), " projects missing county information).\n",
      "Projects spanning multiple counties are counted in each county."
    )
  ) +
  theme_void() +
  theme(
    legend.position = "right",
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 10, color = "gray40"),
    plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
  )

fig_county_choropleth

  ggsave(
  filename = here(maps_dir, "10_county_choropleth.png"),
  plot = fig_county_choropleth,
  width = 14,
  height = 9,
  units = "in",
  dpi = 300
)
cat("  Saved: maps/10_county_choropleth.png\n")


# --- MAP 3: County Choropleth Maps by Process Type (Individual Maps) ---
cat("Creating individual county choropleth maps by process type...\n")

# Count by county, state, and process type (remove NA process types)
county_process_counts <- county_with_state %>%
  filter(!is.na(process_type)) %>%
  count(project_county, first_state, process_type, name = "n_projects")

# Join with county shapefile for each process type
county_map_by_process <- us_counties %>%
  left_join(
    county_process_counts,
    by = c("NAME" = "project_county", "state_name" = "first_state")
  ) %>%
  filter(!is.na(process_type)) %>%
  mutate(
    n_projects = replace_na(n_projects, 0),
    # Create display variable that shows NA for 0 values (will map to grey)
    n_projects_display = ifelse(n_projects == 0, NA_real_, n_projects)
  )

# Function to create choropleth map for a single process type
create_choropleth_map <- function(process_type_value, process_type_label) {

  # Filter data for this process type
  process_data <- county_map_by_process %>%
    filter(process_type == process_type_value)

  if (nrow(process_data) == 0) {
    return(NULL)
  }

  # Create the map

  fig <- ggplot() +
    # Base layer: all counties in light grey
    geom_sf(
      data = us_counties,
      fill = "grey95",
      color = "white",
      size = 0.1
    ) +
    # County fill by project count
    geom_sf(
      data = process_data,
      aes(fill = n_projects_display),
      color = NA
    ) +
    # State boundaries
    geom_sf(
      data = state_map_data,
      fill = NA,
      color = "grey40",
      size = 0.3
    ) +
    scale_fill_gradient(
      low = "#deebf7",
      high = "#08519c",
      name = "Number of\nProjects",
      labels = scales::comma,
      trans = "sqrt",
      na.value = "grey95"
    ) +
    labs(
      title = paste0("Clean Energy Projects by County: ", process_type_label),
      subtitle = "Grey areas indicate no projects",
      caption = paste0(
        "Note: County data available for ", pct_with_county, "% of clean energy projects."
      )
    ) +
    theme_void() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10, color = "gray40"),
      plot.caption = element_text(size = 8, color = "gray50", hjust = 0),
      legend.position = "right"
    )

  return(fig)
}

# Create and save individual maps for each process type
choropleth_process_types <- list(
  list(code = "CE", label = "Categorical Exclusion (CE)"),
  list(code = "EA", label = "Environmental Assessment (EA)"),
  list(code = "EIS", label = "Environmental Impact Statement (EIS)")
)

for (pt in choropleth_process_types) {
  cat("  Creating choropleth map for", pt$code, "...\n")

  fig <- create_choropleth_map(pt$code, pt$label)

  if (!is.null(fig)) {
    print(fig)

    filename <- paste0("11_county_choropleth_", tolower(pt$code), ".png")
    ggsave(
      filename = here(maps_dir, filename),
      plot = fig,
      width = 12,
      height = 8,
      units = "in",
      dpi = 300
    )
    cat("    Saved: maps/", filename, "\n", sep = "")
  }
}

cat("  Saved individual choropleth maps for CE, EA, and EIS\n")


# --- MAP 4: County Choropleth Maps by Process Type (Individual Jenks Breaks) ---
cat("Creating individual county choropleth maps by process type (Jenks breaks)...\n")

# Load classInt package for Jenks breaks
library(classInt)

# Function to create Jenks map for a single process type
create_jenks_map <- function(process_type_value, process_type_label) {

  # Filter data for this process type
  process_data <- county_map_by_process %>%
    filter(process_type == process_type_value & n_projects > 0)

  if (nrow(process_data) == 0) {
    return(NULL)
  }

  # Get values for this process type only
  values <- process_data$n_projects
  n_unique <- length(unique(values))

  # Calculate Jenks breaks for THIS process type
  n_classes <- min(4, n_unique - 1)
  if (n_classes < 2) n_classes <- 2

  jenks_breaks <- tryCatch({
    breaks <- classIntervals(values, n = n_classes, style = "jenks")
    breaks$brks
  }, error = function(e) {
    # Fallback to quantile breaks
    quantile(values, probs = seq(0, 1, length.out = n_classes + 1))
  })

  # Ensure breaks are unique (can happen with ties in data)
  jenks_breaks <- unique(jenks_breaks)

  # If we only have 2 unique breaks (min and max), create more by using unique values
  if (length(jenks_breaks) < 3) {
    sorted_unique <- sort(unique(values))
    if (length(sorted_unique) >= 3) {
      # Use unique values as breaks
      n_to_use <- min(6, length(sorted_unique))
      indices <- round(seq(1, length(sorted_unique), length.out = n_to_use))
      jenks_breaks <- sorted_unique[indices]
    } else {
      # Very few unique values - just use them all
      jenks_breaks <- c(min(values) - 0.5, sorted_unique + 0.5)
    }
    jenks_breaks <- unique(jenks_breaks)
  }

  # Apply classification first, then calculate labels from actual data
  process_data <- process_data %>%
    mutate(
      jenks_class_num = cut(n_projects, breaks = jenks_breaks,
                            labels = FALSE, include.lowest = TRUE)
    )

  # Create labels based on actual min/max values in each class (no gaps)
  n_classes_actual <- max(process_data$jenks_class_num, na.rm = TRUE)
  labels <- character(n_classes_actual)
  prev_upper <- 0

  for (i in 1:n_classes_actual) {
    class_values <- process_data$n_projects[process_data$jenks_class_num == i]
    if (length(class_values) == 0) next

    min_val <- min(class_values)
    max_val <- max(class_values)

    # Ensure continuous ranges: lower bound is previous upper + 1 (except for first class)
    if (i == 1) {
      lower <- min_val
    } else {
      lower <- prev_upper + 1
    }
    upper <- max_val
    prev_upper <- upper

    if (lower == upper) {
      labels[i] <- as.character(lower)
    } else {
      labels[i] <- paste0(lower, "-", upper)
    }
  }

  # Apply labels to data
  process_data <- process_data %>%
    mutate(
      jenks_class = factor(labels[jenks_class_num], levels = labels, ordered = TRUE)
    )

  # Create the map
  fig <- ggplot() +
    # Base layer: all counties in light grey
    geom_sf(
      data = us_counties,
      fill = "grey95",
      color = "white",
      size = 0.1
    ) +
    # County fill by Jenks classification
    geom_sf(
      data = process_data,
      aes(fill = jenks_class),
      color = NA
    ) +
    # State boundaries
    geom_sf(
      data = state_map_data,
      fill = NA,
      color = "grey40",
      size = 0.3
    ) +
    scale_fill_brewer(
      palette = "Blues",
      name = "Project Count",
      na.value = "grey95",
      drop = FALSE
    ) +
    labs(
      title = paste0("Clean Energy Projects by County: ", process_type_label),
      subtitle = "Jenks natural breaks classification; grey areas indicate no projects",
      caption = paste0(
        "Note: County data available for ", pct_with_county, "% of clean energy projects.\n",
        "Classification uses Jenks natural breaks calculated specifically for this process type."
      )
    ) +
    theme_void() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10, color = "gray40"),
      plot.caption = element_text(size = 8, color = "gray50", hjust = 0),
      legend.position = "right"
    )

  return(fig)
}

# Create and save individual maps for each process type
process_types <- list(
  list(code = "CE", label = "Categorical Exclusion (CE)"),
  list(code = "EA", label = "Environmental Assessment (EA)"),
  list(code = "EIS", label = "Environmental Impact Statement (EIS)")
)

for (pt in process_types) {
  cat("  Creating Jenks map for", pt$code, "...\n")

  fig <- create_jenks_map(pt$code, pt$label)

  if (!is.null(fig)) {
    print(fig)

    filename <- paste0("12_county_jenks_", tolower(pt$code), ".png")
    ggsave(
      filename = here(maps_dir, filename),
      plot = fig,
      width = 12,
      height = 8,
      units = "in",
      dpi = 300
    )
    cat("    Saved: maps/", filename, "\n", sep = "")
  }
}

cat("  Saved individual Jenks maps for CE, EA, and EIS\n")


# --------------------------
# ANALYSIS: COUNTY DATA COVERAGE BY PROCESS TYPE
# --------------------------

cat("\n=== County Data Coverage Analysis ===\n")

# Overall county data availability
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

# Statistical summary
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

# --------------------------
# EA AND EIS MISSING COUNTY DATA ANALYSIS
# --------------------------

cat("\n=== EA Projects Missing County Data ===\n")
ea_missing_county <- clean_energy %>%
  filter(process_type == "EA" & (is.na(project_county) | project_county == "[]")) %>%
  select(project_id, project_title, project_location, project_state, project_county, lead_agency, project_department)

cat("Total EA projects missing county:", nrow(ea_missing_county), "out of",
    sum(clean_energy$process_type == "EA"), "EA projects\n")
cat("Sample of EA projects missing county data:\n")
print(ea_missing_county %>% slice_head(n = 15))

# Look at location field patterns for EA
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

# Look at location field patterns for EIS
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

# Examples of each pattern type for EIS
cat("\n=== Sample EIS Projects by Location Pattern ===\n")
for (loc_type in unique(eis_location_patterns$location_type)) {
  cat("\n", loc_type, ":\n")
  sample <- eis_location_patterns %>%
    filter(location_type == loc_type) %>%
    select(project_title, project_location, project_state) %>%
    slice_head(n = 3)
  print(sample)
}

# Check if coordinates are available for geocoding
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

# Check agency patterns
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
    pct_missing = round(100 * sum(!has_county) / n(), 1)
  ) %>%
  filter(total >= 5) %>%
  arrange(process_type, desc(pct_missing))

print(ea_eis_missing_by_agency)

# --------------------------
# ANALYSIS
# --------------------------

cat("\n=== Location Analysis ===\n")

# States with most EIS (complex projects)
eis_states <- location_data %>%
  filter(process_type == "EIS") %>%
  count(project_state, name = "n_eis") %>%
  arrange(desc(n_eis))

cat("\nTop 10 states by EIS count (most complex projects):\n")
eis_states %>% slice_head(n = 10) %>% print()

# States with highest EIS ratio
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

# Multi-state projects analysis
multi_state <- clean_energy %>%
  mutate(n_states = str_count(project_state, "\\|") + 1) %>%
  filter(n_states > 1)

cat("\nMulti-state projects:", nrow(multi_state), "\n")
cat("(Projects spanning multiple states)\n")


# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Location Script Complete ===\n")
cat("Tables saved to:", tables_dir, "\n")
cat("Figures saved to:", figures_dir, "\n")

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
