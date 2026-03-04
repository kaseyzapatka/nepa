# --------------------------
# DELIVERABLE 1: DECARBONIZTION TECHNOLOGY BY LOCATION
# --------------------------
# Table 3: Decarbonization Technology by State
# Geographic analysis of decarbonization technology projects

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable01", "00_setup.R"))


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
# TABLE 3: BY STATE
# --------------------------

cat("\nCreating Table 3: Decarbonization Technology by Location (State)...\n")

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
    x = "Number of Projects Tagged with Decarbonization Technologies",
    y = NULL,
    title = "Top 20 States for Projects Tagged with Decarbonization Technologies"
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

state_totals <- top15_state_process %>%
  distinct(project_state, total)

fig_state_process <- top15_state_process %>%
  ggplot(aes(x = reorder(project_state, total), y = percent, fill = process_type)) +
  geom_col() +
  geom_text(
    aes(label = ifelse(percent >= 3, paste0(round(percent), "%"), "")),
    position = position_stack(vjust = 0.5),
    size = 3,
    color = "white"
  ) +
  geom_text(
    data = state_totals,
    aes(x = reorder(project_state, total), y = 101, label = scales::comma(total)),
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
    title = "Process Type Distribution by State (Top 15)",
    caption = "Note: Percentage labels below 3% are excluded for readability."
  ) +
  scale_fill_manual(
    values = c("CE" = catf_dark_blue, "EA" = catf_teal, "EIS" = catf_magenta)
  ) +
  scale_y_continuous(labels = function(x) paste0(x, "%"),
                     expand = expansion(mult = c(0, 0.08))) +
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
    "(", pct_with_county, "% of decarbonization technology projects)\n")
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
    title = "Projects Tagged with Decarbonization Technologies by State",
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
    title = "Projects Tagged with Decarbonization Technologies by County",
    subtitle = paste0(scales::comma(sum(county_counts$n_projects)), " project-county pairs shown"),
    caption = paste0(
      "Note: County data available for ", pct_with_county, "% of projects tagged with decarbonization technologies ",
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
      title = paste0("Projects Tagged with Decarbonization Technologies by County: ", process_type_label),
      subtitle = "Grey areas indicate no projects",
      caption = paste0(
        "Note: County data available for ", pct_with_county, "% of projects tagged with decarbonization technologies."
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
      title = paste0("Projects Tagged with Decarbonization Technologies by County: ", process_type_label),
      subtitle = "Jenks natural breaks classification; grey areas indicate no projects",
      caption = paste0(
        "Note: County data available for ", pct_with_county, "% of projects tagged with decarbonization technologies.\n",
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
# DEEP DIVE: TOP COUNTIES BY PROCESS TYPE
# --------------------------
cat("\n=== Creating Deep Dive Tables for Top Counties ===\n")

# State name to abbreviation mapping
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

# Helper function to clean project type labels
clean_project_type <- function(x) {
  # Parse JSON if needed
  if (is.character(x) && grepl("^\\[", x)) {
    types <- tryCatch(fromJSON(x), error = function(e) x)
  } else if (is.list(x)) {
    types <- unlist(x)
  } else {
    types <- x
  }

  # Clean each type
  cleaned <- sapply(types, function(t) {
    # Apply cleanups
    t <- str_replace(t, "^Renewable Energy Production - ", "")
    t <- str_replace(t, "Utilities \\(electricity, gas, telecommunications\\)", "Utilities")
    t <- str_replace(t, "Carbon Capture and Sequestration", "Carbon Capture")
    return(t)
  })

  # Return as comma-separated string (remove duplicates)
  paste(unique(cleaned), collapse = ", ")
}

# Get county counts by process type
county_process_summary <- county_with_state %>%
  count(project_county, first_state, process_type, name = "n_projects") %>%
  arrange(process_type, desc(n_projects))

# Function to get top N counties for a process type
get_top_counties <- function(process, n = 10) {
  county_process_summary %>%
    filter(process_type == process) %>%
    slice_head(n = n) %>%
    select(project_county, first_state, n_projects)
}

# Get top 10 counties for each process type
top_ce_counties <- get_top_counties("CE", 10)
top_ea_counties <- get_top_counties("EA", 10)
top_eis_counties <- get_top_counties("EIS", 10)

cat("Top 10 CE counties:\n")
print(top_ce_counties)
cat("\nTop 10 EA counties:\n")
print(top_ea_counties)
cat("\nTop 10 EIS counties:\n")
print(top_eis_counties)

# Function to create project table for specific counties and process type
create_county_project_table <- function(counties_df, process_type_filter) {

  # Get projects in these counties with the specified process type
  projects_in_counties <- county_with_state %>%
    filter(process_type == process_type_filter) %>%
    inner_join(
      counties_df,
      by = c("project_county" = "project_county", "first_state" = "first_state")
    ) %>%
    select(project_id, project_title, project_type, project_county, first_state) %>%
    distinct() %>%
    mutate(
      # Clean project type labels
      project_type_clean = sapply(project_type, clean_project_type),
      # Create county/state label with state abbreviation
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

# --------------------------
# DEEP DIVE FIGURES: Technology breakdown for top 10 counties
# --------------------------
cat("\n=== Creating Deep Dive Figures ===\n")

# Function to create technology breakdown for top counties
create_tech_breakdown <- function(counties_df, process_type_filter) {

  # Get projects and explode technology types
  tech_counts <- county_with_state %>%
    filter(process_type == process_type_filter) %>%
    inner_join(
      counties_df,
      by = c("project_county" = "project_county", "first_state" = "first_state")
    ) %>%
    select(project_id, project_type) %>%
    distinct() %>%
    mutate(
      # Parse project_type JSON
      tech_list = map(project_type, ~ {
        if (is.character(.x) && grepl("^\\[", .x)) {
          tryCatch(fromJSON(.x), error = function(e) .x)
        } else if (is.list(.x)) {
          unlist(.x)
        } else {
          .x
        }
      })
    ) %>%
    unnest(tech_list) %>%
    # Clean technology names
    mutate(
      technology = tech_list,
      technology = str_replace(technology, "^Renewable Energy Production - ", ""),
      technology = str_replace(technology, "Utilities \\(electricity, gas, telecommunications\\)", "Utilities"),
      technology = str_replace(technology, "Carbon Capture and Sequestration", "Carbon Capture")
    ) %>%
    count(technology, name = "n_projects") %>%
    arrange(desc(n_projects))

  return(tech_counts)
}

# Create technology breakdowns
ce_tech_breakdown <- create_tech_breakdown(top_ce_counties, "CE")
ea_tech_breakdown <- create_tech_breakdown(top_ea_counties, "EA")
eis_tech_breakdown <- create_tech_breakdown(top_eis_counties, "EIS")

# Function to create bar chart with filtering and custom x-axis breaks
create_tech_bar_chart <- function(tech_df, process_label, fill_color,
                                   min_count = 0, x_break = 10) {
  # Filter by minimum count
  filtered_df <- tech_df %>% filter(n_projects > min_count)

  # Calculate max for x-axis breaks
  max_val <- max(filtered_df$n_projects, na.rm = TRUE)
  x_breaks <- seq(0, ceiling(max_val / x_break) * x_break, by = x_break)

  # Create caption based on filtering
  caption_text <- if (min_count > 0) {
    paste0("Note: Technologies with ", min_count, " or fewer projects excluded for readability.")
  } else {
    NULL
  }

  ggplot(filtered_df, aes(x = n_projects, y = reorder(technology, n_projects))) +
    geom_col(fill = fill_color) +
    geom_text(aes(label = scales::comma(n_projects)), hjust = -0.1, size = 3) +
    labs(
      x = "Number of Projects",
      y = NULL,
      title = paste0("Technology Distribution: Top 10 ", process_label, " Counties"),
      caption = caption_text
    ) +
    scale_x_continuous(
      expand = expansion(mult = c(0, 0.15)),
      breaks = x_breaks
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 9),
      plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
    )
}

# Create and save figures with filtering and custom x-axis breaks
# CE: filter > 10, x-axis breaks every 100
# EA: filter > 1, x-axis breaks every 10
# EIS: filter > 1, x-axis breaks every 10
fig_ce_tech <- create_tech_bar_chart(ce_tech_breakdown, "CE", catf_dark_blue,
                                     min_count = 10, x_break = 200)
fig_ea_tech <- create_tech_bar_chart(ea_tech_breakdown, "EA", catf_teal,
                                     min_count = 1, x_break = 10)
fig_eis_tech <- create_tech_bar_chart(eis_tech_breakdown, "EIS", catf_magenta,
                                      min_count = 1, x_break = 10)

ggsave(
  filename = here(figures_dir, "13_deep_dive_ce_tech.png"),
  plot = fig_ce_tech,
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)
cat("  Saved: 13_deep_dive_ce_tech.png\n")

ggsave(
  filename = here(figures_dir, "13_deep_dive_ea_tech.png"),
  plot = fig_ea_tech,
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)
cat("  Saved: 13_deep_dive_ea_tech.png\n")

ggsave(
  filename = here(figures_dir, "13_deep_dive_eis_tech.png"),
  plot = fig_eis_tech,
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)
cat("  Saved: 13_deep_dive_eis_tech.png\n")

# Create sample tables for report (random 20 from top 2 counties only)
top_2_ce <- top_ce_counties %>% slice_head(n = 2)
top_2_ea <- top_ea_counties %>% slice_head(n = 2)
top_2_eis <- top_eis_counties %>% slice_head(n = 2)

set.seed(42)  # For reproducibility
table_ce_sample <- create_county_project_table(top_2_ce, "CE") %>% slice_sample(n = min(20, nrow(.)))
table_ea_sample <- create_county_project_table(top_2_ea, "EA") %>% slice_sample(n = min(20, nrow(.)))
table_eis_sample <- create_county_project_table(top_2_eis, "EIS") %>% slice_sample(n = min(20, nrow(.)))

# Save sample tables for the report
write_csv(table_ce_sample, here(tables_dir, "deep_dive_ce_sample.csv"))
write_csv(table_ea_sample, here(tables_dir, "deep_dive_ea_sample.csv"))
write_csv(table_eis_sample, here(tables_dir, "deep_dive_eis_sample.csv"))
cat("  Saved sample tables for report\n")


# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Location Script Complete ===\n")
cat("Tables saved to:", tables_dir, "\n")
cat("Figures saved to:", figures_dir, "\n")
