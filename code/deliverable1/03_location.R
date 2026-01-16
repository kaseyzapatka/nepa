# --------------------------
# DELIVERABLE 1: CLEAN ENERGY BY LOCATION
# --------------------------
# Table 3: Clean Energy by State
# Geographic analysis of clean energy projects

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable1", "00_setup.R"))


# --------------------------
# EXPLORATORY ANALYSIS
# --------------------------

# share that needs geocoding
clean_energy |> 
  count(project_location_needs_geocoding) |> 
  glimpse() # 45/26,621

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
cat("  Saved: table3_by_state.csv\n")


# --------------------------
# FIGURES
# --------------------------

# Figure 1: Top 20 States Bar Chart
top_states <- state_counts %>%
  slice_head(n = 20)

fig_top_states <- top_states %>%
  ggplot(aes(x = n_projects, y = reorder(project_state, n_projects))) +
  geom_col(fill = "darkorange") +
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
  coord_flip() +
  labs(
    x = NULL,
    y = "Percent of Projects",
    fill = "Process Type",
    title = "Process Type Distribution by State (Top 15)"
  ) +
  scale_fill_brewer(palette = "Set1") +
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


# --- MAP 3: County Centroid Dot Map by Process Type ---
cat("Creating county centroid dot map by process type...\n")

# Get county centroids
county_centroids <- us_counties %>%
  st_centroid() %>%
  select(NAME, state_name, geometry)

# Count by county, state, and process type
county_process_counts <- county_with_state %>%
  count(project_county, first_state, process_type, name = "n_projects")

# Join with centroids
centroid_data <- county_centroids %>%
  inner_join(
    county_process_counts,
    by = c("NAME" = "project_county", "state_name" = "first_state")
  )

# Create dot map - size scaled to count, color by process type
fig_county_dots <- ggplot() +
  # Base map (states outline)
  geom_sf(data = state_map_data, fill = "gray95", color = "gray70", size = 0.3) +
  # Dots at county centroids - sized by project count
  geom_sf(
    data = centroid_data,
    aes(size = n_projects, color = process_type),
    alpha = 0.5
  ) +
  scale_size_continuous(
    range = c(0.3, 5),
    guide = "none"  # Remove size legend - density shows concentration
  ) +
  scale_color_manual(
    name = "Process Type",
    values = c("CE" = "#e41a1c", "EA" = "#377eb8", "EIS" = "#4daf4a"),
    labels = c("CE" = "Categorical Exclusion", "EA" = "Environmental Assessment", "EIS" = "Environmental Impact Statement")
  ) +
  labs(
    title = "Clean Energy Projects by County and Process Type",
    subtitle = "Dot size reflects project count; overlapping dots indicate higher concentration",
    caption = paste0(
      "Note: County data available for ", pct_with_county, "% of clean energy projects ",
      "(", scales::comma(n_missing_county), " projects missing county information).\n",
      "Dots placed at county centroids. Larger/denser areas indicate more projects."
    )
  ) +
  theme_void() +
  theme(
    legend.position = "right",
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 10, color = "gray40"),
    plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
  ) +
  guides(
    color = guide_legend(override.aes = list(size = 4, alpha = 1))
  )

fig_county_dots

ggsave(
  filename = here(maps_dir, "11_county_dots_process_type.png"),
  plot = fig_county_dots,
  width = 14,
  height = 9,
  units = "in",
  dpi = 300
)
cat("  Saved: maps/11_county_dots_process_type.png\n")


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
