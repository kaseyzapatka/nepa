# --------------------------
# DELIVERABLE 4: MUTLI-STATE, MULTI-DEPARTMENT
# --------------------------
# Geographic analysis of clean energy projects with across multiple states and departments (agencies)

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable4", "00_setup.R"))


# --------------------------
# PROCESS
# --------------------------

# Create multi-state dataframe
multi_state_data <- 
  clean_energy |> 
  filter(project_multi_state) |> 
  glimpse()  # 858

# Create multi-county dataframe
multi_county_data <- 
  clean_energy |> 
  filter(project_multi_county) |> 
  glimpse()  # 335

# Create multi-department dataframe
multi_department_data <- 
  clean_energy |> 
  filter(project_multi_county) |> 
  glimpse()  # 335


# --------------------------
# EXPLORATORY
# --------------------------

# share that needs geocoding
clean_energy |> 
  filter(project_multi_state) |> 
  select(project_multi_state) |> 
  glimpse() 

# --------------------------
# TABLE 3: BY STATE
# --------------------------

cat("\nCreating Table 3: Clean Energy by Location (State)...\n")

table <- create_crosstab(multi_state_data, "project_state")

table |> print(n = 100)

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
# MAP: STATE FLOW MAP
# --------------------------



state_links <- 
  table |> 
  # filter out DC
  filter(Total != 113) |> 
  glimpse()


#
# OLD MAP
# ----------------------------------------------------

library(sf)
library(tidyverse)
library(tidycensus)
library(jsonlite)

set.seed(42)

# ---- State geometries ----
states_sf <- tigris::states(cb = TRUE, year = 2022) %>%
  filter(!NAME %in% c("Alaska", "Hawaii", "Puerto Rico")) %>%
  select(state_name = NAME, geometry)

state_centroids <- states_sf %>%
  st_centroid() %>%
  mutate(
    lon = st_coordinates(.)[, 1],
    lat = st_coordinates(.)[, 2]
  ) %>%
  st_drop_geometry()

# ---- Build weighted interstate connections ----
edges_df <- state_links %>%                     # your data object
  mutate(state_list = map(project_state, fromJSON)) %>%
  filter(Total >= 10) %>%                       # threshold
  unnest(state_list) %>%
  select(project_state, state_list, Total) %>%
  left_join(state_centroids,
            by = c("state_list" = "state_name")) %>%
  group_by(project_state) %>%
  arrange(lat, .by_group = TRUE) %>%            # south → north ordering
  summarise(
    ordered_states = list(state_list),
    Total = first(Total),
    .groups = "drop"
  ) %>%
  mutate(
    pairs = map(
      ordered_states,
      ~ tibble(from = .x[-length(.x)], to = .x[-1])
    )
  ) %>%
  unnest(pairs) %>%
  group_by(from, to) %>%
  summarise(
    n_projects = sum(Total),                     # 🔑 WEIGHTED COUNT
    .groups = "drop"
  ) %>%
  left_join(state_centroids,
            by = c("from" = "state_name")) %>%
  rename(lon_from = lon, lat_from = lat) %>%
  left_join(state_centroids,
            by = c("to" = "state_name")) %>%
  rename(lon_to = lon, lat_to = lat) %>%
  filter(
    between(lon_from, -130, -66),
    between(lon_to,   -130, -66),
    between(lat_from,  24, 50),
    between(lat_to,    24, 50)
  ) %>%
  mutate(
    lon_from = lon_from + rnorm(n(), 0, 0.35),
    lat_from = lat_from + rnorm(n(), 0, 0.35),
    lon_to   = lon_to   + rnorm(n(), 0, 0.35),
    lat_to   = lat_to   + rnorm(n(), 0, 0.35)
  )

# ---- Plot ----
ggplot() +
  geom_sf(
    data = states_sf,
    fill = "antiquewhite",
    color = "white",
    linewidth = 0.2
  ) +
  geom_curve(
    data = edges_df,
    aes(
      x = lon_from,
      y = lat_from,
      xend = lon_to,
      yend = lat_to,
      size = n_projects
    ),
    curvature = 0.25,
    color = "steelblue",
    alpha = 0.7,
    lineend = "round"
  ) +
  scale_size_continuous(
    range = c(0.4, 4.5),
    breaks = c(25, 50, 100, 250, 500),
    name = "Number of Projects"
  ) +
  coord_sf(
    xlim = c(-125, -66),
    ylim = c(24, 50),
    expand = FALSE
  ) +
  theme_void() +
  labs(
    title = "Major Interstate Project Corridors",
    subtitle = "Connections weighted by total number of shared projects"
  )



#
# UPDATED MAP
# ----------------------------------------------------


library(sf)
library(tidyverse)
library(jsonlite)
library(tigris)
options(tigris_use_cache = TRUE)

set.seed(42)

# ---- State geometries ----
states_sf <- tigris::states(cb = TRUE, year = 2022) %>%
  filter(!NAME %in% c("Alaska", "Hawaii", "Puerto Rico")) %>%
  select(state_name = NAME, geometry) %>%
  st_transform(4326)  # WGS84 for lat/lon coordinates

state_centroids <- states_sf %>%
  st_centroid() %>%
  mutate(
    lon = st_coordinates(.)[, 1],
    lat = st_coordinates(.)[, 2]
  ) %>%
  st_drop_geometry()

# ---- Build weighted interstate connections ----
edges_df <- state_links %>%                     # your data object
  mutate(state_list = map(project_state, fromJSON)) %>%
  filter(Total >= 10) %>%                       # threshold
  unnest(state_list) %>%
  select(project_state, state_list, Total) %>%
  left_join(state_centroids,
            by = c("state_list" = "state_name")) %>%
  group_by(project_state) %>%
  arrange(lat, .by_group = TRUE) %>%            # south → north ordering
  summarise(
    ordered_states = list(state_list),
    Total = first(Total),
    .groups = "drop"
  ) %>%
  mutate(
    pairs = map(
      ordered_states,
      ~ tibble(from = .x[-length(.x)], to = .x[-1])
    )
  ) %>%
  unnest(pairs) %>%
  group_by(from, to) %>%
  summarise(
    n_projects = sum(Total),                     # 🔑 WEIGHTED COUNT
    .groups = "drop"
  ) %>%
  left_join(state_centroids,
            by = c("from" = "state_name")) %>%
  rename(lon_from = lon, lat_from = lat) %>%
  left_join(state_centroids,
            by = c("to" = "state_name")) %>%
  rename(lon_to = lon, lat_to = lat) %>%
  filter(
    between(lon_from, -130, -66),
    between(lon_to,   -130, -66),
    between(lat_from,  24, 50),
    between(lat_to,    24, 50)
  ) %>%
  mutate(
    lon_from = lon_from + rnorm(n(), 0, 0.35),
    lat_from = lat_from + rnorm(n(), 0, 0.35),
    lon_to   = lon_to   + rnorm(n(), 0, 0.35),
    lat_to   = lat_to   + rnorm(n(), 0, 0.35)
  )

# ---- Prepare edges with tiers for visual hierarchy ----
# Rebuild edges without jitter for cleaner highway-style lines
edges_styled <- state_links %>%

  mutate(state_list = map(project_state, fromJSON)) %>%
  filter(Total >= 10) %>%
  unnest(state_list) %>%
  select(project_state, state_list, Total) %>%
  left_join(state_centroids, by = c("state_list" = "state_name")) %>%
  group_by(project_state) %>%
  arrange(lat, .by_group = TRUE) %>%
  summarise(
    ordered_states = list(state_list),
    Total = first(Total),
    .groups = "drop"
  ) %>%
  mutate(
    pairs = map(ordered_states, ~ tibble(from = .x[-length(.x)], to = .x[-1]))
  ) %>%
  unnest(pairs) %>%
  group_by(from, to) %>%
  summarise(n_projects = sum(Total), .groups = "drop") %>%
  # Join centroids for from/to states (no jitter)
  left_join(state_centroids, by = c("from" = "state_name")) %>%
  rename(lon_from = lon, lat_from = lat) %>%
  left_join(state_centroids, by = c("to" = "state_name")) %>%
  rename(lon_to = lon, lat_to = lat) %>%
  filter(
    !is.na(lon_from), !is.na(lon_to),
    between(lon_from, -130, -66),
    between(lon_to, -130, -66),
    between(lat_from, 24, 50),
    between(lat_to, 24, 50)
  ) %>%
  # Create tiers for visual hierarchy
  mutate(
    tier = case_when(
      n_projects >= quantile(n_projects, 0.9) ~ "top",
      n_projects >= quantile(n_projects, 0.7) ~ "high",
      n_projects >= quantile(n_projects, 0.4) ~ "medium",
      TRUE ~ "low"
    ),
    tier = factor(tier, levels = c("low", "medium", "high", "top"))
  )

# Get top N connections for labeling
top_connections <- edges_styled %>%
  slice_max(n_projects, n = 8) %>%
  mutate(
    # Position label at midpoint of connection
    label_x = (lon_from + lon_to) / 2,
    label_y = (lat_from + lat_to) / 2
  )

# ---- Plot: Highway-style connection map ----
fig_state_connections <- ggplot() +
  # Base map - subtle gray states
  geom_sf(
    data = states_sf,
    fill = "gray97",
    color = "gray80",
    linewidth = 0.3
  ) +
  # Lower-tier connections (background)
  geom_segment(
    data = edges_styled %>% filter(tier %in% c("low", "medium")),
    aes(
      x = lon_from, y = lat_from,
      xend = lon_to, yend = lat_to,
      linewidth = n_projects
    ),
    color = catf_light_blue,
    alpha = 0.5,
    lineend = "round"
  ) +
  # Higher-tier connections (foreground)
  geom_segment(
    data = edges_styled %>% filter(tier %in% c("high", "top")),
    aes(
      x = lon_from, y = lat_from,
      xend = lon_to, yend = lat_to,
      linewidth = n_projects
    ),
    color = catf_dark_blue,
    alpha = 0.85,
    lineend = "round"
  ) +
  # Top connection labels with project count
  geom_label(
    data = top_connections,
    aes(x = label_x, y = label_y, label = n_projects),
    size = 2.8,
    fontface = "bold",
    fill = "white",
    color = catf_navy,
    label.size = 0.3,
    label.padding = unit(0.15, "lines"),
    label.r = unit(0.1, "lines")
  ) +
  scale_linewidth_continuous(
    range = c(0.3, 4),
    guide = "none"  # No legend - visual hierarchy speaks for itself
  ) +
  coord_sf(
    xlim = c(-125, -66),
    ylim = c(24, 50),
    expand = FALSE
  ) +
  labs(
    title = "Interstate Clean Energy Project Corridors",
    subtitle = "Line thickness reflects number of shared projects; labels show top corridor counts",
    caption = "Connections shown for state pairs with 10+ shared projects"
  ) +
  theme_void() +
  theme(
    plot.title = element_text(
      size = 14, face = "bold", color = catf_navy,
      margin = margin(b = 5)
    ),
    plot.subtitle = element_text(
      size = 10, color = catf_dark_blue,
      margin = margin(b = 10)
    ),
    plot.caption = element_text(
      size = 8, color = "gray50", hjust = 0,
      margin = margin(t = 10)
    ),
    plot.margin = margin(10, 10, 10, 10)
  )

fig_state_connections

# Save the figure
ggsave(
  filename = here(maps_dir, "12_state_connections_highway.png"),
  plot = fig_state_connections,
  width = 12,
  height = 8,
  units = "in",
  dpi = 300
)
cat("  Saved: maps/12_state_connections_highway.png\n")


state_links |> 
  select(!CE:EA) |> 
  print(n = 100)





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
maps_dir <- here("output", "deliverable4", "maps")
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
  # Base map
  geom_sf(
    data = state_map_data,
    fill = "gray95",
    color = "gray70",
    size = 0.3
  ) +
  # County centroids, sized by project count
  geom_sf(
    data = centroid_data,
    aes(size = n_projects),
    color = catf_dark_blue,
    alpha = 0.5
  ) +
  scale_size_continuous(
    range = c(0.3, 5),
    guide = "none"
  ) +
  facet_wrap(
    ~ process_type,
    ncol = 1,
    labeller = as_labeller(c(
      CE  = "Categorical Exclusion",
      EA  = "Environmental Assessment",
      EIS = "Environmental Impact Statement"
    ))
  ) +
  labs(
    title = "Clean Energy Projects by County and NEPA Process Type",
    subtitle = "Each panel shows county-level project concentration by process type; dot size reflects project count",
    caption = paste0(
      "Note: County data available for ", pct_with_county, "% of clean energy projects ",
      "(", scales::comma(n_missing_county), " projects missing county information).\n",
      "Dots placed at county centroids."
    )
  ) +
  theme_void() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 10, color = "gray40"),
    plot.caption = element_text(size = 8, color = "gray50", hjust = 0)
  )
fig_county_dots

# save
ggsave(
  filename = here(maps_dir, "11_county_dots_process_type.png"),
  plot = fig_county_dots,
  width = 10,
  height = 16,
  units = "in",
  dpi = 300
)




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

