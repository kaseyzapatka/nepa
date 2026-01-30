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

# Create multi-department dataframe
multi_department_data <- 
  clean_energy |> 
  filter(project_multi_department) |> 
  glimpse()  # 21


# --------------------------
# EXPLORATORY
# --------------------------

# share that needs geocoding
clean_energy |> 
  filter(project_multi_state) |> 
  select(project_multi_state) |> 
  glimpse() 

# --------------------------
# MULTI-STATE PROJECTS
# --------------------------

#
# Process multi-state data
# ----------------------------------------------------
state_links <- create_crosstab(multi_state_data, "project_state") |> 
  filter(!row_number() == 1) |> # remove DC, Washington which is false multi-state
  print(n = 100)


#
# Table
# ----------------------------------------------------
# Rename for clarity
tbl_state_links <- 
  state_links %>%
  mutate(project_state = map_chr(project_state, ~ paste(fromJSON(.x), collapse = ", "))) |> 
  rename(
    `State connections` = project_state,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS,
    `Categorical Exclusion` = CE
  ) |> 
  print(n = 10)

# Add totals row
tbl_state_links_clean <- add_totals_row(tbl_state_links, "project_state") |> 
  select(-project_state) |> 
  mutate(`State connections` = replace_na(`State connections`, "Total")) |>
  print()

# save
write_csv(tbl_state_links_clean, here(tables_dir, "table_by_state.csv"))


#
# Map
# ----------------------------------------------------

set.seed(43)

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
top_links <- edges_styled %>%
  slice_max(n_projects, n = 10) %>%
  mutate(
    # Position label at midpoint of connection
    label_x = (lon_from + lon_to) / 2,
    label_y = (lat_from + lat_to) / 2
  )

# ---- Plot: Highway-style connection map ----
map_state_links <- ggplot() +
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
    lineend = "butt",
    linejoin = "round"
  ) +
  # Top connection labels with project count
  geom_label(
    data = top_links,
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
    title = "Interstate Clean Energy Project Connections",
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

map_state_links

# Save the figure
ggsave(
  filename = here(maps_dir, "map_state_links.png"),
  plot = map_state_links,
  width = 12,
  height = 8,
  units = "in",
  dpi = 300
)


#
# Process Type Breakdown Bar Chart
# ----------------------------------------------------
cat("\nCreating process type breakdown for multi-state projects...\n")

# Summarize by process type
process_breakdown <- multi_state_data %>%
  count(process_type, name = "n_projects") %>%
  mutate(
    pct = n_projects / sum(n_projects) * 100,
    label = paste0(n_projects, "\n(", round(pct, 1), "%)")
  )

# Create bar chart
fig_process_breakdown <- ggplot(process_breakdown, aes(x = reorder(process_type, -n_projects), y = n_projects)) +
geom_col(fill = catf_dark_blue, width = 0.7) +
  geom_text(aes(label = label), vjust = -0.3, size = 3.5, color = catf_navy) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Multi-State Projects by NEPA Process Type",
    subtitle = paste0("n = ", sum(process_breakdown$n_projects), " multi-state clean energy projects"),
    x = NULL,
    y = "Number of Projects"
  ) +
  theme_catf() +
  theme(
    panel.grid.major.x = element_blank(),
    axis.line.x = element_blank()
  )

fig_process_breakdown

ggsave(
  filename = here(figures_dir, "fig_multistate_process_type.png"),
  plot = fig_process_breakdown,
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)
cat("  Saved: figures/fig_multistate_process_type.png\n")


#
# Top Connections Analysis
# ----------------------------------------------------
cat("\nCreating top corridors analysis...\n")

# Get top 10 state pair connections with project details
top_connections <- create_crosstab(
  multi_state_data,
  "project_state",
  keep_cols = c("project_title", "project_type")
) %>%
  filter(!row_number() == 1) %>%  # remove DC, Washington
  slice_head(n = 10) %>%
  mutate(
    # Parse JSON state list to readable format
    state_pair = map_chr(project_state, ~ paste(fromJSON(.x), collapse = " - ")),
    # Extract distinct project types, sorted by frequency
    project_type = map_chr(project_type, ~ {
      # Remove JSON brackets and quotes, split on commas
      clean_str <- str_replace_all(.x, '\\[|\\]|"', "")
      all_types <- str_split(clean_str, ",\\s*|\\s*\\|\\s*")[[1]] %>%
        str_trim() %>%
        .[. != ""]  # Remove empty strings
      # Count frequency and sort by most common
      type_counts <- table(all_types)
      sorted_types <- names(sort(type_counts, decreasing = TRUE))
      paste(sorted_types, collapse = ", ")
    })
  ) %>%
  select(state_pair, project_type, CE, EA, EIS, Total) |>
  print()

# Create clean table for export
tbl_top_connections <- top_connections %>%
  rename(
    `State Connections` = state_pair,
    `Distinct Project Types` = project_type,
    `Categorical Exclusion` = CE,
    `Environmental Assessment` = EA,
    `Environmental Impact Statement` = EIS
  )

tbl_top_connections %>% print()

# Save
write_csv(tbl_top_connections, here(tables_dir, "table_top_connections.csv"))



#
# Word Cloud of Project Types
# ----------------------------------------------------
cat("\nCreating word cloud of project types...\n")


# Extract and count all project types from multi-state data
project_type_counts <- multi_state_data %>%
  # Parse JSON arrays in project_type column
  mutate(
    types_list = map(project_type, ~ {
      clean_str <- str_replace_all(.x, '\\[|\\]|"', "")
      str_split(clean_str, ",\\s*")[[1]] %>%
        str_trim() %>%
        .[. != ""]
    })
  ) %>%
  unnest(types_list) %>%
  count(types_list, name = "freq", sort = TRUE) %>%
  filter(!is.na(types_list) & types_list != "") %>%
  rename(word = types_list)

# Create word cloud
set.seed(42)
fig_wordcloud <- ggplot(project_type_counts, aes(label = word, size = freq, color = freq)) +
  geom_text_wordcloud_area(
    shape = "square",
    rm_outside = TRUE,
    area_corr = TRUE
  ) +
  scale_size_area(max_size = 60) +
  scale_color_gradientn(colors = c(catf_light_blue, catf_dark_blue, catf_navy)) +
  theme_void()

fig_wordcloud

ggsave(
  filename = here(figures_dir, "fig_project_types_wordcloud.png"),
  plot = fig_wordcloud,
  width = 8,
  height = 5,
  units = "in",
  dpi = 300
)

# --------------------------
# MULTI-DEPARTMENT
# --------------------------

multi_department_data |>
  select(project_department, lead_agency, project_sponsor, project_multi_department) |>
  print(n = 50)

#
# Process data
# ----------------------------------------------------
department_links <- create_crosstab(
  multi_department_data,
  "lead_agency",
  keep_cols = c("project_title", "project_type")
) |>
  print()

#
# Table
# ----------------------------------------------------
tbl_department_links <-
  department_links |>
  mutate(
    lead_agency = map_chr(lead_agency, ~ paste(fromJSON(.x), collapse = ", ")),
    # Extract distinct project types, sorted by frequency
    project_type = map_chr(project_type, ~ {
      # Remove JSON brackets and quotes, split on commas
      clean_str <- str_replace_all(.x, '\\[|\\]|"', "")
      all_types <- str_split(clean_str, ",\\s*|\\s*\\|\\s*")[[1]] %>%
        str_trim() %>%
        .[. != ""]
      # Count frequency and sort by most common
      type_counts <- table(all_types)
      sorted_types <- names(sort(type_counts, decreasing = TRUE))
      paste(sorted_types, collapse = ", ")
    })
  ) |>
  select(-CE, -project_title) |>
  rename(
    `Department connections` = lead_agency,
    `Distinct Project Types` = project_type
  ) |>
  print()

# save
write_csv(tbl_department_links, here(tables_dir, "table_by_department.csv"))




# --------------------------
# MULTI-DEPARTMENT
# --------------------------

multi_department_data |>
  select(project_department, lead_agency, project_sponsor, project_multi_department) |>
  print(n = 50)


# Create multi-department dataframe
multi_agency_data <- 
  clean_energy |> 
  filter(dataset_source != "CE") |> 
  select(lead_agency, project_sponsor) |> 
  slice_sample(n = 100) |> 
  print(n = 100)
  glimpse()
  
  sfilter(project_multi_department) |> 
  glimpse()  # 21

#
# Process data
# ----------------------------------------------------
department_links <- create_crosstab(
  multi_department_data,
  "lead_agency",
  keep_cols = c("project_title", "project_type")
) |>
  print()




# --------------------------
# BUREAU OF LAND MANAGEMENT SPECIFIC QUESTION
# --------------------------
# Geothermal team - association of project review speed with BLM office -- What to know: Different offices offer faster review processing?

# NEED TO ESTABLISH TIMELINE ANALYSIS FIRST 

projects %>%
  filter(str_detect(project_type, regex("geothermal", ignore_case = TRUE))) |> 
  filter(str_detect(lead_agency, regex("land management", ignore_case = TRUE))) |> 
  #select(project_title, dataset_source, project_department, project_sponsor, lead_agency, project_energy_type) |> 
  select(dataset_source, project_department, project_sponsor,lead_agency, project_energy_type) |> 
  slice_sample(n = 20) |> 
  print(n = 20 )
  glimpse()
