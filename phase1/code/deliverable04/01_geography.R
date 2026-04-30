# --------------------------
# DELIVERABLE 4: MUTLI-STATE, MULTI-DEPARTMENT
# --------------------------
# Geographic analysis of decarbonization technology projects with across multiple states and departments (agencies)

# --------------------------
# SETUP
# --------------------------

source(here::here("phase1", "code", "deliverable04", "00_setup.R"))

# --------------------------
# PROCESS
# --------------------------

# Create multi-state dataframe
multi_state_data <- 
  clean_energy |> 
  filter(project_multi_state) |> 
  glimpse()  # 858

# Create multi-agency dataframe (metadata OR coagency high-confidence text signal)
multi_department_data <-
  clean_energy_multiagency |>
  filter(project_multi_agency) |>
  glimpse()

# Keep metadata-only subset for reference / QA
multi_department_metadata_only <-
  clean_energy_multiagency |>
  filter(project_multi_department)

# Expanded-only projects: not strict metadata multi-department, but high-confidence
# coagency text signal detected.
multi_agency_expanded_only <-
  clean_energy_multiagency |>
  filter(!project_multi_department, project_has_coagency_signal_high_conf)

# Summary table used in report (strict vs expanded definitions)
tbl_multi_agency_summary <- tibble(
  Category = c(
    "Multi-state projects",
    "Multi-department projects"
  ),
  Count = c(
    nrow(multi_state_data),
    nrow(multi_department_data)
  )
)

write_csv(tbl_multi_agency_summary, here(tables_dir, "table_multi_agency_summary.csv"))

# --------------------------
# SHARED HELPERS
# --------------------------

parse_jsonish_vector <- function(x) {
  if (is.null(x) || is.na(x) || x == "") return(character(0))
  if (is.character(x) && str_detect(x, "^\\[")) {
    parsed <- tryCatch(fromJSON(x), error = function(e) NULL)
    if (!is.null(parsed) && length(parsed) > 0) return(str_trim(as.character(parsed)))
  }
  vals_pipe <- str_split(as.character(x), "\\s*\\|\\s*")[[1]] %>%
    str_trim() %>%
    .[. != ""]
  if (length(vals_pipe) == 0) return(character(0))

  vals <- map(vals_pipe, ~ {
    token <- .x
    if (str_detect(token, "^\\[")) {
      parsed_token <- tryCatch(fromJSON(token), error = function(e) NULL)
      if (!is.null(parsed_token) && length(parsed_token) > 0) {
        return(str_trim(as.character(parsed_token)))
      }
    }
    if (str_detect(token, ",")) {
      return(str_split(token, ",\\s*")[[1]] %>% str_trim() %>% .[. != ""])
    }
    token
  }) %>%
    unlist(use.names = FALSE) %>%
    str_trim()

  vals[vals != ""]
}

derive_clean_type_bucket <- function(project_type_text) {
  txt <- str_to_lower(coalesce(project_type_text, ""))
  case_when(
    str_detect(txt, "solar") ~ "Solar",
    str_detect(txt, "wind") ~ "Wind",
    str_detect(txt, "geothermal") ~ "Geothermal",
    str_detect(txt, "electricity transmission|utilities") ~ "Transmission & Grid",
    str_detect(txt, "energy storage") ~ "Energy Storage",
    str_detect(txt, "hydro|hydrokinetic") ~ "Hydro",
    str_detect(txt, "nuclear") ~ "Nuclear",
    TRUE ~ "Other Clean"
  )
}

build_wordcloud_counts <- function(df) {
  df %>%
    mutate(types_list = map(project_type, parse_jsonish_vector)) %>%
    unnest(types_list) %>%
    filter(!is.na(types_list), types_list != "") %>%
    count(types_list, name = "freq", sort = TRUE) %>%
    rename(word = types_list)
}

generate_wordcloud_panels <- function(df, figure_prefix, panel_table_name) {
  process_levels <- c("CE", "EA", "EIS")

  panel_meta <- tibble(
    panel_index = seq_along(process_levels),
    process_type = process_levels,
    n_projects = map_int(process_levels, ~ sum(df$process_type == .x, na.rm = TRUE)),
    figure_file = paste0(figure_prefix, "_", str_to_lower(process_levels), ".png")
  )

  walk2(panel_meta$process_type, panel_meta$figure_file, ~ {
    panel_df <- df %>% filter(process_type == .x)
    panel_counts <- build_wordcloud_counts(panel_df)
    if (nrow(panel_counts) == 0) {
      panel_counts <- tibble(word = paste("No", .x, "projects"), freq = 1)
    }

    fig_panel <- ggplot(panel_counts, aes(label = word, size = freq, color = freq)) +
      geom_text_wordcloud_area(
        shape = "square",
        rm_outside = TRUE,
        area_corr = TRUE
      ) +
      scale_size_area(max_size = 70) +
      scale_color_gradientn(colors = c(catf_light_blue, catf_dark_blue, catf_navy)) +
      theme_void()

    ggsave(
      filename = here(figures_dir, .y),
      plot = fig_panel,
      width = 10,
      height = 6,
      units = "in",
      dpi = 300
    )
  })

  write_csv(panel_meta, here(tables_dir, panel_table_name))
  panel_meta
}


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
    title = "Interstate Decarbonization Technology Project Connections",
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
fig_process_breakdown <- ggplot(process_breakdown, aes(x = factor(process_type, levels = c("CE", "EA", "EIS")), y = n_projects)) +
geom_col(fill = catf_dark_blue, width = 0.7) +
  geom_text(aes(label = label), vjust = -0.3, size = 3.5, color = catf_navy) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Multi-State Projects by NEPA Process Type",
    subtitle = paste0("n = ", sum(process_breakdown$n_projects), " multi-state decarbonization technology projects"),
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
# NOTE: "Top connections" = the 10 state-pair combinations with the highest
# total project count. create_crosstab() groups multi-state projects by their
# full state combination (e.g., "California, Nevada"), counts projects per
# process type, sums to a Total column, and sorts descending. slice_head(n=10)
# then keeps only the 10 highest-volume state-pair corridors.
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
# Word Clouds of Project Types by NEPA Process Type
# ----------------------------------------------------
cat("\nCreating process-type word cloud panels for multi-state projects...\n")

tbl_multistate_wordcloud_panels <- generate_wordcloud_panels(
  df = multi_state_data,
  figure_prefix = "fig_multistate_project_types_wordcloud",
  panel_table_name = "table_multistate_wordcloud_panels.csv"
)
print(tbl_multistate_wordcloud_panels)


#
# Sample of Complex Multi-State Projects
# ----------------------------------------------------
cat("\nCreating sample table of complex multi-state projects...\n")

max_state_spread_km <- function(state_vec, centroid_tbl) {
  coords <- centroid_tbl %>%
    filter(state_name %in% state_vec) %>%
    select(lon, lat)
  if (nrow(coords) < 2) return(0)
  max(as.matrix(dist(coords))) * 111
}

multistate_complex_ranked <- multi_state_data %>%
  mutate(
    state_list = map(project_state, parse_jsonish_vector),
    state_count = lengths(state_list),
    has_non_contiguous_state = map_lgl(
      state_list,
      ~ any(
        .x %in% c(
          "Alaska", "Hawaii", "Puerto Rico", "Guam",
          "American Samoa", "Northern Mariana Islands", "U.S. Virgin Islands"
        )
      )
    ),
    spread_km = map_dbl(state_list, max_state_spread_km, centroid_tbl = state_centroids),
    project_types = map_chr(project_type, ~ paste(parse_jsonish_vector(.x), collapse = ", ")),
    state_footprint = map_chr(state_list, ~ paste(.x, collapse = ", "))
  ) %>%
  arrange(desc(has_non_contiguous_state), desc(state_count), desc(spread_km)) %>%
  distinct(project_title, .keep_all = TRUE)

tbl_multistate_complex_sample <- bind_rows(
  multistate_complex_ranked %>% slice_head(n = 5),
  multistate_complex_ranked %>%
    slice(-(1:5)) %>%
    arrange(desc(spread_km), desc(state_count)) %>%
    slice_head(n = 5)
) %>%
  distinct(project_title, .keep_all = TRUE) %>%
  transmute(
    `Project Title` = project_title,
    `State Footprint` = state_footprint,
    `Number of States` = state_count,
    `Project Types` = project_types
  )

write_csv(
  tbl_multistate_complex_sample,
  here(tables_dir, "table_multistate_complex_sample.csv")
)
print(tbl_multistate_complex_sample)

# --------------------------
# MULTI-DEPARTMENT
# --------------------------

multi_department_data |>
  select(any_of(c(
    "project_department",
    "process_type",
    "lead_agency",
    "project_sponsor",
    "project_multi_department",
    "project_has_coagency_signal_high_conf",
    "project_multi_agency",
    "project_coagency_signal_source"
  ))) |>
  print(n = 50)

#
# Process Type Breakdown Figure
# ----------------------------------------------------
cat("\nCreating process type breakdown for multi-agency projects...\n")

process_breakdown_multiagency <- multi_department_data %>%
  count(process_type, name = "n_projects") %>%
  mutate(
    pct = n_projects / sum(n_projects) * 100,
    label = paste0(n_projects, "\n(", round(pct, 1), "%)")
  )

write_csv(
  process_breakdown_multiagency,
  here(tables_dir, "table_multiagency_process_type.csv")
)

fig_process_breakdown_multiagency <- ggplot(
  process_breakdown_multiagency,
  aes(x = factor(process_type, levels = c("CE", "EA", "EIS")), y = n_projects)
) +
  geom_col(fill = catf_teal, width = 0.7) +
  geom_text(aes(label = label), vjust = -0.3, size = 3.5, color = catf_navy) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "Multi-Department Projects by NEPA Process Type",
    subtitle = paste0("n = ", sum(process_breakdown_multiagency$n_projects), " multi-department decarbonization technology projects"),
    x = NULL,
    y = "Number of Projects"
  ) +
  theme_catf() +
  theme(
    panel.grid.major.x = element_blank(),
    axis.line.x = element_blank()
  )

ggsave(
  filename = here(figures_dir, "fig_multiagency_process_type.png"),
  plot = fig_process_breakdown_multiagency,
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

#
# Department Connections Data
# ----------------------------------------------------
# NOTE: Use lead_agency for crosstab since project_department only reflects
# the first department (due to how classify_department works in extract_data.py).
# We then map agencies to departments in the table display.
department_links <- create_crosstab(
  multi_department_metadata_only,
  "lead_agency",
  keep_cols = c("project_title", "project_type")
) |>
  print()
#
# Table
# ----------------------------------------------------
# Helper function to map agency names to departments
map_agency_to_department <- function(agency) {
  case_when(
    str_detect(agency, "^Department of Energy") ~ "Department of Energy",
    str_detect(agency, "^Department of the Interior") ~ "Department of the Interior",
    str_detect(agency, "^Department of Agriculture") ~ "Department of Agriculture",
    str_detect(agency, "^Department of Defense") ~ "Department of Defense",
    str_detect(agency, "^Department of Homeland Security") ~ "Department of Homeland Security",
    str_detect(agency, "^Department of Transportation") ~ "Department of Transportation",
    str_detect(agency, "^Department of Commerce") ~ "Department of Commerce",
    str_detect(agency, "^Major Independent Agencies") ~ "Major Independent Agencies",
    str_detect(agency, "^Other Independent Agencies") ~ "Other Independent Agencies",
    TRUE ~ agency  # Keep original if no match
  )
}

tbl_department_links <-
  department_links |>
  mutate(
    # Parse lead_agency JSON array and map each agency to its department
    department_connections = map_chr(lead_agency, ~ {
      agencies <- parse_jsonish_vector(.x)
      departments <- unique(map_chr(agencies, map_agency_to_department))
      paste(departments, collapse = ", ")
    }),
    # Extract distinct project types, sorted by frequency
    project_type = map_chr(project_type, ~ {
      all_types <- parse_jsonish_vector(.x)
      # Count frequency and sort by most common
      type_counts <- table(all_types)
      sorted_types <- names(sort(type_counts, decreasing = TRUE))
      paste(sorted_types, collapse = ", ")
    })
  ) |>
  select(
    `Department connections` = department_connections,
    `Distinct Project Types` = project_type,
    Total
  ) |>
  print()

# save
write_csv(tbl_department_links, here(tables_dir, "table_by_department.csv"))


#
# Department Collaboration Hubs (Creative Relationship Table)
# ----------------------------------------------------
cat("\nCreating department collaboration hubs table...\n")

department_projects <- multi_department_metadata_only %>%
  mutate(
    department_list = map(lead_agency, ~ {
      agencies <- parse_jsonish_vector(.x)
      depts <- unique(map_chr(agencies, map_agency_to_department))
      sort(depts[depts != ""])
    })
  ) %>%
  filter(lengths(department_list) >= 2)

department_pairs <- department_projects %>%
  transmute(
    project_id,
    department_pairs = map(department_list, ~ {
      combo <- combn(.x, 2, simplify = FALSE)
      tibble(
        department_1 = map_chr(combo, 1),
        department_2 = map_chr(combo, 2)
      )
    })
  ) %>%
  unnest(department_pairs)

# Add reversed pairs so every department appears on both sides of the Sankey
department_pairs <- bind_rows(
  department_pairs,
  department_pairs %>% rename(department_1 = department_2, department_2 = department_1)
)

pair_counts <- department_pairs %>%
  count(department_1, department_2, name = "shared_projects", sort = TRUE)

tbl_department_collaboration_hubs <- bind_rows(
  pair_counts %>%
    transmute(
      department = department_1,
      partner = department_2,
      shared_projects
    ),
  pair_counts %>%
    transmute(
      department = department_2,
      partner = department_1,
      shared_projects
    )
) %>%
  group_by(department) %>%
  summarise(
    `Unique partner departments` = n_distinct(partner),
    `Collaborative project ties` = sum(shared_projects),
    `Most frequent partner` = partner[which.max(shared_projects)],
    `Projects with top partner` = max(shared_projects),
    `Bridge score` = round(`Unique partner departments` * log1p(`Collaborative project ties`), 2),
    .groups = "drop"
  ) %>%
  arrange(desc(`Bridge score`), desc(`Collaborative project ties`))

write_csv(
  tbl_department_collaboration_hubs,
  here(tables_dir, "table_department_collaboration_hubs.csv")
)
print(tbl_department_collaboration_hubs)


# Figure for collaboration hubs
fig_department_collaboration_hubs <- tbl_department_collaboration_hubs %>%
  mutate(department = fct_reorder(department, `Bridge score`)) %>%
  ggplot(aes(x = `Bridge score`, y = department, fill = `Collaborative project ties`)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = `Most frequent partner`),
    hjust = 0,
    nudge_x = 0.15,
    size = 3,
    color = catf_navy
  ) +
  scale_fill_gradientn(colors = c(catf_light_blue, catf_dark_blue, catf_navy)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.45))) +
  labs(
    title = "Department Collaboration Hubs",
    subtitle = "Bar length shows bridge score; labels show most frequent partner",
    x = "Bridge score",
    y = NULL,
    fill = "Collaborative\nproject ties",
    caption = "Bridge score is a normalized measure of cross-department collaboration ties (unique partner departments × log(1 + total shared project ties))."
  ) +
  theme_catf()

ggsave(
  filename = here(figures_dir, "fig_department_collaboration_hubs.png"),
  plot = fig_department_collaboration_hubs,
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

# Sankey/alluvial view of cross-department ties
base_department_sankey <- ggplot(
  pair_counts,
  aes(axis1 = department_1, axis2 = department_2, y = shared_projects)
) +
  ggalluvial::geom_alluvium(
    aes(fill = department_1),
    width = 1 / 10,
    alpha = 0.8
  ) +
  ggalluvial::geom_stratum(
    width = 1 / 8,
    fill = "gray96",
    color = "gray60"
  ) +
  scale_x_discrete(
    limits = c("axis1", "axis2"),
    labels = NULL,
    expand = c(0.03, 0.03)
  ) +
  scale_fill_manual(values = rep(catf_palette, length.out = n_distinct(pair_counts$department_1))) +
  labs(
    title = "Cross-Department Project Flows",
    subtitle = "Departments only (agencies rolled up to department level); flow width reflects shared projects",
    y = NULL,
    x = NULL
  ) +
  theme_catf() +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.line = element_blank()
  )

sankey_label_data <- ggplot_build(base_department_sankey)$data[[2]] %>%
  as_tibble() %>%
  transmute(
    x = x,
    y = y,
    label = str_remove(stratum, "^Department of ") %>%
      str_replace_all(c(
        "^the Interior$"               = "Interior",
        "^Housing and Urban Development$" = "HUD",
        "^General Services Administration$" = "GSA",
        "^Homeland Security$"          = "Homeland\nSecurity",
        "^International Assistance Programs$" = "Assistance"
      )) %>%
      str_wrap(width = 14)
  )

fig_department_sankey <- base_department_sankey +
  geom_text(
    data = sankey_label_data,
    aes(x = x, y = y, label = label),
    hjust = 0.5,
    inherit.aes = FALSE,
    size = 3.5,
    lineheight = 0.9,
    color = "gray20"
  ) +
  theme(
    plot.margin = margin(10, 10, 10, 10)
  )

ggsave(
  filename = here(figures_dir, "fig_department_sankey.png"),
  plot = fig_department_sankey,
  width = 13,
  height = 7,
  units = "in",
  dpi = 300
)


#
# Filtered Sankey: Top Departments Only
# ----------------------------------------------------
# Threshold rationale: compute total shared-project ties per department (summing
# pair_counts as source; symmetric data so this equals total as destination).
# Departments below the threshold are peripheral collaborators — they add visual
# clutter without meaningfully shifting the story. A threshold of 20 retains the
# core agencies that drive clean-energy NEPA collaboration and drops the tail.
cat("\nCreating filtered department Sankey (top departments only)...\n")

DEPT_TOP_N <- 6  # keep the N most collaborative departments

dept_totals <- pair_counts |>
  group_by(department = department_1) |>
  summarise(total_ties = sum(shared_projects), .groups = "drop") |>
  arrange(desc(total_ties))

cat("\nDepartment collaborative project ties (before filtering):\n")
print(dept_totals, n = Inf)

top_depts      <- dept_totals |> slice_head(n = DEPT_TOP_N) |> pull(department)
excluded_df    <- dept_totals |> filter(!department %in% top_depts)
excluded_depts <- excluded_df$department
# Effective floor: the minimum tie count among kept departments (for footnote)
DEPT_THRESHOLD <- dept_totals |> filter(department %in% top_depts) |> pull(total_ties) |> min()

cat(sprintf(
  "\nTop-%d filter (effective floor: %d project ties)\n  Kept (%d): %s\n  Excluded (%d): %s\n",
  DEPT_TOP_N, DEPT_THRESHOLD,
  length(top_depts),    paste(top_depts,     collapse = ", "),
  length(excluded_depts), paste(excluded_depts, collapse = ", ")
))

# Filter to pairs where BOTH departments clear the threshold
pair_counts_filtered <- pair_counts |>
  filter(department_1 %in% top_depts, department_2 %in% top_depts)

# Order strata: largest department (most ties) at top → set factor levels descending
dept_order <- dept_totals |>
  filter(department %in% top_depts) |>
  arrange(desc(total_ties)) |>
  pull(department)

pair_counts_filtered <- pair_counts_filtered |>
  mutate(
    department_1 = factor(department_1, levels = dept_order),
    department_2 = factor(department_2, levels = dept_order)
  )

# Build footnote text
excluded_label <- if (length(excluded_depts) > 0) {
  paste0(
    "Note: Showing top ", length(top_depts), " of ", nrow(dept_totals),
    " departments by collaborative activity (minimum ", DEPT_THRESHOLD,
    " shared project ties). Excluded (", length(excluded_depts), "): ",
    paste(excluded_depts, collapse = "; "), "."
  )
} else {
  "All departments shown."
}

# Build base filtered Sankey
base_department_sankey_filtered <- ggplot(
  pair_counts_filtered,
  aes(axis1 = department_1, axis2 = department_2, y = shared_projects)
) +
  ggalluvial::geom_alluvium(
    aes(fill = department_1),
    width = 1 / 10,
    alpha = 0.8
  ) +
  ggalluvial::geom_stratum(
    width = 1 / 8,
    fill = "gray96",
    color = "gray60"
  ) +
  scale_x_discrete(
    limits = c("axis1", "axis2"),
    labels = NULL,
    expand = c(0.03, 0.03)
  ) +
  scale_fill_manual(
    values = rep(catf_palette, length.out = n_distinct(pair_counts_filtered$department_1))
  ) +
  labs(
    title = "Cross-Department Project Flows",
    subtitle = paste0(
      "Top ", length(top_depts), " departments by collaborative activity; ",
      "flow width reflects shared projects"
    ),
    caption = str_wrap(excluded_label, width = 200),
    y = NULL,
    x = NULL
  ) +
  theme_catf() +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.line = element_blank(),
    plot.caption = element_text(
      hjust = 0, size = 8, color = "gray40",
      margin = margin(t = 10)
    ),
    plot.margin = margin(10, 10, 15, 10)
  )

# Add stratum labels using stat = "stratum" — avoids ggplot_build() fragility
# with factor axes and uses full department names (no prefix removal)
fig_department_sankey_filtered <- base_department_sankey_filtered +
  geom_text(
    stat = ggalluvial::StatStratum,
    aes(label = str_wrap(after_stat(stratum), width = 18)),
    hjust = 0.5,
    size = 3.5,
    lineheight = 0.9,
    color = "gray20"
  )

ggsave(
  filename = here(figures_dir, "fig_department_sankey_filtered.png"),
  plot = fig_department_sankey_filtered,
  width = 13,
  height = 7,
  units = "in",
  dpi = 300
)
cat("  Saved: figures/fig_department_sankey_filtered.png\n")


#
# Word Clouds by NEPA Process Type
# ----------------------------------------------------
cat("\nCreating process-type word cloud panels for multi-department projects...\n")

tbl_multiagency_wordcloud_panels <- generate_wordcloud_panels(
  df = multi_department_data,
  figure_prefix = "fig_multiagency_project_types_wordcloud",
  panel_table_name = "table_multiagency_wordcloud_panels.csv"
)
print(tbl_multiagency_wordcloud_panels)
