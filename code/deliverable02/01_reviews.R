# --------------------------
# DELIVERABLE 2: PROGRAMMATIC & TIERED REVIEW ANALYSIS
# --------------------------
# Figures and tables for the programmatic/tiered reviews deliverable.
# Answers two core questions:
#   1. How many tiered/programmatic reviews are there compared to total?
#   2. Are tiered reviews completed faster?
#
# Outputs (figures):
#   02_review_share.png       - Review type distribution (counts + %)
#   02_review_by_process.png  - Review type by NEPA process (100% stacked bar)
#   02_agency.png             - Top agencies for non-standard reviews
#   02_state.png              - Geographic distribution (top states)
#   02_duration.png           - Duration by review type and process type
#   02_tiered_parents.png     - Parent programmatic reviews cited by tiered projects
#
# Outputs (tables):
#   02_snapshot.csv           - Count and % by review_type x process_type
#   02_duration_summary.csv   - Duration descriptive statistics

# --------------------------
# SETUP
# --------------------------
rm(list=ls())
source(here::here("code", "deliverable02", "00_setup.R"))


# --------------------------
# SETUP
# --------------------------

reviews |> distinct(project_energy_type)
reviews |> count(process_type)
reviews |> count(review_type)


reviews |> glimpse()
reviews |> dim()
reviews |> glimpse()
duration_data|> count(review_type)
# 
# Programmatic
# ---------------------------------

# 0c3d63979201d57ca792f2bf380b8538 - This programmatic EIS
# 90f52e1a39168fe1d642731f1eacbdd0 - Generic Environmental Impact Statement
# a6c7af5b4682bbdf6a5ee6bc1295a795 - California Offshore Wind Draft Programmatic Environmental Impact Statement
# 35a07173481cc54b1bb1907fb5096331 - New York Bight Programmatic Environmental Impact Statement

sample_review("Programmatic", "EIS")
sample_review("Programmatic", "EIS")
sample_review("Programmatic", "EIS")
sample_review("Programmatic", "EIS")


# f95ec9530b352e3dd46e6473cb80dccf - Tier 1 EIS
# debe659941dc65ed630daab88d5fbf81 - Programmatic EA
# e4f17bdb94ef13df214876fefb844074 - Uranium Leasing Program Final Programmatic Environmental Assessment
# b8dbf48325b74bca43976283460ba1ef - generic environmental impact statement

sample_review("Programmatic", "EA")
sample_review("Programmatic", "EA")
sample_review("Programmatic", "EA")
sample_review("Programmatic", "EA")
  

reviews |> 
  #filter(project_id == "7f58211d8e13a419cc57083e545ba4b7") |> 
  #filter(project_id == "f95ec9530b352e3dd46e6473cb80dccf") |> 
  filter(project_id == "0c3d63979201d57ca792f2bf380b8538") |> 
  select(project_id, project_type, process_type, project_review_type:review_type) |>
  #pull(project_review_match_text)
  glimpse()

# 
# Tiered
# ---------------------------------
# 6c093ea21877201b04a2452b5c59fca9 - generic environmental impact statement
# 5c29e4983e3c45262048a8b0c6cba9cf - This EA tiers from\nthe SWEIS and a re-analysis of the operations per say will not be provided in this EA.
# e76f247aff5b44a943603ffb515644b2 - This EA tiers from the following environmental impact statements completed at the BLM state or national \n
# e76f247aff5b44a943603ffb515644b2 - The EA tiers to the Desert Renewable Energy Conservation Plan (DRECP) EIS and the WWEC \n

sample_review("Tiered", "EA")
sample_review("Tiered", "EA")
sample_review("Tiered", "EA")
sample_review("Tiered", "EA")
sample_review("Tiered", "EA")


# 
# Evaluate medium confidence docs
# ---------------------------------
medconf <- 
  reviews |> 
  filter(project_review_confidence == "medium") |> 
  filter(review_type != "Standard") |> 
  select(project_id) |> 
  glimpse()

sample_conf <- 
  medconf |> 
  select(project_id) |> 
  slice_sample(n=1) |> 
  pull()

# look at a random sample
reviews |> 
  filter(project_id %in% sample_conf) |> 
  select(project_id, project_type, process_type, project_review_type:review_type) |>
  glimpse()

# distinct review context for all medium confidence cases
reviews |> 
  filter(project_review_confidence == "medium") |> 
  filter(review_type != "Standard") |> 
  distinct(project_review_match_text) |> 
  print(n = 100)



# --------------------------
# FIGURE 1: REVIEW TYPE DISTRIBUTION
# --------------------------
# Horizontal bar chart showing all three review types with counts and
# percentage of total. Annotates the two non-standard types with a
# bracket to emphasize their combined share.

cat("\nCreating Figure 1: Review type distribution...\n")

review_counts <- reviews %>%
  count(review_type, name = "n") %>%
  mutate(
    pct   = n / sum(n),
    label = sprintf("%s  (%.1f%%)", comma(n), pct * 100)
  ) %>%
  arrange(desc(review_type))   # Tiered, Programmatic, Standard (bottom to top)

fig_share <- ggplot(review_counts,
                    aes(x = n, y = review_type, fill = review_type)) +
  geom_col(width = 0.55, alpha = 0.9) +
  geom_text(aes(label = label), hjust = -0.08, size = 3.6, color = "gray20") +
  scale_fill_manual(values = review_type_colors) +
  scale_x_continuous(
    expand = expansion(mult = c(0, 0.22)),
    labels = comma
  ) +
  labs(
    title    = "Review Type Distribution",
    subtitle = sprintf(
      "Clean energy EA/EIS projects in NEPATEC 2.0  (n = %s total)",
      comma(nrow(reviews))
    ),
    x = "Number of projects",
    y = NULL
  ) +
  theme_catf() +
  theme(
    legend.position    = "none",
    panel.grid.major.y = element_blank()
  )

fig_share_path <- here(figures_dir, "02_review_share.png")
ggsave(fig_share_path, fig_share, width = 9, height = 4, dpi = 300)
cat("  Saved:", fig_share_path, "\n")
print(fig_share)
# --------------------------
# FIGURE 2: REVIEW TYPE BY NEPA PROCESS
# --------------------------
# 100% stacked horizontal bar chart: each bar = one process type (EA / EIS),
# segments show proportion of standard / programmatic / tiered reviews.
# A companion panel on the right zooms in on the non-standard segment.

cat("\nCreating Figure 2: Review type by process type...\n")

# -- Panel A: 100% stacked proportions --
by_process <- reviews %>%
  count(process_type, review_type) %>%
  group_by(process_type) %>%
  mutate(
    total = sum(n),
    pct   = n / total,
    label = if_else(pct >= 0.05,
                    sprintf("%.1f%%\n(n=%s)", pct * 100, comma(n)),
                    "")
  ) %>%
  ungroup()

# Reorder levels so Standard appears first in stacked bar
by_process <- by_process %>%
  mutate(review_type = factor(review_type,
                              levels = rev(review_type_levels)))

panel_a <- ggplot(by_process,
                  aes(x = pct, y = process_type, fill = review_type)) +
  geom_col(width = 0.55, alpha = 0.9) +
  geom_text(aes(label = label),
            position = position_stack(vjust = 0.5),
            size = 3, color = "white", fontface = "bold") +
  scale_fill_manual(
    values = rev(review_type_colors),
    breaks = review_type_levels,
    name   = "Review type"
  ) +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(
    title    = "Review Type by NEPA Process",
    subtitle = "Each bar = 100% of projects in that process type",
    x = "Share of projects",
    y = NULL
  ) +
  theme_catf() +
  theme(legend.position = "bottom")

# -- Panel B: Zoomed non-standard absolute counts (vertical bars) --
non_std_counts <- by_process %>%
  filter(review_type != "Standard") %>%
  mutate(review_type = droplevels(review_type))

panel_b <- ggplot(non_std_counts,
                  aes(x = review_type, y = n, fill = process_type)) +
  geom_col(position = position_dodge(width = 0.65), width = 0.55, alpha = 0.9) +
  geom_text(aes(label = n),
            position = position_dodge(width = 0.65),
            vjust = -0.4, size = 3.5, color = "gray20") +
  scale_fill_manual(
    values = c("EA" = catf_blue, "EIS" = catf_dark_blue),
    name   = "Process"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22)),
                     labels = comma) +
  labs(
    title    = "Non-Standard Reviews",
    subtitle = "Counts by review type and NEPA process",
    x        = NULL,
    y        = "Number of projects"
  ) +
  theme_catf() +
  theme(legend.position = "bottom")

library(patchwork)

fig_process <- panel_a + panel_b +
  plot_layout(widths = c(2.2, 1)) +
  plot_annotation(
    caption = "Standard = stand-alone EA/EIS; Programmatic = PEIS/PEA; Tiered = tiers from a programmatic review."
  )

fig_process_path <- here(figures_dir, "02_review_by_process.png")
ggsave(fig_process_path, fig_process, width = 12, height = 4.5, dpi = 300)
cat("  Saved:", fig_process_path, "\n")
print(fig_process)

# --------------------------
# FIGURE 3: TOP AGENCIES FOR NON-STANDARD REVIEWS
# --------------------------
# Horizontal bar chart, faceted by review type (Programmatic vs Tiered).
# Shows the top 8 agencies across both review types.

cat("\nCreating Figure 3: Agency breakdown...\n")

top_agencies <- reviews_long_agency %>%
  filter(project_review_type %in% c("programmatic", "tiered")) %>%
  count(agency, review_type, name = "n") %>%
  group_by(agency) %>%
  mutate(total = sum(n)) %>%
  ungroup() %>%
  slice_max(order_by = total, n = 8 * 2, with_ties = FALSE) %>%  # keep top 8
  group_by(agency) %>%
  filter(any(total >= sort(unique(total), decreasing = TRUE)[min(8, n_distinct(agency))])) %>%
  ungroup()

# Re-identify top 8 agencies by total
top8_agencies <- reviews_long_agency %>%
  filter(project_review_type %in% c("programmatic", "tiered")) %>%
  count(agency, name = "total") %>%
  slice_max(order_by = total, n = 8) %>%
  pull(agency)

agency_data <- reviews_long_agency %>%
  filter(project_review_type %in% c("programmatic", "tiered"),
         agency %in% top8_agencies) %>%
  count(agency, review_type, name = "n") %>%
  # Fill zeros for missing combinations
  complete(agency, review_type = c("Programmatic", "Tiered"), fill = list(n = 0)) %>%
  group_by(agency) %>%
  mutate(total = sum(n)) %>%
  ungroup() %>%
  mutate(agency = fct_reorder(agency, total))

fig_agency <- ggplot(agency_data,
                     aes(x = n, y = agency, fill = review_type)) +
  geom_col(position = position_dodge(width = 0.65), width = 0.6, alpha = 0.9) +
  geom_text(aes(label = if_else(n > 0, as.character(n), "")),
            position = position_dodge(width = 0.65),
            hjust = -0.2, size = 3.2, color = "gray20") +
  scale_fill_manual(
    values = review_type_colors[c("Programmatic", "Tiered")],
    name   = "Review type"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25)),
                     labels = comma) +
  labs(
    title    = "Programmatic & Tiered Reviews by Lead Agency",
    subtitle = "Top 8 agencies by total non-standard reviews",
    x        = "Number of projects",
    y        = NULL,
    caption  = "Agency determined by lead_agency_harmonized; multi-agency projects counted once per agency."
  ) +
  theme_catf() +
  theme(
    legend.position    = "top",
    panel.grid.major.y = element_blank()
  )

fig_agency_path <- here(figures_dir, "02_agency.png")
ggsave(fig_agency_path, fig_agency, width = 10, height = 6, dpi = 300)
cat("  Saved:", fig_agency_path, "\n")
print(fig_agency)

# --------------------------
# FIGURE 3b: TOP DEPARTMENTS FOR NON-STANDARD REVIEWS
# --------------------------
# Same story as Figure 3 but at the department level (project_department is a
# scalar column — no unnesting needed).

cat("\nCreating Figure 3b: Department breakdown...\n")

dept_data <- non_standard %>%
  filter(!is.na(project_department), project_department != "") %>%
  count(project_department, review_type, name = "n") %>%
  complete(project_department, review_type = c("Programmatic", "Tiered"),
           fill = list(n = 0)) %>%
  group_by(project_department) %>%
  mutate(total = sum(n)) %>%
  ungroup() %>%
  mutate(project_department = fct_reorder(project_department, total))

fig_dept <- ggplot(dept_data,
                   aes(x = n, y = project_department, fill = review_type)) +
  geom_col(position = position_dodge(width = 0.65), width = 0.6, alpha = 0.9) +
  geom_text(aes(label = if_else(n > 0, as.character(n), "")),
            position = position_dodge(width = 0.65),
            hjust = -0.2, size = 3.2, color = "gray20") +
  scale_fill_manual(
    values = review_type_colors[c("Programmatic", "Tiered")],
    name   = "Review type"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25)),
                     labels = comma) +
  labs(
    title    = "Programmatic & Tiered Reviews by Department",
    subtitle = "All departments with at least one non-standard review",
    x        = "Number of projects",
    y        = NULL,
    caption  = "Department determined by project_department (scalar field; no multi-department unnesting needed)."
  ) +
  theme_catf() +
  theme(
    legend.position    = "top",
    panel.grid.major.y = element_blank()
  )

fig_dept_path <- here(figures_dir, "02_department.png")
ggsave(fig_dept_path, fig_dept, width = 10, height = 6, dpi = 300)
cat("  Saved:", fig_dept_path, "\n")
print(fig_dept)

# --------------------------
# FIGURE 4: GEOGRAPHIC DISTRIBUTION
# --------------------------
# Top states for non-standard reviews (programmatic + tiered combined),
# horizontal bar, colored by review type.

cat("\nCreating Figure 4: State distribution...\n")

top12_states <- reviews_long_state %>%
  filter(project_review_type %in% c("programmatic", "tiered")) %>%
  count(state, name = "total") %>%
  slice_max(order_by = total, n = 12) %>%
  pull(state)

state_data <- reviews_long_state %>%
  filter(project_review_type %in% c("programmatic", "tiered"),
         state %in% top12_states) %>%
  count(state, review_type, name = "n") %>%
  complete(state, review_type = c("Programmatic", "Tiered"), fill = list(n = 0)) %>%
  group_by(state) %>%
  mutate(total = sum(n)) %>%
  ungroup() %>%
  mutate(state = fct_reorder(state, total))

# Labels inside the Programmatic segment (center = n_prog / 2)
state_prog_labels <- state_data %>%
  filter(review_type == "Programmatic", n > 0) %>%
  mutate(label = sprintf("%d (%.0f%%)", n, n / total * 100))

fig_state <- ggplot(state_data,
                    aes(x = n, y = state, fill = review_type)) +
  geom_col(width = 0.65, alpha = 0.9) +   # stacked for total clarity
  scale_fill_manual(
    values = review_type_colors[c("Programmatic", "Tiered")],
    name   = "Review type"
  ) +
  # White label inside the programmatic segment showing count (% of row total)
  geom_text(
    data = state_prog_labels,
    aes(x = n / 2, y = state, label = label),
    inherit.aes = FALSE,
    color = "white", size = 2.8, fontface = "bold"
  ) +
  geom_text(
    data = state_data %>% group_by(state) %>% summarise(total = sum(n), .groups = "drop"),
    aes(x = total, y = state, label = total),
    inherit.aes = FALSE,
    hjust = -0.2, size = 3.2, color = "gray20"
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18)),
                     labels = comma) +
  labs(
    title    = "Geographic Distribution of Non-Standard Reviews",
    subtitle = "Top 12 states (programmatic + tiered; multi-state projects counted per state)",
    x        = "Number of projects",
    y        = NULL,
    caption  = "Western states dominate, reflecting BLM's large federal land footprint in the region."
  ) +
  theme_catf() +
  theme(
    legend.position    = "top",
    panel.grid.major.y = element_blank()
  )

fig_state_path <- here(figures_dir, "02_state.png")
ggsave(fig_state_path, fig_state, width = 9, height = 6, dpi = 300)
cat("  Saved:", fig_state_path, "\n")
print(fig_state)
# --------------------------
# FIGURE 5: DURATION COMPARISON
# --------------------------
# Box plot + jittered individual points showing review duration
# (initiation to decision) by review type, faceted by process type.
# Addresses the key deliverable question: are tiered reviews faster?

cat("\nCreating Figure 5: Duration comparison...\n")

# Sample sizes per group (for labels)
dur_n <- duration_data %>%
  group_by(process_type, review_type) %>%
  summarise(
    n           = n(),
    median_days = median(duration_days),
    .groups     = "drop"
  ) %>%
  mutate(n_label = paste0("n = ", n))

# Cap at p97 per process type for readability
p97 <- quantile(duration_data$duration_days, 0.97, na.rm = TRUE)

fig_duration <- ggplot(
  duration_data,
  aes(x = review_type, y = duration_days, fill = review_type, color = review_type)
) +
  # Violin layer (only meaningful for larger n)
  geom_violin(
    alpha = 0.25, trim = FALSE, color = NA,
    data = duration_data %>% group_by(process_type, review_type) %>%
      filter(n() >= 10) %>% ungroup()
  ) +
  # Box plot for all groups
  geom_boxplot(
    width = 0.25,
    outlier.shape = NA,
    fill  = "white",
    color = catf_navy,
    linewidth = 0.5,
    alpha = 0.8
  ) +
  # Individual points (jittered)
  geom_jitter(width = 0.12, size = 1.3, alpha = 0.4, show.legend = FALSE) +
  # Median label
  stat_summary(
    fun = median, geom = "point",
    shape = 18, size = 3.5, color = catf_navy,
    show.legend = FALSE
  ) +
  # n= label above each group
  geom_text(
    data = dur_n,
    aes(x = review_type, y = p97 * 1.06, label = n_label),
    inherit.aes = FALSE,
    size = 3, color = "gray40", vjust = 0
  ) +
  # Median annotation below each box
  geom_text(
    data = dur_n,
    aes(x = review_type, y = median_days,
        label = sprintf("median\n%s d", comma(round(median_days)))),
    inherit.aes = FALSE,
    hjust = -0.15, size = 2.6, color = catf_navy, fontface = "italic"
  ) +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(values  = review_type_colors) +
  scale_color_manual(values = review_type_colors) +
  coord_cartesian(ylim = c(0, p97 * 1.15)) +
  scale_y_continuous(labels = comma) +
  labs(
    title    = "Review Duration by Type",
    subtitle = "Days from initiation to decision (projects with complete timelines only)",
    x        = NULL,
    y        = "Duration (days)",
    caption  = paste0(
      "Box = IQR; diamond = median; points = individual projects (jittered). ",
      "Violin shown only for groups with n \u2265 10. Y-axis capped at 97th percentile."
    )
  ) +
  theme_catf() +
  theme(legend.position = "none")

fig_duration_path <- here(figures_dir, "02_duration.png")
ggsave(fig_duration_path, fig_duration, width = 11, height = 7, dpi = 300)
cat("  Saved:", fig_duration_path, "\n")
print(fig_duration)

# --------------------------
# FIGURE 6: TIERED REVIEW PARENTAGE
# --------------------------
# Which parent programmatic reviews generate the most downstream tiered work?
# Classifies the (often noisy) tiers_from text into named programmatic reviews.

cat("\nCreating Figure 6: Tiered review parents...\n")

tiered_projects <- reviews %>%
  filter(project_review_type == "tiered") %>%
  select(project_id, project_title, project_review_tiers_from)

classify_parent <- function(x) {
  x_lower <- tolower(x)
  dplyr::case_when(
    str_detect(x_lower, "vegetation.*herbicide|herbicide.*vegetation") ~
      "BLM Vegetation Treatment\nUsing Herbicides PEIS",
    str_detect(x_lower, "tva.*irp|integrated resource plan") ~
      "TVA Integrated\nResource Plan EIS",
    str_detect(x_lower, "drecp|desert renewable energy") ~
      "Desert Renewable Energy\nConservation Plan EIS",
    str_detect(x_lower, "great plains.*wind|wind.*great plains|ugp.*wind") ~
      "Upper Great Plains\nWind Energy PEIS",
    str_detect(x_lower, "geotherm") ~
      "Geothermal Leasing\nPEIS (Western States)",
    str_detect(x_lower, "tsr") ~
      "TSR PEIS",
    str_detect(x_lower, "montrose|montrose-nucla") ~
      "Montrose Transmission\nLine PEIS",
    str_detect(x_lower, "sweis") ~
      "SWEIS",
    str_detect(x_lower, "hazard.*removal|hazard.*vegetation") ~
      "BLM Hazard Removal &\nVegetation PEIA",
    str_detect(x_lower, "rp eis") ~
      "RP EIS",
    TRUE ~ "Reference not\nclearly identified"
  )
}

parent_counts <- tiered_projects %>%
  mutate(parent = classify_parent(
    if_else(is.na(project_review_tiers_from), "", project_review_tiers_from)
  )) %>%
  count(parent, name = "n") %>%
  mutate(
    identified = parent != "Reference not\nclearly identified",
    parent     = fct_reorder(parent, n)
  )

# Color: identified vs. not
parent_colors <- c(
  "TRUE"  = catf_dark_blue,
  "FALSE" = "gray70"
)

fig_parents <- ggplot(parent_counts,
                      aes(x = fct_reorder(parent, n), y = n,
                          color = as.character(identified))) +
  geom_segment(
    aes(xend = fct_reorder(parent, n), yend = 0),
    linewidth = 1.0, color = "gray80"
  ) +
  geom_point(size = 5, alpha = 0.95) +
  geom_text(aes(label = n), hjust = -1.4, size = 3.5, color = "gray20",
            show.legend = FALSE) +
  scale_color_manual(
    values = c("TRUE" = catf_dark_blue, "FALSE" = "gray65"),
    labels = c("TRUE" = "Identified programmatic review",
               "FALSE" = "Vague / unclear reference"),
    name = NULL
  ) +
  scale_y_continuous(
    breaks = 0:max(parent_counts$n),
    expand = expansion(mult = c(0, 0.3))
  ) +
  coord_flip() +
  labs(
    title    = "Parent Programmatic Reviews Cited by Tiered Projects",
    subtitle = sprintf(
      "Which PEIS/PEA do tiered EAs/EISs reference? (%d tiered projects total)", nrow(tiered_projects)
    ),
    x        = NULL,
    y        = "Number of tiered projects",
    caption  = paste0(
      "Tiers-from references extracted from the first 60 pages of each project's documents.\n",
      "Some references are vague and cannot be linked to a specific programmatic review."
    )
  ) +
  theme_catf() +
  theme(
    legend.position    = "top",
    panel.grid.major.y = element_blank(),
    plot.caption       = element_text(hjust = 0)
  )

fig_parents_path <- here(figures_dir, "02_tiered_parents.png")
ggsave(fig_parents_path, fig_parents, width = 10, height = 6, dpi = 300)
cat("  Saved:", fig_parents_path, "\n")
print(fig_parents)
# --------------------------
# TABLE 1: SNAPSHOT CROSS-TABULATION
# --------------------------
# Rows: EA / EIS / Total
# Columns: Standard / Programmatic / Tiered / Total
# Cells: count (n) and share (% of row total)

cat("\nSaving Table 1: Snapshot cross-tabulation...\n")

snapshot <- reviews %>%
  count(process_type, review_type, name = "n") %>%
  group_by(process_type) %>%
  mutate(
    row_total = sum(n),
    pct       = n / row_total * 100
  ) %>%
  ungroup()

# Add "All" row
snapshot_total <- reviews %>%
  count(review_type, name = "n") %>%
  mutate(
    process_type = "All",
    row_total    = sum(n),
    pct          = n / row_total * 100
  )

snapshot_full <- bind_rows(snapshot, snapshot_total) %>%
  mutate(
    cell = sprintf("%s (%.1f%%)", comma(n), pct),
    process_type = factor(process_type, levels = c("EA", "EIS", "All"))
  ) %>%
  select(process_type, review_type, n, pct, row_total) %>%
  arrange(process_type, review_type)

snapshot_wide <- snapshot_full %>%
  select(process_type, review_type, n, pct) %>%
  pivot_wider(
    names_from  = review_type,
    values_from = c(n, pct),
    values_fill = list(n = 0L, pct = 0)
  ) %>%
  left_join(
    snapshot_full %>%
      distinct(process_type, row_total),
    by = "process_type"
  )

snapshot_csv_path <- here(tables_dir, "02_snapshot.csv")
write_csv(snapshot_full, snapshot_csv_path)
cat("  Saved:", snapshot_csv_path, "\n")

# --------------------------
# TABLE 2: DURATION DESCRIPTIVE STATISTICS
# --------------------------

cat("\nSaving Table 2: Duration descriptive statistics...\n")

duration_summary <- duration_data %>%
  group_by(process_type, review_type) %>%
  summarise(
    n            = n(),
    mean_days    = round(mean(duration_days),   0),
    median_days  = round(median(duration_days), 0),
    sd_days      = round(sd(duration_days),     0),
    p25_days     = round(quantile(duration_days, 0.25), 0),
    p75_days     = round(quantile(duration_days, 0.75), 0),
    min_days     = min(duration_days),
    max_days     = max(duration_days),
    .groups = "drop"
  ) %>%
  arrange(process_type, review_type)

duration_csv_path <- here(tables_dir, "02_duration_summary.csv")
write_csv(duration_summary, duration_csv_path)
cat("  Saved:", duration_csv_path, "\n")
print(duration_summary)

# --------------------------
# CONSOLE SUMMARY
# --------------------------

cat("\n=== Deliverable 2 Analysis Complete ===\n")

n_total        <- nrow(reviews)
n_programmatic <- sum(reviews$project_review_type == "programmatic")
n_tiered       <- sum(reviews$project_review_type == "tiered")

cat(sprintf("Total projects:        %s\n", comma(n_total)))
cat(sprintf("  Standard:            %s (%.1f%%)\n",
            comma(n_total - n_programmatic - n_tiered),
            (n_total - n_programmatic - n_tiered) / n_total * 100))
cat(sprintf("  Programmatic:        %s (%.1f%%)\n",
            comma(n_programmatic), n_programmatic / n_total * 100))
cat(sprintf("  Tiered:              %s (%.1f%%)\n",
            comma(n_tiered), n_tiered / n_total * 100))
cat(sprintf("  Non-standard total:  %s (%.1f%%)\n",
            comma(n_programmatic + n_tiered),
            (n_programmatic + n_tiered) / n_total * 100))
cat("\nDuration summary (complete timelines):\n")
print(duration_summary %>%
        select(process_type, review_type, n, median_days, mean_days))
cat("\nFigures saved to:", figures_dir, "\n")
cat("Tables  saved to:", tables_dir, "\n")

