# --------------------------
# DELIVERABLE 5: PAGE COUNT ANALYSIS
# --------------------------
# Figures analyzing document length trends and FRA impact
# for clean energy EA and EIS projects
#
# Figures produced:
#   1. Coverage funnel (inclusion criteria)
#   2. Average pages over time (monthly line chart with FRA line)
#   3. Pre/Post FRA comparison (bar chart, mean pages)
#   4. Pre/Post FRA distribution (violin + box plot)
#   5. Project-level scatter with LOESS trend
#   6. Annual average pages by decision year

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable05", "00_setup.R"))

# --------------------------
# FIGURE 1: COVERAGE / INCLUSION CRITERIA
# --------------------------
# Shows how many clean energy EA/EIS projects survive each filter step

cat("\nCreating Figure 1: Coverage funnel...\n")

# Add percentage relative to total for each process type
coverage_plot <- coverage %>%
  group_by(process_type) %>%
  mutate(
    total = max(n),
    pct = n / total * 100,
    label = sprintf("%s (%.0f%%)", comma(n), pct)
  ) %>%
  ungroup()

fig_coverage <- ggplot(coverage_plot, aes(x = n, y = fct_rev(step), fill = process_type)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.85) +
  geom_text(
    aes(label = label),
    position = position_dodge(width = 0.7),
    hjust = -0.05,
    size = 3,
    color = "gray30"
  ) +
  scale_fill_manual(
    values = c("EA" = catf_dark_blue, "EIS" = catf_teal),
    labels = c("EA" = "Environmental Assessment",
               "EIS" = "Environmental Impact Statement")
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25)), labels = comma) +
  labs(
    title = "Analysis Coverage: Inclusion Criteria",
    subtitle = "Number of clean energy projects retained at each filtering step",
    x = "Number of Projects",
    y = NULL,
    fill = NULL
  ) +
  theme_catf() +
  theme(legend.position = "top")

fig_coverage_path <- here(figures_dir, "05_coverage.png")
ggsave(fig_coverage_path, fig_coverage, width = 10, height = 6, dpi = 300)
cat("  Saved:", fig_coverage_path, "\n")
print(fig_coverage)

# --------------------------
# FIGURE 2: AVERAGE DOCUMENT LENGTH OVER TIME
# --------------------------
# Monthly average page count with 6-month rolling average
# Red dashed vertical line at FRA enactment date
# Faceted by EA vs EIS

cat("\nCreating Figure 2: Average pages over time (monthly)...\n")

# Filter to reasonable date range for readability
pages_for_time <- pages_analysis %>%
  filter(decision_year >= 2000, decision_year <= 2025)

monthly_pages <- pages_for_time %>%
  group_by(process_type, decision_month) %>%
  summarise(
    mean_pages = mean(total_pages, na.rm = TRUE),
    median_pages = median(total_pages, na.rm = TRUE),
    n_projects = n(),
    .groups = "drop"
  ) %>%
  arrange(process_type, decision_month) %>%
  group_by(process_type) %>%
  mutate(
    rolling_mean_6m = zoo::rollmean(mean_pages, k = 6, fill = NA, align = "right")
  ) %>%
  ungroup()

fig_pages_over_time <- ggplot(monthly_pages, aes(x = decision_month)) +
  geom_line(aes(y = mean_pages), color = catf_light_blue, alpha = 0.6, linewidth = 0.5) +
  geom_point(aes(y = mean_pages, size = n_projects),
             color = catf_light_blue, alpha = 0.4) +
  geom_line(aes(y = rolling_mean_6m), color = catf_dark_blue, linewidth = 1.1, na.rm = TRUE) +
  geom_vline(xintercept = fra_date, linetype = "dashed", color = "red", linewidth = 0.8) +
  annotate(
    "text", x = fra_date + 45, y = Inf,
    label = "FRA enacted\n(June 3, 2023)",
    vjust = 1.5, hjust = 0, size = 3, color = "red", fontface = "italic"
  ) +
  facet_wrap(~process_type, ncol = 1, scales = "free_y") +
  scale_x_date(date_labels = "%Y", date_breaks = "2 years") +
  scale_size_continuous(range = c(0.5, 3), guide = "none") +
  labs(
    title = "Average Document Length Over Time",
    subtitle = "Monthly average (light blue) with 6-month rolling average (dark blue)",
    x = "Decision Date (Month)",
    y = "Average Total Pages",
    caption = "Point size reflects number of projects in that month. Projects with complete timelines only."
  ) +
  theme_catf()

fig_pages_over_time_path <- here(figures_dir, "05_pages_over_time.png")
ggsave(fig_pages_over_time_path, fig_pages_over_time, width = 12, height = 8, dpi = 300)
cat("  Saved:", fig_pages_over_time_path, "\n")
print(fig_pages_over_time)

# --------------------------
# FIGURE 3: PRE/POST FRA BAR CHART (NORMALIZED)
# --------------------------
# Mean total pages by process type, before and after FRA
# Includes sample sizes and median markers

cat("\nCreating Figure 3: Pre/Post FRA comparison (bar chart)...\n")

fra_summary <- pages_analysis %>%
  group_by(process_type, fra_period) %>%
  summarise(
    mean_pages = mean(total_pages, na.rm = TRUE),
    median_pages = median(total_pages, na.rm = TRUE),
    sd_pages = sd(total_pages, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    bar_label = sprintf("%.0f pages\n(n = %s)", mean_pages, comma(n)),
    median_label = sprintf("median: %.0f", median_pages)
  )

fig_pre_post <- ggplot(fra_summary, aes(x = fra_period, y = mean_pages, fill = fra_period)) +
  geom_col(alpha = 0.85, width = 0.6) +
  geom_text(aes(label = bar_label), vjust = -0.2, size = 3.3, color = "gray20") +
  geom_point(aes(y = median_pages), shape = 18, size = 4, color = catf_navy) +
  geom_text(
    aes(y = median_pages, label = median_label),
    hjust = -0.15, size = 2.8, color = catf_navy
  ) +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(
    values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.3))) +
  labs(
    title = "Document Length: Pre vs Post Fiscal Responsibility Act",
    subtitle = "Bar height = mean total pages; diamond = median; projects classified by decision date",
    x = NULL,
    y = "Total Pages",
    fill = NULL
  ) +
  theme_catf() +
  theme(legend.position = "none")

fig_pre_post_path <- here(figures_dir, "05_pages_pre_post_fra.png")
ggsave(fig_pre_post_path, fig_pre_post, width = 10, height = 6, dpi = 300)
cat("  Saved:", fig_pre_post_path, "\n")
print(fig_pre_post)

# --------------------------
# FIGURE 4: DISTRIBUTION PRE/POST FRA (VIOLIN + BOX PLOT)
# --------------------------
# Shows the full distribution of page counts before and after FRA

cat("\nCreating Figure 4: Pre/Post FRA distribution (violin + box plot)...\n")

# Cap y-axis at p99 for readability (extreme outliers distort the view)
p99_pages <- quantile(pages_analysis$total_pages, 0.99, na.rm = TRUE)

# Add n labels per group
n_labels <- pages_analysis %>%
  group_by(process_type, fra_period) %>%
  summarise(
    n = n(),
    label = paste0("n = ", comma(n)),
    .groups = "drop"
  )

fig_distribution <- ggplot(pages_analysis, aes(x = fra_period, y = total_pages, fill = fra_period)) +
  geom_violin(alpha = 0.2, trim = FALSE, color = NA) +
  geom_boxplot(
    width = 0.2,
    outlier.alpha = 0.25,
    outlier.size = 0.8,
    fill = "white",
    color = catf_navy,
    linewidth = 0.4
  ) +
  stat_summary(fun = median, geom = "point", shape = 21, size = 2.5,
               fill = catf_navy, color = "white") +
  geom_text(
    data = n_labels,
    aes(x = fra_period, y = 0, label = label),
    inherit.aes = FALSE,
    vjust = 1.5, size = 3, color = "gray40"
  ) +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(
    values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)
  ) +
  coord_cartesian(ylim = c(0, p99_pages)) +
  labs(
    title = "Document Length Distribution: Pre vs Post FRA",
    subtitle = "Violin + boxplot overlay; dot = median (y-axis capped at p99)",
    x = NULL,
    y = "Total Pages",
    fill = NULL
  ) +
  theme_catf() +
  theme(legend.position = "none")

fig_distribution_path <- here(figures_dir, "05_pages_distribution_boxplot.png")
ggsave(fig_distribution_path, fig_distribution, width = 10, height = 6, dpi = 300)
cat("  Saved:", fig_distribution_path, "\n")
print(fig_distribution)

# --------------------------
# FIGURE 5: SCATTER WITH LOESS TREND
# --------------------------
# Individual projects plotted by decision date and total pages
# LOESS smoothing shows overall trend

cat("\nCreating Figure 5: Pages over time scatter with trend...\n")

fig_scatter <- ggplot(
  pages_analysis %>% filter(decision_year >= 2000, decision_year <= 2025),
  aes(x = timeline_decision_date, y = total_pages)
) +
  geom_point(aes(color = fra_period), alpha = 0.35, size = 1.5) +
  geom_smooth(
    method = "loess", se = TRUE,
    color = catf_navy, fill = catf_light_blue,
    alpha = 0.2, linewidth = 1
  ) +
  geom_vline(xintercept = fra_date, linetype = "dashed", color = "red", linewidth = 0.7) +
  facet_wrap(~process_type, ncol = 1, scales = "free_y") +
  scale_color_manual(
    values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)
  ) +
  scale_x_date(date_labels = "%Y", date_breaks = "2 years") +
  labs(
    title = "Document Length by Decision Date",
    subtitle = "Each point = one project; blue curve = LOESS trend with 95% CI",
    x = "Decision Date",
    y = "Total Pages",
    color = NULL,
    caption = "Red dashed line = FRA enactment (June 3, 2023)"
  ) +
  theme_catf() +
  theme(legend.position = "top")

fig_scatter_path <- here(figures_dir, "05_pages_scatter.png")
ggsave(fig_scatter_path, fig_scatter, width = 12, height = 8, dpi = 300)
cat("  Saved:", fig_scatter_path, "\n")
print(fig_scatter)

# --------------------------
# SUMMARY TABLE (CSV)
# --------------------------

cat("\nSaving summary table...\n")

summary_table <- pages_analysis %>%
  group_by(process_type, fra_period) %>%
  summarise(
    n_projects = n(),
    mean_pages = round(mean(total_pages, na.rm = TRUE), 0),
    median_pages = median(total_pages, na.rm = TRUE),
    sd_pages = round(sd(total_pages, na.rm = TRUE), 0),
    min_pages = min(total_pages, na.rm = TRUE),
    max_pages = max(total_pages, na.rm = TRUE),
    p25_pages = quantile(total_pages, 0.25, na.rm = TRUE),
    p75_pages = quantile(total_pages, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

summary_table_path <- here(tables_dir, "05_pages_summary.csv")
write_csv(summary_table, summary_table_path)
cat("  Saved:", summary_table_path, "\n")
print(summary_table)

# Coverage table
coverage_wide <- coverage %>%
  pivot_wider(names_from = process_type, values_from = n, values_fill = 0) %>%
  mutate(Total = EA + EIS)

coverage_table_path <- here(tables_dir, "05_coverage.csv")
write_csv(coverage_wide, coverage_table_path)
cat("  Saved:", coverage_table_path, "\n")
print(coverage_wide)

# --------------------------
# CONSOLE SUMMARY
# --------------------------

cat("\n=== Deliverable 5 Page Analysis Complete ===\n\n")

cat("Analysis sample:", nrow(pages_analysis), "projects with complete timelines\n")
cat("  EA:", sum(pages_analysis$process_type == "EA"), "\n")
cat("  EIS:", sum(pages_analysis$process_type == "EIS"), "\n")
cat("  Pre-FRA:", sum(pages_analysis$fra_period == "Pre-FRA"), "\n")
cat("  Post-FRA:", sum(pages_analysis$fra_period == "Post-FRA"), "\n\n")

cat("Page count summary by process type and FRA period:\n")
print(summary_table)

cat("\nFigures saved to:", figures_dir, "\n")
cat("Tables saved to:", tables_dir, "\n")
