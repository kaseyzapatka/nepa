# --------------------------
# DELIVERABLE 5: PAGE COUNT ANALYSIS
# --------------------------
# Figures analyzing document length trends and FRA impact
# for clean energy EA and EIS projects
#
# Figures produced:
#   1. Coverage funnel (inclusion criteria)
#   2. Document length over time (individual points + 3-month rolling average with FRA line)
#   3. Pre/Post FRA comparison (bar chart, mean pages)
#   4. Pre/Post FRA distribution (violin + box plot)
#   5. FRA page limit compliance (Post-FRA projects only)

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
  scale_x_continuous(expand = expansion(mult = c(0, 0.25)), labels = comma,
                     breaks = seq(0, 2000, by = 200)) +
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
# 3-month rolling average of regulatory page counts
# Red dashed vertical line at FRA enactment date
# Faceted by EA vs EIS

cat("\nCreating Figure 2: Average pages over time (rolling average)...\n")

# Filter to 2010+ for readability; drop projects without regulatory_pages
pages_for_time <- pages_analysis %>%
  filter(decision_year >= 2010, decision_year <= 2025, !is.na(regulatory_pages))

monthly_pages <- pages_for_time %>%
  group_by(process_type, decision_month) %>%
  summarise(
    mean_pages = mean(regulatory_pages, na.rm = TRUE),
    n_projects = n(),
    .groups = "drop"
  ) %>%
  arrange(process_type, decision_month) %>%
  group_by(process_type) %>%
  mutate(
    rolling_mean_3m = zoo::rollmean(mean_pages, k = 3, fill = NA, align = "right")
  ) %>%
  ungroup()

fig_pages_over_time <- ggplot() +
  # Individual project points (low alpha, colored by FRA period)
  geom_point(
    data = pages_for_time,
    aes(x = timeline_decision_date, y = regulatory_pages, color = fra_period),
    alpha = 0.32, size = 1.2
  ) +
  # 3-month rolling average line
  geom_line(
    data = monthly_pages,
    aes(x = decision_month, y = rolling_mean_3m),
    color = catf_navy, linewidth = 1.2, na.rm = TRUE
  ) +
  geom_vline(xintercept = fra_date, linetype = "dashed", color = "red", linewidth = 0.8) +
  annotate(
    "text", x = fra_date + 45, y = Inf,
    label = "Fiscal Responsibility Act\nof 2023 enacted\n(June 3, 2023)",
    vjust = 1.5, hjust = 0, size = 3, color = "red", fontface = "italic"
  ) +
  facet_wrap(~process_type, ncol = 1, scales = "free_y") +
  scale_x_date(date_labels = "%Y", date_breaks = "2 years") +
  scale_color_manual(
    values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)
  ) +
  labs(
    title = "Document Length Over Time",
    subtitle = "Points = individual projects (colored by FRA period); line = 3-month rolling average",
    x = "Decision Date",
    y = "Regulatory Pages (body word count ÷ 500)",
    color = NULL,
    caption = "Note: Projects with complete timelines only. Regulatory pages exclude embedded appendices and low-content pages per 40 C.F.R. § 1508.1(bb)."
  ) +
  theme_catf() +
  theme(legend.position = "top")

fig_pages_over_time_path <- here(figures_dir, "05_pages_over_time.png")
ggsave(fig_pages_over_time_path, fig_pages_over_time, width = 12, height = 8, dpi = 300)
cat("  Saved:", fig_pages_over_time_path, "\n")
print(fig_pages_over_time)

# --------------------------
# FIGURE 3: PRE/POST FRA BAR CHART
# --------------------------
# Mean regulatory pages by process type, before and after FRA
# Includes sample sizes and median markers

cat("\nCreating Figure 3: Pre/Post FRA comparison (bar chart)...\n")

fra_summary <- pages_analysis %>%
  filter(!is.na(regulatory_pages)) %>%
  group_by(process_type, fra_period) %>%
  summarise(
    mean_pages = mean(regulatory_pages, na.rm = TRUE),
    median_pages = median(regulatory_pages, na.rm = TRUE),
    sd_pages = sd(regulatory_pages, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    bar_label = sprintf("average\n%.0f pages\n(n = %s)", mean_pages, comma(n)),
    median_label = sprintf("median: %.0f", median_pages)
  )

# EA post-FRA needs reversed label positions (median above bar, average below diamond)
# because the short bar makes the default placement hard to read
ea_post <- fra_summary %>% filter(process_type == "EA", fra_period == "Post-FRA")
other_bars <- fra_summary %>% filter(!(process_type == "EA" & fra_period == "Post-FRA"))

fig_pre_post <- ggplot(fra_summary, aes(x = fra_period, y = mean_pages, fill = fra_period)) +
  geom_col(alpha = 0.85, width = 0.6) +
  # Average label — standard bars: above bar, dark text
  geom_text(
    data = other_bars,
    aes(label = bar_label), vjust = -0.2, size = 3.3, color = "gray20"
  ) +
  # Average label — EA post-FRA: reversed to below-diamond position, white text
  geom_text(
    data = ea_post,
    aes(y = median_pages, label = bar_label), vjust = 1.8, size = 3.3, color = "white"
  ) +
  geom_point(aes(y = median_pages), shape = 18, size = 4, color = catf_navy) +
  # Median label — standard bars: below diamond, white text
  geom_text(
    data = other_bars,
    aes(y = median_pages, label = median_label),
    vjust = 1.8, size = 2.8, color = "white"
  ) +
  # Median label — EA post-FRA: reversed to above-bar position, black text
  geom_text(
    data = ea_post,
    aes(y = mean_pages, label = median_label),
    vjust = -0.2, size = 2.8, color = "black"
  ) +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(
    values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.3))) +
  labs(
    title = "Document Length: Pre vs Post Fiscal Responsibility Act",
    subtitle = "Bar height = mean regulatory pages; diamond = median; projects classified by decision date",
    x = NULL,
    y = "Regulatory Pages (body word count ÷ 500)",
    fill = NULL
  ) +
  theme_catf() +
  theme(legend.position = "none")

fig_pre_post_path <- here(figures_dir, "05_pages_pre_post_fra.png")
ggsave(fig_pre_post_path, fig_pre_post, width = 10, height = 6, dpi = 300)
cat("  Saved:", fig_pre_post_path, "\n")
print(fig_pre_post)

# --------------------------
# FIGURE 3b: REGULATORY vs BODY PAGES COMPARISON (PRE/POST FRA)
# --------------------------
# Shows how much the FRA word-count normalisation (regulatory_pages) differs
# from simply counting physical body pages (body_pages).
#
# body_pages    = count of physical pages in the body (before appendix) with
#                 >= 50 words.  Excludes appendices but still counts sparse pages
#                 (covers, section dividers, tables) as full pages.
# regulatory_pages = body_word_count / 500.  The FRA statutory definition:
#                 sparse pages contribute < 1 page; only dense text pages
#                 count toward the limit.
#
# The gap between the two bars = pages "lost" to the word-count normalisation.

cat("\nCreating Figure 3b: Regulatory vs body pages comparison...\n")

# Pivot to long form: one row per project × measure
compare_long <- pages_analysis %>%
  filter(!is.na(regulatory_pages), !is.na(body_pages)) %>%
  select(project_id, process_type, fra_period, regulatory_pages, body_pages) %>%
  pivot_longer(
    cols = c(regulatory_pages, body_pages),
    names_to  = "measure",
    values_to = "pages"
  ) %>%
  mutate(
    measure = factor(
      measure,
      levels = c("body_pages", "regulatory_pages"),
      labels = c("Body pages\n(physical, ≥50 words)", "Regulatory pages\n(word count ÷ 500)")
    )
  )

compare_summary <- compare_long %>%
  group_by(process_type, fra_period, measure) %>%
  summarise(
    mean_pages   = mean(pages, na.rm = TRUE),
    median_pages = median(pages, na.rm = TRUE),
    n            = n(),
    .groups = "drop"
  )

# Print console comparison table
cat("  Regulatory vs body pages comparison (means):\n")
print(
  compare_summary %>%
    select(process_type, fra_period, measure, mean_pages, median_pages, n) %>%
    mutate(across(c(mean_pages, median_pages), round, 0)) %>%
    arrange(process_type, fra_period, measure),
  n = Inf
)

fig_compare <- ggplot(
  compare_summary,
  aes(x = fra_period, y = mean_pages, fill = measure)
) +
  geom_col(
    position = position_dodge(width = 0.7),
    width = 0.6, alpha = 0.88
  ) +
  geom_point(
    aes(y = median_pages),
    position = position_dodge(width = 0.7),
    shape = 18, size = 3.5, color = catf_navy
  ) +
  geom_text(
    aes(label = sprintf("%.0f", mean_pages)),
    position = position_dodge(width = 0.7),
    vjust = -0.4, size = 3, color = "gray20"
  ) +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(
    values = c(
      "Body pages\n(physical, ≥50 words)"   = catf_light_blue,
      "Regulatory pages\n(word count ÷ 500)" = catf_dark_blue
    )
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(
    title    = "Document Length: Regulatory Pages vs Physical Body Pages",
    subtitle = "Bar height = mean; diamond = median | Body pages = physical pages before appendix with ≥50 words",
    caption  = paste0(
      "Regulatory pages = body word count ÷ 500 per 40 C.F.R. § 1508.1(bb). ",
      "Gap between bars reflects sparse pages (headers, tables, dividers) that count ",
      "as full physical pages but contribute < 1 regulatory page."
    ),
    x    = NULL,
    y    = "Pages",
    fill = NULL
  ) +
  theme_catf() +
  theme(
    legend.position  = "top",
    plot.caption     = element_text(size = rel(0.75), hjust = 0, color = "gray40"),
    plot.caption.position = "plot"
  )

fig_compare_path <- here(figures_dir, "05_pages_reg_vs_body.png")
ggsave(fig_compare_path, fig_compare, width = 11, height = 6, dpi = 300)
cat("  Saved:", fig_compare_path, "\n")
print(fig_compare)

# --------------------------
# FIGURE 4: DISTRIBUTION PRE/POST FRA (VIOLIN + BOX PLOT)
# --------------------------
# Shows the full distribution of regulatory page counts before and after FRA

cat("\nCreating Figure 4: Pre/Post FRA distribution (violin + box plot)...\n")

pages_analysis_reg <- pages_analysis %>% filter(!is.na(regulatory_pages))

# Cap y-axis at p99 for readability (extreme outliers distort the view)
p99_pages <- quantile(pages_analysis_reg$regulatory_pages, 0.99, na.rm = TRUE)

# Add n and median labels per group
dist_labels <- pages_analysis_reg %>%
  group_by(process_type, fra_period) %>%
  summarise(
    n = n(),
    median_pages = median(regulatory_pages, na.rm = TRUE),
    n_label = paste0("n = ", comma(n)),
    median_label = sprintf("median: %.0f", median_pages),
    .groups = "drop"
  )

fig_distribution <- ggplot(pages_analysis_reg,
                           aes(x = fra_period, y = regulatory_pages, fill = fra_period)) +
  geom_violin(alpha = 0.35, trim = FALSE, color = NA) +
  geom_boxplot(
    width = 0.2,
    outlier.alpha = 0.25,
    outlier.size = 0.8,
    fill = "white",
    color = catf_navy,
    linewidth = 0.4
  ) +
  stat_summary(fun = median, geom = "point", shape = 18, size = 3.5,
               color = catf_navy) +
  geom_label(
    data = dist_labels,
    aes(x = fra_period, y = median_pages, label = paste0("median\n", comma(median_pages))),
    inherit.aes = FALSE,
    hjust = -0.15, size = 2.5, color = catf_navy,
    fill = "white", label.size = 0.2, label.padding = unit(0.15, "lines")
  ) +
  geom_text(
    data = dist_labels,
    aes(x = fra_period, y = 0, label = n_label),
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
    subtitle = "Violin + boxplot overlay; diamond = median (y-axis capped at p99)",
    x = NULL,
    y = "Regulatory Pages (body word count ÷ 500)",
    fill = NULL
  ) +
  theme_catf() +
  theme(legend.position = "none")

fig_distribution_path <- here(figures_dir, "05_pages_distribution_boxplot.png")
ggsave(fig_distribution_path, fig_distribution, width = 10, height = 6, dpi = 300)
cat("  Saved:", fig_distribution_path, "\n")
print(fig_distribution)

# --------------------------
# FIGURE 5: FRA PAGE LIMIT COMPLIANCE (POST-FRA ONLY)
# --------------------------
# FRA page limits:
#   EA:  75 pages max
#   EIS: 150 pages max (standard), 300 pages max (extraordinarily complex)
# Shows how well post-FRA projects comply with these limits

cat("\nCreating Figure 6: FRA page limit compliance...\n")

# Compliance is assessed using regulatory_pages (word-count-based, excluding embedded
# appendices and low-content pages), which better reflects 40 C.F.R. § 1508.1(bb).
# Projects with no extractable text (regulatory_pages = NA) are dropped from this figure.
post_fra <- pages_analysis %>%
  filter(fra_period == "Post-FRA", !is.na(regulatory_pages)) %>%
  mutate(
    compliance = case_when(
      process_type == "EA" & regulatory_pages <= 75 ~ "Compliant",
      process_type == "EA" & regulatory_pages > 75 ~ "Exceeds limit",
      process_type == "EIS" & regulatory_pages <= 150 ~ "Compliant",
      process_type == "EIS" & regulatory_pages > 150 & regulatory_pages <= 300 ~ "Exceeds standard limit",
      process_type == "EIS" & regulatory_pages > 300 ~ "Exceeds limit"
    )
  )

# Order the compliance categories (Exceeds limit rightmost/last in stack)
ea_levels <- c("Compliant", "Exceeds limit")
eis_levels <- c("Compliant", "Exceeds standard limit", "Exceeds limit")
all_levels <- c("Compliant", "Exceeds standard limit", "Exceeds limit")

post_fra <- post_fra %>%
  mutate(compliance = factor(compliance, levels = all_levels))

# Compliance colors: teal for compliant, amber for middle tier, magenta for exceeds
compliance_colors <- c(
  "Compliant" = catf_teal,
  "Exceeds standard limit" = "#E8A317",
  "Exceeds limit" = catf_magenta
)

# Summarise for plotting
compliance_summary <- post_fra %>%
  count(process_type, compliance, .drop = FALSE) %>%
  group_by(process_type) %>%
  # Drop levels that don't belong to this process type
  filter(
    (process_type == "EA" & compliance %in% ea_levels) |
    (process_type == "EIS" & compliance %in% eis_levels)
  ) %>%
  mutate(
    total = sum(n),
    pct = n / total * 100,
    label = sprintf("%s\n(%.0f%%)", comma(n), pct)
  ) %>%
  ungroup()

fig_compliance <- ggplot(compliance_summary,
                         aes(x = process_type, y = n, fill = compliance)) +
  geom_col(width = 0.6, alpha = 0.9) +
  geom_text(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 3.2, color = "white", fontface = "bold"
  ) +
  scale_fill_manual(values = compliance_colors) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(
    title = "FRA Page Limit Compliance: Post-FRA Projects",
    subtitle = paste0(
      "EA limit: 75 pages | EIS limit: 150 pages (300 for extraordinarily complex)\n",
      "n = ", comma(sum(compliance_summary$n[compliance_summary$process_type == "EA"])),
      " EA, ",
      comma(sum(compliance_summary$n[compliance_summary$process_type == "EIS"])),
      " EIS post-FRA projects"
    ),
    x = NULL,
    y = "Number of Projects",
    fill = NULL
  ) +
  theme_catf() +
  theme(legend.position = "bottom")

fig_compliance_path <- here(figures_dir, "05_fra_compliance.png")
ggsave(fig_compliance_path, fig_compliance, width = 10, height = 7, dpi = 300)
cat("  Saved:", fig_compliance_path, "\n")
print(fig_compliance)

# Save compliance summary table
compliance_table_path <- here(tables_dir, "05_fra_compliance.csv")
write_csv(compliance_summary, compliance_table_path)
cat("  Saved:", compliance_table_path, "\n")
print(compliance_summary)

# --------------------------
# SUMMARY TABLE (CSV)
# --------------------------

cat("\nSaving summary table...\n")

summary_table <- pages_analysis %>%
  group_by(process_type, fra_period) %>%
  summarise(
    n_projects = n(),
    # Regulatory page counts (primary measure; falls back to raw total_pages when
    # OCR extraction was unavailable — see reg_pages_source for breakdown)
    mean_pages   = round(mean(regulatory_pages, na.rm = TRUE), 0),
    median_pages = median(regulatory_pages, na.rm = TRUE),
    sd_pages     = round(sd(regulatory_pages, na.rm = TRUE), 0),
    p25_pages    = quantile(regulatory_pages, 0.25, na.rm = TRUE),
    p75_pages    = quantile(regulatory_pages, 0.75, na.rm = TRUE),
    # Source breakdown (how many projects used each method)
    n_ocr           = sum(reg_pages_source == "ocr",              na.rm = TRUE),
    n_no_appx_file  = sum(reg_pages_source == "no_appendix_file", na.rm = TRUE),
    n_raw_fallback  = sum(reg_pages_source == "raw_fallback",     na.rm = TRUE),
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
