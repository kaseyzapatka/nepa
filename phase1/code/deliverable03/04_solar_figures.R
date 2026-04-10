# --------------------------
# DELIVERABLE 3: SOLAR-FILTERED FIGURES (CLIENT DELIVERY)
# --------------------------
# Recreates two figures from the full clean-energy analysis,
# restricted to projects tagged "Renewable Energy Production - Solar".
#
# Outputs:
#   output/deliverable3/figures/03_duration_summary_intervals_by_process_solar.png
#   output/deliverable3/figures/06_capacity_distribution_violin_box_solar.png
#
# Note: Does not modify any existing deliverable scripts or reports.

source(here::here("phase1", "code", "deliverable03", "00_setup.R"))

SOLAR_TAG <- "Renewable Energy Production - Solar"

# --------------------------
# FIGURE A: DURATION SUMMARY INTERVALS BY PROCESS (SOLAR ONLY)
# Mirrors code/deliverable03/03_timeline.R lines 20-311
# --------------------------

cat("=== Figure A: Duration Summary (Solar Only) ===\n")

timeline <- load_timeline_for_deliverable3()

process_levels <- c("CE", "EA", "EIS")

timeline <- timeline %>%
  mutate(
    source_for_plot = toupper(as.character(coalesce(dataset_source, process_type))),
    process_group = factor(source_for_plot, levels = process_levels),
    bert_decision_date = as.Date(bert_decision_date),
    bert_application_date = as.Date(bert_application_date),
    bert_inferred_application_date = as.Date(bert_inferred_application_date),
    bert_earliest_review_date = as.Date(bert_earliest_review_date),
    bert_initiation_date_final = as.Date(bert_initiation_date_final),
    bert_decision_date_final = as.Date(bert_decision_date_final),
    timeline_complete = !is.na(bert_initiation_date_final) & !is.na(bert_decision_date_final),
    bert_year = as.integer(format(bert_decision_date_final, "%Y")),
    decision_year = coalesce(decision_year, bert_year),
    bert_start_date = coalesce(bert_application_date, bert_inferred_application_date, bert_initiation_date_final),
    bert_duration_days = as.numeric(bert_decision_date_final - bert_start_date)
  )

# Filter to solar projects
timeline_solar <- timeline %>%
  filter(str_detect(as.character(project_type), fixed(SOLAR_TAG)))

cat("Solar projects in timeline:", nrow(timeline_solar), "\n")
cat("By process group:\n")
print(count(timeline_solar, process_group))

duration_complete <- timeline_solar %>%
  filter(!is.na(process_group), timeline_complete) %>%
  mutate(duration_months = bert_duration_days / 30.44) %>%
  filter(!is.na(duration_months), duration_months >= 0)

duration_summary <- duration_complete %>%
  group_by(process_group) %>%
  summarise(
    n = n(),
    p10 = quantile(duration_months, 0.10, na.rm = TRUE),
    p25 = quantile(duration_months, 0.25, na.rm = TRUE),
    median_months = median(duration_months, na.rm = TRUE),
    p75 = quantile(duration_months, 0.75, na.rm = TRUE),
    p90 = quantile(duration_months, 0.90, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    median_label = paste0(process_group, ": ~", round(median_months), " months"),
    label_hjust = if_else(process_group == "CE", 0, 0.5)
  )

cat("\nDuration summary (solar):\n")
print(duration_summary)

fig_duration_solar <- ggplot(duration_summary, aes(y = process_group, color = process_group)) +
  geom_segment(aes(x = p10, xend = p90, yend = process_group), linewidth = 1.8, alpha = 0.35) +
  geom_segment(aes(x = p25, xend = p75, yend = process_group), linewidth = 5.5, alpha = 0.55) +
  geom_point(aes(x = median_months), size = 3.2) +
  geom_text(
    aes(x = median_months, label = median_label, hjust = label_hjust),
    nudge_y = 0.28,
    size = 3.2,
    fontface = "bold",
    color = "gray20"
  ) +
  geom_text(
    aes(x = p90, label = paste0("n=", scales::comma(n))),
    nudge_x = 1.2,
    hjust = 0,
    size = 3,
    color = "gray30"
  ) +
  scale_color_catf(drop = FALSE) +
  scale_x_continuous(
    labels = scales::label_number(accuracy = 1),
    expand = expansion(mult = c(0.02, 0.12))
  ) +
  labs(
    title = "Timeline Duration Summary by Review Process (Solar Projects)",
    subtitle = "Thin bar = p10 to p90, thick bar = IQR (p25 to p75), point = median",
    x = "Duration (months)",
    y = "Review Process",
    color = NULL
  ) +
  theme_catf() +
  theme(legend.position = "none")

out_a <- here(figures_dir, "03_duration_summary_intervals_by_process_solar.png")
ggsave(out_a, fig_duration_solar, width = 10, height = 6, dpi = 300)
cat("  Saved:", out_a, "\n")

# --------------------------
# FIGURE B: CAPACITY DISTRIBUTION VIOLIN+BOXPLOT (SOLAR ONLY)
# Mirrors code/deliverable03/02_capacity.R lines 17-730
# --------------------------

cat("\n=== Figure B: Capacity Distribution (Solar Only) ===\n")

gencap_path <- here("phase1", "data", "analysis", "projects_gencap.parquet")

if (!file.exists(gencap_path)) {
  stop("No gencap parquet found. Run: python code/extract/extract_gencap.py --run regex --parallel 3")
}

gencap_projects <- read_parquet(gencap_path) %>%
  mutate(
    capacity_value_use = coalesce(project_gencap_final_value, project_gencap_value),
    capacity_unit_use  = coalesce(project_gencap_final_unit,  project_gencap_unit),
    has_capacity = !is.na(capacity_value_use) & !is.na(capacity_unit_use)
  )

generation_type_tags <- c(
  "Carbon Capture and Sequestration",
  "Conventional Energy Production - Nuclear",
  "Conventional Energy Production - Other",
  "Renewable Energy Production - Biomass",
  "Renewable Energy Production - Energy Storage",
  "Renewable Energy Production - Geothermal",
  "Renewable Energy Production - Hydrokinetic",
  "Renewable Energy Production - Hydropower",
  "Renewable Energy Production - Other",
  "Renewable Energy Production - Solar",
  "Renewable Energy Production - Wind, Offshore",
  "Renewable Energy Production - Wind, Onshore",
  "Nuclear Technology"
)

# Filter: generation-tagged + solar
gencap_solar <- gencap_projects %>%
  filter(
    map_lgl(project_type, function(pt) {
      if (is.null(pt) || is.na(pt) || pt == "") return(FALSE)
      tags <- tryCatch(jsonlite::fromJSON(as.character(pt)), error = function(e) as.character(pt))
      any(tags %in% generation_type_tags)
    })
  ) %>%
  filter(str_detect(as.character(project_type), fixed(SOLAR_TAG))) %>%
  mutate(
    capacity_mw = case_when(
      capacity_unit_use == "GW" ~ capacity_value_use * 1000,
      capacity_unit_use == "kW" ~ capacity_value_use / 1000,
      TRUE ~ capacity_value_use
    )
  )

cat("Solar generation-tagged projects:", nrow(gencap_solar), "\n")

gencap_reasonable <- gencap_solar %>%
  filter(!is.na(capacity_mw) & capacity_mw > 0 & capacity_mw <= 5000) %>%
  mutate(dataset_source = factor(dataset_source, levels = c("CE", "EA", "EIS")))

cat("Solar projects with reasonable capacity:", nrow(gencap_reasonable), "\n")
cat("By process:\n")
print(count(gencap_reasonable, dataset_source))

process_fill <- c("CE" = catf_light_blue, "EA" = catf_blue, "EIS" = catf_dark_blue)

fig_cap_solar <- gencap_reasonable %>%
  ggplot(aes(x = dataset_source, y = capacity_mw, fill = dataset_source)) +
  geom_violin(alpha = 0.5, color = NA, trim = FALSE) +
  geom_boxplot(
    width = 0.16,
    alpha = 0.9,
    outlier.alpha = 0.15,
    outlier.size = 0.8,
    color = "gray20"
  ) +
  stat_summary(
    fun = median,
    geom = "point",
    shape = 21,
    size = 2.8,
    fill = "white",
    color = "black"
  ) +
  labs(
    title = "Distribution of Generation Capacity by Process Type (Solar Projects)",
    subtitle = "Violin shows density; boxplot shows median and interquartile range (MW, log scale)",
    x = "Process Type",
    y = "Generation Capacity (MW, log scale)"
  ) +
  scale_y_log10(
    breaks = c(1, 5, 10, 50, 100, 500, 1000, 5000),
    labels = label_number(big.mark = ",")
  ) +
  scale_fill_manual(values = process_fill, guide = "none") +
  theme_catf() +
  theme(plot.caption = element_text(size = 8, color = "gray50", hjust = 0))

out_b <- here(figures_dir, "06_capacity_distribution_violin_box_solar.png")
ggsave(out_b, fig_cap_solar, width = 10, height = 6, units = "in", dpi = 300)
cat("  Saved:", out_b, "\n")

cat("\n=== Done ===\n")
cat("Run source('code/deliverable03/04_solar_figures.R') to regenerate.\n")
