# --------------------------
# DELIVERABLE 3: TIMELINE ANALYSIS
# --------------------------
# Harmonized timeline extraction for clean energy projects
# Current inputs:
# - CE: data/analysis/projects_timeline_bert.parquet (BERT final dates)
# - EA: data/analysis/projects_timeline_bert_ea_llm.parquet (LLM dates)
# - EIS: placeholder in 00_setup.R (enable once available)

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable03", "00_setup.R"))

# --------------------------
# LOAD HARMONIZED TIMELINE DATA
# --------------------------

timeline <- load_timeline_for_deliverable3(include_eis = FALSE)
process_levels <- c("CE", "EA", "EIS")

# Derive harmonized timeline fields and process grouping for plotting.
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
    # Year from decision date
    bert_year = as.integer(format(bert_decision_date_final, "%Y")),
    # Duration: decision minus best available start date
    bert_start_date = coalesce(bert_application_date, bert_inferred_application_date, bert_initiation_date_final),
    bert_duration_days = as.numeric(bert_decision_date_final - bert_start_date)
  )

timeline_sources <- process_levels[process_levels %in% unique(as.character(na.omit(timeline$process_group)))]
timeline_sources_label <- paste(timeline_sources, collapse = "+")
if (!("EIS" %in% timeline_sources)) {
  cat("Timeline sources:", timeline_sources_label, "| EIS pending\n")
} else {
  cat("Timeline sources:", timeline_sources_label, "\n")
}
cat("Projects loaded:", nrow(timeline), "\n\n")

timeline |>
  glimpse()

# --------------------------
# TABLE 1: EXTRACTION COVERAGE SUMMARY
# --------------------------

cat("=== Timeline Extraction Coverage ===\n\n")

n_total <- nrow(timeline)
n_has_decision <- sum(!is.na(timeline$bert_decision_date))
n_has_app <- sum(!is.na(timeline$bert_application_date))
n_has_inferred_app <- sum(!is.na(timeline$bert_inferred_application_date))
n_has_review <- sum(timeline$bert_n_review_dates > 0)
n_has_any_start <- sum(!is.na(timeline$bert_start_date))
n_has_duration <- sum(!is.na(timeline$bert_duration_days) & timeline$bert_duration_days >= 0)
n_errors <- sum(!is.na(timeline$bert_error))

coverage_table <- tibble(
  Metric = c(
    sprintf("Total clean energy projects (%s)", timeline_sources_label),
    "Decision date found",
    "Explicit initiation date found",
    "Inferred initiation (earliest review as proxy)",
    "Any start date (explicit or inferred)",
    "Review dates found (at least one)",
    "Duration calculable (decision + start, >= 0 days)",
    "Errors (no dates extracted)"
  ),
  Count = c(n_total, n_has_decision, n_has_app, n_has_inferred_app,
            n_has_any_start, n_has_review, n_has_duration, n_errors),
  Percent = sprintf("%.1f%%", 100 * Count / n_total)
)

# --------------------------
# FIGURE 1: COMPLETE TIMELINE SHARE BY PROCESS (BOXPLOT)
# --------------------------

cat("\nCreating Figure: Complete timeline share by review process...\n")

process_summary <- timeline %>%
  filter(!is.na(process_group)) %>%
  group_by(process_group) %>%
  summarise(
    n_projects = n(),
    n_complete = sum(timeline_complete, na.rm = TRUE),
    share_complete = n_complete / n_projects,
    .groups = "drop"
  )

process_summary <- tibble(process_group = factor(process_levels, levels = process_levels)) %>%
  left_join(process_summary, by = "process_group") %>%
  mutate(
    n_projects = replace_na(n_projects, 0L),
    n_complete = replace_na(n_complete, 0L),
    share_complete = if_else(n_projects > 0, share_complete, NA_real_),
    label = case_when(
      n_projects == 0 ~ "Pending",
      TRUE ~ sprintf("%s/%s (%.0f%%)", scales::comma(n_complete), scales::comma(n_projects), 100 * share_complete)
    )
  )

complete_box <- timeline %>%
  filter(!is.na(process_group)) %>%
  mutate(complete_num = as.numeric(timeline_complete))

fig_complete_share <- ggplot(complete_box, aes(x = process_group, y = complete_num, fill = process_group)) +
  geom_boxplot(outlier.shape = NA, width = 0.55, alpha = 0.35, na.rm = TRUE) +
  stat_summary(fun = mean, geom = "point", size = 3, color = catf_navy) +
  geom_text(
    data = process_summary,
    aes(x = process_group, y = 1.07, label = label),
    inherit.aes = FALSE,
    size = 3,
    color = "gray30"
  ) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = c(0, 1.12),
    breaks = seq(0, 1, by = 0.2)
  ) +
  scale_fill_catf(drop = FALSE) +
  labs(
    title = "Share of Projects with Complete Timelines",
    subtitle = "Boxplot shows project-level completion (0/1); dot is mean share by process",
    x = "Review Process",
    y = "Completion Share"
  ) +
  theme_catf() +
  theme(legend.position = "none")

fig_complete_share
fig_complete_share_path <- here(figures_dir, "03_complete_timeline_share_boxplot.png")
ggsave(fig_complete_share_path, fig_complete_share, width = 9, height = 6, dpi = 300)
cat("  Saved:", fig_complete_share_path, "\n")
print(fig_complete_share)

# --------------------------
# FIGURE 2: PROJECT INITIATION -> DECISION SPANS (FACETED)
# --------------------------

cat("\nCreating Figure: Initiation and decision dates by project...\n")

spans_df <- timeline %>%
  filter(!is.na(process_group)) %>%
  filter(
    !is.na(bert_initiation_date_final),
    !is.na(bert_decision_date_final),
    bert_decision_date_final >= bert_initiation_date_final
  ) %>%
  group_by(process_group) %>%
  arrange(bert_initiation_date_final, bert_decision_date_final, .by_group = TRUE) %>%
  mutate(
    project_order = row_number()
  ) %>%
  ungroup()

segments_df <- spans_df

points_df <- bind_rows(
  spans_df %>%
    transmute(process_group, project_order, date = bert_initiation_date_final, point_type = "Initiation"),
  spans_df %>%
    transmute(process_group, project_order, date = bert_decision_date_final, point_type = "Decision")
)

fig_timeline_spans <- ggplot() +
  geom_segment(
    data = segments_df,
    aes(
      x = bert_initiation_date_final, xend = bert_decision_date_final,
      y = project_order, yend = project_order
    ),
    color = catf_light_blue,
    alpha = 0.45,
    linewidth = 0.35
  ) +
  geom_point(
    data = points_df,
    aes(x = date, y = project_order, color = point_type),
    alpha = 0.6,
    size = 1.0
  ) +
  facet_wrap(~process_group, scales = "free_y", ncol = 1, drop = FALSE) +
  scale_color_manual(values = c("Initiation" = catf_teal, "Decision" = catf_magenta)) +
  labs(
    title = "Project Timelines by Review Process",
    subtitle = "Complete timelines only; projects ordered by initiation date within each process",
    x = "Date",
    y = "Projects (ordered within process)",
    color = NULL
  ) +
  theme_catf() +
  theme(
    legend.position = "top",
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid.major.y = element_blank()
  )

fig_timeline_spans_path <- here(figures_dir, "03_project_timeline_spans_by_process.png")
ggsave(fig_timeline_spans_path, fig_timeline_spans, width = 12, height = 10, dpi = 300)
cat("  Saved:", fig_timeline_spans_path, "\n")
print(fig_timeline_spans)

# --------------------------
# FIGURE 3: PROJECTS BY DECISION YEAR (FACETED BY PROCESS)
# --------------------------

cat("\nCreating Figure: Projects by decision year (by process)...\n")

year_counts <- timeline %>%
  filter(!is.na(process_group), !is.na(bert_year)) %>%
  filter(bert_year >= 2000, bert_year <= 2025) %>%
  count(process_group, bert_year, name = "n_projects")

fig_by_year <- ggplot(year_counts, aes(x = bert_year, y = n_projects)) +
  geom_col(fill = catf_dark_blue, alpha = 0.85) +
  geom_text(
    aes(label = scales::comma(n_projects)),
    vjust = -0.3,
    size = 2.6,
    color = "gray30"
  ) +
  facet_wrap(~process_group, scales = "free_y", ncol = 1, drop = FALSE) +
  scale_x_continuous(breaks = seq(2000, 2025, by = 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)), labels = scales::comma) +
  labs(
    title = "Clean Energy Projects by Decision Year",
    subtitle = "Faceted by NEPA review process",
    x = "Decision Year",
    y = "Number of Projects",
    caption = "Year derived from harmonized final decision date."
  ) +
  theme_catf() +
  theme(
    axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5)
  )

fig_by_year_path <- here(figures_dir, "03_projects_by_year.png")
ggsave(fig_by_year_path, fig_by_year, width = 11, height = 9, dpi = 300)
cat("  Saved:", fig_by_year_path, "\n")
print(fig_by_year)

# --------------------------
# FIGURE 4: TIMELINE STATUS MIX BY PROCESS (ADDITIONAL)
# --------------------------

cat("\nCreating Figure: Timeline status mix by process...\n")

status_by_process <- timeline %>%
  mutate(
    timeline_status_plot = case_when(
      !is.na(bert_initiation_date_final) & !is.na(bert_decision_date_final) ~ "Complete",
      !is.na(bert_initiation_date_final) & is.na(bert_decision_date_final) ~ "Missing decision",
      is.na(bert_initiation_date_final) & !is.na(bert_decision_date_final) ~ "Missing initiation",
      TRUE ~ "Missing both"
    )
  ) %>%
  filter(!is.na(process_group)) %>%
  count(process_group, timeline_status_plot, name = "n") %>%
  group_by(process_group) %>%
  mutate(pct = 100 * n / sum(n)) %>%
  ungroup()

status_levels <- c("Complete", "Missing decision", "Missing initiation", "Missing both")
status_colors <- c(
  "Complete" = catf_teal,
  "Missing decision" = catf_magenta,
  "Missing initiation" = catf_blue,
  "Missing both" = catf_navy
)

status_by_process <- status_by_process %>%
  mutate(timeline_status_plot = factor(timeline_status_plot, levels = status_levels))

fig_status_mix <- ggplot(status_by_process, aes(x = process_group, y = pct, fill = timeline_status_plot)) +
  geom_col(alpha = 0.9) +
  scale_fill_manual(values = status_colors, drop = FALSE) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(labels = scales::label_percent(scale = 1), expand = expansion(mult = c(0, 0.03))) +
  labs(
    title = "Timeline Coverage Mix by Review Process",
    subtitle = "Share of projects with complete vs missing timeline components",
    x = "Review Process",
    y = "Percent of Projects",
    fill = NULL
  ) +
  theme_catf()

fig_status_mix_path <- here(figures_dir, "03_timeline_status_by_process.png")
ggsave(fig_status_mix_path, fig_status_mix, width = 10, height = 6, dpi = 300)
cat("  Saved:", fig_status_mix_path, "\n")
print(fig_status_mix)

# --------------------------
# FIGURE 5: DURATION DISTRIBUTION BY PROCESS (ADDITIONAL)
# --------------------------

cat("\nCreating Figure: Duration distribution by process...\n")

duration_by_process <- timeline %>%
  filter(!is.na(process_group)) %>%
  mutate(duration_months = as.numeric(bert_decision_date_final - bert_initiation_date_final) / 30.44) %>%
  filter(!is.na(duration_months), duration_months >= 0)

duration_p99 <- quantile(duration_by_process$duration_months, 0.99, na.rm = TRUE)
duration_break_step <- dplyr::case_when(
  duration_p99 <= 36 ~ 3,
  duration_p99 <= 96 ~ 6,
  TRUE ~ 12
)
duration_breaks <- seq(
  0,
  ceiling(duration_p99 / duration_break_step) * duration_break_step,
  by = duration_break_step
)

fig_duration_by_process <- ggplot(duration_by_process, aes(x = process_group, y = duration_months, fill = process_group)) +
  geom_violin(alpha = 0.25, trim = FALSE, color = NA) +
  geom_boxplot(
    width = 0.18,
    outlier.alpha = 0.2,
    outlier.size = 0.6,
    linewidth = 0.4,
    fill = "white",
    color = catf_navy
  ) +
  stat_summary(fun = median, geom = "point", shape = 21, size = 2.2, fill = catf_navy, color = "white") +
  coord_cartesian(ylim = c(0, duration_p99)) +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(
    breaks = duration_breaks,
    labels = scales::label_number(accuracy = 1)
  ) +
  scale_fill_catf(drop = FALSE) +
  labs(
    title = "Project Duration Distribution by Review Process",
    subtitle = "Violin + boxplot overlay; complete timelines only (y-axis capped at p99)",
    x = "Review Process",
    y = "Duration (months)"
  ) +
  theme_catf() +
  theme(legend.position = "none")

fig_duration_by_process_path <- here(figures_dir, "03_duration_by_process_boxplot.png")
ggsave(fig_duration_by_process_path, fig_duration_by_process, width = 10, height = 6, dpi = 300)
cat("  Saved:", fig_duration_by_process_path, "\n")
print(fig_duration_by_process)

# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Timeline Analysis Summary ===\n\n")
cat("Decision date coverage:", sprintf("%.0f%%", 100 * n_has_decision / n_total), "\n")
cat("Explicit initiation coverage:", sprintf("%.0f%%", 100 * n_has_app / n_total), "\n")
cat("Inferred initiation coverage:", sprintf("%.0f%%", 100 * n_has_inferred_app / n_total), "\n")
cat("Duration calculable:", sprintf("%.0f%%", 100 * n_has_duration / n_total), "\n")
cat("Median dates per project:", median(timeline$bert_n_dates_found), "\n\n")

cat("Files saved to:", figures_dir, "\n")
cat("Tables saved to:", tables_dir, "\n")

# --------------------------
# BERT TIMELINE EXAMPLES (for client review)
# --------------------------
# Source: code/exploratory/timeline/01_compare_decisions.R
# Shows 6 curated project examples from BERT v8 classification

cat("\n=== BERT Timeline Examples ===\n\n")

# --- helpers (JSON parsing) ---

safe_fromJSON <- function(x) {
  tryCatch(fromJSON(x, flatten = TRUE), error = function(e) NULL)
}

normalize_parsed <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.data.frame(x)) return(as_tibble(x))
  if (is.list(x)) {
    if (!is.null(names(x)) && length(names(x)) > 0) {
      return(as_tibble(x))
    }
    return(bind_rows(lapply(x, as_tibble)))
  }
  NULL
}

extract_contexts <- function(df, json_col, model_label) {
  df %>%
    mutate(parsed = map(.data[[json_col]], safe_fromJSON)) %>%
    mutate(parsed = map(parsed, normalize_parsed)) %>%
    select(project_id, project_title, lead_agency, parsed) %>%
    unnest(parsed) %>%
    mutate(model = model_label) %>%
    select(project_id, project_title, lead_agency, model, type, date, source, confidence, everything())
}

# --- load BERT v8 results ---

bert_path <- here("data", "analysis", "test50_bert_v8.parquet")
if (file.exists(bert_path)) {
  bert <- read_parquet(bert_path)
  bert_ctx <- extract_contexts(bert, "bert_dates_json", "bert")

  cat("BERT v8 results loaded:", nrow(bert), "projects,",
      nrow(bert_ctx), "date contexts\n\n")

  # --- curated project examples ---
  example_ids <- c(
    "3e3bb9f5-f5ab-651d-b2d1-50ec99d99db0",
    "46f4da85-af1c-0e66-a706-9a7292dd9689",
    "824ba268-8ddf-a34f-f9a7-625e7727c242",
    "f2812da0-16c5-fbd1-9e16-10bf8e67c514",
    "dec68c6f-da24-f178-7bf9-30dcd886fb12",
    "5c512493-33a9-ff2c-5f13-3a8d55464b93"
  )

  examples_list <- list()

  for (i in seq_along(example_ids)) {
    ex <- bert_ctx %>%
      filter(project_id == example_ids[i]) %>%
      select(project_title, type, date, source) %>%
      arrange(date) %>%
      mutate(example = i)

    examples_list[[i]] <- ex

    cat(sprintf("Example %d (%s): %d date contexts\n",
                i, example_ids[i], nrow(ex)))
  }

  # Combine all examples into one table
  examples_all <- bind_rows(examples_list)
  examples_all |> glimpse()

  # Save combined CSV
  examples_csv_path <- here(tables_dir, "03_bert_client_examples.csv")
  write_csv(examples_all, examples_csv_path)
  cat("\nSaved combined examples:", examples_csv_path, "\n")

  # Save individual CSVs
  for (i in seq_along(examples_list)) {
    ex_path <- here(tables_dir, sprintf("03_bert_example%d.csv", i))
    write_csv(examples_list[[i]], ex_path)
  }
  cat("Saved individual example CSVs to:", tables_dir, "\n")

  # Write to Google Sheets
  gs_url <- "https://docs.google.com/spreadsheets/d/1HuvVNDiPAG3WegTy58yn_LLUQ8RnSFwTg0BeabcyM08/edit?usp=sharing"

  #for (i in seq_along(examples_list)) {
  #  sheet_write(
  #    data = examples_list[[i]],
  #    ss = gs_url,
  #    sheet = sprintf("example%d", i)
  #  )
  #}
  cat("Written examples to Google Sheet\n")
} else {
  cat("Skipping BERT examples: missing file ", bert_path, "\n", sep = "")
}
