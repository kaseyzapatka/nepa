#!/usr/bin/env Rscript
# factsheet_figures.R
# Regenerates all factsheet figures with client-requested title/label updates.
# Run with: Rscript phase1/code/factsheet_figures.R
# Output:   phase1/output/factsheet/figures/

library(here)
library(arrow)
library(tidyverse)
library(jsonlite)
library(scales)
library(zoo)
library(ggalluvial)

out_dir <- here("phase1", "output", "factsheet", "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# CATF brand colors
# ---------------------------------------------------------------------------
catf_dark_blue  <- "#0047BB"
catf_blue       <- "#00B5E2"
catf_magenta    <- "#C22A90"
catf_purple     <- "#75246C"
catf_lime       <- "#93D500"
catf_teal       <- "#00AE8D"
catf_light_blue <- "#8AB7E9"
catf_navy       <- "#002169"
catf_palette    <- c("#0047BB","#00B5E2","#00AE8D","#93D500","#C22A90","#75246C","#8AB7E9","#002169")
catf_sequential <- c("#8AB7E9","#00B5E2","#0047BB","#002169")

theme_catf <- function(base_size = 11, base_family = "Helvetica") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      plot.title       = element_text(face = "bold", size = rel(1.2), color = catf_navy,
                                      margin = margin(b = 10)),
      plot.subtitle    = element_text(size = rel(0.9), color = catf_dark_blue,
                                      margin = margin(b = 10)),
      plot.caption     = element_text(size = rel(0.8), color = "gray50", hjust = 0),
      axis.title       = element_text(size = rel(0.9), color = catf_navy),
      axis.text        = element_text(size = rel(0.85), color = "gray30"),
      axis.line        = element_line(color = "gray70", linewidth = 0.3),
      legend.title     = element_text(face = "bold", size = rel(0.9), color = catf_navy),
      legend.text      = element_text(size = rel(0.85), color = "gray30"),
      legend.position  = "bottom",
      legend.key.size  = unit(0.8, "lines"),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      strip.text       = element_text(face = "bold", size = rel(0.9), color = catf_navy),
      strip.background = element_rect(fill = "gray95", color = NA),
      plot.margin      = margin(15, 15, 10, 10)
    )
}
scale_color_catf <- function(...) scale_color_manual(values = catf_palette, ...)
scale_fill_catf  <- function(...) scale_fill_manual(values = catf_palette, ...)
theme_set(theme_catf())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

explode_column <- function(df, col_name) {
  df %>%
    mutate(!!col_name := sapply(.data[[col_name]], function(x) {
      if (is.null(x) || length(x) == 0 || (is.character(x) && x == "")) return(NA_character_)
      if (is.character(x) && grepl("^\\[", x)) {
        parsed <- tryCatch(jsonlite::fromJSON(x), error = function(e) x)
        if (is.character(parsed) && length(parsed) > 1) return(paste(parsed, collapse = "|"))
        return(as.character(parsed))
      }
      if (is.list(x)) return(paste(unlist(x), collapse = "|"))
      return(as.character(x))
    })) %>%
    separate_rows(!!col_name, sep = "\\|")
}

parse_jsonish_vector <- function(x) {
  if (is.null(x) || is.na(x) || x == "") return(character(0))
  if (is.character(x) && str_detect(x, "^\\[")) {
    parsed <- tryCatch(fromJSON(x), error = function(e) NULL)
    if (!is.null(parsed) && length(parsed) > 0) return(str_trim(as.character(parsed)))
  }
  vals_pipe <- str_split(as.character(x), "\\s*\\|\\s*")[[1]]
  vals_pipe <- str_trim(vals_pipe)
  vals_pipe <- vals_pipe[vals_pipe != ""]
  if (length(vals_pipe) == 0) return(character(0))
  vals <- map(vals_pipe, ~ {
    token <- .x
    if (str_detect(token, "^\\[")) {
      parsed_token <- tryCatch(fromJSON(token), error = function(e) NULL)
      if (!is.null(parsed_token) && length(parsed_token) > 0)
        return(str_trim(as.character(parsed_token)))
    }
    if (str_detect(token, ",")) {
      parts <- str_split(token, ",\\s*")[[1]]
      parts <- str_trim(parts)
      return(parts[parts != ""])
    }
    token
  }) %>% unlist(use.names = FALSE) %>% str_trim()
  vals[vals != ""]
}

map_agency_to_department <- function(agency) {
  case_when(
    str_detect(agency, "^Department of Energy")           ~ "Department of Energy",
    str_detect(agency, "^Department of the Interior")     ~ "Department of the Interior",
    str_detect(agency, "^Department of Agriculture")      ~ "Department of Agriculture",
    str_detect(agency, "^Department of Defense")          ~ "Department of Defense",
    str_detect(agency, "^Department of Homeland Security") ~ "Department of Homeland Security",
    str_detect(agency, "^Department of Transportation")   ~ "Department of Transportation",
    str_detect(agency, "^Department of Commerce")         ~ "Department of Commerce",
    str_detect(agency, "^Major Independent Agencies")     ~ "Major Independent Agencies",
    str_detect(agency, "^Other Independent Agencies")     ~ "Other Independent Agencies",
    TRUE ~ agency
  )
}

# ---------------------------------------------------------------------------
# Shared data: all projects
# ---------------------------------------------------------------------------
message("Loading projects_combined.parquet...")
projects     <- read_parquet(here("phase1", "data", "analysis", "projects_combined.parquet"))
clean_energy <- projects %>% filter(project_energy_type == "Clean")
message("  Total: ", nrow(projects), " | Clean: ", nrow(clean_energy))

# ---------------------------------------------------------------------------
# Fig 1 — Energy type composition (02_energy_type_composition.png)
# ---------------------------------------------------------------------------
message("\n--- Fig 1: Energy type composition ---")

fig1_data <- projects %>%
  group_by(project_energy_type, process_type) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(project_energy_type) %>%
  mutate(total_energy_type = sum(n), pct = 100 * n / total_energy_type) %>%
  ungroup()

fig1_totals <- fig1_data %>%
  mutate(project_energy_type = if_else(project_energy_type == "Clean", "Decarbonized",
                                       project_energy_type)) %>%
  distinct(project_energy_type, total_energy_type)

fig1 <- fig1_data %>%
  mutate(project_energy_type = if_else(project_energy_type == "Clean", "Decarbonized",
                                       project_energy_type)) %>%
  ggplot(aes(x = reorder(project_energy_type, total_energy_type), y = pct, fill = process_type)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = ifelse(pct > 5, paste0(round(pct, 0), "%"), "")),
    position = position_stack(vjust = 0.5),
    color = "white", size = 3.5, fontface = "bold"
  ) +
  geom_text(
    data = fig1_totals,
    aes(x = reorder(project_energy_type, total_energy_type), y = 101,
        label = scales::comma(total_energy_type)),
    inherit.aes = FALSE, hjust = 0, size = 3, color = "gray30"
  ) +
  coord_flip() +
  labs(
    title = "Reviews for Decarbonization Technologies Use Categorical Exclusions More Than\nFossil Technologies or Other Types of Federal Actions",
    x       = NULL,
    y       = "Share of Reviews",
    fill    = "Review Type",
    caption = str_wrap(paste0(
      "Note: NEPA review processes: CE (Categorical Exclusion), EA (Environmental Assessment), ",
      "EIS (Environmental Impact Statement). ",
      "Percentages calculated within each energy type category. ",
      "Percentages below 5% omitted for clarity."
    ), width = 150)
  ) +
  scale_y_continuous(labels = percent_format(scale = 1), expand = expansion(mult = c(0, 0.08))) +
  scale_fill_catf() +
  theme_catf()

ggsave(file.path(out_dir, "02_energy_type_composition.png"), fig1, width = 10, height = 5, dpi = 300)
message("  Saved: 02_energy_type_composition.png")

# ---------------------------------------------------------------------------
# Fig 2 — Agency process: DOE + BLM (02_agency_process.png)
# ---------------------------------------------------------------------------
message("\n--- Fig 2: Agency process (DOE + BLM) ---")

agency_data_raw <- clean_energy %>%
  explode_column("lead_agency") %>%
  filter(!is.na(lead_agency) & lead_agency != "") %>%
  mutate(department = project_department)

agency_harmonized <- clean_energy %>%
  explode_column("lead_agency") %>%
  filter(!is.na(lead_agency) & lead_agency != "") %>%
  mutate(
    lead_agency_exp = lead_agency %>%
      str_replace("^DOE\\s*-\\s*",  "Department of Energy - ") %>%
      str_replace("^DOI\\s*-\\s*",  "Department of the Interior - ") %>%
      str_replace("^USDA\\s*-\\s*", "Department of Agriculture - ") %>%
      str_replace("^DOD\\s*-\\s*",  "Department of Defense - ") %>%
      str_replace("^DOT\\s*-\\s*",  "Department of Transportation - "),
    department = case_when(
      str_detect(lead_agency_exp, " - ") ~
        str_extract(lead_agency_exp, "^.+?(?= - )") %>% str_trim(),
      str_detect(lead_agency_exp, "^Department of ")       ~ lead_agency_exp,
      str_detect(lead_agency_exp, "^Major Independent ")   ~ "Major Independent Agencies",
      str_detect(lead_agency_exp, "^Other Independent ")   ~ "Other Independent Agencies",
      str_detect(lead_agency_exp, "^General Services Admin") ~ "General Services Administration",
      TRUE ~ project_department
    ),
    lead_agency_harmonized = if_else(
      str_detect(lead_agency_exp, " - "),
      str_extract(lead_agency_exp, "(?<= - ).+$") %>% str_trim(),
      lead_agency_exp
    )
  ) %>%
  select(-lead_agency_exp)

coverage_verified <- bind_rows(
  agency_data_raw %>%
    filter(department == "Department of Energy") %>%
    mutate(agency_label = "Dept. of Energy (DOE)", dept_label = "Department of Energy"),
  agency_harmonized %>%
    filter(str_detect(lead_agency_harmonized,
                      regex("bureau of land management", ignore_case = TRUE))) %>%
    mutate(
      agency_label = "Bureau of Land Management (BLM)",
      dept_label   = "Bureau of Land Management (BLM)"
    )
) %>%
  count(dept_label, agency_label, process_type) %>%
  group_by(agency_label) %>%
  mutate(total = sum(n), share = n / total) %>%
  ungroup()

coverage_totals <- coverage_verified %>% distinct(dept_label, agency_label, total)

dept_order_cv  <- c("Department of Energy", "Bureau of Land Management (BLM)")
dept_labels_cv <- c(
  "Department of Energy"            = "Department\nof Energy",
  "Bureau of Land Management (BLM)" = "Bureau of Land\nManagement (BLM)"
)

coverage_verified <- coverage_verified %>%
  mutate(dept_label = factor(dept_label, levels = dept_order_cv,
                             labels = dept_labels_cv[dept_order_cv]))
coverage_totals <- coverage_totals %>%
  mutate(dept_label = factor(dept_label, levels = dept_order_cv,
                             labels = dept_labels_cv[dept_order_cv]))

fig2 <- coverage_verified %>%
  ggplot(aes(x = dept_label, y = share, fill = process_type)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = ifelse(share >= 0.03, scales::percent(share, accuracy = 1), "")),
    position = position_stack(vjust = 0.5), size = 3, color = "white"
  ) +
  geom_text(
    data = coverage_totals,
    aes(x = dept_label, y = 1.02, label = scales::comma(total)),
    inherit.aes = FALSE, hjust = 0, size = 3, color = "gray30"
  ) +
  coord_flip(clip = "off") +
  labs(
    x       = NULL,
    y       = "Share of Reviews",
    fill    = "Review Type",
    title   = "Both DOE and BLM Overwhelmingly Use Categorical Exclusions for Decarbonization-Related\nFederal Actions",
    caption = str_wrap("Note: Numbers to the right show total project count. Only DOE and BLM have complete CE, EA, and EIS coverage in NEPATEC 2.0.", width = 160)
  ) +
  scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.08))) +
  scale_fill_manual(values = c("CE" = catf_dark_blue, "EA" = catf_teal, "EIS" = catf_magenta)) +
  theme_minimal() +
  theme(
    legend.position   = "bottom",
    axis.text.x       = element_blank(),
    axis.ticks.x      = element_blank(),
    axis.line.x       = element_blank(),
    axis.text.y       = element_text(size = 9),
    panel.grid.major.y = element_blank(),
    plot.caption      = element_text(size = 8, color = "gray40", hjust = 0),
    plot.margin       = margin(10, 30, 10, 10)
  )

ggsave(file.path(out_dir, "02_agency_process.png"), fig2, width = 10, height = 5, dpi = 300)
message("  Saved: 02_agency_process.png")

# ---------------------------------------------------------------------------
# Timeline data: CE + EA + EIS  (shared for Figs 3 & 4)
# ---------------------------------------------------------------------------
message("\n--- Loading CE/EA/EIS timelines ---")

timeline_ce_path  <- here("phase1", "data", "analysis", "projects_timeline_bert.parquet")
timeline_ea_path  <- here("phase1", "data", "analysis", "projects_timeline_bert_ea_llm.parquet")
timeline_eis_path <- here("phase1", "data", "analysis", "projects_timeline_bert_eis_llm.parquet")

for (p in c(timeline_ce_path, timeline_ea_path, timeline_eis_path)) {
  if (!file.exists(p)) stop("Missing timeline file: ", p)
}

ce_df  <- read_parquet(timeline_ce_path)  %>% mutate(timeline_input_file = basename(timeline_ce_path))
ea_df  <- read_parquet(timeline_ea_path)  %>% mutate(timeline_input_file = basename(timeline_ea_path))
eis_df <- read_parquet(timeline_eis_path) %>% mutate(timeline_input_file = basename(timeline_eis_path))

timeline_raw <- bind_rows(ce_df, ea_df, eis_df)
if (!"dataset_source" %in% names(timeline_raw)) {
  timeline_raw <- timeline_raw %>% mutate(dataset_source = NA_character_)
}

timeline_harmonized <- timeline_raw %>%
  mutate(
    dataset_source = toupper(as.character(dataset_source)),
    timeline_initiation_date_final = as.Date(case_when(
      dataset_source %in% c("EA", "EIS") ~ llm_initiation_date,
      TRUE ~ bert_initiation_date_final
    )),
    timeline_decision_date_final = as.Date(case_when(
      dataset_source %in% c("EA", "EIS") ~ llm_decision_date,
      TRUE ~ bert_decision_date_final
    )),
    bert_initiation_date_final     = timeline_initiation_date_final,
    bert_decision_date_final       = timeline_decision_date_final,
    bert_decision_date             = timeline_decision_date_final,
    bert_application_date = if_else(
      dataset_source %in% c("EA", "EIS"),
      timeline_initiation_date_final,
      as.Date(bert_application_date)
    ),
    bert_inferred_application_date = if_else(
      dataset_source %in% c("EA", "EIS"),
      as.Date(NA),
      as.Date(bert_inferred_application_date)
    ),
    bert_timeline_status = case_when(
      !is.na(timeline_decision_date_final) & !is.na(timeline_initiation_date_final) ~ "complete",
      !is.na(timeline_decision_date_final) & is.na(timeline_initiation_date_final)  ~ "missing_initiation",
      is.na(timeline_decision_date_final)  & !is.na(timeline_initiation_date_final) ~ "missing_decision",
      TRUE ~ "no_dates"
    ),
    decision_year = as.integer(format(timeline_decision_date_final, "%Y"))
  )

message("  Timeline records: ", nrow(timeline_harmonized))

# ---------------------------------------------------------------------------
# Fig 3 — Solar duration (03_duration_summary_intervals_by_process_solar.png)
# ---------------------------------------------------------------------------
message("\n--- Fig 3: Solar duration ---")

SOLAR_TAG      <- "Renewable Energy Production - Solar"
process_levels <- c("CE", "EA", "EIS")

timeline_for_solar <- timeline_harmonized %>%
  mutate(
    source_for_plot = toupper(as.character(coalesce(dataset_source, process_type))),
    process_group   = factor(source_for_plot, levels = process_levels),
    bert_decision_date             = as.Date(bert_decision_date),
    bert_application_date          = as.Date(bert_application_date),
    bert_inferred_application_date = as.Date(bert_inferred_application_date),
    bert_initiation_date_final     = as.Date(bert_initiation_date_final),
    bert_decision_date_final       = as.Date(bert_decision_date_final),
    timeline_complete  = !is.na(bert_initiation_date_final) & !is.na(bert_decision_date_final),
    bert_year          = as.integer(format(bert_decision_date_final, "%Y")),
    decision_year      = coalesce(decision_year, bert_year),
    bert_start_date    = coalesce(bert_application_date, bert_inferred_application_date,
                                  bert_initiation_date_final),
    bert_duration_days = as.numeric(bert_decision_date_final - bert_start_date)
  )

timeline_solar <- timeline_for_solar %>%
  filter(str_detect(as.character(project_type), fixed(SOLAR_TAG)))

duration_complete_solar <- timeline_solar %>%
  filter(!is.na(process_group), timeline_complete) %>%
  mutate(duration_months = bert_duration_days / 30.44) %>%
  filter(!is.na(duration_months), duration_months >= 0)

duration_summary_solar <- duration_complete_solar %>%
  group_by(process_group) %>%
  summarise(
    n             = n(),
    p10           = quantile(duration_months, 0.10, na.rm = TRUE),
    p25           = quantile(duration_months, 0.25, na.rm = TRUE),
    median_months = median(duration_months, na.rm = TRUE),
    p75           = quantile(duration_months, 0.75, na.rm = TRUE),
    p90           = quantile(duration_months, 0.90, na.rm = TRUE),
    .groups       = "drop"
  ) %>%
  mutate(
    median_label = paste0(
      process_group, ": ~", round(median_months),
      ifelse(round(median_months) == 1, " month", " months")
    ),
    label_hjust = if_else(process_group == "CE", 0, 0.5)
  )

message("  Solar projects in duration summary: ", sum(duration_summary_solar$n))
print(duration_summary_solar)

fig3 <- ggplot(duration_summary_solar, aes(y = process_group, color = process_group)) +
  geom_segment(aes(x = p10, xend = p90, yend = process_group), linewidth = 1.8, alpha = 0.35) +
  geom_segment(aes(x = p25, xend = p75, yend = process_group), linewidth = 5.5, alpha = 0.55) +
  geom_point(aes(x = median_months), size = 3.2) +
  geom_text(
    aes(x = median_months, label = median_label, hjust = label_hjust),
    nudge_y = 0.28, size = 3.2, fontface = "bold", color = "gray20"
  ) +
  geom_text(
    aes(x = p90, label = paste0("n=", scales::comma(n))),
    nudge_x = 1.2, hjust = 0, size = 3, color = "gray30"
  ) +
  scale_color_catf(drop = FALSE) +
  scale_x_continuous(
    labels = scales::label_number(accuracy = 1),
    expand = expansion(mult = c(0.02, 0.12))
  ) +
  labs(
    title = "NEPA Reviews for Solar Projects\nAre Completed More Quickly Than for Decarbonization Projects as a Whole",
    x     = "Duration (months)",
    y     = "Review Process",
    color = NULL
  ) +
  theme_catf() +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "03_duration_summary_intervals_by_process_solar.png"),
       fig3, width = 10, height = 6, dpi = 300)
message("  Saved: 03_duration_summary_intervals_by_process_solar.png")

# ---------------------------------------------------------------------------
# Fig 4 — Solar capacity distribution (06_capacity_distribution_violin_box_solar.png)
# ---------------------------------------------------------------------------
message("\n--- Fig 4: Solar capacity distribution ---")

gencap_path <- here("phase1", "data", "analysis", "projects_gencap.parquet")
if (!file.exists(gencap_path)) {
  stop("Missing projects_gencap.parquet. Run: python code/extract/extract_gencap.py --run")
}

gencap_projects <- read_parquet(gencap_path) %>%
  mutate(
    capacity_value_use = coalesce(project_gencap_final_value, project_gencap_value),
    capacity_unit_use  = coalesce(project_gencap_final_unit,  project_gencap_unit),
    has_capacity       = !is.na(capacity_value_use) & !is.na(capacity_unit_use)
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

gencap_reasonable <- gencap_solar %>%
  filter(!is.na(capacity_mw) & capacity_mw > 0 & capacity_mw <= 5000) %>%
  mutate(dataset_source = factor(dataset_source, levels = c("CE", "EA", "EIS")))

message("  Solar projects with reasonable capacity: ", nrow(gencap_reasonable))
print(count(gencap_reasonable, dataset_source))

process_fill_solar <- c("CE" = catf_light_blue, "EA" = catf_blue, "EIS" = catf_dark_blue)

fig4 <- gencap_reasonable %>%
  ggplot(aes(x = dataset_source, y = capacity_mw, fill = dataset_source)) +
  geom_violin(alpha = 0.5, color = NA, trim = FALSE) +
  geom_boxplot(
    width         = 0.16,
    alpha         = 0.9,
    outlier.alpha = 0.15,
    outlier.size  = 0.8,
    color         = "gray20"
  ) +
  stat_summary(fun = median, geom = "point", shape = 21, size = 2.8,
               fill = "white", color = "black") +
  labs(
    title   = "Solar Projects With Higher Generation Capacities Are More Likely to Go Through More Intensive\nLevels of NEPA Review",
    x       = "Process Type",
    y       = "Generation Capacity (MW, log scale)",
    caption = str_wrap("Note: Violin illustrate density, while boxplots illustrate median and interquartile range. The megawatt (MW) on the y-axis is logged for visual comparison.", width = 160)
  ) +
  scale_y_log10(
    breaks = c(1, 5, 10, 50, 100, 500, 1000, 5000),
    labels = label_number(big.mark = ",")
  ) +
  scale_fill_manual(values = process_fill_solar, guide = "none") +
  theme_catf() +
  theme(plot.caption = element_text(size = 8, color = "gray50", hjust = 0))

ggsave(file.path(out_dir, "06_capacity_distribution_violin_box_solar.png"),
       fig4, width = 10, height = 6, dpi = 300)
message("  Saved: 06_capacity_distribution_violin_box_solar.png")

# ---------------------------------------------------------------------------
# Fig 5 — Review duration (02_duration.png)
# ---------------------------------------------------------------------------
message("\n--- Fig 5: Review duration ---")

reviews_path <- here("phase1", "data", "analysis", "projects_reviews.parquet")
if (!file.exists(reviews_path)) stop("Missing projects_reviews.parquet")

reviews_raw <- read_parquet(reviews_path) %>% as_tibble()

review_type_levels <- c("Standard", "Programmatic", "Tiered")
review_type_colors <- c("Standard" = "gray75", "Programmatic" = "#0047BB", "Tiered" = "#00B5E2")

reviews <- reviews_raw %>%
  mutate(
    review_type  = factor(str_to_sentence(project_review_type), levels = review_type_levels),
    process_type = factor(dataset_source, levels = c("EA", "EIS"))
  )

# EA/EIS timelines with LLM dates (ea_df and eis_df already loaded above)
timeline_d2 <- bind_rows(
  ea_df  %>% select(project_id, llm_initiation_date, llm_decision_date),
  eis_df %>% select(project_id, llm_initiation_date, llm_decision_date)
) %>%
  distinct(project_id, .keep_all = TRUE) %>%
  mutate(
    initiation_date = as.Date(llm_initiation_date),
    decision_date   = as.Date(llm_decision_date)
  ) %>%
  select(project_id, initiation_date, decision_date)

# Patch with targeted re-adjudication if available
targeted_path <- here("phase1", "data", "analysis", "projects_timeline_targeted_llm.parquet")
if (file.exists(targeted_path)) {
  targeted <- read_parquet(targeted_path) %>%
    select(project_id,
           targeted_initiation_date = llm_initiation_date,
           targeted_decision_date   = llm_decision_date)
  timeline_d2 <- timeline_d2 %>%
    left_join(targeted, by = "project_id") %>%
    mutate(
      initiation_date = coalesce(as.Date(targeted_initiation_date), initiation_date),
      decision_date   = coalesce(as.Date(targeted_decision_date),   decision_date)
    ) %>%
    select(-targeted_initiation_date, -targeted_decision_date)
  message("  Targeted re-adjudication applied: ", nrow(targeted), " projects")
}

# Manual date overrides — matches deliverable02 analysis for consistency
tl_full_for_noi <- bind_rows(ea_df, eis_df) %>% distinct(project_id, .keep_all = TRUE)
noi_cf2 <- as.Date(
  tl_full_for_noi %>%
    filter(project_id == "cf2fbe90d43ac57a9460fa857f34af6c") %>%
    pull(noi_publication_date) %>%
    first()
)

manual_overrides <- tibble(
  project_id          = c(
    "cf2fbe90d43ac57a9460fa857f34af6c",
    "f95ec9530b352e3dd46e6473cb80dccf",
    "49cdaa3ff2e6c505c6822e8e9803eb9b",
    "4af8ad4f47941e4ccb53fe4349c258c3",
    "00d09887554d7ab68e49e9ab628583bf",
    "8d13822f3d8b469efcdb2706caa463c7",
    "6890cacf404f0068be5c1e94470e6c58",
    "5445a80334ce78493711d6bc3d24fd81"
  ),
  override_initiation = as.Date(c(noi_cf2, NA, NA, "1993-01-01", NA, NA, NA, NA)),
  override_decision   = as.Date(c(NA, "2019-04-01", "2023-05-01", "1995-09-01",
                                   "2025-06-01", "2022-03-01", "2022-02-25", "2012-09-01"))
)

timeline_d2 <- timeline_d2 %>%
  left_join(manual_overrides, by = "project_id") %>%
  mutate(
    initiation_date = coalesce(override_initiation, initiation_date),
    decision_date   = coalesce(override_decision,   decision_date)
  ) %>%
  select(-override_initiation, -override_decision)

reviews_tl <- reviews %>%
  left_join(timeline_d2, by = "project_id") %>%
  mutate(
    duration_days   = as.numeric(decision_date - initiation_date),
    duration_months = duration_days / 30.44,
    has_duration    = !is.na(duration_days) & duration_days > 0
  )

duration_data <- reviews_tl %>% filter(has_duration)

dur_n <- duration_data %>%
  group_by(process_type, review_type) %>%
  summarise(n = n(), median_days = median(duration_days), .groups = "drop") %>%
  mutate(n_label = paste0("n = ", n))

p97 <- quantile(duration_data$duration_days, 0.97, na.rm = TRUE)

fig5 <- ggplot(
  duration_data,
  aes(x = review_type, y = duration_days, fill = review_type, color = review_type)
) +
  geom_violin(
    alpha = 0.25, trim = FALSE, color = NA,
    data  = duration_data %>% group_by(process_type, review_type) %>%
      filter(n() >= 10) %>% ungroup()
  ) +
  geom_boxplot(
    width         = 0.25,
    outlier.shape = NA,
    fill          = "white",
    color         = catf_navy,
    linewidth     = 0.5,
    alpha         = 0.8
  ) +
  geom_jitter(width = 0.12, size = 1.3, alpha = 0.4, show.legend = FALSE) +
  stat_summary(fun = median, geom = "point", shape = 18, size = 3.5,
               color = catf_navy, show.legend = FALSE) +
  geom_text(
    data = dur_n,
    aes(x = review_type, y = p97 * 1.06, label = n_label),
    inherit.aes = FALSE, size = 3, color = "gray40", vjust = 0
  ) +
  geom_text(
    data = dur_n,
    aes(x = review_type, y = median_days,
        label = sprintf("median\n%s d", comma(round(median_days)))),
    inherit.aes = FALSE, hjust = -0.85, size = 2.6, color = catf_navy, fontface = "italic"
  ) +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(values  = review_type_colors) +
  scale_color_manual(values = review_type_colors) +
  coord_cartesian(ylim = c(0, p97 * 1.15)) +
  scale_y_continuous(labels = comma) +
  labs(
    title   = "Tiered EISs Take Less Time Than Standard EISs",
    x       = NULL,
    y       = "Duration (days)",
    caption = str_wrap(paste0(
      "Note: Box = IQR; diamond = median; points = individual projects (jittered). ",
      "Violin shown only for groups with n ≥ 10. Y-axis capped at 97th percentile. ",
      "Days from initiation to decision (reviews with complete timelines only)."
    ), width = 180)
  ) +
  theme_catf() +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "02_duration.png"), fig5, width = 11, height = 7, dpi = 300)
message("  Saved: 02_duration.png")

# ---------------------------------------------------------------------------
# Fig 6 — Department collaboration hubs, EIS only (fig_department_collaboration_hubs.png)
# ---------------------------------------------------------------------------
message("\n--- Fig 6: Collaboration hubs (EIS only) ---")

# project_multi_department (metadata flag) is what drives lead_agency having multiple entries.
# project_multi_agency (text-signal, from coagency_projects.parquet) does NOT guarantee
# lead_agency has >1 agency — those projects still show only 1 department after parsing.
# So we use project_multi_department throughout; fall back to all process types if EIS has 0.
multi_dept_eis <- clean_energy %>%
  filter(project_multi_department == TRUE, process_type == "EIS")
message("  EIS multi-department projects (metadata): ", nrow(multi_dept_eis))

fig6_scope_label <- "EIS projects only"
if (nrow(multi_dept_eis) == 0) {
  warning("No EIS multi-department projects in metadata — falling back to all process types")
  multi_dept_eis <- clean_energy %>% filter(project_multi_department == TRUE)
  fig6_scope_label <- "all process types"
  message("  Fallback: all process types, n = ", nrow(multi_dept_eis))
}

message("  Final Fig 6 scope: ", fig6_scope_label)

department_projects_eis <- multi_dept_eis %>%
  mutate(
    department_list = map(lead_agency, ~ {
      agencies <- parse_jsonish_vector(.x)
      depts    <- unique(map_chr(agencies, map_agency_to_department))
      sort(depts[depts != ""])
    })
  ) %>%
  filter(lengths(department_list) >= 2)

message("  Projects with ≥2 departments: ", nrow(department_projects_eis))

if (nrow(department_projects_eis) == 0) {
  stop("Fig 6: no projects with ≥2 identifiable departments after all fallbacks — cannot generate figure")
}

department_pairs_eis <- department_projects_eis %>%
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

if (nrow(department_pairs_eis) == 0 || !all(c("department_1", "department_2") %in% names(department_pairs_eis))) {
  stop("Fig 6: unnest produced no department pairs — check lead_agency values")
}

department_pairs_eis <- bind_rows(
  department_pairs_eis,
  department_pairs_eis %>% rename(department_1 = department_2, department_2 = department_1)
)

pair_counts_eis <- department_pairs_eis %>%
  count(department_1, department_2, name = "shared_projects", sort = TRUE)

tbl_collab_hubs_eis <- bind_rows(
  pair_counts_eis %>% transmute(department = department_1, partner = department_2, shared_projects),
  pair_counts_eis %>% transmute(department = department_2, partner = department_1, shared_projects)
) %>%
  group_by(department) %>%
  summarise(
    `Unique partner departments` = n_distinct(partner),
    `Collaborative project ties` = sum(shared_projects),
    `Most frequent partner`      = partner[which.max(shared_projects)],
    `Projects with top partner`  = max(shared_projects),
    `Bridge score`               = round(`Unique partner departments` * log1p(`Collaborative project ties`), 2),
    .groups = "drop"
  ) %>%
  arrange(desc(`Bridge score`), desc(`Collaborative project ties`))

message("  Collaboration hubs (EIS only):")
print(tbl_collab_hubs_eis)

fig6 <- tbl_collab_hubs_eis %>%
  mutate(department = fct_reorder(department, `Bridge score`)) %>%
  ggplot(aes(x = `Bridge score`, y = department, fill = `Collaborative project ties`)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = `Most frequent partner`),
    hjust = 0, nudge_x = 0.15, size = 3, color = catf_navy
  ) +
  scale_fill_gradientn(colors = c(catf_light_blue, catf_dark_blue, catf_navy)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.45))) +
  labs(
    title    = "Department Collaboration Hubs",
    subtitle = paste0(str_to_sentence(fig6_scope_label), ". Bar length shows bridge score; labels show most frequent partner"),
    x        = "Bridge score",
    y        = NULL,
    fill     = "Collaborative\nproject ties",
    caption  = str_wrap(paste0(
      "Note: Bridge score = unique partner departments × log(1 + total shared project ties). ",
      "Restricted to decarbonization ", fig6_scope_label, ". ",
      "Bar length shows bridge score and label shows most frequent partner."
    ), width = 130)
  ) +
  theme_catf()

ggsave(file.path(out_dir, "fig_department_collaboration_hubs.png"),
       fig6, width = 10, height = 6, dpi = 300)
message("  Saved: fig_department_collaboration_hubs.png")

# ---------------------------------------------------------------------------
# Fig 7 — Sankey: cross-department project flows (fig_department_sankey_filtered.png)
# ---------------------------------------------------------------------------
message("\n--- Fig 7: Sankey ---")

# All-process multi-department projects (metadata flag only, all process types)
dept_projects_sankey <- clean_energy %>%
  filter(project_multi_department == TRUE) %>%
  mutate(
    department_list = map(lead_agency, ~ {
      agencies <- parse_jsonish_vector(.x)
      depts    <- unique(map_chr(agencies, map_agency_to_department))
      sort(depts[depts != ""])
    })
  ) %>%
  filter(lengths(department_list) >= 2)

dept_pairs_sankey <- dept_projects_sankey %>%
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

dept_pairs_sankey <- bind_rows(
  dept_pairs_sankey,
  dept_pairs_sankey %>% rename(department_1 = department_2, department_2 = department_1)
)

pair_counts_sankey <- dept_pairs_sankey %>%
  count(department_1, department_2, name = "shared_projects", sort = TRUE)

DEPT_TOP_N <- 6

dept_totals_sankey <- pair_counts_sankey %>%
  group_by(department = department_1) %>%
  summarise(total_ties = sum(shared_projects), .groups = "drop") %>%
  arrange(desc(total_ties))

top_depts_sankey      <- dept_totals_sankey %>% slice_head(n = DEPT_TOP_N) %>% pull(department)
excluded_depts_sankey <- dept_totals_sankey %>% filter(!department %in% top_depts_sankey) %>% pull(department)
DEPT_THRESHOLD_SANKEY <- dept_totals_sankey %>% filter(department %in% top_depts_sankey) %>% pull(total_ties) %>% min()

excluded_label_sankey <- if (length(excluded_depts_sankey) > 0) {
  paste0(
    "Note: Showing top ", length(top_depts_sankey), " of ", nrow(dept_totals_sankey),
    " departments by collaborative activity (minimum ", DEPT_THRESHOLD_SANKEY,
    " shared project ties). Excluded (", length(excluded_depts_sankey), "): ",
    paste(excluded_depts_sankey, collapse = "; "), "."
  )
} else {
  "All departments shown."
}

# Order: descending total_ties so most collaborative department is at top of Sankey
dept_order_sankey <- dept_totals_sankey %>%
  filter(department %in% top_depts_sankey) %>%
  arrange(desc(total_ties)) %>%
  pull(department)

pair_counts_filtered_sankey <- pair_counts_sankey %>%
  filter(department_1 %in% top_depts_sankey, department_2 %in% top_depts_sankey) %>%
  mutate(
    department_1 = factor(department_1, levels = dept_order_sankey),
    department_2 = factor(department_2, levels = dept_order_sankey)
  )

fig7 <- ggplot(
  pair_counts_filtered_sankey,
  aes(axis1 = department_1, axis2 = department_2, y = shared_projects)
) +
  ggalluvial::geom_alluvium(aes(fill = department_1), width = 1/10, alpha = 0.8) +
  ggalluvial::geom_stratum(width = 1/8, fill = "gray96", color = "gray60") +
  geom_text(
    stat      = ggalluvial::StatStratum,
    aes(label = str_wrap(after_stat(stratum), width = 18)),
    hjust     = 0.5, size = 3.5, lineheight = 0.9, color = "gray20"
  ) +
  scale_x_discrete(limits = c("axis1", "axis2"), labels = NULL, expand = c(0.03, 0.03)) +
  scale_fill_manual(
    values = rep(catf_palette, length.out = n_distinct(pair_counts_filtered_sankey$department_1))
  ) +
  labs(
    title   = "Cross-Department Project Flows",
    caption = str_wrap(paste0(
      "Top ", length(top_depts_sankey), " departments by collaborative activity; ",
      "flow width reflects shared projects. ", excluded_label_sankey
    ), width = 160),
    y = NULL,
    x = NULL
  ) +
  theme_catf() +
  theme(
    legend.position  = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    axis.title       = element_blank(),
    axis.text        = element_blank(),
    axis.ticks       = element_blank(),
    axis.line        = element_blank(),
    plot.caption     = element_text(hjust = 0, size = 8, color = "gray40", margin = margin(t = 10)),
    plot.margin      = margin(10, 10, 15, 10)
  )

ggsave(file.path(out_dir, "fig_department_sankey_filtered.png"),
       fig7, width = 13, height = 7, dpi = 300)
message("  Saved: fig_department_sankey_filtered.png")

# ---------------------------------------------------------------------------
# Pages data: shared pipeline for Figs 8, 9, 10
# ---------------------------------------------------------------------------
message("\n--- Loading pages data (Figs 8-10) ---")

fra_date <- as.Date("2023-06-03")

projects_d5 <- projects %>%
  filter(project_energy_type == "Clean", process_type %in% c("EA", "EIS"))

# EA + EIS timelines with LLM-adjudicated dates (ea_df / eis_df already loaded)
timeline_d5 <- bind_rows(
  ea_df  %>% mutate(tl_source = "ea_llm"),
  eis_df %>% mutate(tl_source = "eis_llm")
) %>%
  mutate(
    timeline_initiation_date = as.Date(llm_initiation_date),
    timeline_decision_date   = as.Date(llm_decision_date)
  ) %>%
  select(project_id, tl_source, timeline_initiation_date, timeline_decision_date) %>%
  distinct(project_id, .keep_all = TRUE)

# Documents: final only (FEIS for EIS projects, EA for EA projects)
documents_path <- here("phase1", "data", "analysis", "documents_combined.parquet")
if (!file.exists(documents_path)) stop("Missing documents_combined.parquet")
documents_all <- read_parquet(documents_path)

final_docs <- documents_all %>%
  filter(
    (dataset_source == "EIS" & document_type_clean == "FEIS") |
    (dataset_source == "EA"  & document_type_clean == "EA")
  ) %>%
  mutate(total_pages = as.numeric(total_pages))

final_docs_dedup <- final_docs %>%
  mutate(is_main = (main_document == "YES")) %>%
  group_by(project_id) %>%
  arrange(desc(is_main), desc(total_pages), .by_group = TRUE) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  select(-is_main)

pages_data <- projects_d5 %>%
  inner_join(timeline_d5, by = "project_id") %>%
  inner_join(
    final_docs_dedup %>%
      select(project_id, total_pages, document_type_clean, document_id, main_document),
    by = "project_id"
  )

# Regulatory page counts
page_counts_path <- here("phase1", "data", "analysis", "projects_page_counts.parquet")
if (file.exists(page_counts_path)) {
  page_counts_raw <- read_parquet(page_counts_path)
  if (!"regulatory_pages_method" %in% colnames(page_counts_raw)) {
    page_counts_raw <- page_counts_raw %>% mutate(regulatory_pages_method = "ocr")
  }
  page_counts <- page_counts_raw %>%
    select(project_id, regulatory_pages, body_pages, low_content_pages,
           appendix_start_page, regulatory_pages_method)
  pages_data <- pages_data %>% left_join(page_counts, by = "project_id")
  message("  Joined regulatory_pages for ", sum(!is.na(pages_data$regulatory_pages)),
          " projects")
} else {
  warning("projects_page_counts.parquet not found — regulatory_pages will be NA")
  pages_data <- pages_data %>%
    mutate(regulatory_pages = NA_real_, body_pages = NA_real_,
           low_content_pages = NA_real_, appendix_start_page = NA_real_,
           regulatory_pages_method = NA_character_)
}

pages_data <- pages_data %>%
  mutate(
    reg_pages_source = case_when(
      regulatory_pages_method == "no_appendix_file" ~ "no_appendix_file",
      regulatory_pages_method == "ocr"              ~ "ocr",
      !is.na(total_pages)                           ~ "raw_fallback",
      TRUE                                           ~ NA_character_
    ),
    regulatory_pages = coalesce(regulatory_pages, as.numeric(total_pages))
  )

pages_analysis <- pages_data %>%
  filter(!is.na(timeline_initiation_date), !is.na(timeline_decision_date)) %>%
  mutate(
    fra_period      = factor(
      if_else(timeline_decision_date >= fra_date, "Post-FRA", "Pre-FRA"),
      levels = c("Pre-FRA", "Post-FRA")
    ),
    decision_year   = year(timeline_decision_date),
    decision_month  = floor_date(timeline_decision_date, "month"),
    duration_days   = as.numeric(timeline_decision_date - timeline_initiation_date),
    duration_months = duration_days / 30.44
  )

message("  pages_analysis: ", nrow(pages_analysis), " projects")
message("    Pre-FRA:  ", sum(pages_analysis$fra_period == "Pre-FRA"))
message("    Post-FRA: ", sum(pages_analysis$fra_period == "Post-FRA"))

# ---------------------------------------------------------------------------
# Fig 8 — Pages over time, rolling avg broken at FRA (05_pages_over_time_break.png)
# ---------------------------------------------------------------------------
message("\n--- Fig 8: Pages over time ---")

pages_for_time <- pages_analysis %>%
  filter(decision_year >= 2010, decision_year <= 2025, !is.na(regulatory_pages))

monthly_pages_break <- pages_for_time %>%
  mutate(rolling_segment = if_else(timeline_decision_date < fra_date, "Pre-FRA", "Post-FRA")) %>%
  group_by(process_type, rolling_segment, decision_month) %>%
  summarise(
    mean_pages = mean(regulatory_pages, na.rm = TRUE),
    n_projects = n(),
    .groups    = "drop"
  ) %>%
  arrange(process_type, rolling_segment, decision_month) %>%
  group_by(process_type, rolling_segment) %>%
  mutate(rolling_mean_3m = zoo::rollmean(mean_pages, k = 3, fill = NA, align = "right")) %>%
  ungroup()

fig8 <- ggplot() +
  geom_point(
    data = pages_for_time,
    aes(x = timeline_decision_date, y = regulatory_pages, color = fra_period),
    alpha = 0.32, size = 1.2
  ) +
  geom_line(
    data = monthly_pages_break,
    aes(x = decision_month, y = rolling_mean_3m, group = rolling_segment),
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
  scale_color_manual(values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)) +
  labs(
    title   = "NEPA Document Length Over Time (3-Month Rolling Average Broken at FRA)",
    x       = "Decision Date",
    y       = "Regulatory Pages (body word count ÷ 500)",
    color   = NULL,
    caption = str_wrap(paste0(
      "Note: Analysis is limited to 20,725 decarbonization NEPA reviews. ",
      "Points represent an individual project; the solid line represents a 3-month rolling average ",
      "computed separately for Pre- and Post-FRA."
    ), width = 200)
  ) +
  theme_catf() +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "05_pages_over_time_break.png"), fig8, width = 12, height = 8, dpi = 300)
message("  Saved: 05_pages_over_time_break.png")

# ---------------------------------------------------------------------------
# Fig 9 — Pre/post FRA bar chart (05_pages_pre_post_fra.png)
# ---------------------------------------------------------------------------
message("\n--- Fig 9: Pre/post FRA comparison ---")

fra_summary <- pages_analysis %>%
  filter(!is.na(regulatory_pages)) %>%
  group_by(process_type, fra_period) %>%
  summarise(
    mean_pages   = mean(regulatory_pages, na.rm = TRUE),
    median_pages = median(regulatory_pages, na.rm = TRUE),
    n            = n(),
    .groups      = "drop"
  ) %>%
  mutate(
    bar_label    = sprintf("average\n%.0f pages\n(n = %s)", mean_pages, comma(n)),
    median_label = sprintf("median: %.0f", median_pages)
  )

ea_post    <- fra_summary %>% filter(process_type == "EA",  fra_period == "Post-FRA")
other_bars <- fra_summary %>% filter(!(process_type == "EA" & fra_period == "Post-FRA"))

fig9 <- ggplot(fra_summary, aes(x = fra_period, y = mean_pages, fill = fra_period)) +
  geom_col(alpha = 0.85, width = 0.6) +
  geom_text(data = other_bars, aes(label = bar_label),
            vjust = -0.2, size = 3.3, color = "gray20") +
  geom_text(data = ea_post,    aes(y = median_pages, label = bar_label),
            vjust = 1.8, size = 3.3, color = "white") +
  geom_point(aes(y = median_pages), shape = 18, size = 4, color = catf_navy) +
  geom_text(data = other_bars, aes(y = median_pages, label = median_label),
            vjust = 1.8, size = 2.8, color = "white") +
  geom_text(data = ea_post,    aes(y = mean_pages, label = median_label),
            vjust = -0.2, size = 2.8, color = "black") +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.3))) +
  labs(
    title   = "Average and Median Regulatory Page Counts Mostly Decline After the FRA Was Enacted in 2023",
    x       = NULL,
    y       = "Regulatory Pages (body word count ÷ 500)",
    fill    = NULL,
    caption = str_wrap(paste0(
      "Note: Analysis is limited to 20,725 decarbonization NEPA reviews. ",
      "Bar height represents the mean regulatory pages, while diamond represents the median. ",
      "Project reviews are classified by their decision date."
    ), width = 160)
  ) +
  theme_catf() +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "05_pages_pre_post_fra.png"), fig9, width = 10, height = 6, dpi = 300)
message("  Saved: 05_pages_pre_post_fra.png")

# ---------------------------------------------------------------------------
# Fig 10 — FRA page limit compliance (05_fra_compliance.png)
# ---------------------------------------------------------------------------
message("\n--- Fig 10: FRA compliance ---")

post_fra <- pages_analysis %>%
  filter(fra_period == "Post-FRA", !is.na(regulatory_pages)) %>%
  mutate(
    compliance = case_when(
      process_type == "EA"  & regulatory_pages <= 75  ~ "Compliant",
      process_type == "EA"  & regulatory_pages > 75   ~ "Exceeds limit",
      process_type == "EIS" & regulatory_pages <= 150 ~ "Compliant",
      process_type == "EIS" & regulatory_pages > 150  & regulatory_pages <= 300 ~ "Exceeds standard limit",
      process_type == "EIS" & regulatory_pages > 300  ~ "Exceeds limit"
    )
  )

ea_levels  <- c("Compliant", "Exceeds limit")
eis_levels <- c("Compliant", "Exceeds standard limit", "Exceeds limit")
all_levels <- c("Compliant", "Exceeds standard limit", "Exceeds limit")
post_fra <- post_fra %>% mutate(compliance = factor(compliance, levels = all_levels))

compliance_colors <- c(
  "Compliant"              = catf_teal,
  "Exceeds standard limit" = "#E8A317",
  "Exceeds limit"          = catf_magenta
)

compliance_summary <- post_fra %>%
  count(process_type, compliance, .drop = FALSE) %>%
  group_by(process_type) %>%
  filter(
    (process_type == "EA"  & compliance %in% ea_levels) |
    (process_type == "EIS" & compliance %in% eis_levels)
  ) %>%
  mutate(
    total = sum(n),
    pct   = n / total * 100,
    label = sprintf("%s\n(%.0f%%)", comma(n), pct)
  ) %>%
  ungroup()

eis_rows            <- compliance_summary %>% filter(process_type == "EIS")
n_within_300        <- sum(eis_rows$n[as.character(eis_rows$compliance) %in%
                                        c("Compliant", "Exceeds standard limit")])
n_exceeds_limit_eis <- sum(eis_rows$n[as.character(eis_rows$compliance) == "Exceeds limit"])
n_eis_total         <- sum(eis_rows$n)
pct_within_300      <- round(n_within_300 / n_eis_total * 100)

x_tick <- 2.33; x_vert <- 2.50; x_label <- 2.55
y_bot  <- n_exceeds_limit_eis
y_top  <- n_eis_total
y_mid  <- (y_bot + y_top) / 2
tick_h <- n_eis_total * 0.018

bracket_label <- paste0(
  "Total ", pct_within_300, "%\ncompliant by\nextraordinary\ncomplexity\nthreshold\n(300 pages)"
)

fig10 <- ggplot(compliance_summary, aes(x = process_type, y = n, fill = compliance)) +
  geom_col(width = 0.6, alpha = 0.9) +
  geom_text(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 3.2, color = "white", fontface = "bold"
  ) +
  annotate("segment", x = x_tick, xend = x_vert,
           y = y_top - tick_h, yend = y_top - tick_h,
           color = "black", linewidth = 0.55) +
  annotate("segment", x = x_vert, xend = x_vert,
           y = y_bot + tick_h, yend = y_top - tick_h,
           color = "black", linewidth = 0.55) +
  annotate("segment", x = x_tick, xend = x_vert,
           y = y_bot + tick_h, yend = y_bot + tick_h,
           color = "black", linewidth = 0.55) +
  annotate("text", x = x_label, y = y_mid, label = bracket_label,
           hjust = 0, vjust = 0.5, size = 3.2, color = "black", lineheight = 0.88) +
  scale_fill_manual(values = compliance_colors) +
  scale_x_discrete(expand = expansion(add = c(0.5, 0.8))) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  coord_cartesian(clip = "off") +
  labs(
    title   = "Despite Declining Page Counts, Many Post-FRA Reviews Are Not in Compliance",
    x       = NULL,
    y       = "Number of Projects",
    fill    = NULL,
    caption = str_wrap(paste0(
      "Note: EA limit: 75 pages | EIS limit: 150 pages (300 for extraordinarily complex). ",
      "n = ", comma(sum(compliance_summary$n[compliance_summary$process_type == "EA"])),
      " EA, ",
      comma(sum(compliance_summary$n[compliance_summary$process_type == "EIS"])),
      " EIS post-FRA projects."
    ), width = 160)
  ) +
  theme_catf() +
  theme(legend.position = "bottom", plot.margin = margin(5, 40, 5, 5))

ggsave(file.path(out_dir, "05_fra_compliance.png"), fig10, width = 10, height = 7, dpi = 300)
message("  Saved: 05_fra_compliance.png")

# ---------------------------------------------------------------------------
# Appendix — Energy type breakdown (00_energy_type_breakdown.png)
# ---------------------------------------------------------------------------
message("\n--- Appendix: Energy type breakdown ---")

energy_type_summary <- projects %>%
  count(project_energy_type, name = "projects") %>%
  mutate(
    share = projects / sum(projects),
    project_energy_type = factor(
      if_else(project_energy_type == "Clean", "Decarbonized", project_energy_type),
      levels = c("Decarbonized", "Fossil", "Other")
    )
  )

fig_appendix <- energy_type_summary %>%
  ggplot(aes(x = reorder(project_energy_type, -projects), y = projects,
             fill = project_energy_type)) +
  geom_col() +
  geom_text(
    aes(label = paste0(scales::comma(projects), "\n(",
                       scales::percent(share, accuracy = 0.1), ")")),
    vjust = -0.2, size = 3.5
  ) +
  labs(
    x     = NULL,
    y     = "Number of Projects",
    title = "NEPA Reviews by Energy Type"
  ) +
  scale_y_continuous(labels = scales::comma, expand = expansion(mult = c(0, 0.15))) +
  scale_fill_manual(values = c(
    "Decarbonized" = catf_teal,
    "Fossil"       = catf_navy,
    "Other"        = catf_light_blue
  )) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "00_energy_type_breakdown.png"),
       fig_appendix, width = 8, height = 5, dpi = 300)
message("  Saved: 00_energy_type_breakdown.png")

message("\n=== Done. All factsheet figures written to: ", out_dir, " ===")
