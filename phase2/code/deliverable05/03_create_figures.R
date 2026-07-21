# D5 / 03 — CE Spikes After Major Legislation: analysis + figures + tables
#
# Reproduces and extends the Phase 1 D3 by-year spike figures with this deliverable's own code,
# then adds the Phase-2-specific law-citation attribution and CE-category characterization.
#
# Base population: CE projects placeable by a determination date — year_date =
#   coalesce(decision_date, initiation_date) (initiation is a safe same-year proxy for CEs;
#   median CE duration is 20 days). date_basis records which was used. This is broader than the
#   complete-timeline base used for duration in D4.
#
# Inputs:
#   phase2/data/analysis/timeline/timeline_project_dates.parquet   (decision/initiation dates)
#   phase1/data/analysis/projects_combined.parquet                 (energy, department, agency, type)
#   phase2/data/analysis/deliverable05/law_citations.parquet       (script 01)
#   phase2/data/analysis/deliverable05/ce_categories.parquet       (script 02)
#
# Outputs: phase2/output/deliverable05/{figures,diagnostics}/
#
# Usage: Rscript phase2/code/deliverable05/03_create_figures.R

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(stringr)
  library(lubridate); library(arrow); library(ggplot2); library(scales)
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PHASE2 <- here::here("phase2")
OUT    <- file.path(PHASE2, "output", "deliverable05")
FIGS   <- file.path(OUT, "figures")
DIAG   <- file.path(OUT, "diagnostics")
dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)
dir.create(DIAG, recursive = TRUE, showWarnings = FALSE)

TIMELINE <- file.path(PHASE2, "data", "analysis", "timeline", "timeline_project_dates.parquet")
PROJECTS <- here::here("phase1", "data", "analysis", "projects_combined.parquet")
CITES    <- file.path(PHASE2, "data", "analysis", "deliverable05", "law_citations.parquet")
CATS     <- file.path(PHASE2, "data", "analysis", "deliverable05", "ce_categories.parquet")

# ---------------------------------------------------------------------------
# CATF theme (matches Phase 1 D3 / D4)
# ---------------------------------------------------------------------------
catf_dark_blue <- "#0047BB"; catf_blue <- "#00B5E2"; catf_magenta <- "#C22A90"
catf_purple <- "#75246C"; catf_lime <- "#93D500"; catf_teal <- "#00AE8D"
catf_light_blue <- "#8AB7E9"; catf_navy <- "#012169"

PROCESS_LEVELS <- c("CE", "EA", "EIS")
PROCESS_COLORS <- c("CE" = catf_lime, "EA" = catf_dark_blue, "EIS" = catf_navy)
ENERGY_LEVELS  <- c("Decarb", "Fossil", "Other")
ENERGY_COLORS  <- c("Decarb" = catf_lime, "Fossil" = catf_dark_blue, "Other" = catf_navy)

theme_catf <- function(base_size = 11, base_family = "Helvetica") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      plot.title    = element_text(face = "bold", size = rel(1.2), color = catf_navy, margin = margin(b = 10)),
      plot.subtitle = element_text(size = rel(0.9), color = catf_dark_blue, margin = margin(b = 10)),
      plot.caption  = element_text(size = rel(0.8), color = "gray50", hjust = 0),
      axis.title    = element_text(size = rel(0.9), color = catf_navy),
      axis.text     = element_text(size = rel(0.85), color = "gray30"),
      axis.line     = element_line(color = "gray70", linewidth = 0.3),
      legend.title  = element_text(face = "bold", size = rel(0.9), color = catf_navy),
      legend.text   = element_text(size = rel(0.85), color = "gray30"),
      legend.position = "bottom", legend.key.size = unit(0.8, "lines"),
      panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      strip.text = element_text(face = "bold", size = rel(0.9), color = catf_navy),
      strip.background = element_rect(fill = "gray95", color = NA),
      plot.margin = margin(15, 15, 10, 10)
    )
}
theme_set(theme_catf())

# Legislative markers
MARKERS <- tibble(
  year  = c(2009, 2021, 2022),
  label = c("ARRA\nFeb 09", "BIL\nNov 21", "IRA\nAug 22"),
  hjust = c(-0.08, 1.08, -0.08)
)
YEAR_MIN <- 2000; YEAR_MAX <- 2025

# Windows: year-based for figures; category baselines defined separately
LAWS <- tribble(
  ~law,   ~win_start,   ~win_end,     ~base_start,  ~base_end,
  "ARRA", "2009-03-01", "2011-12-31", NA,           NA,
  "BIL",  "2021-12-01", "2023-12-31", "2018-12-01", "2021-11-30",
  "IRA",  "2022-09-01", "2024-12-31", "2019-09-01", "2022-08-31"
) |> mutate(across(c(win_start, win_end, base_start, base_end), as.Date))

# ---------------------------------------------------------------------------
# Load + derive
# ---------------------------------------------------------------------------
message("Loading inputs...")
dates <- read_parquet(TIMELINE)
meta  <- read_parquet(PROJECTS, col_select = c("project_id", "project_energy_type",
                                               "project_department", "lead_agency_harmonized",
                                               "project_type"))

d <- dates |>
  left_join(meta, by = "project_id") |>
  mutate(
    decision_date   = as.Date(decision_date),
    initiation_date = as.Date(initiation_date),
    year_date  = coalesce(decision_date, initiation_date),
    date_basis = case_when(!is.na(decision_date) ~ "decision",
                           !is.na(initiation_date) ~ "initiation_proxy",
                           TRUE ~ NA_character_),
    year  = year(year_date),
    process_group = factor(process_type, levels = PROCESS_LEVELS),
    energy_type   = factor(recode(coalesce(project_energy_type, "Other"), "Clean" = "Decarb"),
                           levels = ENERGY_LEVELS),
    department = coalesce(project_department, "Other / Unclassified"),
    agency2 = case_when(
      str_detect(coalesce(lead_agency_harmonized, ""), "Department of Energy") ~ "DOE",
      str_detect(coalesce(lead_agency_harmonized, ""), "Bureau of Land Management") ~ "BLM",
      TRUE ~ "Other")
  )

ce <- d |> filter(process_group == "CE")
n_ce_placeable <- sum(!is.na(ce$year))
message("  CE projects: ", nrow(ce), " | placeable by a date: ", n_ce_placeable,
        " (", round(100 * n_ce_placeable / nrow(ce), 1), "%)")

CE_BASE_CAPTION <- sprintf(
  "Base: CE projects placeable by a determination date (n = %s, %.1f%%); year = decision date, or initiation date as a same-year proxy where decision is absent.",
  comma(n_ce_placeable), 100 * n_ce_placeable / nrow(ce))

# Helper: dashed legislative markers
add_markers <- function(p) {
  p + geom_vline(xintercept = MARKERS$year, linetype = "dashed",
                 color = catf_teal, linewidth = 0.75, alpha = 0.9)
}
# Helper: marker text labels in the top/only panel (data carries the panel facet value if needed)
marker_text <- function(panel_df = NULL) {
  md <- MARKERS
  if (!is.null(panel_df)) md <- bind_cols(md, panel_df[rep(1, nrow(MARKERS)), , drop = FALSE])
  geom_text(data = md, aes(x = year, y = Inf, label = label, hjust = hjust),
            vjust = 1.3, size = 2.3, color = catf_teal, lineheight = 0.85, inherit.aes = FALSE)
}

save_fig <- function(p, name, w = 11, h = 7) {
  ggsave(file.path(FIGS, name), p, width = w, height = h, dpi = 300)
  # .rds sidecar (same basename) so downstream scripts can readRDS + retitle.
  saveRDS(p, file.path(FIGS, sub("\\.png$", ".rds", name)))
  message("Wrote ", name)
}

# ===========================================================================
# ANALYSIS A — Temporal counts (the spike)
# ===========================================================================

# A1 [all] — all CE by year, single panel
a_all <- ce |> filter(!is.na(year), between(year, YEAR_MIN, YEAR_MAX)) |>
  count(year, name = "n")
p <- add_markers(ggplot(a_all, aes(year, n))) +
  geom_col(fill = catf_lime, alpha = 0.9) +
  geom_text(aes(label = comma(n)), vjust = -0.3, size = 2.6, color = "gray30") +
  marker_text() +
  scale_x_continuous(breaks = seq(YEAR_MIN, YEAR_MAX, 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18)), labels = comma) +
  labs(title = "Categorical Exclusions by Year", subtitle = "All CE reviews. Dashed lines mark major legislation.",
       x = "Year", y = "Number of CEs", caption = CE_BASE_CAPTION)
save_fig(p, "fig_d5_ce_counts_by_year_all.png", h = 6)

# A2 [by energy] — CE by year stacked by energy type
a_en <- ce |> filter(!is.na(year), between(year, YEAR_MIN, YEAR_MAX)) |>
  count(year, energy_type, name = "n")
a_en_tot <- a_en |> group_by(year) |> summarise(n = sum(n), .groups = "drop")
p <- add_markers(ggplot(a_en, aes(year, n, fill = energy_type))) +
  geom_col(alpha = 0.9) +
  geom_text(data = a_en_tot, aes(year, n, label = comma(n)), vjust = -0.3, size = 2.4,
            color = "gray30", inherit.aes = FALSE) +
  marker_text() +
  scale_fill_manual(values = ENERGY_COLORS, drop = FALSE) +
  scale_x_continuous(breaks = seq(YEAR_MIN, YEAR_MAX, 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18)), labels = comma) +
  labs(title = "Categorical Exclusions by Year and Energy Type",
       subtitle = "All CE reviews, stacked by energy type. Total labeled above each bar.",
       x = "Year", y = "Number of CEs", fill = NULL, caption = CE_BASE_CAPTION)
save_fig(p, "fig_d5_ce_counts_by_year_byenergy.png", h = 6)

# A3 [by process] — CE/EA/EIS by year, faceted (recreation of Phase 1 03_projects_by_year)
a_proc <- d |> filter(!is.na(process_group), !is.na(year), between(year, YEAR_MIN, YEAR_MAX)) |>
  count(process_group, year, name = "n")
mk_proc <- MARKERS |> mutate(process_group = factor("CE", levels = PROCESS_LEVELS))
p <- add_markers(ggplot(a_proc, aes(year, n))) +
  geom_col(aes(fill = process_group), alpha = 0.9) +
  geom_text(aes(label = comma(n)), vjust = -0.3, size = 2.3, color = "gray30") +
  geom_text(data = mk_proc, aes(x = year, y = Inf, label = label, hjust = hjust),
            vjust = 1.3, size = 2.2, color = catf_teal, lineheight = 0.85, inherit.aes = FALSE) +
  facet_wrap(~process_group, scales = "free_y", ncol = 1, drop = FALSE) +
  scale_fill_manual(values = PROCESS_COLORS, guide = "none") +
  scale_x_continuous(breaks = seq(YEAR_MIN, YEAR_MAX, 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22)), labels = comma) +
  labs(title = "NEPA Reviews by Year and Review Type",
       subtitle = "Faceted by process. The CE spike is absent in EA/EIS. Dashed lines mark major legislation.",
       x = "Year", y = "Number of Reviews", caption = CE_BASE_CAPTION)
save_fig(p, "fig_d5_counts_by_year_byprocess.png", w = 11, h = 9)

# A4 [by process x energy] — faceted by process, stacked by energy
a_pe <- d |> filter(!is.na(process_group), !is.na(year), between(year, YEAR_MIN, YEAR_MAX)) |>
  count(process_group, energy_type, year, name = "n")
p <- add_markers(ggplot(a_pe, aes(year, n, fill = energy_type))) +
  geom_col(alpha = 0.9) +
  facet_wrap(~process_group, scales = "free_y", ncol = 1, drop = FALSE) +
  scale_fill_manual(values = ENERGY_COLORS, drop = FALSE) +
  scale_x_continuous(breaks = seq(YEAR_MIN, YEAR_MAX, 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)), labels = comma) +
  labs(title = "NEPA Reviews by Year, Review Type, and Energy Type",
       subtitle = "Faceted by process, stacked by energy type. Dashed lines mark major legislation.",
       x = "Year", y = "Number of Reviews", fill = NULL, caption = CE_BASE_CAPTION)
save_fig(p, "fig_d5_counts_by_year_byprocess_byenergy.png", w = 11, h = 9)

# A5 [agency: department rollup] — CE by year by department (top depts)
top_depts <- ce |> count(department, sort = TRUE) |> slice_head(n = 4) |> pull(department)
a_dept <- ce |> filter(!is.na(year), between(year, YEAR_MIN, YEAR_MAX)) |>
  mutate(dept = if_else(department %in% top_depts, department, "Other")) |>
  count(dept, year, name = "n")
dept_levels <- c(top_depts, "Other")
a_dept <- a_dept |> mutate(dept = factor(dept, levels = dept_levels))
p <- add_markers(ggplot(a_dept, aes(year, n, fill = dept))) +
  geom_col(alpha = 0.9) +
  marker_text() +
  scale_fill_manual(values = c(catf_lime, catf_dark_blue, catf_purple, catf_magenta, catf_light_blue)[seq_along(dept_levels)]) +
  scale_x_continuous(breaks = seq(YEAR_MIN, YEAR_MAX, 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12)), labels = comma) +
  labs(title = "Categorical Exclusions by Year and Department",
       subtitle = "All CE reviews, stacked by lead department (project_department rollup).",
       x = "Year", y = "Number of CEs", fill = NULL, caption = CE_BASE_CAPTION)
save_fig(p, "fig_d5_ce_counts_by_year_bydept.png", h = 6)

# A6 [agency: DOE & BLM only] — the headline finding figure
a_db <- ce |> filter(agency2 %in% c("DOE", "BLM"), !is.na(year), between(year, YEAR_MIN, YEAR_MAX)) |>
  mutate(agency2 = factor(agency2, levels = c("DOE", "BLM"))) |>
  count(agency2, year, name = "n")
mk_db <- MARKERS |> mutate(agency2 = factor("DOE", levels = c("DOE", "BLM")))
p <- add_markers(ggplot(a_db, aes(year, n))) +
  geom_col(aes(fill = agency2), alpha = 0.9) +
  geom_text(aes(label = comma(n)), vjust = -0.3, size = 2.3, color = "gray30") +
  geom_text(data = mk_db, aes(x = year, y = Inf, label = label, hjust = hjust),
            vjust = 1.3, size = 2.2, color = catf_teal, lineheight = 0.85, inherit.aes = FALSE) +
  facet_wrap(~agency2, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = c("DOE" = catf_dark_blue, "BLM" = catf_purple), guide = "none") +
  scale_x_continuous(breaks = seq(YEAR_MIN, YEAR_MAX, 2)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22)), labels = comma) +
  labs(title = "Categorical Exclusions by Year: DOE vs BLM",
       subtitle = "The post-ARRA CE spike is a DOE phenomenon; BLM (the other major CE user) is flat.",
       x = "Year", y = "Number of CEs", caption = CE_BASE_CAPTION)
save_fig(p, "fig_d5_ce_counts_by_year_doe_blm.png", w = 11, h = 8)

# ===========================================================================
# ANALYSIS B — Citation attribution
# ===========================================================================
cites <- read_parquet(CITES)
LAW_LEVELS  <- c("ARRA", "BIL", "IRA")
LAW_COLORS  <- c("ARRA" = catf_dark_blue, "BIL" = catf_teal, "IRA" = catf_magenta)

# project-level citation flags joined to year/process/energy
cite_proj <- cites |>
  filter(law_name %in% LAW_LEVELS) |>
  distinct(project_id, law_name) |>
  mutate(cited = TRUE)

base_pe <- d |> filter(!is.na(year), between(year, YEAR_MIN, YEAR_MAX)) |>
  select(project_id, process_group, energy_type, year, agency2)

# B1 [by process] — law-citing review counts by year (line), faceted by process
cite_year <- base_pe |>
  inner_join(cite_proj, by = "project_id", relationship = "many-to-many") |>
  mutate(law_name = factor(law_name, levels = LAW_LEVELS)) |>
  count(process_group, law_name, year, name = "n")
p <- ggplot(cite_year, aes(year, n, color = law_name)) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.4, alpha = 0.8) +
  facet_wrap(~process_group, scales = "free_y", ncol = 1) +
  scale_color_manual(values = LAW_COLORS) +
  scale_x_continuous(breaks = seq(YEAR_MIN, YEAR_MAX, 2), limits = c(2007, YEAR_MAX)) +
  labs(title = "Law-Citing Reviews by Year",
       subtitle = "Count of reviews whose documents explicitly cite each law. A citation can only follow passage.",
       x = "Year", y = "Reviews citing the law", color = NULL,
       caption = "Citations detected in document text (script 01). Acronyms disambiguated by context.")
save_fig(p, "fig_d5_citations_by_year_byprocess.png", w = 11, h = 9)

# B2 — citation rate within spike window vs baseline (CE; bars + N), all + by energy
cite_rate_tbl <- function(df, scope_label) {
  out <- list()
  for (i in seq_len(nrow(LAWS))) {
    L <- LAWS[i, ]
    win <- df |> filter(year_date >= L$win_start, year_date <= L$win_end)
    cited_win <- mean(win$project_id %in% cite_proj$project_id[cite_proj$law_name == L$law])
    rec <- tibble(law = L$law, scope = scope_label, period = "spike window",
                  n = nrow(win), pct_cited = 100 * cited_win)
    out[[length(out) + 1]] <- rec
    if (!is.na(L$base_start)) {
      bse <- df |> filter(year_date >= L$base_start, year_date <= L$base_end)
      cited_b <- mean(bse$project_id %in% cite_proj$project_id[cite_proj$law_name == L$law])
      out[[length(out) + 1]] <- tibble(law = L$law, scope = scope_label, period = "baseline",
                                       n = nrow(bse), pct_cited = 100 * cited_b)
    }
  }
  bind_rows(out)
}
rate_all <- cite_rate_tbl(ce |> filter(!is.na(year_date)), "All CE")
p <- ggplot(rate_all, aes(law, pct_cited, fill = period)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.1f%%\n(n=%s)", pct_cited, comma(n))),
            position = position_dodge(width = 0.7), vjust = -0.2, size = 3, color = "gray20") +
  scale_fill_manual(values = c("spike window" = catf_dark_blue, "baseline" = catf_light_blue)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(title = "CE Law-Citation Rate: Spike Window vs Baseline",
       subtitle = "Share of CEs whose documents cite the law (ARRA has no usable pre-law baseline).",
       x = NULL, y = "% of CEs citing the law", fill = NULL, caption = CE_BASE_CAPTION)
save_fig(p, "fig_d5_citation_rate_window_vs_baseline.png", w = 10, h = 6)

# ===========================================================================
# ANALYSIS C — CE category mix (CE-only)  [marquee Q3]
# ===========================================================================
cats <- read_parquet(CATS)
# category baselines: ARRA uses a stable 2016-2019 window (no usable pre-period); IRA uses its baseline
CAT_WINDOWS <- tribble(
  ~law,   ~win_start,   ~win_end,     ~base_start,  ~base_end,
  "ARRA", "2009-01-01", "2011-12-31", "2016-01-01", "2019-12-31",
  "IRA",  "2022-09-01", "2024-12-31", "2019-01-01", "2022-08-31"
) |> mutate(across(c(win_start, win_end, base_start, base_end), as.Date))

ce_dates <- ce |> select(project_id, year_date, energy_type) |> filter(!is.na(year_date))
cat_doe <- cats |> filter(schedule == "DOE (10 CFR 1021)") |>
  select(project_id, code_norm, code_description) |>
  inner_join(ce_dates, by = "project_id")

category_shift <- function(law_row, scope = "All CE", energy = NULL) {
  cd <- cat_doe
  if (!is.null(energy)) cd <- cd |> filter(energy_type == energy)
  win <- cd |> filter(year_date >= law_row$win_start, year_date <= law_row$win_end)
  bse <- cd |> filter(year_date >= law_row$base_start, year_date <= law_row$base_end)
  n_win <- n_distinct(win$project_id); n_bse <- n_distinct(bse$project_id)
  w <- win |> count(code_norm, code_description, name = "n") |> mutate(pct = 100 * n / n_win, period = "spike window")
  b <- bse |> count(code_norm, code_description, name = "n") |> mutate(pct = 100 * n / n_bse, period = "baseline")
  bind_rows(w, b) |> mutate(law = law_row$law, scope = scope, n_win = n_win, n_bse = n_bse)
}

# ARRA category-shift figure (top codes by spike-window share)
arra_cat <- category_shift(CAT_WINDOWS[CAT_WINDOWS$law == "ARRA", ])
top_codes <- arra_cat |> filter(period == "spike window") |> slice_max(pct, n = 8) |> pull(code_norm)
arra_plot <- arra_cat |> filter(code_norm %in% top_codes) |>
  mutate(lab = ifelse(is.na(code_description) | code_description == code_norm, code_norm,
                      paste0(code_norm, " — ", code_description)),
         lab = factor(lab),
         period = factor(period, levels = c("baseline", "spike window")))
ord <- arra_plot |> filter(period == "spike window") |> arrange(pct) |> pull(lab)
arra_plot <- arra_plot |> mutate(lab = factor(lab, levels = ord))
p <- ggplot(arra_plot, aes(pct, lab, fill = period)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.65, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.1f%% (n=%s)", pct, comma(n))),
            position = position_dodge(width = 0.7), hjust = -0.05, size = 2.8, color = "gray20") +
  scale_fill_manual(values = c("spike window" = catf_dark_blue, "baseline" = catf_light_blue)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(title = "What Types of CEs: DOE Category Mix, ARRA Window vs Baseline",
       subtitle = "Within-window share of DOE categorical-exclusion codes. ARRA window 2009-2011 vs 2016-2019 baseline.",
       x = "% of DOE CEs in window invoking the code", y = NULL, fill = NULL,
       caption = "DOE 10 CFR 1021 categorical-exclusion codes (script 02). B5.1 = ARRA's energy-efficiency stimulus.")
save_fig(p, "fig_d5_ce_category_shift_arra.png", w = 11, h = 6.5)

# ===========================================================================
# ANALYSIS D — Technology / sector mix (spike window vs baseline)
# ===========================================================================
explode_types <- function(df) {
  # project_type is a JSON array string whose elements can contain internal commas
  # (e.g. "Utilities (electricity, gas, telecommunications)"), so split on the
  # quote-comma-quote element boundary, not on every comma.
  df |> mutate(pt = str_remove_all(coalesce(project_type, ""), '^\\[|\\]$')) |>
    separate_rows(pt, sep = '",\\s*"') |>
    mutate(pt = str_remove_all(pt, '"'), pt = str_trim(pt)) |>
    filter(pt != "", !is.na(pt), pt != "NA")
}
tech_shift <- function(law_row) {
  win <- ce |> filter(year_date >= law_row$win_start, year_date <= law_row$win_end) |> explode_types()
  n_win <- n_distinct(win$project_id)
  win |> count(pt, name = "n") |> mutate(pct = 100 * n / n_win, law = law_row$law, n_win = n_win)
}
arra_tech <- tech_shift(LAWS[LAWS$law == "ARRA", ]) |> slice_max(pct, n = 10)
p <- ggplot(arra_tech, aes(pct, reorder(pt, pct))) +
  geom_col(fill = catf_lime, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.0f%% (n=%s)", pct, comma(n))), hjust = -0.05, size = 2.8, color = "gray20") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(title = "Technology Mix of ARRA-Window CEs (2009-2011)",
       subtitle = "Top project-type tags among CEs issued in the ARRA spike window.",
       x = "% of ARRA-window CEs with the tag", y = NULL, caption = CE_BASE_CAPTION)
save_fig(p, "fig_d5_technology_shift_arra.png", w = 11, h = 6)

# ===========================================================================
# DIAGNOSTIC TABLES
# ===========================================================================
# T5 raw series
ce |> filter(!is.na(year), between(year, YEAR_MIN, YEAR_MAX)) |>
  count(year, energy_type, name = "n") |> write_csv(file.path(DIAG, "d5_counts_by_year.csv"))
ce |> filter(!is.na(year), between(year, YEAR_MIN, YEAR_MAX)) |>
  count(year, department, name = "n") |> write_csv(file.path(DIAG, "d5_counts_by_year_department.csv"))

# T6 date coverage by year x process and date_basis
d |> filter(!is.na(process_group)) |>
  group_by(process_group, year) |>
  summarise(n_total = n(),
            n_placeable = sum(!is.na(year_date)),
            n_decision = sum(date_basis == "decision", na.rm = TRUE),
            n_init_proxy = sum(date_basis == "initiation_proxy", na.rm = TRUE),
            .groups = "drop") |>
  filter(!is.na(year), between(year, YEAR_MIN, YEAR_MAX)) |>
  write_csv(file.path(DIAG, "d5_date_coverage_by_year.csv"))

# T1 spike summary (CE monthly mean, window vs baseline, overall + DOE + BLM)
monthly_counts <- function(df) df |> filter(!is.na(year_date)) |>
  mutate(m = floor_date(year_date, "month")) |> count(m, name = "n")
spike_summary <- list()
for (subset_label in c("All CE", "DOE", "BLM")) {
  sub <- switch(subset_label, "All CE" = ce, "DOE" = ce |> filter(agency2 == "DOE"),
                "BLM" = ce |> filter(agency2 == "BLM"))
  mc <- monthly_counts(sub)
  for (i in seq_len(nrow(LAWS))) {
    L <- LAWS[i, ]
    win_mean <- mc |> filter(m >= L$win_start, m <= L$win_end) |> pull(n) |> mean()
    base_mean <- if (!is.na(L$base_start)) mc |> filter(m >= L$base_start, m <= L$base_end) |> pull(n) |> mean() else NA_real_
    spike_summary[[length(spike_summary) + 1]] <- tibble(
      subset = subset_label, law = L$law, window_mean_monthly = win_mean,
      baseline_mean_monthly = base_mean, spike_ratio = win_mean / base_mean)
  }
}
bind_rows(spike_summary) |> write_csv(file.path(DIAG, "d5_spike_summary.csv"))

# T2 citation rates (all CE + by energy + by process)
ct <- bind_rows(
  cite_rate_tbl(ce |> filter(!is.na(year_date)), "All CE"),
  cite_rate_tbl(ce |> filter(!is.na(year_date), energy_type == "Decarb"), "Decarb CE"),
  cite_rate_tbl(ce |> filter(!is.na(year_date), energy_type == "Fossil"), "Fossil CE"),
  cite_rate_tbl(d |> filter(process_group == "EA", !is.na(year_date)), "All EA"),
  cite_rate_tbl(d |> filter(process_group == "EIS", !is.na(year_date)), "All EIS")
)
write_csv(ct, file.path(DIAG, "d5_citation_rates.csv"))

# T3 category shift (ARRA + IRA, all + by energy)
cs <- bind_rows(
  category_shift(CAT_WINDOWS[CAT_WINDOWS$law == "ARRA", ]),
  category_shift(CAT_WINDOWS[CAT_WINDOWS$law == "IRA", ]),
  category_shift(CAT_WINDOWS[CAT_WINDOWS$law == "ARRA", ], "Decarb CE", "Decarb"),
  category_shift(CAT_WINDOWS[CAT_WINDOWS$law == "ARRA", ], "Fossil CE", "Fossil")
)
write_csv(cs, file.path(DIAG, "d5_category_shift.csv"))

# T4 technology shift
bind_rows(lapply(seq_len(nrow(LAWS)), function(i) tech_shift(LAWS[i, ]))) |>
  write_csv(file.path(DIAG, "d5_technology_shift.csv"))

message("\n=== D5 / 03 complete ===")
message("Figures: ", FIGS)
message("Tables:  ", DIAG)
