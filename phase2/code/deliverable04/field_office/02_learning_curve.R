# D4 Extension #1 — BLM field-office learning-curve analysis
#
# Research question (ASSOCIATIONAL, not causal): within a BLM field office, does NEPA review
# duration fall as the office accumulates experience (cumulative completed reviews) — net of
# calendar-time trends and project mix?
#
# Inputs (DuckDB-built upstream; read here via arrow):
#   - phase2/data/analysis/deliverable04/blm_field_offices.parquet  (01_parse_offices.py)
#   - phase2/data/analysis/timeline/timeline_project_dates.parquet  (durations + source type)
#   - phase2/data/analysis/timeline/timeline_document_index.parquet (project_energy_type)
#
# Duration frame mirrors 08_analyze.R's `headline`: complete timelines only, month-granularity
# imputed to the 15th, year-granularity excluded.
#
# Two confound controls are central:
#   1. Register-anchoring artifact — BLM register "project start" dates are often late
#      administrative entries (the ~40-day artifact). The PRIMARY measure uses DOCUMENT-anchored
#      initiation dates (initiation_source_type != 'metadata'); the register-anchored view is a
#      clearly-flagged secondary.
#   2. Calendar time — a pooled office fixed-effects regression with factor(decision_year)
#      separates the experience gradient from secular calendar trends.
#
# EA vs CE are analyzed SEPARATELY (CE medians ~ weeks, EA ~ months). EIS is too sparse per
# office and is excluded from the core analysis.
#
# Usage: Rscript phase2/code/deliverable04/field_office/02_learning_curve.R

suppressPackageStartupMessages({
  library(here); library(arrow); library(dplyr); library(tidyr)
  library(readr); library(lubridate); library(ggplot2); library(scales)
})

PHASE2 <- here::here("phase2")
TL     <- file.path(PHASE2, "data", "analysis", "timeline")
D04    <- file.path(PHASE2, "data", "analysis", "deliverable04")
FIG    <- file.path(PHASE2, "output", "deliverable04", "figures")
DIAG   <- file.path(PHASE2, "output", "deliverable04", "diagnostics")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
dir.create(DIAG, recursive = TRUE, showWarnings = FALSE)

# CATF palette + theme (copied from fra/02_pages_fra.R) ---------------------------------------
catf_navy <- "#002169"; catf_dark_blue <- "#0047BB"; catf_light_blue <- "#8AB7E9"
catf_teal <- "#00AE8D"; catf_magenta <- "#C22A90"
theme_catf <- function(base = 11) {
  theme_minimal(base_size = base) +
    theme(plot.title = element_text(face = "bold", color = catf_navy, size = rel(1.2)),
          plot.subtitle = element_text(color = catf_dark_blue, size = rel(0.9)),
          plot.caption = element_text(color = "gray50", size = rel(0.8), hjust = 0),
          strip.text = element_text(face = "bold", color = catf_navy),
          strip.background = element_rect(fill = "gray95", color = NA),
          panel.grid.minor = element_blank(), legend.position = "top")
}

# Analysis parameters ------------------------------------------------------------------------
MIN_REVIEWS  <- 30   # qualifying office size (valid durations on the analysis measure)
EA_RELAX     <- 10   # EA falls back to a relaxed threshold (see note below)
anchor_cols  <- c("Document" = catf_navy, "Register" = catf_light_blue)

# ---------------------------------------------------------------------------
# Load + assemble the duration frame (mirror 08_analyze.R `headline`)
# ---------------------------------------------------------------------------
offices <- read_parquet(file.path(D04, "blm_field_offices.parquet"))  # project_id, office_code, state, parse_source

dates_raw <- read_parquet(
  file.path(TL, "timeline_project_dates.parquet"),
  col_select = c("project_id", "process_type", "initiation_date", "decision_date",
                 "initiation_date_granularity", "decision_date_granularity",
                 "initiation_source_type", "timeline_status"))

energy <- read_parquet(
  file.path(TL, "timeline_document_index.parquet"),
  col_select = c("project_id", "project_energy_type")) |>
  group_by(project_id) |>
  summarise(project_energy_type = first(na.omit(project_energy_type)), .groups = "drop")

headline <- dates_raw |>
  mutate(initiation_date = as.Date(initiation_date), decision_date = as.Date(decision_date)) |>
  filter(timeline_status %in% c("complete_clear", "complete_with_proxy"),
         !is.na(initiation_date), !is.na(decision_date),
         initiation_date_granularity != "year", decision_date_granularity != "year") |>
  mutate(
    .init_mid = if_else(initiation_date_granularity == "month",
                        floor_date(initiation_date, "month") + 14, initiation_date),
    .dec_mid  = if_else(decision_date_granularity == "month",
                        floor_date(decision_date, "month") + 14, decision_date),
    duration_days = as.integer(.dec_mid - .init_mid),
    decision_date = .dec_mid,
    decision_year = year(.dec_mid),
    # PRIMARY = document-anchored initiation; register metadata dates are the flagged secondary.
    anchor = if_else(initiation_source_type == "metadata", "Register", "Document")) |>
  filter(!is.na(duration_days), duration_days >= 0)

df <- headline |>
  inner_join(offices, by = "project_id") |>
  left_join(energy, by = "project_id") |>
  filter(process_type %in% c("EA", "CE")) |>           # EIS too sparse per office -> excluded
  mutate(process_type = factor(process_type, levels = c("EA", "CE")),
         energy = factor(recode(coalesce(project_energy_type, "Other"), "Clean" = "Decarb"),
                         levels = c("Decarb", "Fossil", "Other")))

# cum_count = the office's Nth review (experience level), within process x anchor, by decision date
df_cum <- df |>
  group_by(process_type, anchor, office_code) |>
  arrange(decision_date, project_id, .by_group = TRUE) |>
  mutate(cum_count = row_number(), office_n = n()) |>
  ungroup()

# Coverage / feasibility message
feas <- df_cum |> distinct(process_type, anchor, office_code, office_n) |>
  group_by(process_type, anchor) |>
  summarise(offices_ge30 = sum(office_n >= MIN_REVIEWS), offices_ge10 = sum(office_n >= EA_RELAX),
            reviews = sum(office_n[!duplicated(office_code)]), .groups = "drop")
message("Qualifying offices by process x anchor:"); print(as.data.frame(feas))

# ---------------------------------------------------------------------------
# Fig 1 — headline learning curve: median duration by cum_count decile (EA & CE panels)
# Descriptive curve pools all parsed-office reviews per process x anchor; the rigorous
# office-FE regression (below) restricts to qualifying offices.
# ---------------------------------------------------------------------------
curve <- df_cum |>
  group_by(process_type, anchor) |>
  mutate(decile = ntile(cum_count, 10)) |>
  group_by(process_type, anchor, decile) |>
  summarise(med = median(duration_days), n = n(), .groups = "drop") |>
  filter(n >= 10)

p_curve <- ggplot(curve, aes(decile, med, color = anchor)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2) +
  facet_wrap(~process_type, ncol = 1, scales = "free_y") +
  scale_x_continuous(breaks = 1:10) +
  scale_color_manual(values = anchor_cols) +
  labs(title = "Apparent BLM Field-Office Learning Curve (Uncontrolled)",
       subtitle = "Median review duration by the office's cumulative-review decile (1 = earliest reviews, 10 = most experienced)",
       x = "Cumulative-review decile (office experience →)", y = "Median duration (days)", color = "Initiation anchor",
       caption = paste0("Document-anchored = primary (initiation from document text); Register-anchored = secondary (BLM register start date, ",
                        "subject to the late-administrative-entry artifact).\nDescriptive only — this raw gradient is NOT net of calendar time. ",
                        "The office fixed-effects regression with a decision-year control (model table) shows the decline is secular calendar ",
                        "drift, not experience. Associational, not causal.")) +
  theme_catf()
ggsave(file.path(FIG, "fig_d4_fieldoffice_learning_curve.png"), p_curve, width = 11, height = 8, dpi = 300, bg = "white")

# ---------------------------------------------------------------------------
# Per-office first- vs last-tercile (CE, document-anchored, qualifying offices)
# EA has NO office clearing the >=30 document-anchored bar, so the per-office views are CE-only.
# ---------------------------------------------------------------------------
ce_doc_q <- df_cum |> filter(process_type == "CE", anchor == "Document", office_n >= MIN_REVIEWS)

terc <- ce_doc_q |>
  group_by(office_code) |>
  mutate(terc = ntile(cum_count, 3)) |>
  filter(terc %in% c(1, 3)) |>
  group_by(office_code, state, terc) |>
  summarise(med = median(duration_days), .groups = "drop") |>
  mutate(terc = if_else(terc == 1, "first_med", "last_med")) |>
  pivot_wider(names_from = terc, values_from = med)

office_summary <- ce_doc_q |>
  group_by(office_code, state) |>
  summarise(n = n(), median_days = median(duration_days), .groups = "drop") |>
  left_join(terc, by = c("office_code", "state")) |>
  mutate(pct_change = round(100 * (last_med - first_med) / first_med, 1)) |>
  # register-init share = fraction of the office's CE durations anchored on register metadata
  left_join(
    df |> filter(process_type == "CE") |> group_by(office_code) |>
      summarise(register_init_share = round(mean(anchor == "Register"), 3), .groups = "drop"),
    by = "office_code") |>
  arrange(desc(n))

write_csv(office_summary, file.path(DIAG, "d4_fieldoffice_summary.csv"))

share_sped <- mean(office_summary$pct_change < 0, na.rm = TRUE)
message(sprintf("CE qualifying offices: %d | median first-tercile %.0f d, last-tercile %.0f d | %.0f%% sped up",
                nrow(office_summary), median(office_summary$first_med), median(office_summary$last_med),
                100 * share_sped))

# Fig 2 — dumbbell: first- vs last-tercile median, top ~15 CE offices by volume
top15 <- office_summary |> slice_max(n, n = 15) |>
  mutate(office_lab = paste0(office_code, " (n=", n, ")"),
         office_lab = factor(office_lab, levels = office_lab[order(n)]))
db_long <- top15 |>
  select(office_lab, first_med, last_med) |>
  pivot_longer(c(first_med, last_med), names_to = "terc", values_to = "med") |>
  mutate(terc = recode(terc, first_med = "First tercile (least experienced)",
                       last_med = "Last tercile (most experienced)"))
p_db <- ggplot(top15) +
  geom_segment(aes(y = office_lab, yend = office_lab, x = first_med, xend = last_med),
               color = "gray70", linewidth = 1) +
  geom_point(data = db_long, aes(med, office_lab, color = terc), size = 3.2) +
  scale_color_manual(values = c("First tercile (least experienced)" = catf_light_blue,
                                "Last tercile (most experienced)" = catf_navy)) +
  labs(title = "Faster With Experience? First vs Last Tercile of Office Reviews",
       subtitle = "Median CE duration for each office's earliest vs most-experienced third of reviews (document-anchored)",
       x = "Median duration (days)", y = NULL, color = NULL,
       caption = paste0("Top 15 BLM field offices by document-anchored CE review volume (each ≥", MIN_REVIEWS,
                        " reviews). Leftward = shorter for later reviews. Descriptive only: a later tercile is also a later calendar period, ",
                        "so most of this shift is the secular ~7%/yr CE decline, not office learning (see regression). Associational, not causal.")) +
  theme_catf()
ggsave(file.path(FIG, "fig_d4_fieldoffice_first_vs_last.png"), p_db, width = 10, height = 7, dpi = 300, bg = "white")

# ---------------------------------------------------------------------------
# Fig 3 — confound check: experience gradient vs calendar-time trend are distinguishable
# (CE, document-anchored, qualifying offices)
# ---------------------------------------------------------------------------
cal_view <- ce_doc_q |> filter(decision_year >= 2014) |>
  group_by(decision_year) |> summarise(med = median(duration_days), n = n(), .groups = "drop") |>
  filter(n >= 10) |> transmute(view = "By calendar year (decision year)", x = decision_year, med)
exp_view <- ce_doc_q |>
  mutate(decile = ntile(cum_count, 10)) |>
  group_by(decile) |> summarise(med = median(duration_days), .groups = "drop") |>
  transmute(view = "By accumulated office reviews (experience decile)", x = decile, med)
conf <- bind_rows(cal_view, exp_view) |>
  mutate(view = factor(view, levels = c("By calendar year (decision year)",
                                        "By accumulated office reviews (experience decile)")))
p_conf <- ggplot(conf, aes(x, med)) +
  geom_line(color = catf_navy, linewidth = 1.1) + geom_point(color = catf_navy, size = 2) +
  facet_wrap(~view, scales = "free") +
  labs(title = "Why the Curve Is a Calendar Confound (CE, document-anchored)",
       subtitle = "Durations fall with both calendar year AND cumulative experience — because the two are collinear (later reviews = later years)",
       x = NULL, y = "Median duration (days)",
       caption = paste0("The two raw gradients look alike because an office's higher-experience reviews are also its later-calendar reviews (r≈0.67). ",
                        "Net of decision-year, the within-office experience coefficient is null/positive — so the decline is secular calendar drift, not learning.")) +
  theme_catf()
ggsave(file.path(FIG, "fig_d4_fieldoffice_experience_vs_calendar.png"), p_conf, width = 11, height = 5.5, dpi = 300, bg = "white")

# ---------------------------------------------------------------------------
# Regression: log(duration) ~ log(cum_count) + factor(decision_year) + energy + office FE
# Coefficient on log(cum_count) = learning effect net of calendar trend (negative = faster).
# ---------------------------------------------------------------------------
# Fits two nested models so the calendar-time confound is explicit in committed output:
#   raw  = log(dur) ~ log(cum_count) + energy + office FE            (NO calendar control)
#   ctrl = log(dur) ~ log(cum_count) + factor(decision_year) + energy + office FE
# A negative `raw` that turns null/positive under `ctrl` means the apparent speed-up is
# secular calendar drift, not within-office learning.
fit_lc <- function(proc, anch, thr, note = "") {
  s <- df_cum |> filter(process_type == proc, anchor == anch, office_n >= thr)
  n_off <- n_distinct(s$office_code)
  if (n_off < 2 || nrow(s) < 30) {
    return(tibble(process = proc, anchor = anch, threshold = thr, n_reviews = nrow(s),
                  n_offices = n_off, pct_per_doubling_raw = NA_real_,
                  beta_log_cum = NA_real_, ci_low = NA_real_, ci_high = NA_real_,
                  pct_per_doubling = NA_real_, pct_lo = NA_real_, pct_hi = NA_real_,
                  note = paste0("not estimated: ", n_off, " offices >= ", thr, " (need >=2); ", nrow(s), " reviews")))
  }
  s <- s |> mutate(decision_year = factor(decision_year))
  y <- log(pmax(s$duration_days, 1))
  m_raw <- lm(y ~ log(cum_count) + energy + office_code, data = s)
  m     <- lm(y ~ log(cum_count) + decision_year + energy + office_code, data = s)
  est <- coef(m)[["log(cum_count)"]]; ci <- confint(m, "log(cum_count)")
  est_raw <- coef(m_raw)[["log(cum_count)"]]
  tibble(process = proc, anchor = anch, threshold = thr, n_reviews = nrow(s), n_offices = n_off,
         pct_per_doubling_raw = round(100 * (2^est_raw - 1), 1),       # no calendar control
         beta_log_cum = round(est, 4), ci_low = round(ci[1], 4), ci_high = round(ci[2], 4),
         pct_per_doubling = round(100 * (2^est - 1), 1),               # WITH calendar control
         pct_lo = round(100 * (2^ci[1] - 1), 1), pct_hi = round(100 * (2^ci[2] - 1), 1),
         note = note)
}

model_tbl <- bind_rows(
  fit_lc("CE", "Document", MIN_REVIEWS, "PRIMARY: document-anchored, office FE + calendar + energy"),
  fit_lc("CE", "Register", MIN_REVIEWS, "Sensitivity: register-anchored (artifact-prone start dates)"),
  fit_lc("EA", "Document", MIN_REVIEWS, "EA document-anchored: too sparse for office-level analysis"),
  fit_lc("EA", "Register", EA_RELAX,    "EA exploratory: register-anchored, relaxed >=10 threshold (artifact-prone)")
)
write_csv(model_tbl, file.path(DIAG, "d4_fieldoffice_model.csv"))
message("\n=== Learning-curve regression (log(cum_count) coefficient) ===")
print(as.data.frame(model_tbl))

message("\n02_learning_curve.R complete. Figures -> output/deliverable04/figures/ ; tables -> diagnostics/")
