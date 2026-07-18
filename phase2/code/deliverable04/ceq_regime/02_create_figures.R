# D4 CEQ regulatory-regime timelines: two figures from the diagnostics CSVs built by
# 01_build_tables.py. Segments review durations by the CEQ NEPA implementing regulation
# in effect at the DECISION date. This is figures ONLY — no parquet reads, no aggregation
# beyond trivial plot prep; all data construction lives in 01_build_tables.py.
#
# Reads (phase2/output/deliverable04/diagnostics/):
#   d4_duration_by_ceq_regime.csv  — collapsed 4-level regimes + 1978 recent-window sensitivity
#   d4_duration_by_year.csv        — existing 08_create_figures.R diagnostic (Figure B trend)
#
# Outputs (phase2/output/deliverable04/figures/):
#   fig_d4_duration_by_ceq_regime.png       — interval chart, collapsed regime x process
#   fig_d4_duration_trend_ceq_regime.png    — annual median-duration trend with CEQ + FRA markers
#
# CEQ rule effective dates (decimal-year x-positions on the trend): 2020-09-14 -> 2020.70,
# 2022-05-20 -> 2022.38, 2024-07-01 -> 2024.50, 2025-04-11 (rescission effective) -> 2025.28.
# FRA (statutory, 2023-06-03 -> 2023.42) is shown distinctly (teal dotted) — it is NOT a CEQ rule.
#
# Usage: Rscript phase2/code/deliverable04/ceq_regime/02_create_figures.R

suppressPackageStartupMessages({
  library(here); library(dplyr); library(tidyr)
  library(readr); library(stringr); library(ggplot2); library(scales)
})

PHASE2 <- here::here("phase2")
DIAG   <- file.path(PHASE2, "output", "deliverable04", "diagnostics")
FIGS   <- file.path(PHASE2, "output", "deliverable04", "figures")
dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# CATF brand colors and theme (verbatim from 08_create_figures.R)
# ---------------------------------------------------------------------------

catf_dark_blue  <- "#0047BB"
catf_blue       <- "#00B5E2"
catf_magenta    <- "#C22A90"
catf_purple     <- "#75246C"
catf_lime       <- "#93D500"
catf_teal       <- "#00AE8D"
catf_light_blue <- "#8AB7E9"
catf_navy       <- "#012169"

PROCESS_LEVELS <- c("CE", "EA", "EIS")
PROCESS_COLORS <- c("CE" = catf_lime, "EA" = catf_dark_blue, "EIS" = catf_navy)
MONTHS <- 30.44
MIN_N  <- 30

theme_catf <- function(base_size = 11, base_family = "Helvetica") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      plot.title       = element_text(face = "bold", size = rel(1.2), color = catf_navy,
                                      margin = margin(b = 10)),
      plot.subtitle    = element_text(size = rel(0.9), color = catf_dark_blue,
                                      margin = margin(b = 10)),
      plot.caption     = element_text(size = rel(0.8), color = "gray50", hjust = 1),
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

theme_set(theme_catf())

# ---------------------------------------------------------------------------
# Figure A: median duration by collapsed CEQ regime, faceted by process
# ---------------------------------------------------------------------------

REGIME_LABELS <- c(
  "1978"             = "1978 rules",
  "2020_trump"       = "2020 (Trump)",
  "2022_phase1"      = "2022 (Phase 1)",
  "2024_phase2_plus" = "2024+ (Phase 2 / rescission)"
)
# Factor levels ordered bottom -> top so 1978 sits at the TOP of each facet.
REGIME_ORDER <- rev(names(REGIME_LABELS))

ceq <- read_csv(file.path(DIAG, "d4_duration_by_ceq_regime.csv"), show_col_types = FALSE)

reg_df <- ceq |>
  filter(regime_level == "collapsed", anchor == "decision") |>
  mutate(
    p10           = p10_days / MONTHS,
    p25           = p25_days / MONTHS,
    p75           = p75_days / MONTHS,
    p90           = p90_days / MONTHS,
    median_months = median_days / MONTHS,
    process_group = factor(process_type, levels = PROCESS_LEVELS),
    regime_f      = factor(ceq_regime, levels = REGIME_ORDER, labels = REGIME_LABELS[REGIME_ORDER]),
    row_color     = if_else(display, "shown", "suppressed"),
    n_label       = if_else(display, paste0("n=", comma(n)),
                            paste0("n=", comma(n), " (n<", MIN_N, ")"))
  )

# 1978 recent-window comparator tick (2015-2020 decisions), one per process facet.
recent_tick <- ceq |>
  filter(regime_level == "sensitivity", anchor == "decision") |>
  transmute(process_group = factor(process_type, levels = PROCESS_LEVELS),
            recent_months = median_days / MONTHS,
            regime_f = factor("1978 rules", levels = REGIME_LABELS[REGIME_ORDER]))

fig_regime <- ggplot(reg_df, aes(y = regime_f)) +
  # p10-p90 whisker + IQR bar + median point; grey where n<MIN_N (none currently, kept robust)
  geom_segment(aes(x = p10, xend = p90, yend = regime_f, color = process_group, alpha = row_color),
               linewidth = 1.8) +
  geom_segment(aes(x = p25, xend = p75, yend = regime_f, color = process_group, alpha = row_color),
               linewidth = 5.5) +
  geom_point(aes(x = median_months, color = process_group, alpha = row_color), size = 3.0) +
  # dashed reference tick: 1978-rule median for 2015-2020 decisions only (secular-decline comparator)
  geom_segment(data = recent_tick, inherit.aes = FALSE, linetype = "dashed", color = "gray35",
               aes(x = recent_months, xend = recent_months,
                   y = as.numeric(regime_f) - 0.36, yend = as.numeric(regime_f) + 0.36)) +
  geom_text(aes(x = median_months, label = sprintf("%.0f mo", median_months)),
            nudge_y = 0.30, size = 3.0, fontface = "bold", color = "gray20") +
  geom_text(aes(x = p90, label = n_label), nudge_x = 1.5, hjust = 0, size = 2.8, color = "gray40") +
  facet_wrap(~process_group, scales = "free_x", ncol = 1) +
  scale_color_manual(values = PROCESS_COLORS, guide = "none") +
  scale_alpha_manual(values = c("shown" = 0.55, "suppressed" = 0.20), guide = "none") +
  scale_x_continuous(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.02, 0.16))) +
  labs(
    title    = "Review Duration by CEQ Regulatory Regime",
    subtitle = paste0("Thin bar = p10-p90, thick bar = IQR, point = median. Dashed tick on the 1978 row = ",
                      "1978-rule median for 2015-2020 decisions only.\nRows with n<", MIN_N, " are greyed."),
    x = "Duration (months)", y = NULL,
    caption = "Regime = CEQ rule in effect at the decision date. 2024 Phase 2 + 2025 rescission collapsed."
  )

ggsave(file.path(FIGS, "fig_d4_duration_by_ceq_regime.png"),
       fig_regime, width = 11, height = 9.5, dpi = 300)
saveRDS(fig_regime, file.path(FIGS, "fig_d4_duration_by_ceq_regime.rds"))
message("Wrote fig_d4_duration_by_ceq_regime.png")

# ---------------------------------------------------------------------------
# Figure B: annual median-duration trend with CEQ rule + FRA markers
# ---------------------------------------------------------------------------

TREND_START <- 2005   # x-axis start: pre-2005 annual medians are sparse/noisy (few reviews)
dur_year <- read_csv(file.path(DIAG, "d4_duration_by_year.csv"), show_col_types = FALSE) |>
  filter(n >= 5, decision_year >= TREND_START) |>   # CSV unchanged; this trims the *view* only
  mutate(process_type = factor(process_type, levels = PROCESS_LEVELS))

# CEQ rule + FRA markers as decimal years. Single-line abbreviated labels, angled and
# drawn on the TOP facet only (process_type = "CE") so they never repeat or collide.
marks <- tibble(
  x     = c(2020.70,       2022.38,     2023.42,  2024.50,     2025.28),
  label = c("2020 rule",   "Phase 1",   "FRA",    "Phase 2",   "Rescind"),
  kind  = c("CEQ",         "CEQ",       "FRA",    "CEQ",       "CEQ")
)
ceq_x <- marks$x[marks$kind == "CEQ"]
fra_x <- marks$x[marks$kind == "FRA"]
# Anchor the labels at a real y INSIDE the top (CE) panel's headroom (not y=Inf,
# which would clip against the facet strip). Horizontal labels are wider than the
# angled ones were, so alternate two height tiers to keep neighbors collision-free.
ce_top  <- max(dur_year$median_months[dur_year$process_type == "CE"], na.rm = TRUE)
lab_y   <- c("2020 rule" = 1.28, "Phase 1" = 1.10, "FRA" = 1.28,
             "Phase 2" = 1.10, "Rescind" = 1.28)
marks_lab <- marks |> mutate(process_type = factor("CE", levels = PROCESS_LEVELS),
                             y = ce_top * lab_y[label])

# Faceted by process, stacked (ncol = 1) with free y — CE runs in fractions of a
# month while EIS runs in tens of months, so a shared scale would flatten CE/EA.
fig_trend <- ggplot(dur_year, aes(x = decision_year, y = median_months, color = process_type)) +
  # rule markers lightened (alpha) so the data lines dominate: CEQ navy dashed, FRA grey dotted
  geom_vline(xintercept = ceq_x, linetype = "dashed", color = catf_navy, linewidth = 0.4, alpha = 0.45) +
  geom_vline(xintercept = fra_x, linetype = "dotted", color = "gray35", linewidth = 0.6, alpha = 0.6) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.6, alpha = 0.8) +
  # horizontal marker labels on the top (CE) facet only, one color, two height tiers
  geom_text(data = marks_lab, inherit.aes = FALSE, aes(x = x, y = y, label = label),
            hjust = 0.5, vjust = 0.5, size = 2.5, color = catf_navy) +
  facet_wrap(~process_type, ncol = 1, scales = "free_y") +
  scale_color_manual(values = PROCESS_COLORS, guide = "none") +
  scale_x_continuous(breaks = seq(TREND_START, 2025, 5), limits = c(TREND_START, 2026.5)) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.34))) +   # headroom for the angled top labels
  coord_cartesian(clip = "off") +
  labs(
    title    = "Median Review Duration by Year, with CEQ Rule Markers",
    subtitle = paste0("Faceted by process (free y-axis: CE runs in fractions of a month, EIS in tens). ",
                      "Navy dashed = CEQ rule effective dates;\ngrey dotted = FRA (statutory). Years with n>=5 only."),
    x = "Decision year", y = "Median duration (months)", color = NULL,
    caption = paste0("View starts at ", TREND_START, "; sparse pre-", TREND_START,
                     " annual medians are omitted for clarity (data unchanged).\n",
                     "Line ends at 2025 (partial-2026 below the n>=5 floor); ",
                     "the 2025 median straddles the April rescission.")
  )

ggsave(file.path(FIGS, "fig_d4_duration_trend_ceq_regime.png"),
       fig_trend, width = 10, height = 8.5, dpi = 300)
saveRDS(fig_trend, file.path(FIGS, "fig_d4_duration_trend_ceq_regime.rds"))
message("Wrote fig_d4_duration_trend_ceq_regime.png")

# ---------------------------------------------------------------------------
# Figure C: CEQ regulatory-regime orientation timeline (not a data figure)
# ---------------------------------------------------------------------------
# A compact horizontal timeline of the CEQ regime sequence so a reader can see the
# rapid 2020->2025 turnover at a glance. Regime cut dates MIRROR the constants in
# 01_build_tables.py (CEQ_TRUMP / CEQ_PHASE1 / CEQ_PHASE2 / CEQ_RESCIND). Bands run a
# MONOCHROMATIC blue ramp (light -> navy) across the successive CEQ rules, grey for the
# post-rescission gap; the FRA enactment (2023-06-03) is a grey dotted marker because it
# is a *statutory* change to NEPA, not a CEQ implementing rule. Orientation only — no data.

as_d <- function(x) as.Date(x)

# Monochromatic blue ramp anchored to the existing CATF blues (light_blue -> dark_blue ->
# navy) with one interpolated mid-step; NO cyan (catf_blue) and NO teal anywhere. The
# post-rescission gap is grey (no CEQ regs in force).
catf_mid_blue <- "#457FD2"   # midpoint of catf_light_blue (#8AB7E9) and catf_dark_blue (#0047BB)

# Regime bands. 1978's left edge is synthetic (the corpus reaches back ~42 yrs); the
# drawn axis starts in mid-2018.
regime_bands <- tibble(
  regime = factor(c("1978 rules", "2020 Trump Rule", "2022 Phase 1",
                    "2024 Phase 2", "2025 rescission"),
                  levels = c("1978 rules", "2020 Trump Rule", "2022 Phase 1",
                             "2024 Phase 2", "2025 rescission")),
  start  = as_d(c("2018-06-01", "2020-09-14", "2022-05-20", "2024-07-01", "2025-04-11")),
  end    = as_d(c("2020-09-14", "2022-05-20", "2024-07-01", "2025-04-11", "2026-06-01")),
  fill   = c(catf_light_blue, catf_mid_blue, catf_dark_blue, catf_navy, "gray80")
) |>
  mutate(mid = start + as.numeric(end - start) / 2)

# Wide bands carry their label inside; the two narrow recent bands carry it above.
# Contrast: navy text on the light 1978 band, white text on the darker mid/dark bands.
inside_lab <- regime_bands[1:3, ] |>
  mutate(short = c("1978 rules", "2020\nTrump Rule", "2022\nPhase 1"),
         txt   = c(catf_navy, "white", "white"))
above_lab <- regime_bands[4:5, ] |>
  mutate(short = c("2024 Phase 2", "2025 rescission"), ly = c(1.5, 2.05))

# Rule effective-date markers (exclude the synthetic 1978 left edge); stagger the
# close 2024-07 / 2025-04 pair vertically so the date labels do not collide.
rule_marks <- tibble(
  date  = as_d(c("2020-09-14", "2022-05-20", "2024-07-01", "2025-04-11")),
  label = c("Sep 14, 2020", "May 20, 2022", "Jul 1, 2024", "Apr 11, 2025"),
  ly    = c(-0.30, -0.30, -0.30, -0.62)
)
fra_mark <- as_d("2023-06-03")

axis_min <- as_d("2018-06-01"); axis_max <- as_d("2026-06-01")

fig_timeline <- ggplot() +
  # regime bands
  geom_rect(data = regime_bands,
            aes(xmin = start, xmax = end, ymin = 0, ymax = 1, fill = regime),
            color = "white", linewidth = 0.5) +
  scale_fill_manual(values = setNames(regime_bands$fill, levels(regime_bands$regime)),
                    guide = "none") +
  # inside-band labels (wide bands)
  geom_text(data = inside_lab, aes(x = mid, y = 0.5, label = short, color = txt),
            fontface = "bold", size = 2.9, lineheight = 0.85) +
  scale_color_identity() +
  # above-band labels (narrow bands) with leader lines
  geom_segment(data = above_lab, aes(x = mid, xend = mid, y = 1.02, yend = ly - 0.08),
               color = "gray55", linewidth = 0.3) +
  geom_text(data = above_lab, aes(x = mid, y = ly, label = short),
            fontface = "bold", size = 2.7, color = catf_navy) +
  # rule effective-date ticks + date labels below the bar
  geom_segment(data = rule_marks, aes(x = date, xend = date, y = 0, yend = -0.12),
               color = "gray35", linewidth = 0.4) +
  geom_text(data = rule_marks, aes(x = date, y = ly, label = label),
            size = 2.4, color = "gray25") +
  # FRA statutory marker (grey dotted), crossing the whole bar to read as a different
  # kind of change; dotted + full-height distinguishes it from the solid band boundaries.
  geom_segment(aes(x = fra_mark, xend = fra_mark, y = -0.12, yend = 1.12),
               color = "gray35", linetype = "dotted", linewidth = 0.8) +
  annotate("text", x = fra_mark, y = 1.32, label = "FRA takes effect June 3, 2023",
           size = 2.7, color = catf_navy, fontface = "bold") +
  scale_x_date(limits = c(axis_min, axis_max), date_breaks = "1 year",
               date_labels = "%Y", expand = expansion(mult = c(0.01, 0.01))) +
  coord_cartesian(ylim = c(-0.75, 2.35), clip = "off") +
  labs(
    title    = "The CEQ Regulatory Regimes, 2018-2026",
    subtitle = paste0("NEPA's implementing rules were rewritten three times in four years. Bands deepen from light ",
                      "to navy blue across successive\nCEQ rules; grey = post-rescission (agencies revert to own ",
                      "procedures). The dotted grey line marks the FRA (statutory)."),
    x = NULL, y = NULL
  ) +
  theme(
    axis.text.y  = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line    = element_blank(),
    panel.grid   = element_blank(),
    axis.text.x  = element_text(size = rel(0.85), color = "gray30")
  )

ggsave(file.path(FIGS, "fig_d4_ceq_regime_timeline.png"),
       fig_timeline, width = 10, height = 3.6, dpi = 300)
saveRDS(fig_timeline, file.path(FIGS, "fig_d4_ceq_regime_timeline.rds"))
message("Wrote fig_d4_ceq_regime_timeline.png")
