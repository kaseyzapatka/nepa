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
       fig_regime, width = 11, height = 7, dpi = 300)
saveRDS(fig_regime, file.path(FIGS, "fig_d4_duration_by_ceq_regime.rds"))
message("Wrote fig_d4_duration_by_ceq_regime.png")

# ---------------------------------------------------------------------------
# Figure B: annual median-duration trend with CEQ rule + FRA markers
# ---------------------------------------------------------------------------

dur_year <- read_csv(file.path(DIAG, "d4_duration_by_year.csv"), show_col_types = FALSE) |>
  filter(n >= 5) |>
  mutate(process_type = factor(process_type, levels = PROCESS_LEVELS))

# CEQ rule effective dates as decimal years; staggered label heights so the close
# 2024.50 / 2025.28 pair does not collide.
ceq_marks <- tibble(
  x     = c(2020.70,        2022.38,          2024.50,          2025.28),
  label = c("Trump\n2020",  "Phase 1\n2022",  "Phase 2\n2024",  "Rescind\n2025"),
  vjust = c(1.5,            3.0,              1.5,              3.0)
)
FRA_X <- 2023.42  # FRA enactment 2023-06-03 (statutory, not a CEQ rule)

fig_trend <- ggplot(dur_year, aes(x = decision_year, y = median_months, color = process_type)) +
  # CEQ rule markers (navy dashed) + FRA statutory marker (teal dotted)
  geom_vline(xintercept = ceq_marks$x, linetype = "dashed", color = catf_navy, linewidth = 0.5) +
  geom_vline(xintercept = FRA_X, linetype = "dotted", color = catf_teal, linewidth = 0.7) +
  geom_line(linewidth = 0.8) +
  geom_point(aes(size = n), alpha = 0.7) +
  geom_text(data = ceq_marks, inherit.aes = FALSE,
            aes(x = x, y = Inf, label = label), vjust = ceq_marks$vjust, hjust = -0.08,
            size = 2.8, color = catf_navy, lineheight = 0.85) +
  annotate("text", x = FRA_X, y = Inf, label = "FRA\n(statutory)", vjust = 5.2, hjust = -0.08,
           size = 2.8, color = catf_teal, fontface = "italic", lineheight = 0.85) +
  scale_color_manual(values = PROCESS_COLORS) +
  scale_size_continuous(range = c(1, 4), guide = "none") +
  scale_x_continuous(breaks = seq(1990, 2026, 5)) +
  labs(
    title    = "Median Review Duration by Year, with CEQ Rule Markers",
    subtitle = "Navy dashed = CEQ rule effective dates; teal dotted = FRA (statutory). Years with n>=5 only.",
    x = "Decision year", y = "Median duration (months)", color = NULL,
    caption = "Line ends at 2025 (partial-2026 below the n>=5 floor); the 2025 annual median straddles the April rescission."
  )

ggsave(file.path(FIGS, "fig_d4_duration_trend_ceq_regime.png"),
       fig_trend, width = 10, height = 5, dpi = 150)
saveRDS(fig_trend, file.path(FIGS, "fig_d4_duration_trend_ceq_regime.rds"))
message("Wrote fig_d4_duration_trend_ceq_regime.png")
