# D4 geothermal timelines by BLM field office: three figures from the diagnostics CSVs built
# by 01_build_tables.py. Figures ONLY — no parquet reads, no data construction beyond trivial
# plot prep. The `maps` package is used solely for lower-48 state polygons (geometry, not data).
#
# Reads (phase2/output/deliverable04/diagnostics/):
#   d4_geothermal_universe.csv        — funnel/tier/lead-agency rows (subtitle context)
#   d4_geothermal_office_counts.csv   — per office x process (+ ALL) + two baseline rows (Fig A)
#   d4_geothermal_state_map.csv       — per state x cohort, CE only (Fig B)
#   d4_geothermal_timeline_points.csv — CE annual medians + every EA/EIS project (Fig C)
#
# Outputs (phase2/output/deliverable04/figures/):
#   fig_d4_geothermal_offices_by_process.png — office/tier inventory bars + two baseline bars
#   fig_d4_geothermal_map.png                — CE state bubble map (size = median months)
#   fig_d4_geothermal_timeline.png           — CE annual-median line + every EA/EIS as a point
#
# House PROCESS_COLORS (CE = lime, EA = dark blue, EIS = navy) for all process-coded elements;
# cohorts use catf_navy (BLM) vs catf_light_blue (DOE/Other). NO teal/cyan anywhere.
#
# Usage: Rscript phase2/code/deliverable04/geothermal/02_create_figures.R

suppressPackageStartupMessages({
  library(here); library(dplyr); library(tidyr)
  library(readr); library(stringr); library(ggplot2); library(scales); library(maps)
  library(patchwork)
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
COHORT_COLORS  <- c("BLM" = catf_navy, "DOE/Other" = catf_light_blue)
MONTHS <- 30.44

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

universe <- read_csv(file.path(DIAG, "d4_geothermal_universe.csv"), show_col_types = FALSE)
uni <- function(stage, col) universe[[col]][universe$stage == stage]

# ===========================================================================
# Figure A: two side-by-side panels (patchwork).
#   LEFT  — where the 873 geothermal projects sit: one stacked column, three tiers.
#   RIGHT — the 61 office-matched projects by field office, stacked by process.
# ===========================================================================

oc <- read_csv(file.path(DIAG, "d4_geothermal_office_counts.csv"), show_col_types = FALSE)
BASELINE_CODES <- c("(no office match)", "(DOE & other)")

# --- LEFT: tier funnel (stacked column of the 873, three tiers) --------------
# Tier colours stay in the blue family and match the map (BLM darker, DOE lighter).
TIER_LEVELS <- c("BLM, office-matched", "BLM, no office match", "DOE & other")
TIER_COLORS <- c("BLM, office-matched" = catf_navy,
                 "BLM, no office match" = catf_dark_blue,
                 "DOE & other"          = catf_light_blue)
tiers <- tibble(
  tier = factor(TIER_LEVELS, levels = TIER_LEVELS),
  n    = c(uni("office_matched", "n"), uni("unmatched_blm", "n"), uni("doe_other", "n"))
) |>
  arrange(match(tier, rev(TIER_LEVELS))) |>          # stack DOE at bottom, office-matched on top
  mutate(ymax = cumsum(n), ymid = ymax - n / 2)

# The DOE segment is tall enough for an inside label; the two thin BLM segments
# (61 and 48 of 873, ~5% of the bar each) are not — their labels sit outside to
# the right in the tier colour, level with each segment's midpoint.
tiers_in  <- tiers |> filter(tier == "DOE & other")
tiers_out <- tiers |> filter(tier != "DOE & other")
fig_a_left <- ggplot(tiers, aes(x = 1, y = n, fill = tier)) +
  geom_col(width = 0.9, color = "white", linewidth = 0.4) +
  geom_text(data = tiers_in, aes(x = 1, y = ymid, label = paste0(tier, "\n", comma(n))),
            color = catf_navy, fontface = "bold", size = 3.0, lineheight = 0.9) +
  geom_text(data = tiers_out,
            aes(x = 1.5, y = ymid, label = paste0(tier, " — ", comma(n)), color = tier),
            hjust = 0, fontface = "bold", size = 2.9) +
  scale_color_manual(values = TIER_COLORS, guide = "none") +
  scale_fill_manual(values = TIER_COLORS, guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.02))) +
  coord_cartesian(xlim = c(0.5, 3.6), clip = "off") +
  labs(title = "Where the 873 projects sit", x = NULL, y = "Projects") +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank())

# --- RIGHT: the 61 office-matched projects by field office -------------------
off_proc <- oc |>
  filter(!office_code %in% BASELINE_CODES, process_type %in% PROCESS_LEVELS) |>
  mutate(process_type = factor(process_type, levels = PROCESS_LEVELS))
off_tot <- oc |>
  filter(!office_code %in% BASELINE_CODES, process_type == "ALL") |>
  transmute(office_code, tot = n_parsed) |>
  arrange(tot)                                         # ascending -> biggest on top
off_proc <- off_proc |> mutate(office_f = factor(office_code, levels = off_tot$office_code))
off_lab  <- off_tot |> mutate(office_f = factor(office_code, levels = off_tot$office_code),
                              lab = comma(tot))

fig_a_right <- ggplot(off_proc, aes(x = n_parsed, y = office_f, fill = process_type)) +
  geom_col(width = 0.7) +
  geom_text(data = off_lab, inherit.aes = FALSE,
            aes(x = tot + 0.3, y = office_f, label = lab),
            hjust = 0, size = 2.8, color = "gray35") +
  # breaks limited to processes actually present — no office-matched project is
  # an EIS, and an unused navy key reads as a blank swatch at report scale
  scale_fill_manual(values = PROCESS_COLORS, name = "Process", breaks = c("CE", "EA")) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.10))) +
  labs(title = paste0("The ", uni("office_matched", "n"), " office-matched projects by field office"),
       x = "Projects", y = NULL) +
  theme(legend.position = "bottom")

# --- FAR RIGHT: the DOE tier's register offices (light blue) --------------------------------
# Newly feasible: the DOE grant tier links to the CX register's `office` field, so the ~60% of
# non-BLM geothermal projects with a canonical office get a proper inventory — Golden dominant.
doe_oc <- read_csv(file.path(DIAG, "d4_geothermal_doe_office_counts.csv"), show_col_types = FALSE)
doe_total  <- doe_oc |> filter(office == "ALL") |> pull(n_parsed)          # 456 with a register office
DOE_SHORT  <- c("Golden Field Office" = "Golden FO",
                "National Energy Technology Laboratory" = "NETL",
                "Energy Efficiency and Renewable Energy" = "EERE-HQ",
                "RMOTC" = "RMOTC")
doe_named <- doe_oc |> filter(office != "ALL", n_parsed >= 5) |>
  transmute(office = recode(office, !!!DOE_SHORT), n_parsed)
doe_tail_n <- doe_oc |> filter(office != "ALL", n_parsed < 5) |> summarise(n = sum(n_parsed)) |> pull(n)
doe_bars <- bind_rows(doe_named,
                      tibble(office = "Other (truncated)", n_parsed = doe_tail_n)) |>
  arrange(n_parsed) |> mutate(office = factor(office, levels = office))
fig_a_doe <- ggplot(doe_bars, aes(n_parsed, office)) +
  geom_col(fill = catf_light_blue, width = 0.7) +
  geom_text(aes(label = comma(n_parsed)), hjust = -0.2, size = 2.8, color = "gray35") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.14))) +
  labs(title = paste0("The ", comma(doe_total), " DOE-tier projects by register office"),
       x = "Projects", y = NULL) +
  theme(panel.grid.major.y = element_blank())

fig_a <- fig_a_left + fig_a_right + fig_a_doe +
  plot_layout(widths = c(0.85, 1.7, 1.7)) +
  plot_annotation(
    title = "Geothermal review inventory: BLM field offices vs DOE register offices, in a DOE-dominated universe",
    subtitle = paste0("Of ", comma(uni("total_geothermal", "n")), " geothermal projects, only ",
                      comma(uni("blm_led", "n")), " are BLM-led (just ", comma(uni("office_matched", "n")),
                      " with a parseable field-office code); the ", comma(uni("doe_other", "n")),
                      " DOE-tier projects instead link to the CX register, ", comma(doe_total), " to a named office."),
    caption = paste0("Counts are all timeline states (an inventory, not a duration comparison). Middle panel stacked by review process. ",
                     "DOE 'offices' are administering / grant-program offices (Golden, NETL, EERE-HQ), not BLM-style field offices."),
    theme = theme(plot.title    = element_text(face = "bold", size = rel(1.15), color = catf_navy),
                  plot.subtitle = element_text(size = rel(0.85), color = catf_dark_blue),
                  plot.caption  = element_text(size = rel(0.8), color = "gray50", hjust = 0))
  )

ggsave(file.path(FIGS, "fig_d4_geothermal_offices_by_process.png"),
       fig_a, width = 13, height = 6, dpi = 300)
saveRDS(fig_a, file.path(FIGS, "fig_d4_geothermal_offices_by_process.rds"))
message("Wrote fig_d4_geothermal_offices_by_process.png")

# ===========================================================================
# Figure B: CE state bubble map — size = median duration (months), color = cohort
# ===========================================================================

smap <- read_csv(file.path(DIAG, "d4_geothermal_state_map.csv"), show_col_types = FALSE) |>
  filter(display) |>
  mutate(cohort = factor(cohort, levels = c("BLM", "DOE/Other")))

# cohort CE Ns for the subtitle (the two-worlds contrast), read from the office-floor summary
geo_floor <- read_csv(file.path(DIAG, "d4_geothermal_office_floor.csv"), show_col_types = FALSE)
blm_ce_n    <- geo_floor$blm_ce_n[1];    blm_ce_med    <- round(geo_floor$blm_ce_median_days[1])
nonblm_ce_n <- geo_floor$nonblm_ce_n[1]; nonblm_ce_med <- round(geo_floor$nonblm_ce_median_days[1])

# offset the two cohorts horizontally where both are drawn in the same state (NV, CA, UT)
shared <- smap |> count(state) |> filter(n > 1) |> pull(state)
smap <- smap |>
  mutate(lon_off = case_when(
           state %in% shared & cohort == "BLM"       ~ lon - 1.15,
           state %in% shared & cohort == "DOE/Other"  ~ lon + 1.15,
           TRUE ~ lon))

us <- map_data("state")

# Direct on-map cohort labels (a colored key dot + text) placed in open areas, so the
# cohort encoding survives the report's out-width down-scaling without a bottom legend.
cohort_keys <- tibble(
  cohort = factor(c("BLM", "DOE/Other"), levels = c("BLM", "DOE/Other")),
  x   = c(-123.5, -101.0),   # off the Pacific coast; over the southern plains
  y   = c(33.0,   28.5),
  lab = c("BLM (western resource)", "DOE & other (grant CEs, nationwide)")
)

fig_b <- ggplot() +
  geom_polygon(data = us, aes(x = long, y = lat, group = group),
               fill = "gray93", color = "white", linewidth = 0.3) +
  geom_point(data = smap, aes(x = lon_off, y = lat, size = median_months, fill = cohort),
             shape = 21, color = "white", stroke = 0.6, alpha = 0.9) +
  geom_text(data = smap, aes(x = lon_off, y = lat - 1.5, label = paste0("n=", n_complete)),
            size = 2.6, color = "gray25") +
  # direct cohort key: large colored dot + bold label, on the map itself
  geom_point(data = cohort_keys, aes(x = x, y = y, fill = cohort), shape = 21,
             size = 6, color = "white", stroke = 0.8, show.legend = FALSE) +
  geom_text(data = cohort_keys, aes(x = x + 1.2, y = y, label = lab, color = cohort),
            hjust = 0, fontface = "bold", size = 4.2, show.legend = FALSE) +
  scale_fill_manual(values = COHORT_COLORS, guide = "none") +
  scale_color_manual(values = c("BLM" = catf_navy, "DOE/Other" = catf_dark_blue), guide = "none") +
  scale_size_area(max_size = 15, name = "Median duration (months)",
                  breaks = c(0.5, 2, 5, 10, 15)) +
  coord_quickmap() +   # aspect-ratio-correct lower-48; mapproj not installed, so no projection
  guides(size = guide_legend(nrow = 1, override.aes = list(fill = "gray55", color = "white"))) +
  labs(
    title    = "Geothermal CE review duration by state and cohort",
    subtitle = paste0("Two geothermal worlds: BLM (n=", blm_ce_n, ", median ", blm_ce_med,
                      "d) sits in the western resource states (NV/CA/UT); DOE & other (n=", nonblm_ce_n,
                      ", median ", nonblm_ce_med, "d)\nblanket the country as grant CEs. Bubble size = median CE ",
                      "duration in months (not project volume)."),
    x = NULL, y = NULL,
    caption = paste0("One bubble per state x cohort with n ≥ 3 complete CE timelines; n printed ",
                     "below each. Alaska (6 DOE CEs) not drawn.\nShared states (NV/CA/UT) are offset ",
                     "so both cohorts show. Single-state projects only; size encodes duration, not count.")
  ) +
  theme(axis.text = element_blank(), axis.line = element_blank(), axis.ticks = element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        legend.position = "bottom",
        legend.title = element_text(size = rel(1.15), face = "bold", color = catf_navy),
        legend.text  = element_text(size = rel(1.1), color = "gray25"),
        legend.key.size = unit(1.4, "lines"))

ggsave(file.path(FIGS, "fig_d4_geothermal_map.png"),
       fig_b, width = 10, height = 6.5, dpi = 300)
saveRDS(fig_b, file.path(FIGS, "fig_d4_geothermal_map.rds"))
message("Wrote fig_d4_geothermal_map.png")

# ===========================================================================
# Figure C: faceted linear panels, one per review process (ncol = 1, free y).
#   CE  — annual-median line (n >= 5 years, gaps honest) + ARRA guide.
#   EA  — its 8 projects as individual points.
#   EIS — its 14 projects as individual points (the 138-mo 2020 BLM EIS sets the scale).
# Linear axes (zeros plot fine); cohort shape in the EA/EIS panels. Chosen over a
# Gantt/span view because the timeline CSV carries no initiation dates (a Gantt would
# require a builder/CSV change, which this revision keeps frozen).
# ===========================================================================

tp <- read_csv(file.path(DIAG, "d4_geothermal_timeline_points.csv"), show_col_types = FALSE)

CE_MIN_N <- 5
# CE annual median SPLIT BY COHORT: each cohort keeps its own n>=5/yr filter and gap-broken runs.
ce_yr <- tp |>
  filter(row_type == "ce_year", n >= CE_MIN_N) |>
  mutate(cohort = factor(cohort, levels = c("BLM", "DOE/Other"))) |>
  arrange(cohort, yr) |>
  group_by(cohort) |>
  mutate(process_f = factor("CE", levels = PROCESS_LEVELS), val = median_months,
         # break the line across gap years, within each cohort
         run = cumsum(c(1, diff(yr) != 1))) |>
  ungroup()

ee <- tp |>
  filter(row_type == "ea_eis_project") |>
  mutate(process_f = factor(process_type, levels = PROCESS_LEVELS),
         cohort = factor(cohort, levels = c("BLM", "DOE/Other")), val = months)

arra_df <- tibble(process_f = factor("CE", levels = PROCESS_LEVELS), x = 2009)

# Per-cohort overall-median annotations, placed in the empty left half of the CE panel.
ce_med_lab <- tibble(
  process_f = factor("CE", levels = PROCESS_LEVELS),
  cohort = factor(c("BLM", "DOE/Other"), levels = c("BLM", "DOE/Other")),
  x = c(1996, 1996),
  y = c(19.5, 16),
  lab = c(paste0("BLM CEs: median ", blm_ce_med, " d"),
          paste0("DOE & other CEs: median ", nonblm_ce_med, " d")))

# One color scale keyed by string: cohort colors for the CE line, process colors for EA/EIS points.
CE_COHORT_COLORS <- c("BLM" = catf_navy, "DOE/Other" = catf_light_blue)

fig_c <- ggplot() +
  geom_vline(data = arra_df, aes(xintercept = x), linetype = "dashed",
             color = "gray55", linewidth = 0.5) +
  geom_text(data = arra_df, aes(x = x + 0.3, label = "ARRA (2009)"), y = Inf,
            hjust = 0, vjust = 1.5, size = 2.7, color = "gray40") +
  # CE annual median: one gap-broken line per cohort (BLM navy / DOE light blue)
  geom_line(data = ce_yr, aes(x = yr, y = val, group = interaction(cohort, run), color = cohort), linewidth = 0.9) +
  geom_point(data = ce_yr, aes(x = yr, y = val, color = cohort), size = 2.2) +
  geom_text(data = ce_med_lab, aes(x = x, y = y, label = lab, color = cohort),
            hjust = 0, vjust = 0.5, size = 3, fontface = "bold", show.legend = FALSE) +
  # every EA/EIS project as one point: color = process (facet), shape = cohort (unchanged)
  geom_point(data = ee, aes(x = yr, y = val, color = process_f, shape = cohort),
             size = 2.9, stroke = 1.0, fill = "white") +
  facet_wrap(~process_f, ncol = 1, scales = "free_y") +
  scale_color_manual(values = c(PROCESS_COLORS, CE_COHORT_COLORS),
                     breaks = c("BLM", "DOE/Other"), name = "CE cohort") +
  scale_shape_manual(values = c("BLM" = 16, "DOE/Other" = 17), name = "Cohort (EA / EIS)") +
  scale_x_continuous(breaks = seq(1996, 2024, 4), limits = c(1995, 2026)) +
  scale_y_continuous(expand = expansion(mult = c(0.06, 0.18))) +
  guides(shape = guide_legend(override.aes = list(size = 3, color = "gray30")),
         color = guide_legend(override.aes = list(shape = 15, size = 4, linetype = 0))) +
  coord_cartesian(clip = "off") +
  labs(
    title    = "Geothermal CE timelines split by cohort, EA/EIS by decision year",
    subtitle = paste0("CE annual medians now run as two series — BLM western-resource (navy) vs DOE & other grant CEs (light blue), ",
                      "years with n ≥ ", CE_MIN_N, ".\nThe ARRA grant surge (2010–12) is the DOE series. Every EA (",
                      uni("complete_timeline_total", "n_ea"), ") and EIS (", uni("complete_timeline_total", "n_eis"),
                      ") project is one point; shape marks the cohort. Free y-axis per panel."),
    x = "Decision year", y = "Duration (months)",
    caption = paste0("Linear y-axis per panel (CE runs in single months, EIS in tens). CE lines = per-cohort annual medians (n ≥ ",
                     CE_MIN_N, "), broken across gap years;\neach EA/EIS marker is one project (circle = BLM, triangle = DOE & other). ",
                     "Small samples; associational.")
  )

ggsave(file.path(FIGS, "fig_d4_geothermal_timeline.png"),
       fig_c, width = 10, height = 7, dpi = 300)
saveRDS(fig_c, file.path(FIGS, "fig_d4_geothermal_timeline.rds"))
message("Wrote fig_d4_geothermal_timeline.png")
