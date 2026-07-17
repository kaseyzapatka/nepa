# D4 duration-by-technology: two separate figures — one for decarbonization technologies,
# one for fossil-fuel technologies — using the cleaned technology tag (tech_group /
# energy_group from deliverable03/projects_nepa_reviews.parquet, the same variable that
# defines the Decarb-vs-Fossil split). Not faceted; Decarb rendered in lime, Fossil in
# dark blue, matching the energy colours used in 08_analyze.R Figure 6.
#
# Scope: EA + EIS only. CE reviews are uniformly ~1 month across every technology and add no
# contrast; the CE-heavy technologies (geothermal, nuclear, CCS, storage, biomass) therefore
# have too few substantive (EA/EIS) reviews to chart and are suppressed by the n >= 15 floor —
# itself a finding: those technologies' NEPA reviews are almost entirely categorical exclusions.
#
# Duration frame: identical to 08_analyze.R's headline frame (complete_clear + complete_with_proxy,
# year-granularity endpoints excluded, month-granularity imputed to the mid-month 15th, non-negative).
#
# Outputs:
#   phase2/output/deliverable04/figures/fig_d4_duration_by_decarb_tech.png
#   phase2/output/deliverable04/figures/fig_d4_duration_by_fossil_tech.png
#   phase2/output/deliverable04/diagnostics/d4_duration_by_technology.csv
#
# Usage: Rscript phase2/code/deliverable04/fra/04_duration_by_technology.R

suppressPackageStartupMessages({
  library(here); library(arrow); library(dplyr); library(tidyr)
  library(readr); library(stringr); library(lubridate); library(ggplot2); library(scales)
})

PHASE2 <- here::here("phase2")
TL     <- file.path(PHASE2, "data", "analysis", "timeline")
D03    <- file.path(PHASE2, "data", "analysis", "deliverable03")
FIG    <- file.path(PHASE2, "output", "deliverable04", "figures")
DIAG   <- file.path(PHASE2, "output", "deliverable04", "diagnostics")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
dir.create(DIAG, recursive = TRUE, showWarnings = FALSE)

catf_navy <- "#012169"; catf_dark_blue <- "#0047BB"; catf_lime <- "#93D500"
MIN_N <- 15
theme_catf <- function(base = 11) {
  theme_minimal(base_size = base) +
    theme(plot.title    = element_text(face = "bold", color = catf_navy, size = rel(1.2)),
          plot.subtitle = element_text(color = catf_dark_blue, size = rel(0.9)),
          plot.caption  = element_text(color = "gray50", size = rel(0.8), hjust = 0),
          panel.grid.minor = element_blank())
}

# --- headline duration frame (mirrors 08_analyze.R), EA + EIS, joined to the technology tag ---
dates <- read_parquet(
  file.path(TL, "timeline_project_dates.parquet"),
  col_select = c("project_id", "process_type", "timeline_status",
                 "initiation_date", "decision_date",
                 "initiation_date_granularity", "decision_date_granularity")
) |>
  mutate(initiation_date = as.Date(initiation_date),
         decision_date   = as.Date(decision_date))

tech <- read_parquet(
  file.path(D03, "projects_nepa_reviews.parquet"),
  col_select = c("project_id", "tech_group", "energy_group")
) |>
  filter(energy_group %in% c("Decarbonization", "Fossil Fuel"), !is.na(tech_group))

frame <- dates |>
  inner_join(tech, by = "project_id") |>
  filter(timeline_status %in% c("complete_clear", "complete_with_proxy"),
         process_type %in% c("EA", "EIS"),
         !is.na(initiation_date), !is.na(decision_date),
         initiation_date_granularity != "year", decision_date_granularity != "year") |>
  mutate(
    .init_mid = if_else(initiation_date_granularity == "month",
                        floor_date(initiation_date, "month") + 14, initiation_date),
    .dec_mid  = if_else(decision_date_granularity == "month",
                        floor_date(decision_date, "month") + 14, decision_date),
    duration_months = as.integer(.dec_mid - .init_mid) / 30.44
  ) |>
  filter(!is.na(duration_months), duration_months >= 0)

tech_summary <- frame |>
  group_by(energy_group, tech_group) |>
  summarise(
    n             = n(),
    p10           = quantile(duration_months, 0.10, na.rm = TRUE),
    p25           = quantile(duration_months, 0.25, na.rm = TRUE),
    median_months = median(duration_months, na.rm = TRUE),
    p75           = quantile(duration_months, 0.75, na.rm = TRUE),
    p90           = quantile(duration_months, 0.90, na.rm = TRUE),
    .groups = "drop"
  ) |>
  filter(n >= MIN_N)

write_csv(tech_summary, file.path(DIAG, "d4_duration_by_technology.csv"))
message("Wrote d4_duration_by_technology.csv (", nrow(tech_summary), " technology cells)")
print(as.data.frame(tech_summary[, c("energy_group", "tech_group", "n", "median_months")]))

make_tech_fig <- function(eg, colour, title, fname) {
  df <- tech_summary |>
    filter(energy_group == eg) |>
    mutate(
      tech_group = reorder(tech_group, median_months),
      lab = if_else(median_months < 12,
                    sprintf("~%.0f mo", median_months),
                    sprintf("~%.0f mo (%.1f yr)", median_months, median_months / 12))
    )
  g <- ggplot(df, aes(y = tech_group)) +
    geom_segment(aes(x = p10, xend = p90, yend = tech_group), linewidth = 1.8, alpha = 0.32, color = colour) +
    geom_segment(aes(x = p25, xend = p75, yend = tech_group), linewidth = 5.5, alpha = 0.55, color = colour) +
    geom_point(aes(x = median_months), size = 3.2, color = colour) +
    geom_text(aes(x = median_months, label = lab),
              nudge_y = 0.30, size = 3.0, fontface = "bold", color = "gray20") +
    geom_text(aes(x = p90, label = paste0("n=", comma(n))),
              nudge_x = 1.2, hjust = 0, size = 2.9, color = "gray40") +
    scale_x_continuous(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.02, 0.16))) +
    labs(
      title    = title,
      subtitle = "EA + EIS reviews. Thin bar = p10-p90, thick bar = IQR (p25-p75), point = median. Technologies with < 15 reviews suppressed.",
      x = "Duration (months)", y = NULL
    ) +
    theme_catf() + theme(legend.position = "none")
  ggsave(file.path(FIG, fname), g, width = 10, height = 5.5, dpi = 300)
  message("Wrote ", fname)
}

make_tech_fig("Decarbonization", catf_lime,      "Timeline Duration by Decarbonization Technology", "fig_d4_duration_by_decarb_tech.png")
make_tech_fig("Fossil Fuel",     catf_dark_blue,  "Timeline Duration by Fossil-Fuel Technology",      "fig_d4_duration_by_fossil_tech.png")
