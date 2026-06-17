# D4 (report add-on): Document Length Over Time & Pre/Post-FRA page analysis
#
# Reproduces the Phase 1 Deliverable 5 "Document Length Trends Over Time" and
# "Pre vs Post FRA" figures/tables, refreshed on Phase 2 LLM-adjudicated DECISION dates.
# Per the deliverable request, inclusion requires only a DECISION date (not a complete
# timeline) — the FRA period and time axis depend only on the decision date.
#
# Two universes:
#   (A) DECARB regulatory pages (faithful D5 method) — regulatory_pages = body word count
#       / 500 (40 C.F.R. § 1508.1(bb)), available only for clean-energy EA/EIS (Phase 1
#       extract_pages.py output, phase1/data/analysis/projects_page_counts.parquet).
#   (B) FULL-DATASET raw PDF pages (the requested expansion) — all EA/EIS regardless of
#       energy type, using the document index's raw page counts. Raw pages OVERSTATE
#       regulatory length (they include appendices + sparse pages); the regulatory-page
#       expansion to fossil/other requires running extract_pages.py on the full corpus
#       (tracked as a to-do).
#
# FRA date here = ENACTMENT (2023-06-03), matching Phase 1 D5 (the page-limit analysis),
# NOT the CEQ-rule effective date (2023-08-16) used for the timeline FRA-period split.
#
# Usage: Rscript phase2/code/deliverable04/11_pages_fra.R

suppressPackageStartupMessages({
  library(here); library(arrow); library(dplyr); library(tidyr)
  library(readr); library(lubridate); library(ggplot2); library(scales); library(zoo)
})

PHASE2 <- here::here("phase2")
TL     <- file.path(PHASE2, "data", "analysis", "timeline")
FIG    <- file.path(PHASE2, "output", "deliverable04", "figures")
DIAG   <- file.path(PHASE2, "output", "deliverable04", "diagnostics")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
dir.create(DIAG, recursive = TRUE, showWarnings = FALSE)

FRA_DATE <- as.Date("2023-06-03")

# CATF brand colors (Phase 1 D5 setup) ------------------------------------------------
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

# ---------------------------------------------------------------------------
# Load Phase 2 decision dates + per-project metadata
# ---------------------------------------------------------------------------
dates <- read_parquet(file.path(TL, "timeline_project_dates.parquet"),
                      col_select = c("project_id", "process_type", "decision_date")) |>
  filter(process_type %in% c("EA", "EIS"), !is.na(decision_date)) |>
  mutate(decision_date = as.Date(decision_date))

meta <- read_parquet(file.path(TL, "timeline_document_index.parquet"),
                     col_select = c("project_id", "project_energy_type",
                                    "project_title", "doc_page_count")) |>
  group_by(project_id) |>
  summarise(project_energy_type = first(na.omit(project_energy_type)),
            project_title       = first(na.omit(project_title)),
            raw_pages           = suppressWarnings(max(as.numeric(doc_page_count), na.rm = TRUE)),
            .groups = "drop") |>
  mutate(raw_pages = ifelse(is.finite(raw_pages), raw_pages, NA_real_))

# Phase 1 regulatory pages (decarb only)
pc <- read_parquet(here("phase1", "data", "analysis", "projects_page_counts.parquet"),
                   col_select = c("project_id", "regulatory_pages", "body_pages"))

fra_lab <- function(d) factor(if_else(d >= FRA_DATE, "Post-FRA", "Pre-FRA"),
                              levels = c("Pre-FRA", "Post-FRA"))

# ===================================================================================
# (A) DECARB REGULATORY PAGES — faithful D5 reproduction, decision-date-only
# ===================================================================================
decarb <- dates |>
  inner_join(meta, by = "project_id") |>
  filter(project_energy_type == "Clean") |>
  inner_join(pc, by = "project_id") |>
  filter(!is.na(regulatory_pages)) |>
  mutate(fra_period = fra_lab(decision_date),
         decision_month = floor_date(decision_date, "month"),
         decision_year = year(decision_date))

message(sprintf("(A) decarb regulatory sample: %d (EA %d / EIS %d) | Pre %d / Post %d",
                nrow(decarb), sum(decarb$process_type == "EA"), sum(decarb$process_type == "EIS"),
                sum(decarb$fra_period == "Pre-FRA"), sum(decarb$fra_period == "Post-FRA")))

# --- Fig A1: pages over time (points + 3-month rolling mean, FRA line) ---
pt <- decarb |> filter(decision_year >= 2010, decision_year <= 2025)
monthly <- pt |>
  group_by(process_type, decision_month) |>
  summarise(mean_pages = mean(regulatory_pages), .groups = "drop") |>
  arrange(process_type, decision_month) |>
  group_by(process_type) |>
  mutate(roll3 = zoo::rollmean(mean_pages, 3, fill = NA, align = "right")) |>
  ungroup()

p_time <- ggplot() +
  geom_point(data = pt, aes(decision_date, regulatory_pages, color = fra_period),
             alpha = 0.32, size = 1.2) +
  geom_line(data = monthly, aes(decision_month, roll3), color = catf_navy,
            linewidth = 1.2, na.rm = TRUE) +
  geom_vline(xintercept = FRA_DATE, linetype = "dashed", color = "red", linewidth = 0.8) +
  annotate("text", x = FRA_DATE + 45, y = Inf, vjust = 1.5, hjust = 0, size = 3,
           color = "red", fontface = "italic",
           label = "Fiscal Responsibility Act\nof 2023 (June 3, 2023)") +
  facet_wrap(~process_type, ncol = 1, scales = "free_y") +
  scale_x_date(date_labels = "%Y", date_breaks = "2 years") +
  scale_color_manual(values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)) +
  labs(title = "Document Length Over Time (Decarbonization EA/EIS)",
       subtitle = "Points = individual projects; navy line = 3-month rolling mean of monthly mean regulatory pages",
       x = "Decision Date", y = "Regulatory Pages (body word count ÷ 500)", color = NULL,
       caption = "Decision-date inclusion only. Regulatory pages exclude embedded appendices and low-content pages per 40 C.F.R. § 1508.1(bb).") +
  theme_catf()
ggsave(file.path(FIG, "fig_d4_pages_over_time.png"), p_time, width = 12, height = 8, dpi = 300)

# --- Fig A2: pre/post FRA bar (mean + median) ---
fra_sum <- decarb |>
  group_by(process_type, fra_period) |>
  summarise(mean_pages = mean(regulatory_pages), median_pages = median(regulatory_pages),
            sd_pages = sd(regulatory_pages), n = n(), .groups = "drop") |>
  mutate(bar_label = sprintf("mean %.0f\n(n=%s)", mean_pages, comma(n)),
         median_label = sprintf("median %.0f", median_pages))

p_bar <- ggplot(fra_sum, aes(fra_period, mean_pages, fill = fra_period)) +
  geom_col(alpha = 0.85, width = 0.6) +
  geom_text(aes(label = bar_label), vjust = -0.2, size = 3.2, color = "gray20") +
  geom_point(aes(y = median_pages), shape = 18, size = 4, color = catf_navy) +
  geom_text(aes(y = median_pages, label = median_label), vjust = 1.8, size = 2.8, color = "white") +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.3))) +
  labs(title = "Document Length: Pre vs Post Fiscal Responsibility Act (Decarbonization)",
       subtitle = "Bar = mean regulatory pages; diamond = median; classified by decision date (>= 2023-06-03 = Post-FRA)",
       x = NULL, y = "Regulatory Pages (body word count ÷ 500)", fill = NULL) +
  theme_catf() + theme(legend.position = "none")
ggsave(file.path(FIG, "fig_d4_pages_pre_post_fra.png"), p_bar, width = 10, height = 6, dpi = 300)

# --- Fig A3: distribution violin/box (y capped at p99) ---
p99 <- quantile(decarb$regulatory_pages, 0.99, na.rm = TRUE)
dlab <- decarb |> group_by(process_type, fra_period) |>
  summarise(n = n(), median_pages = median(regulatory_pages), .groups = "drop")
p_dist <- ggplot(decarb, aes(fra_period, regulatory_pages, fill = fra_period)) +
  geom_violin(alpha = 0.35, trim = FALSE, color = NA) +
  geom_boxplot(width = 0.2, outlier.alpha = 0.25, outlier.size = 0.8, fill = "white",
               color = catf_navy, linewidth = 0.4) +
  geom_text(data = dlab, aes(x = fra_period, y = 0, label = paste0("n=", comma(n))),
            inherit.aes = FALSE, vjust = 1.5, size = 3, color = "gray40") +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)) +
  coord_cartesian(ylim = c(0, p99)) +
  labs(title = "Document Length Distribution: Pre vs Post FRA (Decarbonization)",
       subtitle = "Violin + boxplot; y-axis capped at p99", x = NULL,
       y = "Regulatory Pages (body word count ÷ 500)", fill = NULL) +
  theme_catf() + theme(legend.position = "none")
ggsave(file.path(FIG, "fig_d4_pages_distribution.png"), p_dist, width = 10, height = 6, dpi = 300)

# --- Fig A4: FRA page-limit compliance (Post-FRA only) ---
post <- decarb |> filter(fra_period == "Post-FRA") |>
  mutate(compliance = case_when(
    process_type == "EA"  & regulatory_pages <= 75  ~ "Compliant",
    process_type == "EA"  & regulatory_pages >  75  ~ "Exceeds limit",
    process_type == "EIS" & regulatory_pages <= 150 ~ "Compliant",
    process_type == "EIS" & regulatory_pages <= 300 ~ "Exceeds standard limit",
    TRUE ~ "Exceeds limit"))
comp_levels <- c("Compliant", "Exceeds standard limit", "Exceeds limit")
comp_sum <- post |> mutate(compliance = factor(compliance, levels = comp_levels)) |>
  count(process_type, compliance, .drop = FALSE) |>
  group_by(process_type) |> mutate(total = sum(n), pct = n / total * 100,
                                    label = ifelse(n > 0, sprintf("%d\n(%.0f%%)", n, pct), "")) |> ungroup()
p_comp <- ggplot(comp_sum, aes(process_type, n, fill = compliance)) +
  geom_col(width = 0.6, alpha = 0.9) +
  geom_text(aes(label = label), position = position_stack(vjust = 0.5), size = 3.2,
            color = "white", fontface = "bold") +
  scale_fill_manual(values = c("Compliant" = catf_teal,
                               "Exceeds standard limit" = "#E8A317",
                               "Exceeds limit" = catf_magenta)) +
  labs(title = "FRA Page-Limit Compliance: Post-FRA Decarbonization Projects",
       subtitle = "EA limit 75 pages; EIS 150 (300 if extraordinarily complex). Regulatory pages.",
       x = NULL, y = "Number of Projects", fill = NULL) +
  theme_catf() + theme(legend.position = "bottom")
ggsave(file.path(FIG, "fig_d4_pages_compliance.png"), p_comp, width = 10, height = 6, dpi = 300)

# --- Summary table (decarb regulatory) ---
summary_tbl <- decarb |>
  group_by(process_type, fra_period) |>
  summarise(n_projects = n(), mean_pages = round(mean(regulatory_pages)),
            median_pages = round(median(regulatory_pages)), sd_pages = round(sd(regulatory_pages)),
            p25_pages = round(quantile(regulatory_pages, .25)),
            p75_pages = round(quantile(regulatory_pages, .75)), .groups = "drop")
write_csv(summary_tbl, file.path(DIAG, "d4_pages_summary_decarb.csv"))
write_csv(comp_sum, file.path(DIAG, "d4_pages_compliance_decarb.csv"))

# ===================================================================================
# (B) FULL-DATASET RAW PAGES — the requested expansion beyond decarb (caveated)
# ===================================================================================
full <- dates |> inner_join(meta, by = "project_id") |>
  filter(!is.na(raw_pages), raw_pages > 0) |>
  mutate(fra_period = fra_lab(decision_date),
         energy = recode(coalesce(project_energy_type, "Other"), "Clean" = "Decarb"))

message(sprintf("(B) full-dataset raw-pages sample: %d (EA %d / EIS %d)",
                nrow(full), sum(full$process_type == "EA"), sum(full$process_type == "EIS")))

full_sum <- full |> group_by(process_type, fra_period) |>
  summarise(mean_pages = mean(raw_pages), median_pages = median(raw_pages), n = n(), .groups = "drop")
p_full <- ggplot(full_sum, aes(fra_period, mean_pages, fill = fra_period)) +
  geom_col(alpha = 0.85, width = 0.6) +
  geom_text(aes(label = sprintf("mean %.0f\n(n=%s)", mean_pages, comma(n))),
            vjust = -0.2, size = 3.2, color = "gray20") +
  geom_point(aes(y = median_pages), shape = 18, size = 4, color = catf_navy) +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.3))) +
  labs(title = "Raw Document Length: Pre vs Post FRA (ENTIRE dataset, all energy types)",
       subtitle = "RAW PDF pages (overstate regulatory length); diamond = median; classified by decision date",
       x = NULL, y = "Raw PDF Pages (max document per project)", fill = NULL,
       caption = "Caveat: raw pages include appendices and sparse pages; not comparable to the 75/150-page FRA regulatory limits. Regulatory-page expansion to fossil/other requires running extract_pages.py on the full corpus (to-do).") +
  theme_catf() + theme(legend.position = "none")
ggsave(file.path(FIG, "fig_d4_pages_pre_post_fra_full_raw.png"), p_full, width = 10, height = 6, dpi = 300)
write_csv(full_sum, file.path(DIAG, "d4_pages_summary_full_raw.csv"))

message("11_pages_fra.R complete. Figures -> output/deliverable04/figures/ ; tables -> diagnostics/")
print(summary_tbl)
