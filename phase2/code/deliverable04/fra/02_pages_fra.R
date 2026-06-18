# D4 FRA analysis: Document Length Over Time & Pre/Post-FRA page limits
# (full Phase 2 corpus, ALL energy types)
#
# Reproduces the Phase 1 Deliverable 5 figures/tables, now on:
#   - Phase 2 regulatory page counts for ALL EA/EIS projects (fra/01_extract_pages.py ->
#     phase2/data/analysis/deliverable04/projects_page_counts.parquet), and
#   - Phase 2 LLM-adjudicated DECISION dates (inclusion requires only a decision date).
#
# Regulatory pages = body word count / 500, excluding embedded appendices and low-content
# pages (40 C.F.R. § 1508.1(bb)) — the measure comparable to the FRA limits (EA 75; EIS 150/300).
# FRA date = enactment (2023-06-03), matching Phase 1 D5.
#
# Usage: Rscript phase2/code/deliverable04/fra/02_pages_fra.R

suppressPackageStartupMessages({
  library(here); library(arrow); library(dplyr); library(tidyr)
  library(readr); library(lubridate); library(ggplot2); library(scales); library(zoo)
})

PHASE2 <- here::here("phase2")
TL     <- file.path(PHASE2, "data", "analysis", "timeline")
D04    <- file.path(PHASE2, "data", "analysis", "deliverable04")
FIG    <- file.path(PHASE2, "output", "deliverable04", "figures")
DIAG   <- file.path(PHASE2, "output", "deliverable04", "diagnostics")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
dir.create(DIAG, recursive = TRUE, showWarnings = FALSE)

FRA_DATE <- as.Date("2023-06-03")
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
fra_lab <- function(d) factor(if_else(d >= FRA_DATE, "Post-FRA", "Pre-FRA"),
                              levels = c("Pre-FRA", "Post-FRA"))

# ---------------------------------------------------------------------------
# Assemble: page counts (all EA/EIS) + decision dates + energy category
# ---------------------------------------------------------------------------
pc <- read_parquet(file.path(D04, "projects_page_counts.parquet")) |>
  filter(!is.na(regulatory_pages)) |>
  transmute(project_id, process_type = dataset_source,
            raw_pages = as.numeric(raw_pages), regulatory_pages = as.numeric(regulatory_pages))

dates <- read_parquet(file.path(TL, "timeline_project_dates.parquet"),
                      col_select = c("project_id", "decision_date")) |>
  mutate(decision_date = as.Date(decision_date))

energy <- read_parquet(file.path(TL, "timeline_document_index.parquet"),
                       col_select = c("project_id", "project_energy_type")) |>
  group_by(project_id) |> summarise(energy = first(na.omit(project_energy_type)), .groups = "drop") |>
  mutate(energy = factor(recode(coalesce(energy, "Other"), "Clean" = "Decarb"),
                         levels = c("Decarb", "Fossil", "Other")))

pages <- pc |>
  inner_join(dates, by = "project_id") |>
  filter(!is.na(decision_date)) |>
  left_join(energy, by = "project_id") |>
  mutate(fra_period = fra_lab(decision_date),
         decision_year = year(decision_date),
         decision_month = floor_date(decision_date, "month"),
         process_type = factor(process_type, levels = c("EA", "EIS")))

message(sprintf("FRA regulatory sample (all energy): %d | EA %d / EIS %d | Pre %d / Post %d",
                nrow(pages), sum(pages$process_type == "EA"), sum(pages$process_type == "EIS"),
                sum(pages$fra_period == "Pre-FRA"), sum(pages$fra_period == "Post-FRA")))

# ---------------------------------------------------------------------------
# Fig 1: document length over time (3-month rolling mean, FRA line)
# ---------------------------------------------------------------------------
pt <- pages |> filter(decision_year >= 2010, decision_year <= 2025)
monthly <- pt |> group_by(process_type, decision_month) |>
  summarise(mean_pages = mean(regulatory_pages), .groups = "drop") |>
  arrange(process_type, decision_month) |> group_by(process_type) |>
  mutate(roll3 = zoo::rollmean(mean_pages, 3, fill = NA, align = "right")) |> ungroup()
p_time <- ggplot() +
  geom_point(data = pt, aes(decision_date, regulatory_pages, color = fra_period), alpha = 0.28, size = 1) +
  geom_line(data = monthly, aes(decision_month, roll3), color = catf_navy, linewidth = 1.2, na.rm = TRUE) +
  geom_vline(xintercept = FRA_DATE, linetype = "dashed", color = "red", linewidth = 0.8) +
  annotate("text", x = FRA_DATE + 45, y = Inf, vjust = 1.5, hjust = 0, size = 3, color = "red",
           fontface = "italic", label = "FRA enacted\n(June 3, 2023)") +
  facet_wrap(~process_type, ncol = 1, scales = "free_y") +
  scale_x_date(date_labels = "%Y", date_breaks = "2 years") +
  scale_color_manual(values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)) +
  labs(title = "Document Length Over Time (All EA/EIS, all energy types)",
       subtitle = "Points = projects; navy line = 3-month rolling mean of monthly mean regulatory pages",
       x = "Decision Date", y = "Regulatory Pages (body word count / 500)", color = NULL,
       caption = "Decision-date inclusion. Regulatory pages exclude embedded appendices and low-content pages per 40 C.F.R. § 1508.1(bb).") +
  theme_catf()
ggsave(file.path(FIG, "fig_d4_pages_over_time.png"), p_time, width = 12, height = 8, dpi = 300)

# ---------------------------------------------------------------------------
# Fig 2: pre/post FRA bar (by process)
# ---------------------------------------------------------------------------
fra_sum <- pages |> group_by(process_type, fra_period) |>
  summarise(mean_pages = mean(regulatory_pages), median_pages = median(regulatory_pages),
            sd_pages = sd(regulatory_pages), n = n(), .groups = "drop")
p_bar <- ggplot(fra_sum, aes(fra_period, mean_pages, fill = fra_period)) +
  geom_col(alpha = 0.85, width = 0.6) +
  geom_text(aes(label = sprintf("mean %.0f\n(n=%s)", mean_pages, comma(n))), vjust = -0.2, size = 3.2, color = "gray20") +
  geom_point(aes(y = median_pages), shape = 18, size = 4, color = catf_navy) +
  geom_text(aes(y = median_pages, label = sprintf("median %.0f", median_pages)), vjust = 1.8, size = 2.8, color = "white") +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.3))) +
  labs(title = "Document Length: Pre vs Post FRA",
       subtitle = "Regulatory pages; bar = mean, diamond = median; classified by decision date (>= 2023-06-03 = Post-FRA)",
       x = NULL, y = "Regulatory Pages (body word count / 500)", fill = NULL) +
  theme_catf() + theme(legend.position = "none")
ggsave(file.path(FIG, "fig_d4_pages_pre_post_fra.png"), p_bar, width = 10, height = 6, dpi = 300)

# ---------------------------------------------------------------------------
# Fig 3: pre/post FRA by ENERGY category
# ---------------------------------------------------------------------------
energy_sum <- pages |> group_by(process_type, energy, fra_period) |>
  summarise(mean_pages = round(mean(regulatory_pages)), median_pages = round(median(regulatory_pages)),
            n = n(), .groups = "drop")
p_energy <- ggplot(energy_sum, aes(energy, mean_pages, fill = fra_period)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65, alpha = 0.9) +
  geom_text(aes(label = comma(n)), position = position_dodge(width = 0.75), vjust = -0.3, size = 2.7, color = "gray30") +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  labs(title = "Document Length by Energy Category: Pre vs Post FRA",
       subtitle = "Mean regulatory pages; labels = n projects; all EA/EIS with a decision date",
       x = NULL, y = "Regulatory Pages (body word count / 500)", fill = NULL) +
  theme_catf()
ggsave(file.path(FIG, "fig_d4_pages_pre_post_fra_by_energy.png"), p_energy, width = 11, height = 6, dpi = 300)

# ---------------------------------------------------------------------------
# Fig 4: distribution (violin + box, y capped at p99)
# ---------------------------------------------------------------------------
p99 <- quantile(pages$regulatory_pages, 0.99, na.rm = TRUE)
dlab <- pages |> group_by(process_type, fra_period) |>
  summarise(n = n(), med = round(median(regulatory_pages)), .groups = "drop")
p_dist <- ggplot(pages, aes(fra_period, regulatory_pages, fill = fra_period)) +
  geom_violin(alpha = 0.35, trim = FALSE, color = NA) +
  geom_boxplot(width = 0.2, outlier.alpha = 0.2, outlier.size = 0.7, fill = "white", color = catf_navy, linewidth = 0.4) +
  stat_summary(fun = median, geom = "point", shape = 18, size = 3.2, color = catf_navy) +
  geom_text(data = dlab, aes(x = fra_period, y = med, label = paste0("median ", med)), inherit.aes = FALSE,
            hjust = -0.18, size = 2.7, color = catf_navy, fontface = "bold") +
  geom_text(data = dlab, aes(x = fra_period, y = 0, label = paste0("n=", comma(n))), inherit.aes = FALSE,
            vjust = 1.5, size = 3, color = "gray40") +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)) +
  coord_cartesian(ylim = c(0, p99)) +
  labs(title = "Document Length Distribution: Pre vs Post FRA",
       subtitle = "Regulatory pages; violin + boxplot; diamond = median; y-axis capped at p99", x = NULL,
       y = "Regulatory Pages (body word count / 500)", fill = NULL) +
  theme_catf() + theme(legend.position = "none")
ggsave(file.path(FIG, "fig_d4_pages_distribution.png"), p_dist, width = 10, height = 6, dpi = 300)

# Fig 4b: distribution BY ENERGY (process rows x energy cols)
p_dist_e <- ggplot(pages, aes(fra_period, regulatory_pages, fill = fra_period)) +
  geom_violin(alpha = 0.35, trim = FALSE, color = NA) +
  geom_boxplot(width = 0.2, outlier.alpha = 0.15, outlier.size = 0.6, fill = "white", color = catf_navy, linewidth = 0.35) +
  stat_summary(fun = median, geom = "point", shape = 18, size = 2.4, color = catf_navy) +
  facet_grid(process_type ~ energy, scales = "free_y") +
  scale_fill_manual(values = c("Pre-FRA" = catf_light_blue, "Post-FRA" = catf_dark_blue)) +
  coord_cartesian(ylim = c(0, p99)) +
  labs(title = "Document Length Distribution by Energy Category: Pre vs Post FRA",
       subtitle = "Regulatory pages; violin + boxplot; diamond = median; y-axis capped at p99",
       x = NULL, y = "Regulatory Pages (body word count / 500)", fill = NULL) +
  theme_catf() + theme(legend.position = "none")
ggsave(file.path(FIG, "fig_d4_pages_distribution_by_energy.png"), p_dist_e, width = 11, height = 7, dpi = 300)

# ---------------------------------------------------------------------------
# Fig 5: FRA page-limit compliance (Post-FRA only)
# ---------------------------------------------------------------------------
post <- pages |> filter(fra_period == "Post-FRA") |>
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

# EIS extraordinary-complexity bracket (Phase 1 D5 style): the share within the 300-page
# threshold = Compliant + "Exceeds standard limit". Stack order (Compliant on top, Exceeds
# limit at the bottom) means that span runs from the top of the Exceeds-limit segment to the top.
eis_get <- function(lvl) { v <- comp_sum$n[comp_sum$process_type == "EIS" & as.character(comp_sum$compliance) == lvl]; if (length(v)) sum(v) else 0 }
n_lim_eis <- eis_get("Exceeds limit"); n_std_eis <- eis_get("Exceeds standard limit"); n_ok_eis <- eis_get("Compliant")
n_eis_tot <- n_lim_eis + n_std_eis + n_ok_eis
pct_w300  <- if (n_eis_tot > 0) round(100 * (n_ok_eis + n_std_eis) / n_eis_tot) else 0
y_bot <- n_lim_eis; y_top <- n_eis_tot; tick_h <- max(n_eis_tot * 0.02, 0.4)
xt <- 2.32; xv <- 2.48; xl <- 2.54

p_comp <- ggplot(comp_sum, aes(process_type, n, fill = compliance)) +
  geom_col(width = 0.6, alpha = 0.9) +
  geom_text(aes(label = label), position = position_stack(vjust = 0.5), size = 3.2, color = "white", fontface = "bold") +
  annotate("segment", x = xt, xend = xv, y = y_top - tick_h, yend = y_top - tick_h, color = "black", linewidth = 0.5) +
  annotate("segment", x = xv, xend = xv, y = y_bot + tick_h, yend = y_top - tick_h, color = "black", linewidth = 0.5) +
  annotate("segment", x = xt, xend = xv, y = y_bot + tick_h, yend = y_bot + tick_h, color = "black", linewidth = 0.5) +
  annotate("text", x = xl, y = (y_bot + y_top) / 2, hjust = 0, vjust = 0.5, size = 2.6, color = "black",
           lineheight = 0.9, label = paste0("Total ", pct_w300, "%\ncompliant by the\n300-page\nextraordinary-\ncomplexity\nthreshold")) +
  scale_fill_manual(values = c("Compliant" = catf_teal, "Exceeds standard limit" = "#E8A317", "Exceeds limit" = catf_magenta)) +
  scale_x_discrete(expand = expansion(add = c(0.5, 1.1))) +
  coord_cartesian(clip = "off") +
  labs(title = "FRA Page-Limit Compliance: Post-FRA Projects",
       subtitle = "Regulatory pages. EA limit 75; EIS 150 (300 if extraordinarily complex).",
       x = NULL, y = "Number of Projects", fill = NULL) +
  theme_catf() + theme(legend.position = "bottom", plot.margin = margin(5, 45, 5, 5))
ggsave(file.path(FIG, "fig_d4_pages_compliance.png"), p_comp, width = 10, height = 6, dpi = 300)

# ---------------------------------------------------------------------------
# Fig 6: regulatory vs raw pages (how much appendices/sparse pages inflate raw counts)
# ---------------------------------------------------------------------------
rr <- pages |> select(process_type, fra_period, regulatory_pages, raw_pages) |>
  pivot_longer(c(regulatory_pages, raw_pages), names_to = "measure", values_to = "pages") |>
  mutate(measure = factor(measure, levels = c("raw_pages", "regulatory_pages"),
                          labels = c("Raw PDF pages", "Regulatory pages (word count / 500)"))) |>
  group_by(process_type, measure) |> summarise(mean_pages = mean(pages), .groups = "drop")
p_rr <- ggplot(rr, aes(measure, mean_pages, fill = measure)) +
  geom_col(width = 0.6, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.0f", mean_pages)), vjust = -0.3, size = 3.3, color = "gray20") +
  facet_wrap(~process_type, scales = "free_y") +
  scale_fill_manual(values = c("Raw PDF pages" = catf_light_blue, "Regulatory pages (word count / 500)" = catf_dark_blue)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(title = "Raw vs Regulatory Pages (All EA/EIS)",
       subtitle = "Mean pages. The gap = appendices + sparse pages that count as raw PDF pages but not toward the FRA limit.",
       x = NULL, y = "Mean Pages", fill = NULL) +
  theme_catf() + theme(legend.position = "none")
ggsave(file.path(FIG, "fig_d4_pages_reg_vs_raw.png"), p_rr, width = 10, height = 6, dpi = 300)

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
summ_tbl <- pages |> group_by(process_type, fra_period) |>
  summarise(n_projects = n(), mean_pages = round(mean(regulatory_pages)),
            median_pages = round(median(regulatory_pages)), sd_pages = round(sd(regulatory_pages)),
            p25_pages = round(quantile(regulatory_pages, .25)), p75_pages = round(quantile(regulatory_pages, .75)),
            .groups = "drop")
write_csv(summ_tbl, file.path(DIAG, "d4_pages_summary.csv"))
write_csv(energy_sum, file.path(DIAG, "d4_pages_summary_by_energy.csv"))
write_csv(comp_sum, file.path(DIAG, "d4_pages_compliance.csv"))

message("02_pages_fra.R complete. Figures -> output/deliverable04/figures/ ; tables -> diagnostics/")
print(summ_tbl)
