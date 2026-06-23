# D6 / n08 — analysis figures for the report (final step in the chain)
#
# Reads the D6 analysis artifacts (n04 base rates, n05 mitigation, n06 CE
# landscape, n07 verdicts) and builds the report figures. Mirrors the house
# pattern (cf. D5 03_analyze_spikes.R): Python builds the data, this final
# numbered R script builds the figures, and reports/deliverable06.qmd embeds them.
#
# Inputs:  phase2/data/analysis/deliverable06/{candidate_verdicts, candidate_mitigation_summary,
#                                              candidate_base_rates}.parquet
#          phase2/output/deliverable06/ce_landscape_summary.csv
# Outputs: phase2/output/deliverable06/figures/*.png
#
# Usage: Rscript phase2/code/deliverable06/n08_analyze.R

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(stringr)
  library(arrow); library(ggplot2); library(scales)
})

PHASE2 <- here::here("phase2")
ANALYSIS <- file.path(PHASE2, "data", "analysis", "deliverable06")
OUT  <- file.path(PHASE2, "output", "deliverable06")
FIGS <- file.path(OUT, "figures")
dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)

# CATF palette / theme (matches D4/D5)
catf_navy <- "#012169"; catf_dark_blue <- "#0047BB"; catf_lime <- "#93D500"
catf_teal <- "#00AE8D"; catf_magenta <- "#C22A90"; catf_light_blue <- "#8AB7E9"
VERDICT_COLORS <- c(new = catf_lime, expand = catf_dark_blue, adopt = catf_teal,
                    already_covered = "gray70", contrast = catf_magenta)

theme_catf <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(plot.title = element_text(face = "bold", color = catf_navy, margin = margin(b = 8)),
          plot.subtitle = element_text(color = catf_dark_blue, margin = margin(b = 8)),
          plot.caption = element_text(size = rel(0.8), color = "gray50", hjust = 0),
          axis.title = element_text(color = catf_navy),
          legend.position = "bottom",
          panel.grid.minor = element_blank(),
          plot.background = element_rect(fill = "white", color = NA))
}

save_fig <- function(p, name, w = 8, h = 5) {
  ggsave(file.path(FIGS, name), p, width = w, height = h, dpi = 300)
  message("  wrote ", name)
}

short_label <- function(x) x %>% str_replace(" \\(.*\\)", "") %>% str_wrap(22)

# ---------------------------------------------------------------------------
verdicts <- read_parquet(file.path(ANALYSIS, "candidate_verdicts.parquet"))
mit      <- read_parquet(file.path(ANALYSIS, "candidate_mitigation_summary.parquet"))

# Fig 1 — candidate verdicts (new / expand / adopt / contrast)
p1 <- verdicts %>%
  mutate(lab = short_label(candidate_label),
         verdict = factor(verdict, levels = names(VERDICT_COLORS))) %>%
  ggplot(aes(reorder(lab, rank_score), n_observed_fonsi, fill = verdict)) +
  geom_col() +
  geom_text(aes(label = n_observed_fonsi), hjust = -0.2, size = 3, color = catf_navy) +
  coord_flip() +
  scale_fill_manual(values = VERDICT_COLORS, drop = FALSE, name = "CE verdict") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "D6 candidate categories by CE verdict",
       subtitle = "Observed clean-energy EA-source FONSI projects; verdict vs. the existing-CE catalog (ce.json)",
       x = NULL, y = "Observed FONSI projects",
       caption = "Deterministic first pass; verdicts pending LLM verification. NEW is empty — all current candidates already map to a CE.") +
  theme_catf()
save_fig(p1, "fig_d6_verdicts.png")

# Fig 2 — CE-shaped (profile) evidence volume
p2 <- verdicts %>%
  filter(verdict != "contrast") %>%
  mutate(lab = short_label(candidate_label)) %>%
  ggplot(aes(reorder(lab, n_profile_fonsi), n_profile_fonsi, fill = verdict)) +
  geom_col() + geom_text(aes(label = n_profile_fonsi), hjust = -0.3, size = 3, color = catf_navy) +
  coord_flip() +
  scale_fill_manual(values = VERDICT_COLORS, drop = FALSE, name = "CE verdict") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "CE-shaped evidence per candidate",
       subtitle = "Profile-subtype FONSI projects (the bounded, low-impact slice)",
       x = NULL, y = "CE-shaped FONSI projects") +
  theme_catf()
save_fig(p2, "fig_d6_evidence_volume.png")

# Fig 3 — mitigated-FONSI share (Track B)
p3 <- mit %>%
  mutate(lab = short_label(candidate_category)) %>%
  ggplot(aes(reorder(lab, mitigated_share), mitigated_share)) +
  geom_col(fill = catf_dark_blue) +
  geom_text(aes(label = percent(mitigated_share, accuracy = 1)), hjust = -0.2, size = 3, color = catf_navy) +
  coord_flip() +
  scale_y_continuous(labels = percent, expand = expansion(mult = c(0, 0.15)), limits = c(0, 1)) +
  labs(title = "Mitigated-FONSI share by candidate (Track B)",
       subtitle = "Share whose no-significant-impact finding leans on committed mitigation (dual-signal)",
       x = NULL, y = "Mitigated-FONSI share",
       caption = "Preliminary; consistent mitigations can become codifiable CE design criteria.") +
  theme_catf()
save_fig(p3, "fig_d6_mitigated_share.png")

# Fig 4 — existing-CE landscape: CEs per agency (top 15)
land <- file.path(OUT, "ce_landscape_summary.csv")
if (file.exists(land)) {
  p4 <- read_csv(land, show_col_types = FALSE) %>%
    slice_max(n_ces, n = 15) %>%
    ggplot(aes(reorder(agency_unit, n_ces), n_ces)) +
    geom_col(fill = catf_teal) +
    geom_text(aes(label = n_ces), hjust = -0.3, size = 3, color = catf_navy) +
    coord_flip() +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(title = "Existing CE landscape — CEs per agency (top 15)",
         subtitle = "From the CE Explorer catalog (ce.json); context for adopt/consolidate",
         x = NULL, y = "Number of categorical exclusions") +
    theme_catf()
  save_fig(p4, "fig_d6_ce_per_agency.png")
}

message("[n08] figures written to ", FIGS)
