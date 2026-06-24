# D6 / 08 — analysis figures for the report (final step in the chain)
#
# Reads the D6 analysis artifacts (01 corpus, 06 CE landscape/clusters, 07
# verdicts) and builds the report figures. Mirrors the house pattern (cf. D5
# 03_analyze_spikes.R): Python builds the data, this final numbered R script
# builds the figures, and reports/deliverable06.qmd embeds them.
#
# Figures (deliberately few — each makes one point):
#   fig_d6_funnel.png       the narrowing: clean FONSIs -> candidate -> CE-shaped -> net-new
#   fig_d6_adoption_gap.png per adopt candidate, how many agencies lack the existing CE
#
# Inputs:  phase2/data/analysis/deliverable06/{candidate_corpus, candidate_verdicts,
#                                              fonsi_project_inventory}.parquet
# Outputs: phase2/output/deliverable06/figures/*.png
#
# Usage: Rscript phase2/code/deliverable06/08_analyze.R

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(stringr)
  library(arrow); library(ggplot2); library(scales); library(forcats)
})

PHASE2 <- here::here("phase2")
ANALYSIS <- file.path(PHASE2, "data", "analysis", "deliverable06")
OUT  <- file.path(PHASE2, "output", "deliverable06")
FIGS <- file.path(OUT, "figures")
dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)

# CATF palette / theme (matches D4/D5)
catf_navy <- "#012169"; catf_dark_blue <- "#0047BB"; catf_lime <- "#93D500"
catf_teal <- "#00AE8D"; catf_magenta <- "#C22A90"; catf_light_blue <- "#8AB7E9"

theme_catf <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(plot.title = element_text(face = "bold", color = catf_navy, margin = margin(b = 6)),
          plot.subtitle = element_text(color = catf_dark_blue, margin = margin(b = 10)),
          plot.caption = element_text(size = rel(0.8), color = "gray50", hjust = 0),
          axis.title = element_text(color = catf_navy),
          legend.position = "none",
          panel.grid.minor = element_blank(),
          plot.background = element_rect(fill = "white", color = NA))
}

save_fig <- function(p, name, w = 8, h = 4.5) {
  ggsave(file.path(FIGS, name), p, width = w, height = h, dpi = 300)
  message("  wrote ", name)
}

short_label <- function(x) x %>% str_replace(" \\(.*\\)", "") %>% str_wrap(26)

# ---------------------------------------------------------------------------
inv      <- read_parquet(file.path(ANALYSIS, "fonsi_project_inventory.parquet"))
corp     <- read_parquet(file.path(ANALYSIS, "candidate_corpus.parquet"))
verdicts <- read_parquet(file.path(ANALYSIS, "candidate_verdicts.parquet"))

corp_fonsi   <- corp %>% filter(is_fonsi)
n_clean      <- inv %>% filter(project_energy_type == "Clean") %>% distinct(project_id) %>% nrow()
n_candidate  <- corp_fonsi %>% distinct(project_id) %>% nrow()
n_ce_shaped  <- corp_fonsi %>% filter(is_profile_subtype) %>% distinct(project_id) %>% nrow()
n_new        <- sum(verdicts$verdict == "new")

# Fig 1 — the funnel (the headline narrowing)
funnel <- tibble(
  stage = c("Clean-energy EA → FONSI projects",
            "In a candidate action type",
            "CE-shaped (bounded, low-impact)",
            "Net-new CE candidates"),
  n = c(n_clean, n_candidate, n_ce_shaped, n_new),
  fill = c(catf_light_blue, catf_light_blue, catf_teal, catf_lime)
) %>% mutate(stage = fct_inorder(stage) %>% fct_rev())

p1 <- ggplot(funnel, aes(stage, n, fill = fill)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = comma(n)), hjust = -0.2, size = 4.2, fontface = "bold", color = catf_navy) +
  scale_fill_identity() +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = "From every clean-energy FONSI down to CE candidates",
       subtitle = "All CE-shaped candidates already map to an existing CE — 0 are net-new",
       x = NULL, y = "Distinct FONSI projects",
       caption = "Clean-energy EA-source FONSI corpus. 'CE-shaped' = the bounded, low-impact slice of each action type.") +
  theme_catf()
save_fig(p1, "fig_d6_funnel.png")

# Fig 2 — the adoption gap (the actionable product)
adopt <- verdicts %>%
  filter(verdict == "adopt") %>%
  mutate(n_lacking = str_count(adopt_targets, ",") + 1L,
         lab = short_label(candidate_label))

p2 <- ggplot(adopt, aes(reorder(lab, n_lacking), n_lacking)) +
  geom_col(width = 0.66, fill = catf_teal) +
  geom_text(aes(label = adopt_targets), hjust = -0.05, size = 3.4, color = catf_navy) +
  coord_flip() +
  scale_y_continuous(breaks = scales::breaks_width(1),
                     expand = expansion(mult = c(0, 0.55))) +
  labs(title = "The adoption gap",
       subtitle = "Agencies running this action through a full EA→FONSI that lack an existing CE",
       x = NULL, y = "Number of agencies that could adopt an existing CE",
       caption = "Each action already has a CE at another agency (see report table). Agencies at bar end are the adopt targets.") +
  theme_catf()
save_fig(p2, "fig_d6_adoption_gap.png")

message("[08] figures written to ", FIGS)
