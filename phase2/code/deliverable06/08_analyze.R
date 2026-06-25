# D6 / 08 — analysis figures for the report (final step in the chain)
#
# Reads the D6 analysis artifacts (01 corpus, 06 CE landscape/clusters, 07
# verdicts) and builds the report figures. Mirrors the house pattern (cf. D5
# 03_analyze_spikes.R): Python builds the data, this final numbered R script
# builds the figures, and reports/deliverable06.qmd embeds them.
#
# Figures (deliberately few — each makes one point), one per analysis branch:
#   fig_d6_funnel.png          (Track A) narrowing: clean FONSIs -> action type -> develop/expand/adopt
#   fig_d6_adoption_gap.png    (Track A) per adopt candidate, how many agencies lack the existing CE
#   fig_d6_mitigated_share.png (Track B) share of FONSIs conditioned on committed mitigation
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
mit      <- read_parquet(file.path(ANALYSIS, "candidate_mitigation_summary.parquet"))
ce_land  <- read_parquet(file.path(ANALYSIS, "ce_landscape_ces.parquet"))

corp_fonsi   <- corp %>% filter(is_fonsi)
n_clean      <- inv %>% filter(project_energy_type == "Clean") %>% distinct(project_id) %>% nrow()
n_candidate  <- corp_fonsi %>% distinct(project_id) %>% nrow()
n_ce_shaped  <- corp_fonsi %>% filter(is_profile_subtype) %>% distinct(project_id) %>% nrow()
# outcome split by verdict (distinct bounded, low-impact projects)
outcome <- corp_fonsi %>%
  filter(is_profile_subtype) %>%
  select(project_id, candidate_category) %>%
  left_join(verdicts %>% select(candidate_category, verdict), by = "candidate_category") %>%
  distinct(project_id, verdict) %>%
  count(verdict)
get_v <- function(v) { x <- outcome$n[outcome$verdict == v]; if (length(x)) x[1] else 0L }
n_develop <- get_v("new"); n_expand <- get_v("expand"); n_adopt_f <- get_v("adopt")

# Fig 1 — the funnel: narrowing (blue) then the develop/expand/adopt outcomes (green)
funnel <- tibble(
  stage = c("Clean-energy EA → FONSI projects",
            "In a recurring action type",
            "→ Develop a new CE",
            "→ Expand an existing CE",
            "→ Adopt an existing CE"),
  n = c(n_clean, n_candidate, n_develop, n_expand, n_adopt_f),
  fill = c(catf_light_blue, catf_light_blue,
           catf_lime, catf_teal, catf_teal)
) %>% mutate(stage = fct_inorder(stage) %>% fct_rev())

p1 <- ggplot(funnel, aes(stage, n, fill = fill)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = comma(n)), hjust = -0.2, size = 4.2, fontface = "bold", color = catf_navy) +
  scale_fill_identity() +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = "From every clean-energy FONSI down to CE outcomes",
       subtitle = glue::glue("The {n_adopt_f} bounded, low-impact actions are candidate adopt opportunities"),
       x = NULL, y = "Distinct FONSI projects",
       caption = "Candidate adopt opportunities (pending CE-coverage verification). No deterministic expand candidates.") +
  theme_catf()
save_fig(p1, "fig_d6_funnel.png", w = 9, h = 4.6)

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

# Fig 3 — (Track B) mitigated-FONSI share, candidates only (wind/contrast excluded)
mit_fig <- mit |>
  left_join(verdicts |> select(candidate_category, candidate_label, verdict),
            by = "candidate_category") |>
  filter(verdict != "contrast") |>
  mutate(lab = short_label(candidate_label))

p3 <- ggplot(mit_fig, aes(reorder(lab, mitigated_share), mitigated_share)) +
  geom_col(width = 0.66, fill = catf_dark_blue) +
  geom_text(aes(label = percent(mitigated_share, accuracy = 1)),
            hjust = -0.2, size = 3.6, fontface = "bold", color = catf_navy) +
  coord_flip() +
  scale_y_continuous(labels = percent, limits = c(0, 1),
                     expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Mitigated-FONSI share by candidate",
       subtitle = "Share whose 'no significant impact' finding is conditioned on committed mitigation",
       x = NULL, y = "Mitigated-FONSI share",
       caption = "A CE must encode recurring mitigations as design criteria — it cannot rely on case-by-case commitments.") +
  theme_catf()
save_fig(p3, "fig_d6_mitigated_share.png")

# ---------------------------------------------------------------------------
# Analysis 1 walk-through figures (sorting -> bounded subset -> match -> limits)
# Fig 4 — where the FONSIs land: action type, split into bounded vs set-aside
dist <- corp_fonsi %>%
  group_by(candidate_category) %>%
  summarise(total = n_distinct(project_id),
            bounded = n_distinct(project_id[is_profile_subtype]), .groups = "drop") %>%
  mutate(other = total - bounded) %>%
  left_join(verdicts %>% select(candidate_category, candidate_label), by = "candidate_category") %>%
  mutate(lab = short_label(candidate_label))
distL <- dist %>%
  select(lab, total, bounded, other) %>%
  pivot_longer(c(bounded, other), names_to = "subset", values_to = "n") %>%
  mutate(subset = factor(subset, levels = c("other", "bounded"),
                         labels = c("Other (set aside)", "Bounded, low-impact")))
p4 <- ggplot(distL, aes(reorder(lab, total), n, fill = subset)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = c("Other (set aside)" = catf_light_blue,
                               "Bounded, low-impact" = catf_teal), name = NULL) +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(title = "Where the clean-energy FONSIs land",
       subtitle = "Each FONSI sorted into an action type; teal = the bounded, low-impact subset",
       x = NULL, y = "FONSI projects") +
  theme_catf() + theme(legend.position = "bottom")
save_fig(p4, "fig_d6_action_distribution.png", h = 4.2)

# Fig 5 — best CE-match strength per candidate (ranking aid; 0.40 = "treat as new")
mfit <- verdicts %>% filter(verdict != "contrast") %>% mutate(lab = short_label(candidate_label))
p5 <- ggplot(mfit, aes(reorder(lab, best_ce_match_score), best_ce_match_score)) +
  geom_col(width = 0.6, fill = catf_teal) +
  geom_hline(yintercept = 0.40, linetype = "dashed", color = catf_magenta) +
  geom_text(aes(label = sprintf("%.2f", best_ce_match_score)), hjust = -0.25, size = 3.6, color = catf_navy) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1))) +
  labs(title = "How strongly each action matches an existing CE",
       subtitle = "Similarity of the best-matching existing CE (dashed line = 0.40 'treat as new' cutoff)",
       x = NULL, y = "Best-match similarity (0–1)",
       caption = "A ranking aid, not verified coverage — confirm the matched CE against its eCFR text.") +
  theme_catf()
save_fig(p5, "fig_d6_ce_match.png", h = 4.0)

# Fig 6 — do the matched CEs state a numeric limit? (none do -> 0 expand)
metrics_lab <- c(bound_acres = "acres", bound_miles = "miles", bound_kv = "kV", bound_mw = "MW")
lim <- verdicts %>% filter(verdict == "adopt") %>%
  transmute(lab = short_label(candidate_label), ce = best_ce_structured_id) %>%
  left_join(ce_land %>% select(structured_id, bound_acres, bound_miles, bound_kv, bound_mw),
            by = c("ce" = "structured_id")) %>%
  pivot_longer(starts_with("bound_"), names_to = "metric", values_to = "val") %>%
  mutate(metric = factor(recode(metric, !!!metrics_lab), levels = c("acres", "miles", "kV", "MW")),
         stated = ifelse(is.na(val), "No limit stated", "Stated"))
p6 <- ggplot(lim, aes(metric, lab, fill = stated)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = ifelse(is.na(val), "—", formatC(val))), color = catf_navy, size = 4) +
  scale_fill_manual(values = c("No limit stated" = "grey88", "Stated" = catf_teal), name = NULL) +
  labs(title = "Do the matched CEs state a numeric limit?",
       subtitle = "None state a parsed limit — so no deterministic expand candidates (not verified within-bounds)",
       x = NULL, y = NULL) +
  theme_catf() + theme(panel.grid = element_blank(), legend.position = "none")
save_fig(p6, "fig_d6_ce_limits.png", w = 6.5, h = 3.2)

message("[08] figures written to ", FIGS)
