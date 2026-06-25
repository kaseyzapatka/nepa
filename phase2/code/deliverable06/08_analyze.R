# D6 / 08 — analysis figures for the report (final step in the chain)
#
# Reads the D6 analysis artifacts and builds the report figures (Python builds the
# data; this R script builds the figures; deliverable06.qmd embeds them).
#
# Methodology / Analysis-1 figure set (each makes one point):
#   fig_d6_outcomes.png        narrowing: 452 clean -> 293 in a type -> 53 bounded -> adopt
#   fig_d6_action_distribution sort step: every clean FONSI by action type (bounded highlighted)
#   fig_d6_ce_match            best-CE match strength per candidate (0.40 "treat as new" cutoff)
#   fig_d6_sizes               size spread of the bounded FONSIs (the candidate CE bounds)
#   fig_d6_classification      how each candidate's rank score is composed
#   fig_d6_timeline            where the 53 bounded FONSIs fall vs the FRA (Jun 2023) line
#   fig_d6_states              US map of the transmission-upgrade FONSI states
#   fig_d6_adoption_gap        per adopt candidate: evidence weight + who could adopt
#   fig_d6_ce_by_agency        Analysis 3: the existing CE landscape by agency
#   fig_d6_mitigated_share     Analysis 2: share conditioned on committed mitigation
#
# Usage: Rscript phase2/code/deliverable06/08_analyze.R

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(stringr)
  library(arrow); library(ggplot2); library(scales); library(forcats)
  library(sf); library(tigris)
})
options(tigris_use_cache = TRUE)

PHASE2 <- here::here("phase2")
ANALYSIS <- file.path(PHASE2, "data", "analysis", "deliverable06")
OUT  <- file.path(PHASE2, "output", "deliverable06")
FIGS <- file.path(OUT, "figures")
dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)

catf_navy <- "#012169"; catf_dark_blue <- "#0047BB"; catf_lime <- "#93D500"
catf_teal <- "#00AE8D"; catf_magenta <- "#C22A90"; catf_light_blue <- "#8AB7E9"
catf_grey <- "#C9CED6"

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
  ggsave(file.path(FIGS, name), p, width = w, height = h, dpi = 300); message("  wrote ", name)
}
short_label <- function(x) x %>% str_replace(" \\(.*\\)", "") %>% str_wrap(26)

# ---------------------------------------------------------------------------
inv      <- read_parquet(file.path(ANALYSIS, "fonsi_project_inventory.parquet")) %>% mutate(project_id = as.character(project_id))
corp     <- read_parquet(file.path(ANALYSIS, "candidate_corpus.parquet")) %>% mutate(project_id = as.character(project_id))
verdicts <- read_parquet(file.path(ANALYSIS, "candidate_verdicts.parquet"))
mit      <- read_parquet(file.path(ANALYSIS, "candidate_mitigation_summary.parquet"))
ce_land  <- read_parquet(file.path(ANALYSIS, "ce_landscape_ces.parquet"))
facts    <- read_parquet(file.path(ANALYSIS, "candidate_facts.parquet")) %>% mutate(project_id = as.character(project_id))

corp_fonsi   <- corp %>% filter(is_fonsi)
n_clean      <- inv %>% filter(project_energy_type == "Clean") %>% distinct(project_id) %>% nrow()
n_candidate  <- corp_fonsi %>% distinct(project_id) %>% nrow()
n_ce_shaped  <- corp_fonsi %>% filter(is_profile_subtype) %>% distinct(project_id) %>% nrow()
outcome <- corp_fonsi %>% filter(is_profile_subtype) %>%
  select(project_id, candidate_category) %>%
  left_join(verdicts %>% select(candidate_category, verdict), by = "candidate_category") %>%
  distinct(project_id, verdict) %>% count(verdict)
get_v <- function(v) { x <- outcome$n[outcome$verdict == v]; if (length(x)) x[1] else 0L }
n_develop <- get_v("new"); n_expand <- get_v("expand"); n_adopt_f <- get_v("adopt")

# === Fig: outcomes funnel — 452 -> 293 -> 53 -> adopt (clean 4-step narrowing) ===
funnel <- tibble(
  stage = c("Clean-energy EA→FONSI projects", "In a recurring action type",
            "Bounded & low-impact", "Candidate to adopt an existing CE"),
  n = c(n_clean, n_candidate, n_ce_shaped, n_adopt_f),
  fill = c(catf_light_blue, catf_light_blue, catf_teal, catf_navy),
  lab = c(comma(n_clean), comma(n_candidate),
          paste0(n_ce_shaped, "  (", percent(n_ce_shaped / n_clean, 1), " of clean)"),
          paste0(n_adopt_f, "  (all bounded actions resolve to adopt)"))
) %>% mutate(stage = fct_inorder(stage) %>% fct_rev())
p_out <- ggplot(funnel, aes(stage, n, fill = fill)) +
  geom_col(width = 0.68) +
  geom_text(aes(label = lab), hjust = -0.03, size = 3.9, fontface = "bold", color = catf_navy) +
  scale_fill_identity() + coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.35))) +
  labs(title = "Scaling CEs: from every clean-energy FONSI to an adopt opportunity",
       subtitle = glue::glue("{n_ce_shaped} bounded, low-impact actions — 0 develop, 0 expand, {n_adopt_f} adopt"),
       x = NULL, y = "Distinct FONSI projects",
       caption = "Narrowing of the clean-energy FONSI corpus. 'Adopt' = a peer agency already has a categorical exclusion for the action.") +
  theme_catf()
save_fig(p_out, "fig_d6_outcomes.png", w = 9, h = 4.2)

# === Fig: sort step — every clean FONSI by action type, incl. the uncategorized pool ===
dist <- corp_fonsi %>% group_by(candidate_category) %>%
  summarise(total = n_distinct(project_id),
            bounded = n_distinct(project_id[is_profile_subtype]), .groups = "drop") %>%
  mutate(other = total - bounded) %>%
  left_join(verdicts %>% select(candidate_category, candidate_label), by = "candidate_category") %>%
  mutate(lab = short_label(candidate_label)) %>% select(lab, total, bounded, other)
dist <- bind_rows(dist, tibble(lab = "Uncategorized\n(net-new pool)",
                               total = n_clean - n_candidate, bounded = 0, other = n_clean - n_candidate))
distL <- dist %>% pivot_longer(c(bounded, other), names_to = "subset", values_to = "n") %>%
  mutate(subset = factor(subset, levels = c("other", "bounded"),
                         labels = c("Broader / set aside", "Bounded, low-impact (CE-shaped)")))
p_sort <- ggplot(distL, aes(reorder(lab, total), n, fill = subset)) +
  geom_col(width = 0.72) +
  geom_text(data = dist, aes(x = lab, y = total, label = total), inherit.aes = FALSE,
            hjust = -0.2, size = 3.5, fontface = "bold", color = catf_navy) +
  scale_fill_manual(values = c("Broader / set aside" = catf_grey,
                               "Bounded, low-impact (CE-shaped)" = catf_teal), name = NULL) +
  coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Step 1–2 · Sorting the 452 clean-energy FONSIs",
       subtitle = "Each FONSI sorted into an action type; teal = the bounded, low-impact subset kept for matching",
       x = NULL, y = "Distinct FONSI projects") +
  theme_catf() + theme(legend.position = "bottom")
save_fig(p_sort, "fig_d6_action_distribution.png", w = 9, h = 4.6)

# === Fig: CE-match strength per candidate (ranking aid; 0.40 cutoff) ===
mfit <- verdicts %>% filter(verdict != "contrast") %>% mutate(lab = short_label(candidate_label))
p_match <- ggplot(mfit, aes(reorder(lab, best_ce_match_score), best_ce_match_score)) +
  geom_col(width = 0.6, fill = catf_teal) +
  geom_hline(yintercept = 0.40, linetype = "dashed", color = catf_magenta) +
  geom_text(aes(label = sprintf("%.2f → %s", best_ce_match_score, best_ce_structured_id)),
            hjust = -0.05, size = 3.3, color = catf_navy) +
  coord_flip() + scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.45))) +
  labs(title = "Step 3 · How strongly each action matches an existing CE",
       subtitle = "Best-match similarity to the federal CE catalog (dashed = 0.40 'treat as new' cutoff)",
       x = NULL, y = "Best-match similarity (0–1)",
       caption = "A ranking aid, not verified coverage — every match is confirmed against its eCFR text before action.") +
  theme_catf()
save_fig(p_match, "fig_d6_ce_match.png", h = 4.0)

# === Fig: size spread of the bounded FONSIs (the candidate CE bounds) ===
sz <- facts %>% filter(is_profile_subtype) %>%
  transmute(candidate_category,
            `Transmission — line miles` = ifelse(candidate_category == "transmission_upgrade", max_miles, NA),
            `Transmission — voltage (kV)` = ifelse(candidate_category == "transmission_upgrade", max_kilovolts, NA),
            `Geothermal — wells drilled` = ifelse(candidate_category == "geothermal_exploration", n_wells, NA)) %>%
  pivot_longer(-candidate_category, names_to = "metric", values_to = "value") %>%
  filter(!is.na(value), value > 0, value < 1000)   # drop study-area outliers
p_sizes <- ggplot(sz, aes(metric, value)) +
  geom_boxplot(width = 0.45, outlier.shape = NA, fill = catf_light_blue, color = catf_navy, alpha = 0.5) +
  geom_jitter(width = 0.12, height = 0, size = 2, color = catf_navy, alpha = 0.7) +
  facet_wrap(~metric, scales = "free", ncol = 3) +
  labs(title = "Step 3 · The size spread of the bounded FONSIs",
       subtitle = "These observed ranges are the candidate numeric bounds a CE could encode",
       x = NULL, y = NULL,
       caption = "Bounded, low-impact subset only. Each point is one FONSI; study-area outliers excluded.") +
  theme_catf() + theme(axis.text.x = element_blank(), strip.text = element_text(color = catf_navy, face = "bold"))
save_fig(p_sizes, "fig_d6_sizes.png", w = 9, h = 3.6)

# === Fig: classification — how each candidate's rank score is composed ===
comp_lab <- c(rank_novelty = "Novelty (develop>expand>adopt)", rank_volume = "Volume (# FONSIs)",
              rank_diversity = "Agency/state spread", rank_limits = "Has size limits",
              rank_mitigation = "Low mitigation dependence", rank_role = "Profile candidate")
cls <- verdicts %>% filter(verdict != "contrast") %>%
  mutate(lab = short_label(candidate_label)) %>%
  select(lab, rank_score, all_of(names(comp_lab))) %>%
  pivot_longer(all_of(names(comp_lab)), names_to = "component", values_to = "contribution") %>%
  mutate(component = factor(recode(component, !!!comp_lab), levels = unname(comp_lab)))
ord <- verdicts %>% filter(verdict != "contrast") %>% arrange(rank_score) %>% pull(candidate_label) %>% short_label()
cls$lab <- factor(cls$lab, levels = ord)
p_cls <- ggplot(cls, aes(lab, contribution, fill = component)) +
  geom_col(width = 0.66) + coord_flip() +
  scale_fill_manual(values = c(catf_navy, catf_dark_blue, catf_teal, catf_lime, catf_light_blue, catf_magenta), name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Step 4 · How each candidate is scored and ranked",
       subtitle = "Transparent multi-factor rank score (0–1); bar length = total, colors = each factor's contribution",
       x = NULL, y = "Rank score") +
  theme_catf() + theme(legend.position = "bottom") + guides(fill = guide_legend(nrow = 2))
save_fig(p_cls, "fig_d6_classification.png", w = 9, h = 4.4)

# === Fig: FRA timeline — where the 53 bounded FONSIs fall ===
yr_of <- function(x) suppressWarnings(as.integer(str_extract(as.character(x), "\\d{4}")))
dt <- inv %>% transmute(project_id,
                        year = coalesce(yr_of(blm_decision_date), yr_of(doe_decision_date),
                                        yr_of(document_date_from_file_name))) %>%
  mutate(year = ifelse(!is.na(year) & year >= 1995 & year <= 2025, year, NA_integer_))
tl <- corp_fonsi %>% filter(is_profile_subtype) %>% distinct(project_id) %>%
  left_join(dt, by = "project_id")
n_known <- sum(!is.na(tl$year)); n_unk <- sum(is.na(tl$year))
tlc <- tl %>% filter(!is.na(year)) %>% count(year)
p_tl <- ggplot(tlc, aes(year, n)) +
  geom_col(width = 0.8, fill = catf_teal) +
  geom_vline(xintercept = 2023.42, linetype = "dashed", color = catf_magenta, linewidth = 0.8) +
  annotate("text", x = 2023.2, y = max(tlc$n), label = "FRA enacted\nJun 2023", hjust = 1.05,
           size = 3.2, color = catf_magenta, fontface = "bold", lineheight = 0.9) +
  scale_x_continuous(limits = c(2000, 2025), breaks = seq(2000, 2024, 4)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Step 0 · When were these EAs decided?",
       subtitle = glue::glue("The {n_ce_shaped} bounded FONSIs by decision year — every dated one predates the FRA's CE-adoption authority"),
       x = NULL, y = "Bounded FONSIs",
       caption = glue::glue("Date known for {n_known}/{n_ce_shaped}; {n_unk} undated (filename/decision-date fields). ",
                            "A D4-timeline merge would firm up coverage. FRA (Jun 2023) let agencies adopt peer CEs.")) +
  theme_catf()
save_fig(p_tl, "fig_d6_timeline.png", w = 9, h = 4.0)

# === Fig: US map of transmission-upgrade FONSI states (tigris/sf — house pattern) ===
tx_state <- corp_fonsi %>%
  filter(is_profile_subtype, candidate_category == "transmission_upgrade") %>%
  mutate(s = str_remove_all(as.character(project_state), '[\\[\\]"]')) %>%
  separate_rows(s, sep = ",\\s*") %>% mutate(state_name = str_squish(s)) %>%
  filter(state_name != "", !is.na(state_name)) %>% count(state_name, name = "n")
n_tx_states <- nrow(tx_state)
states_sf <- tigris::states(cb = TRUE, year = 2022, progress_bar = FALSE) %>%
  filter(!NAME %in% c("Alaska", "Hawaii", "Puerto Rico", "United States Virgin Islands",
                      "Guam", "Commonwealth of the Northern Mariana Islands", "American Samoa")) %>%
  select(state_name = NAME, geometry) %>%
  left_join(tx_state, by = "state_name")
p_map <- ggplot(states_sf) +
  geom_sf(aes(fill = n), color = "white", linewidth = 0.25) +
  scale_fill_gradient(low = catf_light_blue, high = catf_navy, na.value = "grey92",
                      name = "FONSIs", breaks = pretty_breaks(4)) +
  labs(title = glue::glue("Where the transmission-upgrade FONSIs are — {n_tx_states} states"),
       subtitle = "Bounded, low-impact in-corridor transmission FONSIs, concentrated in the West (BLM / BPA territory)",
       x = NULL, y = NULL,
       caption = "Each could adopt TVA's existing transmission-maintenance CE (#17) instead of running a full EA.") +
  theme_catf() + theme(legend.position = "right", axis.text = element_blank(), panel.grid = element_blank())
save_fig(p_map, "fig_d6_states.png", w = 8.5, h = 5.0)

# === Fig: adoption gap (evidence weight + who could adopt) ===
adopt <- verdicts %>% filter(verdict == "adopt") %>%
  mutate(lab = short_label(candidate_label),
         n_lacking = str_count(adopt_targets, ",") + 1L,
         tag = paste0(n_profile_fonsi, " FONSIs → adopt ", best_ce_structured_id, " (", best_ce_agency, ")"))
p_gap <- ggplot(adopt, aes(reorder(lab, n_profile_fonsi), n_profile_fonsi)) +
  geom_col(width = 0.62, fill = catf_teal) +
  geom_text(aes(label = tag), hjust = -0.03, size = 3.2, color = catf_navy) +
  coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.75))) +
  labs(title = "The adoption gap, by evidence weight",
       subtitle = "Bar = bounded FONSIs run as full EAs; label = the existing CE (and holder) they could adopt instead",
       x = NULL, y = "Bounded, low-impact FONSIs run as full EA→FONSI",
       caption = "Each action already has a categorical exclusion at another agency; adopting it avoids the full EA.") +
  theme_catf()
save_fig(p_gap, "fig_d6_adoption_gap.png", w = 9, h = 3.8)

# === Fig (Analysis 3): the existing CE landscape by agency ===
agc <- ce_land %>% count(agency_name, sort = TRUE) %>% filter(!is.na(agency_name), agency_name != "") %>% head(12)
p_agc <- ggplot(agc, aes(reorder(agency_name, n), n)) +
  geom_col(width = 0.7, fill = catf_dark_blue) +
  geom_text(aes(label = n), hjust = -0.2, size = 3.4, fontface = "bold", color = catf_navy) +
  coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(title = glue::glue("The existing CE landscape: {comma(nrow(ce_land))} CEs across {n_distinct(ce_land$agency_unit)} agency units"),
       subtitle = "Top 12 agencies by number of categorical exclusions on the books",
       x = NULL, y = "Categorical exclusions",
       caption = "Source: CE Explorer export. The breadth is the precedent for adopt — agencies routinely share CE families.") +
  theme_catf()
save_fig(p_agc, "fig_d6_ce_by_agency.png", w = 9, h = 4.4)

# === Fig (Analysis 2): mitigated-FONSI share by candidate ===
mit_fig <- mit %>% left_join(verdicts %>% select(candidate_category, candidate_label, verdict),
                             by = "candidate_category") %>%
  filter(verdict != "contrast") %>% mutate(lab = short_label(candidate_label))
p_mit <- ggplot(mit_fig, aes(reorder(lab, mitigated_share), mitigated_share)) +
  geom_col(width = 0.66, fill = catf_dark_blue) +
  geom_text(aes(label = percent(mitigated_share, accuracy = 1)), hjust = -0.2, size = 3.6, fontface = "bold", color = catf_navy) +
  coord_flip() + scale_y_continuous(labels = percent, limits = c(0, 1), expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Mitigated-FONSI share by candidate",
       subtitle = "Share whose 'no significant impact' finding is conditioned on committed mitigation",
       x = NULL, y = "Mitigated-FONSI share",
       caption = "A CE must encode recurring mitigations as design criteria — it cannot rely on case-by-case commitments.") +
  theme_catf()
save_fig(p_mit, "fig_d6_mitigated_share.png")

message("[08] figures written to ", FIGS)
