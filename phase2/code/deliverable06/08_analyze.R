# D6 / 08 — analysis figures for the report (final step in the chain)
#
# Reads the D6 analysis artifacts and builds the report figures (Python builds the
# data; this R script builds the figures; deliverable06.qmd embeds them).
#
# Methodology / Analysis-1 figure set (each makes one point):
#   fig_d6_outcomes_waffle     data at a glance: 452 clean -> not-recurring / recurring / bounded
#   fig_d6_action_distribution sort step: every clean FONSI by action type (bounded highlighted)
#   fig_d6_keep_bounded        keep step: the low-impact siting fingerprint of the bounded subset
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
  library(sf); library(tigris); library(ggwordcloud); library(ggbeeswarm)
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

# "bounded" / CE-shaped = the Rule-B flag from 09 (LLM-bounded + transmission shape gate).
# candidate_facts is now the candidate set itself: one row per FONSI, keyed on corrected action_category.
facts        <- facts %>% mutate(is_bounded = is_ce_shaped %in% TRUE)
corp_fonsi   <- facts
n_clean      <- inv %>% filter(project_energy_type == "Clean") %>% distinct(project_id) %>% nrow()
n_candidate  <- corp_fonsi %>% distinct(project_id) %>% nrow()
n_ce_shaped  <- corp_fonsi %>% filter(is_bounded) %>% distinct(project_id) %>% nrow()
outcome <- corp_fonsi %>% filter(is_bounded) %>%
  select(project_id, candidate_category) %>%
  left_join(verdicts %>% select(candidate_category, verdict), by = "candidate_category") %>%
  distinct(project_id, verdict) %>% count(verdict)
get_v <- function(v) { x <- outcome$n[outcome$verdict == v]; if (length(x)) x[1] else 0L }
n_develop <- get_v("new"); n_expand <- get_v("expand"); n_adopt_f <- get_v("adopt")

# === Fig 1 (data at a glance): waffle of the 452 FONSIs (each square ~= 4.5 FONSIs) ===
n_broader <- n_candidate - n_ce_shaped          # in a recurring type but not bounded
n_uncat   <- n_clean - n_candidate              # not in any recurring type (net-new pool)
ord1 <- c("Not recurring", "Recurring", "Bounded")          # top -> bottom in grid + legend
wv1  <- c(`Not recurring` = 35, Recurring = 53, Bounded = 12)[ord1]
cnt1 <- c(`Not recurring` = n_uncat, Recurring = n_broader, Bounded = n_ce_shaped)[ord1]
waf1 <- tibble(cat = factor(rep(ord1, wv1), levels = ord1)) %>%
  mutate(i = row_number() - 1, x = i %% 10, y = i %/% 10)
pal1 <- c(Bounded = catf_navy, Recurring = catf_light_blue, `Not recurring` = catf_grey)
lab_pos <- waf1 %>% group_by(cat) %>% summarise(y = mean(y), .groups = "drop") %>%
  mutate(lab = paste0(cnt1[as.character(cat)], " (", round(cnt1[as.character(cat)] / n_clean * 100), "%)"))
p_waffle1 <- ggplot(waf1, aes(x, y, fill = cat)) +
  geom_tile(color = "white", linewidth = 1.6) +
  geom_text(data = lab_pos, aes(x = -0.9, y = y, label = lab), inherit.aes = FALSE,
            hjust = 1, fontface = "bold", size = 3.6, color = catf_navy) +
  scale_fill_manual(values = pal1, name = NULL) +
  scale_x_continuous(expand = expansion(add = c(2.8, 0.2))) +
  coord_equal(clip = "off") + scale_y_reverse() +
  labs(title = "The 452 decarbonization FONSIs at a glance",
       subtitle = str_wrap(glue::glue("Of the {comma(n_clean)} FONSIs, {n_candidate} recur ",
                 "({n_broader} broad + {n_ce_shaped} bounded) while {n_uncat} are uncategorized"), 90),
       caption = glue::glue("Each square ≈ {round(n_clean / 100, 1)} FONSIs.")) +
  theme_void(base_size = 12) +
  theme(legend.position = "right", plot.title = element_text(face = "bold", color = catf_navy),
        plot.subtitle = element_text(color = catf_dark_blue), plot.caption = element_text(color = "gray50", hjust = 0))
save_fig(p_waffle1, "fig_d6_outcomes_waffle.png", w = 9, h = 5)

# === Fig: sort step — every clean FONSI by action type, incl. the uncategorized pool ===
dd <- corp_fonsi %>% group_by(candidate_category) %>%
  summarise(total = n_distinct(project_id),
            bounded = n_distinct(project_id[is_bounded]), .groups = "drop") %>%
  left_join(verdicts %>% select(candidate_category, candidate_label), by = "candidate_category") %>%
  mutate(lab = short_label(candidate_label)) %>% filter(!is.na(lab))
sortL <- dd %>% mutate(Broader = total - bounded, Bounded = bounded) %>%
  pivot_longer(c(Bounded, Broader), names_to = "subset", values_to = "n") %>%
  group_by(lab) %>% mutate(share = n / sum(n)) %>% ungroup() %>%
  mutate(subset = factor(subset, levels = c("Broader", "Bounded")))
p_sort <- ggplot(sortL, aes(y = reorder(lab, total), x = n, fill = subset)) +
  geom_col(position = "fill", width = 0.7) +
  geom_text(aes(label = ifelse(share >= 0.06, paste0(percent(share, 1), " (", n, ")"), "")),
            position = position_fill(vjust = 0.5), color = "white", fontface = "bold", size = 3) +
  geom_text(data = dd, aes(y = lab, x = 1, label = paste0("n = ", total)), inherit.aes = FALSE,
            hjust = -0.12, size = 3, color = catf_navy) +
  scale_fill_manual(values = c("Broader" = catf_grey, "Bounded" = catf_navy), name = NULL,
                    guide = guide_legend(reverse = TRUE)) +
  scale_x_continuous(labels = percent, expand = expansion(mult = c(0, 0.14))) +
  labs(title = glue::glue("Sorting the {n_clean} decarbonization FONSIs"),
       subtitle = str_wrap(glue::glue("Within each action type, the share that is bounded & low-impact (teal, kept for ",
                 "matching) vs broader (grey, set aside). A further {n_uncat} FONSIs are uncategorized."), 96),
       x = "Share of the action type", y = NULL,
       caption = "Teal = bounded, low-impact subset carried to Step 3; grey = broader. n = total FONSIs of that type.") +
  theme_catf() + theme(legend.position = "bottom")
save_fig(p_sort, "fig_d6_action_distribution.png", w = 9, h = 4.2)

# === Fig: what makes the kept FONSIs "bounded, low-impact" (the keep step) ===
prof_keep <- facts %>% filter(is_bounded)
keep_attr <- tibble(
  attribute = c("On previously disturbed / developed land", "Within an existing right-of-way",
                "Temporary / short-duration work", "No new permanent access road*"),
  share = c(mean(prof_keep$previously_disturbed_land, na.rm = TRUE),
            mean(prof_keep$within_existing_row, na.rm = TRUE),
            mean(prof_keep$is_temporary, na.rm = TRUE),
            mean(prof_keep$no_new_access_road, na.rm = TRUE)))
p_keep <- ggplot(keep_attr, aes(share, reorder(attribute, share))) +
  geom_col(width = 0.6, fill = catf_dark_blue) +
  geom_text(aes(label = percent(share, 1)), hjust = -0.2, size = 3.8, fontface = "bold", color = catf_navy) +
  scale_x_continuous(labels = percent, limits = c(0, 1), expand = expansion(mult = c(0, 0.12))) +
  labs(title = glue::glue("What makes the {n_ce_shaped} kept FONSIs 'bounded, low-impact'"),
       subtitle = "Share of the bounded subset with each low-impact siting trait",
       x = NULL, y = NULL,
       caption = str_wrap(paste("The kept subset skews to in-corridor work on already-disturbed land — the CE-shaped profile.",
                "*Counts only FONSIs that explicitly state no new road, so it under-counts (most simply don't address roads)."), 120)) +
  theme_catf()
save_fig(p_keep, "fig_d6_keep_bounded.png", w = 8.5, h = 2.8)

# === Fig: CE-match strength per candidate (ranking aid; 0.40 cutoff) ===
mfit <- verdicts %>% filter(verdict != "contrast") %>% mutate(lab = short_label(candidate_label))
p_match <- ggplot(mfit, aes(reorder(lab, best_ce_match_score), best_ce_match_score)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = 0.20, fill = catf_grey, alpha = 0.5) +
  annotate("text", x = Inf, y = 0.10, label = "baseline similarity", hjust = 0.5, vjust = -0.7,
           size = 2.9, color = "gray30", fontface = "italic") +
  geom_col(width = 0.6, fill = catf_dark_blue) +
  geom_text(aes(label = sprintf("%.2f", best_ce_match_score)), hjust = -0.3, size = 3.5,
            fontface = "bold", color = catf_navy) +
  coord_flip(clip = "off") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "How strongly each action matches an existing CE",
       subtitle = "Closest existing CE by text similarity (0–1)",
       x = NULL, y = "Best-match similarity (0–1)",
       caption = str_wrap(paste("Blended semantic + word-overlap similarity. Grey band = baseline, where unrelated CEs score",
                "(≤ ~0.20); the matches sit 2–6× above it. A ranking aid — every match is confirmed against its eCFR text (see table above)."), 118)) +
  theme_catf()
save_fig(p_match, "fig_d6_ce_match.png", h = 4.0)

# === Fig: bounded FONSIs vs existing-CE stated limits (the expand-test comparison) ===
prof_sz <- facts %>% filter(is_bounded)
BLUE <- catf_dark_blue; GREEN <- "#1D9E75"   # Existing CE limits = blue; Bounded FONSIs = green
grp_pal <- c("Bounded FONSIs" = GREEN, "Existing CE limits" = BLUE)
sz <- bind_rows(
  tibble(metric = "Line length (miles)", group = "Bounded FONSIs",     value = prof_sz$max_miles),
  tibble(metric = "Voltage (kV)",        group = "Bounded FONSIs",     value = prof_sz$max_kilovolts),
  tibble(metric = "Disturbance (acres)", group = "Bounded FONSIs",     value = prof_sz$max_acres),
  tibble(metric = "Line length (miles)", group = "Existing CE limits", value = ce_land$bound_miles),
  tibble(metric = "Voltage (kV)",        group = "Existing CE limits", value = ce_land$bound_kv),
  tibble(metric = "Disturbance (acres)", group = "Existing CE limits", value = ce_land$bound_acres)
) %>%
  filter(!is.na(value), value > 0,
         !(metric == "Disturbance (acres)" & value > 10000),   # drop study-area outliers
         !(metric == "Line length (miles)" & value > 200)) %>%
  mutate(metric = factor(metric, levels = c("Line length (miles)", "Voltage (kV)", "Disturbance (acres)")),
         group  = factor(group, levels = c("Bounded FONSIs", "Existing CE limits")))
ncount  <- sz %>% count(metric, group)
# shade ONLY the panel(s) where the bounded FONSIs exceed the CE limits (the flag)
winners <- sz %>% group_by(metric, group) %>% summarise(med = median(value), .groups = "drop") %>%
  group_by(metric) %>% slice_max(med, n = 1, with_ties = FALSE) %>% ungroup() %>%
  filter(group == "Bounded FONSIs")
p_sizes <- ggplot(sz, aes(x = value, y = group)) +
  geom_rect(data = winners, aes(fill = group), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf,
            alpha = 0.12, inherit.aes = FALSE) +
  geom_jitter(aes(color = group), height = 0.18, width = 0, size = 1.5, alpha = 0.28) +   # dots UNDER the box, faint
  geom_boxplot(aes(color = group), width = 0.55, fill = NA, outlier.shape = NA, linewidth = 0.75, fatten = 2.2) +  # box ON TOP so median is visible
  geom_text(data = ncount, aes(x = Inf, y = group, label = paste0("n = ", n)), inherit.aes = FALSE,
            hjust = 1.1, size = 2.8, color = "gray45") +
  facet_wrap(~metric, scales = "free_x", ncol = 1) +
  scale_x_log10(labels = label_comma()) +
  scale_color_manual(values = grp_pal, name = NULL) +
  scale_fill_manual(values = grp_pal, guide = "none") +
  labs(title = "Our bounded FONSIs vs the limits existing CEs state",
       subtitle = str_wrap(paste("The miles panel is shaded green — the one dimension where our bounded FONSIs run",
                "past the limits CEs typically state; on voltage and acres the existing limits already cover them."), 100),
       x = "Stated size (log scale)", y = NULL,
       caption = str_wrap(paste("Box = median & middle 50%; dots = individual FONSIs / CE limits. Existing CEs bound voltage",
                "only twice (n=2), so that row is indicative. Study-area outliers excluded."), 115)) +
  theme_catf() + theme(legend.position = "top", strip.text = element_text(color = catf_navy, face = "bold"),
                       axis.text.y = element_blank(), axis.ticks.y = element_blank(), panel.spacing = unit(1.5, "lines"))
save_fig(p_sizes, "fig_d6_sizes.png", w = 9, h = 7.7)

# === Fig: classification — how each candidate's rank score is composed ===
comp_lab <- c(rank_novelty = "Novelty", rank_volume = "Volume",
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
  scale_fill_manual(values = c("#012169", "#23457F", "#3D6BAB", "#5E8FD0", "#8AB7E9", "#C2CBD6"), name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(title = "How each candidate is scored and ranked",
       subtitle = "Transparent multi-factor rank score (0–1); bar length = total, colors = each factor's contribution",
       x = NULL, y = "Rank score") +
  theme_catf() + theme(legend.position = "bottom") + guides(fill = guide_legend(nrow = 2))
save_fig(p_cls, "fig_d6_classification.png", w = 9, h = 4.4)

# === Fig: FRA timeline — the 53 bounded FONSIs by D4 decision date ===
tl <- facts %>% filter(is_bounded) %>% distinct(project_id, decision_date) %>%
  mutate(d = suppressWarnings(as.Date(decision_date)),
         year = as.integer(format(d, "%Y")),
         year = ifelse(!is.na(year) & year >= 1995 & year <= 2026, year, NA_integer_),
         era = case_when(is.na(year) ~ "Undated", d < as.Date("2023-06-03") ~ "Pre-FRA", TRUE ~ "Post-FRA"))
n_pre <- sum(tl$era == "Pre-FRA"); n_post <- sum(tl$era == "Post-FRA"); n_unk <- sum(tl$era == "Undated")
tlc    <- tl %>% filter(!is.na(year)) %>% count(year, era)
yr_tot <- tlc %>% group_by(year) %>% summarise(n = sum(n), .groups = "drop")
ymax   <- max(yr_tot$n)
p_tl <- ggplot(tlc, aes(year, n, fill = era)) +
  geom_col(width = 0.8) +
  geom_text(data = yr_tot, aes(year, n, label = n), inherit.aes = FALSE,
            vjust = -0.4, size = 2.8, color = catf_navy) +
  geom_vline(xintercept = 2023.42, linetype = "dashed", color = "gray45", linewidth = 0.8) +
  annotate("text", x = 2023.0, y = ymax, label = "FRA enacted\nJun 2023", hjust = 1.05, vjust = 1,
           size = 3.2, color = "gray45", fontface = "bold", lineheight = 0.9) +
  scale_fill_manual(values = c("Pre-FRA" = catf_navy, "Post-FRA" = "#9AA1AC", "Undated" = catf_grey), name = NULL) +
  scale_x_continuous(limits = c(2002, 2026), breaks = seq(2004, 2026, 4)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = "When were these EAs decided?",
       subtitle = str_wrap(glue::glue("The {n_ce_shaped} bounded FONSIs by decision year — ",
                 "{n_pre} pre-FRA, {n_post} post-FRA, {n_unk} undated"), 95),
       x = NULL, y = "Bounded FONSIs",
       caption = str_wrap(glue::glue("Decision dates merged from the D4 timeline, known for {n_pre + n_post}/{n_ce_shaped}. ",
                 "FRA (Jun 2023) gave agencies authority to adopt another agency's CE."), 110)) +
  theme_catf() + theme(legend.position = "bottom")
save_fig(p_tl, "fig_d6_timeline.png", w = 9, h = 4.0)

# === Fig: US map of transmission-upgrade FONSI states (tigris/sf — house pattern) ===
tx_state <- corp_fonsi %>%
  filter(is_bounded, candidate_category == "transmission_upgrade") %>%
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
  geom_sf_text(data = filter(states_sf, !is.na(n)), aes(label = n),
               color = "white", fontface = "bold", size = 3.2) +
  scale_fill_gradient(low = catf_light_blue, high = catf_navy, na.value = "grey92",
                      name = "FONSIs", breaks = pretty_breaks(4)) +
  labs(title = glue::glue("Where the transmission-upgrade FONSIs are — {n_tx_states} states"),
       subtitle = "Bounded, low-impact in-corridor transmission FONSIs, concentrated in the West (BLM / BPA territory)",
       x = NULL, y = NULL,
       caption = str_wrap(paste("Count = transmission FONSIs touching each state; 4 projects span two states, so the 37 projects",
                "sum to 41 state-counts. Each maps to a TVA transmission CE (#17 reconductoring / #19 rebuild)."), 110)) +
  theme_catf() + theme(legend.position = "right", axis.text = element_blank(), panel.grid = element_blank())
save_fig(p_map, "fig_d6_states.png", w = 8.5, h = 5.0)

# === Fig: the transmission FONSIs — adopt-ready (LLM-bounded) vs too-big (expand/develop) ===
# Adopt vs expand = the LLM's is_bounded_low_impact judgment; #17 vs #19 from the action text.
tx_split <- facts %>% filter(candidate_category == "transmission_upgrade", is_profile_subtype) %>%
  distinct(project_id, .keep_all = TRUE) %>%
  mutate(txt = tolower(paste(coalesce(quoted_span, ""), coalesce(action_definition, ""))),
         is_rebuild = str_detect(txt, "rebuild|reconstruct|(remov|replac|install|new)\\w*[^.]{0,40}(structure|pole|tower|h-frame|monopole)"),
         bucket = case_when(!is_bounded ~ "Too big → expand / develop",
                            is_rebuild  ~ "Adopt CE #19 (rebuild ≤ 25 mi)",
                            TRUE        ~ "Adopt CE #17 (modify / reconductor)"),
         bucket = factor(bucket, levels = c("Adopt CE #17 (modify / reconductor)",
                                            "Adopt CE #19 (rebuild ≤ 25 mi)", "Too big → expand / develop")))
n_adopt_tx <- sum(tx_split$is_bounded); n_exp_tx <- sum(!tx_split$is_bounded)
p_cesplit <- ggplot(count(tx_split, bucket), aes(x = bucket, y = n, fill = bucket)) +
  geom_col(width = 0.66) +
  geom_text(aes(label = n), hjust = -0.35, fontface = "bold", color = catf_navy, size = 4.4) +
  scale_fill_manual(values = c("Adopt CE #17 (modify / reconductor)" = catf_navy,
                               "Adopt CE #19 (rebuild ≤ 25 mi)" = catf_dark_blue,
                               "Too big → expand / develop" = catf_light_blue), guide = "none") +
  scale_x_discrete(labels = function(x) str_wrap(x, 18)) +   # wrap so the CE #17/#19 labels aren't cut off
  coord_flip(clip = "off") + scale_y_continuous(expand = expansion(mult = c(0, 0.14))) +
  labs(title = glue::glue("The {nrow(tx_split)} transmission FONSIs: {n_adopt_tx} adopt-ready, {n_exp_tx} too big"),
       subtitle = str_wrap(glue::glue("The model judged {n_adopt_tx} small / in-corridor (adopt — CE #17 modify or #19 rebuild ≤ 25 mi); ",
                "the other {n_exp_tx} are large rebuilds it flagged as not low-impact — the expand / develop case."), 96),
       x = NULL, y = "Transmission FONSIs (rule-profiled)",
       caption = "Adopt vs expand = the model's is_bounded_low_impact judgment; #17 vs #19 from the action text.") +
  theme_catf() + theme(axis.text.y = element_text(color = catf_navy, face = "bold"))
save_fig(p_cesplit, "fig_d6_ce_split.png", w = 9, h = 3.0)

# === Fig: adoption gap (evidence weight + who could adopt) ===
adopt <- verdicts %>% filter(verdict == "adopt") %>%
  mutate(lab = short_label(candidate_label),
         n_lacking = str_count(adopt_targets, ",") + 1L,
         tag = paste0(n_profile_fonsi, " FONSIs → adopt ", best_ce_structured_id, " (", best_ce_agency, ")"))
p_gap <- ggplot(adopt, aes(reorder(lab, n_profile_fonsi), n_profile_fonsi)) +
  geom_col(width = 0.62, fill = catf_dark_blue) +
  geom_text(aes(label = tag), hjust = -0.03, size = 3.2, color = catf_navy) +
  coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.75))) +
  labs(title = "The adoption gap, by evidence weight",
       subtitle = "Bar = bounded FONSIs run as full EAs; label = the existing CE (and holder) they could adopt instead",
       x = NULL, y = "Bounded, low-impact FONSIs run as full EA→FONSI",
       caption = "Each action already has a categorical exclusion at another agency; adopting it avoids the full EA.") +
  theme_catf()
save_fig(p_gap, "fig_d6_adoption_gap.png", w = 9, h = 3.8)

# === Analysis 3: roll up to DEPARTMENT (the agency_unit prefix, e.g. "DOI - BLM" -> DOI) ===
ce_land <- ce_land %>% mutate(dept = str_trim(str_extract(agency_unit, "^[^-]+")),
                              dept = ifelse(is.na(dept) | dept == "", "Other", dept))
n_dept    <- n_distinct(ce_land$dept)
ce_dept   <- ce_land %>% count(dept, name = "ce") %>% arrange(desc(ce)) %>% mutate(rank = row_number())
total_all <- sum(ce_dept$ce)
k50       <- which(cumsum(ce_dept$ce) / total_all >= 0.50)[1]
top4      <- ce_dept$dept[1:4]
BLUE4     <- c("#012169", "#2A5499", "#5A86C4", "#8AB7E9"); GREY <- catf_grey
DEPT_FULL <- c(DOI = "Department of the Interior (DOI)", DOD = "Department of Defense (DOD)",
               DOT = "Department of Transportation (DOT)", DHS = "Department of Homeland Security (DHS)",
               DOC = "Department of Commerce (DOC)", HHS = "Department of Health and Human Services (HHS)",
               DOE = "Department of Energy (DOE)", USDA = "Department of Agriculture (USDA)")
dept_full <- function(d) ifelse(d %in% names(DEPT_FULL), DEPT_FULL[d], d)

# --- Waffle: 10x10 (each square ~= total/100 CEs), mirroring Figure 1 ---
sq  <- round(ce_dept$ce[1:4] / total_all * 100); sq <- c(sq, 100 - sum(sq))
cnt <- c(ce_dept$ce[1:4], total_all - sum(ce_dept$ce[1:4]))
ordw <- c(unname(dept_full(top4)), paste0(n_dept - 4, " other departments"))   # DOI..DHS top, rest bottom
waf <- tibble(cat = factor(rep(ordw, sq), levels = ordw)) %>%
  mutate(i = row_number() - 1, x = i %% 10, y = i %/% 10)
pal_w <- setNames(c(BLUE4, GREY), ordw)
labw  <- waf %>% group_by(cat) %>% summarise(y = mean(y), .groups = "drop") %>%
  mutate(idx = as.integer(cat), lab = paste0(comma(cnt[idx]), " (", sq[idx], "%)"))
p_waffle <- ggplot(waf, aes(x, y, fill = cat)) +
  geom_tile(color = "white", linewidth = 1.6) +
  geom_text(data = labw, aes(x = -0.9, y = y, label = lab), inherit.aes = FALSE,
            hjust = 1, fontface = "bold", size = 3.4, color = catf_navy) +
  scale_fill_manual(values = pal_w, name = "Departments") +
  scale_x_continuous(expand = expansion(add = c(2.8, 0.2))) +
  coord_equal(clip = "off") + scale_y_reverse() +
  labs(title = "Four departments hold half the CE landscape",
       subtitle = glue::glue("Of {comma(total_all)} categorical exclusions across {n_dept} departments, the top four hold 50%"),
       caption = glue::glue("Each square ≈ {round(total_all / 100)} CEs.")) +
  theme_void(base_size = 12) +
  theme(legend.position = "right", plot.title = element_text(face = "bold", color = catf_navy),
        plot.subtitle = element_text(color = catf_dark_blue), plot.caption = element_text(color = "gray50", hjust = 0),
        plot.background = element_rect(fill = "white", color = NA))
save_fig(p_waffle, "fig_d6_ce_waffle.png", w = 9.5, h = 5)

# --- Figure 13: top 20 agencies, colored by the top-4 dept ramp (+ grey) ---
agc <- ce_land %>% filter(!is.na(agency_name), agency_name != "") %>%
  count(agency_name, dept, sort = TRUE) %>% slice_head(n = 20) %>%
  mutate(col = ifelse(dept %in% top4, dept, "Other dept"),
         col = factor(col, levels = c(top4, "Other dept")))
pal13 <- c(setNames(BLUE4, top4), "Other dept" = GREY)
p_agc <- ggplot(agc, aes(reorder(agency_name, n), n, fill = col)) +
  geom_col(width = 0.74) +
  geom_text(aes(label = n), hjust = -0.25, size = 2.9, color = catf_navy) +
  scale_fill_manual(values = pal13, name = "Department") +
  coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Top 20 agencies by number of categorical exclusions",
       subtitle = "Colored by department; the four that hold half the catalog are in teal",
       x = NULL, y = "Categorical exclusions", caption = "Source: CE Explorer export.") +
  theme_catf() + theme(legend.position = "right")
save_fig(p_agc, "fig_d6_ce_by_agency.png", w = 9.5, h = 5.5)

# --- Figure 14a: only 86 of 2,105 state a numeric limit — as a waffle (mirrors Figure 1/14) ---
n_any_bound <- sum(ce_land$states_any_bound, na.rm = TRUE)
nb <- n_any_bound; ntot <- nrow(ce_land); nq <- ntot - nb
sqn <- round(c(nb, nq) / ntot * 100); sqn[2] <- 100 - sqn[1]
ordn <- c("States a numeric limit", "Qualitative limits only"); cntn <- c(nb, nq)
wafn <- tibble(cat = factor(rep(ordn, sqn), levels = ordn)) %>%
  mutate(i = row_number() - 1, x = i %% 10, y = i %/% 10)
pal_n <- setNames(c(catf_dark_blue, GREY), ordn)
labn  <- wafn %>% group_by(cat) %>% summarise(y = mean(y), .groups = "drop") %>%
  mutate(idx = as.integer(cat), lab = paste0(comma(cntn[idx]), " (", percent(cntn[idx] / ntot, 1), ")"))
p_numlim <- ggplot(wafn, aes(x, y, fill = cat)) +
  geom_tile(color = "white", linewidth = 1.6) +
  geom_text(data = labn, aes(x = -0.9, y = y, label = lab), inherit.aes = FALSE,
            hjust = 1, fontface = "bold", size = 3.4, color = catf_navy) +
  scale_fill_manual(values = pal_n, name = "Limit type") +
  scale_x_continuous(expand = expansion(add = c(2.8, 0.2))) +
  coord_equal(clip = "off") + scale_y_reverse() +
  labs(title = glue::glue("Only {nb} of {comma(ntot)} CEs state an explicit numeric limit"),
       subtitle = str_wrap("The rest bound the action qualitatively — 'routine', 'minor', 'small-scale', 'temporary' — not with numbers", 90),
       caption = glue::glue("Each square ≈ {round(ntot / 100)} CEs.")) +
  theme_void(base_size = 12) +
  theme(legend.position = "right", plot.title = element_text(face = "bold", color = catf_navy),
        plot.subtitle = element_text(color = catf_dark_blue), plot.caption = element_text(color = "gray50", hjust = 0),
        plot.background = element_rect(fill = "white", color = NA))
save_fig(p_numlim, "fig_d6_ce_numlimit.png", w = 9.5, h = 5)

# --- Fig: every stated numeric limit as a lollipop, height = how many CEs use that value ---
# Shows "no common threshold" directly: the values sprawl across the log axis, none dominates.
bcnt <- ce_land %>% transmute(`Acreage limit (acres)` = bound_acres, `Length limit (miles)` = bound_miles) %>%
  pivot_longer(everything(), names_to = "metric", values_to = "value") %>% filter(!is.na(value), value > 0) %>%
  count(metric, value) %>%
  group_by(metric) %>% mutate(metric_n = paste0(metric, "  (", sum(n), " CEs · ", n_distinct(value), " different values)")) %>% ungroup()
p_bnd3 <- ggplot(bcnt, aes(value, n)) +
  geom_segment(aes(xend = value, yend = 0), color = catf_light_blue, linewidth = 0.8) +
  geom_point(color = catf_navy, size = 2.6) +
  facet_wrap(~metric_n, scales = "free", ncol = 1) +
  scale_x_log10(labels = label_comma()) + scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "No common threshold — every CE picks a different number",
       subtitle = str_wrap(paste("Each stick is one stated limit; its height is how many CEs use exactly that number.",
                "The values sprawl across the log axis and none dominates."), 95),
       x = "Stated limit (log scale)", y = "CEs using that exact value",
       caption = "Acreage and mileage limits among the 86 CEs that state a number.") +
  theme_catf() + theme(strip.text = element_text(color = catf_navy, face = "bold"), panel.spacing = unit(1.4, "lines"))
save_fig(p_bnd3, "fig_d6_ce_bounds_lolli.png", w = 8.5, h = 5)

# --- Figure 15: relatedness map — t-SNE layout, KMeans clusters, convex hulls ---
# curated topic labels for the 8 deterministic k-means families (cluster_km -> topic); see the D6
# architecture doc. Tied to the deterministic clustering in 06; revisit if the clustering changes.
CE_TOPICS <- c("0" = "Property leases, licenses, and permits", "1" = "Geological surveys and site assessments",
               "2" = "Routine maintenance and minor ground work", "3" = "Hazmat and disposal",
               "4" = "Goods, services, and personnel procurement", "5" = "Rules, standards, and guidance",
               "6" = "Monitoring and rights-of-way", "7" = "Airport layout plans and monitoring equipment")
if ("coord_x" %in% names(ce_land) && any(!is.na(ce_land$coord_x))) {
  sc    <- ce_land %>% filter(!is.na(coord_x)) %>% mutate(cl = factor(cluster_km))
  pal_cl <- setNames(RColorBrewer::brewer.pal(max(3, nlevels(sc$cl)), "Set2")[seq_len(nlevels(sc$cl))], levels(sc$cl))
  topic_labs <- dplyr::coalesce(CE_TOPICS[levels(sc$cl)], levels(sc$cl))   # legend keys = the topic labels
  p_scatter <- ggplot(sc, aes(coord_x, coord_y)) +
    geom_point(aes(color = cl), size = 1.5, alpha = 0.7) +
    scale_color_manual(values = pal_cl, labels = topic_labs, name = NULL,
                       guide = guide_legend(override.aes = list(size = 4.5, alpha = 1), ncol = 2)) +
    labs(title = "How related are the existing CEs?",
         subtitle = str_wrap(paste("Each point is one CE, laid out by t-SNE of its text embedding; color = topic family.",
                  "Closer = more similar wording; families recur across departments."), 95),
         x = NULL, y = NULL,
         caption = "Many families recur across departments — the precedent for adopt.") +
    theme_catf() + theme(axis.text = element_blank(), panel.grid = element_blank(),
                         legend.position = "bottom", legend.text = element_text(size = 9),
                         legend.key.size = unit(0.55, "cm"))
  save_fig(p_scatter, "fig_d6_ce_scatter.png", w = 9.5, h = 7.2)
}

# Appendix: how many CE clusters? inertia elbow + silhouette across k. The silhouette is low and
# flat, so k = 8 is a readability choice for the scatter, not a natural optimum.
ksel_path <- file.path(ANALYSIS, "ce_kselection.parquet")
if (file.exists(ksel_path)) {
  ksel <- read_parquet(ksel_path)
  ksl <- ksel %>% tidyr::pivot_longer(c(inertia, silhouette), names_to = "metric", values_to = "value") %>%
    mutate(metric = recode(metric, inertia = "Inertia (elbow)", silhouette = "Silhouette (separation)"))
  p_elbow <- ggplot(ksl, aes(k, value)) +
    geom_line(color = catf_dark_blue, linewidth = 0.8) + geom_point(color = catf_navy, size = 1.8) +
    geom_vline(xintercept = 8, linetype = "dashed", color = catf_grey) +
    facet_wrap(~ metric, scales = "free_y") +
    scale_x_continuous(breaks = ksel$k) +
    labs(title = "Choosing the number of CE clusters (k = 8)",
         subtitle = str_wrap(paste("Inertia falls smoothly (no sharp elbow) and the silhouette is low and flat",
                  "(~0.035 at every k) — the CE text does not form well-separated clusters, so k = 8 (dashed) is a",
                  "readability choice for the scatter, not a natural optimum."), 104),
         x = "Number of clusters (k)", y = NULL) +
    theme_catf()
  save_fig(p_elbow, "fig_d6_ce_elbow.png", w = 9, h = 3.6)
}

# Fig (A3.2): UpSet of the cross-agency CE 'twin' families — which DEPARTMENTS share a near-identical
# CE (the precedent for adopt). Mirrors the Phase 1 geothermal UpSet (ggupset, list-column of sets).
clusters_file <- file.path(ANALYSIS, "ce_clusters.parquet")
if (file.exists(clusters_file) && requireNamespace("ggupset", quietly = TRUE)) {
  xa <- read_parquet(clusters_file) %>% filter(n_agencies >= 2)
  xa$depts <- lapply(xa$agencies, function(a) {
    toks <- trimws(strsplit(as.character(a), ",")[[1]])
    d <- unique(trimws(sub("\\s*-.*$", "", toks)))           # department = the prefix before " - "
    sort(d[d != ""])
  })
  xa <- xa[vapply(xa$depts, length, integer(1)) >= 1, ]
  p_upset <- ggplot(xa, aes(x = depts)) +
    geom_bar(fill = catf_navy, width = 0.62) +
    stat_count(geom = "text", aes(label = after_stat(count)), vjust = -0.45, size = 3,
               fontface = "bold", color = catf_navy) +
    ggupset::scale_x_upset(n_intersections = 12) +
    ggupset::theme_combmatrix(combmatrix.panel.point.color.fill = catf_dark_blue,
                              combmatrix.panel.point.color.empty = "gray85",
                              combmatrix.panel.line.color = "gray70",
                              combmatrix.label.text = element_text(size = 9, color = catf_navy)) +
    labs(title = glue::glue("Which departments share the same CE 'twin' families ({nrow(xa)} cross-agency families)"),
         subtitle = str_wrap(paste("Each bar = the number of near-identical CE families shared by exactly that set of",
                  "departments (top 12 combinations); a filled dot below marks the departments in the combination."), 104),
         x = NULL, y = "Twin families") +
    theme(plot.title = element_text(face = "bold", color = catf_navy, size = 13),
          plot.subtitle = element_text(color = catf_dark_blue),
          axis.title.y = element_text(color = catf_navy),
          panel.grid.major.x = element_blank())
  save_fig(p_upset, "fig_d6_ce_upset.png", w = 10, h = 6)
}

# === Analysis 2: corpus-wide mitigation (read the enrichment — NOT limited to candidates) ===
enr <- read_parquet(file.path(ANALYSIS, "fonsi_enrichment.parquet")) %>%
  filter(!is.na(action_summary)) %>%
  mutate(is_mit = is_mitigated_fonsi %in% TRUE)
n_enr <- nrow(enr); n_mit <- sum(enr$is_mit)

# Fig 11: waffle of the whole corpus by mitigation status (mitigated / not / unknown) — matches the
# data-at-a-glance waffle. Null is_mitigated_fonsi is shown as its own 'Unknown', never lumped in.
n_notmit <- sum(enr$is_mitigated_fonsi %in% FALSE)
n_unk    <- sum(is.na(enr$is_mitigated_fonsi))
ordm <- c("Unknown", "Not mitigated", "Mitigated")          # top -> bottom in grid + legend
cntm <- c(Unknown = n_unk, `Not mitigated` = n_notmit, Mitigated = n_mit)[ordm]
# largest-remainder allocation so the squares sum to EXACTLY 100 (no orphan square in an 11th row)
.lr100 <- function(counts) {
  raw <- counts / sum(counts) * 100; fl <- floor(raw); rem <- 100 - sum(fl)
  if (rem > 0) { o <- order(raw - fl, decreasing = TRUE); fl[o[seq_len(rem)]] <- fl[o[seq_len(rem)]] + 1 }
  fl
}
wvm  <- .lr100(cntm)
wafm <- tibble(cat = factor(rep(ordm, wvm), levels = ordm)) %>%
  mutate(i = row_number() - 1, x = i %% 10, y = i %/% 10)
palm <- c(Mitigated = catf_dark_blue, `Not mitigated` = catf_grey, Unknown = "#D9DCE1")
labm <- wafm %>% group_by(cat) %>% summarise(y = mean(y), .groups = "drop") %>%
  mutate(lab = paste0(cntm[as.character(cat)], " (", round(cntm[as.character(cat)] / n_enr * 100), "%)"))
p_ov <- ggplot(wafm, aes(x, y, fill = cat)) +
  geom_tile(color = "white", linewidth = 1.6) +
  geom_text(data = labm, aes(x = -0.9, y = y, label = lab), inherit.aes = FALSE,
            hjust = 1, fontface = "bold", size = 3.6, color = catf_navy) +
  scale_fill_manual(values = palm, name = NULL) +
  scale_x_continuous(expand = expansion(add = c(2.8, 0.2))) +
  coord_equal(clip = "off") + scale_y_reverse() +
  labs(title = glue::glue("{n_mit} of {n_enr} decarbonization FONSIs are 'mitigated' ({percent(n_mit/n_enr,1)})"),
       subtitle = str_wrap("A 'mitigated FONSI' reaches no-significant-impact only because the applicant committed to mitigation; 'Unknown' = the read did not say", 92),
       caption = glue::glue("Each square ≈ {round(n_enr / 100, 1)} FONSIs.")) +
  theme_void(base_size = 12) +
  theme(legend.position = "right", plot.title = element_text(face = "bold", color = catf_navy),
        plot.subtitle = element_text(color = catf_dark_blue), plot.caption = element_text(color = "gray50", hjust = 0))
save_fig(p_ov, "fig_d6_mitigated_overall.png", w = 9, h = 4.6)

# Fig 12: action-type breakdown of all 451 — a horizontal bar (bar length = # FONSIs = the breakdown),
# shaded by mitigated-FONSI share, labeled with count + % of total + mitigated %. Legible at every size.
share <- enr %>% group_by(action_category) %>%
  summarise(n = n(), mit = sum(is_mit), .groups = "drop") %>%
  mutate(share = mit / n, pct = n / sum(n),
         lab = str_to_title(str_replace_all(action_category, "_", " ")),
         lab = recode(lab, "Temporary Resource Assessment" = "Temporary assessment"))
p_share <- ggplot(share, aes(x = n, y = reorder(lab, n), fill = share)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = paste0(n, "  (", percent(pct, 1), " of total) · ", percent(share, 1), " mitigated")),
            hjust = -0.04, size = 3.1, color = catf_navy) +
  scale_fill_gradient(low = catf_light_blue, high = catf_navy, limits = c(0, 1), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.5))) +
  labs(title = "How the 451 decarbonization FONSIs break down by action type",
       subtitle = "Bar length = number of FONSIs; darker = higher mitigated-FONSI share (incl. the large 'Other' pool)",
       x = "Number of FONSIs", y = NULL,
       caption = "A CE must encode the recurring mitigations as design criteria — it cannot rely on case-by-case commitments.") +
  theme_catf() + theme(legend.position = "none")
save_fig(p_share, "fig_d6_mitigated_share.png", w = 10, h = 4.2)

# Fig: PHRASE cloud (2-3 grams) of committed-mitigation language — surfaces measures, not generic words
ng_stop <- unique(c(tidytext::stop_words$word, letters,
            "project","projects","mitigation","measures","measure","impacts","impact","action","proposed","applicant",
            "construction","area","areas","resources","resource","plan","plans","sites","federal","state","local",
            "appropriate","implement","implemented","minimize","reduce","reducing","avoid","potential","including",
            "activities","management","require","required","ensure","provide","within","prior","conducted","completed"))
mit_df <- enr %>% filter(is_mit, !is.na(mitigation_summary)) %>%
  transmute(doc = row_number(), text = str_replace_all(tolower(mitigation_summary), "[^a-z ]", " "))
phrases <- bind_rows(
    tidytext::unnest_tokens(mit_df, ngram, text, token = "ngrams", n = 2),
    tidytext::unnest_tokens(mit_df, ngram, text, token = "ngrams", n = 3)) %>%
  filter(!is.na(ngram), ngram != "") %>%
  tidyr::separate(ngram, into = c("w1", "w2", "w3"), sep = " ", fill = "right", remove = FALSE) %>%
  mutate(wl = if_else(is.na(w3) | w3 == "", w2, w3)) %>%
  filter(!w1 %in% ng_stop, !wl %in% ng_stop, nchar(w1) >= 3, nchar(wl) >= 3,
         !str_detect(ngram, "mitigat|measure")) %>%   # phrase bounded by content words; drop framing words (item 2)
  count(ngram, sort = TRUE) %>% filter(n >= 2)
# collapse plural / substring variants so one recurring entity (e.g. 'desert tortoise') doesn't fill
# the cloud as 'desert tortoise' + 'desert tortoises' + 'desert tortoise habitat' — keep the top form
.kept <- character(0)
for (.g in phrases$ngram) {
  .gn <- gsub("s\\b", "", .g)
  if (!any(vapply(.kept, function(k) { .kn <- gsub("s\\b", "", k)
        grepl(.gn, .kn, fixed = TRUE) || grepl(.kn, .gn, fixed = TRUE) }, logical(1))))
    .kept <- c(.kept, .g)
}
phrases <- phrases %>% filter(ngram %in% .kept) %>% slice_head(n = 40)
set.seed(6)
p_wc <- ggplot(phrases, aes(label = ngram, size = n, color = n)) +
  geom_text_wordcloud_area(shape = "square", rm_outside = TRUE, area_corr = TRUE, eccentricity = 0.65) +
  scale_size_area(max_size = 26) +
  scale_color_gradient(low = catf_light_blue, high = catf_navy) +
  labs(title = "The committed-mitigation language is project-specific",
       subtitle = str_wrap(glue::glue("Most-frequent 2–3 word phrases across the {n_mit} mitigated FONSIs' mitigation ",
                             "summaries — no phrase dominates, consistent with case-specific (not standardized) measures"), 95)) +
  theme_void(base_size = 12) +
  theme(plot.title = element_text(face = "bold", color = catf_navy),
        plot.subtitle = element_text(color = catf_dark_blue, margin = margin(b = 2)),
        plot.margin = margin(1, 1, 1, 1), legend.position = "none")
save_fig(p_wc, "fig_d6_mitigation_wordcloud.png", w = 8, h = 4.6)

# Fig: the recurring significance THRESHOLDS — what agencies said WOULD make an impact significant
parse_sig <- function(j) {
  x <- tryCatch(jsonlite::fromJSON(j, simplifyVector = FALSE), error = function(e) list())
  vapply(x, function(t) if (is.null(t$statement)) "" else t$statement, character(1))
}
sig_stmts <- enr %>% filter(!is.na(significance_thresholds), significance_thresholds != "[]") %>%
  pull(significance_thresholds) %>% lapply(parse_sig) %>% unlist()
sig_cond <- sig_stmts[nchar(sig_stmts) > 40 &
  str_detect(tolower(sig_stmts), "significant if|would be|would not|unless|exceed|result in|loss of|greater than|contaminat|degrad")]
sig_df <- tibble(doc = seq_along(sig_cond), text = str_replace_all(tolower(sig_cond), "[^a-z ]", " "))
sig_phr <- bind_rows(
    tidytext::unnest_tokens(sig_df, ngram, text, token = "ngrams", n = 2),
    tidytext::unnest_tokens(sig_df, ngram, text, token = "ngrams", n = 3)) %>%
  filter(!is.na(ngram), ngram != "") %>%
  tidyr::separate(ngram, into = c("w1", "w2", "w3"), sep = " ", fill = "right", remove = FALSE) %>%
  mutate(wl = if_else(is.na(w3) | w3 == "", w2, w3)) %>%
  filter(!w1 %in% ng_stop, !wl %in% ng_stop, nchar(w1) >= 4, nchar(wl) >= 4) %>%
  count(ngram, sort = TRUE) %>% slice_head(n = 14)
p_sig <- ggplot(sig_phr, aes(n, reorder(ngram, n))) +
  geom_col(fill = catf_dark_blue, width = 0.7) +
  geom_text(aes(label = n), hjust = -0.3, size = 3.2, color = catf_navy, fontface = "bold") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(title = "But the significance thresholds recur — that is the codifiable part",
       subtitle = str_wrap(paste("Most-common phrases in the agencies' explicit significance-threshold statements",
                "(\"would be significant if ...\") — unlike the mitigations, these conditions repeat across projects"), 95),
       x = "Times the phrase appears across mitigated FONSIs", y = NULL,
       caption = "From the enrichment's significance_thresholds; the recurring conditions are the natural CE bounds.") +
  theme_catf()
save_fig(p_sig, "fig_d6_significance_thresholds.png", w = 9, h = 4.2)

message("[08] figures written to ", FIGS)
