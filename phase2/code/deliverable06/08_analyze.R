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

# === Fig: outcomes — one stacked bar of the universe (452) -> recurring (293) -> adopt (53) ===
n_broader <- n_candidate - n_ce_shaped          # in a recurring type but not bounded
n_uncat   <- n_clean - n_candidate              # not in any recurring type (net-new pool)
comp <- tibble(
  segment = factor(c("Bounded", "Recurring", "Not Recurring"),
                   levels = c("Not Recurring", "Recurring", "Bounded")),
  n = c(n_ce_shaped, n_broader, n_uncat),
  fill = c(catf_navy, catf_light_blue, catf_grey)
)
p_out <- ggplot(comp, aes(x = glue::glue("All decarbonization\nFONSIs (n = {comma(n_clean)})"),
                          y = n, fill = segment)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(n, "\n(", percent(n / n_clean, 1), ")")),
            position = position_stack(vjust = 0.5), size = 4, color = "white", fontface = "bold", lineheight = 0.85) +
  scale_fill_manual(values = setNames(comp$fill, comp$segment), name = NULL, guide = guide_legend(reverse = TRUE)) +
  coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.02))) +
  labs(title = "Scaling CEs: where the decarbonization FONSIs fall",
       subtitle = str_wrap(glue::glue("Of the {comma(n_clean)} FONSIs, {n_candidate} recur ",
                 "({n_broader} broad + {n_ce_shaped} bounded) while {n_uncat} are uncategorized"), 95),
       x = NULL, y = "Distinct FONSI projects") +
  theme_catf() + theme(legend.position = "bottom",
                       axis.text.y = element_text(face = "bold", color = catf_navy)) +
  guides(fill = guide_legend(nrow = 1, reverse = TRUE))
save_fig(p_out, "fig_d6_outcomes.png", w = 10, h = 3.0)

# === Fig: sort step — every clean FONSI by action type, incl. the uncategorized pool ===
dd <- corp_fonsi %>% group_by(candidate_category) %>%
  summarise(total = n_distinct(project_id),
            bounded = n_distinct(project_id[is_profile_subtype]), .groups = "drop") %>%
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
  scale_fill_manual(values = c("Broader" = catf_grey, "Bounded" = catf_teal), name = NULL,
                    guide = guide_legend(reverse = TRUE)) +
  scale_x_continuous(labels = percent, expand = expansion(mult = c(0, 0.14))) +
  labs(title = glue::glue("Step 1–2 · Sorting the {n_clean} decarbonization FONSIs"),
       subtitle = str_wrap(glue::glue("Within each action type, the share that is bounded & low-impact (teal, kept for ",
                 "matching) vs broader (grey, set aside). A further {n_uncat} FONSIs are uncategorized."), 96),
       x = "Share of the action type", y = NULL,
       caption = "Teal = bounded, low-impact subset carried to Step 3; grey = broader. n = total FONSIs of that type.") +
  theme_catf() + theme(legend.position = "bottom")
save_fig(p_sort, "fig_d6_action_distribution.png", w = 9, h = 4.2)

# === Fig: CE-match strength per candidate (ranking aid; 0.40 cutoff) ===
mfit <- verdicts %>% filter(verdict != "contrast") %>% mutate(lab = short_label(candidate_label))
p_match <- ggplot(mfit, aes(reorder(lab, best_ce_match_score), best_ce_match_score)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = 0.20, fill = catf_grey, alpha = 0.5) +
  annotate("text", x = Inf, y = 0.10, label = "baseline similarity", hjust = 0.5, vjust = -0.7,
           size = 2.9, color = "gray30", fontface = "italic") +
  geom_col(width = 0.6, fill = catf_teal) +
  geom_text(aes(label = sprintf("%.2f", best_ce_match_score)), hjust = -0.3, size = 3.5,
            fontface = "bold", color = catf_navy) +
  coord_flip(clip = "off") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Step 3 · How strongly each action matches an existing CE",
       subtitle = "Closest existing CE by text similarity (0–1)",
       x = NULL, y = "Best-match similarity (0–1)",
       caption = str_wrap(paste("Blended semantic + word-overlap similarity. Grey band = baseline, where unrelated CEs score",
                "(≤ ~0.20); the matches sit 2–6× above it. A ranking aid — every match is confirmed against its eCFR text (see table above)."), 118)) +
  theme_catf()
save_fig(p_match, "fig_d6_ce_match.png", h = 4.0)

# === Fig: size spread of the bounded FONSIs (the candidate CE bounds) ===
sz <- facts %>% filter(is_profile_subtype) %>%
  transmute(candidate_category,
            `Transmission — line miles` = ifelse(candidate_category == "transmission_upgrade", max_miles, NA),
            `Transmission — voltage (kV)` = ifelse(candidate_category == "transmission_upgrade", max_kilovolts, NA),
            `Geothermal — wells drilled` = ifelse(candidate_category == "geothermal_exploration", n_wells, NA)) %>%
  pivot_longer(-candidate_category, names_to = "metric", values_to = "value") %>%
  filter(!is.na(value), value > 0, value < 1000) %>%   # drop study-area outliers
  group_by(metric) %>% mutate(metric_n = paste0(metric, "\n(n = ", n(), ")")) %>% ungroup()
p_sizes <- ggplot(sz, aes(x = "", value)) +
  geom_boxplot(width = 0.55, fill = catf_light_blue, color = catf_navy, alpha = 0.2,
               linewidth = 0.5, outlier.shape = NA) +
  geom_beeswarm(cex = 3, size = 2.3, color = catf_navy, alpha = 0.7) +
  facet_wrap(~metric_n, scales = "free_y", ncol = 3) +
  scale_x_discrete(expand = expansion(add = 0.7)) +
  labs(title = "Step 3 · The size range of the bounded FONSIs",
       subtitle = str_wrap("What the bounded projects actually measure — the observed range a CE's threshold could be set against", 90),
       x = NULL, y = NULL,
       caption = str_wrap(paste("Bounded, low-impact subset only. Each dot is one FONSI (n per panel);",
                "box = median & middle 50%, whiskers = range. Study-area outliers excluded."), 110)) +
  theme_catf() + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
                       strip.text = element_text(color = catf_navy, face = "bold"),
                       panel.spacing = unit(1.8, "lines"))
save_fig(p_sizes, "fig_d6_sizes.png", w = 9.5, h = 3.8)

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
  scale_fill_manual(values = c("#012169", "#28518F", "#4E7CB5", "#3FA7A0", "#8AB7E9", "#C2CBD6"), name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Step 4 · How each candidate is scored and ranked",
       subtitle = "Transparent multi-factor rank score (0–1); bar length = total, colors = each factor's contribution",
       x = NULL, y = "Rank score") +
  theme_catf() + theme(legend.position = "bottom") + guides(fill = guide_legend(nrow = 2))
save_fig(p_cls, "fig_d6_classification.png", w = 9, h = 4.4)

# === Fig: FRA timeline — the 53 bounded FONSIs by D4 decision date ===
tl <- facts %>% filter(is_profile_subtype) %>% distinct(project_id, decision_date) %>%
  mutate(d = suppressWarnings(as.Date(decision_date)),
         year = as.integer(format(d, "%Y")),
         year = ifelse(!is.na(year) & year >= 1995 & year <= 2026, year, NA_integer_),
         era = case_when(is.na(year) ~ "Undated", d < as.Date("2023-06-03") ~ "Pre-FRA", TRUE ~ "Post-FRA"))
n_pre <- sum(tl$era == "Pre-FRA"); n_post <- sum(tl$era == "Post-FRA"); n_unk <- sum(tl$era == "Undated")
tld <- tl %>% filter(!is.na(year))
p_tl <- ggplot(tld, aes(x = year, fill = era)) +
  geom_dotplot(binwidth = 1, method = "histodot", dotsize = 0.78, stackratio = 1.1,
               color = "white", stackgroups = TRUE, binpositions = "all") +
  geom_vline(xintercept = 2023.42, linetype = "dashed", color = catf_magenta, linewidth = 0.8) +
  annotate("text", x = 2023.0, y = 0.97, label = "FRA enacted\nJun 2023", hjust = 1.05, vjust = 1,
           size = 3.2, color = catf_magenta, fontface = "bold", lineheight = 0.9) +
  scale_fill_manual(values = c("Pre-FRA" = catf_teal, "Post-FRA" = catf_magenta, "Undated" = catf_grey), name = NULL) +
  scale_x_continuous(limits = c(2002, 2026), breaks = seq(2004, 2026, 4)) +
  scale_y_continuous(NULL, breaks = NULL) +
  labs(title = "When were these EAs decided?",
       subtitle = str_wrap(glue::glue("The {n_ce_shaped} bounded FONSIs by decision year — each dot is one FONSI ",
                 "({n_pre} pre-FRA, {n_post} post-FRA, {n_unk} undated)"), 95),
       x = NULL,
       caption = str_wrap(glue::glue("Decision dates merged from the D4 timeline, known for {n_pre + n_post}/{n_ce_shaped}. ",
                 "FRA (Jun 2023) gave agencies authority to adopt another agency's CE."), 110)) +
  theme_catf() + theme(legend.position = "bottom")
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

# === Analysis 3: roll up to DEPARTMENT (the agency_unit prefix, e.g. "DOI - BLM" -> DOI) ===
ce_land <- ce_land %>% mutate(dept = str_trim(str_extract(agency_unit, "^[^-]+")),
                              dept = ifelse(is.na(dept) | dept == "", "Other", dept))
dept_pal <- c(DOI = "#012169", DOD = "#0047BB", USDA = "#00AE8D", DOT = "#93D500", DHS = "#C22A90",
              DOE = "#8AB7E9", DOC = "#E8A33D", HHS = "#6A4C93", Other = "#9AA0AA")
grp <- function(d) ifelse(d %in% names(dept_pal), d, "Other")
n_dept <- n_distinct(ce_land$dept)

# Fig: CEs by department (rolled up — the most complete view; placed first)
deptc <- ce_land %>% count(dept, sort = TRUE) %>% slice_head(n = 14) %>% mutate(g = grp(dept))
p_dept <- ggplot(deptc, aes(reorder(dept, n), n, fill = g)) +
  geom_col(width = 0.72) +
  geom_text(aes(label = n), hjust = -0.2, size = 3.4, fontface = "bold", color = catf_navy) +
  scale_fill_manual(values = dept_pal, guide = "none") +
  coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "The existing CE landscape, by department",
       subtitle = str_wrap(glue::glue("{comma(nrow(ce_land))} categorical exclusions rolled up across {n_dept} departments / independent agencies"), 90),
       x = NULL, y = "Categorical exclusions") +
  theme_catf()
save_fig(p_dept, "fig_d6_ce_by_dept.png", w = 9, h = 4.4)

# Fig: top 20 agencies, colored by department
agc <- ce_land %>% filter(!is.na(agency_name), agency_name != "") %>%
  count(agency_name, dept, sort = TRUE) %>% slice_head(n = 20) %>% mutate(g = grp(dept))
p_agc <- ggplot(agc, aes(reorder(agency_name, n), n, fill = g)) +
  geom_col(width = 0.74) +
  geom_text(aes(label = n), hjust = -0.25, size = 2.9, color = catf_navy) +
  scale_fill_manual(values = dept_pal, name = "Department") +
  coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Top 20 agencies by number of categorical exclusions",
       subtitle = "Bars colored by department (rolled up from the agency unit code)",
       x = NULL, y = "Categorical exclusions",
       caption = "Source: CE Explorer export.") +
  theme_catf() + theme(legend.position = "right")
save_fig(p_agc, "fig_d6_ce_by_agency.png", w = 9.5, h = 5.5)

# Fig: distribution of stated numeric bounds in existing CEs (acres + miles; kV/MW too rare)
n_any_bound <- sum(ce_land$states_any_bound, na.rm = TRUE)
bnd <- ce_land %>% transmute(`Acreage limit (acres)` = bound_acres, `Length limit (miles)` = bound_miles) %>%
  pivot_longer(everything(), names_to = "metric", values_to = "value") %>% filter(!is.na(value)) %>%
  group_by(metric) %>% mutate(metric_n = paste0(metric, "\n(n = ", n(), ")")) %>% ungroup()
set.seed(6)
p_bnd <- ggplot(bnd, aes(x = "", value)) +
  geom_boxplot(width = 0.16, fill = catf_light_blue, color = catf_navy, alpha = 0.25, outlier.shape = NA) +
  geom_jitter(width = 0.32, height = 0, size = 2.2, color = catf_navy, alpha = 0.6) +
  facet_wrap(~metric_n, scales = "free_y") + scale_x_discrete(expand = expansion(add = 0.6)) +
  labs(title = "Numeric bounds in existing CEs, where stated",
       subtitle = str_wrap(glue::glue("Only {n_any_bound} of {comma(nrow(ce_land))} CEs state any numeric limit — acres and miles dominate (kV/MW appear only a handful of times)"), 92),
       x = NULL, y = NULL,
       caption = "Each dot is one CE's stated limit; box = median & middle 50%, whiskers = range.") +
  theme_catf() + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
                       strip.text = element_text(color = catf_navy, face = "bold"), panel.spacing = unit(1.8, "lines"))
save_fig(p_bnd, "fig_d6_ce_bounds.png", w = 8, h = 3.6)

# Fig: relatedness map — every CE projected to 2D (PCA of text embeddings), colored by department
if ("coord_x" %in% names(ce_land) && any(!is.na(ce_land$coord_x))) {
  sc <- ce_land %>% filter(!is.na(coord_x)) %>% mutate(g = grp(dept))
  p_scatter <- ggplot(sc, aes(coord_x, coord_y, color = g)) +
    geom_point(size = 1.5, alpha = 0.55) +
    scale_color_manual(values = dept_pal, name = "Department") +
    labs(title = "How related are the existing CEs?",
         subtitle = str_wrap(paste("Each point is one CE, placed by the similarity of its text (2D PCA of embeddings);",
                  "tight groups are near-identical CEs, often shared across agencies"), 92),
         x = NULL, y = NULL,
         caption = "Proximity = textual similarity. Cross-department overlap is the precedent for adopt.") +
    theme_catf() +
    theme(legend.position = "right", axis.text = element_blank(), panel.grid = element_blank()) +
    guides(color = guide_legend(override.aes = list(size = 3)))
  save_fig(p_scatter, "fig_d6_ce_scatter.png", w = 9, h = 5.5)
}

# === Analysis 2: corpus-wide mitigation (read the enrichment — NOT limited to candidates) ===
enr <- read_parquet(file.path(ANALYSIS, "fonsi_enrichment.parquet")) %>%
  filter(!is.na(action_summary)) %>%
  mutate(is_mit = is_mitigated_fonsi %in% TRUE)
n_enr <- nrow(enr); n_mit <- sum(enr$is_mit)

# Fig: how many of the whole corpus are mitigated (stacked bar over ALL FONSIs)
ov <- tibble(segment = factor(c("Mitigated FONSI", "Not mitigated (inherently low-impact)"),
                              levels = c("Not mitigated (inherently low-impact)", "Mitigated FONSI")),
             n = c(n_mit, n_enr - n_mit))
p_ov <- ggplot(ov, aes(x = "", y = n, fill = segment)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = n), position = position_stack(vjust = 0.5), color = "white", fontface = "bold", size = 5) +
  scale_fill_manual(values = c("Mitigated FONSI" = catf_dark_blue,
                               "Not mitigated (inherently low-impact)" = catf_grey), name = NULL,
                    guide = guide_legend(reverse = TRUE)) +
  coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.04))) +
  labs(title = glue::glue("{n_mit} of {n_enr} decarbonization FONSIs are 'mitigated' ({percent(n_mit/n_enr,1)})"),
       subtitle = "A 'mitigated FONSI' reaches no-significant-impact only because the applicant committed to mitigation",
       x = NULL, y = "Decarbonization FONSIs") +
  theme_catf() + theme(legend.position = "bottom", axis.text.y = element_blank(), axis.ticks.y = element_blank())
save_fig(p_ov, "fig_d6_mitigated_overall.png", w = 10, h = 2.6)

# Fig: mitigated share by action type — ALL 451, including the 'Other' pool (not just candidates)
share <- enr %>% group_by(action_category) %>%
  summarise(n = n(), mit = sum(is_mit), .groups = "drop") %>%
  mutate(share = mit / n, lab = str_to_title(str_replace_all(action_category, "_", " "))) %>%
  filter(n >= 3)
p_share <- ggplot(share, aes(reorder(lab, share), share)) +
  geom_col(width = 0.66, fill = catf_dark_blue) +
  geom_text(aes(label = paste0(percent(share, 1), "  (", mit, "/", n, ")")), hjust = -0.08, size = 3.4, color = catf_navy) +
  coord_flip() + scale_y_continuous(labels = percent, limits = c(0, 1.18), breaks = seq(0, 1, .25),
                                    expand = expansion(mult = c(0, 0))) +
  labs(title = "Mitigated-FONSI share by action type (all 451 FONSIs)",
       subtitle = "Not limited to the Analysis-1 candidates — the 'Other' pool (61% mitigated) is included",
       x = NULL, y = "Mitigated-FONSI share",
       caption = "A CE must encode the recurring mitigations as design criteria — it cannot rely on case-by-case commitments.") +
  theme_catf()
save_fig(p_share, "fig_d6_mitigated_share.png", h = 4.0)

# Fig: word cloud of the committed-mitigation language (shows how project-specific it is)
stop_w <- c(letters, "the","and","for","with","would","that","this","are","all","any","will","from","not","its",
            "during","including","include","includes","such","other","which","been","were","has","have","also",
            "project","projects","mitigation","measures","measure","impacts","impact","action","proposed","applicant",
            "construction","area","areas","resources","resource","plan","plans","sites","federal","state","local",
            "use","used","using","appropriate","implement","implemented","minimize","reduce","avoid","potential",
            "activities","management","require","required","ensure","provide","within","prior","specific","including",
            "associated","through","under","conducted","monitoring","best","practices","standard","compliance")
words <- enr %>% filter(is_mit, !is.na(mitigation_summary)) %>% pull(mitigation_summary) %>%
  paste(collapse = " ") %>% tolower() %>% str_extract_all("[a-z]{4,}") %>% unlist()
wf <- tibble(word = words) %>% filter(!word %in% stop_w) %>% count(word, sort = TRUE) %>% slice_head(n = 130)
set.seed(6)
p_wc <- ggplot(wf, aes(label = word, size = n, color = n)) +
  geom_text_wordcloud_area(rm_outside = TRUE, eccentricity = 1) +
  scale_size_area(max_size = 13) +
  scale_color_gradient(low = catf_light_blue, high = catf_navy) +
  labs(title = "The committed-mitigation language is project-specific",
       subtitle = glue::glue("Most-frequent words across the {n_mit} mitigated FONSIs' mitigation summaries — no term dominates, ",
                             "consistent with case-specific (not standardized) measures")) +
  theme_catf() + theme(panel.grid = element_blank())
save_fig(p_wc, "fig_d6_mitigation_wordcloud.png", w = 9, h = 5.5)

message("[08] figures written to ", FIGS)
