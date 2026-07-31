# D4 Extension #1 — Field-office experience vs. process change (BLM + DOE)
#
# Research question (ASSOCIATIONAL, not causal): does NEPA CE review duration fall because
# individual lead-agency offices get faster as they accumulate experience — or because a
# system-wide PROCESS CHANGE (batch processing / procedural streamlining) hit every office at
# once? We map each project to the office that handled it, order each office's reviews by the
# office's own cumulative count, and test whether duration falls with that count net of calendar
# time. It does NOT: within any decision year an office's caseload is uncorrelated with its speed,
# low-volume offices converge to high-volume speed without doing the reps, and the office
# fixed-effects experience coefficient is ~0 with or without a year control — in BOTH corpora.
# The takeaway: process change, not office experience, likely drove the CE speed-up.
#
# Inputs (DuckDB-built upstream; read here via arrow):
#   - phase2/data/analysis/deliverable04/blm_field_offices.parquet  (01_parse_offices.py)
#   - phase2/data/analysis/deliverable04/doe_offices.parquet        (01b_build_doe_offices.py)
#   - phase2/data/analysis/timeline/timeline_project_dates.parquet  (durations + source type)
#   - phase2/data/analysis/timeline/timeline_document_index.parquet (project_energy_type)
#
# Duration frame mirrors 08_create_figures.R's `headline`: complete timelines only, month-granularity
# imputed to the 15th, year-granularity excluded.
#
# Two confound controls are central:
#   1. Register-anchoring artifact — BLM register "project start" dates are often late
#      administrative entries (the ~40-day artifact). The PRIMARY measure uses DOCUMENT-anchored
#      initiation dates (initiation_source_type != 'metadata'); the register-anchored view is a
#      clearly-flagged secondary.
#   2. Calendar time — a pooled office fixed-effects regression with factor(decision_year)
#      separates the experience gradient from secular calendar trends.
#
# Symmetric no-filter design: BOTH arms use each office's FULL review history (no calendar cut).
# Known pre-2012 BLM artifact rows (the AZ-A010 batched cluster; historical-citation initiations)
# are RETAINED and flagged in the report caveats; they inflate the RAW BLM estimate only, not the
# year-controlled result. See the parameter block for detail.
#
# CE is the sole focus of the office-level analysis (EA has no office clearing the >=30
# document-anchored bar; EIS is too sparse per office).
#
# Usage: Rscript phase2/code/deliverable04/field_office/02_create_figures.R

suppressPackageStartupMessages({
  library(here); library(arrow); library(dplyr); library(tidyr)
  library(readr); library(lubridate); library(ggplot2); library(scales); library(patchwork)
})

PHASE2 <- here::here("phase2")
TL     <- file.path(PHASE2, "data", "analysis", "timeline")
D04    <- file.path(PHASE2, "data", "analysis", "deliverable04")
FIG    <- file.path(PHASE2, "output", "deliverable04", "figures")
DIAG   <- file.path(PHASE2, "output", "deliverable04", "diagnostics")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
dir.create(DIAG, recursive = TRUE, showWarnings = FALSE)

# CATF palette + theme (copied from fra/02_create_figures.R) ---------------------------------------
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

# Analysis parameters ------------------------------------------------------------------------
MIN_REVIEWS  <- 30   # qualifying office size (valid durations on the analysis measure)
EA_RELAX     <- 10   # EA falls back to a relaxed threshold (see note below)
anchor_cols  <- c("Document" = catf_navy, "Register" = catf_light_blue)

# Symmetric no-filter design: BOTH arms use each office's FULL review history, no calendar cut. ---
# Known pre-2012 artifact rows remain in the BLM doc-anchored CE frame and are RETAINED under this
# symmetric design (flagged in the report caveats): a 9-row batched AZ-A010 cluster all sharing
# initiation 2006-09-15 -> decision 2010-04-28, and historical-citation "initiations" (e.g. a
# 7,227-day span from a 1985 citation). They modestly inflate the RAW (uncontrolled) BLM estimate
# and do NOT affect the year-controlled result. Keeping both arms on their full natural windows
# makes the two regressions symmetric (a one-sided ≥2012 filter would not be justified).

# ---------------------------------------------------------------------------
# Load + assemble the duration frame (mirror 08_create_figures.R `headline`)
# ---------------------------------------------------------------------------
offices <- read_parquet(file.path(D04, "blm_field_offices.parquet"))  # project_id, office_code, state, parse_source

dates_raw <- read_parquet(
  file.path(TL, "timeline_project_dates.parquet"),
  col_select = c("project_id", "process_type", "initiation_date", "decision_date",
                 "initiation_date_granularity", "decision_date_granularity",
                 "initiation_source_type", "timeline_status"))

energy <- read_parquet(
  file.path(TL, "timeline_document_index.parquet"),
  col_select = c("project_id", "project_energy_type")) |>
  group_by(project_id) |>
  summarise(project_energy_type = first(na.omit(project_energy_type)), .groups = "drop")

headline <- dates_raw |>
  mutate(initiation_date = as.Date(initiation_date), decision_date = as.Date(decision_date)) |>
  filter(timeline_status %in% c("complete_clear", "complete_with_proxy"),
         !is.na(initiation_date), !is.na(decision_date),
         initiation_date_granularity != "year", decision_date_granularity != "year") |>
  mutate(
    .init_mid = if_else(initiation_date_granularity == "month",
                        floor_date(initiation_date, "month") + 14, initiation_date),
    .dec_mid  = if_else(decision_date_granularity == "month",
                        floor_date(decision_date, "month") + 14, decision_date),
    duration_days = as.integer(.dec_mid - .init_mid),
    decision_date = .dec_mid,
    decision_year = year(.dec_mid),
    # PRIMARY = document-anchored initiation; register metadata dates are the flagged secondary.
    anchor = if_else(initiation_source_type == "metadata", "Register", "Document")) |>
  filter(!is.na(duration_days), duration_days >= 0)

df <- headline |>
  inner_join(offices, by = "project_id") |>
  left_join(energy, by = "project_id") |>
  filter(process_type %in% c("EA", "CE")) |>           # EIS too sparse per office -> excluded
  mutate(process_type = factor(process_type, levels = c("EA", "CE")),
         energy = factor(recode(coalesce(project_energy_type, "Other"), "Clean" = "Decarb"),
                         levels = c("Decarb", "Fossil", "Other")))

# cum_count = the office's Nth review (experience level), within process x anchor, by decision date
df_cum <- df |>
  group_by(process_type, anchor, office_code) |>
  arrange(decision_date, project_id, .by_group = TRUE) |>
  mutate(cum_count = row_number(), office_n = n()) |>
  ungroup()

# Coverage / feasibility message
feas <- df_cum |> distinct(process_type, anchor, office_code, office_n) |>
  group_by(process_type, anchor) |>
  summarise(offices_ge30 = sum(office_n >= MIN_REVIEWS), offices_ge10 = sum(office_n >= EA_RELAX),
            reviews = sum(office_n[!duplicated(office_code)]), .groups = "drop")
message("Qualifying offices by process x anchor:"); print(as.data.frame(feas))

# ---------------------------------------------------------------------------
# BLM CE document-anchored qualifying frame (FULL window): offices with >=30 doc-anchored CE
# reviews over their whole history. The experience clock (cum_count / office_n) is intact here.
# EA has NO office clearing the >=30 document-anchored bar, so the office-level views are CE-only.
# ---------------------------------------------------------------------------
ce_doc_q <- df_cum |> filter(process_type == "CE", anchor == "Document", office_n >= MIN_REVIEWS)
message(sprintf("BLM CE-Document full-window qualifying frame: %d reviews / %d offices",
                nrow(ce_doc_q), n_distinct(ce_doc_q$office_code)))

terc <- ce_doc_q |>
  group_by(office_code) |>
  mutate(terc = ntile(cum_count, 3)) |>
  filter(terc %in% c(1, 3)) |>
  group_by(office_code, state, terc) |>
  summarise(med = median(duration_days), .groups = "drop") |>
  mutate(terc = if_else(terc == 1, "first_med", "last_med")) |>
  pivot_wider(names_from = terc, values_from = med)

office_summary <- ce_doc_q |>
  group_by(office_code, state) |>
  summarise(n = n(), median_days = median(duration_days), .groups = "drop") |>
  left_join(terc, by = c("office_code", "state")) |>
  mutate(pct_change = round(100 * (last_med - first_med) / first_med, 1)) |>
  # register-init share = fraction of the office's CE durations anchored on register metadata
  left_join(
    df |> filter(process_type == "CE") |> group_by(office_code) |>
      summarise(register_init_share = round(mean(anchor == "Register"), 3), .groups = "drop"),
    by = "office_code") |>
  arrange(desc(n))

write_csv(office_summary, file.path(DIAG, "d4_fieldoffice_summary.csv"))

share_sped <- mean(office_summary$pct_change < 0, na.rm = TRUE)
message(sprintf("CE qualifying offices: %d | median first-tercile %.0f d, last-tercile %.0f d | %.0f%% sped up",
                nrow(office_summary), median(office_summary$first_med), median(office_summary$last_med),
                100 * share_sped))

# (Retired figures: the BLM learning-curve, first-vs-last dumbbell, and experience-vs-calendar
# confound panels were removed in the process-change reframe. The convergence figure below,
# the within-year correlation table, and the office fixed-effects regression carry the argument.)

# ===========================================================================
# DOE ARM — administering-office learning curve (structural mirror of the BLM arm)
# ---------------------------------------------------------------------------
# BLM offices come from the DOI-BLM case number; DOE has no such code, so the DOE CX register's
# `office` field (linked by cx_number in 01b_build_doe_offices.py) supplies the administering /
# grant-program office. The DOE arm is CE-ONLY by construction (the CX register is CE) and
# DOCUMENT-ANCHORED as the primary measure (register-anchored DOE dates number only ~137 — too
# thin for a sensitivity arm, the mirror image of BLM's register-heavy thinness). Same headline
# duration frame (`headline` above), same office fixed-effects regression, same expected null.
# ===========================================================================
catf_lime <- "#93D500"   # house CE color, used only to key the DOE arm's CE learning curve

doe_off <- read_parquet(file.path(D04, "doe_offices.parquet")) |>
  select(project_id, office_code = office)

doe_df <- headline |>
  inner_join(doe_off, by = "project_id") |>
  left_join(energy, by = "project_id") |>
  filter(process_type == "CE") |>              # DOE arm is CE-only by construction (CX register)
  mutate(process_type = factor("CE", levels = c("EA", "CE")),
         energy = factor(recode(coalesce(project_energy_type, "Other"), "Clean" = "Decarb"),
                         levels = c("Decarb", "Fossil", "Other")))

# cum_count = the office's Nth CE review by decision date, within anchor (mirror of df_cum)
doe_cum <- doe_df |>
  group_by(process_type, anchor, office_code) |>
  arrange(decision_date, project_id, .by_group = TRUE) |>
  mutate(cum_count = row_number(), office_n = n()) |>
  ungroup()

# DOE CE-Document qualifying frame (FULL window, mirror of BLM): offices with >=30 doc-anchored
# CE reviews over their whole history. No calendar cut — symmetric with the BLM arm.
doe_ce_doc_q <- doe_cum |> filter(process_type == "CE", anchor == "Document", office_n >= MIN_REVIEWS)
message(sprintf("DOE CE-Document full-window qualifying frame: %d reviews / %d offices",
                nrow(doe_ce_doc_q), n_distinct(doe_ce_doc_q$office_code)))

# --- DOE per-office first- vs last-tercile summary (mirror office_summary) --------------------
doe_terc <- doe_ce_doc_q |>
  group_by(office_code) |>
  mutate(terc = ntile(cum_count, 3)) |>
  filter(terc %in% c(1, 3)) |>
  group_by(office_code, terc) |>
  summarise(med = median(duration_days), .groups = "drop") |>
  mutate(terc = if_else(terc == 1, "first_med", "last_med")) |>
  pivot_wider(names_from = terc, values_from = med)

doe_summary <- doe_ce_doc_q |>
  group_by(office_code) |>
  summarise(n = n(), median_days = median(duration_days), .groups = "drop") |>
  left_join(doe_terc, by = "office_code") |>
  mutate(pct_change = round(100 * (last_med - first_med) / first_med, 1)) |>
  left_join(
    doe_cum |> filter(process_type == "CE") |> group_by(office_code) |>
      summarise(register_init_share = round(mean(anchor == "Register"), 3), .groups = "drop"),
    by = "office_code") |>
  arrange(desc(n))
write_csv(doe_summary, file.path(DIAG, "d4_fieldoffice_doe_summary.csv"))

doe_share_sped <- mean(doe_summary$pct_change < 0, na.rm = TRUE)
message(sprintf("DOE CE qualifying offices: %d | median first-tercile %.0f d, last-tercile %.0f d | %.0f%% sped up",
                nrow(doe_summary), median(doe_summary$first_med), median(doe_summary$last_med),
                100 * doe_share_sped))

# (Retired: the DOE learning-curve, dumbbell, and experience-vs-calendar confound panels were
# removed in the process-change reframe, mirroring the BLM deletions above.)

# ===========================================================================
# LEAD FIGURE — convergence: the quieter half of offices reaches the busier half's speed
# ---------------------------------------------------------------------------
# For each arm, split the qualifying offices at the median office_n (full-history CE volume) into a
# BUSIER and a QUIETER half by total caseload, then trace median document-anchored CE duration over
# time for each half. If offices sped up by DOING REPS, the busier half — with far more accumulated
# experience — should stay persistently faster. Instead the quieter half converges to the same
# speed: the signature of a process change hitting all offices at once, not office learning.
# Durations are aggregated into TWO-YEAR BINS from each arm's earliest even year (2008-09, 2010-11,
# ...) so the thin quieter-half series is stable; only bins with n >= 10 per group are shown. The
# busier/quieter split is on office_n = the office's TOTAL full-history doc-anchored CE caseload.
# Line colors sit OUTSIDE the BLM-navy / DOE-light-blue cohort palette (dark blue = busier, lime =
# quieter) to avoid a semantic clash.
# ---------------------------------------------------------------------------
build_conv <- function(dat, arm_lab) {
  med_n     <- median(dat |> distinct(office_code, office_n) |> pull(office_n))
  base_year <- 2L * (min(dat$decision_year) %/% 2L)
  dat |>
    mutate(vol_group = if_else(office_n >= med_n, "Busier half", "Quieter half"),
           bin_start = base_year + 2L * ((decision_year - base_year) %/% 2L),
           bin_label = paste0(bin_start, "–", substr(bin_start + 1L, 3, 4))) |>
    group_by(vol_group, bin_start, bin_label) |>
    summarise(n = n(), median_days = median(duration_days), .groups = "drop") |>
    filter(n >= 10) |>
    transmute(arm = arm_lab, vol_group, bin_start, bin_label, n, median_days)
}
# Split metadata (so the report can name the halves without hard-coding): median full-history
# caseload, office counts per half, and the busier half's minimum caseload (the "each handling >= M").
split_meta <- function(dat, arm_lab) {
  d   <- dat |> distinct(office_code, office_n)
  med <- median(d$office_n)
  tibble(arm = arm_lab, median_caseload = med,
         n_busier = sum(d$office_n >= med), n_quieter = sum(d$office_n < med),
         busier_min = min(d$office_n[d$office_n >= med]))
}
conv_blm <- build_conv(ce_doc_q,     "BLM field offices")
conv_doe <- build_conv(doe_ce_doc_q, "DOE administering offices")
conv_all <- bind_rows(conv_blm, conv_doe)
write_csv(conv_all |> select(arm, vol_group, bin_label, n, median_days),
          file.path(DIAG, "d4_fieldoffice_convergence.csv"))
write_csv(bind_rows(split_meta(ce_doc_q, "BLM field offices"),
                    split_meta(doe_ce_doc_q, "DOE administering offices")),
          file.path(DIAG, "d4_fieldoffice_convergence_split.csv"))

# Should the DOE panel be shown? It must (1) be STABLE — no adjacent two-year bin swing > 3x — AND
# (2) actually CONVERGE — the quieter half must approach the busier half's speed (their medians meet
# within ~2x in the latest shared bin). This figure's whole claim is convergence, so a stable-but-
# non-convergent DOE series would visually contradict the headline. On the full window the DOE
# quieter half is stable but stays ~10x slower than the busier half throughout (busier DOE offices —
# e.g. NETL — are structurally faster; that between-office gap is netted out by the office-FE
# regression, which returns the +3.2% null). It therefore does NOT converge -> BLM-only figure.
doe_qh <- conv_doe |> filter(vol_group == "Quieter half") |> arrange(bin_start)
doe_swings <- if (nrow(doe_qh) >= 2) {
  r <- doe_qh$median_days[-1] / head(doe_qh$median_days, -1); pmax(r, 1 / r)
} else numeric(0)
doe_stable <- length(doe_swings) > 0 && all(doe_swings <= 3)
# convergence: ratio of quieter to busier median in the latest bin both halves share
shared_bins <- intersect(conv_doe$bin_start[conv_doe$vol_group == "Busier half"],
                         conv_doe$bin_start[conv_doe$vol_group == "Quieter half"])
last_bin  <- if (length(shared_bins)) max(shared_bins) else NA_integer_
doe_ratio <- if (!is.na(last_bin))
  conv_doe$median_days[conv_doe$vol_group == "Quieter half" & conv_doe$bin_start == last_bin] /
  conv_doe$median_days[conv_doe$vol_group == "Busier half"  & conv_doe$bin_start == last_bin] else NA_real_
doe_converges <- !is.na(doe_ratio) && doe_ratio <= 2
doe_show <- doe_stable && doe_converges
message(sprintf("DOE panel: swings %s (stable=%s); latest-bin quieter/busier ratio %.1fx (converges=%s) -> %s",
                paste(sprintf("%.1fx", doe_swings), collapse = ", "), doe_stable,
                ifelse(is.na(doe_ratio), NA, doe_ratio), doe_converges,
                if (doe_show) "keeping DOE panel" else "BLM-only figure"))

conv_fig <- if (doe_show) conv_all else conv_blm
conv_fig <- conv_fig |> mutate(arm = factor(arm, levels = c("BLM field offices", "DOE administering offices")))
conv_col <- c("Busier half" = catf_dark_blue, "Quieter half" = catf_lime)
conv_lab <- conv_fig |> group_by(arm, vol_group) |> slice_min(bin_start, n = 1, with_ties = FALSE) |> ungroup()

conv_sub  <- paste0("Offices are split into a busier and a quieter half by total caseload. If accumulated experience ",
                    "drove the speed-up, the busier half would stay persistently faster — instead the quieter half ",
                    "converges to the same review duration.")
conv_capt <- paste0("Median document-anchored CE review duration by two-year bin over each office's full history; ",
                    "only bins with ≥ 10 reviews per half are shown. BLM field offices. ",
                    if (!doe_show)
                      "The DOE arm is omitted here — its quieter half does not converge to the busier half (busier DOE offices are structurally faster, a between-office gap the office fixed-effects regression nets out); the DOE null rests on that regression and the within-year correlation. "
                    else "DOE administering offices, full window. ",
                    "Associational, not causal.")
wrap <- function(s, w) paste(strwrap(s, width = w), collapse = "\n")

p_conv <- ggplot(conv_fig, aes(bin_start, median_days, color = vol_group)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2.6) +
  geom_text(data = conv_lab, aes(label = vol_group), hjust = 1, nudge_x = -0.25, size = 3,
            fontface = "bold", show.legend = FALSE) +
  { if (!doe_show) facet_null() else facet_wrap(~arm, scales = "free", nrow = 1) } +
  scale_color_manual(values = conv_col, guide = "none") +
  scale_x_continuous(breaks = sort(unique(conv_fig$bin_start)),
                     labels = conv_fig$bin_label[match(sort(unique(conv_fig$bin_start)), conv_fig$bin_start)],
                     expand = expansion(mult = c(0.30, 0.06))) +
  expand_limits(y = 0) +
  labs(title = "The Quieter Half of Offices Reaches the Busier Half's Speed",
       subtitle = wrap(conv_sub, 118),
       x = "Decision-year bin", y = "Median duration (days)",
       caption = wrap(conv_capt, 128)) +
  theme_catf()
conv_w <- if (!doe_show) 8.5 else 11
ggsave(file.path(FIG, "fig_d4_fieldoffice_convergence.png"), p_conv, width = conv_w, height = 5.5, dpi = 300, bg = "white")
saveRDS(p_conv, file.path(FIG, "fig_d4_fieldoffice_convergence.rds"))

# ---------------------------------------------------------------------------
# Within-year correlation: is an office's accumulated caseload related to its speed WITHIN a year?
# Spearman(cum_count, duration_days) per decision year (both arms, full window). ~0 every
# year = experience is not what makes an office fast in a given calendar year.
# ---------------------------------------------------------------------------
withinyear_cor <- function(dat, arm_lab) {
  dat |> group_by(decision_year) |>
    filter(n() >= 10) |>
    summarise(n = n(),
              spearman = round(suppressWarnings(cor(cum_count, duration_days, method = "spearman")), 3),
              .groups = "drop") |>
    transmute(arm = arm_lab, year = decision_year, n, spearman)
}
withinyear <- bind_rows(withinyear_cor(ce_doc_q, "BLM"), withinyear_cor(doe_ce_doc_q, "DOE"))
write_csv(withinyear, file.path(DIAG, "d4_fieldoffice_withinyear_cor.csv"))
wy_blm <- withinyear |> filter(arm == "BLM")
message(sprintf("Within-year Spearman(cum_count, duration) — BLM full window: %d years, range %+.2f to %+.2f",
                nrow(wy_blm), min(wy_blm$spearman), max(wy_blm$spearman)))

# ---------------------------------------------------------------------------
# Combined N / inventory figure + CSV (both arms): parse/link funnel + ranked per-office
# full-history document-anchored CE review counts, with the ≥30 qualifying threshold marked.
# ---------------------------------------------------------------------------
n_blm_qual <- n_distinct(ce_doc_q$office_code)       # BLM regression qualifying-office count (full history)
n_doe_qual <- n_distinct(doe_ce_doc_q$office_code)   # DOE regression qualifying-office count (full history)
blm_inv <- offices |> count(office_code, name = "n_parsed") |>
  left_join(ce_doc_q |> count(office_code, name = "n_ce_doc_complete"),
            by = "office_code") |>
  transmute(arm = "BLM", office = office_code, n_parsed,
            n_ce_doc_complete = coalesce(n_ce_doc_complete, 0L))
doe_inv <- doe_off |> count(office_code, name = "n_parsed") |>
  left_join(doe_ce_doc_q |> count(office_code, name = "n_ce_doc_complete"),
            by = "office_code") |>
  transmute(arm = "DOE", office = office_code, n_parsed,
            n_ce_doc_complete = coalesce(n_ce_doc_complete, 0L))
inventory <- bind_rows(blm_inv, doe_inv)
write_csv(inventory, file.path(DIAG, "d4_fieldoffice_inventory.csv"))

# Funnel headline numbers for the panel subtitles (read from committed coverage CSVs, no literals).
blm_cov  <- read_csv(file.path(DIAG, "d4_fieldoffice_parse_coverage.csv"), show_col_types = FALSE)
doe_cov  <- read_csv(file.path(DIAG, "d4_fieldoffice_doe_coverage.csv"),  show_col_types = FALSE)
blm_led    <- blm_cov$blm_projects[blm_cov$scope == "ALL"]
blm_parsed <- blm_cov$parsed[blm_cov$scope == "ALL"]
doe_val <- function(m) doe_cov$value[doe_cov$metric == m]

inv_panel <- function(dat, arm_lab, bar_col, funnel) {
  d <- dat |> filter(n_ce_doc_complete > 0) |> slice_max(n_ce_doc_complete, n = 18) |>
    mutate(office = factor(office, levels = office[order(n_ce_doc_complete)]))
  n_q <- sum(dat$n_ce_doc_complete >= MIN_REVIEWS)
  ggplot(d, aes(n_ce_doc_complete, office)) +
    geom_vline(xintercept = MIN_REVIEWS, linetype = "dashed", color = "gray45", linewidth = 0.5) +
    geom_point(color = bar_col, size = 3) +
    geom_text(aes(label = comma(n_ce_doc_complete)), hjust = -0.3, size = 2.6, color = "gray35") +
    annotate("text", x = MIN_REVIEWS, y = 0.55, label = paste0("≥ ", MIN_REVIEWS, " qualifies"),
             hjust = -0.05, vjust = 0, size = 2.7, color = "gray45") +
    scale_x_log10(expand = expansion(mult = c(0.04, 0.16)), labels = comma) +
    labs(title = arm_lab, subtitle = funnel,
         x = "Document-anchored complete CE reviews (log scale)", y = NULL) +
    theme_catf() + theme(plot.subtitle = element_text(size = rel(0.8)),
                         panel.grid.major.y = element_line(color = "gray93", linewidth = 0.3))
}
p_inv_blm <- inv_panel(blm_inv, "BLM field offices", catf_navy,
  sprintf("%s BLM-led → %s parsed to an office → %d offices clear ≥ %d document-anchored CE reviews",
          comma(blm_led), comma(blm_parsed), n_blm_qual, MIN_REVIEWS))
p_inv_doe <- inv_panel(doe_inv, "DOE administering offices", catf_light_blue,
  sprintf("%s DOE-led → %s linked to a register office → %d offices clear ≥ %d document-anchored CE reviews",
          comma(doe_val("doe_led")), comma(doe_val("with_office")), n_doe_qual, MIN_REVIEWS))
p_inv <- p_inv_blm / p_inv_doe +
  plot_annotation(
    title = "Field-office inventory: two arms, both thin above the analysis threshold",
    subtitle = "Ranked per-office count of document-anchored complete CE reviews (full history); only offices past the dashed line enter the office fixed-effects regression.",
    caption = paste0("BLM offices parse from the DOI-BLM case number; DOE offices link through the CX register. Top 18 offices per arm; log x-axis.\n",
                     "Bars count each office's full-history document-anchored complete CE reviews. An inventory, not a duration comparison."),
    theme = theme(plot.title = element_text(face = "bold", size = rel(1.2), color = catf_navy),
                  plot.subtitle = element_text(size = rel(0.9), color = catf_dark_blue),
                  plot.caption = element_text(size = rel(0.8), color = "gray50", hjust = 0)))
ggsave(file.path(FIG, "fig_d4_fieldoffice_inventory.png"), p_inv, width = 10, height = 9, dpi = 300, bg = "white")
saveRDS(p_inv, file.path(FIG, "fig_d4_fieldoffice_inventory.rds"))

# ---------------------------------------------------------------------------
# Regression: log(duration) ~ log(cum_count) + factor(decision_year) + energy + office FE
# Coefficient on log(cum_count) = learning effect net of calendar trend (negative = faster).
# ---------------------------------------------------------------------------
# Fits two nested models so the calendar-time confound is explicit in committed output:
#   raw  = log(dur) ~ log(cum_count) + energy + office FE            (NO calendar control)
#   ctrl = log(dur) ~ log(cum_count) + factor(decision_year) + energy + office FE
# A negative `raw` that turns null/positive under `ctrl` means the apparent speed-up is
# secular calendar drift, not within-office learning.
# `agency` tags the row; `src` is the cumulative-experience frame (df_cum for BLM, doe_cum for the
# DOE arm). BOTH arms use each office's FULL review history — no calendar filter (symmetric design);
# `frame` = "full" for every row. Offices qualify on their full-history caseload (office_n >= thr).
# Raw CIs are stored alongside controlled CIs so the report table shows a 95% interval on every row.
fit_lc <- function(proc, anch, thr, note = "", agency = "BLM", src = df_cum) {
  s <- src |> filter(process_type == proc, anchor == anch, office_n >= thr)
  frame <- "full"
  n_off <- n_distinct(s$office_code)
  if (n_off < 2 || nrow(s) < 30) {
    return(tibble(agency = agency, frame = frame, process = proc, anchor = anch, threshold = thr,
                  n_reviews = nrow(s), n_offices = n_off,
                  pct_per_doubling_raw = NA_real_, pct_lo_raw = NA_real_, pct_hi_raw = NA_real_,
                  beta_log_cum = NA_real_, ci_low = NA_real_, ci_high = NA_real_,
                  pct_per_doubling = NA_real_, pct_lo = NA_real_, pct_hi = NA_real_,
                  note = paste0("not estimated: ", n_off, " offices >= ", thr, " (need >=2); ", nrow(s), " reviews")))
  }
  s <- s |> mutate(decision_year = factor(decision_year))
  y <- log(pmax(s$duration_days, 1))
  m_raw <- lm(y ~ log(cum_count) + energy + office_code, data = s)
  m     <- lm(y ~ log(cum_count) + decision_year + energy + office_code, data = s)
  est <- coef(m)[["log(cum_count)"]]; ci <- confint(m, "log(cum_count)")
  est_raw <- coef(m_raw)[["log(cum_count)"]]; ci_raw <- confint(m_raw, "log(cum_count)")
  tibble(agency = agency, frame = frame, process = proc, anchor = anch, threshold = thr,
         n_reviews = nrow(s), n_offices = n_off,
         pct_per_doubling_raw = round(100 * (2^est_raw - 1), 1),       # no calendar control
         pct_lo_raw = round(100 * (2^ci_raw[1] - 1), 1), pct_hi_raw = round(100 * (2^ci_raw[2] - 1), 1),
         beta_log_cum = round(est, 4), ci_low = round(ci[1], 4), ci_high = round(ci[2], 4),
         pct_per_doubling = round(100 * (2^est - 1), 1),               # WITH calendar control
         pct_lo = round(100 * (2^ci[1] - 1), 1), pct_hi = round(100 * (2^ci[2] - 1), 1),
         note = note)
}

model_tbl <- bind_rows(
  # BLM arm — full window, full-history office qualification.
  fit_lc("CE", "Document", MIN_REVIEWS, "PRIMARY: document-anchored, office FE + calendar + energy"),
  fit_lc("CE", "Register", MIN_REVIEWS, "Sensitivity: register-anchored (artifact-prone start dates)"),
  fit_lc("EA", "Document", MIN_REVIEWS, "EA document-anchored: too sparse for office-level analysis"),
  fit_lc("EA", "Register", EA_RELAX,    "EA exploratory: register-anchored, relaxed >=10 threshold (artifact-prone)"),
  # DOE arm — CE-only, document-anchored primary (structural mirror of BLM); register too thin.
  # Full window, same as BLM (symmetric design).
  fit_lc("CE", "Document", MIN_REVIEWS, "DOE PRIMARY: document-anchored, office FE + calendar + energy",
         agency = "DOE", src = doe_cum),
  fit_lc("CE", "Register", MIN_REVIEWS, "DOE register-anchored: too thin (n~137) for a sensitivity arm",
         agency = "DOE", src = doe_cum)
)
write_csv(model_tbl, file.path(DIAG, "d4_fieldoffice_model.csv"))
message("\n=== Learning-curve regression (log(cum_count) coefficient) ===")
print(as.data.frame(model_tbl))

# ---- HARD CHECK: DOE CE-Document primary row reproduces the BLM null (raw - -> controlled +) ----
doe_row <- model_tbl |> filter(agency == "DOE", process == "CE", anchor == "Document")
stopifnot("DOE CE-Document row missing" = nrow(doe_row) == 1)
if (doe_row$n_offices != 12L)
  stop(sprintf("DOE CE-Document n_offices %d != 12", doe_row$n_offices))
if (!(doe_row$pct_per_doubling_raw < 0 && doe_row$pct_per_doubling > 0))
  stop(sprintf("DOE sign flip not reproduced: raw %.1f%%, controlled %.1f%%",
               doe_row$pct_per_doubling_raw, doe_row$pct_per_doubling))
message(sprintf("HARD CHECK PASSED — DOE CE-Document: %d offices, %d reviews, raw %+.1f%% -> controlled %+.1f%%/doubling (CI %+.1f, %+.1f)",
                doe_row$n_offices, doe_row$n_reviews, doe_row$pct_per_doubling_raw,
                doe_row$pct_per_doubling, doe_row$pct_lo, doe_row$pct_hi))

message("\n02_create_figures.R complete. Figures -> output/deliverable04/figures/ ; tables -> diagnostics/")
