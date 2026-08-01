#!/usr/bin/env Rscript
# D2 Phase 6 — significance analysis (plan v2.11 §8).
#
# Reads the determination dataset (+ threshold child + corpus) and produces the headline
# tables. HEADLINE-DENOMINATOR GATE: every primary table filters to
# agency_scope_status == 'primary_blm_doe_family' (plus in-scope time). context_other_agency /
# manual_scope_review rows are reported separately, never folded into A1 primary rates.
# Dual denominators (projects AND determinations); cells below MIN_CELL suppressed.
# ANALYTIC GRAIN: headline counts are taken over `primary_dr` = one row per
# (document x resource x determination_class), NOT raw determination instances — so a resource/class
# concluded more than once in a document (a duplicate the LLM may emit, or two sub-findings that
# crosswalk to the same 12-area bucket) counts once, not twice. The raw instances stay in the
# parquet for provenance; only the counting unit changes here.
#
# NOTE: on a --dry-run determinations table every row is extraction_method='regex' &
# needs_human_review=TRUE, so these tables are ILLUSTRATIVE until the billable LLM pass + gold.
#
# Run:  Rscript phase2/code/deliverable02/06_create_figures.R
suppressMessages({library(arrow); library(dplyr); library(tidyr); library(readr); library(stringr)})
options(arrow.skip_nul = TRUE)   # EIS evidence/rationale text carries embedded nul bytes (PDF artifact)

A <- "phase2/data/analysis/deliverable02"
OUT <- "phase2/output/deliverable02/analysis"; dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
MIN_CELL <- 5
NON_DET <- c("not_a_determination", "ambiguous")

# Run with NO flags. `det`/`thr` are the FONSI track only and feed the FONSI headline tables; the
# EIS-track block below runs whenever significance_determinations_eis.parquet exists and reads the
# _eis parquets directly, so the two tracks never mix. (A removed --with-eis flag used to fold EIS
# rows into `det` BEFORE the headline gate, silently corrupting every FONSI table and figure —
# 193 analyzed projects became 325. Removed 2026-07-15; combined-track exploration should read the
# parquets in an ad-hoc session instead.)
det <- read_parquet(file.path(A, "significance_determinations.parquet"))
thr <- read_parquet(file.path(A, "determination_thresholds.parquet"))
eis_det_path <- file.path(A, "significance_determinations_eis.parquet")
if (length(commandArgs(trailingOnly = TRUE)) > 0)
  stop("06_create_figures.R takes no flags (the EIS block auto-runs off the _eis parquet); ",
       "got: ", paste(commandArgs(trailingOnly = TRUE), collapse = " "))

# ---- headline gate ----
primary <- det %>%
  filter(agency_scope_status == "primary_blm_doe_family",
         analysis_scope == "primary",
         !determination_class %in% NON_DET)
cat(sprintf("determinations total=%d  primary-scope determinations=%d  (projects=%d)\n",
            nrow(det), nrow(primary), n_distinct(primary$project_id)))

# analytic-grain rollup: one row per (document x resource x class). Collapses same-resource/class
# repeats within a document; per-document attributes (agency, cohort) are constant within the group
# so first() is safe; mitigation is reconciled with any() (a document's resource conclusion is
# mitigation-dependent if ANY of its collapsed sub-findings is), never keep-first-arbitrary.
primary_dr <- primary %>%
  group_by(project_id, document_id, shared_resource_area, determination_class) %>%
  summarise(agency = first(agency), cohort_by_date = first(cohort_by_date),
            mitigation_dependent = any(as.logical(mitigation_dependent)), .groups = "drop")
cat(sprintf("analytic determinations (document x resource x class) = %d\n", nrow(primary_dr)))

suppress <- function(df, col = "n") { df[[paste0(col, "_suppressed")]] <- df[[col]] < MIN_CELL; df }
w <- function(df, name) { write_csv(df, file.path(OUT, name)); cat("  wrote", name, "\n") }

# 1. headline cross-resource significance map (resource x class)
w(primary_dr %>% count(shared_resource_area, determination_class) %>%
    suppress() %>% arrange(desc(n)), "resource_by_class.csv")

# 2. class distribution + dual denominators (analytic determinations AND distinct projects)
w(bind_rows(
    primary_dr %>% count(determination_class, name = "n_determinations"),
    primary %>% distinct(project_id, determination_class) %>%
      count(determination_class, name = "n_projects") %>% rename(n_determinations = n_projects) %>%
      mutate(determination_class = paste0(determination_class, " [project-level]"))
  ), "class_distribution_dual_denominator.csv")

# 3. cross-agency (BLM vs DOE-family subagencies, within primary scope)
w(primary_dr %>% count(agency, determination_class) %>% suppress(), "agency_by_class.csv")

# 4. cross-cohort (FRA label = 2023-06-03)
w(primary_dr %>% count(cohort_by_date, determination_class) %>% suppress(), "cohort_by_class.csv")

# 5. threshold profile from the CHILD table (not the scalar summary)
thr_primary <- thr %>% semi_join(primary, by = "determination_instance_id") %>%
  left_join(primary %>% select(determination_instance_id, determination_class), by = "determination_instance_id")
w(thr_primary %>% count(threshold_type, determination_class) %>% suppress(), "threshold_by_class.csv")

# 6. MITIGATION at BOTH levels (the deliverable centerpiece: mitigated FONSIs).
#    Reported at two grains because they answer different questions and use different signals:
#    - DOCUMENT level = the classic "mitigated FONSI" rate: is the FONSI's overall no-significant-
#      impact finding mitigation-dependent? A document counts as mitigated if ANY of its
#      determinations is. The window-level mitigation flag is fine here (we OR to the document).
#    - RESOURCE level = which resource areas' conclusions depend on mitigation. Uses the PRECISE
#      per-resource text signal (the less_than_significant_with_mitigation class), NOT the
#      window-shared flag, which over-attributes across a multi-resource window.

# 6a. document-level mitigated-FONSI rate (project-level count reported alongside)
doc_mit <- primary %>%
  group_by(project_id, document_id) %>%
  summarise(mitigated_dependent = any(as.logical(mitigation_dependent)),
            mitigated_class_signal = any(determination_class == "less_than_significant_with_mitigation"),
            .groups = "drop")
doc_summary <- doc_mit %>%
  summarise(n_documents = n(), n_projects = n_distinct(project_id),
            n_mitigated_dependent = sum(mitigated_dependent),
            share_mitigated_dependent = round(mean(mitigated_dependent), 3),
            n_mitigated_class_signal = sum(mitigated_class_signal),
            share_mitigated_class_signal = round(mean(mitigated_class_signal), 3))
w(doc_summary, "mitigation_document_level.csv")
cat(sprintf("mitigated-FONSI rate (document level): %d/%d = %.1f%% (any mitigation-dependent) | %.1f%% (LTS-with-mitigation class)\n",
            doc_summary$n_mitigated_dependent, doc_summary$n_documents,
            100 * doc_summary$share_mitigated_dependent, 100 * doc_summary$share_mitigated_class_signal))

# 6b. resource-level: which resource areas carry mitigation-dependent conclusions (precise signal).
#     Restricted to the below-the-line FONSI classes (the handful of significant_* determinations in
#     FONSIs are anomalies, held out here) so counts reconcile with the resource map + figures.
BELOW_LINE <- c("no_significant_impact", "less_than_significant", "less_than_significant_with_mitigation")
w(primary_dr %>%
    filter(shared_resource_area != "project_wide", determination_class %in% BELOW_LINE) %>%
    group_by(shared_resource_area) %>%
    summarise(n_determinations = n(),
              n_mit_class = sum(determination_class == "less_than_significant_with_mitigation"),
              share_mit_class = round(mean(determination_class == "less_than_significant_with_mitigation"), 3),
              n_mit_dependent = sum(as.logical(mitigation_dependent)), .groups = "drop") %>%
    arrange(desc(n_mit_class)) %>% suppress(col = "n_determinations"),
  "mitigation_by_resource.csv")

# 6c. class x mitigation cross-tab (analytic grain), kept for continuity
w(primary_dr %>% count(determination_class, mitigation_dependent) %>% suppress(), "mitigation_by_class.csv")

# 6d. #52(a) AGGREGATE any-overlap resource-match FINDING.
#     Distinct from the class-signal shares above (6a/6b): this is the join-based pairing of a
#     flagged effect to a same-resource committed condition, under the any-overlap (D6 multi-label)
#     rule = the shipped `mitigation_resource_matched` column. Denominator = the flagged
#     significant / less-than-significant determinations (impact-acknowledged classes; a
#     no-significant-impact conclusion is not a flagged effect). Analytic grain, reconciled with
#     any() like the rest of section 6. Per-resource split is DESCRIPTIVE ONLY (exact attribution is
#     weaker, ~0.76 precision on the condition tags) — never a per-project claim.
IMPACT_CLASSES <- c("significant_adverse", "significant_unavoidable",
                    "less_than_significant", "less_than_significant_with_mitigation")
rmatch_dr <- primary %>%
  filter(determination_class %in% IMPACT_CLASSES) %>%
  group_by(project_id, document_id, shared_resource_area, determination_class) %>%
  summarise(rmatched = any(as.logical(mitigation_resource_matched)), .groups = "drop")
rmatch_doc <- primary %>%
  filter(determination_class %in% IMPACT_CLASSES) %>%
  group_by(document_id) %>%
  summarise(rmatched = any(as.logical(mitigation_resource_matched)), .groups = "drop")
rmatch_overall <- tibble(
  n_determinations = nrow(rmatch_dr),
  n_matched        = sum(rmatch_dr$rmatched),
  share_matched    = round(mean(rmatch_dr$rmatched), 3),
  n_documents      = nrow(rmatch_doc),
  n_docs_matched   = sum(rmatch_doc$rmatched),
  share_docs_matched = round(mean(rmatch_doc$rmatched), 3))
w(rmatch_overall, "mitigation_resource_match_overall.csv")
cat(sprintf("resource-match FINDING (#52a any-overlap): %d/%d determinations = %.1f%% | docs %d/%d = %.1f%%\n",
            rmatch_overall$n_matched, rmatch_overall$n_determinations, 100 * rmatch_overall$share_matched,
            rmatch_overall$n_docs_matched, rmatch_overall$n_documents, 100 * rmatch_overall$share_docs_matched))
w(rmatch_dr %>% group_by(shared_resource_area) %>%
    summarise(n_determinations = n(), n_matched = sum(rmatched),
              share_matched = round(mean(rmatched), 3), .groups = "drop") %>%
    arrange(desc(share_matched)) %>% suppress(col = "n_determinations"),
  "mitigation_resource_match_by_resource.csv")

# 7. context universe reported SEPARATELY (never in primary rates)
w(det %>% filter(!determination_class %in% NON_DET) %>%
    count(agency_scope_status, determination_class) %>% suppress(),
  "context_universe_by_scope.csv")

# 8. association layer (interpretable): significant(0/1) ~ threshold flags + agency + cohort.
#    Guarded — needs enough adjudicated (non-regex) rows; skipped on a dry-run table.
adjudicated <- primary %>% filter(extraction_method == "regex+llm")
if (nrow(adjudicated) >= 200 && n_distinct(adjudicated$determination_class) > 1) {
  d <- adjudicated %>%
    mutate(significant = as.integer(determination_class %in%
             c("significant_adverse", "significant_unavoidable", "eis_required")))
  thr_wide <- thr %>% semi_join(adjudicated, by = "determination_instance_id") %>%
    distinct(determination_instance_id, threshold_type) %>% mutate(v = 1) %>%
    pivot_wider(names_from = threshold_type, values_from = v, values_fill = 0,
                names_prefix = "thr_")
  d <- d %>% left_join(thr_wide, by = "determination_instance_id")
  thr_cols <- grep("^thr_", names(d), value = TRUE)
  form <- as.formula(paste("significant ~ agency + cohort_by_date +",
                           paste(thr_cols, collapse = " + ")))
  fit <- glm(form, data = d, family = binomial())
  or <- data.frame(term = names(coef(fit)), odds_ratio = exp(coef(fit)))
  w(or, "association_odds_ratios.csv")
  cat("  association layer fit on", nrow(d), "adjudicated determinations\n")
} else {
  cat("  [association layer skipped] needs >=200 adjudicated (regex+llm) determinations; ",
      "run the billable LLM pass first.\n")
}

# ---- FIGURES (CATF-styled PNGs for the report) ----
fig_ok <- tryCatch({ library(ggplot2); library(patchwork); source("phase2/code/utils/utils.R"); TRUE },
                   error = function(e) { cat("[figures skipped]", conditionMessage(e), "\n"); FALSE })
if (fig_ok) tryCatch({
  set.seed(42)   # reproducible figures/tables (any jitter, repel, or tie-breaks are stable)
  res_label <- c(air_quality="Air quality", water="Water", biological="Biological",
    cultural="Cultural / historic", visual="Visual", noise="Noise", soils_geology="Soils / geology",
    socioeconomic="Socioeconomic", transportation="Transportation", land_use="Land use",
    climate_ghg="Climate / GHG", public_health="Public health", unknown="Unplaced")
  class_label <- c(no_significant_impact="No significant impact",
    less_than_significant="Less than significant",
    less_than_significant_with_mitigation="Committed mitigation")
  relab <- function(x, m) ifelse(is.na(m[x]), x, m[x])
  savefig <- function(p, name, w = 8, h = 5) {
    suppressMessages(ggsave(file.path(OUT, name), p, width = w, height = h, dpi = 300))
    # .rds sidecar (same basename) so downstream scripts can readRDS + retitle.
    saveRDS(p, file.path(OUT, sub("\\.png$", ".rds", name)))
  }

  # clean-energy technology from the dataset's own `project_type` classification (a curated multi-tag
  # field, 100% populated). Assign one primary technology per project by priority: generation types
  # first, then nuclear/CCS, then transmission. ~22% resolve to "Other / mixed". Used by both tracks.
  techmap <- tryCatch({
    pc <- read_parquet(file.path(dirname(A), "projects_combined.parquet"),
                       col_select = c("project_id", "project_type"))
    pt <- ifelse(is.na(pc$project_type), "", pc$project_type)
    tibble(project_id = pc$project_id, tech = dplyr::case_when(
      grepl("Renewable Energy Production - Solar", pt) ~ "Solar",
      grepl("Wind, Onshore|Wind, Offshore|Renewable Energy Production - Wind", pt) ~ "Wind",
      grepl("Hydropower", pt) ~ "Hydro",
      grepl("Geothermal", pt) ~ "Geothermal",
      grepl("Biomass", pt) ~ "Biomass",
      grepl("Energy Storage", pt) ~ "Storage",
      grepl("Nuclear Technology", pt) ~ "Nuclear",
      grepl("Carbon Capture", pt) ~ "Carbon capture",
      grepl("Electricity Transmission", pt) ~ "Transmission",
      grepl("Renewable Energy Production - Other", pt) ~ "Renewable (other)",
      TRUE ~ "Other / mixed"))
  }, error = function(e) { cat("[techmap skipped]", conditionMessage(e), "\n"); NULL })

  # resource-level analytic determinations: below-the-line FONSI classes only (the significant_*
  # anomalies are held out), so figure counts reconcile with the resource map + mitigation tables.
  res_lvl <- primary_dr %>%
    filter(shared_resource_area != "project_wide", determination_class %in% BELOW_LINE)

  # Fig — significance outcomes by resource (100% stacked; counts + %, sorted by mitigation reliance)
  odata <- res_lvl %>% filter(shared_resource_area != "unknown") %>%
    count(shared_resource_area, determination_class) %>%
    group_by(shared_resource_area) %>% mutate(share = n / sum(n), tot = sum(n)) %>% ungroup() %>%
    mutate(Resource = relab(shared_resource_area, res_label),
           Outcome = factor(relab(determination_class, class_label), levels = rev(unname(class_label))))
  mit_order <- odata %>% filter(determination_class == "less_than_significant_with_mitigation") %>%
    select(shared_resource_area, mit_share = share)
  odata <- odata %>% left_join(mit_order, by = "shared_resource_area") %>%
    mutate(mit_share = coalesce(mit_share, 0))
  otot <- odata %>% distinct(Resource, tot, mit_share)
  savefig(ggplot(odata, aes(reorder(Resource, mit_share), share, fill = Outcome)) +
        geom_col() +
        geom_text(aes(label = ifelse(share >= 0.08, sprintf("%d (%d%%)", n, round(100 * share)), "")),
                  position = position_fill(vjust = 0.5), size = 2.4, color = "white") +
        geom_text(data = otot, aes(x = reorder(Resource, mit_share), y = 1.0, label = paste0("n=", tot)),
                  inherit.aes = FALSE, hjust = -0.1, size = 2.8, color = "gray35") +
        coord_flip() +
        scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.12))) +
        scale_fill_manual(values = c("No significant impact" = catf_light_blue,
                                     "Less than significant" = catf_dark_blue,
                                     "Committed mitigation" = catf_magenta)) +
        guides(fill = guide_legend(reverse = TRUE)) +  # legend left-to-right = bar segment order
        labs(title = "How agencies stay below the line, by resource",
             subtitle = "Share of each resource's FONSI determinations by outcome — sorted by reliance on mitigation (top = most)",
             x = NULL, y = NULL, fill = NULL) + theme_catf() + theme(legend.position = "bottom"),
    "fig_outcomes_by_resource.png", 8, 5.5)

  mitr <- res_lvl %>% filter(shared_resource_area != "unknown") %>%
    group_by(shared_resource_area) %>%
    summarise(n = n(), n_mit = sum(determination_class == "less_than_significant_with_mitigation"),
              share = mean(determination_class == "less_than_significant_with_mitigation"),
              .groups = "drop") %>% filter(n >= MIN_CELL) %>%
    mutate(Resource = relab(shared_resource_area, res_label))

  # Fig — mitigation intensity by resource (lollipop; label carries share AND the underlying counts)
  savefig(mitr %>% ggplot(aes(reorder(Resource, share), share)) +
        geom_segment(aes(xend = reorder(Resource, share), y = 0, yend = share), color = "gray80") +
        geom_point(aes(size = n), color = catf_magenta) +
        geom_text(aes(label = sprintf("%s  (%d of %d)", scales::percent(share, accuracy = 1), n_mit, n)),
                  hjust = -0.15, size = 2.8, color = "gray35") +
        coord_flip() + scale_size_area(max_size = 7, guide = "none") +
        scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.28))) +
        labs(title = "Which resources drive mitigation",
             subtitle = "Share of a resource's FONSI conclusions that depend on committed mitigation",
             x = NULL, y = NULL,
             caption = "Label = share (mitigation-dependent of total). Point size = total determinations. Per-resource class signal.") + theme_catf(),
    "fig_mitigation_by_resource.png", 8, 5)

  # Fig 4 — the mitigation landscape (scatter)
  f4 <- ggplot(mitr, aes(n, share)) +
    geom_hline(yintercept = mean(mitr$share), linetype = "dashed", color = "gray70") +
    geom_point(aes(size = n * share), color = catf_dark_blue, alpha = 0.8) +
    scale_size_area(max_size = 11, guide = "none") + scale_y_continuous(labels = scales::percent) +
    labs(title = "The mitigation landscape",
         subtitle = "Resources both frequently analyzed and frequently mitigated sit upper-right",
         x = "How often analyzed (determinations)", y = "Mitigation-dependent share",
         caption = paste0("Dashed line = the average mitigation share across resources (",
                          scales::percent(mean(mitr$share), accuracy = 1),
                          "); points above it rely on mitigation more than average.\n",
                          "Point size approx. number of mitigation-dependent determinations.")) + theme_catf()
  f4 <- if (requireNamespace("ggrepel", quietly = TRUE))
          f4 + ggrepel::geom_text_repel(aes(label = Resource), size = 3, color = catf_navy, seed = 1)
        else f4 + geom_text(aes(label = Resource), vjust = -0.9, size = 3, color = catf_navy)
  savefig(f4, "fig_mitigation_landscape.png", 7.5, 6)

  # Fig 5 — analysis breadth per FONSI (violin + box)
  savefig(res_lvl %>% group_by(project_id, document_id) %>%
      summarise(n_res = n_distinct(shared_resource_area), .groups = "drop") %>%
      left_join(doc_mit %>% select(project_id, document_id, mitigated_class_signal),
                by = c("project_id", "document_id")) %>%
      mutate(Group = ifelse(mitigated_class_signal, "Mitigated FONSI", "Non-mitigated FONSI")) %>%
      filter(!is.na(Group)) %>%
      ggplot(aes(Group, n_res, fill = Group)) +
        geom_violin(alpha = 0.35, color = NA) +
        geom_boxplot(width = 0.16, outlier.size = 0.5, color = catf_navy, fill = "white") +
        scale_fill_manual(values = c("Mitigated FONSI" = catf_magenta,
                                     "Non-mitigated FONSI" = catf_dark_blue), guide = "none") +
        scale_y_continuous(breaks = seq(0, 20, 2)) +
        labs(title = "How broad is a FONSI's significance analysis?",
             subtitle = "Distinct resource areas addressed per FONSI document",
             x = NULL, y = "Resource areas per FONSI") + theme_catf(),
    "fig_breadth_per_fonsi.png", 7, 5)

  # Fig — extraction accuracy (dumbbell: all-400 vs held-out per metric)
  val_fig <- tryCatch(read_parquet(file.path(A, "validation_metrics.parquet")), error = function(e) NULL)
  if (!is.null(val_fig) && nrow(val_fig) > 0) {
    metric_lab <- c(candidate_is_determination = "Finds a determination",
      resource_determination_detection = "Assigns the right resource",
      determination_class_macro_f1 = "Gets the class right",
      mitigation_dependent_f1 = "Flags mitigation-dependence",
      primary_threshold_type_accuracy = "Identifies the threshold")
    vdat <- val_fig %>% mutate(score = coalesce(f1, precision)) %>%
      filter(scope %in% c("overall", "holdout"), metric %in% names(metric_lab)) %>%
      mutate(Metric = factor(metric_lab[metric], levels = rev(unname(metric_lab))),
             Scope = ifelse(scope == "holdout", "Held-out test", "All 400"))
    vwide <- vdat %>% select(Metric, Scope, score) %>%
      tidyr::pivot_wider(names_from = Scope, values_from = score)
    savefig(ggplot(vdat, aes(score, Metric)) +
          # shade the bottom two rows (secondary attributes that matter less)
          annotate("rect", xmin = 0, xmax = 1.12, ymin = 0.5, ymax = 2.5, fill = "gray92", alpha = 0.7) +
          geom_vline(xintercept = 0.8, linetype = "dashed", color = "gray60") +
          geom_segment(data = vwide, aes(x = `All 400`, xend = `Held-out test`, y = Metric, yend = Metric),
                       inherit.aes = FALSE, color = "gray70", linewidth = 1) +
          geom_point(aes(color = Scope), size = 4.5, alpha = 0.75) +
          geom_text(data = dplyr::filter(vdat, Scope == "All 400"),
                    aes(label = sprintf("%.2f", score)), nudge_y = 0.24, size = 2.7, color = catf_magenta) +
          geom_text(data = dplyr::filter(vdat, Scope == "Held-out test"),
                    aes(label = sprintf("%.2f", score)), nudge_y = -0.24, size = 2.7, color = catf_dark_blue) +
          scale_color_manual(values = c("All 400" = catf_magenta, "Held-out test" = catf_dark_blue)) +
          scale_x_continuous(limits = c(0, 1.12), breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
          labs(title = "The extraction was graded before anything was reported",
               subtitle = "Agreement with the AI-human reviewed answer key: full sample vs the held-out test",
               x = "Score (F1; threshold row = accuracy)", y = NULL, color = NULL,
               caption = "Dashed line = 0.80, the standard bar. Blue = held-out test (the honest score); magenta = all 400.\nShaded rows are secondary attributes that matter less to the findings.") +
          theme_catf() + theme(legend.position = "bottom"),
      "fig_validation_accuracy.png", 8, 4.8)
  }
  # NOTE: fig_corpus_overview (projects + documents bars, FONSI + EIS resource waffles) is now
  # built in the EIS block below, where the EIS resource mix is available for the second waffle.
  # corpus_fig is still needed here by the agency-scope waffle and the sub-agency heatmap.
  corpus_fig <- tryCatch(read_parquet(file.path(A, "significance_corpus.parquet")), error = function(e) NULL)

  # Fig — regulatory-threshold profile (descriptive; threshold ID is the least-accurate field)
  thr_lab <- c(other_quantitative = "Quantitative numeric limits", wetland_floodplain = "Wetland / floodplain",
    NHPA_adverse_effect = "NHPA §106 adverse effect", visual_vrm = "Visual (VRM)", ESA_take = "ESA take",
    NAAQS = "NAAQS (air)", ESA_jeopardy = "ESA jeopardy", noise_threshold = "Noise threshold", PSD = "PSD (air)")
  tprof <- thr %>% semi_join(distinct(primary, determination_instance_id), by = "determination_instance_id") %>%
    filter(!threshold_type %in% c("none", "unknown", "")) %>% count(threshold_type) %>%
    mutate(Threshold = ifelse(is.na(thr_lab[threshold_type]), threshold_type, thr_lab[threshold_type]))
  savefig(tprof %>% ggplot(aes(reorder(Threshold, n), n)) +
        geom_col(fill = catf_dark_blue) + geom_text(aes(label = n), hjust = -0.2, size = 3, color = "gray30") +
        coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.09))) +
        labs(title = "Which regulatory thresholds the conclusions lean on",
             subtitle = "Threshold citations across primary FONSI determinations",
             x = NULL, y = "Citations",
             caption = "Descriptive only: threshold identification is the pipeline's least-accurate field (see validation).\nExcludes conclusions not anchored to a specific threshold.") + theme_catf(),
    "fig_threshold_profile.png", 8, 4.5)

  # Fig — agency scope waffle: BLM + DOE kept, everything else dropped
  if (!is.null(corpus_fig)) {
    asc <- corpus_fig %>% filter(process_type == "EA") %>% distinct(project_id, agency) %>% count(agency) %>%
      mutate(grp = recode(agency, "DOE-family" = "DOE family", "other" = "Other (dropped)"),
             grp = factor(grp, levels = c("BLM", "DOE family", "Other (dropped)"))) %>% arrange(grp)
    asc <- asc %>% mutate(sq = round(100 * n / sum(n)))
    asc$sq[which.max(asc$sq)] <- asc$sq[which.max(asc$sq)] + (100 - sum(asc$sq))
    agrid <- expand.grid(y = 1:10, x = 1:10)
    agrid$grp <- factor(rep(as.character(asc$grp), asc$sq)[1:100], levels = levels(asc$grp))
    savefig(ggplot(agrid, aes(x, y, fill = grp)) +
          geom_tile(color = "white", linewidth = 1.1) + coord_equal() +
          scale_fill_manual(values = c("BLM" = catf_dark_blue, "DOE family" = catf_navy,
                                       "Other (dropped)" = "gray80"), name = NULL) +
          labs(title = "Who leads these FONSIs — and what the analysis keeps",
               subtitle = "Each square ≈ 1% of the 452 decarbonization FONSIs. BLM + DOE are analyzed; the rest is set aside.") +
          theme_void(base_family = "Helvetica") +
          theme(plot.title = element_text(face = "bold", color = catf_navy, size = rel(1.15)),
                plot.subtitle = element_text(color = catf_dark_blue, size = rel(0.85)),
                legend.text = element_text(size = rel(0.9)), legend.position = "right"),
      "fig_agency_scope.png", 8, 5)
  }

  # Fig — FONSI coverage funnel: how the 452-project corpus narrows to the analyzed set. Mirrors
  # fig_eis_funnel's project panel; alpha lightens the set-aside stages so the analyzed 193 reads
  # as the saturated end of the same corpus. Finding-section coverage needs the D6 spans file, so
  # that stage (and the funnel) degrades gracefully when it is absent.
  if (!is.null(corpus_fig)) {
    fcor <- corpus_fig %>% filter(process_type == "EA") %>%
      distinct(project_id, agency_scope_status, analysis_scope)
    fprim <- fcor %>% filter(agency_scope_status == "primary_blm_doe_family",
                             analysis_scope == "primary")
    n_located <- tryCatch({
      sp <- read_parquet(file.path(dirname(A), "deliverable06", "fonsi_evidence_spans.parquet"),
                         col_select = c("project_id", "span_type", "manifest_role"))
      fprim %>% semi_join(sp %>% filter(span_type == "finding",
                                        manifest_role %in% c("linked_ea", "canonical_fonsi",
                                                             "supporting_fonsi")) %>%
                            distinct(project_id), by = "project_id") %>% nrow()
    }, error = function(e) NA)
    ffun_stages <- c("Decarbonization FONSI corpus", "Led by BLM or the DOE family",
                     "Dated decision, 2009–present", "Finding sections located",
                     "Analyzed: ≥1 coded determination")
    # explicit FONSI-only count (defensive: `primary` should already be single-track)
    n_analyzed_fonsi <- primary %>% filter(process_type == "EA") %>%
      distinct(project_id) %>% nrow()
    ffun <- tibble(stage = factor(ffun_stages, levels = rev(ffun_stages)),
                   n = c(nrow(fcor), sum(fcor$agency_scope_status == "primary_blm_doe_family"),
                         nrow(fprim), n_located, n_analyzed_fonsi)) %>%
      filter(!is.na(n))
    w(ffun %>% transmute(metric = as.character(stage), n), "fonsi_coverage_funnel.csv")
    savefig(ggplot(ffun, aes(n, stage)) +
          geom_col(aes(alpha = stage), fill = catf_navy, width = 0.62) +
          geom_text(aes(label = scales::comma(n)), hjust = -0.15, size = 3.4, color = "gray25") +
          scale_alpha_manual(values = setNames(c(0.30, 0.45, 0.62, 0.80, 1), ffun_stages),
                             guide = "none") +
          scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
          labs(title = "From 452 FONSIs to the 193 analyzed",
               subtitle = "Projects retained at each step; darker = closer to the analyzed set",
               x = NULL, y = NULL,
               caption = paste0("Set-asides, top to bottom: other-agency FONSIs (kept as context); no reliable decision\n",
                                "date, pre-2009, or boundary review; finding statements the extraction did not recognize;\n",
                                "flagged text with no codable determination. Every rate in this report describes the analyzed set.")) +
          theme_catf(),
      "fig_fonsi_funnel.png", 8.5, 4.8)
  }

  # Fig — mitigated vs not (single 100% stacked bar) for the top of the mitigated section
  ms <- doc_mit %>% mutate(grp = ifelse(mitigated_class_signal, "Mitigated FONSI", "Not mitigated")) %>%
    count(grp) %>% mutate(grp = factor(grp, levels = c("Not mitigated", "Mitigated FONSI")),
                          share = n / sum(n))
  savefig(ggplot(ms, aes(x = 1, y = share, fill = grp)) + geom_col(width = 0.5) +
        geom_text(aes(label = sprintf("%s: %d (%d%%)", grp, n, round(100 * share))),
                  position = position_stack(vjust = 0.5), color = "white", size = 3.6, fontface = "bold") +
        coord_flip() + scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
        scale_fill_manual(values = c("Mitigated FONSI" = catf_magenta, "Not mitigated" = catf_light_blue),
                          guide = "none") +
        labs(title = "Most analyzed decarbonization FONSIs are mitigated",
             subtitle = "The 258 analyzed FONSI decision documents (193 projects; some projects file more than one), split by whether\nthe not-significant finding depends on committed mitigation",
             x = NULL, y = NULL) +
        theme_catf() + theme(axis.text = element_blank(), axis.ticks = element_blank(),
                             panel.grid = element_blank()),
    "fig_mitigated_share.png", 8, 2.7)

  # department + sub-agency analysis (analytic grain, below-line resource determinations)
  dept <- primary %>%
    filter(shared_resource_area %in% names(res_label), !shared_resource_area %in% c("project_wide", "unknown"),
           determination_class %in% BELOW_LINE) %>%
    distinct(project_id, document_id, shared_resource_area, determination_class, agency) %>%
    mutate(mit = determination_class == "less_than_significant_with_mitigation")

  # Fig — BLM vs DOE: mitigation reliance by resource (dumbbell)
  deptr <- dept %>% group_by(shared_resource_area, agency) %>%
    summarise(n = n(), mit = mean(mit), .groups = "drop") %>% filter(n >= 5) %>%
    mutate(Resource = relab(shared_resource_area, res_label))
  # order by the BLM - DOE gap: DOE-leaning resources at the bottom, BLM-leaning at the top
  ord <- deptr %>% select(Resource, agency, mit) %>%
    tidyr::pivot_wider(names_from = agency, values_from = mit) %>%
    mutate(diff = coalesce(BLM, 0) - coalesce(`DOE-family`, 0)) %>% arrange(diff)
  deptr <- deptr %>% mutate(Resource = factor(Resource, levels = ord$Resource))
  # background band per row, tinted by the department with the larger share (alpha-light),
  # with a horizontal wrapped label in the free corner of each band (top-right of the BLM band,
  # bottom-right of the DOE band — rows whose points sit far left)
  lead_bg <- ord %>%
    mutate(yi = match(Resource, levels(deptr$Resource)),
           Leads = ifelse(diff >= 0, "BLM", "DOE-family"))
  lead_lab <- lead_bg %>% group_by(Leads) %>%
    summarise(y = ifelse(first(Leads) == "BLM", max(yi), min(yi)), .groups = "drop") %>%
    mutate(label = ifelse(Leads == "BLM", "BLM has a\nlarger share", "DOE-family has\na larger share"),
           col = ifelse(Leads == "BLM", catf_purple, catf_dark_blue))
  x_lab <- max(deptr$mit) * 1.06
  savefig(ggplot(deptr, aes(mit, Resource)) +
        geom_rect(data = lead_bg, aes(xmin = -Inf, xmax = Inf, ymin = yi - 0.5, ymax = yi + 0.5,
                                      fill = Leads),
                  inherit.aes = FALSE, alpha = 0.10) +
        scale_fill_manual(values = c("BLM" = catf_purple, "DOE-family" = catf_dark_blue),
                          guide = "none") +
        geom_line(aes(group = Resource), color = "gray78", linewidth = 1) +
        geom_point(aes(color = agency), size = 3.6, alpha = 0.9) +
        geom_text(data = lead_lab, aes(x = x_lab, y = y, label = label),
                  inherit.aes = FALSE, hjust = 1, size = 3, fontface = "bold",
                  lineheight = 0.95, color = lead_lab$col) +
        scale_color_manual(values = c("BLM" = catf_purple, "DOE-family" = catf_dark_blue), name = NULL) +
        scale_x_continuous(labels = scales::percent, expand = expansion(mult = c(0.02, 0.08))) +
        labs(title = "Does a resource trigger mitigation more for BLM or DOE?",
             subtitle = "Share of a resource's FONSI conclusions that depend on committed mitigation, by department",
             x = "Mitigation-dependent share", y = NULL) +
        theme_catf() + theme(legend.position = "bottom"),
    "fig_dept_by_resource.png", 8, 5.5)

  # Fig — sub-agency x resource mitigation heatmap
  if (!is.null(corpus_fig)) {
    suba <- corpus_fig %>% filter(process_type == "EA", agency_scope_status == "primary_blm_doe_family") %>%
      distinct(project_id, lead_agency_harmonized) %>%
      mutate(sub = str_squish(str_remove_all(as.character(lead_agency_harmonized), '\\[|\\]|"')),
             sub = recode(sub, "Department of Energy" = "DOE (dept.)", "Power Marketing Administration" = "Power Marketing",
                          "Bureau of Land Management" = "BLM", "National Nuclear Security Administration" = "NNSA"))
    subr <- dept %>% left_join(suba, by = "project_id") %>% filter(!is.na(sub), sub != "") %>%
      group_by(sub, shared_resource_area) %>% summarise(n = n(), mit = mean(mit), .groups = "drop") %>%
      mutate(Resource = relab(shared_resource_area, res_label))
    keep_sub <- subr %>% group_by(sub) %>% summarise(tot = sum(n), .groups = "drop") %>%
      filter(tot >= 40) %>% pull(sub)
    subr <- subr %>% filter(sub %in% keep_sub, n >= 3)
    # x labels horizontal (wrapped) and colored by department: BLM magenta, DOE-family blue
    sub_axis <- sort(unique(subr$sub))
    sub_cols <- ifelse(sub_axis == "BLM", catf_magenta, catf_dark_blue)
    savefig(ggplot(subr, aes(sub, reorder(Resource, mit), fill = mit)) +
          geom_tile(color = "white", linewidth = 1) +
          geom_text(aes(label = scales::percent(mit, accuracy = 1), color = mit > 0.25), size = 2.6) +
          scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "gray15"), guide = "none") +
          scale_fill_gradientn(colors = c("#eef3fb", catf_light_blue, catf_dark_blue, catf_navy),
                               labels = scales::percent, breaks = c(0, 0.2, 0.4), name = "Mitigation\nshare") +
          scale_x_discrete(labels = function(x) stringr::str_wrap(x, width = 10)) +
          labs(title = "Which resources drive mitigation, by sub-agency",
               subtitle = "Share of a resource's conclusions that depend on mitigation (cells with ≥3 determinations)",
               x = NULL, y = NULL,
               caption = "Sub-agencies with ≥40 determinations (BLM in magenta, DOE-family in blue). Small cells are noisy; read the pattern, not the decimals.") +
          guides(fill = guide_colorbar(barheight = grid::unit(4, "cm"))) +
          theme_catf() + theme(axis.text.x = element_text(angle = 0, hjust = 0.5,
                                                          color = sub_cols, face = "bold"),
                               panel.grid = element_blank(), legend.position = "right"),
      "fig_subagency_by_resource.png", 8.5, 6)
  }

  # example mitigations — up to 2 per resource area, ordered by how mitigation-heavy the resource is
  ex <- primary %>%
    filter(shared_resource_area %in% names(res_label), shared_resource_area != "unknown",
           determination_class == "less_than_significant_with_mitigation", !is.na(rationale_text)) %>%
    mutate(example = str_squish(rationale_text), L = nchar(example)) %>%
    filter(L >= 40, L <= 240) %>%
    arrange(shared_resource_area, L, example) %>%   # `example` breaks length ties deterministically
    group_by(shared_resource_area) %>% slice_head(n = 2) %>% ungroup() %>%
    left_join(mitr %>% select(shared_resource_area, share), by = "shared_resource_area") %>%
    mutate(share = coalesce(share, 0), Resource = relab(shared_resource_area, res_label)) %>%
    arrange(desc(share), Resource, L) %>%
    select(Resource, example)
  w(ex, "mitigation_examples.csv")

  # Fig — FONSI mitigation-dependence by clean-energy technology (which techs lean on mitigation)
  if (!is.null(techmap)) {
    ftech <- primary_dr %>% filter(determination_class %in% BELOW_LINE,
             !shared_resource_area %in% c("project_wide", "unknown")) %>%
      left_join(techmap, by = "project_id") %>% filter(!is.na(tech), !tech %in% c("Other / mixed", "Renewable (other)")) %>%
      group_by(tech) %>%
      summarise(n = n(), n_mit = sum(determination_class == "less_than_significant_with_mitigation"),
                share = mean(determination_class == "less_than_significant_with_mitigation"),
                proj = n_distinct(project_id), .groups = "drop") %>% filter(n >= 40)
    w(ftech %>% arrange(desc(share)), "fonsi_technology.csv")
    savefig(ggplot(ftech, aes(reorder(tech, share), share)) +
          geom_col(fill = catf_magenta, width = 0.72) +
          geom_text(aes(label = sprintf("%s  (%d of %d)", scales::percent(share, accuracy = 1), n_mit, n)),
                    hjust = -0.1, size = 2.9, color = "gray30") +
          coord_flip() + scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.30))) +
          labs(title = "Which technologies lean on mitigation?",
               subtitle = "Share of a technology's FONSI conclusions that depend on committed mitigation",
               x = NULL, y = NULL,
               caption = "Technology from the dataset's project_type classification (one primary type per project; ~1 in 5 resolve to Other / mixed and are excluded).") +
          theme_catf(),
      "fig_fonsi_technology.png", 8, 4.4)

    # Fig — FONSI mitigation enforceability by resource (is the committed measure actually binding?)
    enf <- primary %>% filter(determination_class == "less_than_significant_with_mitigation",
             shared_resource_area %in% names(res_label), !shared_resource_area %in% c("project_wide", "unknown")) %>%
      distinct(project_id, document_id, shared_resource_area, mitigation_enforceability) %>%
      group_by(shared_resource_area) %>%
      summarise(n = n(), n_enf = sum(mitigation_enforceability == "permit_condition", na.rm = TRUE),
                share = mean(mitigation_enforceability == "permit_condition", na.rm = TRUE), .groups = "drop") %>%
      filter(n >= 10) %>% mutate(Resource = relab(shared_resource_area, res_label))
    w(enf %>% arrange(desc(share)), "fonsi_enforceability.csv")
    savefig(ggplot(enf, aes(reorder(Resource, share), share)) +
          geom_col(fill = catf_dark_blue, width = 0.72) +
          geom_text(aes(label = sprintf("%s  (%d of %d)", scales::percent(share, accuracy = 1), n_enf, n)),
                    hjust = -0.1, size = 2.9, color = "gray30") +
          coord_flip() + scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.34))) +
          labs(title = "Is the committed mitigation actually enforceable?",
               subtitle = "Share of a resource's mitigated FONSI conclusions tied to an enforceable permit condition",
               x = NULL, y = NULL,
               caption = "Resources with strong regulatory hooks (ESA, Clean Air/Water Acts) carry enforceable conditions; softer resources rely on non-binding commitments.") +
          theme_catf(),
      "fig_fonsi_enforceability.png", 8, 4.4)
  }

  cat("  wrote figures to", OUT, "\n")
}, error = function(e) cat("[figures error]", conditionMessage(e), "\n"))

# =====================================================================================
# EIS TRACK — above-the-line analysis (parallel to the FONSI section above).
# Self-contained: reads the *_eis parquets fresh; auto-runs whenever they exist (no flag).
# Same headline gate (primary_blm_doe_family + primary scope) and analytic grain
# (document x resource x class). Writes *_eis.csv tables + fig_*_eis.png figures so the report's
# EIS section mirrors the FONSI one. Skipped cleanly if the EIS extraction has not been run.
# =====================================================================================
if (file.exists(eis_det_path)) tryCatch({
  cat("\n--- EIS track analysis ---\n")
  edet <- read_parquet(eis_det_path)
  ethr <- if (file.exists(file.path(A, "determination_thresholds_eis.parquet")))
    read_parquet(file.path(A, "determination_thresholds_eis.parquet")) else NULL
  ABOVE <- c("significant_adverse", "significant_unavoidable", "eis_required")

  # EIS universe = ALL agencies, ALL eras. The descriptive EIS analysis (which resources cross the
  # line) does NOT use the decision date, so undated + pre-ARRA projects are kept rather than dropped;
  # BLM + DOE is called out only where a finding is agency-sensitive. (FONSI stays BLM+DOE primary,
  # because FONSI has only partial coverage outside it — see the report Methods.)
  eprimary <- edet %>% filter(!determination_class %in% NON_DET)
  # BLM+DOE, in-window subset — used ONLY for the like-for-like FONSI-vs-EIS comparison figure,
  # where both tracks must share a scope (FONSI is BLM+DOE-only).
  eis_bd <- eprimary %>% filter(agency_scope_status == "primary_blm_doe_family", analysis_scope == "primary")
  # analytic grain, carrying the EIS-only attributes for the factor/impact/alternative cuts
  edr <- eprimary %>%
    distinct(project_id, document_id, shared_resource_area, determination_class,
             alternative_name, significance_factor, impact_type, mitigation_dependent)
  edr_rc <- edr %>% distinct(project_id, document_id, shared_resource_area, determination_class)
  cat(sprintf("EIS determinations (all agencies)=%d  analytic(document x resource x class)=%d  projects=%d  (BLM+DOE subset=%d)\n",
              nrow(eprimary), nrow(edr_rc), n_distinct(eprimary$project_id), n_distinct(eis_bd$project_id)))

  # --- tables ---
  w(edr_rc %>% count(determination_class, name = "n_determinations") %>%
      left_join(eprimary %>% distinct(project_id, determination_class) %>%
                  count(determination_class, name = "n_projects"), by = "determination_class") %>%
      arrange(desc(n_determinations)), "eis_class_distribution.csv")

  # significant share by resource (the "which resources cross the line" table)
  eres <- edr_rc %>% filter(!shared_resource_area %in% c("project_wide", "unknown")) %>%
    group_by(shared_resource_area) %>%
    summarise(n = n(),
              n_adverse = sum(determination_class == "significant_adverse"),
              n_unavoid = sum(determination_class == "significant_unavoidable"),
              n_sig = sum(determination_class %in% ABOVE),
              share_sig = round(mean(determination_class %in% ABOVE), 3), .groups = "drop") %>%
    arrange(desc(share_sig)) %>% suppress(col = "n")
  w(eres, "eis_resource_significance.csv")

  # FONSI vs EIS: per resource, EIS-significant share vs FONSI-mitigation share. BOTH sides are the
  # BLM+DOE subset so the cross-track comparison is like-for-like (FONSI is BLM+DOE-only).
  # NOTE: fon_mit_share uses ALL primary_dr rows (including the rare significant_* FONSI anomalies),
  # so it differs slightly from mitigation_by_resource.csv's share_mit_class, whose denominator is
  # BELOW_LINE-only (6b above). The two tables are not drop-in interchangeable.
  fon_dr <- primary_dr %>% filter(!shared_resource_area %in% c("project_wide", "unknown"))
  fon_res <- fon_dr %>% group_by(shared_resource_area) %>%
    summarise(fon_n = n(), fon_mit = sum(determination_class == "less_than_significant_with_mitigation"),
              fon_mit_share = round(mean(determination_class == "less_than_significant_with_mitigation"), 3),
              .groups = "drop")
  eis_bd_res <- eis_bd %>%
    distinct(project_id, document_id, shared_resource_area, determination_class) %>%
    filter(!shared_resource_area %in% c("project_wide", "unknown")) %>%
    group_by(shared_resource_area) %>%
    summarise(eis_n = n(), eis_sig = sum(determination_class %in% ABOVE),
              eis_sig_share = round(mean(determination_class %in% ABOVE), 3), .groups = "drop")
  cmp <- eis_bd_res %>% full_join(fon_res, by = "shared_resource_area")
  w(cmp, "eis_fonsi_vs_eis.csv")

  # significance factors + impact type (above-line only) — the "why significant" cut
  w(edr %>% filter(determination_class %in% ABOVE, significance_factor != "") %>%
      distinct(project_id, document_id, shared_resource_area, determination_class, significance_factor) %>%
      count(significance_factor, name = "n") %>% arrange(desc(n)) %>% suppress(),
    "eis_significance_factors.csv")
  w(edr %>% filter(determination_class %in% ABOVE, impact_type != "") %>%
      distinct(project_id, document_id, shared_resource_area, determination_class, impact_type) %>%
      count(impact_type, name = "n") %>% arrange(desc(n)) %>% suppress(),
    "eis_impact_type.csv")

  # doc-level mitigation vs significance rates
  edoc <- eprimary %>% group_by(project_id, document_id) %>%
    summarise(mit = any(determination_class == "less_than_significant_with_mitigation"),
              sig = any(determination_class %in% ABOVE), .groups = "drop")
  w(edoc %>% summarise(n_documents = n(), n_projects = n_distinct(project_id),
                       n_mitigated = sum(mit), share_mitigated = round(mean(mit), 3),
                       n_with_significant = sum(sig), share_with_significant = round(mean(sig), 3)),
    "eis_mitigation_document_level.csv")

  # --- figures (reuse the FONSI figure environment: theme_catf, palette, savefig, res_label) ---
  if (exists("savefig")) {
    ecls_label <- c(no_significant_impact = "No significant impact",
      less_than_significant = "Less than significant",
      less_than_significant_with_mitigation = "Committed mitigation",
      significant_adverse = "Significant adverse", significant_unavoidable = "Significant unavoidable")

    # EIS validation dumbbell (mirror of the FONSI one; EIS scores are lower — task is harder)
    eval_fig <- tryCatch(read_parquet(file.path(A, "validation_metrics_eis.parquet")), error = function(e) NULL)
    if (!is.null(eval_fig) && nrow(eval_fig) > 0) {
      metric_lab <- c(candidate_is_determination = "Finds a determination",
        resource_determination_detection = "Assigns the right resource",
        determination_class_macro_f1 = "Gets the class right",
        mitigation_dependent_f1 = "Flags mitigation-dependence",
        primary_threshold_type_accuracy = "Identifies the threshold")
      vdat <- eval_fig %>% mutate(score = coalesce(f1, precision)) %>%
        filter(scope %in% c("overall", "holdout"), metric %in% names(metric_lab)) %>%
        mutate(Metric = factor(metric_lab[metric], levels = rev(unname(metric_lab))),
               Scope = ifelse(scope == "holdout", "Held-out test", "All 400"))
      vwide <- vdat %>% select(Metric, Scope, score) %>%
        tidyr::pivot_wider(names_from = Scope, values_from = score)
      savefig(ggplot(vdat, aes(score, Metric)) +
            # shade the bottom two rows (secondary attributes that matter less) — matches the FONSI figure
            annotate("rect", xmin = 0, xmax = 1.12, ymin = 0.5, ymax = 2.5, fill = "gray92", alpha = 0.7) +
            geom_vline(xintercept = 0.8, linetype = "dashed", color = "gray60") +
            geom_segment(data = vwide, aes(x = `All 400`, xend = `Held-out test`, y = Metric, yend = Metric),
                         inherit.aes = FALSE, color = "gray70", linewidth = 1) +
            geom_point(aes(color = Scope), size = 4.5, alpha = 0.75) +
            geom_text(data = dplyr::filter(vdat, Scope == "All 400"),
                      aes(label = sprintf("%.2f", score)), nudge_y = 0.24, size = 2.7, color = catf_magenta) +
            geom_text(data = dplyr::filter(vdat, Scope == "Held-out test"),
                      aes(label = sprintf("%.2f", score)), nudge_y = -0.24, size = 2.7, color = catf_dark_blue) +
            scale_color_manual(values = c("All 400" = catf_magenta, "Held-out test" = catf_dark_blue)) +
            scale_x_continuous(limits = c(0, 1.12), breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
            labs(title = "The EIS extraction was graded the same way — a harder task",
                 subtitle = "Agreement with the AI-human reviewed answer key: full sample vs the held-out test",
                 x = "Score (F1; threshold row = accuracy)", y = NULL, color = NULL,
                 caption = "Dashed line = 0.80. Blue = held-out test (the honest score); magenta = all 400.\nShaded rows are secondary attributes that matter less. EIS distinctions are harder than FONSI — the two coders agreed only ~58% on the class.") +
            theme_catf() + theme(legend.position = "bottom"),
        "fig_validation_accuracy_eis.png", 8, 4.8)
    }

    # Fig — which resources cross the line (significant share by resource, adverse + unavoidable stacked)
    ea <- eres %>% filter(!n_suppressed) %>%
      select(shared_resource_area, n, share_sig, n_adverse, n_unavoid) %>%
      tidyr::pivot_longer(c(n_adverse, n_unavoid), names_to = "band", values_to = "cnt") %>%
      mutate(Resource = relab(shared_resource_area, res_label),
             share_band = cnt / n,
             # order: adverse (less intense) left, unavoidable (more intense) right — see reverse below
             Band = factor(ifelse(band == "n_unavoid", "Significant unavoidable", "Significant adverse"),
                           levels = c("Significant adverse", "Significant unavoidable")))
    etot <- eres %>% filter(!n_suppressed) %>% mutate(Resource = relab(shared_resource_area, res_label))
    savefig(ggplot(ea, aes(reorder(Resource, share_sig), share_band, fill = Band)) +
          geom_col(width = 0.72, position = position_stack(reverse = TRUE)) +
          # per-segment label = each type's share AND count (of the resource's total determinations)
          geom_text(aes(label = ifelse(share_band >= 0.05, sprintf("%s (%d)",
                        scales::percent(share_band, accuracy = 1), cnt), "")),
                    position = position_stack(vjust = 0.5, reverse = TRUE), size = 2.4, color = "white",
                    fontface = "bold") +
          # end label = total significant share and count of / total determinations for the resource
          geom_text(data = etot, aes(x = reorder(Resource, share_sig), y = share_sig,
                    label = sprintf("%s significant  (%d of %s)", scales::percent(share_sig, accuracy = 1),
                                    n_sig, scales::comma(n))),
                    inherit.aes = FALSE, hjust = -0.1, size = 2.6, color = "gray30") +
          coord_flip() +
          scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.30))) +
          scale_fill_manual(values = c("Significant adverse" = catf_magenta,
                                       "Significant unavoidable" = catf_purple)) +
          labs(title = "Which resources cross the line",
               subtitle = "Share of each resource's EIS determinations judged significant — sorted (top = most likely to cross)",
               x = NULL, y = NULL, fill = NULL,
               caption = "Percentages (and counts) are of the resource's total EIS determinations. In-bar = each type's share and count; end label = total significant share. Intensity increases left→right: adverse, then unavoidable.") +
          theme_catf() + theme(legend.position = "bottom"),
      "fig_eis_above_line.png", 8, 5.5)

    # Fig — significant unavoidable, the wall (count lollipop; impacts mitigation can't erase)
    eun <- eres %>% filter(n_unavoid >= MIN_CELL) %>% mutate(Resource = relab(shared_resource_area, res_label))
    savefig(ggplot(eun, aes(reorder(Resource, n_unavoid), n_unavoid)) +
          geom_segment(aes(xend = reorder(Resource, n_unavoid), y = 0, yend = n_unavoid), color = "gray80") +
          geom_point(color = catf_purple, size = 4) +
          # fixed absolute offset so every count clears its dot (small dots near the axis don't overlap)
          geom_text(aes(label = n_unavoid), nudge_y = max(eun$n_unavoid) * 0.03, hjust = 0, size = 3, color = "gray30") +
          coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.10))) +
          labs(title = "The wall: significant unavoidable",
               subtitle = "Determinations judged significant that mitigation cannot bring below the line",
               x = NULL, y = "Significant-unavoidable determinations",
               caption = "Each bar counts a resource's significant-unavoidable determinations — e.g. 126 of visual's 316 significant determinations. Visual, biological, and air quality hit the wall most.") +
          theme_catf(),
      "fig_eis_unavoidable.png", 8, 4.6)

    # Fig — FONSI vs EIS: a quadrant scatter. x = mitigated below the line (FONSI), y = crosses the
    # line (EIS). The y=x diagonal separates "cross-over" resources (above) from "managed-below" (below).
    cmpf <- cmp %>% filter(!is.na(eis_sig_share), !is.na(fon_mit_share), eis_n >= 20, fon_n >= 20) %>%
      mutate(Resource = relab(shared_resource_area, res_label),
             Shape = factor(ifelse(eis_sig_share >= fon_mit_share,
                            "More likely to be significant (EIS)", "More likely to be mitigated (FONSI)"),
                            levels = c("More likely to be significant (EIS)", "More likely to be mitigated (FONSI)")))
    axmax <- max(cmpf$eis_sig_share, cmpf$fon_mit_share) * 1.15
    p15 <- ggplot(cmpf, aes(fon_mit_share, eis_sig_share)) +
      annotate("segment", x = 0, y = 0, xend = axmax, yend = axmax, linetype = "dashed", color = "gray70") +
      geom_point(aes(color = Shape, size = eis_n + fon_n), alpha = 0.85) +
      scale_color_manual(values = c("More likely to be significant (EIS)" = catf_magenta,
                                    "More likely to be mitigated (FONSI)" = catf_dark_blue), name = NULL) +
      scale_size_area(max_size = 8, guide = "none") +
      scale_x_continuous(labels = scales::percent, limits = c(0, axmax), expand = c(0, 0)) +
      scale_y_continuous(labels = scales::percent, limits = c(0, axmax), expand = c(0, 0)) +
      coord_fixed(ratio = 1) +
      guides(color = guide_legend(override.aes = list(size = 4))) +
      labs(title = "Two ways a resource can be a problem",
           subtitle = "Each resource: how often it crosses the line (EIS) vs is mitigated below it (FONSI) — BLM + DOE",
           x = "Mitigated below the line  (FONSI)", y = "Crosses the line  (EIS significant)",
           caption = "Dashed line = equal odds (a resource crosses as often as it is mitigated). Point size ≈ number of determinations.")
    p15 <- if (requireNamespace("ggrepel", quietly = TRUE))
      p15 + ggrepel::geom_text_repel(aes(label = Resource), size = 3, color = catf_navy, seed = 1,
              max.overlaps = 20, box.padding = 0.5, point.padding = 0.4, min.segment.length = 0, force = 2)
    else p15 + geom_text(aes(label = Resource), vjust = -1, size = 3, color = catf_navy)
    savefig(p15 + theme_catf() + theme(legend.position = "bottom"), "fig_fonsi_vs_eis.png", 7.5, 7.6)

    # Fig — why significant: factors + impact type (juxtaposed, mirror of corpus_overview layout)
    fac_lab <- c(magnitude = "Sheer magnitude", protected_resource = "Protected resource",
      cumulative = "Cumulative effect", regulatory_threshold = "Regulatory threshold",
      mitigable = "Significant but mitigable", geographic_extent = "Geographic extent",
      duration = "Duration / permanence", uncertainty = "Scientific uncertainty",
      controversy = "Controversy", none = "Unspecified")
    fac <- edr %>% filter(determination_class %in% ABOVE, significance_factor != "") %>%
      distinct(project_id, document_id, shared_resource_area, determination_class, significance_factor) %>%
      count(significance_factor, name = "n") %>% filter(n >= MIN_CELL) %>%
      mutate(Factor = ifelse(is.na(fac_lab[significance_factor]), significance_factor, fac_lab[significance_factor]))
    fac_order <- fac %>% arrange(n) %>% pull(Factor)
    # FIG A — factor totals (its own figure, larger)
    savefig(ggplot(fac, aes(factor(Factor, levels = fac_order), n)) +
          geom_col(fill = catf_magenta) + geom_text(aes(label = n), hjust = -0.2, size = 3.4, color = "gray30") +
          coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.10))) +
          labs(title = "Why an impact is significant",
               subtitle = "The factor cited behind each significant determination (one determination can cite several)",
               x = NULL, y = "Significant determinations") + theme_catf(),
      "fig_eis_factors.png", 8, 4.8)
    # FIG B — WHERE each factor bites. Colour is ROW-NORMALIZED (share within each factor) so the
    # concentrations pop per row (e.g. sheer magnitude -> visual/water/land use); the cell label keeps
    # the raw count for magnitude context.
    fr <- edr %>% filter(determination_class %in% ABOVE, !significance_factor %in% c("", "none"),
                         !shared_resource_area %in% c("project_wide", "unknown")) %>%
      distinct(project_id, document_id, shared_resource_area, determination_class, significance_factor) %>%
      count(significance_factor, shared_resource_area, name = "n") %>%
      mutate(Factor = factor(ifelse(is.na(fac_lab[significance_factor]), significance_factor,
                                    fac_lab[significance_factor]), levels = fac_order),
             Resource = relab(shared_resource_area, res_label)) %>%
      filter(!is.na(Factor)) %>%
      # drop tiny-total factors (e.g. controversy, n≈11) — row-normalizing them over-saturates a
      # near-empty row; they add noise, not signal.
      group_by(Factor) %>% filter(sum(n) >= 30) %>% mutate(row_share = n / sum(n)) %>% ungroup() %>%
      mutate(Factor = droplevels(Factor))
    res_ord <- fr %>% group_by(Resource) %>% summarise(t = sum(n), .groups = "drop") %>% arrange(t) %>% pull(Resource)
    fr_thr <- 0.55 * max(fr$row_share)   # white text on the darkest cells, relative to this grid's range
    savefig(ggplot(fr, aes(factor(Resource, levels = res_ord), Factor, fill = row_share)) +
          geom_tile(color = "white", linewidth = 0.7) +
          geom_text(aes(label = ifelse(n >= 5, n, ""), color = row_share > fr_thr), size = 2.7) +
          scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "gray15"), guide = "none") +
          scale_fill_gradientn(colors = c("#eef3fb", catf_light_blue, catf_dark_blue, catf_navy),
                               labels = scales::percent, name = "Share of the\nfactor's findings") +
          labs(title = "Where each factor bites",
               subtitle = "Each row shaded by share within that factor — darker = where that reason concentrates (cell = count)",
               x = NULL, y = NULL) +
          guides(fill = guide_colorbar(barheight = grid::unit(4, "cm"))) +
          theme_catf() + theme(axis.text.x = element_text(angle = 25, hjust = 1), panel.grid = element_blank(),
                               legend.position = "right"),
      "fig_eis_factor_heatmap.png", 9.5, 6.5)

    # Fig — breadth of significance per EIS (document/project level): when an EIS crosses the line,
    # on how many resource areas? (The FONSI breadth figure's project-level analog.)
    ebr <- edr_rc %>% filter(!shared_resource_area %in% c("project_wide", "unknown")) %>%
      group_by(project_id, document_id) %>%
      summarise(n_sig = n_distinct(shared_resource_area[determination_class %in% ABOVE]), .groups = "drop")
    brk <- ebr %>% filter(n_sig >= 1) %>%
      mutate(bucket = factor(ifelse(n_sig >= 7, "7+", as.character(n_sig)), levels = c(as.character(1:6), "7+"))) %>%
      count(bucket, name = "docs")
    n_cross <- sum(brk$docs); n_narrow <- sum(ebr$n_sig == 1); n_broad <- sum(ebr$n_sig >= 3); n_zero <- sum(ebr$n_sig == 0)
    w(brk, "eis_breadth.csv")
    savefig(ggplot(brk, aes(bucket, docs)) +
          geom_col(fill = catf_magenta, width = 0.78) +
          geom_text(aes(label = docs), vjust = -0.4, size = 3.2, color = "gray30") +
          scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
          labs(title = "When an EIS crosses the line, how broadly?",
               subtitle = sprintf("Distinct resource areas judged significant per EIS document (%s documents with a significant finding)",
                                  scales::comma(n_cross)),
               x = "Resource areas found significant", y = "EIS documents",
               caption = sprintf("Most EISs that cross the line do so narrowly (%s on a single resource); a long tail crosses on three or more (%s documents).",
                                 scales::comma(n_narrow), scales::comma(n_broad))) +
          theme_catf(),
      "fig_eis_breadth.png", 8, 4.6)

    # verbatim EXAMPLE tables — significant & unavoidable (per resource) and significance factors
    pick_examples <- function(df, grp, n_each = 2) df %>%
      mutate(example = str_squish(rationale_text), L = nchar(example)) %>%
      filter(!is.na(rationale_text), L >= 45, L <= 210) %>%
      distinct(.data[[grp]], example, .keep_all = TRUE) %>%
      arrange(.data[[grp]], L, example) %>% group_by(.data[[grp]]) %>%
      slice_head(n = n_each) %>% ungroup()
    eis_unav_ex <- eprimary %>%
      filter(determination_class == "significant_unavoidable",
             shared_resource_area %in% names(res_label), shared_resource_area != "unknown") %>%
      pick_examples("shared_resource_area") %>%
      left_join(eres %>% select(shared_resource_area, n_unavoid), by = "shared_resource_area") %>%
      mutate(Resource = relab(shared_resource_area, res_label)) %>%
      arrange(desc(n_unavoid), Resource, L) %>% select(Resource, example)
    w(eis_unav_ex, "eis_unavoidable_examples.csv")
    eis_fac_ex <- eprimary %>%
      filter(determination_class %in% ABOVE, !significance_factor %in% c("", "none")) %>%
      pick_examples("significance_factor") %>%
      mutate(Factor = ifelse(is.na(fac_lab[significance_factor]), significance_factor, fac_lab[significance_factor])) %>%
      left_join(fac %>% select(significance_factor, ftot = n), by = "significance_factor") %>%
      mutate(ftot = coalesce(ftot, 0L)) %>% arrange(desc(ftot), Factor, L) %>% select(Factor, example)
    w(eis_fac_ex, "eis_factor_examples.csv")

    # Fig — does it differ by agency? Which lead agencies most often cross the line (EIS is all-agency).
    lah <- tryCatch(read_parquet(file.path(A, "significance_corpus.parquet")) %>%
                      distinct(project_id, lead_agency_harmonized) %>%
                      mutate(agency = str_squish(str_remove_all(as.character(lead_agency_harmonized), '\\[|\\]|"'))),
                    error = function(e) NULL)
    if (!is.null(lah)) {
      ag_abbr <- c("Corps of Engineers--Civil Works" = "Army Corps", "Bureau of Land Management" = "BLM",
        "Bureau of Ocean Energy Management" = "BOEM", "Nuclear Regulatory Commission" = "NRC",
        "Department of Energy" = "DOE", "Forest Service" = "USFS", "Power Marketing Administration" = "Power Marketing",
        "United States Fish and Wildlife Service" = "USFWS", "Bureau of Indian Affairs" = "BIA",
        "Bureau of Reclamation" = "Reclamation", "Tennessee Valley Authority" = "TVA", "Navy, Marine Corps" = "Navy/USMC",
        "Federal Railroad Administration" = "FRA", "Rural Utilities Service" = "Rural Utilities",
        "National Aeronautics and Space Administration" = "NASA", "Energy Programs" = "DOE (Energy Programs)")
      doe_fam <- c("Department of Energy", "Power Marketing Administration", "Energy Programs",
                   "National Nuclear Security Administration")
      eag <- edr_rc %>% left_join(lah, by = "project_id") %>% filter(!is.na(agency), agency != "") %>%
        group_by(agency) %>%
        summarise(n = n(), n_sig = sum(determination_class %in% ABOVE),
                  share = mean(determination_class %in% ABOVE), proj = n_distinct(project_id), .groups = "drop") %>%
        filter(n >= 150) %>%
        mutate(Agency = ifelse(is.na(ag_abbr[agency]), agency, ag_abbr[agency]),
               Coverage = ifelse(agency == "Bureau of Land Management" | agency %in% doe_fam,
                                 "BLM + DOE (complete coverage)", "Other agency (partial coverage)"))
      w(eag %>% arrange(desc(share)) %>% select(agency, Agency, Coverage, n, n_sig, share, proj), "eis_agency.csv")
      savefig(ggplot(eag, aes(reorder(Agency, share), share, fill = Coverage)) +
            geom_col(width = 0.74) +
            geom_text(aes(label = sprintf("%s  (%d of %s)", scales::percent(share, accuracy = 1), n_sig,
                          scales::comma(n))), hjust = -0.1, size = 2.7, color = "gray30") +
            coord_flip() +
            scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.28))) +
            scale_fill_manual(values = c("BLM + DOE (complete coverage)" = catf_dark_blue,
                                         "Other agency (partial coverage)" = catf_light_blue), name = NULL) +
            labs(title = "Does crossing the line differ by agency?",
                 subtitle = "Share of a lead agency's EIS determinations judged significant (agencies with ≥150 determinations)",
                 x = NULL, y = NULL,
                 caption = "Bar label = significant share (significant of the agency's total). Land/water-facing agencies cross most, standardized nuclear/utility programs least. Only BLM + DOE have complete coverage; others partial.") +
            theme_catf() + theme(legend.position = "bottom"),
        "fig_eis_by_agency.png", 8.5, 6)
    }

    # Fig — EIS coverage & significance funnel (ALL agencies): how 753 corpus projects narrow to the
    # analyzed set, the date-status of that set, and how its determinations split above/below the line.
    n_corpus_eis <- tryCatch(nrow(distinct(filter(read_parquet(file.path(A, "significance_corpus.parquet")),
                                    process_type == "EIS"), project_id)), error = function(e) NA)
    n_cand_eis <- tryCatch(n_distinct(read_parquet(file.path(A, "significance_section_candidates_eis.parquet"))$project_id),
                           error = function(e) NA)
    n_analyzed <- n_distinct(eprimary$project_id)
    n_dets  <- nrow(edr_rc)
    n_above <- sum(edr_rc$determination_class %in% ABOVE)
    n_unav  <- sum(edr_rc$determination_class == "significant_unavoidable")
    date_lab <- c(in_scope_dated = "Dated, in-window", missing_decision_date = "No decision date",
                  pre_ARRA_dated = "Pre-2009 (pre-ARRA)", boundary_review = "Boundary")
    efun_date <- eprimary %>% distinct(project_id, time_scope_status) %>% count(time_scope_status) %>%
      mutate(Status = factor(ifelse(is.na(date_lab[time_scope_status]), time_scope_status,
                                     date_lab[time_scope_status]), levels = rev(unname(date_lab))))
    w(bind_rows(
        tibble(metric = c("corpus_projects","projects_with_sections","projects_analyzed",
                          "analytic_determinations","above_the_line","significant_unavoidable"),
               n = c(n_corpus_eis, n_cand_eis, n_analyzed, n_dets, n_above, n_unav)),
        efun_date %>% transmute(metric = paste0("date_", time_scope_status), n)), "eis_coverage_funnel.csv")

    efun_proj <- tibble(stage = factor(c("EIS corpus", "Sections retrieved", "Analyzed"),
                          levels = rev(c("EIS corpus", "Sections retrieved", "Analyzed"))),
                        n = c(n_corpus_eis, n_cand_eis, n_analyzed))
    pA <- ggplot(efun_proj, aes(n, stage)) + geom_col(fill = catf_navy, width = 0.62) +
      geom_text(aes(label = scales::comma(n)), hjust = -0.15, size = 3.4, color = "gray25") +
      scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
      labs(title = "Projects: corpus to analyzed", subtitle = "EIS projects retained at each step (all agencies)",
           x = NULL, y = NULL) + theme_catf()
    # nested funnel: all determinations ⊃ significant ⊃ significant-unavoidable. Light→hot as it
    # narrows to the intense subset (matches the report's significant=magenta, unavoidable=purple).
    det_stages <- c("All determinations", "Significant (crosses the line)", "Significant unavoidable")
    pC <- ggplot(tibble(stage = factor(det_stages, levels = rev(det_stages)),
                        n = c(n_dets, n_above, n_unav)), aes(n, stage, fill = stage)) +
      geom_col(width = 0.62) + geom_text(aes(label = scales::comma(n)), hjust = -0.15, size = 3.4, color = "gray25") +
      scale_fill_manual(values = setNames(c(catf_light_blue, catf_magenta, catf_purple), det_stages),
                        guide = "none") +
      scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
      labs(title = "Determinations: how many cross the line", subtitle = "From the analyzed EIS projects",
           x = NULL, y = NULL) + theme_catf()
    pB <- ggplot(efun_date, aes(x = 1, y = n, fill = Status)) + geom_col(width = 0.5) +
      geom_text(aes(label = ifelse(n >= 120, sprintf("%s: %d", Status, n), "")),
                position = position_stack(vjust = 0.5), color = "white", size = 3, fontface = "bold") +
      coord_flip() + scale_y_continuous(expand = c(0, 0)) +
      # brand sequential blues (data-completeness, not severity) — no clashing cyan/purple
      scale_fill_manual(values = c("Dated, in-window" = catf_navy, "No decision date" = catf_dark_blue,
                                   "Pre-2009 (pre-ARRA)" = catf_light_blue, "Boundary" = "gray75"),
                        breaks = c("Dated, in-window", "No decision date", "Pre-2009 (pre-ARRA)", "Boundary"),
                        name = NULL) +
      labs(title = "Do the analyzed projects have decision dates?",
           subtitle = "The EIS analysis is descriptive and does not use the date, so undated projects are kept",
           x = NULL, y = NULL) +
      theme_catf() + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
                           panel.grid = element_blank(), legend.position = "bottom")
    savefig((pA | pC) / pB + patchwork::plot_layout(heights = c(1, 0.6)), "fig_eis_funnel.png", 11, 6.6)

    # Fig — corpus overview: projects + documents (bars, top) over FONSI + EIS resource waffles
    # (bottom), the two waffles sharing one bottom legend. Built here so the EIS resource mix exists.
    corpus_fig <- tryCatch(read_parquet(file.path(A, "significance_corpus.parquet")), error = function(e) NULL)
    if (!is.null(corpus_fig)) {
      trk_lv <- c("FONSI (EA)", "EIS")   # FONSI left, EIS right — matches the waffles + FONSI-first report order
      proj <- corpus_fig %>% distinct(project_id, process_type) %>% count(process_type) %>%
        mutate(Track = factor(recode(process_type, EA = "FONSI (EA)", EIS = "EIS"), levels = trk_lv))
      docs <- bind_rows(
        det  %>% distinct(project_id, document_id) %>% summarise(n = n()) %>% mutate(Track = "FONSI (EA)"),
        edet %>% distinct(project_id, document_id) %>% summarise(n = n()) %>% mutate(Track = "EIS")) %>%
        mutate(Track = factor(Track, levels = trk_lv))
      bar_cols <- c("FONSI (EA)" = catf_dark_blue, "EIS" = catf_navy)
      mk_bar <- function(df, title, sub, ylab) ggplot(df, aes(Track, n, fill = Track)) +
        geom_col(width = 0.62) +
        geom_text(aes(label = scales::comma(n)), vjust = -0.4, size = 3.6, color = "gray30") +
        scale_y_continuous(expand = expansion(mult = c(0, 0.16))) +
        scale_fill_manual(values = bar_cols, guide = "none") +
        labs(title = title, subtitle = sub, x = NULL, y = ylab) + theme_catf()
      p_proj <- mk_bar(proj, "Scale of the analysis", "Clean-energy projects by review type", "Projects")
      p_docs <- mk_bar(docs, "Documents read", "Decision & supporting documents parsed for findings", "Documents")

      # fixed resource -> color map (same palette family as the FONSI waffle), SHARED by both waffles
      # so the collected legend is a single one; resources ordered by combined FONSI+EIS frequency.
      fon_res_ct <- primary_dr %>% filter(!shared_resource_area %in% c("project_wide", "unknown")) %>%
        count(shared_resource_area, name = "tot")
      eis_res_ct <- edr_rc %>% filter(!shared_resource_area %in% c("project_wide", "unknown")) %>%
        count(shared_resource_area, name = "tot")
      res_rank <- bind_rows(fon_res_ct, eis_res_ct) %>% group_by(shared_resource_area) %>%
        summarise(tot = sum(tot), .groups = "drop") %>% arrange(desc(tot)) %>%
        mutate(Resource = relab(shared_resource_area, res_label))
      # harmonized brand ramp: cool→warm through the report's own hues (light blue → blue → purple →
      # magenta), one shade per resource — no off-palette greens/reds.
      waffle_pal <- grDevices::colorRampPalette(c(catf_light_blue, catf_dark_blue, catf_purple, catf_magenta))(nrow(res_rank))
      waf_cols <- setNames(waffle_pal, res_rank$Resource)
      res_levels <- res_rank$Resource
      mk_waffle <- function(ct, title) {
        wfd <- ct %>% mutate(Resource = relab(shared_resource_area, res_label)) %>%
          arrange(match(Resource, res_levels)) %>% mutate(sq = round(100 * tot / sum(tot)))
        wfd$sq[which.max(wfd$tot)] <- wfd$sq[which.max(wfd$tot)] + (100 - sum(wfd$sq))
        wfd <- wfd %>% filter(sq > 0)
        g <- expand.grid(y = 1:10, x = 1:10)
        g$grp <- factor(rep(wfd$Resource, wfd$sq)[1:100], levels = res_levels)
        ggplot(g, aes(x, y, fill = grp)) + geom_tile(color = "white", linewidth = 1.1) + coord_equal() +
          scale_fill_manual(values = waf_cols, limits = res_levels, drop = FALSE, name = NULL) +
          labs(title = title) + theme_void(base_family = "Helvetica") +
          theme(plot.title = element_text(face = "bold", color = catf_navy, size = rel(1.15)),
                legend.text = element_text(size = rel(0.85)))
      }
      p_wf_fon <- mk_waffle(fon_res_ct, "What resources FONSIs cover")
      p_wf_eis <- mk_waffle(eis_res_ct, "What resources EISs cover")
      savefig(((p_proj | p_docs) / (p_wf_fon | p_wf_eis)) +
                patchwork::plot_layout(heights = c(0.7, 1.15), guides = "collect") &
                theme(legend.position = "bottom"),
              "fig_corpus_overview.png", 11, 8.8)
    }

    # Fig — EIS significant-share by clean-energy technology (which techs cross the line)
    if (exists("techmap") && !is.null(techmap)) {
      etech_dr <- edr_rc %>% filter(!shared_resource_area %in% c("project_wide", "unknown")) %>%
        left_join(techmap, by = "project_id") %>% filter(!is.na(tech), !tech %in% c("Other / mixed", "Renewable (other)"))
      etech <- etech_dr %>% group_by(tech) %>%
        summarise(n = n(), n_sig = sum(determination_class %in% ABOVE),
                  share = mean(determination_class %in% ABOVE), proj = n_distinct(project_id), .groups = "drop") %>%
        filter(n >= 60)
      w(etech %>% arrange(desc(share)), "eis_technology.csv")
      savefig(ggplot(etech, aes(reorder(tech, share), share)) +
            geom_col(fill = catf_magenta, width = 0.72) +
            geom_text(aes(label = sprintf("%s  (%d of %s)", scales::percent(share, accuracy = 1), n_sig,
                          scales::comma(n))), hjust = -0.1, size = 2.9, color = "gray30") +
            coord_flip() + scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.30))) +
            labs(title = "Which technologies cross the line?",
                 subtitle = "Share of a technology's EIS determinations judged significant",
                 x = NULL, y = NULL,
                 caption = "Technology from the dataset's project_type classification (one primary type per project; ~1 in 5 resolve to Other / mixed and are excluded).") +
            theme_catf(),
        "fig_eis_technology.png", 8, 4.4)

      # Fig — technology × resource "signature": row-normalized so each tech's crossing pattern shows.
      # Keep only techs with enough significant determinations (>=30) so a row isn't one saturated cell.
      tr <- etech_dr %>% filter(determination_class %in% ABOVE, tech %in% etech$tech) %>%
        count(tech, shared_resource_area, name = "n") %>%
        group_by(tech) %>% filter(sum(n) >= 30) %>% mutate(row_share = n / sum(n)) %>% ungroup() %>%
        mutate(Resource = relab(shared_resource_area, res_label))
      tech_ord <- etech %>% filter(tech %in% unique(tr$tech)) %>% arrange(share) %>% pull(tech)
      res_ord2 <- tr %>% group_by(Resource) %>% summarise(t = sum(n), .groups = "drop") %>% arrange(t) %>% pull(Resource)
      tr_thr <- 0.55 * max(tr$row_share)   # white text on the darkest cells, relative to this grid's range
      savefig(ggplot(tr, aes(factor(Resource, levels = res_ord2), factor(tech, levels = tech_ord), fill = row_share)) +
            geom_tile(color = "white", linewidth = 0.7) +
            geom_text(aes(label = ifelse(n >= 5, n, ""), color = row_share > tr_thr), size = 2.7) +
            scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "gray15"), guide = "none") +
            scale_fill_gradientn(colors = c("#eef3fb", catf_light_blue, catf_dark_blue, catf_navy),
                                 labels = scales::percent, name = "Share of the\ntech's crossings") +
            labs(title = "Each technology's significance signature",
                 subtitle = "Where each technology crosses the line — row-normalized (cell = count of significant determinations)",
                 x = NULL, y = NULL) +
            theme_catf() + theme(axis.text.x = element_text(angle = 25, hjust = 1), panel.grid = element_blank(),
                                 legend.position = "right"),
        "fig_eis_technology_resource.png", 9, 5)
    }

    # Fig + examples — significant-but-mitigable (the EIS analog of a mitigated FONSI: significant, but reducible)
    emit <- edr %>% filter(determination_class %in% ABOVE, significance_factor == "mitigable",
             !shared_resource_area %in% c("project_wide", "unknown")) %>%
      distinct(project_id, document_id, shared_resource_area) %>%
      count(shared_resource_area, name = "n") %>% filter(n >= MIN_CELL) %>%
      mutate(Resource = relab(shared_resource_area, res_label))
    w(emit %>% arrange(desc(n)), "eis_mitigable.csv")
    savefig(ggplot(emit, aes(reorder(Resource, n), n)) +
          geom_col(fill = catf_purple, width = 0.72) +
          geom_text(aes(label = n), hjust = -0.3, size = 3, color = "gray30") +
          coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
          labs(title = "Significant, but reducible",
               subtitle = "Significant EIS determinations the agency flags as still mitigable, by resource",
               x = NULL, y = "Determinations", caption = "The above-the-line analog of a mitigated FONSI: the impact crosses the line, but the agency notes mitigation can lessen it.") +
          theme_catf(),
      "fig_eis_mitigable.png", 8, 4.2)
    emit_ex <- eprimary %>%
      filter(determination_class %in% ABOVE, significance_factor == "mitigable") %>%
      pick_examples("shared_resource_area", n_each = 1) %>%
      mutate(Resource = relab(shared_resource_area, res_label)) %>%
      left_join(emit %>% select(shared_resource_area, nres = n), by = "shared_resource_area") %>%
      mutate(nres = coalesce(nres, 0L)) %>% arrange(desc(nres), Resource, L) %>% select(Resource, example)
    w(emit_ex, "eis_mitigable_examples.csv")

    cat("  wrote EIS figures to", OUT, "\n")
  }
}, error = function(e) cat("[EIS analysis error]", conditionMessage(e), "\n")) else
  cat("\n[EIS track skipped] no", eis_det_path, "\n")

cat("\nDone. Primary-scope tables in", OUT, "\n")
cat("Reminder: dry-run tables are illustrative; regenerate after the LLM pass + gold validation.\n")
