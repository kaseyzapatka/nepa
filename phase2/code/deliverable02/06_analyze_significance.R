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
# Run:  Rscript phase2/code/deliverable02/06_analyze_significance.R
suppressMessages({library(arrow); library(dplyr); library(tidyr); library(readr); library(stringr)})

A <- "phase2/data/analysis/deliverable02"
OUT <- "phase2/output/deliverable02/analysis"; dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
MIN_CELL <- 5
NON_DET <- c("not_a_determination", "ambiguous")

# FONSI track is the default; pass --with-eis to also fold in the EIS track
# (04 writes *_eis.parquet so the two tracks never clobber each other).
WITH_EIS <- "--with-eis" %in% commandArgs(trailingOnly = TRUE)

det <- read_parquet(file.path(A, "significance_determinations.parquet"))
thr <- read_parquet(file.path(A, "determination_thresholds.parquet"))
eis_det_path <- file.path(A, "significance_determinations_eis.parquet")
if (WITH_EIS && file.exists(eis_det_path)) {
  det <- bind_rows(det, read_parquet(eis_det_path))
  eis_thr_path <- file.path(A, "determination_thresholds_eis.parquet")
  if (file.exists(eis_thr_path)) thr <- bind_rows(thr, read_parquet(eis_thr_path))
  cat("combined FONSI + EIS tracks. determinations by process_type:\n")
  print(table(det$process_type))
} else if (WITH_EIS) {
  cat("--with-eis passed but no", eis_det_path, "found — FONSI only.\n")
} else {
  cat("FONSI track only (pass --with-eis to combine the EIS track).\n")
}

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
  res_label <- c(air_quality="Air quality", water="Water", biological="Biological",
    cultural="Cultural / historic", visual="Visual", noise="Noise", soils_geology="Soils / geology",
    socioeconomic="Socioeconomic", transportation="Transportation", land_use="Land use",
    climate_ghg="Climate / GHG", public_health="Public health", unknown="Unplaced")
  class_label <- c(no_significant_impact="No significant impact",
    less_than_significant="Less than significant",
    less_than_significant_with_mitigation="LTS with mitigation")
  relab <- function(x, m) ifelse(is.na(m[x]), x, m[x])
  savefig <- function(p, name, w = 8, h = 5)
    suppressMessages(ggsave(file.path(OUT, name), p, width = w, height = h, dpi = 300))

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
                                     "LTS with mitigation" = catf_magenta)) +
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
               subtitle = "Agreement with the human answer key: full sample vs the held-out test",
               x = "Score (F1; threshold row = accuracy)", y = NULL, color = NULL,
               caption = "Dashed line = 0.80, the standard bar. Blue = held-out test (the honest score); magenta = all 400.\nShaded rows are secondary attributes that matter less to the findings.") +
          theme_catf() + theme(legend.position = "bottom"),
      "fig_validation_accuracy.png", 8, 4.8)
  }
  # Fig — corpus at a glance: FONSI vs EIS magnitude + FONSI resource waffle (juxtaposed)
  corpus_fig <- tryCatch(read_parquet(file.path(A, "significance_corpus.parquet")), error = function(e) NULL)
  if (!is.null(corpus_fig)) {
    mag <- corpus_fig %>% distinct(project_id, process_type) %>% count(process_type) %>%
      mutate(Track = recode(process_type, EA = "FONSI (EA)", EIS = "EIS"))
    p_mag <- ggplot(mag, aes(Track, n, fill = Track)) +
      geom_col(width = 0.62) + geom_text(aes(label = scales::comma(n)), vjust = -0.4, size = 3.6, color = "gray30") +
      scale_y_continuous(expand = expansion(mult = c(0, 0.14))) +
      scale_fill_manual(values = c("FONSI (EA)" = catf_dark_blue, "EIS" = catf_navy), guide = "none") +
      labs(title = "Scale of the analysis", subtitle = "Clean-energy projects by review type",
           x = NULL, y = "Projects") + theme_catf()
    # manual 10x10 waffle of the FONSI resource mix (top 6 + Other)
    top6 <- otot %>% arrange(desc(tot)) %>% slice_head(n = 6) %>% pull(Resource)
    wfd <- otot %>% mutate(grp = ifelse(Resource %in% top6, as.character(Resource), "Other")) %>%
      group_by(grp) %>% summarise(tot = sum(tot), .groups = "drop") %>%
      mutate(is_other = grp == "Other") %>% arrange(is_other, desc(tot)) %>%
      mutate(sq = round(100 * tot / sum(tot)))
    wfd$sq[1] <- wfd$sq[1] + (100 - sum(wfd$sq))
    wgrid <- expand.grid(y = 1:10, x = 1:10)
    wgrid$grp <- factor(rep(wfd$grp, wfd$sq)[1:100], levels = wfd$grp)
    waf_cols <- setNames(c(catf_navy, catf_dark_blue, catf_blue, catf_teal, catf_lime,
                           catf_magenta, "gray75")[seq_len(nrow(wfd))], wfd$grp)
    p_waf <- ggplot(wgrid, aes(x, y, fill = grp)) +
      geom_tile(color = "white", linewidth = 1.1) + coord_equal() +
      scale_fill_manual(values = waf_cols, name = NULL) +
      labs(title = "What resources FONSIs cover", subtitle = "Each square ≈ 1% of determinations") +
      theme_void(base_family = "Helvetica") +
      theme(plot.title = element_text(face = "bold", color = catf_navy, size = rel(1.15)),
            plot.subtitle = element_text(color = catf_dark_blue, size = rel(0.85)),
            legend.text = element_text(size = rel(0.8)))
    savefig((p_mag | p_waf) + patchwork::plot_layout(widths = c(0.8, 1.15)),
            "fig_corpus_overview.png", 11, 4.6)
  }

  # Fig — regulatory-threshold profile (descriptive; threshold ID is the least-accurate field)
  thr_lab <- c(other_quantitative = "Other quantitative", wetland_floodplain = "Wetland / floodplain",
    NHPA_adverse_effect = "NHPA §106 adverse effect", visual_vrm = "Visual (VRM)", ESA_take = "ESA take",
    NAAQS = "NAAQS (air)", ESA_jeopardy = "ESA jeopardy", noise_threshold = "Noise threshold", PSD = "PSD (air)")
  tprof <- thr %>% semi_join(distinct(primary, determination_instance_id), by = "determination_instance_id") %>%
    filter(!threshold_type %in% c("none", "unknown", "")) %>% count(threshold_type) %>%
    mutate(Threshold = ifelse(is.na(thr_lab[threshold_type]), threshold_type, thr_lab[threshold_type]))
  savefig(tprof %>% ggplot(aes(reorder(Threshold, n), n)) +
        geom_col(fill = catf_teal) + geom_text(aes(label = n), hjust = -0.2, size = 3, color = "gray30") +
        coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, 0.09))) +
        labs(title = "Which regulatory thresholds the conclusions lean on",
             subtitle = "Threshold citations across primary FONSI determinations",
             x = NULL, y = "Citations",
             caption = "Descriptive only: threshold identification is the pipeline's least-accurate field (see validation).\nExcludes conclusions not anchored to a specific threshold.") + theme_catf(),
    "fig_threshold_profile.png", 8, 4.5)

  # example mitigations — up to 2 per resource area, ordered by how mitigation-heavy the resource is
  ex <- primary %>%
    filter(shared_resource_area %in% names(res_label), shared_resource_area != "unknown",
           determination_class == "less_than_significant_with_mitigation", !is.na(rationale_text)) %>%
    mutate(example = str_squish(rationale_text), L = nchar(example)) %>%
    filter(L >= 40, L <= 240) %>%
    arrange(shared_resource_area, L) %>%
    group_by(shared_resource_area) %>% slice_head(n = 2) %>% ungroup() %>%
    left_join(mitr %>% select(shared_resource_area, share), by = "shared_resource_area") %>%
    mutate(share = coalesce(share, 0), Resource = relab(shared_resource_area, res_label)) %>%
    arrange(desc(share), Resource, L) %>%
    select(Resource, example)
  w(ex, "mitigation_examples.csv")

  cat("  wrote figures to", OUT, "\n")
}, error = function(e) cat("[figures error]", conditionMessage(e), "\n"))

cat("\nDone. Primary-scope tables in", OUT, "\n")
cat("Reminder: dry-run tables are illustrative; regenerate after the LLM pass + gold validation.\n")
