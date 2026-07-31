# D4: Sample-check candidates for eyeballing.
# Pulls ~20 projects per review type (CE/EA/EIS), stratified across coverage states
# (complete / missing-init / missing-decision / missing-both), and lists EVERY candidate
# for those projects with its role, dates, model probabilities, rank, and selected flags —
# so you can eyeball whether the right initiation/decision dates are being picked.
#
# Output:
#   phase2/output/deliverable04/sample_check_candidates.csv   (one row per candidate)
#   phase2/output/deliverable04/sample_check_projects.csv     (one row per project: selected dates)
# Console: per-project blocks (selected dates + candidate count).
#
# Usage:  Rscript phase2/code/deliverable04/09_sample_check.R [seed]

suppressPackageStartupMessages({
  library(arrow); library(dplyr); library(readr); library(tidyr); library(stringr)
})

PHASE2 <- here::here("phase2")
DATA   <- file.path(PHASE2, "data", "analysis", "timeline")
OUT    <- file.path(PHASE2, "output", "deliverable04")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
args <- commandArgs(trailingOnly = TRUE)
seed <- if (length(args) >= 1) as.integer(args[1]) else 42
N_PER_STATE <- 5  # up to 5 projects per (process x coverage state) -> ~20 per process

proj <- read_parquet(file.path(DATA, "timeline_project_dates.parquet")) |>
  filter(process_type %in% c("CE", "EA", "EIS")) |>
  mutate(state = case_when(
    !is.na(initiation_date) & !is.na(decision_date) ~ "complete",
    is.na(initiation_date)  & !is.na(decision_date) ~ "missing_initiation",
    !is.na(initiation_date) & is.na(decision_date)  ~ "missing_decision",
    TRUE                                            ~ "missing_both"))

set.seed(seed)
samp <- proj |>
  group_by(process_type, state) |>
  slice_sample(n = N_PER_STATE) |>
  ungroup()
ids <- samp$project_id
message("Sampled ", length(ids), " projects (", N_PER_STATE, "/state/process, seed ", seed, ")")

# Project-level summary (the "likely" selected dates)
proj_out <- samp |>
  transmute(process_type, project_id, state,
            initiation_date, initiation_is_proxy, decision_date, decision_is_proxy,
            has_rod, decision_is_feis_fallback, timeline_status, timeline_flags) |>
  arrange(process_type, state, project_id)
write_csv(proj_out, file.path(OUT, "sample_check_projects.csv"))

# Candidate-level detail for those projects
cands <- read_parquet(file.path(DATA, "timeline_candidates.parquet")) |>
  filter(project_id %in% ids) |>
  transmute(process_type, project_id, candidate_role, parsed_date, date_granularity,
            p_init_cal = suppressWarnings(as.numeric(p_init_cal)),
            p_dec_cal  = suppressWarnings(as.numeric(p_dec_cal)),
            ranking_score = suppressWarnings(as.numeric(ranking_score)),
            selected_for_initiation, selected_for_decision,
            raw_date_text,
            context = str_trunc(str_squish(coalesce(context_text, "")), 180)) |>
  left_join(select(samp, project_id, state), by = "project_id") |>
  arrange(process_type, state, project_id,
          desc(selected_for_decision), desc(selected_for_initiation),
          desc(coalesce(p_init_cal, p_dec_cal, 0)))
write_csv(cands, file.path(OUT, "sample_check_candidates.csv"))

# Console: readable per-project blocks
cat("\n==== SAMPLE CHECK (", length(ids), "projects ) ====\n")
for (pr in c("CE", "EA", "EIS")) {
  cat("\n########## ", pr, " ##########\n")
  pp <- proj_out |> filter(process_type == pr)
  for (i in seq_len(nrow(pp))) {
    r <- pp[i, ]
    cat(sprintf("\n[%s] %s  (%s)\n  SELECTED init=%s%s  decision=%s%s\n",
                r$state, r$project_id, r$timeline_status,
                ifelse(is.na(r$initiation_date), "—", as.character(r$initiation_date)),
                ifelse(isTRUE(r$initiation_is_proxy), " (proxy)", ""),
                ifelse(is.na(r$decision_date), "—", as.character(r$decision_date)),
                ifelse(isTRUE(r$decision_is_feis_fallback), " (FEIS-fallback)",
                       ifelse(isTRUE(r$decision_is_proxy), " (proxy)", ""))))
    cc <- cands |> filter(project_id == r$project_id)
    if (nrow(cc) == 0) { cat("    (no candidates)\n"); next }
    for (j in seq_len(min(nrow(cc), 12))) {
      c <- cc[j, ]
      sel <- paste0(ifelse(isTRUE(c$selected_for_initiation), "I", "."),
                    ifelse(isTRUE(c$selected_for_decision), "D", "."))
      cat(sprintf("    %s %-16s %-10s pi=%-4s pd=%-4s | %s\n",
                  sel, str_trunc(c$candidate_role, 16),
                  ifelse(is.na(c$parsed_date), "—", as.character(c$parsed_date)),
                  ifelse(is.na(c$p_init_cal), "—", sprintf("%.2f", c$p_init_cal)),
                  ifelse(is.na(c$p_dec_cal), "—", sprintf("%.2f", c$p_dec_cal)),
                  str_trunc(c$context, 90)))
    }
    if (nrow(cc) > 12) cat(sprintf("    ... (%d more candidates; see CSV)\n", nrow(cc) - 12))
  }
}
cat("\nWrote: ", file.path(OUT, "sample_check_candidates.csv"), "\n      ",
    file.path(OUT, "sample_check_projects.csv"), "\n")
