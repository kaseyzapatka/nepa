# --------------------------
# VALIDATION: COMPARE REGEX RUNS
# --------------------------
# Side-by-side comparison of baseline vs refactored regex candidates.
# Set SRC below to "CE", "EA", or "EIS" — everything else updates automatically.

rm(list = ls())
source(here::here("code", "validation", "timeline", "00_setup.R"))

# --------------------------
# CONFIGURE: change SRC to switch sources
# --------------------------
SRC <- "EIS"   # "CE", "EA", or "EIS"

paths      <- regex_paths[[SRC]]
baseline   <- load_candidates(paths$baseline,   label = "baseline",   source = SRC)
refactored <- load_candidates(paths$refactored, label = "refactored", source = SRC)

cat(sprintf("\n=== Comparing %s: baseline vs refactored ===\n", SRC))

both <- bind_rows(baseline, refactored)

# --------------------------
# 1. HIGH-LEVEL COUNTS
# --------------------------
cat("\n── Row counts ───────────────────────────────────────────\n")
both |>
  count(run_label, name = "n_rows") |>
  mutate(delta = n_rows - n_rows[run_label == "baseline"]) |>
  print()

cat("\n── Project counts ───────────────────────────────────────\n")
both |>
  group_by(run_label) |>
  summarise(n_projects = n_distinct(project_id), .groups = "drop") |>
  print()

# --------------------------
# 2. PER-PROJECT CANDIDATE COUNTS
# --------------------------
per_project <- both |>
  count(run_label, project_id) |>
  pivot_wider(names_from = run_label, values_from = n, values_fill = 0) |>
  mutate(delta = refactored - baseline)

cat("\n── Per-project delta summary ────────────────────────────\n")
per_project |>
  summarise(
    projects_identical = sum(delta == 0),
    projects_gained    = sum(delta > 0),
    projects_lost      = sum(delta < 0),
    median_delta       = median(delta),
    max_gained         = max(delta),
    max_lost           = min(delta)
  ) |>
  print()

cat("\n── Top gainers (refactored has more candidates) ─────────\n")
per_project |> arrange(desc(delta)) |> slice_head(n = 10) |> print()

cat("\n── Top losers (refactored has fewer candidates) ─────────\n")
per_project |> arrange(delta) |> slice_head(n = 10) |> print()

# --------------------------
# 3. SIDE-BY-SIDE FOR A SPECIFIC PROJECT
# --------------------------
# Inspect any project_id from the tables above
inspect_project <- function(pid) {
  cat("\n── Project:", pid, "────────────────────────────────────────\n")
  cat("BASELINE:\n")
  baseline |>
    filter(project_id == pid) |>
    select(date, match, position_pct, doc_type) |>
    arrange(position_pct) |>
    print(n = 30)
  cat("REFACTORED:\n")
  refactored |>
    filter(project_id == pid) |>
    select(date, match, position_pct, doc_type, main_document_imputed) |>
    arrange(position_pct) |>
    print(n = 30)
}

# Automatically show one top gainer and one top loser
top_gainer <- per_project |> slice_max(delta, n = 1) |> pull(project_id)
top_loser  <- per_project |> slice_min(delta, n = 1) |> pull(project_id)

inspect_project(top_gainer)
inspect_project(top_loser)


# To inspect any other project, call:
#   inspect_project("your-project-id-here")

# --------------------------
# 4. DELTA DISTRIBUTION PLOT
# --------------------------
p_delta <- per_project |>
  filter(delta != 0) |>
  ggplot(aes(x = delta)) +
  geom_histogram(binwidth = 1, fill = source_palette[[SRC]], colour = "white") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = catf_colors["grey"]) +
  labs(
    title    = sprintf("%s — per-project candidate count change: refactored vs baseline", SRC),
    subtitle = sprintf(
      "%d projects changed  |  median delta = %+.0f  |  range [%+d, %+d]",
      sum(per_project$delta != 0),
      median(per_project$delta),
      min(per_project$delta),
      max(per_project$delta)
    ),
    x = "Δ candidates (refactored − baseline)",
    y = "Number of projects"
  ) +
  theme_nepa()

print(p_delta)

# --------------------------
# 5. MAIN_DOCUMENT_IMPUTED (EA / EIS only)
# --------------------------
if (SRC %in% c("EA", "EIS")) {
  cat(sprintf("\n── main_document_imputed (fallback projects) — %s ──────\n", SRC))
  refactored |>
    count(main_document_imputed) |>
    mutate(pct = round(n / sum(n) * 100, 1)) |>
    print()
}

# --------------------------
# 6. DATE RANGE COMPARISON
# --------------------------
cat("\n── Date range by run ────────────────────────────────────\n")
both |>
  filter(!is.na(date)) |>
  group_by(run_label) |>
  summarise(
    min_date     = min(date),
    max_date     = max(date),
    pct_pre1990  = round(mean(date < as.Date("1990-01-01")) * 100, 1),
    pct_post2030 = round(mean(date > as.Date("2030-01-01")) * 100, 1),
    .groups = "drop"
  ) |>
  print()

# --------------------------
# EXPLORATORY
# --------------------------
refactored |> 
  #select(project_id, date, context) |> 
  select(date, context) |> 
  slice_sample(n = 10) |> 
  print()
