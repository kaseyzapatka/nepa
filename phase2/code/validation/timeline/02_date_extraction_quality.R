# --------------------------
# VALIDATION: DATE EXTRACTION QUALITY
# --------------------------
# Inspect the regex candidate dates across all sources (CE, EA, EIS).
# Covers: year distribution, match patterns, position in document,
# coverage gaps, and potential false positives / noise.

rm(list = ls())
source(here::here("code", "validation", "timeline", "00_setup.R"))

# --------------------------
# LOAD ALL SOURCES
# --------------------------
# Uses refactored path when available, falls back to baseline
pick_path <- function(src) {
  new <- regex_paths[[src]]$refactored
  if (file.exists(new)) new else regex_paths[[src]]$baseline
}

ce  <- load_candidates(pick_path("CE"),  label = "CE",  source = "CE")
ea  <- load_candidates(pick_path("EA"),  label = "EA",  source = "EA")
eis <- load_candidates(pick_path("EIS"), label = "EIS", source = "EIS")

cat("Paths used:\n")
cat(" CE: ",  pick_path("CE"),  "\n")
cat(" EA: ",  pick_path("EA"),  "\n")
cat(" EIS:", pick_path("EIS"), "\n")

all_cands <- bind_rows(ce, ea, eis)

cat("Loaded:\n")
all_cands |> count(source) |> print()

# --------------------------
# 1. COVERAGE: projects with zero candidates
# --------------------------
cat("\n── Coverage: projects with at least one candidate ───────\n")
# We need the full project list to compute the denominator
projects <- read_parquet(here("data", "analysis", "projects_combined.parquet")) |>
  as_tibble() |>
  filter(project_energy_type == "Clean") |>
  select(project_id, dataset_source)

coverage <- projects |>
  left_join(
    all_cands |> count(project_id, name = "n_candidates"),
    by = "project_id"
  ) |>
  mutate(has_candidates = !is.na(n_candidates))

coverage |>
  group_by(dataset_source) |>
  summarise(
    n_projects         = n(),
    with_candidates    = sum(has_candidates),
    coverage_pct       = round(mean(has_candidates) * 100, 1),
    median_candidates  = median(n_candidates, na.rm = TRUE),
    .groups = "drop"
  ) |>
  print()

# --------------------------
# 2. YEAR DISTRIBUTION
# --------------------------
cat("\n── Date year distribution ───────────────────────────────\n")
year_dist <- all_cands |>
  filter(!is.na(date)) |>
  mutate(year = as.integer(format(date, "%Y"))) |>
  filter(year >= 1970, year <= 2030)   # filter obvious outliers for display

p_year <- year_dist |>
  count(source, year) |>
  ggplot(aes(x = year, y = n, fill = source)) +
  geom_col() +
  facet_wrap(~source, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = source_palette, guide = "none") +
  scale_x_continuous(breaks = seq(1970, 2030, 5)) +
  labs(
    title    = "Year distribution of extracted date candidates",
    subtitle = "Obvious outliers (<1970, >2030) excluded from plot",
    x = "Year", y = "Candidate count"
  ) +
  theme_nepa()

print(p_year)

# Flag suspicious date ranges
cat("\n── Suspicious dates ─────────────────────────────────────\n")
all_cands |>
  filter(!is.na(date)) |>
  mutate(year = as.integer(format(date, "%Y"))) |>
  summarise(
    pre_1980    = sum(year < 1980),
    post_2030   = sum(year > 2030),
    future      = sum(date > Sys.Date()),
    .groups = "drop"
  ) |>
  print()

# --------------------------
# 3. MATCH PATTERN FREQUENCY
# --------------------------
cat("\n── Top 30 match patterns ────────────────────────────────\n")
# Normalise matches to pattern type: "Month YYYY", "Month DD, YYYY", etc.
all_cands |>
  mutate(
    pattern = case_when(
      str_detect(match, "^\\d{1,2}/\\d{1,2}/\\d{2,4}$")       ~ "MM/DD/YYYY",
      str_detect(match, "^\\d{4}-\\d{2}-\\d{2}$")              ~ "YYYY-MM-DD",
      str_detect(match, "^[A-Za-z]+ \\d{1,2}, \\d{4}$")        ~ "Month DD, YYYY",
      str_detect(match, "^[A-Za-z]+ \\d{4}$")                  ~ "Month YYYY",
      str_detect(match, "^[A-Za-z]+ \\d{1,2}(st|nd|rd|th)")    ~ "Month DDth ...",
      str_detect(match, "\\d{4}")                               ~ "Other with year",
      TRUE                                                       ~ "Other"
    )
  ) |>
  count(source, pattern, sort = TRUE) |>
  group_by(source) |>
  mutate(pct = round(n / sum(n) * 100, 1)) |>
  ungroup() |>
  pivot_wider(names_from = source, values_from = c(n, pct), values_fill = 0) |>
  arrange(desc(n_CE)) |>
  print(n = 30)

# --------------------------
# 4. POSITION DISTRIBUTION
# --------------------------
p_pos <- all_cands |>
  filter(!is.na(position_pct)) |>
  ggplot(aes(x = position_pct, fill = source)) +
  geom_histogram(binwidth = 5, colour = "white") +
  facet_wrap(~source, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = source_palette, guide = "none") +
  labs(
    title    = "Where in documents are dates found?",
    subtitle = "position_pct = 0 is document start, 100 is end",
    x = "Position in document (%)", y = "Candidate count"
  ) +
  theme_nepa()

print(p_pos)

# --------------------------
# 5. DOC TYPE BREAKDOWN (EA / EIS)
# --------------------------
cat("\n── Candidate counts by doc_type (EA + EIS) ─────────────\n")
bind_rows(ea, eis) |>
  filter(!is.na(doc_type), doc_type != "") |>
  count(source, doc_type, sort = TRUE) |>
  group_by(source) |>
  mutate(pct = round(n / sum(n) * 100, 1)) |>
  ungroup() |>
  print(n = 30)

# --------------------------
# 6. CANDIDATES PER PROJECT DISTRIBUTION
# --------------------------
p_cands <- all_cands |>
  count(source, project_id) |>
  ggplot(aes(x = n, fill = source)) +
  geom_histogram(binwidth = 5, colour = "white") +
  facet_wrap(~source, scales = "free_y", ncol = 1) +
  scale_fill_manual(values = source_palette, guide = "none") +
  coord_cartesian(xlim = c(0, 150)) +
  labs(
    title    = "Candidates per project (capped at 150 for display)",
    subtitle = "Very high counts may indicate noise-heavy documents",
    x = "Candidate count", y = "Projects"
  ) +
  theme_nepa()

print(p_cands)

# Projects with very high candidate counts — potential noise
cat("\n── Projects with >100 candidates (top 15) ───────────────\n")
all_cands |>
  count(source, project_id, sort = TRUE) |>
  filter(n > 100) |>
  slice_head(n = 15) |>
  print()

