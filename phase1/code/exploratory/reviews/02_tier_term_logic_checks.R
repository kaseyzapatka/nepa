# --------------------------
# REVIEWS QA 02: TIER TERM LOGIC CHECKS
# --------------------------
# Purpose:
#   Validate tier terminology logic assumptions:
#   - Tier 1 language should generally map to programmatic reviews
#   - Tier 2 language should generally map to tiered reviews
#
# Outputs:
#   output/exploratory/reviews/02_tier_term_summary.csv
#   output/exploratory/reviews/02_tier1_not_programmatic.csv
#   output/exploratory/reviews/02_tier2_not_tiered.csv

library(here)
library(arrow)
library(tidyverse)

reviews_path <- here("phase1", "data", "analysis", "projects_reviews.parquet")
out_dir <- here("phase1", "output", "exploratory", "reviews")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(reviews_path)) {
  stop("Missing input file: ", reviews_path)
}

to_chr <- function(x) if_else(is.na(x), "", as.character(x))

tier1_pattern <- regex("\\btier\\s*(?:1|i|one)\\b", ignore_case = TRUE)
tier2_pattern <- regex("\\btier\\s*(?:2|ii|two)\\b", ignore_case = TRUE)

reviews <- read_parquet(reviews_path) %>%
  as_tibble() %>%
  transmute(
    project_id,
    dataset_source,
    project_title = to_chr(project_title),
    review_type = str_to_lower(to_chr(project_review_type)),
    review_source = to_chr(project_review_source),
    tiers_from = to_chr(project_review_tiers_from),
    match_text = to_chr(project_review_match_text),
    tiers_from_context = to_chr(project_review_tiers_from_context),
    text_for_qc = str_squish(str_c(
      project_title, " || ", match_text, " || ",
      tiers_from, " || ", tiers_from_context
    ))
  ) %>%
  mutate(
    has_tier1 = str_detect(text_for_qc, tier1_pattern),
    has_tier2 = str_detect(text_for_qc, tier2_pattern)
  )

summary_tbl <- reviews %>%
  count(review_type, has_tier1, has_tier2, name = "n") %>%
  arrange(review_type, desc(has_tier1), desc(has_tier2))

tier1_not_programmatic <- reviews %>%
  filter(has_tier1, review_type != "programmatic") %>%
  mutate(
    check_reason = "contains_tier1_but_not_programmatic",
    qc_excerpt = str_trunc(text_for_qc, 400)
  ) %>%
  select(
    check_reason, project_id, dataset_source, project_title,
    review_type, review_source, tiers_from, match_text, qc_excerpt
  ) %>%
  arrange(dataset_source, review_type, project_title)

tier2_not_tiered <- reviews %>%
  filter(has_tier2, review_type != "tiered") %>%
  mutate(
    check_reason = "contains_tier2_but_not_tiered",
    qc_excerpt = str_trunc(text_for_qc, 400)
  ) %>%
  select(
    check_reason, project_id, dataset_source, project_title,
    review_type, review_source, tiers_from, match_text, qc_excerpt
  ) %>%
  arrange(dataset_source, review_type, project_title)

write_csv(summary_tbl, here(out_dir, "02_tier_term_summary.csv"))
write_csv(tier1_not_programmatic, here(out_dir, "02_tier1_not_programmatic.csv"))
write_csv(tier2_not_tiered, here(out_dir, "02_tier2_not_tiered.csv"))

cat("\n[02_tier_term_logic_checks] complete\n")
cat("  tier1_not_programmatic:", nrow(tier1_not_programmatic), "\n")
cat("  tier2_not_tiered:", nrow(tier2_not_tiered), "\n")
