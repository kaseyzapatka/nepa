# --------------------------
# REVIEWS QA 03: TIERED LINKAGE CHECKS
# --------------------------
# Purpose:
#   Check whether tiered projects have usable parent-link information.
#   Focuses on:
#   - missing or weak `project_review_tiers_from`
#   - whether tiering context appears to mention the extracted parent
#
# Outputs:
#   output/exploratory/reviews/03_tiered_parent_frequency.csv
#   output/exploratory/reviews/03_tiered_link_quality.csv
#   output/exploratory/reviews/03_tiered_link_manual_review.csv

library(here)
library(arrow)
library(tidyverse)

reviews_path <- here("data", "analysis", "projects_reviews.parquet")
out_dir <- here("output", "exploratory", "reviews")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(reviews_path)) {
  stop("Missing input file: ", reviews_path)
}

to_chr <- function(x) if_else(is.na(x), "", as.character(x))
norm_text <- function(x) {
  x %>%
    str_to_lower() %>%
    str_replace_all("[^a-z0-9 ]", " ") %>%
    str_squish()
}

tiered <- read_parquet(reviews_path) %>%
  as_tibble() %>%
  filter(str_to_lower(to_chr(project_review_type)) == "tiered") %>%
  transmute(
    project_id,
    dataset_source,
    project_title = to_chr(project_title),
    review_source = to_chr(project_review_source),
    tiers_from = to_chr(project_review_tiers_from),
    tiers_from_context = to_chr(project_review_tiers_from_context),
    match_text = to_chr(project_review_match_text)
  ) %>%
  mutate(
    parent_norm = norm_text(tiers_from),
    context_norm = norm_text(tiers_from_context),
    parent_words = str_count(parent_norm, "\\S+"),
    parent_has_key_terms = str_detect(
      parent_norm,
      regex("peis|pea|programmatic|environmental|impact statement|assessment|plan|eis|ea|generic", ignore_case = TRUE)
    ),
    parent_missing = parent_norm == "",
    parent_too_short = parent_words < 3,
    # exact-string mention is strict; useful to find weak extractions
    context_mentions_parent = if_else(
      parent_missing,
      FALSE,
      str_detect(context_norm, fixed(parent_norm))
    ),
    link_quality = case_when(
      parent_missing ~ "missing_parent",
      parent_too_short & !parent_has_key_terms ~ "weak_parent_text",
      !context_mentions_parent ~ "parent_not_explicit_in_context",
      TRUE ~ "ok"
    )
  )

parent_frequency <- tiered %>%
  mutate(parent_display = if_else(parent_norm == "", "(missing)", tiers_from)) %>%
  count(parent_display, name = "n_projects", sort = TRUE)

link_quality <- tiered %>%
  select(
    project_id, dataset_source, project_title, review_source,
    tiers_from, match_text, tiers_from_context,
    parent_words, parent_has_key_terms, context_mentions_parent, link_quality
  ) %>%
  arrange(link_quality, dataset_source, project_title)

manual_review <- link_quality %>%
  filter(link_quality != "ok")

write_csv(parent_frequency, here(out_dir, "03_tiered_parent_frequency.csv"))
write_csv(link_quality, here(out_dir, "03_tiered_link_quality.csv"))
write_csv(manual_review, here(out_dir, "03_tiered_link_manual_review.csv"))

cat("\n[03_tier_linkage_checks] complete\n")
cat("  tiered projects:", nrow(link_quality), "\n")
cat("  manual review rows:", nrow(manual_review), "\n")
print(count(link_quality, link_quality))
