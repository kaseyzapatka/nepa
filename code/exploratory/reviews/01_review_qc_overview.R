# --------------------------
# REVIEWS QA 01: OVERVIEW CHECKS
# --------------------------
# Purpose:
#   1) sanity-check review classification outputs
#   2) flag potentially inconsistent records for manual review
#
# Outputs:
#   output/exploratory/reviews/01_counts_review_type.csv
#   output/exploratory/reviews/01_counts_review_source.csv
#   output/exploratory/reviews/01_issue_summary.csv
#   output/exploratory/reviews/01_qa_row_flags.csv
rm(list=ls())

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
is_blank <- function(x) str_squish(to_chr(x)) == ""

reviews <- read_parquet(reviews_path) %>%
  as_tibble() %>%
  mutate(
    review_type = str_to_lower(to_chr(project_review_type)),
    review_source = str_to_lower(to_chr(project_review_source)),
    tiers_from = to_chr(project_review_tiers_from),
    match_text = to_chr(project_review_match_text),
    is_programmatic_flag = as.logical(project_review_is_programmatic)
  )

expected_review_types <- c("standard", "programmatic", "tiered")
expected_sources <- c(
  "title", "doc_metadata", "text_regex", "llm",
  "no_documents", "no_pages", "error_reading_pages", "error", "unknown", "none"
)

qa <- reviews %>%
  mutate(
    issue_unknown_review_type = !review_type %in% expected_review_types,
    issue_unknown_source = !review_source %in% expected_sources,
    issue_programmatic_flag_mismatch =
      (review_type == "programmatic" & !coalesce(is_programmatic_flag, FALSE)) |
      (review_type != "programmatic" & coalesce(is_programmatic_flag, FALSE)),
    issue_tiered_missing_parent = review_type == "tiered" & is_blank(tiers_from),
    issue_standard_has_parent = review_type == "standard" & !is_blank(tiers_from),
    issue_nonstandard_missing_match =
      review_type %in% c("programmatic", "tiered") & is_blank(match_text)
  )
qa |> glimpse()

dup_ids <- qa %>%
  count(project_id, name = "n") %>%
  filter(n > 1)

if (nrow(dup_ids) > 0) {
  qa <- qa %>%
    left_join(dup_ids %>% mutate(issue_duplicate_project_id = TRUE), by = "project_id") %>%
    mutate(issue_duplicate_project_id = coalesce(issue_duplicate_project_id, FALSE))
} else {
  qa <- qa %>% mutate(issue_duplicate_project_id = FALSE)
}

type_counts <- qa %>%
  count(dataset_source, review_type, name = "n") %>%
  arrange(dataset_source, review_type) |> 
  print()

source_counts <- qa %>%
  count(review_source, name = "n") %>%
  arrange(desc(n)) |> 
  print()

issue_summary <- tibble(
  issue = c(
    "unknown_review_type",
    "unknown_source",
    "programmatic_flag_mismatch",
    "tiered_missing_parent",
    "standard_has_parent",
    "nonstandard_missing_match",
    "duplicate_project_id"
  ),
  n_projects = c(
    sum(qa$issue_unknown_review_type, na.rm = TRUE),
    sum(qa$issue_unknown_source, na.rm = TRUE),
    sum(qa$issue_programmatic_flag_mismatch, na.rm = TRUE),
    sum(qa$issue_tiered_missing_parent, na.rm = TRUE),
    sum(qa$issue_standard_has_parent, na.rm = TRUE),
    sum(qa$issue_nonstandard_missing_match, na.rm = TRUE),
    sum(qa$issue_duplicate_project_id, na.rm = TRUE)
  )
) |> 
  print()

row_flags <- qa %>%
  mutate(
    any_issue =
      issue_unknown_review_type |
      issue_unknown_source |
      issue_programmatic_flag_mismatch |
      issue_tiered_missing_parent |
      issue_standard_has_parent |
      issue_nonstandard_missing_match |
      issue_duplicate_project_id,
    issue_list = pmap_chr(
      list(
        issue_unknown_review_type,
        issue_unknown_source,
        issue_programmatic_flag_mismatch,
        issue_tiered_missing_parent,
        issue_standard_has_parent,
        issue_nonstandard_missing_match,
        issue_duplicate_project_id
      ),
      function(a, b, c, d, e, f, g) {
        issues <- c(
          if (a) "unknown_review_type",
          if (b) "unknown_source",
          if (c) "programmatic_flag_mismatch",
          if (d) "tiered_missing_parent",
          if (e) "standard_has_parent",
          if (f) "nonstandard_missing_match",
          if (g) "duplicate_project_id"
        )
        paste(issues, collapse = "; ")
      }
    )
  ) %>%
  filter(any_issue) %>%
  select(
    project_id, dataset_source, project_title, review_type, review_source,
    is_programmatic_flag, tiers_from, match_text, issue_list
  ) %>%
  arrange(dataset_source, review_type, project_title) |> 
  print()

write_csv(type_counts, here(out_dir, "01_counts_review_type.csv"))
write_csv(source_counts, here(out_dir, "01_counts_review_source.csv"))
write_csv(issue_summary, here(out_dir, "01_issue_summary.csv"))
write_csv(row_flags, here(out_dir, "01_qa_row_flags.csv"))

cat("\n[01_review_qc_overview] complete\n")
cat("  total projects:", nrow(qa), "\n")
cat("  flagged projects:", nrow(row_flags), "\n")
print(issue_summary)
