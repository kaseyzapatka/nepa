# --------------------------
# REVIEWS QA 04: MANUAL VALIDATION PACKET
# --------------------------
# Purpose:
#   Create a compact, human-readable packet for direct spot-checking.
#   This script avoids heavy page-level reads and instead surfaces the
#   extracted evidence text used by the classifier.
#
# What to review:
#   1) Programmatic rows: evidence should indicate broad plan-level review
#      (PEA/PEIS/generic/tier 1), not a project-specific tiering statement.
#   2) Tiered rows: evidence should show explicit tiering language and a
#      plausible parent review in `tiers_from`.
#   3) Standard controls: evidence should not look programmatic/tiered.
#
# Outputs:
#   output/exploratory/reviews/04_validation_projects.csv
#   output/exploratory/reviews/04_validation_packet.csv

library(here)
library(arrow)
library(tidyverse)

set.seed(42)

reviews_path <- here("data", "analysis", "projects_reviews.parquet")
ea_docs_path <- here("data", "processed", "ea", "documents.parquet")
eis_docs_path <- here("data", "processed", "eis", "documents.parquet")
out_dir <- here("output", "exploratory", "reviews")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

n_programmatic_per_source <- 10
n_standard_per_source <- 10

if (!file.exists(reviews_path)) stop("Missing: ", reviews_path)

to_chr <- function(x) if_else(is.na(x), "", as.character(x))
is_blank <- function(x) str_squish(to_chr(x)) == ""
flatten_struct_chr <- function(x) {
  if (is.data.frame(x) && "value" %in% names(x)) return(as.character(x$value))
  if (is.list(x)) {
    return(map_chr(x, function(el) {
      if (is.null(el)) return(NA_character_)
      if (is.data.frame(el) && "value" %in% names(el)) return(as.character(el$value[[1]]))
      if (is.list(el) && !is.null(el$value)) return(as.character(el$value))
      as.character(el)
    }))
  }
  as.character(x)
}

# 1) Read project-level review outputs
reviews <- read_parquet(reviews_path) %>%
  as_tibble() %>%
  transmute(
    project_id,
    dataset_source,
    project_title = to_chr(project_title),
    process_type = to_chr(process_type),
    review_type = str_to_lower(to_chr(project_review_type)),
    review_source = to_chr(project_review_source),
    project_review_confidence = to_chr(project_review_confidence),
    is_programmatic_flag = as.logical(project_review_is_programmatic),
    tiers_from = to_chr(project_review_tiers_from),
    tiers_from_context = to_chr(project_review_tiers_from_context),
    match_text = to_chr(project_review_match_text)
  )

# 2) Build a manageable validation set
validation_projects <- bind_rows(
  reviews %>%
    filter(review_type == "tiered") %>%
    mutate(validation_group = "tiered_all"),
  reviews %>%
    filter(review_type == "programmatic") %>%
    group_by(dataset_source) %>%
    group_modify(~ slice_sample(.x, n = min(nrow(.x), n_programmatic_per_source))) %>%
    ungroup() %>%
    mutate(validation_group = "programmatic_sample"),
  reviews %>%
    filter(review_type == "standard") %>%
    group_by(dataset_source) %>%
    group_modify(~ slice_sample(.x, n = min(nrow(.x), n_standard_per_source))) %>%
    ungroup() %>%
    mutate(validation_group = "standard_control")
) %>%
  distinct(project_id, .keep_all = TRUE)

write_csv(
  validation_projects %>%
    select(
      project_id, dataset_source, project_title, process_type,
      review_type, review_source, project_review_confidence,
      tiers_from, match_text, validation_group
    ),
  here(out_dir, "04_validation_projects.csv")
)

# 3) Attach one primary document per project (for easy manual drill-down)
docs_primary <- bind_rows(
  read_parquet(ea_docs_path) %>% mutate(dataset_source = "EA"),
  read_parquet(eis_docs_path) %>% mutate(dataset_source = "EIS")
) %>%
  as_tibble() %>%
  mutate(
    project_id = flatten_struct_chr(project_id),
    document_title = to_chr(document_title),
    file_name = to_chr(file_name),
    main_flag = str_to_upper(to_chr(main_document)) == "YES",
    total_pages_num = suppressWarnings(as.numeric(total_pages))
  ) %>%
  semi_join(validation_projects %>% select(project_id, dataset_source), by = c("project_id", "dataset_source")) %>%
  group_by(dataset_source, project_id) %>%
  arrange(desc(main_flag), desc(total_pages_num), document_id, .by_group = TRUE) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  transmute(
    dataset_source,
    project_id,
    primary_file_name = file_name,
    primary_document_title = document_title,
    primary_document_pages = total_pages_num
  )

# 4) Construct evidence text and QC helper flags
packet <- validation_projects %>%
  mutate(
    evidence_text = case_when(
      review_type == "tiered" & !is_blank(tiers_from_context) ~ tiers_from_context,
      !is_blank(match_text) ~ match_text,
      !is_blank(tiers_from) ~ tiers_from,
      TRUE ~ ""
    ),
    evidence_source = case_when(
      review_type == "tiered" & !is_blank(tiers_from_context) ~ "project_review_tiers_from_context",
      !is_blank(match_text) ~ "project_review_match_text",
      !is_blank(tiers_from) ~ "project_review_tiers_from",
      TRUE ~ "none"
    ),
    evidence_lc = str_to_lower(evidence_text),
    has_programmatic_terms = str_detect(
      evidence_lc,
      regex("\\b(programmatic|peis|pea|generic|tier\\s*(1|i|one))\\b", ignore_case = TRUE)
    ),
    has_tier_link_terms = str_detect(
      evidence_lc,
      regex("\\b(tier(s|ed|ing)?\\s+(to|from)|incorporat(e|es|ed|ing)\\s+by\\s+reference)\\b", ignore_case = TRUE)
    ),
    has_tier2_terms = str_detect(evidence_lc, regex("\\btier\\s*(2|ii|two)\\b", ignore_case = TRUE)),
    tiers_from_words = str_count(str_squish(tiers_from), "\\S+"),
    qc_flag = case_when(
      review_type == "programmatic" & !has_programmatic_terms ~ "programmatic_missing_programmatic_terms",
      review_type == "programmatic" & has_tier2_terms ~ "programmatic_mentions_tier2_check",
      review_type == "tiered" & is_blank(tiers_from) ~ "tiered_missing_parent",
      review_type == "tiered" & tiers_from_words < 3 ~ "tiered_parent_too_short",
      review_type == "tiered" & !has_tier_link_terms ~ "tiered_missing_tiering_language",
      TRUE ~ ""
    ),
    qc_focus = case_when(
      review_type == "programmatic" ~ "Confirm this is a broad plan-level review (PEA/PEIS/generic/tier 1).",
      review_type == "tiered" ~ "Confirm this tiers from a specific parent review and `tiers_from` is accurate.",
      TRUE ~ "Confirm this looks standard (no programmatic/tiering language)."
    )
  ) %>%
  left_join(docs_primary, by = c("dataset_source", "project_id")) %>%
  select(
    validation_group, dataset_source, project_id, project_title, process_type,
    review_type, review_source, project_review_confidence, is_programmatic_flag,
    tiers_from, tiers_from_context, match_text,
    evidence_source, evidence_text,
    has_programmatic_terms, has_tier_link_terms, has_tier2_terms,
    qc_flag, qc_focus,
    primary_file_name, primary_document_title, primary_document_pages
  ) %>%
  arrange(desc(qc_flag != ""), validation_group, dataset_source, review_type, project_title)

write_csv(packet, here(out_dir, "04_validation_packet.csv"))

cat("\n[04_manual_validation_packet] complete\n")
cat("  projects in packet:", nrow(packet), "\n")
cat("  flagged rows:", sum(packet$qc_flag != ""), "\n")
cat("  output:", here(out_dir, "04_validation_packet.csv"), "\n")
