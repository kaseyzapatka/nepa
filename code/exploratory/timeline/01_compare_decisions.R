# Compare decision extraction side-by-side (BERT vs LLM)

rm(list = ls())

source(here::here("code", "00_setup.R"))

library(dplyr)
library(purrr)
library(jsonlite)
library(stringr)
library(tidyr)
library(tibble)

safe_fromJSON <- function(x) {
  tryCatch(fromJSON(x, flatten = TRUE), error = function(e) NULL)
}

normalize_parsed <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.data.frame(x)) return(as_tibble(x))
  if (is.list(x)) {
    # Handle list-of-lists (JSON array) or named list (single object)
    if (!is.null(names(x)) && length(names(x)) > 0) {
      return(as_tibble(x))
    }
    return(bind_rows(lapply(x, as_tibble)))
  }
  NULL
}

read_results <- function(path) {
  df <- arrow::read_parquet(here::here(path))
  df
}

extract_contexts <- function(df, json_col, model_label) {
  df %>%
    mutate(parsed = map(.data[[json_col]], safe_fromJSON)) %>%
    mutate(parsed = map(parsed, normalize_parsed)) %>%
    select(project_id, project_title, lead_agency, parsed) %>%
    unnest(parsed) %>%
    mutate(model = model_label) %>%
    select(project_id, project_title, lead_agency, model, type, date, source, confidence, everything())
}

compare_decisions <- function(
  bert_path = "data/analysis/test20_bert.parquet",
  llm_path = "data/analysis/test20_workers.parquet",
  llm_json_col = "llm_dates_json",
  bert_json_col = "bert_dates_json",
  n_projects = 10,
  seed = 42
) {
  set.seed(seed)

  bert <- read_results(bert_path)
  llm <- read_results(llm_path)

  bert_ctx <- extract_contexts(bert, bert_json_col, "bert")
  llm_ctx <- extract_contexts(llm, llm_json_col, "llm")

  common_ids <- intersect(bert_ctx$project_id, llm_ctx$project_id)
  sample_ids <- sample(common_ids, min(n_projects, length(common_ids)))

  side_by_side <- bind_rows(bert_ctx, llm_ctx) %>%
    filter(project_id %in% sample_ids, type == "decision") %>%
    mutate(source = str_squish(source)) %>%
    arrange(project_id, model, date)

  side_by_side
}

# ---- Run comparison ----
side_by_side <- compare_decisions()

# Print a compact view
side_by_side %>%
  select(project_id, model, date, confidence, source) %>%
  print(n = 100)

# Write to CSV for review
readr::write_csv(
  side_by_side,
  here::here("data", "analysis", "compare_decisions_side_by_side.csv")
)
