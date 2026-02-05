# Compare decision extraction side-by-side (BERT vs LLM)

rm(list = ls())

source(here::here("code", "00_setup.R"))

library(dplyr)
library(purrr)
library(jsonlite)
library(stringr)
library(tidyr)
library(tibble)
library(googlesheets4)

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
  #bert_path = "data/analysis/test20_bert.parquet",
  #bert_path = "data/analysis/test20_bert_v2.parquet",
  bert_path = "data/analysis/test20_bert_v6.parquet",
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
    #filter(project_id %in% sample_ids, type == "decision") %>%
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
#readr::write_csv(
#  side_by_side,
#  here::here("data", "analysis", "compare_decisions_side_by_side.csv")
#)

# Write to google sheets for review
sheet_write(
  data = side_by_side,
  ss = "https://docs.google.com/spreadsheets/d/1HuvVNDiPAG3WegTy58yn_LLUQ8RnSFwTg0BeabcyM08/edit?usp=sharing",
  sheet = "side_by_side_v03"
)


# bert analysis 
#bert_path = "data/analysis/test20_bert_v6.parquet"
#bert_path = "data/analysis/test50_bert.parquet"
#bert_path = "data/analysis/test50_bert_v2.parquet"
bert_path = "data/analysis/test50_bert_v8.parquet"
bert <- read_results(bert_path)
bert_json_col = "bert_dates_json"
bert_ctx <- extract_contexts(bert, bert_json_col, "bert")

bert_ctx |> 
  distinct(model) |> 
  glimpse() # misses extra dates

bert_ctx |> 
  select(project_id) |> 
  slice_sample(n = 1 ) |> 
  print()


#
# Examples for Feb 5 meeting 
# --------------------------------------------------
example1 <- 
  bert_ctx |> 
  filter(project_id == "3e3bb9f5-f5ab-651d-b2d1-50ec99d99db0") |> 
  #select(project_title, model:context_cleaned_flag) |> 
  select(type, date, source) |> 
  arrange(date) |> 
  print()

# Write to google sheets for review
sheet_write(
  data = example1,
  ss = "https://docs.google.com/spreadsheets/d/1HuvVNDiPAG3WegTy58yn_LLUQ8RnSFwTg0BeabcyM08/edit?usp=sharing",
  sheet = "example1"
)

example2 <- 
  bert_ctx |> 
  filter(project_id == "46f4da85-af1c-0e66-a706-9a7292dd9689") |> 
  #select(project_title, model:context_cleaned_flag) |> 
  select(type, date, source) |> 
  arrange(date) |> 
  print()

# Write to google sheets for review
sheet_write(
  data = example2,
  ss = "https://docs.google.com/spreadsheets/d/1HuvVNDiPAG3WegTy58yn_LLUQ8RnSFwTg0BeabcyM08/edit?usp=sharing",
  sheet = "example2"
)

example3 <- 
  bert_ctx |> 
  filter(project_id == "824ba268-8ddf-a34f-f9a7-625e7727c242") |> 
  #select(project_title, model:context_cleaned_flag) |> 
  select(type, date, source) |> 
  arrange(date) |> 
  print()

# Write to google sheets for review
sheet_write(
  data = example3,
  ss = "https://docs.google.com/spreadsheets/d/1HuvVNDiPAG3WegTy58yn_LLUQ8RnSFwTg0BeabcyM08/edit?usp=sharing",
  sheet = "example3"
)

example4 <- 
  bert_ctx |> 
  filter(project_id == "f2812da0-16c5-fbd1-9e16-10bf8e67c514") |> 
  #select(project_title, model:context_cleaned_flag) |> 
  select(type, date, source) |> 
  arrange(date) |> 
  print()

# Write to google sheets for review
sheet_write(
  data = example4,
  ss = "https://docs.google.com/spreadsheets/d/1HuvVNDiPAG3WegTy58yn_LLUQ8RnSFwTg0BeabcyM08/edit?usp=sharing",
  sheet = "example4"
)

example5 <- 
  bert_ctx |> 
  filter(project_id == "dec68c6f-da24-f178-7bf9-30dcd886fb12") |> 
  #select(project_title, model:context_cleaned_flag) |> 
  select(type, date, source) |> 
  arrange(date) |> 
  print()


# Write to google sheets for review
sheet_write(
  data = example5,
  ss = "https://docs.google.com/spreadsheets/d/1HuvVNDiPAG3WegTy58yn_LLUQ8RnSFwTg0BeabcyM08/edit?usp=sharing",
  sheet = "example5"
)


example6 <- 
  bert_ctx |> 
  filter(project_id == "5c512493-33a9-ff2c-5f13-3a8d55464b93") |> 
  #select(project_title, model:context_cleaned_flag) |> 
  select(type, date, source) |> 
  arrange(date) |> 
  print()


# Write to google sheets for review
sheet_write(
  data = example6,
  ss = "https://docs.google.com/spreadsheets/d/1HuvVNDiPAG3WegTy58yn_LLUQ8RnSFwTg0BeabcyM08/edit?usp=sharing",
  sheet = "example6"
)

