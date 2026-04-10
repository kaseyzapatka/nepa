# Generate client-facing samples for review

rm(list = ls())

source(here::here("phase1", "code", "00_setup.R"))

library(dplyr)
library(purrr)
library(jsonlite)
library(stringr)

safe_fromJSON <- function(x) {
  tryCatch(fromJSON(x, flatten = TRUE), error = function(e) NULL)
}

read_results <- function(path) {
  arrow::read_parquet(here::here(path))
}

extract_contexts <- function(df, json_col, model_label) {
  df %>%
    mutate(parsed = map(.data[[json_col]], safe_fromJSON)) %>%
    select(project_id, project_title, lead_agency, parsed) %>%
    unnest(parsed) %>%
    mutate(model = model_label) %>%
    select(project_id, project_title, lead_agency, model, type, date, source, confidence, everything())
}

make_samples <- function(
  path,
  json_col,
  model_label,
  n_per_class = 5,
  seed = 123
) {
  set.seed(seed)
  df <- read_results(path)
  ctx <- extract_contexts(df, json_col, model_label) %>%
    mutate(source = str_squish(source))

  classes <- c("decision", "initiation", "other")
  out <- purrr::map_dfr(classes, function(cls) {
    ctx %>%
      filter(type == cls) %>%
      distinct(project_id, date, source, .keep_all = TRUE) %>%
      slice_sample(n = min(n_per_class, n()))
  })

  out
}

# ---- Inputs ----
# Set these to the run you want to send to clients
run_path <- "data/analysis/test50_bert_v8.parquet"
json_col <- "bert_dates_json"
model_label <- "llm"

samples <- make_samples(run_path, json_col, model_label, n_per_class = 5)

readr::write_csv(
  samples,
  here::here("phase1", "data", "analysis", "client_timeline_samples.csv")
)

samples %>%
  select(project_id, model, type, date, source) %>%
  print(n = 50)
