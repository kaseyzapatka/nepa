# Diagnose initiation coverage and likely failure modes

rm(list = ls())

source(here::here("code", "00_setup.R"))

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

coverage_summary <- function(df, decision_col, initiation_col) {
  total <- nrow(df)
  dec <- sum(!is.na(df[[decision_col]]) & df[[decision_col]] != "")
  init <- sum(!is.na(df[[initiation_col]]) & df[[initiation_col]] != "")
  tibble(
    total = total,
    decision = dec,
    decision_pct = round(dec / total * 100, 1),
    initiation = init,
    initiation_pct = round(init / total * 100, 1)
  )
}

# ---- Inputs ----
bert_path <- "data/analysis/test20_bert.parquet"
llm_path <- "data/analysis/test20_workers.parquet"

bert <- read_results(bert_path)
llm <- read_results(llm_path)

# ---- Coverage ----
print("BERT coverage")
print(coverage_summary(bert, "bert_decision_date", "bert_application_date"))

print("LLM coverage")
print(coverage_summary(llm, "llm_decision_date", "llm_application_date"))

# ---- Context diagnostics ----
bert_ctx <- extract_contexts(bert, "bert_dates_json", "bert")
llm_ctx <- extract_contexts(llm, "llm_dates_json", "llm")

all_ctx <- bind_rows(bert_ctx, llm_ctx) %>%
  mutate(source = str_squish(source))

# simple cue buckets to spot weak initiation signals
cue_map <- list(
  scoping = "scoping|notice of intent|noi",
  application = "application received|submitted|submittal",
  consultation = "consultation|initiated|initiation|start of the review|review process started",
  draft_prep = "draft|prepared|preparation|revised|reviewed|document creation"
)

bucketed <- all_ctx %>%
  mutate(cue_bucket = case_when(
    str_detect(source, cue_map$scoping) ~ "scoping",
    str_detect(source, cue_map$application) ~ "application",
    str_detect(source, cue_map$consultation) ~ "consultation",
    str_detect(source, cue_map$draft_prep) ~ "draft_prep",
    TRUE ~ "other"
  ))

print("Initiation contexts by cue bucket")
print(bucketed %>%
  filter(type == "initiation") %>%
  count(model, cue_bucket, sort = TRUE))

# Spot likely false initiation (decision-like cues)
print("Initiation contexts that look decision-like")
print(bucketed %>%
  filter(type == "initiation") %>%
  filter(str_detect(source, "signed|signature|authorizing|compliance officer|approved")) %>%
  select(project_id, model, date, source) %>%
  distinct() %>%
  slice_head(n = 20))
