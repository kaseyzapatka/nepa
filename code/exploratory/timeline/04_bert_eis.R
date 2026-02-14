# --------------------------
# EXPLORATORY: BERT EIS 
# --------------------------
# Explore how well BERT-based timeline extraction worked for clean energy EA projects
# Full run: data/analysis/projects_timeline_bert.parquet

# TAB
#2.28 - 2.17 #| .11

# --------------------------
# SETUP
# --------------------------
rm(list = ls()) 
source(here::here("code", "deliverable03", "00_setup.R"))

# --------------------------
# FUNCTIONS
# --------------------------

safe_fromJSON <- function(x) {
  tryCatch(fromJSON(x, flatten = TRUE), error = function(e) NULL)
}

normalize_parsed <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.data.frame(x)) return(as_tibble(x))
  if (is.list(x)) {
    if (!is.null(names(x)) && length(names(x)) > 0) {
      return(as_tibble(x))
    }
    return(bind_rows(lapply(x, as_tibble)))
  }
  NULL
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

show_timeline <- function(df, pid) {
  df |> 
  filter(project_id == pid) |>
  select("type","final_flag", "date","source") |> 
  arrange(date) |> 
  print(n = 500)
}

final_timeline <- function(df, pid) {
  df |> 
  filter(type %in% c("initiation","decision") & final_flag == TRUE ) |>
  filter(project_id == pid) |>
  select("type","final_flag", "date","source") |> 
  arrange(date) |> 
  print(n = 50)
}

# --------------------------
# LOAD BERT TIMELINE DATA
# --------------------------
# settings
bert_timeline_path <- here("data", "analysis", "test50_eis.parquet")
#bert_timeline_path <- here("data", "analysis", "test50_eis_llm.parquet")
bert_json_col = "bert_dates_json"

# extract
data <- read_parquet(bert_timeline_path)
timeline <- extract_contexts(data, bert_json_col, "timeline")

# glimpse
timeline |> glimpse()

# --------------------------
# VIEW SPECIFIC TIMELINES
# --------------------------

sample <-
  timeline |> 
  slice_sample(n = 1) |> 
  select(project_id) |> 
  pull() |>
  print()

# final timepoints
final_timeline(timeline, sample)
show_timeline(timeline, sample)

sample 

#
# checks
# ------------------------------
show_timeline(timeline, "XXXXXXX") # no decision


# --------------------------
# LLM
# --------------------------
llm <- read_parquet("data/analysis/test50_eis_llm.parquet")


llm |> glimpse()

#
# sample
# ------------------------------
sample <- 
  llm |> 
  select(project_id) |> 
  slice_sample(n =1) |> 
  pull() |> 
  print()

#
# read prompt
# ------------------------------
#llm |> 
#  filter(project_id %in% sample) |> 
#  select(llm_adj_prompt) |> 
#  pull() |> 
#  print()

#
# compare llm with bert 
# ------------------------------
llm |> 
  #filter(project_id %in% sample) |> 
  select(bert_initiation_date_final, bert_decision_date, llm_initiation_date, llm_decision_date) |> 
  #filter(!is.na(bert_initiation_date_final) & !is.na(bert_decision_date)) |> 
  drop_na() |> 
  print(n = 50)

#
# view llm reasoning
# ------------------------------
llm |> 
  filter(project_id %in% sample) |> 
  select(llm_initiation_reasoning) |> 
  pull() |> 
  print()

llm |> 
  filter(project_id %in% sample) |> 
  select(llm_decision_reasoning) |> 
  pull() |> 
  print()

#
# compare llm with bert 
# ------------------------------
show_timeline(timeline, sample) 



# --------------------------
# DATA ANALYSIS
# --------------------------
analysis <-
  llm |> 
  select(project_id, project_title, noi_publication_date, llm_initiation_date:llm_adj_error) |> 
  glimpse()


# bert timeline status
analysis |> 
  filter(!is.na(llm_initiation_date) & !is.na(llm_decision_date)) |> 
  #filter(is.na(llm_initiation_date)) |> 
  glimpse()

analysis |> 
  filter(!is.na(llm_initiation_date) & !is.na(llm_decision_date)) |> 
  glimpse()

# summary of timeline status
analysis |> 
  count(bert_timeline_status)


analysis |> 
  filter(bert_timeline_status == "missing_decision") |> 
  glimpse()

#
# view dates 
# ------------------------------
llm |> 
  filter(!is.na(llm_initiation_date) & !is.na(llm_decision_date)) |> 
  #select(contains("date")) |> 
  mutate(
    init_agree  = case_when(bert_initiation_date_final == llm_initiation_date ~ TRUE), 
    dec_agree  = case_when(bert_decision_date_final == llm_decision_date ~ TRUE) 
  ) |> 
  
  select(bert_initiation_date_final, llm_initiation_date, init_agree, bert_decision_date_final, llm_decision_date, dec_agree) |> 
  mutate(bert_duration = as_date(bert_decision_date_final)-as_date(bert_initiation_date_final)) |> 
  mutate(llm_duration = as_date(llm_decision_date)-as_date(llm_initiation_date)) |> 
  #print(n = 100)
  summarize(
    bert_duration = median(bert_duration, na.rm = TRUE), 
    llm_duration = median(llm_duration, na.rm = TRUE), 
            ) |> 
  print(n = 100)



# --------------------------
# documents
# --------------------------
docs <- read_parquet("data/analysis/documents_combined.parquet") |> 
  filter(dataset_source == "EIS") |> 
  glimpse()

docs |>
  filter(project_id == "1d8e80f7f4201d7cde3a6a71d30e3266") |> 
  glimpse()

projects <- read_parquet("data/analysis/projects_combined.parquet") |> 
  filter(dataset_source == "EIS") |> 
  glimpse()


projects |> 
  select(contains("date")) |> 
  glimpse()

#
# review individual projects
# ------------------------------

sample <-
  llm |> 
  slice_sample(n = 1) |> 
  select(project_id) |> 
  pull() |>
  print()

docs |> 
  #filter(project_id == "be968878a52ba88ffa3b34a9e8510b6b") |> 
  filter(project_id %in% sample) |> 
  select(file_name, document_type, document_type_clean, document_type_category, main_document) |> 
  print()

sample 

docs |> 
  select(document_type_category) |> 
  distinct()
  glimpse()


