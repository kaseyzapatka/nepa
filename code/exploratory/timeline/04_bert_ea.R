# --------------------------
# EXPLORATORY: BERT EA 
# --------------------------
# Explore how well BERT-based timeline extraction worked for clean energy EA projects
# Full run: data/analysis/projects_timeline_bert.parquet

# TAB
#5.00 - 4.91 | .09
#4.91 - 4.75 | .16
#4.75 - 4.63 | .12

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
#bert_timeline_path <- here("data", "analysis", "test50_ea.parquet")
bert_timeline_path <- here("data", "analysis", "test50_ea_llm.parquet")
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


# example 0b5d77cbc60c67c4be589f6d2f8d9f27 - decision should be 2012-10-01

#candidates <- read_csv("data/analysis/filtered_candidates_0b5d.csv")
candidates |> 
 select(type, date, source) |> 
  print(n = 100)
# 
# NOTES
# ------------------------------

# -  "expire" should be its own case
# drop dates + 5 years from earliest date (initiation)


# duplicate dates, preference decision, then initiation, then review, drop other
# initiation and decision can be on the same day - e.g., 5af96b24-dc7c-006e-15ac-1ffafdbcc93e
# if no initiation, first review date becomes initiation
# group by project, order, and overwrite to make sure there is 1 initiation and 1 decision, all others should be review or other. 
#
# checks
# ------------------------------
show_timeline(timeline, "2225ede090009df12ee87dcf48e16103") # no decision
show_timeline(timeline, "064ca592292b00766bfac7cddb952a1e") # seems too old
show_timeline(timeline, "0b5d77cbc60c67c4be589f6d2f8d9f27") # seem off


#
# review 
# ------------------------------

#
# can be cleaned in post processing
# ------------------------------

#
# long CE -- could have LLM read?
# ------------------------------

#
# good
# ------------------------------




# --------------------------
# DATA ANALYSIS
# --------------------------
analysis <-
  data |> 
  select(project_id, project_title, noi_publication_date, bert_decision_date:bert_error) |> 
  glimpse()

# bert error
analysis |> 
  #filter(bert_error == "no_dates_found_by_regex") |> 
  filter(bert_timeline_status == "no_dates") |> 
  glimpse() # 6

# bert timeline status
analysis |> 
  filter(is.na(bert_initiation_date_final)) |> 
  glimpse()

# summary of timeline status
analysis |> 
  count(bert_timeline_status)


analysis |> 
  filter(bert_timeline_status == "missing_decision") |> 
  glimpse()



# --------------------------
# COMPARING ROD VS SAMPLE
# --------------------------
# data
rod <- read_csv("data/analysis/test50_ea_rod_adjudication.csv")


rod_sample <-
  rod |> 
  slice_sample(n = 1) |> 
  select(project_id) |> 
  pull() |> 
  print()
  
test |> 
  filter(project_id %in% rod_sample) |> 
  #select(contains("initiation_date")) |> 
  select(sample_decision_date:notes) |> 
  glimpse()


data|> 
  summary(bert_n_dates_found)
  glimpse()



# --------------------------
# LLM
# --------------------------
llm <- read_parquet("data/analysis/test50_ea_llm.parquet")

llm |> 
  select(project_id) |> 
  slice_sample(n =1) |> 
  print()

llm |> 
  #filter(project_id == "0d717f5248bf4e6ae621cf064239d1b2") |> 
  #filter(project_id == "b2e51628d082d262c18bf76b8919812e") |> 
  filter(project_id == "e2b118a3c04f23e5447082f2373a91b7") |> 
  select(llm_adj_prompt) |> 
  pull() |> 
  print()


llm |> 
  filter(project_id == "b2e51628d082d262c18bf76b8919812e") |> 
  #select(bert_n_dates_found, bert_decision_date, llm_decision_date,  bert_decision_date_source, llm_decision_reasoning) |> 
  select(bert_n_dates_found, bert_initiation_date_final, bert_decision_date) |> 
  print()


llm |> 
  filter(project_id == "b2e51628d082d262c18bf76b8919812e") |> 
  glimpse()
  select(bert_initiation_date_final,) |> 
  print()


timeline |> 
  filter(project_id == "b2e51628d082d262c18bf76b8919812e") |> 
  #filter(type %in% c("initiation","decision") & final_flag == TRUE ) |>
  select("type","final_flag", "date","source") |> 
  arrange(date) |> 
  print(n = 50)


llm |> select(llm_adj_error) |> group_by(llm_adj_error) |> count()

llm |> 
  select(bert_n_dates_found, bert_initiation_date_final, llm_initiation_date, bert_decision_date, llm_decision_date) |> 
  print()


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
  print(n = 100)
  summarize(
    bert_duration = mean(bert_duration, na.rm = TRUE), 
    llm_duration = mean(llm_duration, na.rm = TRUE), 
            ) |> 
  glimpse()
  print(n = 100)

31/50



# --------------------------
# documents
# --------------------------
docs <- read_parquet("data/analysis/documents_combined.parquet") |> 
  filter(dataset_source == "EA") |> 
  glimpse()

projects <- read_parquet("data/analysis/projects_combined.parquet") |> 
  filter(dataset_source == "EA") |> 
  glimpse()

candidates <- read_parquet("data/analysis/regex_candidates_ea.parquet") |> 
  glimpse()

candidates |> 
  filter(project_id == "543b103fec369256675be35047a51d20") |> 
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
  #filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  filter(project_id %in% sample) |> 
  select(file_name, document_type, document_type_clean, document_type_category, main_document) |> 
  print()

sample 

# a94c8dd2bac25c3c52045291fc2b79f9 - FEA should be final

docs |> 
  select(document_type_category) |> 
  distinct()
  glimpse()


