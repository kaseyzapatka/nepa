# --------------------------
# EXPLORATORY: BERT EA 
# --------------------------
# Explore how well BERT-based timeline extraction worked for clean energy EA projects
# Full run: data/analysis/projects_timeline_bert.parquet

# TAB
#5.00 - 4.91 | .09
#4.91 - 4.75 | .16
#4.75 - 4.63 | .12
#4.63 - 4.48 | .15
#4.48 - 4.34 | .14
#4.34 - 4.21 | .13
#4.21 - 4.07 | .14
#4.07 - 3.93 #| .14 - 50 sample 
#3.93 - 2.28 #| 1.65 - 573 sample 

# --------------------------
# SETUP
# --------------------------
rm(list = ls()) 
source(here::here("phase1", "code", "deliverable03", "00_setup.R"))

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
bert_timeline_path <- here("phase1", "data", "analysis", "projects_timeline_bert_ea_llm.parquet")
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
show_timeline(timeline, "2225ede090009df12ee87dcf48e16103") # no decision
show_timeline(timeline, "064ca592292b00766bfac7cddb952a1e") # seems too old
show_timeline(timeline, "0b5d77cbc60c67c4be589f6d2f8d9f27") # seem off
show_timeline(timeline, "41cbef1f58b19d9f391d037a5d0cbfc6") # 2019-07-01 - final decision


# --------------------------
# LLM
# --------------------------
#llm <- read_parquet("data/analysis/test50_ea_llm.parquet")
#llm <- read_parquet("data/analysis/test50_ea_llm.parquet")
#llm <- read_parquet("data/analysis/test50_ea_llm_v3.parquet")
llm <- read_parquet("data/analysis/projects_timeline_bert_ea_llm.parquet")
llm |> glimpse()

llm_summary <- 
  llm |>  
  filter(!is.na(llm_initiation_date) & !is.na(llm_decision_date)) |> 
  select(project_id, llm_initiation_date, llm_decision_date, contains("reasoning")) |> 
  mutate(days = as_date(llm_decision_date) - as_date(llm_initiation_date)) |> 
  mutate(years = (as.numeric)(as_date(llm_decision_date) - as_date(llm_initiation_date))/365.25) |> 
  glimpse()


llm_summary |>
  summarize(
    mean_years = mean(
      as.numeric(as_date(llm_decision_date) - 
                 as_date(llm_initiation_date)),
      na.rm = TRUE
    ) / 365.25
  ) |>
  glimpse()

llm_summary |> 
  slice_sample(n = 10) |>
  select(llm_initiation_date,llm_decision_date, days, years ) |> 
  print()

llm_summary |> 
  slice_sample(n = 1) |>
  pull(llm_decision_reasoning) |> 
  print()

#355/573 # 61%
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
  #filter(project_id == "0b5d77cbc60c67c4be589f6d2f8d9f27") |> # example 0b5d77cbc60c67c4be589f6d2f8d9f27 - decision should be 2012-10-01
  filter(project_id %in% sample) |> 
  select(project_id, bert_initiation_date_final, bert_decision_date, llm_initiation_date, llm_decision_date) |> 
  print(n = 50)

show_timeline(timeline, "41cbef1f58b19d9f391d037a5d0cbfc6") # 2019-03-26 | 2019-07-01 - final decision
show_timeline(timeline, "5c29e4983e3c45262048a8b0c6cba9cf") # no clear decision
show_timeline(timeline, "a94c8dd2bac25c3c52045291fc2b79f9") # 2008-05-22 |       


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
# LLM Claude Validation
# --------------------------
llm |> 
  filter(!is.na(llm_initiation_date) & !is.na(llm_decision_date)) |> 
  select(project_id) |> 
  slice_sample(n = 10 ) |> 
  print()

# "716142ba81ce0beccb68f8cd5b0e930c" - Dates check out
# e5c590bfc90f810aae483d05bfd75c40
# 1c43ac38157653b1c7c339ec8598a85b
# 0ecdffd963b17a55fb2494b1cc5b7f7c
# 8821fb369b963ad337af0a6ac1db6ab4
# 1c2b8cc1484c40f793104e081f0321b2
# 0d717f5248bf4e6ae621cf064239d1b2
# 79cc77ed8727e7744c191ac8dc7372e8
# 6a2156fa14087bb58bd61ed1c11c717d
# 6faf66e9757e4d865ceef6462911a854
# 8b635116b3235c991c71d0d81be39584

llm |> 
  filter(project_id == "716142ba81ce0beccb68f8cd5b0e930c") |> 
  select(llm_initiation_date, llm_decision_date, contains("reasoning")) |> 
  print()

llm |> 
  filter(project_id == "0dadd4221696c70125443b189d73db55") |> 
  select(llm_initiation_date, llm_decision_date) |> 
  #select(llm_initiation_date, llm_decision_date, contains("reasoning")) |> 
  print()

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


