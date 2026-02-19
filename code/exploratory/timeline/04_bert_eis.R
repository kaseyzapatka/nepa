# --------------------------
# EXPLORATORY: BERT EIS 
# --------------------------
# Explore how well BERT-based timeline extraction worked for clean energy EA projects
# Full run: data/analysis/projects_timeline_bert.parquet

# TAB
#2.28 - 2.17 #| .11
#2.17 - 2.04 #| .13
#2.04 #| .13

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
#llm <- read_parquet("data/analysis/test50_eis_llm.parquet")
llm <- read_parquet("data/analysis/projects_timeline_bert_eis_llm.parquet")


llm |> glimpse()

#
# sample
# ------------------------------
sample <- 
  llm |> 
  filter(!is.na(llm_initiation_date) & !is.na(llm_decision_date)) |> 
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
  #select(project_id, bert_initiation_date_final, bert_decision_date, llm_initiation_date, llm_decision_date) |> 
  #filter(!is.na(bert_initiation_date_final) & !is.na(bert_decision_date)) |> 
  #filter(!is.na(llm_initiation_date) & !is.na(llm_decision_date)) |> 
  filter(is.na(llm_initiation_date) & is.na(llm_decision_date)) |> 
  select(project_id) |> 
  #drop_na() |> 
  print(n = 50)


llm |> 
  filter(!is.na(llm_initiation_date) & !is.na(llm_decision_date)) |> 
  dim()

362/753 # 48% 

#
# view llm reasoning
# ------------------------------
llm |> 
  #filter(project_id == "9e2d0d5d3ae33f94a782b79bae9db894") |> 
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

# llm timeline status
analysis |> 
  filter(!is.na(llm_initiation_date) & !is.na(llm_decision_date)) |> 
  select(llm_initiation_date, llm_decision_date) |> 
  mutate(duration = as_date(llm_decision_date) - as_date(llm_initiation_date)) |> 
  print(n = 100)


analysis |> 
  filter(!is.na(llm_initiation_date) & !is.na(llm_decision_date)) |> 
  select(llm_initiation_date, llm_decision_date, contains("reasoning")) |> 
  print(n = 100)


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

llm |> 
  left_join(projects) |> 
  select(noi_publication_date, llm_initiation_date) |> 
  drop_na() |> 
  print()
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

llm |> 
  filter(project_id == "9e2d0d5d3ae33f94a782b79bae9db894") |> 
  pull(llm_decision_reasoning)

# 9e2d0d5d3ae33f94a782b79bae9db894
# =================================
#2008
#
#9/12/2008 — NRC and USACE sign a Memorandum of Understanding for cooperative review of nuclear power plant license applications
#9/22/2008 — Westinghouse submits Revision 17 to the AP1000 Design Certification Amendment
#10/14/2008 — NRC issues "notice of acceptance" for docketing the COL application (published in Federal Register)
#12/19/2008 — EPA provides EIS scoping comments to NRC
#
#2009
#
#3/16/2009 — USACE releases public notice soliciting comments on preconstruction activities
#5/1/2009 — PEF withdraws its Limited Work Authorization (LWA) request
#
#2010
#
#3/3/2010 — DOE submits motion to withdraw its Yucca Mountain permanent repository application
#8/6/2010 — DEIS filed
#8/13/2010 — CEQ Federal Register notice published
#9/15/2010 — NRC issues updated "Waste Confidence" regulation (extending safe storage to 60 years beyond reactor licensed life)
#9/23/2010 — Joint 404 permit public hearing held in Crystal River, FL
#~December 2010 — AP1000 certification review expected to be completed
#
#2011
#
#NRC Safety Evaluation Report anticipated for publication