# --------------------------
# EXPLORATORY: BERT EA 
# --------------------------
# Explore how well BERT-based timeline extraction worked for clean energy EA projects
# Full run: data/analysis/projects_timeline_bert.parquet

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
  print(n = 50)
}

final_timeline <- function(df, pid) {
  df |> 
  filter(type == c("initiation","decision") & final_flag == TRUE ) |>
  filter(project_id == pid) |>
  select("type","final_flag", "date","source") |> 
  print(n = 50)
}

timeline |> glimpse() 
  distinct(type)

# --------------------------
# LOAD BERT TIMELINE DATA
# --------------------------
# settings
bert_timeline_path <- here("data", "analysis", "test50_ea.parquet")
bert_json_col = "bert_dates_json"

# extract
data <- read_parquet(bert_timeline_path)
timeline <- extract_contexts(data, bert_json_col, "timeline")

# glimpse
timeline |> glimpse()

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
# VIEW SPECIFIC TIMELINES
# --------------------------

sample <-
  timeline |> 
  slice_sample(n = 1) |> 
  select(project_id) |> 
  pull() |>
  print()

show_timeline(timeline, sample)
final_timeline(timeline, sample)

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