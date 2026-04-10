# --------------------------
# EXPLORATORY: BERT
# --------------------------
# Explore how well BERT-based timeline extraction worked for clean energy CE projects
# Full run: data/analysis/projects_timeline_bert.parquet

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
  print()
}


# --------------------------
# LOAD BERT TIMELINE DATA
# --------------------------
bert_timeline_path <- here("phase1", "data", "analysis", "projects_timeline_bert.parquet")
#bert_timeline_path <- here("phase1", "data", "analysis", "test50_bert_v9.parquet")
timeline <- read_parquet(bert_timeline_path)
timeline |> glimpse()

bert_json_col = "bert_dates_json"
timeline <- extract_contexts(timeline, bert_json_col, "timeline")
timeline |> glimpse()


cat("Projects loaded:", nrow(timeline), "\n\n")

#timeline |> 
#  mutate(file_name_dates = map(project_file_name_dates, ~fromJSON(.x, flatten = TRUE))) |> 
#  filter(map_int(file_name_dates, length) > 0) %>%
#  slice(1) %>%
#  pull(file_name_dates) %>%
#  .[[1]]
#  glimpse()

# Derive year from decision date (or inferred application date as fallback)
timeline <- timeline %>%
  #select(bert_application_date,bert_inferred_application_date ) |> 
  mutate(
    bert_decision_date = as.Date(bert_decision_date),
    bert_application_date = as.Date(bert_application_date),
    bert_inferred_application_date = as.Date(bert_inferred_application_date),
    bert_earliest_review_date = as.Date(bert_earliest_review_date),
    # Year from decision date
    bert_year = as.integer(format(bert_decision_date, "%Y")),
    # Duration: decision minus best available start date
    bert_start_date = coalesce(bert_application_date, bert_inferred_application_date),
    bert_duration_days = as.numeric(bert_decision_date - bert_start_date)
  ) |> 
  select(project_id,contains("date")) |> 
  glimpse()



# --------------------------
# VIEW
# --------------------------

timeline |> 
  filter(!is.na(bert_error)) |> 
  select(project_id) |> 
  print()
  glimpse()

# no dates
timeline |> 
  filter(project_id == "a572180b-0637-5ddb-144c-16d2dab7cdd1") |> 
  glimpse()


timeline |> 
  filter(project_id == "a8f8ca21-28aa-e951-164f-736cac8136ef") |> 
  glimpse()


timeline |> 
  filter(project_id == "a8f8ca21-28aa-e951-164f-736cac8136ef") |> 
  select(type, date, source) |> 
  print()


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


# -  "expire" should be its own case
# drop dates + 5 years from earliest date (initiation)


# duplicate dates, preference decision, then initiation, then review, drop other
# initiation and decision can be on the same day - e.g., 5af96b24-dc7c-006e-15ac-1ffafdbcc93e
# if no initiation, first review date becomes initiation
# group by project, order, and overwrite to make sure there is 1 initiation and 1 decision, all others should be review or other. 


# checks !
show_timeline(timeline, "1df6f8b5-7e16-2d38-01b6-a042628ea3c8") # Environmental Coordinator > review - why is 2022-09-13 still decision? 
show_timeline(timeline, "b523e342-39f2-fca4-0a2c-745c476dcf88") # 2021-12-31 not expiration?  
show_timeline(timeline, "cec29e92-aa8d-42df-257a-62f290614404") # not capturing "DOE Initiator Signature" and "NEPA Compliance Officer" text correctly in document | initiation is after decision?
show_timeline(timeline, "3e3bb9f5-f5ab-651d-b2d1-50ec99d99db0") # why is 2021-04-30 not expiration? | why is 2021-01-11 final initiation? -- could be right?
show_timeline(timeline, "6149175c-8eb2-78bf-b995-1ab77be60997") # prob fine but there are lots of initiations here | 2014-10-01 should prob be a review
show_timeline(timeline, "5c0911d5-65ab-b391-c956-a713cfa57da5") # why is 2021-07-01 not the final decision here?
show_timeline(timeline, "e74f6ef2-fb99-b7d0-1e67-7c76c1c269a5") # why is 2020-04-30 initiation instead of 2020-04-27?
show_timeline(timeline, "d01c96bc-9ec7-77c5-d72f-da3cb7bbb548") # why does second 2012-08-22 not trigger decision?
show_timeline(timeline, "ca23261c-79f5-5207-fcff-a3158b9b2f9f") # seems to be two CE combined? - last review should be made decision
show_timeline(timeline, "10a5fe92-958e-9004-a46c-3f3ba4c031cc") # 2023-07-10 should be final decision | 2023-07-07 should be review
show_timeline(timeline, "779788a2-7efe-c1b9-9c24-3b98ae5f8499") # should 2018-05-02 be the initiation?
show_timeline(timeline, "b1c99c0e-775d-d29b-e696-89d6cf2213ef") # 2011-09-01 and 2013-01-01 should be expiration dates
show_timeline(timeline, "82fcccb4-f539-604e-5be5-7f9fb867054a") # 2021-08-05 should be decision

# review 
show_timeline(timeline, "58cab57e-ab90-d812-4735-526d78cf48b4") # 2016-08-23 not initiation but other? NO DATES
show_timeline(timeline, "a8f8ca21-28aa-e951-164f-736cac8136ef") # correct - no dates anymore


# can be cleaned in post processing
show_timeline(timeline, "f2b5a957-b5aa-dadb-2cff-89b60dfa5e9b") # correct
show_timeline(timeline, "f2b5a957-b5aa-dadb-2cff-89b60dfa5e9b") # correct
show_timeline(timeline, "97e4029a-56b2-238a-1bd4-425e5cde9e91") # correct
show_timeline(timeline, "78566cce-e233-ee90-6f60-0109a449e89b") # correct
show_timeline(timeline, "e0f39636-313a-fc6c-43e3-6453566a1d1b") # correct
show_timeline(timeline, "5ec95c90-2b23-042d-4bab-8c8dad0151c6") # correct

show_timeline(timeline, "5ec95c90-2b23-042d-4bab-8c8dad0151c6") # why is 2023-05-01 not the final decision? CORRECT
show_timeline(timeline, "8de424f4-3082-0131-f0d5-6a8c7cc07d2f") # 2012-07-01 is not being captured correctly as historical date; decision and initiation should be on same day CORRECT
show_timeline(timeline, "e74f6ef2-fb99-b7d0-1e67-7c76c1c269a5") # 2020-04-30 initiation and not 2020-04-27? CORRECT

# long CE -- could have LLM read?
show_timeline(timeline, "18354a3f-ea8d-982a-e050-0c882dcd3ce9") # correct, but long?
show_timeline(timeline, "19938d50-8678-a9ce-0f7e-b164e85dac34") # correct, but long? | 2022-02-25 likely initiation but means CE took 2years | 2024-02-09 should be initiation?
show_timeline(timeline, "badd0cda-75a3-e09f-138c-2ecbfa39631f") # long CE? classifications are correct here but seems long, other explanation?
show_timeline(timeline, "ed362c4b-cec9-fb82-6292-8cc64fedf5ae") # long CE? classifications are correct here but seems long, other explanation?
show_timeline(timeline, "b3e12657-6fce-6aaf-767e-80e3db00201a") # could 2017-08-14 be the correct initiation date instead? decision date is correct
show_timeline(timeline, "0ba26e6f-880f-fc26-35b1-50a34622eb7c") # could be correct but also initiation date seems far 

# good
show_timeline(timeline, "ea12d384-b7bf-83a0-4ada-d529d21f945b") # no initiation
