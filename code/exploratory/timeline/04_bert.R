# --------------------------
# EXPLORATORY: BERT
# --------------------------
# Explore how well BERT-based timeline extraction worked for clean energy CE projects
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
  print()
}

timeline |> 
  distinct(type) |> 
  glimpse()

# --------------------------
# LOAD BERT TIMELINE DATA
# --------------------------
bert_timeline_path <- here("data", "analysis", "projects_timeline_bert.parquet")
#bert_timeline_path <- here("data", "analysis", "test50_bert_v9.parquet")
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

timeline |> 
  filter(type == "other") |>  
  filter(date < "2008") |> 
  select(project_id, type, date, source) |> 
  slice_sample(n = 20)
  #print(n = 20)

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
show_timeline(timeline, "1df6f8b5-7e16-2d38-01b6-a042628ea3c8") # Environmental Coordinator > review - why is decision 2022-09-13 still decision !
show_timeline(timeline, "5ec95c90-2b23-042d-4bab-8c8dad0151c6") # why is 2023-05-01 not the final decision? 
show_timeline(timeline, "58cab57e-ab90-d812-4735-526d78cf48b4") # all three three historical dates are still not being captured correctly, 2016-08-23 should be the initiation date
show_timeline(timeline, "8de424f4-3082-0131-f0d5-6a8c7cc07d2f") # 2012-07-01 is not being captured correctly as historical date; decision and initation should be on same day
show_timeline(timeline, "e74f6ef2-fb99-b7d0-1e67-7c76c1c269a5") # 2013-08-01 is not being captured as historical date, not being captured, 2020-04-27 should be initiation_final

# can be cleaned in post processing
show_timeline(timeline, "f2b5a957-b5aa-dadb-2cff-89b60dfa5e9b") # 2 decisions, make first a initiation?
show_timeline(timeline, "f2b5a957-b5aa-dadb-2cff-89b60dfa5e9b") # 2 decisions, make first a initiation?

# not sure what to do with these
show_timeline(timeline, "b523e342-39f2-fca4-0a2c-745c476dcf88") # not sure how to deal with  initiation 2021-12-31
show_timeline(timeline, "cec29e92-aa8d-42df-257a-62f290614404") # not capturing "DOE Initiator Signature" and "NEPA Compliance Officer" text correctly
show_timeline(timeline, "97e4029a-56b2-238a-1bd4-425e5cde9e91") # not sure what to do with first initiation 

show_timeline(timeline, "3e3bb9f5-f5ab-651d-b2d1-50ec99d99db0") # not sure what to do here
show_timeline(timeline, "78566cce-e233-ee90-6f60-0109a449e89b") # last review should review and is a duplicate date
show_timeline(timeline, "6149175c-8eb2-78bf-b995-1ab77be60997") # what other bucket can we categorize all these initiations?
show_timeline(timeline, "5c0911d5-65ab-b391-c956-a713cfa57da5") # how do we get the first decision to be an initiation based on date determined
show_timeline(timeline, "ca23261c-79f5-5207-fcff-a3158b9b2f9f") # seems to be two CE combined?
show_timeline(timeline, "e0f39636-313a-fc6c-43e3-6453566a1d1b") # last initiation needs to be a decision
show_timeline(timeline, "19938d50-8678-a9ce-0f7e-b164e85dac34") # not sure how to clean this one
show_timeline(timeline, "18354a3f-ea8d-982a-e050-0c882dcd3ce9") # not sure if review 2010-06-01 is a review or not

# good
show_timeline(timeline, "ea12d384-b7bf-83a0-4ada-d529d21f945b") # no initiation
show_timeline(timeline, "d01c96bc-9ec7-77c5-d72f-da3cb7bbb548") # initiation and decision cannot be on the same day
show_timeline(timeline, "a8f8ca21-28aa-e951-164f-736cac8136ef") # good complicated examples

# some manual checks 
show_timeline(timeline, "e74f6ef2-fb99-b7d0-1e67-7c76c1c269a5") # review 2020-04-27 > should become initiation even though correctly classified
show_timeline(timeline, "5ec95c90-2b23-042d-4bab-8c8dad0151c6") # review 2020-04-27 > should become initiation even though correctly classified


#timeline |> 
#  #filter(project_id == "3e3bb9f5-f5ab-651d-b2d1-50ec99d99db0") |> 
#  filter(project_id == "78566cce-e233-ee90-6f60-0109a449e89b") |> 
#  select(bert_decision_date, bert_earliest_review_date, bert_latest_review_date, bert_inferred_application_date) |> 
#  glimpse()
