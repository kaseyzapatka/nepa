# --------------------------
# DEFINING CLEAN ENERGY
# --------------------------
# This file is an exploratory analyses to help us define clean energy for the 
# the project.

# --------------------------
# SETUP
# --------------------------

# clear environment
rm(list=ls()) 

# source setup files
source(here::here("code", "00_setup.R"))

# load libraries
library(jsonlite)

#
# functions 
# ----------------------------------------
# Safe JSON parser for LLM output
safe_fromJSON <- function(x) {
  tryCatch(
    fromJSON(x, flatten = TRUE),
    error = function(e) NULL
  )
}


# --------------------------
# LOAD DATA
# --------------------------
documents <- read_parquet(here("data", "analysis", "documents_combined.parquet")) |> glimpse()
ea_pages <- read_parquet(here("data", "processed", "ea", "pages.parquet")) |> glimpse()
projects_timeline <- read_parquet(here("data", "analysis", "projects_timeline.parquet")) |> glimpse()

# llm results 
my_results <- read_parquet(here("data", "analysis", "my_results.parquet")) |> glimpse()
my_results2 <- read_parquet(here("data", "analysis", "my_results2.parquet")) |> glimpse()
my_results3 <- read_parquet(here("data", "analysis", "my_results3.parquet")) |> glimpse()
my_results4 <- read_parquet(here("data", "analysis", "my_results4.parquet")) |> glimpse()
my_results70b <- read_parquet(here("data", "analysis", "my_results_70b.parquet")) |> glimpse()
my_resultsqwen <- read_parquet(here("data", "analysis", "my_results_qwen.parquet")) |> glimpse()
my_results100 <- read_parquet(here("data", "analysis", "results_100_qwen14b.parquet")) |> glimpse()

# test runs by varying model size
test10_qwen2_14b <- read_parquet(here("data", "analysis", "test10_qwen14b.parquet")) |> glimpse()
test10_qwen2_7b <- read_parquet(here("data", "analysis", "test10_qwen2:7b.parquet")) |> glimpse()
test10_llama3 <- read_parquet(here("data", "analysis", "test10_llama3.2.parquet")) |> glimpse()
test10_qwen7 <- read_parquet(here("data", "analysis", "test10_hybrid_qwen7.parquet")) |> glimpse()





# --------------------------
# VIEW RESULTS OF LLM RUN
# --------------------------

#
# processing
# ----------------------------------------

#  successes
successes_parsed100 <- 
  my_results100 %>%
  # process
  mutate(
    parsed = map(llm_raw_response, safe_fromJSON),
    parse_success = !map_lgl(parsed, is.null)
  ) %>%
  #filter(parse_success) %>%            # drop malformed JSON rows
  unnest_wider(parsed) %>%
  glimpse()

#  successes
successes_parsed_llama <- 
  test10_llama3 %>%
  # process
  mutate(
    parsed = map(llm_raw_response, safe_fromJSON),
    parse_success = !map_lgl(parsed, is.null)
  ) %>%
  #filter(parse_success) %>%            # drop malformed JSON rows
  unnest_wider(parsed) %>%
  glimpse()

successes_parsed_qwen <- 
  test10_qwen2_7b %>%
  # process
  mutate(
    parsed = map(llm_raw_response, safe_fromJSON),
    parse_success = !map_lgl(parsed, is.null)
  ) %>%
  #filter(parse_success) %>%            # drop malformed JSON rows
  unnest_wider(parsed) %>%
  glimpse()

successes_parsed_qwen14 <- 
  test10_qwen2_14b %>%
  # process
  mutate(
    parsed = map(llm_raw_response, safe_fromJSON),
    parse_success = !map_lgl(parsed, is.null)
  ) %>%
  #filter(parse_success) %>%            # drop malformed JSON rows
  unnest_wider(parsed) %>%
  glimpse()


#  successes
successes_parsed_qwen7 <- 
  test10_qwen7  %>%
  # process
  mutate(
    parsed = map(llm_raw_response, safe_fromJSON),
    parse_success = !map_lgl(parsed, is.null)
  ) %>%
  #filter(parse_success) %>%            # drop malformed JSON rows
  unnest_wider(parsed) %>%
  glimpse()


#
# failures
# ----------------------------------------
test10_qwen2_7b |> 
  filter(!is.na(llm_error)) |> 
  select(total_chars) |> 
  summary()


# failure - characters
my_results100 |> 
  filter(is.na(llm_error)) |> 
  select(total_chars) |> 
  summary()

# success - characters
my_results100 |> 
  filter(!is.na(llm_error)) |> 
  ggplot() +
  geom_histogram(aes(total_chars))

my_results100 |> 
  filter(is.na(llm_error)) |> 
  ggplot() +
  geom_histogram(aes(total_chars))

my_results100 |> 
  filter(!is.na(llm_error)) |> 
  dim() # 26

my_results100 |> 
  filter(!is.na(llm_error)) |> 
  filter(total_chars > 20000) |> 
  dim() # 26


# number of characters tokens in errors
my_results100 |> 
  filter(!is.na(llm_error)) |> 
  select(total_chars) |> 
  summary()

my_results100 |> 
  filter(!is.na(llm_error)) |> 
  dim() # 26

my_results100 |> 
  filter(!is.na(llm_error)) |> 
  filter(total_chars > 20000) |> 
  dim() # 26


#
# successes
# ----------------------------------------
successes_parsed |> 
  unnest(dates) |> 
  group_by(project_id) |> 
  count(type) |> 
  ungroup() |> 
  filter(type == "decision") |> 
  arrange(desc(n)) |> 
  count(n) |> 
  arrange(desc(n)) |> 
  print() # 15 / 50 1 decision

# view where there are double decisions
double_decision <-
  successes_parsed |> 
  unnest(dates) |> 
  group_by(project_id) |> 
  count(type) |> 
  ungroup() |> 
  left_join(test10_qwen2_7b) |> 
  filter(n >= 2) |> 
  slice_sample(n=1) |> 
  glimpse()

successes_parsed |> 
  filter(project_id %in% double_decision) |> 
  unnest(dates) |> 
  select(type, date, source) |> 
  print()
 
#
# view quality of tags by specific project
# ----------------------------------------
target_id <-
  successes_parsed_qwen7 |> 
    filter(parse_success == FALSE) |> 
    select(project_id) |> 
    #slice_sample(n=1) |> 
    glimpse()

#successes_parsed_qwen14 |> 
#  filter(project_id %in% target_id) |> 
#  unnest(dates) |> 
#  select(type, date, source) |> 
#  print()

successes_parsed_qwen |> 
  filter(project_id %in% target_id) |> 
  unnest(dates) |> 
  select(type, date, source) |> 
  print()

successes_parsed_qwen7 |>
  filter(project_id %in% target_id) |> 
  unnest(classifications) |> 
  select(type, date, reason) |> 
  print()



#successes_parsed_llama |> 
#  filter(project_id %in% target_id) |> 
#  unnest(dates) |> 
#  select(type, date, source) |> 
#  print()



successes_parsed100 |> 
  slice_sample(n = 1)
  filter(project_id == "51baada6-e292-4362-7094-625197da2c2") |> 
  unnest(dates) |> 
  select(type, date, source) |> 
  print()


# --------------------------
# DOCUMENT DISTRIBUTION BY REVIEW PROCESS
# --------------------------
# Goal: Determine which review processes have which documents
# to identify the best documents for timeline extraction

#
# Count unique file names by review process (EA, CE, EIS)
# ------------------------------
documents |>
  group_by(dataset_source) |>
  summarise(
    n_documents = n(),
    n_unique_filenames = n_distinct(file_name)
  ) |>
  print()

#
# Distribution of document_type by review process
# ------------------------------
documents |>
  group_by(dataset_source, document_type) |>
  summarise(n = n(), .groups = "drop") |>
  arrange(dataset_source, desc(n)) |>
  print(n = 50)

#
# Distribution of document_type_category by review process
# ------------------------------
documents |>
  group_by(dataset_source, document_type_category) |>
  summarise(n = n(), .groups = "drop") |>
  arrange(dataset_source, desc(n)) |>
  print()

documents |> 
  glimpse()



# --------------------------
# CATEGORICAL EXCLUSION (CE)
# --------------------------
ce <-
  documents |>
  filter(dataset_source == "CE") |>
  glimpse()



  ce |>
  group_by(main_document, document_type_category) |> 
  count() |> 
  print()
  glimpse()


#
# How many documents per project?
# ------------------------------
ce_project_doc_counts <-
  ce |>
  group_by(project_id) |>
  summarise(
    n_docs = n(),
    n_main_docs = sum(main_document == "YES", na.rm = TRUE),
    n_decision_docs = sum(document_type_category == "decision", na.rm = TRUE),
    doc_types = paste(unique(document_type), collapse = ", "),
    has_ce_type = any(document_type == "CE", na.rm = TRUE)
  ) |>
  ungroup() |> 
  print()

# Distribution of documents per project
ce_project_doc_counts |>
  count(n_docs, name = "n_projects") |>
  mutate(pct = n_projects / sum(n_projects) * 100) |>
  print(n = 20)

# How many projects have exactly 1 document vs multiple?
ce_project_doc_counts |>
  mutate(doc_group = case_when(
    n_docs == 1 ~ "1 doc",
    n_docs == 2 ~ "2 docs",
    n_docs <= 5 ~ "3-5 docs",
    TRUE ~ "6+ docs"
  )) |>
  count(doc_group) |>
  mutate(pct = n / sum(n) * 100) |>
  print()

#
# Is main_document reliable for identifying the key document?
# ------------------------------
# How many projects have exactly 1 main document?
ce_project_doc_counts |>
  count(n_main_docs, name = "n_projects") |>
  mutate(pct = n_projects / sum(n_projects) * 100) |>
  print()

# For projects with multiple docs, how many main docs do they have?
ce_project_doc_counts |>
  filter(n_docs > 1) |>
  count(n_main_docs, name = "n_projects") |>
  mutate(pct = n_projects / sum(n_projects) * 100) |>
  print()

#
# What document_type do multi-doc projects have?
# ------------------------------
ce_project_doc_counts |>
  filter(n_docs > 1) |>
  count(doc_types, sort = TRUE) |>
  slice_head(n = 20) |>
  print(n = 20)

#
# Sample: look at a few projects with multiple documents
# ------------------------------
multi_doc_projects <-
  ce_project_doc_counts |>
  filter(n_docs > 1) |>
  slice_sample(n = 5) |>
  pull(project_id)

ce |>
  filter(project_id %in% multi_doc_projects) |>
  select(project_id, document_type, document_type_category, main_document, file_name) |>
  arrange(project_id, desc(main_document)) |>
  print(n = 50)

#
# Load CE pages to examine actual text content
# ------------------------------
ce_pages <- read_parquet(here("data", "processed", "ce", "pages.parquet"))

# Get document_ids for our sample projects
sample_doc_ids <-
  ce |>
  filter(project_id %in% multi_doc_projects) |>
  pull(document_id)

# Join pages with document metadata for sample
sample_pages <-
  ce_pages |>
  filter(document_id %in% sample_doc_ids) |>
  left_join(
    ce |> select(document_id, project_id, document_type, main_document, file_name),
    by = "document_id"
  )

# Summary: how many pages per document in sample?
sample_pages |>
  group_by(project_id, document_id, document_type, main_document, file_name) |>
  summarise(n_pages = n(), .groups = "drop") |>
  arrange(project_id, desc(main_document)) |>
  print(n = 50)

# Look at first page of each document to compare content
sample_pages |>
  filter(page_number == 1) |>
  select(project_id, main_document, file_name, page_text) |>
  mutate(
    # Truncate text for display
    page_text_preview = str_sub(page_text, 1, 500)
  ) |>
  arrange(project_id, desc(main_document)) |>
  select(-page_text) |>
  print(n = 20)

# For one specific project, compare all first pages side by side
one_project <- multi_doc_projects[1]

sample_pages |>
  filter(project_id == one_project) |>
  filter(page_number <= 3) |>  # First 3 pages
  arrange(desc(main_document), page_number) |>
  select(main_document, file_name, page_number, page_text) |>
  mutate(page_text_preview = str_sub(page_text, 1, 1000)) |>
  select(-page_text) |>
  print(n = 20)

#
# Key question: For projects with 1 main doc, is it always document_type == "CE"?
# ------------------------------
ce |>
  filter(main_document == "YES") |>
  count(document_type, sort = TRUE) |>
  mutate(pct = n / sum(n) * 100) |>
  print()

#
# Potential identifier: total_pages - does the main/longest doc have the dates?
# ------------------------------
ce |>
  mutate(total_pages = as.numeric(total_pages)) |>
  group_by(main_document) |>
  summarise(
    n = n(),
    mean_pages = mean(total_pages, na.rm = TRUE),
    median_pages = median(total_pages, na.rm = TRUE),
    max_pages = max(total_pages, na.rm = TRUE)
  ) |>
  print()

#
# documents by main-document?
# ------------------------------
ce |>
  group_by(main_document) |>
  count(document_type, sort = TRUE) |>
  print()


#
# documents by main-document?
# ------------------------------
ce |> 
  filter(main_document == "NO") |> 
  count(file_name, sort = TRUE) |>
  slice_head(n = 30) |>
  print(n = 30)


ce |> 
  filter(document_type == "OTHER") |> 
  filter(document_count > 1) |> 
  count(file_name, sort = TRUE) |>
  slice_head(n = 30) |>
  print(n = 30)


#
# Unique file name patterns by review process (sample of most common)
# ------------------------------

# CE file names
ce |> 
  count(file_name, sort = TRUE) |>
  slice_head(n = 100) |>
  print(n = 100)



# CE file names
documents |> 

  filter(dataset_source == "CE") |>
  filter(project_id == "0002b3e8-1a99-b345-658e-ad94e8605295") |>
  select(file_name, document_type, main_document, document_title) |> 
  print()
  count(file_name, sort = TRUE) |>
  slice_head(n = 10) |>
  print(n = 10)








# --------------------------
# ENVIRONMENTAL ASSESSMENT (EA)
# --------------------------


# EA file names
documents |>
  filter(dataset_source == "EA") |>
  count(file_name, sort = TRUE) |>
  slice_head(n = 30) |>
  print(n = 30)

# EIS file names
documents |>
  filter(dataset_source == "EIS") |>
  count(file_name, sort = TRUE) |>
  slice_head(n = 30) |>
  print(n = 30)

# --------------------------
# TIMELINE EXTRACTION EVALUATION
# --------------------------
# Quick evaluation of CE + Clean energy timeline extraction results

# Load latest timeline data
timeline <- read_parquet(here("data", "analysis", "projects_timeline.parquet"))

# Filter to CE + Clean energy
ce_timeline <-
  timeline |>
  filter(dataset_source == "CE", project_energy_type == "Clean")

print(paste("CE + Clean energy projects:", nrow(ce_timeline)))

#
# 1. Missing data summary
# ------------------------------
ce_timeline |>
  summarise(
    n_projects = n(),
    has_earliest = sum(!is.na(project_date_earliest)),
    has_latest = sum(!is.na(project_date_latest)),
    has_decision = sum(!is.na(project_date_decision)),
    has_year = sum(!is.na(project_year)),
    pct_earliest = round(mean(!is.na(project_date_earliest)) * 100, 1),
    pct_latest = round(mean(!is.na(project_date_latest)) * 100, 1),
    pct_decision = round(mean(!is.na(project_date_decision)) * 100, 1)
  ) |>
  print()

#
# 2. Projects flagged for review
# ------------------------------
ce_timeline |>
  count(project_timeline_needs_review, name = "n") |>
  mutate(pct = round(n / sum(n) * 100, 1)) |>
  print()

#
# 3. Review reasons breakdown
# ------------------------------
ce_timeline |>
  filter(project_timeline_needs_review == TRUE) |>
  mutate(reasons = project_timeline_review_reasons) |>
  count(reasons, sort = TRUE) |>
  slice_head(n = 10) |>
  print()

#
# 4. Duration distribution (for projects with dates)
# ------------------------------
ce_timeline |>
  filter(!is.na(project_duration_days)) |>
  summarise(
    n = n(),
    min_days = min(project_duration_days),
    median_days = median(project_duration_days),
    mean_days = round(mean(project_duration_days)),
    max_days = max(project_duration_days),
    over_1yr = sum(project_duration_days > 365),
    over_5yr = sum(project_duration_days > 1825)
  ) |>
  print()

#
# 5. Document count vs date extraction success
# ------------------------------
ce_timeline |>
  mutate(has_dates = !is.na(project_date_earliest)) |>
  group_by(project_document_count) |>
  summarise(
    n = n(),
    pct_has_dates = round(mean(has_dates) * 100, 1),
    .groups = "drop"
  ) |>
  slice_head(n = 10) |>
  print()

#
# 6. Sample projects needing review (for manual check)
# ------------------------------
ce_timeline |>
  filter(project_timeline_needs_review == TRUE) |>
  select(project_id, project_title, project_date_earliest, project_date_latest,
         project_duration_days, project_timeline_review_reasons) |>
  slice_sample(n = 10) |>
  print()


#
# 7. Sample projects that DON'T NEED  review 
# ------------------------------
ce_timeline |>
  filter(project_timeline_needs_review == FALSE) |>
  select(project_id, project_title, project_date_earliest, project_date_latest,
         project_duration_days, project_timeline_review_reasons) |>
  slice_sample(n = 10) |>
  print()

# Enoch Well House Power Line - 
ce_timeline |> 
  filter(project_id == "abb436a4-3bb4-8f91-0013-d5a5158dbf32") |>
  glimpse()
  select(project_id, project_title, project_dates:project_year) |> 
  glimpse()

# Correct Water Intrusion Issues around 105-L - DATE DETERMINED
ce_timeline |> 
  filter(project_id == "d97d7ac4-8f23-e9cc-e7d3-85b2e75d8484") |>
  select(project_id, project_title, project_dates:project_year) |> 
  glimpse() 

# Create Spool Pieces for 218-1H Trane Chiller Evaporator Piping - DATE DETERMINED
ce_timeline |> 
  filter(project_id == "85c1b6d4-85a5-1597-00c0-5d4e55050a21") |>
  select(project_id, project_title, project_dates:project_year) |> 
  glimpse()



# --------------------------
# DOCUMENTS
# --------------------------


document_sample <- 
  documents |> 
  left_join(projects) |> 
  filter(main_document == "YES") |> 
  slice_sample( n = 100) |> 
  select(project_title,project_id, document_id, document_type, file_name, main_document) |> 
  glimpse()


# save to google sheets so team can view
sheet_write(
  data = document_sample,
  ss = "https://docs.google.com/spreadsheets/d/1Khd_aLhw7ArxO9zMl1C6R7HlFPRr8Nl_S7pIxPvM1tc/edit?usp=sharing",
  sheet = "document_sample"
)


  documents |> 
  left_join(projects) |> 
  filter(project_id == "890d5651-4f06-9c9b-3567-7e90e7adf7c9") |> 
  glimpse()

documents |> glimpse()

#
# check to see if prepared_by column in documents provides any new information -- not really 
# ------------------------------

documents |> 
  filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  glimpse()


doc_title_sample <- 
  documents |> 
  slice_head(n = 100) |> 
  select(project_id, document_title, file_name, main_document, document_type_category) |> 
  glimpse()

# save to google sheets so team can view
sheet_write(
  data = doc_title_sample,
  ss = "https://docs.google.com/spreadsheets/d/1Khd_aLhw7ArxO9zMl1C6R7HlFPRr8Nl_S7pIxPvM1tc/edit?usp=sharing",
  sheet = "doc_title_sample"
)

# sample generatro
documents |> 
  filter(dataset_source == "EIS") |> 
  filter(!is.na(prepared_by)) |> 
  slice_sample(n = 1) |> 
  select(project_id) |> 
  print()

documents |> 
  filter(project_id == "832c45f82b287d606cb23b5e898342fe") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()

documents |> 
  filter(project_id == "4f737e61f906e379aaf5b0670a50ae12") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()

documents |> 
  filter(project_id == "a7c79217cc119261606c44a0b85e7042") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()

documents |> 
  filter(project_id == "c8ef7f23d1dde9411d38fa045d364197") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()

documents |> 
  filter(project_id == "30ceb76704ebe7a9ea177c3ef2b2e84e") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()


documents |> 
  filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()


documents |> 
  filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  left_join(projects) |> 
  select(prepared_by, project_department, lead_agency) |> 
  distinct() |>
  print()



# --------------------------
# TEST CASES
# --------------------------

#
# test 1: 39d059404366825eda2052316c81c34a: Chevron USA Inc. Three Lost Hills Well Abandonments
# ------------------------------
test <- 
  documents |> 
  filter(project_id == "39d059404366825eda2052316c81c34a") |> 
  glimpse()

# save to google sheets so team can view
sheet_write(
  data = test,
  ss = "https://docs.google.com/spreadsheets/d/1Khd_aLhw7ArxO9zMl1C6R7HlFPRr8Nl_S7pIxPvM1tc/edit?usp=sharing",
  sheet = "test"
)

projects_timeline |> 
  glimpse()
   filter(project_id == "39d059404366825eda2052316c81c34a") |> 
  glimpse()

#
# test 2: 6e9fc6608e30977c74305d2a98628a13: Coldfoot Cell tower
# ------------------------------

test2 <- 
  documents |> 
  filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  glimpse()

# save to google sheets so team can view
sheet_write(
  data = test2,
  ss = "https://docs.google.com/spreadsheets/d/1Khd_aLhw7ArxO9zMl1C6R7HlFPRr8Nl_S7pIxPvM1tc/edit?usp=sharing",
  sheet = "test2"
)

projects_timeline |> 
  filter(project_id == "6e9fc6608e30977c74305d2a98628a13") |> 
  pull(project_dates) |> 
  #select(project_dates, project_date_contexts, project_year) |> 
  print()
