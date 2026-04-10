# --------------------------
# EXPLORATORY: PUBLIC COMMENTS
# --------------------------
# Exploraotry analysis of public comments


# --------------------------
# SETUP
# --------------------------

source(here::here("phase1", "code", "deliverable01", "00_setup.R"))


# --------------------------
# LOAD SPECIFIC DATA
# --------------------------

doc_path <- here("phase1", "data", "analysis", "documents_combined.parquet")
docs <- read_parquet(doc_path)

# --------------------------
# ANALYSIS
# --------------------------


docs |> 
  filter(dataset_source == "EA") |> 
  filter(str_detect(file_name, regex("comment", ignore_case = TRUE))) |> 
  glimpse()