# --------------------------
# PROJECT REVIEW: UTILITIES ONLY
# --------------------------

source(here::here("code", "deliverable01", "00_setup.R"))

# There were about 1,623 projects that had some combination of Utilities we didn't want
# to count as clean energy
utilties_only_projects <-
  projects |>
  # identify Utilities + Broadband, Waste Management, or Land Development tags
  select(project_title,project_type, contains("utilities")) |>
  filter(project_utilities_to_exclude) |>
  select(project_title, project_type) |>
  glimpse()

# save
sheet_write(
  data = utilties_only_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "utilties_only_projects"
)
