# --------------------------
# PROJECT REVIEW: MILITARY NUCLEAR
# --------------------------

source(here::here("code", "deliverable01", "00_setup.R"))

# Frank wanted to know the agency mix for defense related nuclear projects
# nearly all of 481 were DOE
# THESE WERE ALL REMOVED FROM CLEAN ENERGY
military_projects <-
  projects |>
  mutate(department = project_department) |>
  filter(str_detect(project_type, "Military and Defense") & str_detect(project_type, "Nuclear")) |>
  select(project_id, project_title, department, project_type) |>
  arrange(department) |>
  glimpse()

# save
sheet_write(
  data = military_projects,
  ss = "https://docs.google.com/spreadsheets/d/11J6hU15ngCQP-Quk8h2eSkwct7cmq8Zigl_XsDbpsi0/edit?usp=sharing",
  sheet = "military_projects_to_exclude")

military_project_ids_to_filter <-
  military_projects |>
  select(project_id) |>
  glimpse()

write_csv(military_project_ids_to_filter, here("notes", "military_project_ids_to_filter.csv"))
