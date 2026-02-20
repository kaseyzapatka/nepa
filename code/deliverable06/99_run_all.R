# --------------------------
# DELIVERABLE 6: RUN ALL
# --------------------------

library(here)

scripts <- c(
  "01_transmission.R",
  "02_geothermal.R",
  "03_pipelines.R",
  "04_identification_qc.R",
  "05_length_validation.R"
)

for (script_name in scripts) {
  script_path <- here("code", "deliverable06", script_name)
  cat("Running", script_path, "...\n")
  source(script_path, local = new.env(parent = globalenv()))
}

cat("Deliverable 6 complete. Outputs in output/deliverable6/{tables,figures}.\n")
