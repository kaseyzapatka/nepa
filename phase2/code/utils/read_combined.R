rm(list = ls())

library(arrow)
library(here)

# Read the combined projects parquet
df <- read_parquet(here("phase2/data/analysis/projects_combined.parquet"))

# Quick overview
cat("Dimensions:", nrow(df), "rows x", ncol(df), "cols\n\n")
cat("Column names:\n")
print(names(df))

cat("\n\nFirst few rows:\n")
print(head(df))

cat("\n\nData types:\n")
print(str(df))

# check project description
df |> select(project_description) |> filter(is.na(project_description))|> glimpse()
