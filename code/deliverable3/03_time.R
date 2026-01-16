# --------------------------
# DELIVERABLE 3: CHANGE OVER TIME
# --------------------------
# Table 3: Change Over Time
# Analyzes clean energy projects by year (placeholder - requires timeline extraction)

# --------------------------
# SETUP
# --------------------------

source(here::here("code", "deliverable3", "00_setup.R"))

# --------------------------
# TABLE 3: CHANGE OVER TIME
# --------------------------

cat("Creating Table 3: Change Over Time...\n")
cat("  Note: Timeline extraction not yet implemented.\n")
cat("  This requires parsing dates from document text.\n")

# Create placeholder table
table3 <- data.frame(
  Year = c("2025", "2024", "2023", "2022", "2021", "Total"),
  `Environmental Assessment` = rep("TBD", 6),
  `Environmental Impact Statement` = rep("TBD", 6),
  `Categorical Exclusion` = rep("TBD", 6),
  check.names = FALSE
)

table3 %>% print()

output_file3 <- here(tables_dir, "table3_by_year_placeholder.csv")
write_csv(table3, output_file3)
cat("  Saved placeholder:", output_file3, "\n")

# --------------------------
# SUMMARY
# --------------------------

cat("\n=== Time Script Complete ===\n")
cat("Tables saved to:", tables_dir, "\n")
