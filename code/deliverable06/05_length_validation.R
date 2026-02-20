# --------------------------
# DELIVERABLE 6: LENGTH EXTRACTION VALIDATION TABLES
# --------------------------

source(here::here("code", "deliverable06", "00_setup.R"))

set.seed(606)

analysis <- prepare_deliverable6_data() %>%
  mutate(
    process_group = toupper(as.character(coalesce(process_group, process_type, dataset_source))),
    transmission_flag = coalesce(project_is_transmission, FALSE),
    pipeline_flag = coalesce(project_is_pipeline, FALSE)
  )

confidence_bucket <- function(x) {
  txt <- str_to_lower(coalesce(as.character(x), ""))
  case_when(
    txt == "" ~ "missing",
    str_detect(txt, "high") ~ "high",
    str_detect(txt, "med") ~ "medium",
    str_detect(txt, "low") ~ "low",
    TRUE ~ "other"
  )
}

length_bucket <- function(x) {
  case_when(
    is.na(x) ~ "missing",
    x < 1 ~ "<1 mile",
    x < 10 ~ "1-10 miles",
    x < 50 ~ "10-50 miles",
    x < 100 ~ "50-100 miles",
    TRUE ~ "100+ miles"
  )
}

sample_balanced_by_strata <- function(df, strata_col, n_target) {
  if (nrow(df) == 0 || n_target <= 0) return(df[0, ])

  df2 <- df %>% mutate(.row_id = row_number())
  n_target <- min(n_target, nrow(df2))
  strata_values <- unique(df2[[strata_col]])
  per_stratum <- ceiling(n_target / max(length(strata_values), 1))

  sampled <- purrr::map_dfr(strata_values, function(stratum_value) {
    chunk <- df2 %>% filter(.data[[strata_col]] == stratum_value)
    if (nrow(chunk) == 0) return(chunk)
    chunk %>% slice_sample(n = min(per_stratum, nrow(chunk)))
  })

  if (nrow(sampled) > n_target) {
    sampled <- sampled %>% slice_sample(n = n_target)
  }

  if (nrow(sampled) < n_target) {
    remaining <- anti_join(df2, sampled, by = ".row_id")
    if (nrow(remaining) > 0) {
      sampled <- bind_rows(
        sampled,
        remaining %>% slice_sample(n = min(n_target - nrow(sampled), nrow(remaining)))
      )
    }
  }

  sampled %>% select(-.row_id)
}

build_validation_sample <- function(df, tech_name, flag_col, length_col, conf_col, source_col,
                                    n_nonmissing = 80, n_missing = 20) {
  base <- df %>%
    filter(.data[[flag_col]]) %>%
    mutate(
      technology = tech_name,
      extracted_length_miles = .data[[length_col]],
      extraction_confidence = as.character(.data[[conf_col]]),
      extraction_source_text = as.character(.data[[source_col]]),
      confidence_bucket = confidence_bucket(.data[[conf_col]]),
      length_bucket = length_bucket(.data[[length_col]])
    )

  nonmissing <- base %>% filter(!is.na(extracted_length_miles))
  missing <- base %>% filter(is.na(extracted_length_miles))

  nonmissing_sample <- sample_balanced_by_strata(nonmissing, "confidence_bucket", n_nonmissing) %>%
    mutate(sample_type = "nonmissing_length_review")
  missing_sample <- sample_balanced_by_strata(missing, "process_group", n_missing) %>%
    mutate(sample_type = "missing_length_gap_review")

  bind_rows(nonmissing_sample, missing_sample) %>%
    transmute(
      technology,
      sample_type,
      project_id,
      process_group,
      project_state_primary,
      project_title = str_squish(project_title_txt),
      project_type = str_squish(project_type_txt),
      extracted_length_miles,
      confidence_bucket,
      extraction_confidence,
      length_bucket,
      extraction_source_text,
      manual_length_found = "",
      manual_length_miles = "",
      manual_source_excerpt = "",
      manual_notes = ""
    )
}

transmission_sample <- build_validation_sample(
  analysis,
  tech_name = "Transmission",
  flag_col = "transmission_flag",
  length_col = "project_transmission_length_miles",
  conf_col = "project_transmission_length_confidence",
  source_col = "project_transmission_length_source_text",
  n_nonmissing = 90,
  n_missing = 30
)

pipeline_sample <- build_validation_sample(
  analysis,
  tech_name = "Pipeline",
  flag_col = "pipeline_flag",
  length_col = "project_pipeline_length_miles",
  conf_col = "project_pipeline_length_confidence",
  source_col = "project_pipeline_length_source_text",
  n_nonmissing = 90,
  n_missing = 50
)

coverage_summary <- bind_rows(
  analysis %>%
    filter(transmission_flag) %>%
    summarise(
      technology = "Transmission",
      n_projects = n(),
      n_with_length = sum(!is.na(project_transmission_length_miles)),
      pct_with_length = 100 * n_with_length / n_projects,
      n_high_conf = sum(confidence_bucket(project_transmission_length_confidence) == "high"),
      n_medium_conf = sum(confidence_bucket(project_transmission_length_confidence) == "medium"),
      n_low_conf = sum(confidence_bucket(project_transmission_length_confidence) == "low"),
      n_missing_conf = sum(confidence_bucket(project_transmission_length_confidence) == "missing"),
      median_length_miles = median(project_transmission_length_miles, na.rm = TRUE)
    ),
  analysis %>%
    filter(pipeline_flag) %>%
    summarise(
      technology = "Pipeline",
      n_projects = n(),
      n_with_length = sum(!is.na(project_pipeline_length_miles)),
      pct_with_length = 100 * n_with_length / n_projects,
      n_high_conf = sum(confidence_bucket(project_pipeline_length_confidence) == "high"),
      n_medium_conf = sum(confidence_bucket(project_pipeline_length_confidence) == "medium"),
      n_low_conf = sum(confidence_bucket(project_pipeline_length_confidence) == "low"),
      n_missing_conf = sum(confidence_bucket(project_pipeline_length_confidence) == "missing"),
      median_length_miles = median(project_pipeline_length_miles, na.rm = TRUE)
    )
)

write_csv(coverage_summary, here(tables_dir, "table_length_extraction_coverage.csv"))
write_csv(transmission_sample, here(tables_dir, "table_transmission_length_validation_sample.csv"))
write_csv(pipeline_sample, here(tables_dir, "table_pipeline_length_validation_sample.csv"))

cat("Saved outputs to:\n", tables_dir, "\n")
cat("Transmission validation sample rows:", nrow(transmission_sample), "\n")
cat("Pipeline validation sample rows:", nrow(pipeline_sample), "\n")
