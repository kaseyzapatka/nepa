# ── Timeline Review Navigator ─────────────────────────────────────────────────
# Two functions:
#
# load_timeline_navigator(dataset, ...)
#   Loads a dataset and installs navigation functions into the global env.
#   Source once, then use these in the console:
#
#     show()              display current project
#     nxt()               next project
#     prv()               previous project
#     goto(n)             jump to index n (1-based)
#     goto_id("abc")      jump to a specific project_id
#     rnd()               jump to a random project
#     use("ea_llm")       switch to a different dataset without re-sourcing
#     datasets()          list all available dataset shortcuts
#
#   dataset shortcuts (pass parquet_path directly for custom files):
#
#   Full runs:
#     "ce"           projects_timeline_bert.parquet        CE post bert-run
#     "ea"           projects_timeline_bert_ea.parquet     EA post bert-run
#     "ea_llm"       projects_timeline_bert_ea_llm.parquet EA post LLM adjudication
#     "eis_llm"      projects_timeline_bert_eis_llm.parquet EIS post LLM adjudication
#
#   Training test samples:
#     "test_ce"      test_ce_post_retrain.parquet          CE 50-project post-retrain
#     "test_ce_100"  test100_bert_refactored.parquet       CE 100-project bert test
#     "test_ce_1k"   test1000_bert_refactored.parquet      CE 1000-project bert test
#     "test_ea"      test100_bert_ea_refactored.parquet    EA 100-project bert test
#     "test_ea_50"   test50_ea.parquet                     EA 50-project bert test
#     "test_ea_llm"  test50_ea_llm_v3.parquet              EA 50-project LLM test
#     "test_eis"     test50_bert_eis_refactored.parquet    EIS 50-project bert test
#     "test_eis_llm" test50_eis_llm.parquet                EIS 50-project LLM test
#
#   Regex candidate caches:
#     "regex_ce"     timeline_regex_ce.parquet             CE regex candidates
#     "regex_ea"     timeline_regex_ea.parquet             EA regex candidates
#     "regex_eis"    timeline_regex_eis.parquet            EIS regex candidates
#
# view_project(pid, dataset)
#   One-off display of a single project_id without loading a navigator.
# ─────────────────────────────────────────────────────────────────────────────

library(dplyr)
library(purrr)
library(jsonlite)
library(stringr)
library(lubridate)
library(arrow)

.TIMELINE_PATHS <- list(
  # ── Full runs (post bert-run / post LLM) ────────────────────────────────────
  ce      = here::here("data", "analysis", "timeline_ce.parquet"),
  ea      = here::here("data", "analysis", "timeline_ea.parquet"),
  ea_llm  = here::here("data", "analysis", "timeline_ea_llm.parquet"),
  eis     = here::here("data", "analysis", "timeline_eis.parquet"),
  eis_llm = here::here("data", "analysis", "timeline_eis_llm.parquet"),

  # ── Training test samples (post bert-train, before full run) ────────────────
  # Update these shortcuts to point to your latest --bert-run --sample output
  test_ce  = here::here("data", "analysis", "timeline_ce_sample20.parquet"),
  test_ea  = here::here("data", "analysis", "timeline_ea_sample20.parquet"),
  test_eis = here::here("data", "analysis", "timeline_eis_sample20.parquet"),

   # ── Regex samples (Rebuild regex cache for all sources) ────────────────
  # Update these shortcuts to point to your latest --bert-run --sample output
  regex_ce  = here::here("data", "analysis", "timeline_regex_ce.parquet"),
  regex_ea  = here::here("data", "analysis", "timeline_regex_ea.parquet"),
  regex_eis = here::here("data", "analysis", "timeline_regex_eis.parquet")
)

# Maps (stage, process) → dataset shortcut. NA = combination doesn't exist.
.STAGE_MAP <- list(
  regex = c(ce = "regex_ce",  ea = "regex_ea",  eis = "regex_eis"),
  post_training = c(ce = "test_ce",  ea = "test_ea",  eis = "test_eis"),
  post_run      = c(ce = "ce",       ea = "ea",        eis = "eis"),
  post_llm      = c(ce = NA,         ea = "ea_llm",    eis = "eis_llm")
)

.resolve_stage <- function(stage, process) {
  if (!stage   %in% names(.STAGE_MAP))
    stop("Unknown stage '", stage,   "'. Options: ", paste(names(.STAGE_MAP), collapse = ", "))
  if (!process %in% names(.STAGE_MAP[[stage]]))
    stop("Unknown process '", process, "'. Options: ce, ea, eis")
  key <- .STAGE_MAP[[stage]][[process]]
  if (is.na(key))
    stop(sprintf("No data available for stage='%s', process='%s'", stage, process))
  key
}

.safe_parse_dates <- function(x) {
  if (is.na(x) || !nzchar(x) || x == "[]") return(NULL)
  tryCatch({
    out <- fromJSON(x, flatten = TRUE)
    if (is.data.frame(out)) return(as_tibble(out))
    if (is.list(out))       return(bind_rows(lapply(out, as_tibble)))
    NULL
  }, error = function(e) NULL)
}

.meta_value <- function(meta, col) {
  if (!col %in% names(meta)) return(NULL)
  value <- meta[[col]]
  if (is.list(value)) value <- value[[1]]
  if (length(value) == 0) return(NULL)
  value[[1]]
}

.meta_chr <- function(meta, col, default = "\u2014") {
  value <- .meta_value(meta, col)
  if (is.null(value) || length(value) == 0 || is.na(value)) return(default)
  value <- as.character(value)[1]
  if (!nzchar(value)) return(default)
  value
}

.parsed_context <- function(parsed) {
  for (col in c("source", "context_cleaned", "context")) {
    if (col %in% names(parsed)) {
      value <- as.character(parsed[[col]])
      value[is.na(value)] <- ""
      return(value)
    }
  }
  rep("", nrow(parsed))
}

.is_regex_candidate_dataset <- function(raw, json_col) {
  !json_col %in% names(raw) &&
    all(c("project_id", "date", "match", "context") %in% names(raw))
}

.read_project_metadata <- function(project_ids) {
  meta_path <- here::here("data", "analysis", "projects_combined.parquet")
  if (!file.exists(meta_path)) {
    return(tibble(
      project_id    = project_ids,
      project_title = NA_character_,
      lead_agency   = NA_character_
    ))
  }

  read_parquet(
    meta_path,
    col_select = c("project_id", "project_title", "lead_agency")
  ) |>
    as_tibble() |>
    filter(project_id %in% project_ids) |>
    distinct(project_id, .keep_all = TRUE)
}

.normalize_regex_candidates <- function(raw, json_col) {
  project_ids <- unique(as.character(raw$project_id))
  meta <- .read_project_metadata(project_ids)

  for (col in c("match", "context", "section_label", "doc_type")) {
    if (!col %in% names(raw)) raw[[col]] <- NA_character_
  }
  if (!"position" %in% names(raw)) raw$position <- NA_integer_
  for (col in c("sig_flag", "ner_decision_signal")) {
    if (!col %in% names(raw)) raw[[col]] <- FALSE
  }

  raw |>
    as_tibble() |>
    mutate(
      project_id = as.character(project_id),
      date = as.character(date),
      match = coalesce(as.character(match), ""),
      context = coalesce(as.character(context), ""),
      section_label = coalesce(as.character(section_label), ""),
      doc_type = coalesce(as.character(doc_type), ""),
      position = suppressWarnings(as.integer(position)),
      sig_flag = coalesce(as.logical(sig_flag), FALSE),
      ner_decision_signal = coalesce(as.logical(ner_decision_signal), FALSE),
      type = case_when(
        sig_flag & ner_decision_signal ~ "sig+ner",
        sig_flag ~ "signature",
        ner_decision_signal ~ "ner_signal",
        TRUE ~ "candidate"
      ),
      source = str_squish(paste0(
        "[match: ", match,
        if_else(nzchar(doc_type), paste0(" | doc: ", doc_type), ""),
        if_else(nzchar(section_label), paste0(" | section: ", section_label), ""),
        if_else(sig_flag, " | sig", ""),
        if_else(ner_decision_signal, " | ner", ""),
        "] ",
        context
      ))
    ) |>
    arrange(project_id, suppressWarnings(as.Date(date)), position, match) |>
    group_by(project_id) |>
    summarise(
      regex_candidate_count = n(),
      !!json_col := toJSON(
        pmap(
          list(type, date, source),
          function(type, date, source) {
            list(type = type, date = date, source = source)
          }
        ),
        auto_unbox = TRUE
      ),
      .groups = "drop"
    ) |>
    left_join(meta, by = "project_id") |>
    select(project_id, project_title, lead_agency, regex_candidate_count, all_of(json_col))
}

.read_review_dataset <- function(path, json_col) {
  raw <- read_parquet(path) |> as_tibble()
  if (.is_regex_candidate_dataset(raw, json_col)) {
    return(.normalize_regex_candidates(raw, json_col))
  }
  raw
}

.print_project <- function(meta, parsed, source_width = 150) {
  cat(strrep("\u2550", 80), "\n")

  if (!is.null(meta$.idx))
    cat(sprintf("[%d / %d]  %s\n", meta$.idx, meta$.n_total, .meta_chr(meta, "project_id", "")))
  else
    cat(sprintf("%s\n", .meta_chr(meta, "project_id", "")))

  cat(sprintf("Title:    %s\n", .meta_chr(meta, "project_title")))
  cat(sprintf("Agency:   %s\n", .meta_chr(meta, "lead_agency")))
  cat(strrep("\u2500", 80), "\n")
  cat(sprintf("BERT initiation : %s\n",
              .meta_chr(meta, "bert_initiation_date_final")))
  cat(sprintf("BERT decision   : %s\n",
              .meta_chr(meta, "bert_decision_date_final")))

  if ("llm_initiation_date" %in% names(meta))
    cat(sprintf("LLM  initiation : %s\n",
                .meta_chr(meta, "llm_initiation_date")))
  if ("llm_decision_date" %in% names(meta))
    cat(sprintf("LLM  decision   : %s\n",
                .meta_chr(meta, "llm_decision_date")))
  if ("regex_candidate_count" %in% names(meta))
    cat(sprintf("Regex candidates: %s\n",
                .meta_chr(meta, "regex_candidate_count")))

  cat(strrep("\u2500", 80), "\n")

  if (is.null(parsed) || nrow(parsed) == 0) {
    cat("(no dates extracted)\n")
  } else {
    if (!"type" %in% names(parsed)) parsed$type <- "date"
    tbl <- parsed |>
      mutate(
        date    = as_date(date),
        context = str_squish(.parsed_context(parsed))
      ) |>
      arrange(date) |>
      select(type, date, context)

    for (i in seq_len(nrow(tbl))) {
      row <- tbl[i, ]
      cat(sprintf("%-12s %s\n", row$type, row$date))
      wrapped <- strwrap(row$context, width = source_width, indent = 2, exdent = 2)
      cat(paste(wrapped, collapse = "\n"), "\n\n")
    }
  }

  cat(strrep("\u2550", 80), "\n")
}

# ── load_timeline_navigator() ─────────────────────────────────────────────────

load_timeline_navigator <- function(
  dataset      = "ce",
  parquet_path = NULL,
  json_col     = "bert_dates_json",
  skip_empty   = TRUE,
  source_width = 150
) {
  # Resolve path
  path <- if (!is.null(parquet_path)) parquet_path else {
    if (!dataset %in% names(.TIMELINE_PATHS))
      stop("Unknown dataset '", dataset, "'. Run datasets() to see options.")
    .TIMELINE_PATHS[[dataset]]
  }

  # Mutable state shared across all closures
  state <- new.env(parent = emptyenv())
  state$json_col     <- json_col
  state$source_width <- source_width
  state$skip_empty   <- skip_empty
  state$idx          <- 1L

  # Inner loader — called on init and by use()
  .load <- function(path, label) {
    state$raw <- NULL  # drop old data before loading new
    gc()

    raw <- .read_review_dataset(path, state$json_col)
    has_dates <- map_lgl(raw[[state$json_col]], function(x) !is.na(x) && nzchar(x) && x != "[]")
    ids <- if (state$skip_empty) raw$project_id[has_dates] else raw$project_id

    state$raw     <- raw
    state$ids     <- ids
    state$n_total <- length(ids)
    state$label   <- label
    state$idx     <- 1L

    cat(strrep("\u2550", 80), "\n")
    cat(sprintf("Dataset : %s\n", label))
    cat(sprintf("File    : %s\n", basename(path)))
    cat(sprintf("Projects: %d (with dates)\n", state$n_total))
    cat(strrep("\u2550", 80), "\n")
  }

  .load(path, if (!is.null(parquet_path)) basename(parquet_path) else dataset)

  .show <- function() {
    pid    <- state$ids[[state$idx]]
    meta   <- state$raw |> filter(project_id == pid) |> slice(1)
    meta$.idx     <- state$idx
    meta$.n_total <- state$n_total
    parsed <- .safe_parse_dates(meta[[state$json_col]])
    .print_project(meta, parsed, source_width = state$source_width)
    invisible(pid)
  }

  .use <- function(new_dataset, parquet_path = NULL) {
    new_path <- if (!is.null(parquet_path)) parquet_path else {
      if (!new_dataset %in% names(.TIMELINE_PATHS))
        stop("Unknown dataset '", new_dataset, "'. Run datasets() to see options.")
      .TIMELINE_PATHS[[new_dataset]]
    }
    label <- if (!is.null(parquet_path)) basename(parquet_path) else new_dataset
    .load(new_path, label)
    .show()
  }

  assign("show",     function()           { .show() },                                       envir = .GlobalEnv)
  assign("nxt",      function()           { state$idx <- min(state$idx + 1L, state$n_total); .show() }, envir = .GlobalEnv)
  assign("prv",      function()           { state$idx <- max(state$idx - 1L, 1L);            .show() }, envir = .GlobalEnv)
  assign("goto",     function(n)          { state$idx <- max(1L, min(as.integer(n), state$n_total)); .show() }, envir = .GlobalEnv)
  assign("goto_id",  function(pid) {
    i <- which(state$ids == pid)
    if (length(i) == 0) { message("project_id not found: ", pid); return(invisible(NULL)) }
    state$idx <- i[[1L]]
    .show()
  }, envir = .GlobalEnv)
  assign("rnd",      function()           { state$idx <- sample.int(state$n_total, 1L); .show() }, envir = .GlobalEnv)
  assign("use",       function(d, parquet_path = NULL) { .use(d, parquet_path) },             envir = .GlobalEnv)
  assign("use_stage", function(stage, process) { .use(.resolve_stage(stage, process), NULL) }, envir = .GlobalEnv)
  assign("datasets", function() {
    cat("Available dataset shortcuts:\n\n")
    cat("  regex:\n")
    cat("    regex_ce  timeline_regex_ce.parquet   CE regex candidates\n")
    cat("    regex_ea  timeline_regex_ea.parquet   EA regex candidates\n")
    cat("    regex_eis timeline_regex_eis.parquet  EIS regex candidates\n")
    cat("\n  post_run:\n")
    cat("    ce       timeline_ce.parquet        CE full bert-run\n")
    cat("    ea       timeline_ea.parquet        EA full bert-run\n")
    cat("    eis      timeline_eis.parquet       EIS full bert-run\n")
    cat("\n  post_llm:\n")
    cat("    ea_llm   timeline_ea_llm.parquet    EA post LLM adjudication\n")
    cat("    eis_llm  timeline_eis_llm.parquet   EIS post LLM adjudication\n")
    cat("\n  post_training (update these paths after each training run):\n")
    cat("    test_ce  timeline_ce_sample{N}.parquet\n")
    cat("    test_ea  timeline_ea_sample{N}.parquet\n")
    cat("    test_eis timeline_eis_sample{N}.parquet\n")
    cat("\n  Or pass any file directly:\n")
    cat('    use("ce", parquet_path = here("data","analysis","timeline_ce_sample20_v2.parquet"))\n')
    invisible(NULL)
  }, envir = .GlobalEnv)

  invisible(NULL)
}

# ── view_project() ────────────────────────────────────────────────────────────

view_project <- function(pid, dataset = "ce", parquet_path = NULL,
                         json_col = "bert_dates_json", source_width = 150) {
  path <- if (!is.null(parquet_path)) parquet_path else {
    if (!dataset %in% names(.TIMELINE_PATHS))
      stop("Unknown dataset '", dataset, "'. Run datasets() to see options.")
    .TIMELINE_PATHS[[dataset]]
  }

  raw  <- .read_review_dataset(path, json_col)
  meta <- raw |> filter(project_id == pid) |> slice(1)

  if (nrow(meta) == 0) {
    message("project_id not found: ", pid)
    return(invisible(NULL))
  }

  parsed <- .safe_parse_dates(meta[[json_col]])
  .print_project(meta, parsed, source_width = source_width)
  invisible(pid)
}
