# ── Phase 2 D4 candidate review — 20-project sample ─────────────────────────
#
# Just source this one file — it handles everything:
#   source("/Users/Dora/git/consulting/nepa/phase2/code/deliverable04/review_sample20.R")
#
# Then navigate with:
#   show()          display current project
#   nxt()           next project
#   prv()           previous project
#   rnd()           random project
#   goto(5)         jump to project #5
#   goto_id("abc")  jump by project_id
#
# What to check in each project:
#   ★DEC  = candidate the pipeline selected as decision date
#   ★INIT = candidate the pipeline selected as initiation date
#
# Good signs:
#   - ★DEC context shows a signature block, NCO date, or digital sig
#   - ★INIT context shows application received, NOI published, scoping began
#
# Red flags:
#   - ★DEC context mentions "comment period" or "scoping period"  <- Fix 3
#   - ★DEC context mentions "drawn by", "checked by", "sheet"     <- Fix 1
#   - invalid_order projects — check which of the two dates is wrong
#   - missing_initiation — check whether any INIT candidate in the list
#     would have been a reasonable pick
# ─────────────────────────────────────────────────────────────────────────────

NAVIGATOR_PATH <- "/Users/Dora/git/consulting/nepa/phase2/code/utils/timeline_review.R"
SAMPLE_PATH    <- "/Users/Dora/git/consulting/nepa/phase2/output/deliverable04/timeline_review_sample20.parquet"

source(NAVIGATOR_PATH)

# ── Settings — change these in the console at any time, then call show() ──────
DATES_ONLY <- FALSE   # TRUE = show only initiation/decision candidates; FALSE = show all

# Toggle function: type  dates_only()  in the console to flip the filter
dates_only <- function() {
  DATES_ONLY <<- !DATES_ONLY
  cat(sprintf("DATES_ONLY = %s  (%s)\n",
    DATES_ONLY,
    if (DATES_ONLY) "showing initiation + decision only" else "showing all candidates"
  ))
  show()
}

# Override .print_project for Phase 2 layout
.print_project <- function(meta, parsed, source_width = 160) {
  cat(strrep("═", 80), "\n")
  if (!is.null(meta$.idx))
    cat(sprintf("[%d / %d]  %s\n", meta$.idx, meta$.n_total, .meta_chr(meta, "project_id", "")))
  else
    cat(sprintf("%s\n", .meta_chr(meta, "project_id", "")))

  cat(sprintf("Title:       %s\n", .meta_chr(meta, "project_title")))
  cat(sprintf("Agency:      %s\n", .meta_chr(meta, "lead_agency")))
  cat(sprintf("Status:      %s\n", .meta_chr(meta, "timeline_status")))
  cat(sprintf("Review type: %s\n", .meta_chr(meta, "process_type")))
  cat(strrep("─", 80), "\n")
  cat(sprintf("Pipeline initiation : %s\n", .meta_chr(meta, "pipeline_init_date")))
  cat(sprintf("Pipeline decision   : %s\n", .meta_chr(meta, "pipeline_dec_date")))
  cat(sprintf("Candidates          : %s\n", .meta_chr(meta, "n_candidates")))
  if (DATES_ONLY) cat("  [filter: initiation + decision only — type dates_only() to toggle]\n")
  cat(strrep("─", 80), "\n")

  if (is.null(parsed) || nrow(parsed) == 0) {
    cat("(no candidates)\n")
  } else {
    tbl <- parsed |>
      mutate(context = stringr::str_squish(source)) |>
      arrange(as.Date(date)) |>
      select(type, date, context)

    if (DATES_ONLY) {
      tbl <- tbl |>
        filter(grepl("dec|init", type, ignore.case = TRUE))
    }

    if (nrow(tbl) == 0) {
      cat("(no initiation or decision candidates)\n")
    } else {
      for (i in seq_len(nrow(tbl))) {
        row <- tbl[i, ]
        cat(sprintf("%-24s  %s\n", row$type, row$date))
        wrapped <- strwrap(row$context, width = source_width, indent = 4, exdent = 4)
        cat(paste(wrapped, collapse = "\n"), "\n\n")
      }
    }
  }
  cat(strrep("═", 80), "\n")
}

load_timeline_navigator(
  parquet_path = SAMPLE_PATH,
  json_col     = "candidates_json",
  skip_empty   = FALSE,
  source_width = 160
)

show()
nxt()
