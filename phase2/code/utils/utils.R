# --------------------------
# SHARED R UTILITIES
# --------------------------
# Source this file from each deliverable's 00_setup.R:
#   source(here::here("code", "utils", "utils.R"))

library(jsonlite)

# --------------------------
# unpack_json()
# --------------------------
# Unpack a JSON-object column into multiple named columns, one row per input row.
#
# Usage (pipe-friendly):
#   df |> unpack_json(my_json_col)
#
# The original column is replaced by its parsed sub-columns, prefixed with
# "<col>_" (controlled by names_sep in unnest).
#
# Guarantees:
#   - nrow(out) == nrow(df)  — no rows are added or dropped
#   - Rows with NULL / NA / "" / "null" JSON get NA in all output columns
#   - Unexpected parse failures fall back to NA (no silent errors)

# Detect column names from the first successfully parsed value
.detect_json_col_names <- function(x_vec) {
  for (x in x_vec) {
    if (is.null(x) || is.na(x) || x == "" || x == "null") next
    result <- tryCatch(
      jsonlite::fromJSON(x, simplifyDataFrame = TRUE),
      error = function(e) NULL
    )
    if (is.null(result)) next
    if (is.data.frame(result) && ncol(result) > 0) return(names(result))
    if (is.list(result) && !is.null(names(result)))  return(names(result))
    return("value")  # unnamed scalar/vector fallback
  }
  return("value")
}

# Parse a single JSON string into exactly one tibble row.
# na_tibble is pre-built so all fallback rows share the same schema.
.parse_json_row <- function(x, na_tibble) {
  if (is.null(x) || is.na(x) || x == "" || x == "null") return(na_tibble)

  result <- tryCatch(
    jsonlite::fromJSON(x, simplifyDataFrame = TRUE),
    error = function(e) NULL
  )

  if (is.null(result)) return(na_tibble)

  if (is.data.frame(result)) {
    if (nrow(result) == 0) return(na_tibble)
    result[1L, , drop = FALSE]          # clamp to 1 row; avoids row expansion
  } else if (is.list(result) && !is.null(names(result))) {
    as_tibble(as.list(result))
  } else {
    # Scalar or unnamed vector — collapse to a single string value
    tibble(value = paste(as.character(result), collapse = ", "))
  }
}

#' Unpack a JSON-object column into tidy columns
#'
#' @param df   A data frame.
#' @param col  Unquoted name of the JSON column to unpack.
#' @return     The same data frame with the JSON column replaced by its parsed
#'             sub-columns (prefixed "<col>_").  nrow is always preserved.
unpack_json <- function(df, col) {
  col      <- enquo(col)
  col_name <- as_label(col)

  raw_vec   <- df |> pull(!!col)
  col_names <- .detect_json_col_names(raw_vec)

  na_tibble <- tibble(!!!set_names(
    rep(list(NA_character_), length(col_names)),
    col_names
  ))

  df |>
    mutate(!!col_name := map(!!col, .parse_json_row, na_tibble = na_tibble)) |>
    unnest(!!col, names_sep = "_", keep_empty = TRUE)
}
