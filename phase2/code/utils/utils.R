# --------------------------
# SHARED R UTILITIES — PHASE 2
# --------------------------
# Source this file from each deliverable script:
#   source(here::here("phase2", "code", "utils", "utils.R"))

library(jsonlite)

# --------------------------
# CATF BRAND THEME
# --------------------------
# Clean Air Task Force brand colors and ggplot2 theme.
# Ported from phase1/code/deliverable03/00_setup.R so all phase 2 deliverables
# can reference a single authoritative copy.

# Primary colors
catf_dark_blue  <- "#0047BB"
catf_blue       <- "#00B5E2"

# Secondary colors
catf_magenta    <- "#C22A90"
catf_purple     <- "#75246C"
catf_lime       <- "#93D500"
catf_teal       <- "#00AE8D"
catf_light_blue <- "#8AB7E9"
catf_navy       <- "#002169"

# Named lookup
catf_colors <- c(
  dark_blue  = "#0047BB",
  blue       = "#00B5E2",
  magenta    = "#C22A90",
  purple     = "#75246C",
  lime       = "#93D500",
  teal       = "#00AE8D",
  light_blue = "#8AB7E9",
  navy       = "#002169"
)

# Categorical palette — 8 visually distinct colors, ordered for stacked bars
catf_palette <- c(
  "#0047BB",  # dark_blue
  "#00AE8D",  # teal
  "#C22A90",  # magenta
  "#93D500",  # lime
  "#00B5E2",  # blue
  "#75246C",  # purple
  "#8AB7E9",  # light_blue
  "#002169"   # navy
)

# Sequential palette (light → dark blue)
catf_sequential <- c("#8AB7E9", "#00B5E2", "#0047BB", "#002169")

# Diverging palette (teal → blue → navy → magenta)
catf_diverging  <- c("#00AE8D", "#00B5E2", "#0047BB", "#75246C", "#C22A90")

#' CATF ggplot2 theme
theme_catf <- function(base_size = 11, base_family = "Helvetica") {
  ggplot2::theme_minimal(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      plot.title      = ggplot2::element_text(face = "bold", size = ggplot2::rel(1.2),
                                              color = catf_navy, margin = ggplot2::margin(b = 10)),
      plot.subtitle   = ggplot2::element_text(size = ggplot2::rel(0.9),
                                              color = catf_dark_blue, margin = ggplot2::margin(b = 10)),
      plot.caption    = ggplot2::element_text(size = ggplot2::rel(0.8), color = "gray50", hjust = 0),
      axis.title      = ggplot2::element_text(size = ggplot2::rel(0.9), color = catf_navy),
      axis.text       = ggplot2::element_text(size = ggplot2::rel(0.85), color = "gray30"),
      axis.line       = ggplot2::element_line(color = "gray70", linewidth = 0.3),
      legend.title    = ggplot2::element_text(face = "bold", size = ggplot2::rel(0.9), color = catf_navy),
      legend.text     = ggplot2::element_text(size = ggplot2::rel(0.85), color = "gray30"),
      legend.position = "bottom",
      legend.key.size = ggplot2::unit(0.8, "lines"),
      panel.grid.major   = ggplot2::element_line(color = "gray90", linewidth = 0.3),
      panel.grid.minor   = ggplot2::element_blank(),
      panel.background   = ggplot2::element_rect(fill = "white", color = NA),
      plot.background    = ggplot2::element_rect(fill = "white", color = NA),
      strip.text         = ggplot2::element_text(face = "bold", size = ggplot2::rel(0.9), color = catf_navy),
      strip.background   = ggplot2::element_rect(fill = "gray95", color = NA),
      plot.margin        = ggplot2::margin(15, 15, 10, 10)
    )
}

#' CATF discrete fill / color scales
scale_fill_catf  <- function(...) ggplot2::scale_fill_manual(values  = catf_palette, ...)
scale_color_catf <- function(...) ggplot2::scale_color_manual(values = catf_palette, ...)

#' CATF sequential (continuous) fill / color scales
scale_fill_catf_seq  <- function(...) ggplot2::scale_fill_gradientn(colors  = catf_sequential, ...)
scale_color_catf_seq <- function(...) ggplot2::scale_color_gradientn(colors = catf_sequential, ...)

# Set CATF as the default session theme
ggplot2::theme_set(theme_catf())

# --------------------------
# unpack_json()
# --------------------------
# Unpack a JSON-object column into multiple named columns, one row per input row.
#
# Usage (pipe-friendly):
#   df |> unpack_json(my_json_col)

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
    return("value")
  }
  return("value")
}

.parse_json_row <- function(x, na_tibble) {
  if (is.null(x) || is.na(x) || x == "" || x == "null") return(na_tibble)
  result <- tryCatch(
    jsonlite::fromJSON(x, simplifyDataFrame = TRUE),
    error = function(e) NULL
  )
  if (is.null(result)) return(na_tibble)
  if (is.data.frame(result)) {
    if (nrow(result) == 0) return(na_tibble)
    result[1L, , drop = FALSE]
  } else if (is.list(result) && !is.null(names(result))) {
    tibble::as_tibble(as.list(result))
  } else {
    tibble::tibble(value = paste(as.character(result), collapse = ", "))
  }
}

#' Unpack a JSON-object column into tidy columns
unpack_json <- function(df, col) {
  col      <- dplyr::enquo(col)
  col_name <- dplyr::as_label(col)
  raw_vec   <- dplyr::pull(df, !!col)
  col_names <- .detect_json_col_names(raw_vec)
  na_tibble <- tibble::tibble(!!!rlang::set_names(
    rep(list(NA_character_), length(col_names)), col_names
  ))
  df |>
    dplyr::mutate(!!col_name := purrr::map(!!col, .parse_json_row, na_tibble = na_tibble)) |>
    tidyr::unnest(!!col, names_sep = "_", keep_empty = TRUE)
}
