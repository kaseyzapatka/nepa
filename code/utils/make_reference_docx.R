#!/usr/bin/env Rscript
# Generates reports/catf-reference.docx with CATF brand colors applied to heading styles.
# Run once: Rscript reports/make_reference_docx.R

library(here)

ref_path <- here("reports", "catf-reference.docx")
tmp_dir  <- tempfile(pattern = "catf_ref_")
dir.create(tmp_dir)

# 1. Generate default pandoc reference.docx
system2("pandoc", c("--print-default-data-file", "reference.docx"),
        stdout = ref_path)
message("Generated default reference.docx → ", ref_path)

# 2. Unzip into temp dir
system2("unzip", c("-q", ref_path, "-d", tmp_dir))

# 3. Patch word/styles.xml using text replacement
#    Word uses w:themeColor to override w:val, so we must remove it along with
#    w:themeShade/w:themeTint and set the explicit hex color.
styles_file <- file.path(tmp_dir, "word", "styles.xml")
txt <- paste(readLines(styles_file, warn = FALSE), collapse = "\n")

# Helper: within the first <w:style> block matching a given styleId, replace the
# <w:color .../> element's attributes, stripping theme overrides.
patch_color <- function(text, style_id, hex_color) {
  # Match the full <w:color ... /> inside the target style block
  # Strategy: locate the styleId, then find the next w:color tag
  pattern <- sprintf(
    '(styleId="%s"(?s:.)*?<w:color )([^/]*?)(/>)',
    style_id
  )
  replacement <- sprintf('\\1w:val="%s" \\3', hex_color)
  result <- sub(pattern, replacement, text, perl = TRUE)
  if (identical(result, text)) {
    message("  WARNING: no match for ", style_id)
  } else {
    message("  Patched ", style_id, " → #", hex_color)
  }
  result
}

txt <- patch_color(txt, "Heading1", "012169")
txt <- patch_color(txt, "Heading2", "012169")
txt <- patch_color(txt, "Heading3", "0047BB")

writeLines(txt, styles_file)

# 4. Rezip as docx
file.remove(ref_path)
old_wd <- setwd(tmp_dir)
system2("zip", c("-q", "-r", ref_path, "."))
setwd(old_wd)

# 5. Cleanup
unlink(tmp_dir, recursive = TRUE)
message("Done → ", ref_path)
