#!/bin/bash
# Post-render script: generate PDF versions of key pages and place them in docs/reports/
# Runs automatically after `quarto render` completes.

# Guard against infinite recursion: the quarto renders below also trigger this
# post-render hook. Exit immediately if we're already inside a PDF render pass.
if [ -n "$NEPA_RENDERING_PDFS" ]; then
  exit 0
fi
export NEPA_RENDERING_PDFS=1

set -e

echo "Rendering PDFs..."

quarto render phase1/reports/project_overview.qmd --to pdf
echo "  project_overview.pdf done"

quarto render phase1/reports/key_insights.qmd --to pdf
echo "  key_insights.pdf done"

echo "PDFs complete."
