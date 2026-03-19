#!/bin/bash
# Post-render script: generate PDF versions of key pages and place them in docs/reports/
# Runs automatically after `quarto render` completes.

set -e

echo "Rendering PDFs..."

quarto render reports/project_overview.qmd --to pdf --output docs/reports/project_overview.pdf
echo "  project_overview.pdf done"

quarto render reports/key_insights.qmd --to pdf --output docs/reports/key_insights.pdf
echo "  key_insights.pdf done"

echo "PDFs complete."
