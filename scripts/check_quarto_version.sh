#!/usr/bin/env bash
# Pre-render guard: refuse to build with a Quarto older than MIN_VERSION.
#
# Quarto < 1.9 renders this site *successfully* but drops the
# screen-reader-only callout labels (<span class="screen-reader-only">Note</span>),
# so the regression is invisible to a sighted reviewer and only shows up as a
# ~50k-line diff of scaffolding churn. Note that `quarto-required` in _quarto.yml
# does NOT enforce this -- it is only honoured for extensions -- hence this hook.
#
# Install Quarto from https://quarto.org/docs/get-started/, not conda-forge.

set -euo pipefail

MIN_VERSION="1.10.0"

# Ask the Quarto that is actually running this render, not whatever `quarto`
# happens to resolve to on PATH -- those can differ (a stale conda/RStudio copy
# earlier on PATH will otherwise mask the very mismatch this guard exists to catch).
# Quarto exports QUARTO_BIN_PATH to pre-render scripts.
if [ -n "${QUARTO_BIN_PATH:-}" ] && [ -x "${QUARTO_BIN_PATH}/quarto" ]; then
  ACTUAL="$("${QUARTO_BIN_PATH}/quarto" --version)"
else
  ACTUAL="$(quarto --version)"
fi

ver_num() { echo "$1" | awk -F. '{printf "%d%03d%03d", $1, $2, $3}'; }

if [ "$(ver_num "$ACTUAL")" -lt "$(ver_num "$MIN_VERSION")" ]; then
  echo "" >&2
  echo "ERROR: Quarto $ACTUAL is too old -- this project requires >= $MIN_VERSION." >&2
  echo "       Rendering with an older Quarto silently produces a downgraded site." >&2
  echo "       Install from https://quarto.org/docs/get-started/ (not conda-forge)." >&2
  echo "" >&2
  exit 1
fi

echo "Quarto $ACTUAL OK (>= $MIN_VERSION)"
