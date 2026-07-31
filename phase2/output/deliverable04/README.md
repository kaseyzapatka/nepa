# D4 output/ — generated artifacts (regenerable)

**Everything in this directory is produced by the pipeline and can be deleted and rebuilt.**
It holds no inputs. Hand-labeled assets (the things you must not lose) live in
`phase2/training/deliverable04/`, not here.

## Layout

| dir | contents | tracked? |
|---|---|---|
| `diagnostics/` | numbered pipeline diagnostics (`01_…`–`07_…`), `d4_*.csv` analysis tables, `*_eval_errors.csv`, `*_eval_summary.csv`, QA reports | report-read `d4_*.csv` force-tracked (`git add -f`); all other CSVs gitignored (`*.csv`) |
| `figures/` | `fig_d4_*.png` charts | tracked |
| `gold/` | gold-split definitions read by `labeling/` + `07_validate.py` | tracked |

## Rules

1. **Inputs ≠ outputs.** Labels/gold are inputs → `training/deliverable04/`. Anything a script
   writes goes here and must be regenerable from code + the training inputs.
2. **No code in `output/`.** Scripts live in `code/deliverable04/` (one-offs there too, prefixed `_`).
3. **Large + transient artifacts are gitignored**, never committed (the repo-wide `*.csv` rule
   already covers most of this; model `*checkpoints/` are intentionally excluded). Exception:
   the report-read `d4_*.csv` diagnostics (the ~20 tables `deliverable04.qmd` reads via
   `read_diag()`, including `d4_duration_by_technology.csv`) are force-tracked with `git add -f`
   so a public clone can render the report without re-running the pipeline. All other
   `diagnostics/*.csv` (the numbered `01_…`–`07_…` QA tables, `*_eval_*`, sections QA) stay
   gitignored and regenerate on the next pipeline run.
