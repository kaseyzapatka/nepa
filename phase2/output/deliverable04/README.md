# D4 output/ — generated artifacts (regenerable)

**Everything in this directory is produced by the pipeline and can be deleted and rebuilt.**
It holds no inputs. Hand-labeled assets (the things you must not lose) live in
`phase2/training/deliverable04/`, not here.

## Layout

| dir | contents | tracked? |
|---|---|---|
| `diagnostics/` | numbered pipeline diagnostics (`01_…`–`07_…`), `d4_*.csv` analysis tables, `*_eval_errors.csv`, `*_eval_summary.csv`, QA reports | CSVs gitignored (`*.csv`) |
| `figures/` | `fig_d4_*.png` charts | tracked |
| `gold/` | gold-split definitions read by `labeling/` + `07_validate.py` | tracked |
| `review_queues/` | large manual-review / API-validation CSVs (28 MB, 15 MB) | gitignored (big + `*.csv`) |

## Rules

1. **Inputs ≠ outputs.** Labels/gold are inputs → `training/deliverable04/`. Anything a script
   writes goes here and must be regenerable from code + the training inputs.
2. **No code in `output/`.** Scripts live in `code/deliverable04/` (one-offs there too, prefixed `_`).
3. **Large + transient artifacts are gitignored**, never committed (the repo-wide `*.csv` rule
   already covers most of this; `review_queues/` and model `*checkpoints/` are intentionally excluded).
