# D2 handoff — what's built, and your turn

The entire D2 code surface is written and every deterministic stage has been **run and
verified** against live data under `SCHEMA_VERSION = d2_v2_11`. What remains needs your key,
your labels, or your budget approval — those are the only stopping points.

## Built & run (deterministic, key-free) ✅

```bash
conda run -n nepa python phase2/code/deliverable02/_run.py            # 00 regime -> 01 corpus+cohorts
conda run -n nepa python phase2/code/deliverable02/03_build_gold_set_queue.py
conda run -n nepa python phase2/code/deliverable02/02_extract_fonsi_significance.py --dry-run
conda run -n nepa python phase2/code/deliverable02/04_extract_eis_significance.py --dry-run --sample 800
Rscript phase2/code/deliverable02/06_create_figures.R
quarto render phase2/reports/deliverable02.qmd
```

Verified: regime 1,326 rows; corpus 1,205 (FONSI 452); `agency_scope_status` 427/23/2 FONSI &
406/283/64 EIS; determinations 3,478 with **3,478/3,478 unique IDs** (no collisions);
`framework_regime = decision_period` (0 mismatches); mitigation grain clean; report renders to
`docs/phase2/reports/deliverable02.html`.

## Your turn

### 1. Gate 1 — review the corpus (small, ~1 hr)
`phase2/output/deliverable02/corpus_membership_review.csv` — confirm the ~34 primary mitigated
FONSIs; check `off_mission_flag` rows (NNSA/defense/nuclear) for exclusion.

### 2. Gold labeling (the critical-path bottleneck) — multi-determination grain
`significance_gold_queue.csv` (300 pos + 100 neg windows, stratified) is a **reading list**. Both
labelers (Claude + Codex) read every window and write a LONG CSV — **one row per
`(evidence_span_id × resource_area)` determination** — to `gold/labels_claude.csv` /
`gold/labels_codex.csv` per `gold_labeling.md` (this grain matches the extractor, which emits one
determination per resource area). Then:
```bash
conda run -n nepa python phase2/code/deliverable02/gold_agreement.py             # align on (window x resource); auto-accept core agreements; write gold_disagreements.csv
# analyst fills final_* in output/deliverable02/gold_disagreements.csv (blank = drop that row)
conda run -n nepa python phase2/code/deliverable02/gold_agreement.py --finalize  # -> gold/significance_gold.parquet (30% holdout BY WINDOW)
conda run -n nepa python phase2/code/deliverable02/05_validate_significance.py   # window + resource-set + class/mitigation metrics
```

### 3. Billable LLM pass — FONSI first, via the Batch API (needs budget approval; you run it)
Key is in keychain `nepa-anthropic`; `anthropic` is installed in `nepa`.
**Spike first** (~30 windows, synchronous, ~$1), eyeball the output:
```bash
conda run -n nepa python phase2/code/deliverable02/02_extract_fonsi_significance.py --sample 30 --model claude-sonnet-5
```
**Full FONSI via batch — ONE password** (`--batch-run` = submit → poll → fetch → build in one
process; the keychain is read once and cached; 50% price; auto-chunked under the API's
100k-request / 256 MB caps):
```bash
conda run -n nepa python phase2/code/deliverable02/02_extract_fonsi_significance.py --batch-run --model claude-sonnet-5
```
(Prefer not to leave the terminal open? Split form: `--batch-submit` now, `--batch-fetch
[--wait]` later — one password each.)
This upgrades rows to `extraction_method='regex+llm'` and stamps `significance_llm_run_at`.
NOTE: `temperature=0` is only sent on Haiku (Sonnet 5 / Opus 4.8 reject sampling params — 400).
The prompt in `extract_common._prompt_for` is a v1 — expect to tune it after the spike.

### 4. After the FONSI pass — validate + look before paying for EIS
```bash
conda run -n nepa python phase2/code/deliverable02/05_validate_significance.py  # Gate 3 vs gold
Rscript phase2/code/deliverable02/06_create_figures.R                     # FONSI-only tables
quarto render phase2/reports/deliverable02.qmd
```
Decide from these whether the EIS pass is worth it.

### 5. EIS (gated) — only after FONSI Gate 3 passes and you like the FONSI analysis
`04` never touches FONSI outputs (`_eis` suffix). Retrieval spike first, then batch:
```bash
conda run -n nepa python phase2/code/deliverable02/04_extract_eis_significance.py --dry-run --sample 800  # retrieval check, free
conda run -n nepa python phase2/code/deliverable02/04_extract_eis_significance.py --batch-run --sample 0 --model claude-sonnet-5   # one password
Rscript phase2/code/deliverable02/06_create_figures.R --with-eis          # combined analysis
```

## Notes
- All scripts `py_compile` clean. The IDE's `anthropic` "missing" warning is a false positive
  (base-env lint; the import is lazy and only in the real-mode branch).
- Dry-run report tables are **illustrative** (regex-only, all `needs_human_review=TRUE`) — the
  report banner says so until the LLM pass + gold validation land.
