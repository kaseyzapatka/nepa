# Deliverable 6 outputs — what to look at

**Client-facing deliverable (the only thing a client sees):**
→ the rendered report `docs/phase2/reports/deliverable06.html`
  (source: `phase2/reports/deliverable06.qmd`; figures embedded from `figures/`).

**Your one QA table (is the machine right?):**
→ `d6_comparison_table.csv` — every candidate with its CE **verdict**
  (new / expand / adopt / contrast), # of CE-shaped FONSIs, the best-matching
  existing CE + agency, the adopt/expand detail, and a rank score.

**`figures/`** — the report figures (≈16 PNGs, all embedded in the report): the outcomes
waffle, sort/keep/fingerprint, CE-match, sizes, classification, adoption gap, states map,
timeline, mitigated overall/share, mitigation wordcloud, and the Analysis-3 CE-landscape
set (department waffle, agency, numeric-limit, bounds lollipop, t-SNE scatter, CE-split).

**`review/`** — drill-down / QA tables. Open these only when a number in the
comparison table or report looks off:
- `d6_new.csv` / `d6_expand.csv` / `d6_adopt.csv` — the three opportunity lists (full columns).
- `d6_candidate_evidence_<category>.csv` — per-candidate project-level evidence with citations.
- `candidate_*_review.csv` — per-pipeline-step QA (membership, extraction, mitigation, CE match, descriptive, storage scan).
- `ce_landscape_summary.csv` — existing-CE counts per agency.

**Not here:** `phase2/data/analysis/deliverable06/*.parquet` are machine-to-machine
handoffs between pipeline steps — you don't read these directly.

**Authoritative review CSVs:** `d6_comparison_table.csv` (verdicts) and the
`d6_adopt.csv` / `d6_candidate_evidence_*.csv` drill-downs are the current client-facing
QA artifacts; the `candidate_*_review.csv` files are per-step internal QA.

**This deliverable is LLM-backed, not deterministic.** Facts, numeric limits, mitigation,
and significance thresholds come from a one-pass enrichment of the FONSIs
(`03_enrich_llm.py`; model **claude-sonnet-4-6**, schema **d6_enrich_schema_v5**,
451/452 FONSIs enriched, ~97% quote verification). `_run.py` wires that enrichment in and
**aborts** if `fonsi_enrichment.parquet` is missing — so the enrichment must be run first.
CE-coverage matches are **candidate matches pending eCFR verification**, not confirmed
coverage.
