# Review Type Extraction

**Purpose:** Classify clean energy EA/EIS projects as `programmatic`, `tiered`, or `standard`.
**Input:** `data/analysis/projects_combined.parquet` + page-level text.
**Output:** `data/analysis/projects_reviews.parquet`
**Cost:** LLM fallback option adds API cost; production run (regex only) is free.
**Prerequisites:** Base dataset built ([runbook 01](01_base_dataset.md)).

---

## Full production run (recommended)

Default scope: clean energy + EA/EIS only. Uses DuckDB for page loading; `--workers` parallelizes project classification.

```bash
python code/extract/extract_reviews.py --run --workers 8
```

## Optional: with LLM fallback (slower, higher recall)

Use when you want higher recall on borderline/ambiguous phrasing that regex scores as medium confidence, or for a focused QA pass on edge cases.

```bash
python code/extract/extract_reviews.py --run --use-llm --workers 8
```

## Test run (safe — does not overwrite main output)

Writes to `data/analysis/projects_reviews_test.parquet`.

```bash
python code/extract/extract_reviews.py --test --workers 4
```

---

## Notes

- `generic` / `tier 1` stand-in terminology is included by default.
- This output is used by the targeted timeline re-adjudication step (see [runbook 02](02_timeline.md)). Run this before that step.
- Do not use `--use-llm` for routine production refreshes — reserve it for QA passes.
