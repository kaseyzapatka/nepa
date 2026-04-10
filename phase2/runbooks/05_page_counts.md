# Regulatory Page Count Extraction

**Purpose:** Estimate FRA-compliant page counts for clean energy EA and EIS final documents.
**Input:** `data/analysis/projects_combined.parquet` + page-level text.
**Output:** `data/analysis/projects_page_counts.parquet`
**Cost:** Free (no LLM calls).
**Prerequisites:** Base dataset built ([runbook 01](01_base_dataset.md)).

---

## Full production run

```bash
python code/extract/extract_pages.py --run
```

Re-run whenever `projects_combined.parquet` is updated with new projects.

---

## Notes

- The FRA defines a "page" as 500 words, excluding maps, figures, and appendices (40 C.F.R. § 1508.1(bb)).
- Script computes `regulatory_pages = CEIL(body_word_count / 500)` by detecting embedded appendix sections and excluding low-content pages.
- Output is joined into the Deliverable 5 R pipeline by `code/deliverable05/00_setup.R`.
