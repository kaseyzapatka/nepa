# Geography and Multi-Agency Extraction

**Purpose:** Identify co-agency (multi-agency) review projects and build the Deliverable 4 geography outputs.
**Input:** `data/analysis/projects_combined.parquet` + page-level text cues.
**Output:** `data/analysis/coagency_projects.parquet`; rendered `reports/deliverable04.qmd`
**Cost:** Free (no LLM calls).
**Prerequisites:** Base dataset built ([runbook 01](01_base_dataset.md)).

---

## Full run

Run all three commands in sequence:

```bash
# 1. Build co-agency classifications from page-text cues
python code/extract/extract_coagency.py --run

# 2. Rebuild Deliverable 4 tables and figures (strict vs expanded multi-agency outputs)
Rscript code/deliverable04/01_geography.R

# 3. Render the updated Deliverable 4 report
quarto render reports/deliverable04.qmd
```

---

## Notes

- `extract_coagency.py` uses page-level text cues to identify projects with multiple lead or cooperating agencies.
- Step 2 produces both strict and expanded multi-agency counts used in the deliverable.
- Re-run all three steps together whenever `projects_combined.parquet` is updated.
