# Generation Capacity Extraction

**Purpose:** Extract generation capacity (MW/GW/kW) from project documents in two phases: regex over all projects, then LLM adjudication for ambiguous multi-candidate cases.
**Input:** `data/analysis/projects_combined.parquet` + page-level text.
**Output:** `data/analysis/projects_gencap.parquet` (updated in place after phase 2); `data/analysis/gencap_{ce,ea,eis}_llm.parquet` (per-source raw LLM outputs).
**Cost:** LLM adjudication phase is low cost (only runs on ambiguous multi-candidate projects).
**Prerequisites:** Base dataset built ([runbook 01](01_base_dataset.md)).

---

## Phase 1 — Regex (parallel, no API cost)

```bash
python code/extract/extract_gencap.py --run regex --parallel 3
```

Output: `data/analysis/projects_gencap.parquet`

## Phase 2 — LLM adjudication

Runs only on projects with 2+ distinct regex candidates. Updates `projects_gencap.parquet` in place.

```bash
python code/extract/extract_gencap.py --run llm --workers 4
```

> Use `--workers 2` if hitting rate limits.

## Testing / debugging

```bash
# Test on 10 projects first
python code/extract/extract_gencap.py --run llm --sample 10 --workers 1

# Debug a single project
python code/extract/extract_gencap.py --run llm --project-id <UUID>
```

---

## Notes

- Run phase 1 before phase 2; phase 2 reads the regex output.
- Per-source raw LLM outputs (`gencap_{ce,ea,eis}_llm.parquet`) are written alongside the merged file for traceability.
