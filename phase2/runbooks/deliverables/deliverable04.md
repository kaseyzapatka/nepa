# D4 — Project Timelines

**Purpose:** Extract an initiation date and a decision date for every NEPA review (CE, EA, EIS) and produce a project-level timeline database for duration analysis, coverage diagnostics, and FRA-period comparisons.
**Input:** `data/analysis/projects_combined.parquet`, `documents_combined.parquet`, processed pages/sections, and cached Tier A register parquets (BLM/DOE/Federal Register).
**Output:** `data/analysis/timeline/timeline_project_dates.parquet` (one row per project with dates), plus `timeline_candidates.parquet`, `timeline_context_packets.parquet`, `timeline_document_index.parquet`.
**Cost:** LLM adjudication (`06`, optional) ~$0.50–$1.00 (Claude Haiku) for EA+EIS. LLM gold-labeling (`labeling/05`) cost scales with split size.
**Conda env:** `nepa` — all Python via `/opt/anaconda3/envs/nepa/bin/python` (scripts hard-require `CONDA_DEFAULT_ENV=nepa`).

**Scripts** (in `phase2/code/deliverable04/`):
- `00_sample.py`, `00b_sections.py` — one-time setup (gold sample, document sections)
- `01_index.py` → `02_retrieve.py` → `03_extract_candidates.py` → `04_classify_candidates.py` → `05_select_dates.py` → `06_adjudicate_llm.py` (optional)
- `_run.py` — sharded full-corpus orchestrator (runs 02→03→04→05, optional 06)
- `07_validate.py`, `08_analyze.R`, `export_api_validation.py`, `build_review_packet.py` — validation / analysis / helpers
- `labeling/` — gold-set construction and labeling (see "Labeling & training" below)

> **Renumber note (2026-06-01):** the pipeline was flat-renumbered to insert the learned classifier at `04`. Selection moved `04`→`05`, adjudication kept `06`, validation moved `05`→`07`, the orchestrator `07_run_full_corpus_timelines.py`→`_run.py`, and `validation/`→`labeling/`. Older docs/commits may reference the old names.

---

## Pipeline stages

| Stage | Script | Input → Output |
|---|---|---|
| Index | `01_index.py` | projects + documents + Tier A registers → `timeline_document_index.parquet` |
| Retrieve | `02_retrieve.py` | index → `timeline_context_packets.parquet` (5-tier retrieval) |
| Extract | `03_extract_candidates.py` | packets → `timeline_candidates.parquet` (regex dates + role prelabel) |
| **Classify** | `04_classify_candidates.py` | candidates → same parquet + `p_initiation`/`p_decision`/`classifier_*` |
| Select | `05_select_dates.py` | candidates → `timeline_project_dates.parquet` |
| Adjudicate | `06_adjudicate_llm.py` | unresolved projects → updated dates (optional) |

**Candidate roles (script 03):** `clear_initiation`, `clear_decision`, `proxy_initiation`, `proxy_decision`, `review`, `historical`, `reject`, `body_text` (a date in a decision-type doc with no role cue — a holding category for the classifier; a last-resort decision proxy in `05` until a model exists), `unknown`.

**Tier A (authoritative, confidence 5.0, bypass the classifier):** BLM ePlanning, DOE ePlanning, DOE CX register, Federal Register NOI. These are exempt from the learned scorer.

---

## Full corpus run

```bash
conda activate nepa
# 1. Build/refresh the document index (needed if registers or corpus changed)
python phase2/code/deliverable04/01_index.py --process CE EA EIS
# 2. Run the sharded pipeline (02→03→04→05); resumes from manifest, add --with-api for 06
python phase2/code/deliverable04/_run.py --process CE EA EIS
```

`_run.py` shards by process type + SHA-1 hash bucket (default 5 shards/process), maintains `timeline_run_manifest.parquet`, and skips completed shards unless `--force`. The `04` classification stage passes through with neutral scores until a model is trained — so the full pipeline runs end-to-end today and gains accuracy once the classifier is trained.

**Stage-only reruns** (faster than `_run.py` when only patterns/logic changed):

```bash
# regex/role patterns changed (script 03):
python phase2/code/deliverable04/03_extract_candidates.py --process CE EA EIS
python phase2/code/deliverable04/04_classify_candidates.py --process CE EA EIS   # pass-through if no model
python phase2/code/deliverable04/05_select_dates.py --process CE EA EIS

# selection/scoring changed (script 05) only:
python phase2/code/deliverable04/05_select_dates.py --process CE EA EIS
```

### Sample / smoke test

`--sample-ids <file>` isolates a run into `timeline/sample_runs/<stem>/` (scripts 02–05 all honor it):

```bash
python phase2/code/deliverable04/02_retrieve.py --process CE --sample-ids ids.txt
python phase2/code/deliverable04/03_extract_candidates.py --process CE --sample-ids ids.txt
python phase2/code/deliverable04/05_select_dates.py --process CE --sample-ids ids.txt
```

---

## Classifier (`04_classify_candidates.py`)

Two independent binary heads per candidate — `p_initiation` and `p_decision` — over the **ambiguous middle band only**: `role_confidence_score < 5.0` and role in {clear/proxy init/decision, `body_text`, `unknown`}. Exempt: 5.0 (Tier A / strong cue) and `review`/`historical`/`reject`. One shared-encoder model with a `[CE]/[EA]/[EIS]` process token and a multi-label head. Backend-pluggable: **SetFit today, fine-tuned DeBERTa-v3 later** (see the SetFit→BERT criteria in the script docstring).

```bash
pip install setfit datasets          # one-time, in the nepa env
python phase2/code/deliverable04/04_classify_candidates.py --train     # needs gold labels
python phase2/code/deliverable04/04_classify_candidates.py --eval      # held-out test split
python phase2/code/deliverable04/04_classify_candidates.py             # score (default mode)
```

**Open integration step:** once a model is trained, wire `classifier_signal` in `05_select_dates.py` (`_compute_candidate_score`, currently hard-zero) to consume `p_initiation`/`p_decision`.

---

## Labeling & training (`labeling/`)

The classifier trains only on gold labels. **There is no human gold yet** — the `gold/codex_labels/` files are regex echoes; do not train on them.

```bash
# 1. (already built) splits + review packets: labeling/01_build_gold_samples.py, 02_prepare_gold_review_packets.py
# 2. LLM-label a split (dry-run first to preview prompts / cost):
python phase2/code/deliverable04/labeling/05_llm_label_candidates.py --split diagnostic_balanced_v2 --dry-run --limit 3
python phase2/code/deliverable04/labeling/05_llm_label_candidates.py --split diagnostic_balanced_v2
# 3. Import into the gold/training tables:
python phase2/code/deliverable04/labeling/03_import_gold_labels.py \
    --projects   <split>_projects_llm_labeled.csv \
    --candidates <split>_candidates_llm_labeled.csv
# 4. Train + eval the classifier (above)
```

`03_import` writes `timeline/gold/timeline_gold_candidate_training.parquet` (the classifier's training input; test split excluded) and `timeline_gold_candidates.parquet` (carries `split`, used by `--eval`).

⚠️ `labeling/04_codex_prelabel_gold_packets.py` is a **mechanical regex echo**, not an LLM pass. Use `05_llm_label_candidates.py` for real labels.

---

## Overnight runs

Use the `timeline-runner` agent (LAUNCH/REPORT modes) — it sets up logging, a watchdog, and a morning coverage report. See `.claude/agents/timeline_runner.md`.

---

## Validation & analysis

```bash
python phase2/code/deliverable04/07_validate.py --prepare-review          # build review packet
python phase2/code/deliverable04/07_validate.py --validate --reviewed-packet <filled.csv>
Rscript phase2/code/deliverable04/08_analyze.R                            # duration tables + figures
```

---

## `timeline_project_dates.parquet` — key columns

| Column | Notes |
|---|---|
| `project_id`, `process_type` | keys |
| `initiation_date` / `decision_date` | ISO-8601 or null |
| `*_granularity` | `day` / `month` / `year` / `unknown` |
| `*_source_type`, `*_confidence`, `*_is_proxy`, `*_evidence_text` | provenance |
| `duration_days` | decision − initiation (day-granularity both) |
| `timeline_status` | `complete_clear`, `complete_with_proxy`, `missing_initiation`, `missing_decision`, `missing_both`, `manual_review`, `invalid_order` |
| `timeline_flags` | pipe-delimited diagnostics (`year_proxy_decision`, `nepa_case_year_proxy_discarded`, `same_day`, …) |

---

## Notes

- **FRA cut date: August 16, 2023** (CEQ final rule effective date) — splits pre/post regulatory-period analysis.
- **CEs have one date** (determination = initiation). CE coverage leans on registers (DOE CX, BLM) because CE forms put dates in structured fields, not prose.
- **`body_text`** was the old `doc_type_decision` catch-all (~86k CE candidates). It is intentionally NOT `clear_decision` — it has no role evidence. The classifier is the mechanism that promotes/demotes it.
- **July-1 guard:** `YYYY-07-01` dates are frequently NEPA case-number-year proxies; `05` applies a −2.0 penalty and discards the proxy when it would invert ordering with a same-year initiation.
- **Always activate `nepa`** — every script exits immediately if `CONDA_DEFAULT_ENV != nepa`.
