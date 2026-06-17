# D4 Timeline — EA Coverage-Regression Fix Plan

**Audience:** the agent patching the D4 pipeline for the EA side. Self-contained — gives you the
finding, the exact one-line change, the per-project verdict, a mandatory validation/decision rule,
and what to explicitly NOT do. Companion docs: the CE fix is in
[`missing_investigation_CEplan.md`](missing_investigation_CEplan.md); full evidence is in
[`deliverable04/missing_investigation_findings.md`](deliverable04/missing_investigation_findings.md).

**Read the CE plan first.** This EA change is a *one-number extension* of CE Fix 1. If the CE patch
has already been applied, you are editing one value it introduced.

> **Status as of 2026-06-10 (`acdd7ba`):**
> - **EA Fix 1 ✅ IMPLEMENTED** — `TIER_D_CONTEXT_CHARS` dict in `02_retrieve.py` has `"EA": 8_000`.
>   The CE patch (Fix 1) was applied simultaneously; EA value is already 8,000.
> - **Validation NOT yet run** — Step A (isolated cohort check on `missing_ea_ids.txt`) and Step B
>   (full-EA regression diff) in §4 are still pending. Run Step A first; Step B gates whether to keep
>   the change or revert EA to 2,000.

---

## 1. The situation (don't re-derive)

16 EA clean-energy projects that Phase 1 had are absent from Phase 2's
`timeline_project_dates.parquet`. They are listed in `phase2/notes/deliverable04/missing.csv`
(`process_type == "EA"`; full UUIDs below in §3). All 16 are in `timeline_document_index.parquet`
but have **zero rows** in `timeline_candidates.parquet` — the failure is at candidate generation
(`02_retrieve.py` → `03_extract_candidates.py`), not adjudication.

Unlike CE (a clean truncation regression worth ~1,280 projects), **EA is small and mixed, and is
only partly a real regression.** Three sub-causes:

1. **Truncation (recoverable, ~4–6 projects).** The EA *narrative* document is `priority_3`, so it
   is scanned only by `build_tier_d_packets`, which truncates `context_text` to **2,000 chars**
   (`02_retrieve.py` ≈ line 863). The date sits below char 2,000 and is cut off. **This is the only
   recoverable bucket, and the fix below targets it.**
2. **Image-only decision PDFs (unrecoverable, 6 projects).** The FONSI/ROD exists but its
   `page_text` is empty (un-OCR'd scan, `AVG(LENGTH(page_text)) = 0`). The decision date is locked
   in an image; no text path can reach it. Needs OCR — **out of scope, do not attempt here.**
3. **Genuinely date-less (unrecoverable, 3 projects):** `7b598…`, `c5310…` (draft-EA only),
   `db787…` — yield zero candidates even on full page text. Phase 1 also had no real date.

**Important:** Phase 1's EA "dates" were frequently *wrong* — a railroad timetable, a revision-date
stamp, a regulatory-citation date, garbled OCR (see §3). For several of these, Phase 2's *absence*
is more correct than Phase 1's value. **Do NOT reinstate Phase-1-style proxy/fallback dates to
"close the gap."** The goal is only to recover dates that genuinely exist in extractable text.

## 2. The fix (one number)

**File:** `phase2/code/deliverable04/02_retrieve.py`, function `build_tier_d_packets`, the
`context_text` line (≈ line 863).

The CE patch introduces a per-process truncation cap:

```python
# module-level constant (added by the CE patch)
TIER_D_CONTEXT_CHARS = {"CE": 30_000, "EA": 2_000, "EIS": 2_000}
```
```python
# inside build_tier_d_packets
"context_text": _truncate(text, TIER_D_CONTEXT_CHARS.get(process_type, 2000)),
```

**Your change: bump EA from 2,000 to 8,000.**
```python
TIER_D_CONTEXT_CHARS = {"CE": 30_000, "EA": 8_000, "EIS": 2_000}
```

That is the entire code change. (8,000 — not 30,000 — because EA narrative pages are shorter than CE
forms and EA already has a dedicated full-read path for true decision docs
[`build_ea_decision_full_read_packets`, 8,000-char cap]; matching that keeps EA `tier_d` consistent
without flooding the candidate table.)

**If the CE patch has NOT been applied yet:** add the `TIER_D_CONTEXT_CHARS` dict (with
`"EA": 8_000`) and the `.get(...)` line yourself, exactly as above. Leave `"EIS": 2_000`.

**Do not** change `01_index.py`, `03_extract_candidates.py`, the EA full-read path, or any EA scoring.
EIS stays at 2,000 (untouched).

## 3. Per-project verdict (what to expect after the fix)

| project_id (full UUID) | Cause | After Fix | Phase-1 date was… |
|---|---|---|---|
| 7eb0b76647829f279093e10c12439239 | Truncation | **Recovers** (clear_decision in narrative) | wrong (4(d)-rule citation) |
| cbeef882ee3ed575871c812d45e33306 | Truncation | **Recovers** (clear_decision) | proxy (USFWS letter) |
| cea21587f7a7441da41bcddd30a7d494 | Truncation | **Recovers** (clear_decision) | garbled FONSI OCR |
| d649c45e7b994c85bf63185fd445abeb | Truncation | **Recovers** (proxy/init) | init-only |
| 0d717f5248bf4e6ae621cf064239d1b2 | Image-only FONSI + truncated narrative | Partial (narrative proxy only) | weak (programmatic ROD ref) |
| 18fa1b28127615d8d2ca3468099e986f | Image-only FONSI + truncated narrative | Partial (narrative proxy only) | wrong (revision-date stamp) |
| 0937f67244dd987d646b6998b0458635 | Image-only FONSI | No real decision date (image) | wrong (LIRR timetable) |
| 4a12ca0c89143be8dcc58924e6251fe0 | Image-only ROD (no packets) | Unrecoverable | init-only |
| bed860e4046bb3d9773737d14f3cd071 | Image-only FONSI | Unrecoverable | none |
| c7ae21cea4a60637bb52f732ce98d7fc | Image-only FONSI (no packets) | Unrecoverable | init-only |
| 25b36ebc2f912bdf7cbe86a40f930eec | No packets (all `defer`) | Unrecoverable here | none |
| 4b5f3363c76f853b020897a435202ce9 | FONSI has text but no clear decision | Maybe weak proxy | none |
| a01e3a1239f2b0b4e47b6c5cad8d2766 | FONSI has text but no clear decision | Maybe weak proxy | init-only |
| 7b598a73119e9f12e37dff1409872637 | Date-less | Unrecoverable | none |
| c5310bcce988ec1c0164ec3d98aa611d | Date-less (draft EA only) | Unrecoverable | none |
| db787aab7c191ed3a9b88fda16c61910 | FONSI has text, no parseable decision date | Unrecoverable | none |

**Realistic target: ~4 clean recoveries + ~2 partial (proxy) recoveries.** Anything beyond that is
noise. If you recover far more than ~6 EA, something over-fired — inspect before keeping.

## 4. Validation + DECISION RULE (this is the real work — do not skip)

The risk is not the 16 targets; it is **collateral change to the rest of the EA corpus**, because
raising the cap adds candidates to *every* EA project. You must prove the change is net-positive.

**Step A — isolated cohort check (fast).**
```bash
conda activate nepa
cd phase2/code/deliverable04
python - <<'PY'
import pandas as pd
m = pd.read_csv("../../notes/deliverable04/missing.csv")
m[m.process_type=="EA"].project_id.to_csv("../../notes/deliverable04/missing_ea_ids.txt", index=False, header=False)
PY
python 02_retrieve.py --process EA --sample-ids ../../notes/deliverable04/missing_ea_ids.txt
python 03_extract_candidates.py --process EA --sample-ids ../../notes/deliverable04/missing_ea_ids.txt
```
Confirm the sample-run candidates parquet now has rows for the previously-missing EA ids (expect
~10–13 of 16 to get ≥1 candidate; ~4–6 a decision-role candidate). Spot-check 5 contexts to confirm
the captured date is a real signature/decision date, **not** a revision-date stamp or citation.

**Step B — full-EA regression diff (the gate).** Re-run the full EA pipeline and compare the new
`timeline_project_dates.parquet` against the current backup
(`timeline_project_dates.pre_gt_inject_*.parquet`), restricted to `process_type == "EA"`:
- New EA projects gaining a date: should be ≈ the recoveries in §3 (~4–6).
- **Existing EA projects whose `decision_date`/`initiation_date` CHANGED:** this is the danger
  metric. Count them and inspect a sample.

**DECISION RULE:**
- ✅ **Keep `"EA": 8_000`** if Step B shows the ~4–6 recoveries land cleanly AND existing-EA date
  changes are ~zero (or every change is a manifest improvement on inspection).
- ❌ **Revert EA to `2_000`** if existing-EA dates shift materially. 4–6 projects is **not** worth
  destabilizing the EA set (EA coverage is already ~74% and was hard-won). Reverting is a one-line
  change back; leave CE/EIS as they are.

State the diff counts explicitly in your summary so the decision is auditable.

## 5. Explicitly out of scope
- OCR of the 6 image-only FONSIs/RODs (separate effort).
- Any change to `01_index.py`, `03_` scoring/exclusion logic, or the EA full-read path.
- Reinstating Phase-1 proxy/fallback dates. Leave genuinely unrecoverable projects uncovered.
- EIS (untouched).

## 6. How the verdict was derived (if you need to reproduce)
For each EA missing project, pull full page text from `phase2/data/processed/ea/pages.parquet`
(DuckDB, filtered to the project's `document_id`s — never `pd.read_parquet` the whole file) and run
`extract_candidates_from_packet` from `03_` on the full text. "Image-only" = decision doc
(`decision_doc_score >= 4.5`) with `AVG(LENGTH(page_text)) = 0`. Phase-1 dates/sources from
`phase1/data/analysis/projects_timeline_bert_ea_llm.parquet`
(`bert_decision_date_source`, `llm_decision_date`).
