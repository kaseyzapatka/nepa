# D4 Timeline — CE Coverage-Regression Fix Plan

**Audience:** the agent patching the D4 pipeline. This document is self-contained — it gives
you the confirmed root cause, the exact file:line to change, the recommended patch, recovery
estimates, and a validation recipe. You should not need to re-derive the diagnosis.

**Scope:** the 1,360 **CE** projects that Phase 1 produced timeline dates for but Phase 2 dropped.
(The 16 EA cases are being handled separately — do not touch EA/EIS behavior unless a change is
explicitly marked safe for them.)

---

## 1. What's wrong (one sentence)

The CE determination/signature date sits at the **bottom of a dense ~7,000–9,000-char form page**,
but the `tier_d` retrieval path **truncates each page's `context_text` to 2,000 chars** before the
candidate extractor (`03_`) ever sees it — so the date is cut off and no candidate is generated.
A secondary, smaller loss comes from whole-block exclusion-keyword rejection on the same form pages.

## 2. Evidence (already established — for your confidence, not to re-run)

The 1,376 missing projects (1,360 CE + 16 EA) are all present in
`timeline_document_index.parquet` but have **zero rows** in `timeline_candidates.parquet`. The
failure is at candidate generation (scripts `02_retrieve.py` → `03_extract_candidates.py`), not
adjudication.

Project-level root-cause table (all 1,376; CE/EA split):

| Bucket | Cause | n | CE | EA | Recoverable? |
|---|---|---|---|---|---|
| **B** | **Truncation** — date present in full page text but **beyond char 2,000** | **1,034** | **1,029** | 5 | **Yes — Fix 1** |
| **C** | All dates **rejected** — real signature date shares a block with an exclusion keyword | 261 | 261 | 0 | Mostly — Fix 2 (~251) |
| **A** | **No packets** — all of the project's docs are `defer` priority (never scanned) | 46 | 42 | 4 | Partly — Fix 3 |
| **D** | **No parseable date** anywhere in the scanned page(s) | 35 | 28 | 7 | No (genuinely absent) |

Agency split of the CE buckets is **DOE-dominated**: bucket B is DOE 1,015 / BLM-DOI 13 / USDA 4;
bucket C is DOE 239 / BLM-DOI 22. These are overwhelmingly **DOE CX-determination forms**.

Supporting facts:
- Of 1,321 CE missing-cohort packet pages, **1,293 (98%) contain a parseable date in the FULL page
  text**, and for **1,178 (89%) the first date is beyond char 2,000** (truncated away). Median full
  page length = **7,107 chars**; only **2,000** are stored in the packet.
- **1,028 / 1,029** bucket-B CE projects recover through `tier_d` packets
  (`retrieval_reason = page_keyword_score`), avg full text ≈ **8,979 chars**, avg stored = **2,000**.
- **1,030** of the bucket-B projects have **only `priority_3`** documents → `tier_b` (which already
  reads CE pages at full length) never scans them; they hit `tier_d` exclusively.
- **Recovery simulation**: re-running the *real* `03_` extractor (`extract_candidates_from_packet`)
  on the **full** page text yields a candidate for **1,029** CE projects that currently get none.
- **Phase 1 cross-check** — Phase 1's `bert_decision_date_source` for this cohort is literally the
  CE signature block, e.g.
  `"NEPA Compliance Officer Signature: … Date: 3/31/10 FIELD OFFICE MANAGER DETERMINATION"`,
  `"Approved by Jason Anderson, DOE-ID NEPA Compliance Officer, on 09/17/2021."`,
  `"Digitally signed by PETER SIEBACH Date: 2022.09.22"`. Phase 1 read the whole page/document and
  caught the bottom-of-form signature date; Phase 2 truncates it off. Phase 1 had **1,209 decision +
  254 initiation** dates for this CE cohort.

## 3. Why it happens (code path)

`phase2/code/deliverable04/02_retrieve.py`:

- `build_tier_b_packets` (≈ line 548) only scans `priority_1` / `priority_2` documents (line ≈558:
  `priority_docs = doc_rows[doc_rows["scan_priority"].isin(["priority_1", "priority_2"])]`). For CE
  *small* docs it already stores **full** page text (`_truncate(text, 30_000 …)`, line ≈630). **But
  the missing CE docs are `priority_3`,** so this path never runs for them.
- The missing docs are `priority_3` because in `01_index.py`, `DECISION_DOC_SCORES` does not score a
  bare `document_type_clean == "CE"` (it scores `"ce determination"`, `"categorical exclusion
  determination"`, etc.). So `decision_doc_score = 0`, and with the main-doc bonus (+1.5) the
  `scan_priority_score` is ~1.5 → `priority_3`. (Confirmed: 1,212 missing CE docs have
  `document_type_clean = "CE"`, `decision_doc_score = 0.0`, `scan_priority = priority_3`.)
- `priority_3` docs are scanned **only** by `build_tier_d_packets` (≈ line 798:
  `priority_docs = …isin(["priority_1","priority_2","priority_3"])`), and that function stores
  **`"context_text": _truncate(text, 2000)`** at **line ≈863**. ← *This is the cut.*

`phase2/code/deliverable04/03_extract_candidates.py`:

- `extract_candidates_from_packet` only sees `packet["context_text"]` (the 2,000-char slice). The
  date beyond char 2,000 is never in the string it scans.
- `_should_reject_date` (lines ≈525–566) scans the **entire block** for `EXCLUSION_KEYWORDS`. On CE
  forms the real signature date often sits in the same block as `"expiration date"`,
  `"categorical exclusion expires"`, `"expires on"`, statute citations (`"recovery act"`,
  `"policy act"`, `"… management act"`, `"act of 20…"`), `"https://"`, or `"eds."`/`"pp."` — so the
  whole block is rejected, killing the signature date too (bucket C). Call site: line ≈821,
  `reject, _ = _should_reject_date(parsed, block, process_type, source_tier)`.

---

## 4. The fix

Implement **Fix 1** (recovers ~1,029, the bulk). Then **Fix 2** (recovers ~251 more). Fix 3 is
optional cleanup for ~30–40 more.

### Fix 1 — Stop truncating CE `tier_d` page packets (PRIMARY, ~1,029 CE)

**File:** `phase2/code/deliverable04/02_retrieve.py`, `build_tier_d_packets`, the
`"context_text": _truncate(text, 2000)` line (≈863).

**Change:** make the cap process-aware so **CE reads the full page**, while EA/EIS are left
unchanged (do not alter their behavior):

```python
# near the other module constants at top of file
TIER_D_CONTEXT_CHARS = {"CE": 30_000, "EA": 2_000, "EIS": 2_000}
```

```python
# in build_tier_d_packets, replace the context_text line:
"context_text": _truncate(text, TIER_D_CONTEXT_CHARS.get(process_type, 2000)),
```

(30,000 matches the existing CE full-read cap used in `build_tier_b_packets`; CE form pages are
~7k chars so this is effectively "whole page". Keeping EA/EIS at 2,000 means their candidate sets
are byte-identical to today.)

**Why this is safe for the classifier/selection:** `03_`'s `model_context` window
(`_build_model_context`, `MODEL_CONTEXT_CHARS = {"CE": 900, …}`) is still date-centered and capped,
so the SetFit classifier (`04_`) sees the same bounded window per candidate — only the *number* of
candidates per CE page rises. The per-project packet cap (`PACKET_CAPS["CE"] = 25`) is unaffected
(it caps packet count, not candidate count).

**Alternative / complementary (more "correct", broader blast radius — your call):** instead of (or
in addition to) the cap change, promote these docs into the `tier_b` CE full-read path by scoring
`document_type_category == "decision"` in `01_index.py`. In `_compute_scores` (≈ line 198), add:
```python
if str(row.get("document_type_category", "")).strip().lower() == "decision":
    decision_doc_score = max(decision_doc_score, 4.5)   # floor → priority_2 (or priority_1 with main-doc bonus)
```
This routes them through `build_tier_b_packets` (which already reads CE pages at 30k) and also makes
them eligible for `tier_c` sections. It changes `scan_priority` distribution for **all** CE decision
docs, so re-validate scan volume. If you only want the surgical, low-risk fix, do the `tier_d` cap
change above and skip this.

### Fix 2 — Scope exclusion-keyword rejection to a window around the date (SECONDARY, ~251 CE)

**File:** `phase2/code/deliverable04/03_extract_candidates.py`, `_should_reject_date`
(≈525–566) and its text-tier call site (≈821).

**Problem:** the keyword scan over the whole block kills the real signature date when an unrelated
expiration/citation/URL token appears in the same block. Measured: on 264 bucket-C CE pages, a date
**would survive for 251 (95%)** if `EXCLUSION_KEYWORDS` were checked only within **±60 chars** of the
date span instead of across the whole block. The 13 that still reject are genuine
expiration/citation-only pages (correctly dropped).

**Change:** pass the date's char span into `_should_reject_date` and check `EXCLUSION_KEYWORDS`
(and the citation `EXCLUSION_RE`) only inside a ±60-char window; keep `future_date`, the pre-1970
cutoffs, and `REJECT_CUES` as they are. Sketch:

```python
def _should_reject_date(parsed_date, context, process_type, source_tier,
                        date_span: tuple[int, int] | None = None):
    ctx_lower = context.lower()
    if parsed_date.date() > RUN_DATE:
        return True, "future_date"
    if process_type in ("CE", "EA") and parsed_date.year < 1970:
        return True, "pre_1970_hard_reject"
    if process_type == "EIS" and parsed_date.year < 1970:
        return True, "pre_1970_eis_reject"

    # Window the citation/keyword exclusions to the immediate neighborhood of the date so a
    # bottom-of-form signature date isn't killed by an unrelated 'expires on' / statute cite
    # elsewhere on the same dense CE form page.
    if date_span is not None:
        ds, de = date_span
        win = context[max(0, ds - 60):de + 60]
        win_lower = win.lower()
    else:
        win, win_lower = context, ctx_lower

    for kw in EXCLUSION_KEYWORDS:
        if kw in win_lower:
            return True, f"exclusion_keyword:{kw}"
    for pat in EXCLUSION_RE:
        if pat.search(win):
            return True, "exclusion_regex"
    if source_tier == "metadata":
        return False, ""
    if REJECT_CUES.search(win):
        return True, "reject_cue"
    return False, ""
```

Update the two call sites to pass the span. The text-tier site (≈821) has `_ms, _me` (match start/
end within `block`): `reject, _ = _should_reject_date(parsed, block, process_type, source_tier, (_ms, _me))`.
The metadata-tier site (≈740) can pass `None` (unchanged behavior).

**CAUTION — this touches EA/EIS too.** Window-scoping is a broad behavioral change: it could
re-admit citation/bibliographic dates in EA/EIS narrative. Two safe options:
  1. **Gate Fix 2 to CE only** (`win = … if process_type == "CE" else context`), or
  2. Apply the window only to the **expiration/citation** keyword subset and keep OMB/form-boilerplate
     keywords (`"omb control"`, `"previous editions obsolete"`, `"doe f "`, `"paperwork reduction"`)
     **global** (those genuinely invalidate the whole page).
  Prefer option 1 for this CE-focused task; validate against the regression set before widening.

### Fix 3 — Rescue `defer` CE docs (OPTIONAL, ~30–40 CE)

**File:** `01_index.py`. 35 of the 42 bucket-A CE projects have *only* `defer` docs. If you adopt the
Fix-1 **alternative** (`document_type_category == "decision"` floor), most of these get promoted out
of `defer` automatically and into a scanned tier. Otherwise, lower the CE defer threshold for
main documents (e.g. don't `defer` a CE `is_main_document` page). Smaller prize; do last.

---

## 5. Expected recovery

| Fix | CE projects recovered (est.) | Cumulative CE coverage of the 1,360 |
|---|---|---|
| Fix 1 (tier_d CE cap) | ~1,029 | ~76% |
| + Fix 2 (windowed exclusion, CE-gated) | ~251 | ~94% |
| + Fix 3 (defer rescue) | ~30–40 | ~97% |
| Residual unrecoverable (bucket D + genuine expiration/citation-only) | ~40 | — |

## 6. Validation recipe (do this after patching)

1. **Isolated re-run on the missing cohort** (fast, won't clobber canonical outputs). Write the
   missing CE ids to a file and use the `--sample-ids` flag (sample runs auto-isolate to
   `timeline/sample_runs/<stem>/`):
   ```bash
   python - <<'PY'
   import pandas as pd
   m = pd.read_csv("phase2/notes/deliverable04/missing.csv")
   m[m.process_type=="CE"].project_id.to_csv("phase2/notes/deliverable04/missing_ce_ids.txt", index=False, header=False)
   PY
   conda activate nepa
   cd phase2/code/deliverable04
   python 01_index.py   --sample-ids ../../notes/deliverable04/missing_ce_ids.txt
   python 02_retrieve.py --process CE --sample-ids ../../notes/deliverable04/missing_ce_ids.txt
   python 03_extract_candidates.py --process CE --sample-ids ../../notes/deliverable04/missing_ce_ids.txt
   ```
   (Confirm the `--sample-ids` plumbing in `03_`; if it lacks the flag, point it at the sample-run
   packets parquet.) **Success = the candidates parquet now has rows for the previously-missing ids.**
   Count distinct `project_id` with ≥1 candidate; target ≈1,280+.

2. **Spot-check the recovered dates against Phase 1.** Join recovered decision dates to
   `phase1/data/analysis/projects_timeline_bert.parquet` (`bert_decision_date_final`); for the
   signature-block cohort they should match within a few days. Eyeball 15–20 contexts to confirm the
   captured date is the NEPA Compliance Officer / Field Office Manager **signature** date, not the
   `"categorical exclusion expires on …"` expiration date.

3. **Regression guard.** Re-run the **full** CE pipeline only when the sample looks right
   (`02_retrieve.py --process CE --force` writes to `timeline/process_runs/CE/`; merge per your
   normal full-run process). Compare `timeline_project_dates.parquet` against the backup
   (`timeline_project_dates.pre_gt_inject_*.parquet`) to confirm the **existing** ~19,349 covered
   projects' dates don't shift materially — only new coverage is added.

4. **Bucket-C sanity.** After Fix 2, verify you did **not** start admitting citation/expiration dates
   for projects that were previously correct. Check that future-dated and `< 1970` dates are still
   rejected (those guards are intentionally global).

## 7. How to reproduce the diagnosis numbers (if you need to)

The bucketing logic: for each missing project's page-based packets, pull the **full** page text from
`phase2/data/processed/ce/pages.parquet` (DuckDB, filtered to the cohort's `document_id`s — never
`pd.read_parquet` the whole pages file), then run the real `extract_candidates_from_packet` from
`03_` on the full text. `recover` = yields ≥1 candidate; `rejected` = a date parses but all are
rejected; `nodate` = no parseable date; `no_packets` = absent from the packets parquet entirely.
Agency from `project_department` in the document index. (This is exactly how the table in §2 was
produced.)
