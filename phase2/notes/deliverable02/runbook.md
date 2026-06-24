# D2 runbook — significance determinations

Plan: `phase2/plans/deliverable02.md` (v2.5). Code: `phase2/code/deliverable02/`.

## What is built and tested (runnable now)

| File | Status | Output |
|---|---|---|
| `common.py` | ✅ | paths / IO / helpers |
| `significance_taxonomy.py` | ✅ | resource crosswalk, determination/threshold/factor vocab, cue dicts |
| `00_resolve_framework_regime.py` | ✅ tested | `data/analysis/deliverable02/project_regime.parquet` (1,326 projects) |
| `01_build_d2_inventory.py` | ✅ tested | `…/significance_corpus.parquet` (1,205) + `output/deliverable02/corpus_membership_review.csv` |
| `_run.py` | ✅ | runs 00 → 01 |

Run the whole deterministic stage:

```bash
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable02/_run.py
```

### What it produces (verified Jun 18)
- **Regime:** 484 assigned_high, 222 assigned_proxy, 69 boundary_review, 551 missing_date.
  Decision periods skew pre-2020 (613); post-2023 cells thin (2024-rule 27, 2025-removal 11) — as the plan predicted.
- **Corpus tiers:** mitigated_fonsi **58** (36 in the primary 2009-present window — 18 DOE-family, 17 BLM, 1 other), straight_fonsi 394, eis_significant 753. (FONSI total 452 ✓.)
- **Gate 1/2 list:** `corpus_membership_review.csv` = the mitigated FONSIs + EIS projects to eyeball.

### Calibration finding (important)
BLM/DOE FONSIs **do not** use the CEQA-style phrase "less than significant with mitigation" (0 hits).
They use **"would be significant [absent mitigation]"** and **"with incorporation of … mitigation"**.
The Gate-1 screen (`MITIGATED_SCREEN_CUES`) is recall-oriented; the human prunes. The precise extractor
cue (`DETERMINATION_CUES['explicit_mitigated_lts']`) is separate.

## Monday plan

1. **Run the deterministic stage** (above). ~5 s.
2. **Gate 1/2 — eyeball `corpus_membership_review.csv`.** Confirm the 36 primary mitigated FONSIs look right;
   spot-check the `off_mission_flag` rows (NNSA/Defense) for exclusion. This is the centerpiece list.
3. **POC spike (next build, needs your go-ahead on LLM budget).** Hand-code ~30 mitigated FONSIs against the
   schema, run the extractor on the same 30 (`02 --sample 30`), compare. Freeze the schema. Est. cost ~$1.
4. **Gold-set build (yours to label).** ~300 determinations + ~100 negatives via the `03` worksheet.

## Remaining stages (gated — built after the spike)

| File | Blocked on | Why |
|---|---|---|
| `02_extract_significance.py` (LLM) | POC spike + LLM-budget go-ahead | Build/run the extractor against a **frozen** schema (plan: freeze after spike). Haiku 4.5 @ $1/$5 per 1M tok; full run ~$20–35, spike ~$1. Reads **EA body + FONSI** / **FEIS + optional ROD** (v2.5). |
| `03_build_gold_set_queue.py` | — (buildable now) | Emits the labeling worksheet; gold itself is hand-coded by the analyst. |
| `04_extract_eis_significance.py` | FONSI validation passing (Gate 3) | EIS gated behind FONSI per plan; 6.1M-page substrate. |
| `05_validate_significance.py` | the gold set existing | Targets are meaningless without gold labels. |
| `06_analyze_significance.R` + report | extracted table + gold | Includes the Phase-6 association layer (logistic regression; agency's own determination = label). |

## Reproducibility notes
- Every output carries `*_run_at` + `schema_version` (`d2_v2_5`).
- Paths resolve from repo root; reads are read-only over D6 / shared artifacts; writes only to the D2 write set.
- Cut dates verified (incl. 2026-01-08 CEQ removal, 91 FR 618) — re-check the Federal Register before building `00` for production if the regulatory picture moves.
