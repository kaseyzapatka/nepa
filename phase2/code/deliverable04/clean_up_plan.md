# Deliverable 04 — Code Cleanup Plan

Streamline the D4 codebase so the production pipeline is obvious and one-off tools don't
masquerade as load-bearing code. `[x]` = done; `[ ]` = do after Thursday's presentation.

---

## [x] 1. Write `README.md` — the pipeline map  *(done 2026-06-15)*

Single-page map of the pipeline spine in run order (00→08, including the `04b`/`05b`/`05c`
siblings), inputs/outputs, a Tools section, sub-tracks (`labeling/`, `_archived/`), and
conventions. Kills most of the "how does this relate" confusion. Zero behavioral risk.
**File:** `README.md`.

## [x] 3. Archive completed one-off scripts  *(done 2026-06-15)*

Moved finished one-offs out of the top level into `_archived/` (git history preserved):
- `_audit_ea_decision_recall.py` — June-9 EA recovery cycle, complete
- `review_sample20.R` — one-off review helper

Kept at top level: the spine (`00`–`08`), `_diagnostics.py` (reusable), `_run.py`
(orchestrator), and `_phase0_baseline.py` (**archive this too once the current EIS recovery
cycle closes** — it's this cycle's point-in-time baseline).

---

## [ ] 2. Make `_run.py` the single source of truth  *(after Thursday — touches orchestration)*

`_run.py` runs `00b → 01 → 02 → 03 → 04 → 05 → 06` and **omits `04b`, `05b`, `05c`**, so
calibration / ranking / gt-injection are manual steps easy to forget. Add them in the
canonical order (`02 → 03 → 04 → 04b --apply → 05b --apply → 05 → 05c → ...`). These stages
take `--apply` / `--run-dir` flags the others don't, so `run_stage` needs per-stage
invocation handling. **Test on a small `--sample-ids` run before trusting it.**

## [ ] 4. Resolve `06_adjudicate_llm.py`  *(after Thursday)*

It's wired into `_run.py` but is stale (top-3 candidates, raw probs — see notes). Either
**rebuild** it (per-event top-k, calibrated probs, retain authoritative candidates) before
relying on it, or **remove it from `_run.py`** until rebuilt so the orchestrator doesn't
point at stale code. Decide based on whether the LLM adjudication pass is being revived.

---

## Optional / housekeeping
- `_archived/` holds ~25 completed one-offs + `analysis/` + `build/`. Safe to leave (out of
  the way) or delete entirely (git retains history) — your call.
- `labeling/` is a separate gold-labeling sub-pipeline; document its own run order in a short
  `labeling/README.md` if it gets reused for the classifier retrain.
