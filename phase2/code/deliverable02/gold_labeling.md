<!-- ⚠ GRAIN MIGRATION PENDING (2026-07-08). The extractor was redesigned to emit a *list* of
determinations per window (one per resource area), not one. This labeling prompt + worksheet
(`03_build_gold_set_queue.py`), the merge (`gold_agreement.py`), and validation
(`05_validate_significance.py`) still assume one determination per window and must move to
per-(window × resource_area) grain before Gate 3. **Do not start labeling until this banner is
removed.** No labels have been produced yet, so nothing is lost. -->

# D2 Gold-Set Labeling Prompt (dual-labeler: Claude + Codex)

You are an expert NEPA analyst labeling an **answer key** ("gold set") for Deliverable 2:
*Determinations of significance across resource areas*. Your labels will be used to grade an
LLM extraction pipeline (Gate 3), so **accuracy and consistency matter more than speed**. Two
labelers (Claude and Codex) perform this task **independently**; disagreements are adjudicated
by the human analyst. Do not consult the other labeler's output.

## Task

Label **all 400 rows** of the labeling queue. Each row contains a passage (`evidence_text`)
extracted from a NEPA decision document (an EA or its FONSI) for a clean-energy project. For
each passage, decide whether it contains a **significance determination** and, if so, code it.

- **Input:** `phase2/data/analysis/deliverable02/gold/significance_gold_queue.parquet`
  (same content as `phase2/output/deliverable02/significance_gold_queue.csv`)
- **Output:** ONE CSV, depending on who you are:
  - Claude → `phase2/data/analysis/deliverable02/gold/labels_claude.csv`
  - Codex → `phase2/data/analysis/deliverable02/gold/labels_codex.csv`
- **Output columns (exactly these, one row per input row):**
  `evidence_span_id, gold_is_determination, gold_determination_class,
  gold_determination_scope, gold_resource_area, gold_primary_threshold_type,
  gold_primary_threshold_status, gold_mitigation_link, gold_evidence_span_ok,
  gold_needs_human_review, gold_notes, labeler, labeler_confidence`
  - `labeler` = `claude` or `codex`; `labeler_confidence` ∈ {high, medium, low}.
  - Use the **exact controlled-vocabulary strings** below. Booleans as TRUE/FALSE.

## What to read (and what to ignore)

Base every label on **`evidence_text` only**, with `heading_title`, `manifest_role`,
`page_start`/`page_end`, and `project_title` as context. **IGNORE the machine-guess columns**
(`candidate_class_guess`, `resource_area_guess`, `threshold_types_guess`,
`determination_polarity_guess`, `matched_cue_group`) — they are the output of the cheap regex
layer you are helping to grade; anchoring on them defeats the purpose. Do not use outside
knowledge about the specific project; if the passage doesn't support a judgment, say so.

## Decision sequence (apply in order to every row)

### 1. `gold_is_determination` — TRUE/FALSE
Does the passage contain (or directly state the basis for) a significance judgment?
- **FALSE** (canonical junk-text list — identical to the extractor prompt's): tables of
  contents, acronym lists, boilerplate, pure project description, affected-environment/background
  descriptions, methodology text, cross-references, lists of comments received. If FALSE → set
  `gold_determination_class` to
  `not_a_determination`, leave the remaining fields blank except `gold_evidence_span_ok`
  (still answer it), and move on.
- **TRUE:** the agency concludes something about significance — a formal FONSI statement, or a
  resource section that states significance criteria and concludes how the impact compares
  ("impacts would be less than significant", "no significant impacts would occur", "would be
  significant unless…", "impacts would remain significant").
- A section that ONLY states criteria ("impacts would be significant if …") **without any
  conclusion** in the passage: `gold_is_determination = FALSE` and set
  `gold_evidence_span_ok = FALSE` (the determination likely exists but outside this window).
- **No-Action alternative rule:** a statement that the No-Action alternative would have no (or
  reduced) impacts **IS a determination** — `gold_is_determination = TRUE`, scope
  `alternative_specific`, class per its conclusion (usually `no_significant_impact`). Apply this
  consistently; do not treat No-Action conclusions as descriptive text.

### 2. `gold_determination_class` — exactly one of:
| Value | Use when the passage… |
|---|---|
| `no_significant_impact` | is the formal FONSI-style conclusion: no significant impact (project-wide or explicitly "no significant impacts to X") |
| `less_than_significant` | concludes an impact exists but is below the significance line, WITHOUT depending on committed mitigation |
| `less_than_significant_with_mitigation` | concludes below-the-line **because of** committed mitigation ("with implementation of the measures…", "would be significant absent mitigation") |
| `significant_adverse` | concludes a significant adverse impact |
| `significant_unavoidable` | concludes significant AND unavoidable/not fully mitigable |
| `eis_required` | states an EIS is required / impacts warrant an EIS |
| `ambiguous` | a determination is clearly being made but you genuinely cannot tell which; explain in `gold_notes` |
| `not_a_determination` | (paired with `gold_is_determination = FALSE`) |

**The hard pair:** `less_than_significant` vs `less_than_significant_with_mitigation`.
The `_with_mitigation` label requires the conclusion to **depend on** committed measures —
"resource protection measures will be implemented … impacts would be less than significant"
counts; impacts that are minor **by the project's inherent design** (small footprint, existing
right-of-way) do NOT count, even if BMPs are mentioned in passing.

### 3. `gold_determination_scope` — exactly one of:
`project_overall` (the formal FONSI / whole-project conclusion) ·
`resource_specific` (one resource area's conclusion — most Environmental Consequences
sections) · `alternative_specific` (tied to one alternative only) · `threshold_specific`
(the determination IS a threshold finding, e.g. "§106 adverse effect" linked to the NEPA
conclusion) · `programmatic_or_tiered` (tiers from a programmatic review) · `procedural`.
When a resource section also echoes the project conclusion, prefer `resource_specific`.

### 4. `gold_resource_area` — exactly one of:
`air_quality, water, biological, cultural, visual, noise, soils_geology, socioeconomic,
transportation, land_use, climate_ghg, public_health, project_wide, unknown`
- `project_wide` = a project-level / FONSI conclusion not tied to one resource (pairs with
  scope `project_overall`). `unknown` = a resource-specific finding whose resource you genuinely
  cannot place. Do not use `unknown` for project-level conclusions — use `project_wide`.
- Mapping guidance: wetlands/floodplains → `water`; wildlife, vegetation, special-status
  species → `biological`; historic/tribal/§106 → `cultural`; agriculture/farmland, recreation,
  ROW compatibility → `land_use`; EMF/safety/hazmat → `public_health`; EJ/economics →
  `socioeconomic`; GHG/climate → `climate_ghg`.
- `project_overall` rows (the FONSI itself) → **always `project_wide`** (no exceptions — a
  conclusion framed around a single resource should instead be scoped `resource_specific`; this
  keeps the rule identical to the extractor's).
- If the passage genuinely covers multiple resources with one conclusion, pick the dominant
  one and note "multi-resource" in `gold_notes`.

### 5. `gold_primary_threshold_type` / `gold_primary_threshold_status`
Only when the determination is **anchored to a regulatory threshold in the passage**:
- Types: `NAAQS, PSD, ESA_take, ESA_jeopardy, NHPA_adverse_effect, wetland_floodplain,
  noise_threshold, visual_vrm, other_quantitative, none, unknown`
- Status: `exceeds, does_not_exceed, may_exceed, mitigated_below, not_evaluated, unknown`
- A mere mention of a statute is NOT an anchor — the conclusion must lean on it ("emissions
  would not exceed NAAQS" → `NAAQS` / `does_not_exceed`). If several, pick the one doing the
  most work and list others in `gold_notes`. Default `none` / `none`→ leave status blank.

### 6. `gold_mitigation_link` — TRUE/FALSE
TRUE iff the conclusion **depends on committed/required mitigation** (mirrors the class
distinction in step 2; TRUE whenever you chose `less_than_significant_with_mitigation`, and
also TRUE for e.g. a `significant_unavoidable` that relies on partial mitigation).
Baseline design features and voluntary BMPs alone → FALSE.

### 7. `gold_evidence_span_ok` — TRUE/FALSE
TRUE if the passage itself contains the determination you coded. FALSE if you had to infer it
or the operative sentence is clearly outside the window (this measures retrieval quality —
answer it for every row, including negatives).

### 8. `gold_needs_human_review` + `gold_notes`
Set TRUE for anything you'd want the human analyst to look at (genuinely ambiguous, garbled
text, conflicting statements). ALWAYS put a one-line reason in `gold_notes` when
`ambiguous`, `unknown`, `needs_human_review=TRUE`, or `labeler_confidence=low`.

## Worked examples

1. *"Impacts to cultural resources … would be significant if the Project results in adverse
impacts to NRHP-eligible properties that cannot be satisfactorily mitigated … CUL-1 requires
… With implementation of these measures, impacts would not be significant."*
→ TRUE · `less_than_significant_with_mitigation` · `resource_specific` · `cultural` ·
`NHPA_adverse_effect`/`mitigated_below` · mitigation TRUE · span_ok TRUE · confidence high.

2. *"FINDING OF NO SIGNIFICANT IMPACT … Based on the analysis in the EA, Western finds that
the proposed action will not significantly affect the quality of the human environment…"*
→ TRUE · `no_significant_impact` · `project_overall` · `unknown` · `none` · mitigation FALSE
(unless the finding is expressly conditioned on mitigation) · span_ok TRUE.

3. *"…………………………13 2.2.2 Storage Reservoir Geology ………14 2.2.3 King Island Temporar"*
→ FALSE · `not_a_determination` · span_ok FALSE · notes "TOC fragment".

4. *"Construction would temporarily increase dust; with standard practices emissions remain
well below de minimis thresholds; operational emissions are negligible."* (concludes without
committed-mitigation dependence) → TRUE · `less_than_significant` · `resource_specific` ·
`air_quality` · `NAAQS`/`does_not_exceed` (if NAAQS is the stated anchor) · mitigation FALSE.

## Process requirements

- Label **every row** (all 400); never skip. If text is unreadable/garbled → FALSE +
  `needs_human_review=TRUE` + note.
- Work row-by-row; do not batch-guess by heading. Re-read borderline passages once before
  settling.
- Be consistent: same wording pattern → same label, first row to last.
- Do not modify the input file; write only your own output CSV with the exact columns above.

## After both labelers finish

Run `python phase2/code/deliverable02/gold_agreement.py` — it reports per-field agreement,
auto-accepts rows where both labelers agree on the core fields, and writes
`output/deliverable02/gold_disagreements.csv` for the human analyst to adjudicate (fill the
`final_*` columns). Then `gold_agreement.py --finalize` assembles
`gold/significance_gold.parquet` (with a deterministic 30% holdout) for `05_validate`.

**Methodological caveat (for the record):** the extraction pipeline under evaluation also uses
a Claude model. Independence is protected by (a) a second, non-Anthropic labeler (Codex),
(b) human adjudication of every disagreement, and (c) the analyst spot-checking ~40 randomly
chosen *agreed* rows. Report Gate-3 metrics with this design stated.
