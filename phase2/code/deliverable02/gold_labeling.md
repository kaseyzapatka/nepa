# D2 Gold-Set Labeling Prompt (dual-labeler: Claude + Codex)

You are an expert NEPA analyst labeling an **answer key** ("gold set") for Deliverable 2:
*Determinations of significance across resource areas*. Your labels will be used to grade an
LLM extraction pipeline (Gate 3), so **accuracy and consistency matter more than speed**. Two
labelers (Claude and Codex) perform this task **independently**; disagreements are adjudicated
by the human analyst. Do not consult the other labeler's output.

## Task — one row PER RESOURCE DETERMINATION (multi-determination grain)

The extraction pipeline emits **one determination per resource area** a passage concludes on — a
single Environmental Consequences / FONSI window can conclude separately on air, water,
biological, cultural, … resources, plus a project-wide FONSI statement. Your gold set must match
that grain so it can grade the pipeline.

For **each window** in the reading list, decide which resource areas the passage makes a
significance conclusion about, and **emit one output row per (window × resource area)**:

- A window concluding on 3 resources → **3 rows** (same `evidence_span_id`, different
  `gold_resource_area`).
- A window that also states a project-wide FONSI/decision → **add one more row** with
  `gold_resource_area = project_wide`, scope `project_overall`.
- A **junk / non-determination** window (TOC, acronym list, boilerplate, pure project
  description, affected-environment/background, methodology, cross-reference, comment list) →
  **exactly one row** with `gold_resource_area = none`, `gold_is_determination = FALSE`,
  `gold_determination_class = not_a_determination`.

Never emit a determination for a resource the passage does **not** conclude on. Do not merge
several resources into one row.

- **Input (reading list):** `phase2/data/analysis/deliverable02/gold/significance_gold_queue.parquet`
  (same content as `phase2/output/deliverable02/significance_gold_queue.csv`) — one row per window.
- **Output:** ONE long CSV, depending on who you are:
  - Claude → `phase2/data/analysis/deliverable02/gold/labels_claude.csv`
  - Codex → `phase2/data/analysis/deliverable02/gold/labels_codex.csv`
- **Output columns (exactly these; one row per resource determination):**
  `evidence_span_id, gold_resource_area, gold_is_determination, gold_determination_class,
  gold_determination_scope, gold_primary_threshold_type, gold_primary_threshold_status,
  gold_mitigation_link, gold_evidence_span_ok, gold_needs_human_review, gold_notes,
  labeler, labeler_confidence`
  - **`(evidence_span_id, gold_resource_area)` is the row key** — it must be unique in your file.
    If a window truly concludes on the same resource twice with different classes, keep the
    operative/final conclusion and note the other in `gold_notes`.
  - `labeler` = `claude` or `codex`; `labeler_confidence` ∈ {high, medium, low}.
  - Use the **exact controlled-vocabulary strings** below. Booleans as TRUE/FALSE.

## What to read (and what to ignore)

Base every label on **`evidence_text` only**, with `heading_title`, `manifest_role`,
`page_start`/`page_end`, and `project_title` as context. **IGNORE the machine-guess columns**
(`candidate_class_guess`, `resource_area_guess`, `threshold_types_guess`,
`determination_polarity_guess`, `matched_cue_group`) — they are the output of the cheap regex
layer you are helping to grade; anchoring on them defeats the purpose. Do not use outside
knowledge about the specific project; if the passage doesn't support a judgment, say so.

## Decision sequence (apply to every window)

### 0. Enumerate the determinations in the window
Read the whole passage and list every resource area it reaches a significance conclusion about,
plus a project-wide row if it states a FONSI/whole-project decision. Each becomes one output row.
If it reaches **no** conclusion at all → emit the single junk row (see Task) and move on.

### 1. `gold_is_determination` — TRUE/FALSE (per row)
- **FALSE** only for the single junk row of a non-determination window (canonical junk-text list,
  identical to the extractor prompt's): tables of contents, acronym lists, boilerplate, pure
  project description, affected-environment/background descriptions, methodology text,
  cross-references, lists of comments received. Set `gold_determination_class =
  not_a_determination`, `gold_resource_area = none`, leave the remaining vocab fields blank except
  `gold_evidence_span_ok` (still answer it), and move on.
- **TRUE** for every real resource/project determination row: the agency concludes something about
  significance — a formal FONSI statement, or a resource section that states significance criteria
  and concludes how the impact compares ("impacts would be less than significant", "no significant
  impacts to X", "would be significant unless…", "impacts would remain significant").
- A resource that ONLY states criteria ("impacts would be significant if …") **without any
  conclusion** in the passage: do **not** emit a row for it (there is no determination to grade);
  if it is the window's only content, treat the window as junk and set `gold_evidence_span_ok =
  FALSE` on the junk row (the determination likely exists outside this window).
- **No-Action alternative rule:** a statement that the No-Action alternative would have no (or
  reduced) impacts **IS a determination** — emit a row, `gold_is_determination = TRUE`, scope
  `alternative_specific`, class per its conclusion (usually `no_significant_impact`), resource =
  the affected resource (or `project_wide` if framed for the whole project).

### 2. `gold_determination_class` — exactly one of:
| Value | Use when the row's resource conclusion… |
|---|---|
| `no_significant_impact` | is the formal FONSI-style conclusion: no significant impact (project-wide or explicitly "no significant impacts to X") |
| `less_than_significant` | concludes an impact exists but is below the significance line, WITHOUT depending on committed mitigation |
| `less_than_significant_with_mitigation` | concludes below-the-line **because of** committed mitigation ("with implementation of the measures…", "would be significant absent mitigation") |
| `significant_adverse` | concludes a significant adverse impact |
| `significant_unavoidable` | concludes significant AND unavoidable/not fully mitigable |
| `eis_required` | states an EIS is required / impacts warrant an EIS |
| `ambiguous` | a determination is clearly being made but you genuinely cannot tell which; explain in `gold_notes` |
| `not_a_determination` | (junk row only, paired with `gold_is_determination = FALSE`) |

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
The `project_wide` row uses `project_overall`; a resource row that also echoes the project
conclusion stays `resource_specific`.

### 4. `gold_resource_area` — exactly one of:
`air_quality, water, biological, cultural, visual, noise, soils_geology, socioeconomic,
transportation, land_use, climate_ghg, public_health, project_wide, unknown, none`
- This is part of the **row key** — every real determination in a window gets its own resource.
- `project_wide` = a project-level / FONSI conclusion not tied to one resource (pairs with
  scope `project_overall`). `unknown` = a resource-specific finding whose resource you genuinely
  cannot place (note why in `gold_notes`). `none` = the junk row of a non-determination window.
  Do not use `unknown` for project-level conclusions — use `project_wide`.
- Mapping guidance (identical to the extractor): wetlands/floodplains/groundwater → `water`;
  wildlife, vegetation, special-status species → `biological`; historic/tribal/§106 → `cultural`;
  agriculture/farmland, recreation, ROW/land-use plans → `land_use`; worker/public safety, EMF,
  hazmat, solid waste → `public_health`; EJ/economics/public services → `socioeconomic`;
  GHG/climate → `climate_ghg`; traffic/roads/aviation → `transportation`.
- `project_overall` rows (the FONSI itself) → **always `project_wide`**.

### 5. `gold_primary_threshold_type` / `gold_primary_threshold_status` (per row)
Only when THIS resource's determination is **anchored to a regulatory threshold in the passage**:
- Types: `NAAQS, PSD, ESA_take, ESA_jeopardy, NHPA_adverse_effect, wetland_floodplain,
  noise_threshold, visual_vrm, other_quantitative, none, unknown`
- Status: `exceeds, does_not_exceed, may_exceed, mitigated_below, not_evaluated, unknown`
- A mere mention of a statute is NOT an anchor — the conclusion must lean on it ("emissions
  would not exceed NAAQS" → `NAAQS` / `does_not_exceed`). Default `none` / `none` → leave status
  blank.

### 6. `gold_mitigation_link` — TRUE/FALSE (per row)
TRUE iff THIS resource's conclusion **depends on committed/required mitigation** (mirrors the
class distinction in step 2; TRUE whenever you chose `less_than_significant_with_mitigation`, and
also TRUE for e.g. a `significant_unavoidable` that relies on partial mitigation).
Baseline design features and voluntary BMPs alone → FALSE.

### 7. `gold_evidence_span_ok` — TRUE/FALSE (per row)
TRUE if the passage itself contains the determination you coded. FALSE if you had to infer it or
the operative sentence is clearly outside the window (this measures retrieval quality — answer it
for every row, including the junk row).

### 8. `gold_needs_human_review` + `gold_notes` (per row)
Set TRUE for anything you'd want the human analyst to look at (genuinely ambiguous, garbled
text, conflicting statements). ALWAYS put a one-line reason in `gold_notes` when
`ambiguous`, `unknown`, `needs_human_review=TRUE`, or `labeler_confidence=low`.

## Worked examples

1. *Window: "Air emissions during construction would remain below NAAQS and are less than
significant. Impacts to cultural resources … would be significant if the Project results in
adverse impacts to NRHP-eligible properties … CUL-1 requires … With implementation of these
measures, impacts would not be significant. Western finds no significant impact and issues this
FONSI."*
→ **three rows**, same `evidence_span_id`:
   - `air_quality` · TRUE · `less_than_significant` · `resource_specific` · `NAAQS`/`does_not_exceed` · mitigation FALSE · span_ok TRUE
   - `cultural` · TRUE · `less_than_significant_with_mitigation` · `resource_specific` · `NHPA_adverse_effect`/`mitigated_below` · mitigation TRUE · span_ok TRUE
   - `project_wide` · TRUE · `no_significant_impact` · `project_overall` · `none` · mitigation FALSE · span_ok TRUE

2. *"FINDING OF NO SIGNIFICANT IMPACT … Based on the analysis in the EA, Western finds that the
proposed action will not significantly affect the quality of the human environment…"* (only the
project conclusion, no per-resource findings)
→ **one row**: `project_wide` · TRUE · `no_significant_impact` · `project_overall` · `none` ·
mitigation FALSE (unless the finding is expressly conditioned on mitigation) · span_ok TRUE.

3. *"…………………………13 2.2.2 Storage Reservoir Geology ………14 2.2.3 King Island Temporar"*
→ **one row**: `none` · FALSE · `not_a_determination` · span_ok FALSE · notes "TOC fragment".

4. *"Construction would temporarily increase dust; with standard practices emissions remain well
below de minimis thresholds; operational emissions are negligible. Noise at the nearest receptor
would not exceed the county 65 dBA limit."*
→ **two rows**: `air_quality` · TRUE · `less_than_significant` · `resource_specific` ·
`NAAQS`/`does_not_exceed` · mitigation FALSE ; and `noise` · TRUE · `less_than_significant` ·
`resource_specific` · `noise_threshold`/`does_not_exceed` · mitigation FALSE.

## Process requirements

- Cover **every window** in the reading list; never skip one. Emit at least one row per window
  (the junk row if there is no determination).
- Enumerate resources first, then code each row; do not batch-guess by heading. Re-read
  borderline passages once before settling.
- Be consistent: same wording pattern → same label, first window to last.
- Keep `(evidence_span_id, gold_resource_area)` unique in your file.
- Do not modify the input file; write only your own long output CSV with the exact columns above.

## After both labelers finish

Run `python phase2/code/deliverable02/gold_agreement.py` — it aligns the two long CSVs on
`(evidence_span_id, gold_resource_area)`, reports per-field agreement (and resource-set
agreement: rows one labeler emitted that the other did not), auto-accepts rows where both
labelers agree on the core fields, and writes `output/deliverable02/gold_disagreements.csv` for
the human analyst to adjudicate (fill the `final_*` columns). Then
`gold_agreement.py --finalize` assembles `gold/significance_gold.parquet` (with a deterministic
30% holdout **by window**, so a window is entirely in or out) for `05_validate`.

**Methodological caveat (for the record):** the extraction pipeline under evaluation also uses
a Claude model. Independence is protected by (a) a second, non-Anthropic labeler (Codex),
(b) human adjudication of every disagreement, and (c) the analyst spot-checking ~40 randomly
chosen *agreed* rows. Report Gate-3 metrics with this design stated.
