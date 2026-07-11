# D2 Gold-Set Labeling Prompt — EIS TRACK (dual-labeler: Claude + Codex)

> This is the **EIS** labeling task. It is a separate job from the FONSI labeling
> (`gold_labeling.md`) with **its own input and its own output files** — never mix the two.

## ⚑ WHO YOU ARE — READ FIRST (this decides your output file)

Two reviewers label this set **independently and in parallel**. You are exactly one of them — pick
your identity and write ONLY to your own file:

| If you are… | You are… | Write ONLY to | Set `labeler` = |
|---|---|---|---|
| **Claude** | **Reviewer 1** | `phase2/data/analysis/deliverable02/gold/labels_eis_claude.csv` | `claude` |
| **Codex** | **Reviewer 2** | `phase2/data/analysis/deliverable02/gold/labels_eis_codex.csv` | `codex` |

You and the other reviewer run **at the same time**. You write ONLY your assigned file; you NEVER
read, edit, or overwrite the other reviewer's file or the input file. Because each reviewer writes a
different file, simultaneous work cannot conflict. **Do not consult, compare, or coordinate with the
other reviewer** — two *independent* answer keys are the entire point. Work end-to-end without
asking for help; if something is genuinely unresolvable, code your best judgment and flag it in
`gold_notes` rather than stopping.

---

You are an expert NEPA analyst labeling an **answer key** ("gold set") for Deliverable 2:
*Determinations of significance across resource areas* — the **EIS track**. Your labels grade an LLM
extraction pipeline (Gate 3), so **accuracy and consistency matter more than speed**. Disagreements
between the two reviewers are adjudicated later by the human analyst.

**EIS context (how this differs from the FONSI task):** these passages come from **Environmental
Impact Statements** — usually **Environmental Consequences** chapters where the agency analyzes each
resource area's impacts and states whether they are significant. Because an EIS is the *higher* level
of review, you will see **more `significant_adverse` and `significant_unavoidable` conclusions** than
in FONSIs. Two EIS-specific rules:

- **An EIS does NOT issue a FONSI.** It leads to a Record of Decision (ROD). So a project-wide /
  overall conclusion (`gold_resource_area = project_wide`, scope `project_overall`) is **rare** — use
  it only when the passage states an explicit whole-project or ROD-level significance conclusion. The
  large majority of rows are `resource_specific`.
- **`eis_required`** is almost never a determination *inside* an EIS (the EIS already exists). Use it
  only if the passage explicitly says a **further or supplemental EIS** is required; otherwise do not.

## Task — one row PER RESOURCE DETERMINATION (multi-determination grain)

The extraction pipeline emits **one determination per resource area** a passage concludes on — a
single Environmental Consequences section can conclude separately on air, water, biological,
cultural, … resources. Your gold set must match that grain so it can grade the pipeline.

For **each window** in the reading list, decide which resource areas the passage makes a
significance conclusion about, and **emit one output row per (window × resource area)**:

- A section concluding on 3 resources → **3 rows** (same `evidence_span_id`, different
  `gold_resource_area`).
- A section that also states an explicit whole-project / ROD-level conclusion → add one row with
  `gold_resource_area = project_wide`, scope `project_overall` (rare in an EIS — see above).
- A **junk / non-determination** window (methodology, affected-environment/baseline description,
  alternatives description, table of contents, acronym list, cross-reference, comment list) →
  **exactly one row** with `gold_resource_area = none`, `gold_is_determination = FALSE`,
  `gold_determination_class = not_a_determination`.

Never emit a determination for a resource the passage does **not** conclude on. Do not merge several
resources into one row.

- **Input (reading list) — READ-ONLY:** `phase2/output/deliverable02/significance_gold_queue_eis.csv`
  (400 rows, one per window; identical content is also in
  `phase2/data/analysis/deliverable02/gold/significance_gold_queue_eis.parquet` if you prefer
  parquet). **`evidence_text` contains commas and newlines — parse with a real CSV reader
  (`pandas.read_csv` / `csv` module), NEVER by splitting on commas or lines.** Do not modify it.
- **Output:** ONE long CSV — the file assigned to your identity in the "WHO YOU ARE" table above
  (Reviewer 1 / Claude → `labels_eis_claude.csv`; Reviewer 2 / Codex → `labels_eis_codex.csv`).
  Write it with a real CSV writer so long text fields are quoted correctly.
- **Output columns (exactly these, in this order; one row per resource determination):**
  `evidence_span_id, gold_resource_area, gold_is_determination, gold_determination_class,
  gold_determination_scope, gold_primary_threshold_type, gold_primary_threshold_status,
  gold_mitigation_link, gold_evidence_span_ok, gold_needs_human_review, gold_notes,
  labeler, labeler_confidence`
  - **`(evidence_span_id, gold_resource_area)` is the row key** — it must be unique in your file.
    If a window truly concludes on the same resource twice with different classes, keep the
    operative/final conclusion and note the other in `gold_notes`.
  - `labeler` = `claude` or `codex`; `labeler_confidence` ∈ {high, medium, low}.
  - Use the **exact controlled-vocabulary strings** below. Booleans as TRUE/FALSE.

## Save incrementally — checkpoint every 100 windows (do NOT hold everything to the end)

This is a ~400-window job that expands to ~1,000+ output rows. Do **not** accumulate all labels in
memory and write once at the very end — a crash near the end would lose everything. Instead:

1. Process the windows **in the order they appear** in the input file.
2. **After every 100 windows** (at windows 100, 200, 300, 400), **write your output CSV to disk with
   ALL rows completed so far** — overwrite the whole file each time (header + every row you've
   produced). After each write the file is a complete, valid CSV of your progress, so a later error
   never costs more than the last <100 windows.
3. **Resuming after an error:** first read your existing output CSV, collect the `evidence_span_id`s
   you have already labeled, and **continue from the first window you have not yet covered** — do not
   re-label windows already in your file (that would create duplicate keys).
4. When all 400 windows are done, do one final write so the file is complete.

Keep writing to the **same single output file** assigned to your identity — checkpoints overwrite it
in place; never create per-batch files.

## What to read (and what to ignore)

Base every label on **`evidence_text` only**, with `heading_title`, `page_start`/`page_end`, and
`section_id` as context. **IGNORE the machine-guess columns** (`candidate_class_guess`,
`resource_area_guess`, `threshold_types_guess`, `determination_polarity_guess`, `matched_cue_group`)
— they are the output of the cheap regex layer you are helping to grade; anchoring on them defeats
the purpose. Do not use outside knowledge about the specific project; if the passage doesn't support
a judgment, say so.

## Decision sequence (apply to every window)

### 0. Enumerate the determinations in the window
Read the whole passage and list every resource area it reaches a significance conclusion about. Each
becomes one output row. If it reaches **no** conclusion at all → emit the single junk row (see Task)
and move on.

### 1. `gold_is_determination` — TRUE/FALSE (per row)
- **FALSE** only for the single junk row of a non-determination window (methodology,
  affected-environment/baseline description, alternatives description, TOC, acronym list,
  cross-reference, comment list). Set `gold_determination_class = not_a_determination`,
  `gold_resource_area = none`, leave the remaining vocab fields blank except `gold_evidence_span_ok`,
  and move on.
- **TRUE** for every real resource determination: the agency concludes how a resource's impact
  compares to the significance threshold ("impacts would be significant", "less than significant",
  "significant and unavoidable", "would not be significant with mitigation").
- A resource that ONLY states criteria/methodology ("impacts would be significant if …") **without a
  conclusion** in the passage: do not emit a row for it; if that is all the window contains, treat it
  as junk and set `gold_evidence_span_ok = FALSE`.
- **Alternatives:** a conclusion tied to a specific action alternative (or the No-Action alternative)
  IS a determination — emit a row, scope `alternative_specific`, class per its conclusion, resource =
  the affected resource.

### 2. `gold_determination_class` — exactly one of:
| Value | Use when the row's resource conclusion… |
|---|---|
| `no_significant_impact` | concludes no significant impact to this resource |
| `less_than_significant` | concludes an impact exists but is below the significance line, WITHOUT depending on committed mitigation |
| `less_than_significant_with_mitigation` | concludes below-the-line **because of** committed mitigation |
| `significant_adverse` | concludes a significant adverse impact |
| `significant_unavoidable` | concludes significant AND unavoidable / not fully mitigable (common in EIS "significant and unavoidable" findings) |
| `eis_required` | ONLY if the passage explicitly calls for a further / supplemental EIS (rare inside an EIS) |
| `ambiguous` | a determination is clearly made but you genuinely cannot tell which; explain in `gold_notes` |
| `not_a_determination` | (junk row only, paired with `gold_is_determination = FALSE`) |

**The hard pairs:** (a) `less_than_significant` vs `less_than_significant_with_mitigation` — the
`_with_mitigation` label requires the conclusion to **depend on** committed measures, not incidental
BMPs. (b) `significant_adverse` vs `significant_unavoidable` — use `significant_unavoidable` when the
text says the significant impact **cannot be fully mitigated / is unavoidable**; otherwise
`significant_adverse`.

### 3. `gold_determination_scope` — exactly one of:
`resource_specific` (one resource area's conclusion — most EIS Environmental Consequences text) ·
`alternative_specific` (tied to one alternative only) · `threshold_specific` (the determination IS a
threshold finding, e.g. a §106 adverse-effect finding driving the NEPA conclusion) ·
`programmatic_or_tiered` (tiers from a programmatic EIS) · `project_overall` (an explicit
whole-project / ROD-level conclusion — rare) · `procedural`.

### 4. `gold_resource_area` — exactly one of:
`air_quality, water, biological, cultural, visual, noise, soils_geology, socioeconomic,
transportation, land_use, climate_ghg, public_health, project_wide, unknown, none`
- This is part of the **row key** — every real determination in a window gets its own resource.
- `project_wide` = an explicit whole-project / ROD-level conclusion (pairs with scope
  `project_overall`; rare in an EIS). `unknown` = a resource-specific finding whose resource you
  genuinely cannot place (note why). `none` = the junk row of a non-determination window.
- Mapping guidance (identical to the extractor): wetlands/floodplains/groundwater → `water`;
  wildlife, vegetation, special-status species → `biological`; historic/tribal/§106 → `cultural`;
  agriculture/farmland, recreation, ROW/land-use plans → `land_use`; worker/public safety, EMF,
  hazmat, solid waste → `public_health`; EJ/economics/public services → `socioeconomic`;
  GHG/climate → `climate_ghg`; traffic/roads/aviation → `transportation`.

### 5. `gold_primary_threshold_type` / `gold_primary_threshold_status` (per row)
Only when THIS resource's determination is **anchored to a regulatory threshold in the passage**:
- Types: `NAAQS, PSD, ESA_take, ESA_jeopardy, NHPA_adverse_effect, wetland_floodplain,
  noise_threshold, visual_vrm, other_quantitative, none, unknown`
- Status: `exceeds, does_not_exceed, may_exceed, mitigated_below, not_evaluated, unknown`
- A mere mention of a statute is NOT an anchor — the conclusion must lean on it. Default `none` /
  `none` → leave status blank.

### 6. `gold_mitigation_link` — TRUE/FALSE (per row)
TRUE iff THIS resource's conclusion **depends on committed/required mitigation** (mirrors the class
distinction; TRUE whenever you chose `less_than_significant_with_mitigation`, and also TRUE for a
`significant_unavoidable` that relies on partial mitigation). Baseline design features and voluntary
BMPs alone → FALSE.

### 7. `gold_evidence_span_ok` — TRUE/FALSE (per row)
TRUE if the passage itself contains the determination you coded. FALSE if you had to infer it or the
operative sentence is clearly outside the window (this measures retrieval quality — answer it for
every row, including the junk row).

### 8. `gold_needs_human_review` + `gold_notes` (per row)
Set TRUE for anything you'd want the human analyst to look at (genuinely ambiguous, garbled text,
conflicting statements). ALWAYS put a one-line reason in `gold_notes` when `ambiguous`, `unknown`,
`needs_human_review=TRUE`, or `labeler_confidence=low`.

## Worked examples

1. *"Construction emissions would not exceed the NAAQS and impacts to air quality would be less than
significant. Impacts to greater sage-grouse would be **significant and unavoidable** even with the
applicant-committed measures. Noise at the nearest residence would be less than significant with
implementation of the timing restrictions in MM-NOISE-1."*
→ **three rows**, same `evidence_span_id`:
   - `air_quality` · TRUE · `less_than_significant` · `resource_specific` · `NAAQS`/`does_not_exceed` · mitigation FALSE · span_ok TRUE
   - `biological` · TRUE · `significant_unavoidable` · `resource_specific` · `ESA_take`/`unknown` · mitigation TRUE (relies on partial measures) · span_ok TRUE
   - `noise` · TRUE · `less_than_significant_with_mitigation` · `resource_specific` · `noise_threshold`/`mitigated_below` · mitigation TRUE · span_ok TRUE

2. *"3.4 Methodology. This section describes the analysis area and the methods used to evaluate
impacts to visual resources. Impact intensity is defined as negligible, minor, moderate, or major."*
→ **one row**: `none` · FALSE · `not_a_determination` · span_ok FALSE · notes "methodology, no conclusion".

3. *"Under the No Action Alternative, none of the proposed facilities would be built and there would
be no significant impacts to cultural resources."*
→ **one row**: `cultural` · TRUE · `no_significant_impact` · `alternative_specific` · `none` · mitigation FALSE · span_ok TRUE.

## Process requirements

- Cover **every window** in the reading list; never skip one. Emit at least one row per window (the
  junk row if there is no determination).
- Enumerate resources first, then code each row; do not batch-guess by heading. Re-read borderline
  passages once before settling.
- Be consistent: same wording pattern → same label, first window to last.
- Keep `(evidence_span_id, gold_resource_area)` unique in your file.
- Do not modify the input file; write only your own long output CSV with the exact columns above.

## After both labelers finish

Run `python phase2/code/deliverable02/gold_agreement.py --track eis` — it aligns the two long CSVs on
`(evidence_span_id, gold_resource_area)`, reports per-field agreement (and resource-set agreement),
auto-accepts rows where both labelers agree on the core fields, and writes
`output/deliverable02/gold_disagreements_eis.csv` for the analyst to adjudicate (fill the `final_*`
columns). Then `gold_agreement.py --track eis --finalize` assembles
`gold/significance_gold_eis.parquet` (30% holdout by window) for
`05_validate_significance.py --track eis`.

**Methodological caveat (for the record):** the extraction pipeline under evaluation also uses a
Claude model. Independence is protected by (a) a second, non-Anthropic labeler (Codex),
(b) human adjudication of every disagreement, and (c) the analyst spot-checking ~40 randomly chosen
*agreed* rows. Report Gate-3 metrics with this design stated.
