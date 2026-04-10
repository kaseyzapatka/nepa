# Deliverable 06 Transmission Validation Memo

Date: 2026-02-20
Scope: memorialize manual validation notes for transmission extraction/adjudication and define concrete workflow fixes.
Data snapshot: `data/analysis/projects_combined.parquet` as currently present in this repo on 2026-02-20.

## Snapshot Metrics (Current File)
- Clean + strict transmission projects: 205
- Projects with 2+ transmission candidates: 62
- Projects with 2+ distinct candidate values: 59
- Projects with `project_transmission_length_llm_trigger == TRUE`: 26
- Taxonomy counts in strict set:
  - `do_not_sum`: 146
  - `llm`: 26
  - `take_max`: 14
  - `sum`: 12
  - `build_verb_winner`: 6
  - `choose_alternative`: 1

## Case Log From Manual Review

### Cases that appear correct or mostly correct
- `027e8fc7-069e-9bcc-0d19-c87315aedb8a`
  - Current: `sum` to 33.7 miles (15.9 + 17.8).
  - Note: good additive behavior.
- `2acd8c11-4c32-f636-659f-c7a62f0619b8`
  - Current: build-verb winner selects 20 miles.
  - Note: good build-verb behavior.
- `3148ad5b-0eb7-e20b-7688-276559b2268c`
  - Current: build-verb winner selects 9.77 miles.
  - Note: good build-verb behavior.
- `e574fb6c-c0de-5842-3493-32cca57052f4`
  - Current: `sum` to 10.2 miles.
  - Note: additive behavior appears appropriate.
- `d9a9a307-066d-334a-0095-a8f3862a427e`
  - Current: LLM selects 3.8 miles acquisition line; this aligns with intended choice.
- `f493eb06-f330-28f1-4ea9-5e830f78daea`
  - Current: `take_max` selects 1.34 miles; this appears acceptable for this case.
- `a38e1b8a-7776-458d-fa50-f4f325ebad52`
  - Current: LLM selects 18.39 miles and ignores the access-road mileage.
  - Note: desired behavior.

### Cases likely incorrect (selection/adjudication)
- `06ee24b6-e7bd-10d4-4924-31154372b4a3`
  - Current: LLM picks 11.389 miles from 60,132 ft.
  - Expected: add both lengths (6,167 ft + 60,132 ft).
  - Likely issue: LLM prompt only allows selecting one candidate.
- `29402d2a-61cf-25dc-5050-bb2f2d62ff48`
  - Current: LLM picks 26.0 miles from "26.0 miles north of Helena".
  - Expected: 3.14 miles from "BLM authorizes 3.14 miles of this existing powerline".
  - Likely issue: location-distance not fully filtered before adjudication.
- `ff089bb4-8710-815b-7eac-d3e94c6b4e49`
  - Current: `take_max` = 13 miles.
  - Expected: sum the two segments (10 + 13).
  - Likely issue: no additive cue detected even though sentence enumerates two valid segments.
- `9b92344b-1230-be3c-58f4-5c2df57521ba`
  - Current: `sum` = 3.5 miles.
  - Expected: likely 0.5 miles (or at minimum not naive 3 + 0.5).
  - Likely issue: additive rule sums values with mixed semantics (land rights vs transmission-line rights).
- `baa1b870-24af-1703-22ae-cd0120271f40`
  - Current: LLM picks 1.61 miles (public land crossing extent).
  - Expected: 10 miles total line length.
  - Likely issue: partial crossing candidate not downweighted enough.
- `b48dbb2b-e9e5-999b-6cb9-049605356a7b`
  - Current: LLM picks 3.0 miles over ~1/3 mile.
  - Expected: ~0.33 mile.
  - Likely issue: fraction-form parsing/normalization + candidate representation.
- `88637c69-789b-99df-4593-7fb2601ea8d9`
  - Current: LLM picks 4.7 (public lands crossing) instead of clear total 11.7 miles long.
  - Expected: 11.7 miles.
  - Likely issue: total-length cue not strongly prioritized over crossing extents.

### Cases likely should be excluded from transmission-infrastructure analysis
- `1aff267e-235b-abb2-347a-92d3ff989575`
  - Vegetation herbicide follow-up along ROW.
- `284f25aa-e022-7781-51c0-d338390aa866`
  - Access-road maintenance along corridor.
- `ac0254b4-77b3-f1bb-6855-2e8c7f94538c`
  - Vegetation reclamation/removal program.
- `dab94a46-e67a-4865-0437-0fdefe83ba69`
  - Danger-tree management / routine inspections.
- `8d4f94cf-0cab-3ccf-00a0-c7c18dbfb2b9`
  - Routine road maintenance; also vulnerable to location-distance confusion.

## High-Value Fixes

### 1) Tighten transmission inclusion filtering (highest priority)
- Add a stronger exclusion gate for O&M/vegetation/reclamation/inspection projects.
- Keep these as `project_is_transmission_broad`, but default them out of strict infrastructure analysis unless there is explicit line-build/upgrade/rebuild scope.
- Important bug to fix in current code:
  - In `code/extract/extract_technology.py`, maintenance flagging currently uses:
    - `context_text.str.contains(TRANSMISSION_MAINTENANCE_RE, regex=False)`
  - With a compiled regex, `regex=False` prevents regex matching and can silently fail.
  - Should use regex matching (`regex=True` or omit the argument).

### 2) Improve candidate filtering before any adjudication
- Hard-drop location-distance patterns:
  - `X miles north/south/east/west of <place>`
  - `located approximately X miles <direction> of`
- Hard-tag and downrank (or exclude for total-length selection) partial extents:
  - `cross public lands for X miles`
  - `X miles on public/federal/BLM lands`
- Add candidate role tags:
  - `total_length_explicit`, `segment_length`, `partial_land_crossing`, `location_distance`, `access_road_length`, `vegetation_maintenance_extent`, `land_rights_extent`.

### 3) Refine sum/take-max logic
- Do not sum all distinct values whenever additive language appears anywhere in the full text.
- Sum only when candidate roles are compatible and explicitly segmental, for example:
  - same sentence/clause list joined by `and`/commas with shared line object
  - mixed-action projects where `new_build` and `upgrade` segments are both part of one project scope
- Prefer explicit totals (`X miles long`, `total length`) over component/crossing values.

### 4) Upgrade LLM adjudication contract (if used)
- Current prompt forces one-candidate selection. This blocks valid additive outcomes.
- Update LLM output schema to return:
  - `decision_type`: `sum | take_max | choose_alternative | do_not_sum`
  - `selected_candidate_ids`: list
  - `selected_length_miles`
  - `rationale_short`
- Trigger LLM only for unresolved multi-candidate rows after deterministic filters and role-tagging.

### 5) Fraction parsing and numeric normalization
- Preserve and parse fraction expressions like `1/3 mile` before regex extraction.
- Keep both raw token and normalized miles in candidate JSON for QA.

## LLM Prompt Improvements
- Add explicit negative examples inside prompt:
  - "26 miles north of Helena" is location, not line length.
  - "cross public lands for 4.7 miles" is a partial crossing extent.
- Add explicit priority rule:
  - If any candidate sentence says `X miles long`/`in length`, prefer it unless contradicted by a clearer total.
- Allow multi-candidate output (sum) when segments are clearly additive.

## Should We Use Claude?
Short answer: maybe, but only after deterministic fixes above.

Recommended approach:
1. Build a frozen eval set from these reviewed IDs plus additional multi-candidate rows.
2. Compare adjudication accuracy with identical candidate inputs and output schema:
   - current Ollama model
   - Claude (same schema/prompt intent)
3. Promote a model only if it materially improves error classes that rules cannot reliably solve.

Rationale: many current failures are upstream candidate/filter issues; switching models alone will not fix location-distance or maintenance-inclusion errors.

## Implementation Sequence
1. Deterministic filtering hotfixes (maintenance gate + location/partial crossing filtering + fraction parsing).
2. Role-tagged candidate adjudication (sum/take-max/alternative/do-not-sum with stricter compatibility checks).
3. Optional LLM stage for unresolved multi-candidate cases only.
4. Re-run extraction and update this memo with before/after accuracy on the reviewed set.

## Related Files
- `code/extract/extract_technology.py`
- `code/extract/extract_data.py`
- `code/deliverable06/01_transmission.R`
- `tmp/d6_transmission_user_review_ids_diagnostics.csv`

## Follow-up Inspection Queue (User-Requested)

Projects to explicitly inspect in the next QA pass:

- `ff089bb4-8710-815b-7eac-d3e94c6b4e49`
  - Label to verify: example of LLM getting it right.
- `baa1b870-24af-1703-22ae-cd0120271f40`
  - Label to verify: example of LLM getting it right.
- `7094a354-0fee-062a-2d26-5e96f85ed60d`
  - Label to verify: example of LLM getting it right.
- `09299b10-2b00-47bc-9dd1-bb93cc27f659`
  - Label to verify: maintenance (uses "conduct routine road maintenance").
- `1aff267e-235b-abb2-347a-92d3ff989575`
  - Label to verify: maintenance (uses "herbicide application on incompatible vegetation").
- `284f25aa-e022-7781-51c0-d338390aa866`
  - Label to verify: maintenance (uses "road maintenance").
- `3fbe2462-7af6-4c8f-5613-90e8dc9bcc7c`
  - Label to verify: maintenance (uses "crews will survey and inspect vegetation along").
- `02827ece-5b58-b374-1150-f3c0718f81c8`
  - Label to verify: maintenance (uses "perform routine maintenance").
- `091946ad-d76a-c6e5-985b-b6090db23fe9`
  - Label to verify: upgrade (uses "replacement project").
- `0dfa9f0c-9b67-2669-cece-74de2002eac3`
  - Label to verify: upgrade (uses "Replacement" in context).
- `29b57e84-b5ea-e75b-2a63-7b08fc3a1d90`
  - Label to verify: upgrade (uses "proposes to replace").
- `f07667a4-97d1-255d-7710-924e15acc5b3`
  - Label to verify: new_build (uses "proposing to construct").
