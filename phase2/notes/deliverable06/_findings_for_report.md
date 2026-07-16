# Deliverable 06 — Findings for the Report (running list)

Points that must make it into `phase2/reports/deliverable06.qmd`. Mirrors the D4
`facts.md` convention: report-ready facts + the caveats that must travel with them.

**Source run:** D6 v2 (narrow-first) pipeline, `phase2/code/deliverable06/n01–n06`.
**Taxonomy:** `d6_v2_2`. **Extraction:** deterministic (LLM pass / Gate 3 not yet
run). **Compiled:** 2026-06-17

---

## 0. The honest big-picture framing (READ FIRST)

The clean-energy FONSI evidence **largely corroborates and supports
adoption/expansion of EXISTING categorical exclusions** — it does not reveal a
pile of brand-new untapped CE categories. Every substantive candidate already
maps to a real federal CE at *some* agency (see §5). So the deliverable's value
to CATF is two things, and we should say so plainly:

1. **Validation** — these action categories reliably reach a FONSI (no
   significant impact), which is exactly the evidence CEQ's Apr-2026 guidance
   says supports a CE.
2. **Concrete regulatory levers** — *adopt* an existing CE across agencies that
   lack it, or *expand* an existing CE's bounds (acreage, siting). This is more
   actionable for CATF than a speculative "new CE."

Frame the report around adopt/expand/validate, not "here are new CEs."

---

## 1. Candidate funnel (452 → 293 → 63)

| Stage | N | What it is |
|---|---:|---|
| Clean EA-source FONSI projects | 452 | The corpus |
| In a candidate category | 293 | The other 159 are excluded tech_groups (Other Clean 67, Nuclear 33, Biomass 27, Energy-Storage-manufacturing 21, Hydropower 7, CCS 4) |
| CE-shaped "profile" subset | 63 | The specific subtype worth writing up |

---

## 2. The three substantive candidates (with honest N caveats)

| Candidate | CE-shaped N | Strength | CE story |
|---|---:|---|---|
| **Transmission upgrades within existing ROW** | 48 | **The only robust finding** — reconductoring/rebuilds in existing ROW | Adopt TVA #17 across DOE/BLM/PMAs |
| **Geothermal exploration** | 7 | Small N — illustrative, not robust. Cleanest CE *shape* (few acres, wells, no new roads) | Adopt/expand the BLM CE + DOE B3.1 |
| **Solar on disturbed/developed land** | 8 | Small N — illustrative. Maps onto an existing CE (see §4) | Adopt/expand/validate DOE B5.16 |

**Say the N's out loud in the report.** Transmission (48) can carry a real
recurrence argument; geothermal (7) and solar-disturbed (8) are
"worth-a-closer-look" illustrations, not statistically robust categories. If
CATF wants either firmed up, widen the corpus (combined EA/FONSI docs, CE-track
text).

---

## 3. NOT findings (already covered, or out of scope)

- **Temporary resource assessment — NOT a finding.** It is already overwhelmingly
  handled as a CE: **2,317 CE vs 6 EA vs 7 EIS** in the clean universe (only 2
  FONSIs), and DOE B3.1 already covers it. Mention only as: "already CE territory;
  a possible *adoption* target for agencies that lack an equivalent." Do not
  present it as a discovered opportunity.
- **Wind (contrast).** The existing wind CE (DOE B5.18) only covers ≤2 turbines;
  utility wind has case-specific wildlife impacts → genuinely needs case-by-case
  review. Keep as the contrast that shows the method discriminates.
- **Off-scope transmission.** 9 `other_transmission` projects are NEPATEC
  mis-tags (nuclear demo, gas plant, mining, an appliance-efficiency standard)
  that leaked into the "Clean + Transmission" bucket — exclude; note as a
  data-quality caveat.

---

## 4. What "solar maps onto DOE B5.16" means (important)

**DOE B5.16 is an EXISTING categorical exclusion** for solar PV "located within a
previously disturbed or developed area." Our disturbed-site solar FONSIs match it
almost exactly. So this is **not a new-CE opportunity for DOE — DOE already
excludes this action.** The finding therefore becomes:

- **Validation:** real projects in this class received FONSIs / no significant
  impact, corroborating that the category is CE-appropriate.
- **Adoption/expansion lever:** other agencies (BLM, USDA, DoD) adopting a
  similar CE, or expanding B5.16's bounds (larger acreage, beyond "previously
  disturbed").

**CAVEAT:** the crosswalk match is unverified and the existing CE may be narrower
than our projects (scale, agency). Verify B5.16's exact scope against our
projects' scope to decide whether the story is "validate," "adopt," or "expand."
The same adopt/expand logic applies to transmission (TVA #17) and geothermal
exploration (BLM CE / DOE B3.1).

---

## 5. Existing-CE matches (the adopt/expand targets — verify, then REPORT)

The `all-MiniLM` semantic crosswalk (`candidate_ce_comparison.parquet`) mapped
each candidate to a real existing federal CE. Verify against canonical agency
text before citing; these are high-confidence:

| Candidate | Best existing CE | Story |
|---|---|---|
| Transmission upgrades | **TVA #17** — routine upgrade/reconductoring of existing transmission | Adopt across agencies |
| Solar (disturbed) | **DOE B5.16** — solar PV in a previously disturbed/developed area | Validate / adopt / expand (§4) |
| Geothermal exploration | **DOE B3.1** + BLM geothermal-exploration CE | Adopt / expand the proven model |
| Temp resource assessment | **DOE B3.1** site characterization & monitoring | Already CE'd (not a finding) |
| Wind (contrast) | **DOE B5.18** — ≤2 turbines only | Shows utility wind needs case-by-case |

---

## 6. Mitigation dependence — minor supporting signal (do NOT headline)

Preliminary, reused from v1 conditions (~51% "uncertain"). Keep as a supporting
detail in each candidate's bounding-conditions discussion, not the organizing
frame. A high case-specific-mitigation share is a yellow flag for a CE (limits
would need to absorb the mitigation as design features).

---

## 7. Bounding limits — the geothermal model works (medians, not maxes)

Geothermal exploration surfaces the CE template cleanly, e.g.
`11850027d9edfd683a121c0243725b29`: "drill up to twelve shallow exploratory
monitoring wells" → 2.5 acres, 12 wells, **no new access roads** (mirrors the BLM
CE: ≤10 acres, no new roads). Acreage is now disturbance-context-aware, but
**report medians, not maxes** until the LLM pass cleans limit selection.

---

## 8. Storage shows up co-located with solar (Gate 2 evidence)

The `Energy Storage` tech_group is battery *manufacturing*, so it was dropped. A
non-manufacturing scan found 19 storage-deployment mentions elsewhere — 11
co-located with **solar**, 3 transmission, 3 wind, 1 CAES. If CATF wants a
storage CE, the evidence lives in co-located solar+storage FONSIs.

---

## 9. Model selection & cost (LLM extraction pass)

Default **Sonnet 4.6**; escalate to **Opus 4.8** where nuance is missed; Haiku
only if the benchmark confirms it suffices. Run `n06_benchmark_models.py` to pick
the lowest model that clears the accuracy bar (it runs the production prompt
through all three + scores vs. a labeled sample). Cost is negligible — all 293
candidates: Haiku ~$1, Sonnet ~$3, Opus ~$5 (Batch API halves it).
