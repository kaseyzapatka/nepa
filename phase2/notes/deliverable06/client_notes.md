# Deliverable 06 — Client Notes (methodology, findings, open questions)

Working note consolidating how we searched for categorical-exclusion (CE)
candidates in prior FONSIs, what we found, and what we should confirm with CATF
before going further. Companion to `findings_for_report.md` (report-ready facts)
and `phase2/architecture/deliverables/deliverable06.md` (the durable pipeline doc).

**Status:** first pass. Extraction is deterministic (the LLM pass, manual
validation, and the `.qmd` report are still ahead). **Compiled:** 2026-06-17.

---

## 1. Search methodology

### What the deliverable is looking for
Recurring *clean-energy action categories* in prior Environmental Assessments
that ended in a **Finding of No Significant Impact (FONSI)** — because a FONSI is
the evidence a CE needs: documented proof that a class of action reliably has no
significant environmental impact. The unit of interest is not "a technology" but
**a technology + a constraint that keeps its impact small enough to be CE-eligible**
(e.g., transmission *upgrades within an existing right-of-way*, not transmission
generally). That constraint is the boundary of the potential CE.

### v1 approach (replaced): the 14-archetype metadata classifier
The original method classified the **entire BLM/DOE clean-energy universe**
(~20,000 projects across CE + EA + EIS review types) into **14 technology
archetypes** (facility upgrade, transmission, solar, wind, geothermal
exploration, energy storage, pipeline, hydrogen, carbon management, etc.).

- "Metadata classifier" = it assigned archetypes by **keyword-matching each
  project's *metadata*** (`project_type` tags + title), **not by reading the
  documents**. ~42,700 of ~43,500 assignments were metadata-only keyword matches;
  only ~800 were text-supported.
- It then scored all 14 on a 0–2 rubric and assigned recommendation tiers.

**Why it was abandoned:** keyword-matching metadata is too crude. It mislabeled
**10,865 clean-energy projects as "oil and gas,"** every project landed in ~2
archetypes (double-counting), the 0–2 scores came out nearly identical (10 of 14
the same), and *everything* tiered "review." It answered the easy question
loudly (*what technology is this?*) and never addressed the hard one (*is this a
low-impact, bounded action that could be a CE?*).

### v2 approach (current): narrow-first funnel
The key idea: for clean energy the technology types are **already known** — no
classifier needed to discover that solar and transmission exist. The hard part is
isolating, *within* each technology, the slice that is CE-shaped. So the search is
a funnel that reads the actual documents for the slice that matters.

| Step | What we do |
|---|---|
| 1. Define corpus | Start from the **452 clean-energy EA-source FONSI projects** (CE-/EIS-track projects kept only as context). |
| 2. Technology recurrence | Read off FONSI counts per technology from the dataset's own `tech_group` field (no classifier). |
| 3. Choose candidates | Combine **recurrence** + **domain knowledge** of what's CE-shaped + **regulatory signal** (CEQ Apr-2026 guidance; BLM geothermal-exploration CE as the model). |
| 4. Subtype split | Within each technology, rule-based subtyping (reading project text) separates the CE-shaped slice from the high-impact slices. |
| 5. Facts + crosswalk | Extract bounding limits (acres/miles/MW/wells), siting constraints (no new roads, within existing ROW), mitigation dependence; cross-reference every candidate against the CE-Explorer database of existing federal CEs. |

**Technology recurrence (step 2), among 452 clean FONSIs:**
> Transmission 149 · Other Clean 67 · Wind 62 · Solar 61 · Nuclear 33 · Biomass 27 · Energy Storage 21 · Geothermal 21 · Hydropower 7 · CCS 4

**Candidate refinement (step 3):** initial set was transmission, geothermal,
solar, energy storage, temporary resource assessment, + wind (contrast). Data
checks then dropped **energy storage** (it's battery *manufacturing*, not grid
storage) and demoted **temporary resource assessment** (already CE-covered, thin).

**Subtype split (step 4) — isolating the CE-shaped slice:**

| Technology | CE-shaped slice (profiled) | Excluded slices |
|---|---:|---|
| Transmission (149) | `standalone_upgrade` = **48** | new_line, gen_bundled, substation, vegetation mgmt, telecom, off-scope |
| Geothermal (21) | `exploration` = **7** | development (power plant) |
| Solar (61) | `disturbed_developed` = **8** | greenfield_utility, gen_tie, manufacturing |

### One-line contrast
- **v1:** classify the *whole universe* into 14 *technology* buckets by *metadata keywords*, then score.
- **v2:** start from *FONSIs* → read off *technology recurrence* → isolate the *CE-shaped subtype* by reading projects → extract *bounding limits* → check against *existing CEs*.

### Tooling / cost note
Local tools only (spaCy + `all-MiniLM` embeddings); the LLM extraction pass is
gated and would cost ~$1–5 total. Default model **Sonnet 4.6**, escalate to
**Opus 4.8**; `n06_benchmark_models.py` benchmarks all three to pick the lowest
model that clears the accuracy bar.

---

## 2. Main findings

### The funnel
| Stage | N |
|---|---:|
| Clean EA-source FONSI projects | 452 |
| In a candidate category | 293 |
| CE-shaped ("profile") subset | 63 |

### The honest headline
**The evidence supports *adopting / expanding / validating* existing categorical
exclusions — it does not surface a brand-new CE category.** Every substantive
candidate already maps to a real federal CE.

| Candidate | CE-shaped N | Evidence strength | Already a CE? |
|---|---:|---|---|
| **Transmission upgrades in existing ROW** | 48 | Strong (most FONSIs) | **Yes** — TVA #17 ("routine modification/minor upgrade of existing transmission"); routine-upgrade CEs are common, so likely covered at several agencies |
| **Geothermal exploration** | 7 | Weak (small N) — illustrative | **Yes** — BLM geothermal-exploration CE; DOE B3.1 |
| **Solar on disturbed/developed land** | 8 | Weak (small N) — illustrative | **Yes** — DOE B5.16 ("solar PV within a previously disturbed or developed area") |

**Key implication:** none of the three is *both* high-evidence *and* a genuinely
new opportunity. The strongest-evidence candidate (transmission) is also the most
likely already covered. This is a real finding about the regulatory landscape:
the obvious low-impact clean-energy actions have mostly already been excluded.

### What "maps onto an existing CE" means
For solar, DOE B5.16 already excludes "solar PV within a previously disturbed or
developed area." Our disturbed-site solar FONSIs match it almost exactly — so the
finding is **validation** (real projects in this class got FONSIs) plus an
**adoption/expansion lever** (other agencies adopting it; or widening its bounds).
Same logic for transmission (TVA #17) and geothermal (BLM CE / DOE B3.1).

### Not findings
- **Temporary resource assessment** — already overwhelmingly a CE: **2,317 CE vs
  6 EA vs 7 EIS** (only 2 FONSIs); DOE B3.1 covers it. A possible adoption target,
  not a discovery.
- **Wind** — kept as a *contrast*: the existing wind CE (DOE B5.18) only covers
  ≤2 turbines; utility wind has case-specific wildlife impacts and needs
  case-by-case review.

### Standing caveats
- Extraction is **deterministic** (LLM pass not yet run); use medians, not maxes.
- Two of three Ns are **thin** (7 and 8).
- CE crosswalk matches are **unverified** (a ranking aid, not a coverage decision).
- Mitigation-dependence is a **preliminary** supporting signal, not a headline.
- NEPATEC's "Clean + Transmission" tag is **noisy** (a nuclear demo, gas plant,
  mining project, and an appliance-efficiency standard leaked into the corpus and
  are flagged off-scope).

---

## 3. Client feedback worth getting before going further

In priority order — these change what we build next.

1. **Framing — adopt/expand vs. brand-new CEs?** The strongest candidates already
   have CEs somewhere. Is CATF's goal *adoption/expansion/validation of existing
   CEs* (what this corpus best supports) or specifically *net-new CE categories*?
   If net-new, we should **invert the search** — rank recurring FONSI categories
   by how *poorly* they match any existing CE, and look in the buckets we set
   aside (Other Clean, biomass, hydrogen, etc.). **This is the decision
   everything else hinges on.**

2. **Candidate priorities.** Are transmission/geothermal/solar the right targets,
   or does CATF want others elevated (storage, offshore, hydrogen)? Their priority
   list should drive the deep pass.

3. **Depth / definition of "done."** Is this a **triage shortlist** ("here are the
   categories worth pursuing") or do they need **substantiation dossiers**
   (rulemaking/litigation-grade)? That decides whether we run the LLM pass and
   manual validation, or stop at the shortlist.

4. **Small-N tolerance.** Geothermal (7) and solar (8) are thin. Firm them up
   (widen the corpus beyond EA-source FONSIs — combined EA/FONSI docs, CE-track
   text), or are illustrative examples enough?

5. **Target agencies.** The adoption story is "agency X should adopt agency Y's
   CE." Which agencies is CATF actually trying to move (DOE? BLM? FERC? a specific
   Power Marketing Administration)? That shapes which existing CE we benchmark.

6. **Bounding limits.** A CE *is* its limits (e.g. ≤10 acres, no new roads).
   Derive proposed bounds from the FONSI distribution, or test CATF's target
   thresholds?

7. **Scope of "clean."** Transmission is dual-use grid infrastructure — a
   transmission-upgrade CE helps clean energy but isn't clean-only. Confirm
   whether dual-use/grid actions are in scope.

**Recommendation:** get answers to **#1, #2, #3 first** — they determine whether
the next move is the LLM extraction pass, a corpus-widening effort, or a pivot to
hunting net-new categories.
