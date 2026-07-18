# Deliverable 6 — Analysis 1 refactor: five hand-coded categories → a data-driven `tech × action` grid

**Status:** plan (not yet built). Written 2026-06-30.

**The change in one sentence:** stop analyzing five hand-picked action types and instead categorize
**every** clean-energy FONSI by its **technology** (an independent, exhaustive classification we
already have) and its **action** (one cheap LLM label), then read `adopt` / `expand` / `develop`
off **each cell of that grid** — so *develop* (net-new CEs) finally appears in technologies the old
five ignored (e.g. biomass, hydro), with a **codifiability screen** separating genuine CE gaps from
non-physical actions (e.g. battery manufacturing — see §3, §5).

**Why it's better for the client:** the current Analysis 1 ends anticlimactically ("all five are
adopt"). The refactor ends with a **coverage map** — one picture showing, across all clean-energy
technologies, where a CE already exists to *adopt*, where a bound needs to *expand*, and where the
**gaps** are that a new CE should *develop*. That is the actionable story.

> **Guardrails.**
> 1. **Do NOT re-run `03_enrich_llm.py`.** `tech_group`, all sizes/siting/bounded fields, and the
>    Rule-B inputs are already in `fonsi_enrichment.parquet`. The only new LLM work is one cheap
>    **action-sublabel** classify pass on cached summaries (~$1–2).
> 2. **Build on a branch.** The refactor changes headline numbers and every Analysis-1 figure; keep
>    the committed `desktop` report intact until the new one is verified.
> 3. **Streamline.** Add one hero grid and keep the current figures (revisit redundancy after seeing
>    the grid — no cuts committed up front; see §2).
> 4. **Label: `new` in code, "develop" in display.** The verdict engine already emits the string
>    **`new`**; keep it as the internal parquet value (so `07` / `08_create_figures.R` / `deliverable06.qmd` /
>    `qa_deliverable06.py` are not churned across dozens of `verdict == "new"` references) and render
>    it as **"develop"** only in figure legends and report prose. Every client-facing surface reads
>    *develop*; the data stays `new`. This mapping is defined once here and applied in §4.4.

---

## 1. The new Analysis 1 — same 6-step spine, new engine

The report keeps its familiar arc (data → categorize → keep bounded → match → test → rank). Only the
*categorization* and the *ending* change. Before → after, step by step:

| Step | Now (5 hand-coded types) | After (tech × action grid) |
|---|---|---|
| **1. The landscape** | 452 split into candidate / "other" / bounded | **452 split by *technology*** (10 groups, exhaustive — **no "other" black hole**). Lead with the coverage question. |
| **2. Categorize** | LLM forces each FONSI into 5 types or "other" | **Group by `tech_group` → LLM labels the *action*** within each (see the vocab in §4.2) → the **tech × action grid** |
| **3. Keep CE-shaped** | bounded/low-impact subset per type | **unchanged** — same Rule-B filter, per cell |
| **4. Match to CE catalog** | closest CE per type (hardcoded query) | closest CE **per cell** (query derived from the cell's own FONSIs) |
| **5. The three verdicts** | expand test → classify (all → adopt) | **adopt / expand / develop per cell** — *develop* = a recurring cell with **no** existing CE. This is the payoff. |
| **6. Rank** | rank the 4 adopts | rank **all** opportunities across the three verdicts (top-N cells) |
| **Main Finding** | "candidate adoption opportunities" | **the coverage map**: N adopt · M expand · **K develop**, with the develop gaps as the headline |

**Where `develop` comes from:** it is the Step-5 verdict (`new` in code, *develop* in display) for any
cell whose Step-4 CE-match is empty. `07` already emits it — it just never fired because all five
hand-picked cells had CEs. Fill the grid and the no-CE cells appear at the **tech × action-verb**
level (e.g. *Biomass × new_build*). The grid names the gap at that level; the *specific* sub-action
recommendation (a digester-gas CE vs a co-firing CE) comes from the **worked example** (§2.1, table
14), not the grid. A **codifiability screen** keeps non-physical cells (e.g. *Energy Storage ×
manufacturing*) out of the develop shortlist (§3). No separate "Analysis 4."

---

## 2. Figure-by-figure plan

The single biggest move: a **tech × action coverage grid** becomes the new hero exhibit. Every
current figure is **kept and adapted** to the grid's cells; the grid is *added*, not a replacement
(revisit redundancy only after seeing it with real data).

| # | Current figure | Role now | Plan | New form |
|---|---|---|---|---|
| 1 | `fig-d6-outcomes-waffle` | 452 as candidate/other/bounded | **Replace** | **Verdict-composition bar**: 452 by technology, each technology's slice colored adopt/expand/develop/covered. Opens with the coverage story, kills "other". |
| 2 | `fig_d6_action_distribution` (sorting) | FONSIs by 5 types, teal/grey | **Keep, adapt** | per-technology stacked bar (bounded vs broader) across all 10 techs |
| 3 | `fig_d6_keep_bounded` | low-impact siting traits | **Keep** (tech-agnostic) | minor: optionally facet the top techs |
| 4 | `tbl-d6-ce-match` | closest CE per type (4 rows) | **Keep, generalize** | closest CE per **shortlisted cell** (top-N rows) |
| 5 | `fig_d6_ce_match` | match similarity per type | **Keep, adapt** | match similarity per shortlisted cell (top-N) |
| 6 | `fig_d6_sizes` | sizes vs CE limits (3 panels) | **Keep, focus** | same 3 panels, restricted to cells where *expand* is live |
| 7 | `fig_d6_classification` | rank score, 6 factors | **Keep, extend** | now includes **develop** cells; top-N bars |
| 8 | `tbl-d6-rank-sensitivity` | rank under 3 weightings | **Keep** | top-N cells |
| 9 | `tbl-d6-adopt` | adopt opportunities | **Generalize** | the **verdict table**: adopt / expand / develop |
| 10 | `fig-d6-adoption-gap` | evidence weight per adopt opp. | **Keep, generalize** | bounded FONSIs run as full EAs **per opportunity cell**, across all three verdicts (the "prize" behind each) |
| 11 | `fig-d6-states` | geographic reach | **Keep, adapt** | states for the top opportunity cells (reach) |
| 12 | `fig-d6-timeline` | FRA timing | **Keep** (context) | same, over the CE-shaped set |
| 13 | `fig-d6-ce-split` | Transmission CE #17/#19 split (hardcoded to `transmission_upgrade`, `08_create_figures.R:291-315`) | **Keep as a Transmission deep-dive** | re-key from `candidate_category=="transmission_upgrade"` to the **`Transmission__upgrade` cell**; it stays a technology-specific tab, not generalized to all cells |
| 14 | `tbl-d6-transmission-example` | worked example (adopt) | **Keep + add one** | keep the transmission *adopt* example; **add a develop example** (e.g. biomass or storage) so both verdicts are shown concretely |

**Net: keep every current figure, adapted to tech × action, and ADD the hero grid** (+ a second,
*develop* worked example). The grid will visibly overlap a few figures (sorting, ce-match,
adoption-gap, ce-split); we **keep them for now and revisit redundancy only after seeing the grid
with real data** — no cuts committed up front.

### 2.1 The hero: the tech × action coverage grid
- **Rows** = technologies, ordered by frequency (Transmission 149 → CCS 4).
- **Columns** = **action verbs** (`new_build`, `upgrade`, `maintenance`, `decommissioning`,
  `exploration`, `assessment`, …) — the §4.2 vocab. Note the grid is **tech × verb**, *not*
  sub-action: a cell like *Biomass × new_build* aggregates biogas, co-firing, biorefinery, etc.
- **Cell size/number** = CE-shaped FONSI count; **cell color = verdict**, with **five** states, not
  four: `adopt` / `expand` / **`develop` (`verdict==new` ∧ `is_codifiable`)** / **`develop-excluded`
  (`verdict==new` ∧ `is_codifiable==False`, e.g. `manufacturing`)** / `already-covered`; blank = none
  observed. `is_codifiable` is the deterministic field written by `10_action_label.py` (§4 item 1). The
  `develop-excluded` color is essential so a manufacturing-dominated cell (e.g. *Energy Storage ×
  manufacturing*, all 21 Storage FONSIs) is **not** shown to the client as a CE opportunity.
- **What the grid does and does not show:** it tells you *which tech × verb cells recur and their CE
  status* at a glance — the coverage map. It does **not** resolve the *specific* sub-action CE
  (digester-gas vs co-firing); those come from the worked examples (table 14). State this in the
  report so the "one look" claim isn't overread.
- Built in `08_create_figures.R` as a `geom_tile` heatmap (≤10 rows × ≤11 cols).
- Overlaps figures #1, #2, #5, #10, #13 (kept for now, per client direction); revisit redundancy once
  it renders with real data.

---

## 3. The develop wing — and superseding the net-new plan

**Post-refactor develop pool (defined explicitly).** `develop` candidates are **the grid cells whose
Step-4 CE-match is empty** (`verdict == new`) and that pass the codifiability screen — *not* the old
`action_category == "other"` (314) pool. The tech × action grid **is** the categorizer; there is no
separate clustering universe.

**This refactor SUPERSEDES `deliverable06_newCEs.md`.** That plan (marked READY TO IMPLEMENT, round-3
review 2026-06-30) built its whole pipeline — HDBSCAN clustering, eligibility ranking, and the QA
reconciliation `clustered + noise == 314` — on the 314 "other" pool, which **this refactor eliminates**
(every FONSI, including the 314, now lands in a grid cell). To avoid two contradictory ready-to-build
plans:
- **Retire** the net-new plan's front-end: the HDBSCAN clustering, the 314-universe definition, and
  the `== 314` QA. Nothing there was built, so no code is lost.
- **Absorb** its still-valid back-end onto the grid's `develop` cells: the **CE-retrieval gate**
  (nearest CEs → novelty), the **`codifiability`** screen (physical action vs funding/manufacturing),
  the **bounded gate**, **recurrence × spread ranking**, and the **"candidate, pending CE-catalog
  review"** framing.
- **Action item (plan edit, not a script):** mark `deliverable06_newCEs.md` as *Superseded by
  `deliverable06_refactor.md` — back-end absorbed, clustering front-end retired.* Optionally keep
  clustering only as *within-cell* refinement for a big heterogeneous cell (e.g. *Biomass ×
  new_build*).

---

## 4. Data-layer changes (the code)

Localized, but with real **ordering, label, and QA seams** to get right (round-1 review).

0. **Bump the taxonomy version first.** `candidates.py:16` stamps `TAXONOMY_VERSION = "d6_v2_2"` into
   every output parquet and the QA check. Bump to **`"d6_v3_0"`** before any run so old and new outputs
   can never be confused (especially on the branch).
1. **Action sublabel — new pass `10_action_label.py`** (committing to one name; the `10_` slot is freed
   because the net-new plan is superseded, §3). Runs **early — right after `01`, before `04`** — so
   cells exist before CE-match. Cached on `action_summary`, ~$1–2. **Output file (the `10→09`
   interface):** writes `data/analysis/deliverable06/fonsi_action_labels.parquet` with columns
   `(project_id, action, is_codifiable, actionlabel_run_at, actionlabel_llm_run_at,
   actionlabel_prompt_version)` — a **new** file; it does **not** mutate `fonsi_enrichment.parquet`
   (an external prereq — would break the `_run.py:68` enrichment-present guard). `09` **joins it onto
   the enrichment by `project_id`** before building the cell id. **`is_codifiable: bool` is derived
   deterministically from the verb (no extra LLM):** `False` for `manufacturing` and
   `land_or_row_authorization` (non-physical), `True` for the physical-action verbs. This is the field
   that powers the grid's `develop-excluded` color, the develop-shortlist gate in `07`, and the QA
   check (§4.6). Vocab in §4.2. The pass also does the work the `candidates.py` subtypes used to do —
   distinguishing a transmission *upgrade* from *new_build* / *maintenance* / *substation*.
2. **Categorizer — `09_wire_enrichment.py`:** drop the `if cat not in CANDS: continue` gate (09:112) so
   **all 452** are kept; set `candidate_category = f"{tech_group}__{action}"`, carry `tech_group` +
   `action` + `is_codifiable` (joined from `10` by `project_id`) into `candidate_facts.parquet`.
   **Rule-B, corrected:** the transmission shape-gate must fire on
   **`tech_group=="Transmission"` AND `action=="upgrade"` AND `within_existing_row==True`** — *not* on
   `tech_group=="Transmission"` alone. Why: 84 of the 149 Transmission FONSIs are currently "other"
   (vegetation maintenance, new lines, substations, distribution) — 25 with `within_existing_row==False`
   and 22 null. A bare tech-level gate would pull those into the CE-shaped pool and inflate the count;
   the `action` verb re-supplies the subtype exclusion `candidates.py` used to give. Non-transmission
   cells gate on the bounded judgment only (unchanged).
3. **CE-match + ordering — `04_base_rates_and_ce.py` / `_run.py`.** Replace the hardcoded `QUERY_TERMS`
   (04:53) with a query derived from each cell's member summaries; key `candidate_ce_comparison` on the
   cell id. **Resolve the 04-before-09 circularity concretely:** the cell id needs `action`, so change
   the `_run.py` order to **`01 → 10_action_label → 02 → 03 → 04 → 05 → 06 → 09 → 07 → 08`** (sublabel
   before 04). Keep `04`'s base-rate half on the regex corpus if it still needs it; point only its
   CE-crosswalk half at the cell id. **Base rates are descriptive-only:** `07` does **not** read
   `candidate_base_rates` into its verdict or rank (the score uses `n_focus` / `n_agencies` /
   `n_states` / `mit_share` / `has_limits` / `role`), so the old-5-keyed base-rate half does **not**
   silently corrupt verdicts — but if the base-rate *figure* is kept, regroup that half by `tech_group`
   (exhaustive across all 452); otherwise leave it out of the cell pipeline. Either way `07` is
   unaffected. **Budget the `_run.py` surgery** (~1–2 h) in the estimate.
4. **Verdict — `07_classify_and_rank.py`: three changes.** (a) replace the `CAND_ORDER` filter
   (07:48, 07:96) with "all cells present in facts," sorted by tech_group then verdict; (b) replace the
   `candidate_label` lookup that piggybacks off `CAND_ORDER` (07:104) with a per-cell label
   (`"{Tech} — {action}"`); (c) **carry `is_codifiable` from `candidate_facts` into
   `candidate_verdicts.parquet`, and filter the client develop shortlist `d6_new.csv` (07:213) to
   `is_codifiable==True`** — `is_codifiable==False` cells stay in `candidate_verdicts` for the grid
   coloring but are excluded from the ranked output and worked examples. The new/expand/adopt logic is
   otherwise unchanged; the emitted string stays **`new`**, shown as *develop* only in display
   (guardrail 4). **Display recode (R side):** in `08_create_figures.R`,
   add `verdict_display = recode(verdict, new = "develop")` and use `verdict_display` in every legend,
   label, and prose reference, so no figure ships the raw string `new`.
5. **Bounds/mitigation + R figures — `05`, `09` summary, `08_create_figures.R`.** `05` and the `09` mitigation
   summary already `groupby("candidate_category")` → they follow once the key is the cell id. But
   `08_create_figures.R` also has **transmission-hardcoded** figures (`fig_d6_states`, `fig_d6_ce_split`,
   ~08:265-315) that must be re-keyed to the `Transmission__upgrade` cell (§2 rows 11, 13).
6. **QA — `qa_deliverable06.py` (update FIRST, before any test run).** It hardcodes the five: `CANDS`
   (5), `EXPECT` (5 CE-shaped counts), and `assert n_shaped == 63` (qa:39-55) — all fail the instant
   the cell ids change. Replace with grid-level assertions: `sum(cells) == 452`; every cell has a
   verdict; every shortlisted cell has a CE-match row; **no `is_codifiable==False` cell in the client
   develop shortlist** (catches `manufacturing` / `land_or_row_authorization`); per-cell CE-shaped
   counts reconcile to the bounded tally.

### 4.2 Action vocabulary (the `10_action_label.py` controlled set)
Each verb separates a distinct CE profile: `new_build` · `upgrade` (modify/rebuild/reconductor
existing) · `maintenance` (repair, vegetation/ROW upkeep) · `decommissioning` (removal/retirement) ·
`exploration` (resource-investigation drilling) · `assessment` (surveys, met towers, site
characterization) · `research_or_demonstration` (pilot / R&D facility) · `manufacturing`
(component/battery factory — **flagged non-codifiable**) · `interconnection` (gen-tie / grid tap) ·
`land_or_row_authorization` (grant/renewal/amendment — administrative, non-physical) · `other`.
The first five were implicit in the old five categories; `decommissioning`, `research_or_demonstration`,
and `manufacturing` are recurring clean-energy actions the old verbs miss (removal is a strong CE
candidate; R&D/demo is heavily represented; manufacturing is mostly *non*-codifiable and must be
isolated so it doesn't pollute `new_build` or a develop cell). `interconnection` and
`land_or_row_authorization` are also included; at build, fold either into `new_build` / `other` if it
proves rare (< R FONSIs), logging the merge.

---

## 5. Issues anticipated — and how each is headed off

1. **Multi-tag projects (82% carry >1 project_type).** `tech_group` is **already single-valued**
   (D3's primary-technology rule). Inherit it; let the action-sublabel pass confirm the primary
   `(tech, action)` from the document. Multi-membership (one FONSI in two cells) is **deferred** — it
   breaks distinct-project counts. Log the multi-tag rate + any overrides.
2. **Small/vague cells** (CCS 4, Hydro 7; **"Other Clean" 67**). Report all cells; **shortlist** only
   `n_ce_shaped ≥ R` (5 main / 3 exploratory). Decompose "Other Clean" via the action sublabel or flag
   it as a residual. `log()` everything dropped from the shortlist.
3. **Figures for N cells.** The grid (`geom_tile`, ≤10 × ≤11 cols; expect 5–7 non-empty columns in
   practice) is legible where 25 bars aren't; the
   per-cell bar/box figures (classification, sizes) use **top-N** cells; the verdict table facets by
   `tech_group`. This is the bulk of the figure work — budget for it.
4. **Rule-B transmission gate** — re-key on **`tech_group=="Transmission"` AND `action=="upgrade"` AND
   `within_existing_row==True`** (NOT tech alone). The 84 currently-"other" Transmission FONSIs
   (maintenance / new-line / substation) must be excluded via the `action` verb, which replaces the
   `candidates.py` subtype filter that no longer runs (§4 item 2).
5. **Scope: Nuclear (33), CCS (4)** — **kept in the featured analysis** (per client direction). They
   are recurring clean-energy technologies and get the same adopt/expand/develop treatment as the rest.
6. **Headline numbers change** (the committed "63 CE-shaped," the 5-row verdict table). Expect it;
   that's the point. Do it on a branch and re-verify every inline `r n_*` value.
7. **The render/verify loop is slow** (~2–3 min/pass) and the messy cells only surface once you run
   with real N — budget iteration time.
8. **Reproducibility:** new outputs get `run_at`; the sublabel pass gets the two-timestamp audit
   convention + a prompt version.

---

## 6. Sequencing & effort

- **Optional fast prototype first (~2–3 h, ~$0):** a self-contained script that produces the per-cell
  **verdict table + the coverage grid** from the enrichment alone — the *numbers*, without touching
  figures or report. Good for validating the direction (and for a presentation) before the full build.
- **Full refactor (~2–2.5 days, ~$1–2), on a branch:**
  0. **Prep (do first):** bump `TAXONOMY_VERSION` → `d6_v3_0`; rewrite `qa_deliverable06.py` to
     grid-level assertions (else every test run blocks); mark `deliverable06_newCEs.md` superseded.
     *(~1–2 h)*
  1. `10_action_label.py` + corrected Rule-B + categorizer + **`_run.py` reorder** (sublabel before 04)
     *(~½ day + ~1–2 h for the orchestrator)*
  2. Per-cell CE-match + verdict engine (both `07` changes) *(~½ day)*
  3. Figures — hero grid (with the 5-state color incl. `develop-excluded`) first, then the top-N /
     re-keyed reworks incl. the transmission-hardcoded ones *(~¾ day)*
  4. Report prose (Steps 1–6 + coverage-map Main Finding + develop section, with the grid-granularity
     caveat) + render/verify *(~½ day)*

**Bottom line:** the verdict logic is small; the time is (a) the figures + Analysis-1 narrative that are
hand-fitted to five categories, and (b) the ordering/label/QA seams above. The hero grid keeps the
figure work bounded and the result digestible — but the prep step (QA + version + supersede) must come
first or the pipeline won't run.
