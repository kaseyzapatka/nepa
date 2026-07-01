# Deliverable 6 — Analysis 4: Net-new CE discovery from the "other" residual

**Status:** plan, revised after Codex review (not yet built).
**Author:** Claude. **Reviewed:** Codex, 2026-06-30 (`deliverable06_newCEs_feedback.md`).
**Scope:** a new sub-analysis of D6 that examines the FONSIs the candidate pipeline set aside, to
surface **candidate net-new** Categorical Exclusion (CE) themes — recurring low-impact clean-energy
action types that appear to have *no* existing CE, **pending a CE-catalog review gate**.

> **What changed in this revision (Codex findings applied):**
> 1. **"No existing CE" is now a pipeline gate, not a final spot-check** — an automatic CE-catalog
>    retrieval + LLM screen runs before any `net_new_verdict` is final (§4.5–4.6, §7). Client-facing
>    language is **"candidate net-new … pending CE-catalog review"** throughout.
> 2. **Cluster all 314 "other", not the bounded 217** — boundedness becomes an eligibility/ranking
>    field, so `bounded_share` is meaningful and false/unknown-bounded context is preserved (§4.1, §4.7).
> 3. **Composite `cluster_text` with deterministic fallbacks**, not `potential_ce_theme` alone —
>    guards against pass-1 artifacts in the 87 rows reclassified into "other" (§4.2).
> 4. **`sklearn.cluster.HDBSCAN` (already in sklearn 1.8), no new dependency**; stable *hashed*
>    cluster ids; a small stability grid (§4.4, §5, §10).
> 5. **Funding gets an explicit three-state codifiability field**, orthogonal to novelty (§4.6, §9.1).
> 6. **Full audit fields + timestamps + a noise/one-off output** on all tables (§7).
> 7. **Terminology:** "uncategorized **FONSI** residual", never "uncategorized CEs" (§9, throughout).
> 8. **Explicit pipeline placement + fail-loud prerequisites + QA reconciliation** (§6, §8).

---

## 1. The question

Of the clean-energy EA→FONSIs that are **not** one of the 5 existing-CE candidate types
(transmission upgrade, solar, geothermal exploration, temporary assessment, onshore wind),
**which recurring, low-impact action types appear to have no existing CE** — i.e., are **candidate
net-new CEs** (pending CE-catalog review) an agency could write from scratch?

## 2. Why this analysis is needed (net-new is empty *by design*, not by finding)

D6's three verdicts are **develop** (net-new: a recurring action with no existing CE), **expand**
(exceeds an existing CE's bound), and **adopt** (an existing CE at another agency covers it). The
5 candidate types were hand-picked *because they already recur and already have CEs* — they were
built to test adopt/expand. The LLM classifier sorts every FONSI into one of those 5 or **"other."**

Consequence: all 5 candidate types resolve to **adopt**, and **develop/net-new is empty — not
because no net-new actions exist, but because the design only examined types that already have
CEs.** A net-new CE is, by definition, a recurring action type with no existing CE; such types
*cannot* be among the 5, so they sit unexamined in the **"other"** residual. This analysis examines it.

**Framing (Codex finding 7):** these are **EA-to-FONSI actions that might support a future CE** — an
"uncategorized FONSI residual", *not* "uncategorized CEs". Clustering does not establish a CE; it
surfaces **candidate themes pending CE-catalog and eCFR review**.

## 3. Inputs (what we already have — most of the groundwork is done)

Source: `data/analysis/deliverable06/fonsi_enrichment.parquet` (the corrected, Stage-2-classified
enrichment of all **452** clean FONSIs). The "other" residual and the relevant pre-extracted fields:

| Quantity | Count | Notes |
|---|---|---|
| `action_category == "other"` (non-empty summary) | **314** | the residual the candidate pipeline set aside — **the clustering universe** |
| of those, `is_bounded_low_impact == TRUE` | **217** | eligible CE-shaped evidence (low-impact) |
| `is_bounded_low_impact == FALSE` | **84** | kept as contrary context (do **not** drop — Codex finding 2) |
| `is_bounded_low_impact` null/unknown | **13** | surfaced, not silently bounded |
| of the 314, `potential_ce_theme` non-null | **266** | pass-1 named a plausible theme |
| of the 314, pass-1 category was **not** "other" | **87** | `transmission_upgrade` 37 · `wind_onshore` 26 · `solar` 16 · `geothermal_exploration` 8 — reclassified into "other" by Stage 2; may carry weak/missing pass-1 theme fields (Codex finding 3) |

Already-extracted per-FONSI fields we reuse (verified present in the parquet; no new extraction
needed for naming):
- `potential_ce_theme` — short LLM-named plausible net-new theme.
- `action_label_freeform`, `action_category_other` — normalized / freeform action labels.
- `why_not_current_candidate` — one phrase on why it's outside the 5 candidate types.
- `key_activities`, `bounded_rationale` — activity detail and boundedness reasoning.
- `is_bounded_low_impact`, `action_summary`, `lead_agency_harmonized`, `project_state`,
  `project_title`, `tech_group`.
- Stage-2 provenance: `action_category_pass1`, `classification_confidence`, `classification_parse_ok`,
  `classification_prompt_version`; extraction provenance: `extraction_confidence`.

**Themes already visible in the raw fields (pre-clustering):** biomass / biogas / biorefinery ·
combined heat & power at industrial sites · EV-battery manufacturing · landfill- & blast-furnace-gas
power · renewable-energy research labs · weed / vegetation management · small wind + storage demos ·
transmission access roads / interconnection taps. The task is to **aggregate the scattered free-text
themes into sized, recurring clusters**, retrieve nearest existing CEs, then screen and rank.

## 4. Method

### 4.1 Scope the pool — **all 314**, boundedness as an eligibility field (Codex finding 2)
Cluster **all 314** "other" rows with non-empty summaries. Do **not** pre-filter to the bounded 217:
clustering the full residual keeps the 84 false-bounded and 13 unknown rows as *contrary context*
(a cluster of 4 bounded + 25 large projects is not the same opportunity as 4 bounded + 0 contrary),
and it makes `bounded_share` a real ranking signal instead of a constant 1.0. Boundedness enters at
**ranking/eligibility** (§4.7), not as the clustering universe. The 217 bounded rows remain the
"eligible CE-shaped evidence".

### 4.2 Composite clustering text with deterministic fallbacks (Codex finding 3)
Do **not** cluster on `potential_ce_theme` alone — it is a pass-1 artifact, and the 87 reclassified
rows may have weak/empty theme fields. Build a deterministic `cluster_text` per FONSI by
concatenating, in order and skipping empties:

`potential_ce_theme` → `action_label_freeform` → `action_category_other` →
`why_not_current_candidate` → `key_activities` → trimmed `action_summary`.

**QA gate:** every one of the 314 input rows must have non-empty `cluster_text`
(one bounded row is known to lack a theme and needs the fallback). Persist the exact assembled
`cluster_text` and a `cluster_input_sha256` for reproducibility.

### 4.3 Embed
Embed each `cluster_text` with **`sentence-transformers/all-MiniLM-L6-v2`** via the existing
`embeddings.embed()` helper (the model D6 already uses in `04`/`06` and D3). 384-d, L2-normalized,
deterministic. Reuse the on-disk embedding cache pattern from `06` (sha256 of text+model).

### 4.4 Cluster — **`sklearn.cluster.HDBSCAN`** (no new dependency; Codex finding 4)
`sklearn` 1.8 ships `sklearn.cluster.HDBSCAN`, so the recommended path needs **no new package**
(the third-party `hdbscan` is *not* required). On L2-normalized vectors, euclidean is a sound cosine
proxy. HDBSCAN:
- **auto-detects the number of clusters** (no fixed *k*), and
- **labels low-density points as noise (`-1`)** — keep the *dense, recurring* themes, isolate one-offs.

Details:
- `min_cluster_size` ≈ 3–5 (cluster *detection* granularity — decoupled from the shortlist floor R,
  which now applies to bounded evidence at ranking, §4.7); `min_samples` tuned for stability.
- **Stability grid** over `min_cluster_size` × `min_samples`: record cluster count, noise share, and
  top-cluster membership stability; pick and **document** the chosen params.
- **Stable cluster ids (not raw HDBSCAN integers):** `cluster_id = "netnew_" + sha1(sorted member
  project_ids + cluster_method + cluster_params + clustering_version)[:8]`, so ids survive re-runs
  and param bumps are visible. Raw integer labels are kept only as a transient `hdbscan_label`.
- **Fallback (zero-dep):** `sklearn.AgglomerativeClustering` (distance-threshold, cosine) + small-
  cluster filter, matching `06`'s sklearn usage.

Output: `cluster_id` per FONSI (`-1`/`noise` bucket retained), plus a membership table.

### 4.5 Retrieve nearest existing CEs — **the net-new gate** (Codex finding 1, HIGH)
Before any cluster is called net-new, retrieve its nearest existing CEs automatically. For each
**coherent** cluster, embed the cluster label + a few representative member summaries and cosine-
retrieve the nearest CE descriptions from `ce_source.load_ce_catalog()` /
`ce_landscape_ces.parquet` (reuse `06`'s cached CE embeddings; `ce_landscape_ces` already carries
`ce_id`, `agency_unit`, `ce_description`, and a `nearest_xagency_ce` precedent). Write
`net_new_ce_matches` (below) with the top-k nearest `ce_id`s, agencies, descriptions, and cosine
scores. These nearest CEs are **shown to the LLM screen** (§4.6) and drive the novelty verdict. A
theme with a close existing CE is **adopt/expand**, not net-new.

### 4.6 Name + screen each cluster (LLM — one cheap cached pass)
For **each coherent cluster** (not each FONSI — ~20–40 calls total), send the member
`cluster_text`s + representative `action_summary`s **+ the nearest existing CEs from §4.5** to Claude
Sonnet (pinned snapshot, temp 0, tool-use schema, cached on cluster-content hash + `NETNEW_PROMPT_VERSION`
— mirroring `03 --stage classify`). Return **two orthogonal judgments** (my refinement of Codex
findings 1 + 5, which keeps novelty and codifiability from being conflated in one enum):

- `theme_name` — concise recurring-action name.
- `coherent` (bool) — is the cluster one coherent action type?
- **`net_new_verdict`** (novelty, given the nearest CEs) ∈
  `plausible_net_new` / `close_existing_ce_review` / `belongs_to_existing_type` / `not_coherent`.
- **`codifiability`** (funding treatment, Codex finding 5) ∈
  `physical_action_codifiable` / `funded_physical_action_codifiable` / `funding_only_not_codifiable`
  / `physical_action_unknown`. A CE codifies a *physical* action, not financing; funding-only
  clusters are excluded from the shortlist, and any funded-physical evidence is reported explicitly
  as a CE **for the physical activity, not the financing mechanism**.
- `is_bounded_low_impact` (cluster-level judgment, cross-checked against the row-level evidence).
- `matched_ce_ids` (which of the retrieved nearest CEs, if any, it considers a real match) + `rationale`.

For rows whose `action_category_pass1 != "other"` (the 87), the screen must **explicitly** decide
`belongs_to_existing_type` vs genuinely residual.

### 4.7 Rank — eligibility on bounded evidence (Codex finding 2)
Compute cluster-level fields: `n_total`, `n_bounded_true`, `n_bounded_false`, `n_bounded_unknown`,
`bounded_share_known` (= true / (true+false)), `n_agencies`, `n_states`. **Eligible** =
`coherent` ∧ `net_new_verdict == plausible_net_new` ∧ `codifiability ∈
{physical_action_codifiable, funded_physical_action_codifiable}` ∧ `n_bounded_true >= R`
∧ `bounded_share_known >= τ` ∧ no close existing-CE match. Rank eligible clusters by
`recurrence × spread × boundedness` = `f(n_bounded_true) · g(n_agencies, n_states) · bounded_share_known`.
**R = 5** for the main client shortlist; **R = 3** allowed only for multi-agency/multi-state, highly
coherent themes, labeled **exploratory** (Codex answer 5). Unknown-bounded counts are surfaced, never
hidden.

### 4.8 Verify the top themes (eCFR — final human confirmation)
The §4.5 gate makes novelty a first-class pipeline output; §4.8 is the **final** human confirmation
for the shortlist only — confirm "no existing CE" against the CE catalog / eCFR, the same human step
as the adopt worksheet (`ce_verification.py`). Records land in `net_new_ce_matches` with
`manual_ce_review_status`.

## 5. Methodology decision — clustering algorithm (updated after review)

| Option | Auto-*k*? | Isolates one-offs? | New deps | Consistency w/ D6 | Verdict |
|---|---|---|---|---|---|
| **`sklearn.cluster.HDBSCAN`** (+ our embeddings + LLM naming) | yes | **yes (noise = `-1`)** | **none — in sklearn 1.8** | embeddings already standard | **recommended — best task fit, no install** |
| sklearn `AgglomerativeClustering` (distance-threshold, cosine) + filter small clusters | yes | via post-filter | none | matches `06` sklearn usage | strong no-dep fallback |
| third-party `hdbscan` package | yes | yes | 1 (unnecessary) | — | **rejected — sklearn already provides HDBSCAN** |
| sklearn `KMeans` (06's choice) | no (fixed *k*) | no | none | exact `06` precedent | rejected — forces *k*, absorbs one-offs |
| full **BERTopic** | yes | yes | 3 (`bertopic`+`umap`+`hdbscan`) | not used in D6 | rejected — corpus is small, we already have embeddings, and the LLM names clusters better than c-TF-IDF |
| **LLM-only grouping** (no embeddings) | n/a | n/a | none | — | rejected for *grouping* — less reproducible than embedding clustering; use the LLM for naming + viability screen |

**Recommendation:** embeddings (`all-MiniLM-L6-v2`) → **`sklearn.cluster.HDBSCAN`** → nearest-CE
retrieval → LLM naming/screening. This is the lean core of BERTopic (skip UMAP/c-TF-IDF; n≈314 is
small and the LLM names better), HDBSCAN's noise model best matches "recurring vs one-off", and it
runs in the existing `nepa` env with **no new dependency**.

## 6. Implementation

New script **`code/deliverable06/10_net_new.py`** (08 is R analysis, 09 wires enrichment; 10 is free):
- reads `fonsi_enrichment.parquet`; takes **all 314** "other" rows with summaries.
- assembles `cluster_text` with the §4.2 fallback ladder; QA non-empty; stamps `cluster_input_sha256`.
- `embeddings.embed()` for vectors (deterministic, pinned model, cached).
- **`sklearn.cluster.HDBSCAN`** with fixed, documented params; **hashed** stable `cluster_id`;
  Agglomerative fallback behind a flag.
- §4.5 nearest-CE retrieval from the CE catalog (reuse `06` embeddings).
- LLM cluster naming/screening via a new `prompts.build_netnew_prompt()` + tool schema, called
  through an `enrich_lib.call_*`-style helper, **cached** on cluster-content hash +
  `NETNEW_PROMPT_VERSION` (re-runs free; committed output canonical).
- **Audit stamps (project convention):** `netnew_extraction_run_at` on **all** rows at build time;
  `netnew_llm_run_at` per-cluster only when the LLM call succeeds (else `""`).
- writes the outputs in §7. Add a `--dry-run` (cost preview, no key) like `03`.

**Pipeline integration (Codex finding 8):**
- Runs **after `03_enrich_llm.py --stage classify`** (needs corrected `action_category`) **and after
  `06_ce_landscape.py`** (needs CE embeddings for §4.5 retrieval).
- Added to `_run.py` **only after** the classify output is stable.
- **Fail loudly** if `classification_parse_ok` is missing, if classified rows are not stamped with the
  expected `classification_prompt_version`, or if the CE catalog / `06` embeddings are absent.

## 7. Outputs / deliverables
- `data/analysis/deliverable06/net_new_themes.parquet` — one row per cluster:
  `cluster_id`, `theme_name`, `coherent`, `net_new_verdict`, `codifiability`, `n_total`,
  `n_bounded_true`, `n_bounded_false`, `n_bounded_unknown`, `bounded_share_known`, `n_agencies`,
  `n_states`, `rank_score`, `is_exploratory` (R=3 tier), `example_project_ids`, `nearest_ce_ids`,
  `nearest_ce_scores`, `matched_ce_ids`, `manual_ce_review_status`; **provenance:** `cluster_method`,
  `cluster_params`, `cluster_input_sha256`, `embedding_model`, `llm_model`, `netnew_prompt_version`,
  `netnew_schema_sha`, `netnew_extraction_run_at`, `netnew_llm_run_at`.
- `data/analysis/deliverable06/net_new_ce_matches.parquet` — cluster × nearest-CE review rows:
  `cluster_id`, `ce_id`, `agency_unit`, `ce_description`, `cosine`, `is_match` (LLM),
  `manual_ce_review_status` (`pending`/`confirmed_net_new`/`reclassified_adopt_expand`).
- `output/deliverable06/review/net_new_membership.csv` — FONSI → cluster drill-down, carrying
  `project_id`, `project_title`, `lead_agency_harmonized`, `project_state`, `action_category_pass1`,
  `potential_ce_theme`, `action_label_freeform`, `why_not_current_candidate`, `key_activities`,
  `is_bounded_low_impact`, `classification_confidence`, `extraction_confidence`, `cluster_text`,
  `action_summary`.
- `output/deliverable06/review/net_new_noise.csv` — the `-1`/one-off rows, **auditable and
  count-reconciled** (excluded from the shortlist but not hidden; Codex finding 6).
- A report **Analysis 4** section in `deliverable06.qmd` (ranked **candidate** shortlist + a
  theme-landscape figure), and a **linked review page** (mirroring the CE verification worksheet).

## 8. Validation / QA (`qa_deliverable06.py` additions)
- **Reconciliation:** `clustered + noise == 314`; membership totals match; every input row has
  non-empty `cluster_text`.
- **Shortlist integrity:** every ranked theme is `coherent` ∧ `plausible_net_new` ∧ codifiable ∧
  `n_bounded_true >= R`; **no funding-only theme** in the shortlist; **no theme with a confirmed
  close CE** in the shortlist.
- **CE-review status:** each shortlisted theme has a `net_new_ce_matches` row and a
  `manual_ce_review_status` before the report consumes it.
- Spot-check each shortlisted cluster's member summaries (does the theme hold?); no cluster spans
  obviously unrelated actions; thin-n (R=3) themes flagged `is_exploratory`.

## 9. Open decisions — resolved by review
1. **Funding vs physical action** → three-state `codifiability` (§4.6): exclude
   `funding_only_not_codifiable`; keep `funded_physical_action_codifiable` only with explicit
   "CE is for the physical activity, not financing" language. D1 funding fields are a **later
   diagnostic**, not a first-implementation cross-deliverable dependency (Codex finding 5).
2. **Transmission-adjacent themes** (access roads, interconnection taps) → LLM screen tags
   `belongs_to_existing_type`; the 37 pass-1 `transmission_upgrade` rows get explicit attention.
3. **Recurrence threshold R** → **R = 5** main shortlist; **R = 3** exploratory only if
   multi-agency/multi-state and highly coherent. Applies to `n_bounded_true` at ranking (decoupled
   from HDBSCAN `min_cluster_size`).
4. **Pool scope** → **all 314** clustered; boundedness is an eligibility/ranking field (Codex finding 2).
5. **One FONSI → one theme** — `cluster_text` is single-valued per FONSI; no multi-membership.
6. **Cluster stability** → documented param grid + stability check (§4.4).
7. **Terminology** → "uncategorized **FONSI** residual" / "candidate net-new CE themes pending
   CE-catalog review"; never "uncategorized CEs".

## 10. Dependencies & tooling
- **Available in `nepa` env:** `sentence_transformers`, `sklearn` **1.8** (with
  `sklearn.cluster.HDBSCAN`), `anthropic`, the local `embeddings.embed()` helper.
- **No new dependency required.** `bertopic`, `umap-learn`, and the third-party `hdbscan` are
  **not** needed.

## 11. Cost & effort
- Embedding + clustering + CE retrieval: **free** (local, ~314 short texts + cached CE embeddings).
- LLM cluster naming/screening: **~$0.50–$2** (one cached call per cluster, ~20–40 clusters, Sonnet).
- Build effort: ~half a day–one day (one script + CE-retrieval gate + a report section + a linked page).

## 12. Reproducibility
Same guarantees as the classify pass: pinned embedding model, deterministic clustering (fixed,
documented params; **hashed** stable cluster ids), CE retrieval from committed catalog embeddings,
LLM naming on a **pinned Sonnet snapshot at temp 0 with a committed cache** keyed on cluster content +
`NETNEW_PROMPT_VERSION`. Audit timestamps (`netnew_extraction_run_at` on all rows,
`netnew_llm_run_at` per successful cluster call) per project convention. Re-running regenerates the
committed outputs; no hand-edited values.

## 13. Review status
Codex review (2026-06-30) returned **8 findings + answers to 6 questions**; **all applied** above
(findings 1–8; funding split into an orthogonal `codifiability` field as a refinement of findings
1+5). No open questions remain before implementation; the only deferred item is the optional D1
funding-field diagnostic (§9.1), explicitly out of the first implementation.
