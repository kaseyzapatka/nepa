# Deliverable 6 — Analysis 4: Net-new CE discovery from the "other" residual

**Status:** ⚠️ **SUPERSEDED (2026-06-30) by `deliverable06_refactor.md`.** The tech × action grid
replaces this plan's HDBSCAN clustering front-end and its 314-"other" universe (every FONSI now lands
in a grid cell; the `develop` pool is the grid's no-CE cells, not the 314). This plan's **back-end is
absorbed** into the refactor — the CE-retrieval gate, `codifiability` screen, bounded gate, recurrence
× spread ranking, and "candidate, pending review" framing all carry over to the grid's develop cells.
Do **not** implement this plan's clustering/QA (`clustered + noise == 314`) as written. Kept for the
back-end design record only. (Prior status: plan, revised after Codex review, not yet built.)
**Author:** Claude. **Reviewed:** Codex, 2026-06-30 (`phase2/plans/deliverable06_newCEs_feedback.md`);
second independent review round applied 2026-06-30 (see §13).
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

D6's candidate classifier (`07_classify_and_rank.py`, output `d6_new.csv`) emits **`new`** (net-new:
a recurring action with no existing CE), **`expand`** (exceeds an existing CE's bound), **`adopt`**
(an existing CE at another agency covers it), plus `already_covered` and `contrast`. The 5 candidate
types were hand-picked *because they already recur and already have CEs* — they were built to test
adopt/expand. The LLM classifier sorts every FONSI into one of those 5 or **"other."**

Consequence: all 5 candidate types resolve to **`adopt`**, and the **`new`/net-new bucket is empty —
not because no net-new actions exist, but because the design only examined types that already have
CEs.** A net-new CE is, by definition, a recurring action type with no existing CE; such types
*cannot* be among the 5, so they sit unexamined in the **"other"** residual. This analysis examines it.
(Analysis 4's own cluster-level `net_new_verdict` enum in §4.6 is distinct from this per-FONSI `new`
verdict.)

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

**QA gate:** every one of the 314 input rows must have non-empty `cluster_text`. **48 of 314** lack
`potential_ce_theme` and rely on the fallback ladder (47 are bounded-False, ~1 bounded-True);
verified that all 314 resolve to non-empty `cluster_text`. Persist the exact assembled `cluster_text`
and a `cluster_input_sha256` for reproducibility.

### 4.3 Embed
Embed each `cluster_text` with **`sentence-transformers/all-MiniLM-L6-v2`** via the existing
`embeddings.embed()` helper (the model D6 already uses in `04`/`06` and D3). 384-d, L2-normalized,
deterministic. **No disk cache needed** — `embeddings.embed()` is an in-process `lru_cache` on the
*model object* only (not an on-disk vector cache), and recomputing ~314 short texts is instant. (If a
cache is ever wanted, hand-roll a `cluster_input_sha256`+model-keyed npy like `06` does internally.)

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

### 4.5 Retrieve nearest existing CEs — **the net-new gate** (finding 1, HIGH)
Before any cluster is called net-new, retrieve its nearest existing CEs automatically. The mechanics
are made explicit here so there is exactly **one** decision authority, a defined `k`, and a defined
cosine role (round-2 review resolved a three-decider ambiguity):

- **CE corpus + embeddings.** Load the catalog from **`ce_source.load_ce_catalog()`** — it returns
  `ce_id`, `agency_unit`, `ce_description`, which is all the retrieval needs (one source; **not**
  `ce_landscape_ces.parquet`, whose `nearest_xagency_ce` is a CE↔CE precedent irrelevant here).
  **Re-embed the ~2,105 `ce_description`s fresh** with `embeddings.embed()` (free/local, ~instant;
  this matches `04_base_rates_and_ce.py`, which re-embeds rather than reusing a cache). Do **not**
  assume a reusable keyed cache from `06`: `06` persists only an unlabeled `(2105, 384)`
  `ce_embeddings.npy` aligned to its parquet by *implicit row order* and drops the source text, so
  reuse would silently mispair on any re-sort. (If reuse is ever desired, re-derive `06`'s exact
  sort + `normalize_space` and verify against `ce_embeddings.sig`, failing loudly on mismatch.)
- **Retrieval.** For each **coherent** cluster, embed `theme_name` + a few representative member
  `cluster_text`s and cosine-retrieve the **top `k = 5`** nearest CE descriptions; record
  `nearest_ce_ids`, `nearest_ce_scores`, and `nearest_ce_cosine` (= the max).
- **Decision authority = the LLM screen, never a cosine cutoff.** The top-5 are **shown to the LLM**
  (§4.6), which alone decides novelty via `net_new_verdict`. Cosine is *retrieval only* — never an
  auto-reject — matching `04`, which treats every cosine hit as an `unverified_candidate` with no
  accept threshold.
- **Cosine tripwire = a review flag only.** If `nearest_ce_cosine ≥ 0.75`, set
  `requires_ce_review = True`; such a cluster may still be `plausible_net_new`, but it cannot enter
  the **client** shortlist until human CE-catalog/eCFR review clears it (§4.7–4.8). (`06` uses 0.85
  for CE↔CE near-duplicates; 0.75 is a deliberately looser cluster→CE flag.)
- Write `net_new_ce_matches` (§7) with the top-k rows (per-row `cosine`) and `manual_ce_review_status`;
  the **cluster-level** `nearest_ce_cosine` (= max) and the `requires_ce_review` flag live on
  `net_new_themes` (§7). A theme the LLM ties to a close existing CE is **adopt/expand**, not net-new.

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
  Only `plausible_net_new` is eligible (§4.7); `close_existing_ce_review` and `belongs_to_existing_type`
  are both treated as **not net-new** (ineligible), and `not_coherent` drops the cluster. This is the
  **LLM's** novelty decision — distinct from the orthogonal cosine `requires_ce_review` flag (§4.5),
  which can additionally hold even a `plausible_net_new` cluster out of the *client* shortlist until
  human review clears it.
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
∧ `bounded_share_known >= τ`. Novelty is decided **once**, by the LLM screen in §4.6 (which saw the
top-5 nearest CEs) — there is no separate cosine reject clause. A high-cosine cluster
(`requires_ce_review == True`, §4.5) may still be `plausible_net_new` and thus eligible, but it is
held out of the **client** shortlist until human CE-review clears it (§4.8); it appears in the full
output flagged. Rank eligible clusters by
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
- `embeddings.embed()` for vectors (deterministic, pinned model; no disk cache — recompute is instant).
- **`sklearn.cluster.HDBSCAN`** with fixed, documented params; **hashed** stable `cluster_id`;
  Agglomerative fallback behind a flag.
- §4.5 nearest-CE retrieval: **re-embed** the ~2,105 `ce_source.load_ce_catalog()` descriptions fresh
  (not a `06` cache reuse), cosine top-`k=5`, record `nearest_ce_cosine` + `requires_ce_review`.
- LLM cluster naming/screening: **add new code** (not a reuse) mirroring the classify path —
  `prompts.build_netnew_prompt()` + `prompts.netnew_tool_schema()` + `NETNEW_PROMPT_VERSION`, and an
  `enrich_lib.call_netnew()` mirroring `enrich_lib.call_classification` (same never-raises /
  token-return contract). **Cache** on a cluster-content hash + `NETNEW_PROMPT_VERSION` via the same
  `classify_key` / `write_json_atomic` pattern `03` already uses (re-runs free; committed output canonical).
- **Audit stamps (project convention):** `netnew_extraction_run_at` on **all** rows at build time;
  `netnew_llm_run_at` per-cluster only when the LLM call succeeds (else `""`).
- writes the outputs in §7. Add a `--dry-run` (cost preview, no key) like `03`.

**Pipeline integration (finding 5 — corrected sequencing):**
- `03_enrich_llm.py` (the 37-field enrichment + `--stage classify`) is **standalone and *not* in
  `_run.py`**; `fonsi_enrichment.parquet` is produced out-of-band and merely *checked for existence*
  by the `09_wire_enrichment.py` guard. So 10 slots into `_run.py` **after `06_ce_landscape.py`**
  (needs `ce_source.load_ce_catalog()`) with `fonsi_enrichment.parquet` as an **external prerequisite**,
  exactly the way `09` already treats it — not "after 03" inside the orchestrator.
- Add 10 to `_run.py` **only after** the classify output is stable.
- **Fail loudly** if `fonsi_enrichment.parquet` is missing, if `classification_parse_ok` is absent, or
  if classified rows are not stamped with the expected `classification_prompt_version`. (The CE
  catalog is loaded via `ce_source`; 10 re-embeds it, so no `06` embedding artifact is required.)

## 7. Outputs / deliverables
- `data/analysis/deliverable06/net_new_themes.parquet` — one row per cluster:
  `cluster_id`, `theme_name`, `coherent`, `net_new_verdict`, `codifiability`, `n_total`,
  `n_bounded_true`, `n_bounded_false`, `n_bounded_unknown`, `bounded_share_known`, `n_agencies`,
  `n_states`, `rank_score`, `is_exploratory` (R=3 tier), `requires_ce_review`, `in_client_shortlist`,
  `example_project_ids`, `nearest_ce_ids`, `nearest_ce_scores`, `nearest_ce_cosine`, `matched_ce_ids`,
  `manual_ce_review_status`; **provenance:** `cluster_method`,
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
- **Shortlist integrity:** every **client-shortlist** theme (`in_client_shortlist == True`) is
  `coherent` ∧ `plausible_net_new` ∧ codifiable ∧ `n_bounded_true >= R`; **no funding-only theme**;
  and **no `requires_ce_review` theme** enters the client shortlist until its `manual_ce_review_status`
  clears (`confirmed_net_new`). A `plausible_net_new` theme flagged `requires_ce_review` may still be
  *eligible* and appear in the full output — it is just held out of the client-facing list.
- **CE-review status:** each shortlisted theme has ≥1 `net_new_ce_matches` row and a resolved
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
- Embedding + clustering + CE retrieval: **free** (local, ~314 cluster texts + ~2,105 re-embedded CE descriptions).
- LLM cluster naming/screening: **~$0.50–$2** (one cached call per cluster, ~20–40 clusters, Sonnet).
- Build effort: ~half a day–one day (one script + CE-retrieval gate + a report section + a linked page).

## 12. Reproducibility
Same guarantees as the classify pass: pinned embedding model, deterministic clustering (fixed,
documented params; **hashed** stable cluster ids), CE retrieval by re-embedding the catalog fresh
(deterministic; no fragile cache reuse), LLM naming on a **pinned Sonnet snapshot at temp 0 with a
committed cache** keyed on cluster content +
`NETNEW_PROMPT_VERSION`. Audit timestamps (`netnew_extraction_run_at` on all rows,
`netnew_llm_run_at` per successful cluster call) per project convention. Re-running regenerates the
committed outputs; no hand-edited values.

## 13. Review status
- **Round 1 — Codex (2026-06-30):** 8 findings + answers to 6 questions; **all applied** (funding
  split into an orthogonal `codifiability` field as a refinement of findings 1+5).
- **Round 2 — independent code-verified review (2026-06-30):** 8 findings, **all applied**:
  (1) the CE-match gate now has a single decision authority — the LLM screen — with an explicit
  `k = 5` retrieval and cosine used only as a `requires_ce_review` flag (≥ 0.75), not a reject
  clause; the redundant §4.7 "no close CE" clause was removed and §8 reconciled.
  (2) CE embeddings are **re-embedded fresh** in script 10 (the `06` `ce_embeddings.npy` is an
  unlabeled row-order-aligned array with the source text dropped — not a safe keyed cache).
  (3) §2 uses the built classifier's real verdict name **`new`** (output `d6_new.csv`), not
  "develop"; `already_covered`/`contrast` noted.
  (4) §4.5 names **one** CE source (`ce_source.load_ce_catalog()`); the `nearest_xagency_ce`
  red herring dropped.
  (5) §6 sequencing corrected: `03_enrich_llm.py` is **not** in `_run.py`; 10 runs after
  `06_ce_landscape.py` with `fonsi_enrichment.parquet` as an external prerequisite (as `09` treats it).
  (6) §4.2 count corrected (48/314 lack `potential_ce_theme`, not 1).
  (7) §6 names the **new** code to write — `enrich_lib.call_netnew()` + `prompts.build_netnew_prompt()`
  + `netnew_tool_schema()` + `NETNEW_PROMPT_VERSION` — rather than implying a reusable `call_*`.
  (8) feedback-file path corrected.
- **Round 3 — re-review of the round-2 fixes (2026-06-30): READY TO IMPLEMENT.** All 8 round-2 fixes
  verified to hold against the code; 3 residual copy fixes applied: (a) removed the last two "cache"
  mentions (§4.3, §6) that contradicted the re-embed fix — `embeddings.embed()` is an in-process
  `lru_cache` on the model object, not a disk cache; (b) reconciled the `net_new_ce_matches` schema
  (per-row `cosine`) vs the cluster-level `nearest_ce_cosine`/`requires_ce_review` on `net_new_themes`;
  (c) stated the disposition of an LLM `close_existing_ce_review`/`belongs_to_existing_type` verdict
  (ineligible) and its distinction from the cosine `requires_ce_review` flag.
- **No code-blocking defect remains.** The only deferred item is the optional D1 funding-field
  diagnostic (§9.1), explicitly out of the first implementation.
