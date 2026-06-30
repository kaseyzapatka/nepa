# Code review request (Codex): D6 Stage-2 LLM classification

You are reviewing a specific, self-contained change. Be skeptical and concrete. Point to
exact files/lines. The author is a data scientist; the deliverable is client-facing
(Clean Air Task Force), so correctness and reproducibility matter more than cleverness.

## Commit under review

`268943a` — "[D6] add Stage-2 LLM classification (--stage classify) to fix action_category"

Files changed (all under `phase2/code/deliverable06/`):
- `prompts.py` — new `build_classification_prompt()`, `classification_tool_schema()`,
  `CLASSIFICATION_PROMPT_VERSION`, `ACTION_CATEGORIES`.
- `enrich_lib.py` — new `call_classification()` and `classify_preflight()` (mirror
  `call_enrichment` / `preflight`); added two imports from `prompts`.
- `03_enrich_llm.py` — refactored into `run_extraction()` + new `run_classification()`,
  driven by a new `--stage {both,extract,classify}` flag (default `both`).

## Why this change exists

D6 groups ~451 clean-energy EA→FONSIs into action types to find categorical-exclusion (CE)
candidates. Each FONSI is read once by Claude Sonnet in `03` (a 39-field structured
extraction), which also assigns `action_category`. That field's instruction was just six bare
labels with **no definitions and no enum constraint**, so keyword-similar actions were
mislabeled: a botanical "Experimental Garden **Array**" → `solar`; a BLM **land withdrawal** →
`solar`; a VHF two-way **radio** upgrade → `transmission_upgrade`.

Re-running the whole 39-field extraction to fix one field costs ~$19 **and** would regenerate
(and perturb) the already-finalized Analysis-2 fields (mitigation, significance thresholds,
verified quotes). So classification was split into a **separate, cheaply-cached stage** that
re-asks ONLY the category from the **already-extracted summary** (no document re-read), with
real definitions + an enum schema, and **overwrites** `action_category` (preserving the
extraction value as `action_category_pass1`).

## Design intent (verify it actually holds)

- `--stage both` (default): extraction (cache-aware) then classification → a from-scratch run
  is fully correct in one command.
- `--stage classify`: reuse the committed extraction output, re-classify only (~$1.49 for 452,
  Sonnet, temp 0). This is how the author fixes the classifier **now** without re-paying $19.
- Two independent caches: extraction keyed on `PROMPT_VERSION|SCHEMA_VERSION|model|packet`
  (`cache_key`), classification keyed on `CLASSIFICATION_PROMPT_VERSION|model|summary-prompt`
  (`classify_key`). Bumping the classify version forces a classify-only re-run; the extraction
  cache is untouched.
- Reproducibility: committed `fonsi_enrichment.parquet` is canonical; anyone re-runs the script
  (pinned model, temp 0) to regenerate. No hand-edited values.

## Explicitly OUT OF SCOPE for this review (do not flag as missing)

- The **downstream rewiring is not done yet.** `07/08/09` and the report still build Analysis 1
  from the keyword `candidate_category`; switching them to the corrected `action_category` +
  shape fields (`within_existing_row`, `line_miles`, `new_access_road`) is the *next* commit.
- The real classify pass has not been run yet (no API key in this environment); only
  compile + dry-run + a mock end-to-end test were done.

## Review questions

1. **Cache correctness.** Is `classify_key` sufficient to guarantee determinism and correct
   invalidation? The prompt text embeds the summary, so a changed summary changes the key — good
   — but the *definitions* live in `build_classification_prompt`, not in the key except via
   `CLASSIFICATION_PROMPT_VERSION`. If someone edits the prompt wording **without** bumping the
   version, stale cached answers are silently reused. Is that an acceptable footgun, or should
   the key hash the prompt template too?
2. **Overwrite + audit logic** (`run_classification`). `action_category_pass1` is set only if
   absent, then `action_category` is overwritten. Trace a *second* `--stage classify` run on an
   already-classified parquet: is `pass1` still the extraction value (not the
   previously-classified value)? Is the NaN-safe `n_changed` correct?
3. **Failure / skip handling.** No-summary rows are skipped (keep their category); a failed API
   call (`parsed is None`) keeps the pass-1 category. Is silently keeping pass-1 on failure the
   right call for a client deliverable, or should the run fail loudly above some error rate (the
   extraction stage enforces `MIN_SUCCESS_*`; classification does not)?
4. **Is classifying from the summary sound?** The summary is ~100 tokens and was written by the
   same model in pass 1. Does re-classifying from it (rather than the full evidence packet) risk
   propagating a bad summary, or systematically lose signal the packet had? The shape fields
   (`within_existing_row`, `line_miles`) still come from extraction — is that split coherent?
5. **Prompt quality.** Do the category definitions + the "physical action, not keywords" rules in
   `build_classification_prompt` actually kill the named failure modes (garden→other, grant→other,
   radio→other, new-build→other)? What categories or edge cases are under-specified (e.g.,
   battery storage, hydro, distribution vs transmission, hybrid solar+storage)?
6. **The vestigial extraction `action_category`.** Extraction still produces a coarse
   `action_category` that classification overwrites — so a from-scratch `both` run asks for the
   category twice (once wasted). Acceptable wart, or should `action_category` be removed from the
   extraction schema (which would force a one-time $19 re-extraction)?
7. **Refactor fidelity.** `run_extraction` is the old `main()` body moved into a function. Diff it
   against the prior version: is any behavior changed (cache use, checkpointing, the
   `MIN_SUCCESS` gate, the evidence/coverage CSVs, the `suffix`/`mode` handling)?
8. **Thread-safety / SDK use.** `call_classification` mirrors `call_enrichment` (shared client,
   `max_retries`, `tool_choice` forced). Any concurrency or error-path issue?
9. **Anything that would embarrass us with the client**, or a simpler/cheaper/more robust design.

## How to verify locally

```
conda run -n nepa python 03_enrich_llm.py --dry-run            # both-stage cost preview, no key
conda run -n nepa python 03_enrich_llm.py --stage classify --dry-run   # ~$1.49, no key
conda run -n nepa python -m py_compile prompts.py enrich_lib.py 03_enrich_llm.py
```

## Codex review feedback (2026-06-30)

### Findings

1. **High - failed classify reruns can leave stale categories while stamping them as current.**
   In `03_enrich_llm.py:274-292`, `action_category_pass1` is initialized only when absent,
   which correctly preserves the extraction value on a second run. But the failure/default path
   for `action_category` uses the current `df["action_category"]` value (`03_enrich_llm.py:283`),
   not `action_category_pass1`. On the first classify run from the extraction-only parquet, that
   current value is pass 1, so the behavior matches the comment. On any later classify run over
   an already-classified parquet - especially after a `CLASSIFICATION_PROMPT_VERSION` bump, cache
   deletion, or partial API outage - a failed row keeps the previous classified category while
   `classification_run_at` and `classification_prompt_version` are overwritten for every row
   (`03_enrich_llm.py:286-287`). That can produce a mixed-version output that looks fully
   classified under the new prompt. The confidence/rationale columns are also replaced with blank
   defaults for failed rows (`03_enrich_llm.py:284-285`). Recommendation: add per-row
   `classification_parse_ok`, `classification_error`, and probably `classification_cache_hit`;
   stamp version/run metadata only for successful/cached classifications; and either fail before
   writing canonical output when failures occur in full/pilot mode or explicitly fall back to
   `action_category_pass1` if that is the intended policy.

2. **Medium - classification has no success gate before overwriting the canonical parquet.**
   `run_classification()` counts failed calls (`03_enrich_llm.py:290-292`), but `main()` writes
   `clean_out` unconditionally after classification (`03_enrich_llm.py:371-373`). The extraction
   stage has a parse-rate gate (`03_enrich_llm.py:215-218`); classification should have the same
   protection, arguably stricter because the output is cheap to rerun and directly controls the
   corrected category used downstream. For the client-facing full run, I would fail on any
   classification failure or at least below a high threshold (for example 99% or 100%), with the
   looser threshold reserved for `--sample`.

3. **Medium - the classifier prompt has a real category overlap for wind met/resource testing.**
   `temporary_resource_assessment` explicitly includes met towers and temporary site
   characterization (`prompts.py:283-284`), while `wind_onshore` also includes onshore
   wind-resource met testing (`prompts.py:285`). A wind met tower or wind resource assessment fits
   both definitions, so the enum alone will not prevent inconsistent labels. Pick and state a
   priority rule. If "temporary" is intended to mean no permanent generating/transmitting
   facility, wind met testing should probably stay in `temporary_resource_assessment`; if wind
   resource testing is intentionally part of `wind_onshore`, remove met towers from the temporary
   definition or state that wind-specific met testing takes precedence.

4. **Low - the cache-key concern in the prompt is partly already solved, but the tool schema is
   still outside the key.** The code hashes the full classifier prompt, not just the summary:
   `_classify_input()` builds the full prompt (`03_enrich_llm.py:225-232`), and `classify_key()`
   hashes that text with the model and classify prompt version (`03_enrich_llm.py:74-75`,
   `03_enrich_llm.py:248-250`). So edits to the wording/definitions in
   `build_classification_prompt()` should invalidate cache entries even if the version is not
   bumped. What is not covered is the tool schema itself (`prompts.py:294-300`). If the enum,
   required fields, or confidence choices change without prompt text/version changes, stale cache
   entries can still be reused. Low-cost fix: include
   `json.dumps(classification_tool_schema(), sort_keys=True)` in the key or add a separate
   `CLASSIFICATION_SCHEMA_VERSION`.

### Other Review Notes

- I did not find refactor drift in the extraction path. The old `main()` body was moved into
  `run_extraction()` with the same cache behavior, checkpointing, raw/analysis outputs,
  evidence CSV, coverage CSV, suffix handling, and `MIN_SUCCESS` gate.
- `action_category_pass1` does preserve the original extraction value across a normal second
  classify run, because the column is only created when absent (`03_enrich_llm.py:274-276`).
  The `n_changed` calculation is NaN-safe as written (`03_enrich_llm.py:288-289`).
- Classifying from the cached summary is a defensible cost/control tradeoff for this specific
  fix, since it avoids perturbing the finalized Analysis-2 fields. Before using the result
  downstream, QA should sample all changed rows, low-confidence rows, and the named failure
  modes against the original evidence quotes. Summary-only classification can only be as good as
  pass 1's `action_summary`, `key_activities`, and `purpose_and_need`.
- The prompt directly addresses the named failures: garden/experimental array, grants/funding,
  land withdrawals/ROW grants, radio/communications, and new transmission on new ROW. Remaining
  edge cases worth spelling out before the real run: battery storage, hybrid solar+storage,
  substations/interconnection/gen-tie lines, distribution-only work if that is not meant to be
  "transmission_upgrade", and new lines inside existing ROW/corridors.
- Keeping the vestigial extraction `action_category` is acceptable for now. Removing it would
  force a paid schema-changing re-extraction for little benefit; the key is to make the stage-2
  overwrite/audit columns unambiguous.
- `call_classification()` mirrors `call_enrichment()` closely and I did not see a separate
  thread-safety issue. The main operational risk is not SDK concurrency; it is treating failed
  or skipped classification rows as if the classification stage completed cleanly.

### Verification Run

- `conda run -n nepa python -m py_compile prompts.py enrich_lib.py 03_enrich_llm.py` passed.
- `conda run -n nepa python 03_enrich_llm.py --dry-run` passed and reported about `$18.31`
  extraction plus `$1.49` classification for 452 full-run FONSIs.
- `conda run -n nepa python 03_enrich_llm.py --stage classify --dry-run` passed and reported
  about `$1.49`.
- Current `phase2/data/analysis/deliverable06/fonsi_enrichment.parquet` has 452 rows and only
  the original `action_category` column among the classification/audit fields; the classification
  cache file is not present yet, consistent with the note saying the real classify pass has not
  been run.

## Resolution (author, 2026-06-30)

All four findings were **Stage-2 only** — fixed without touching the extraction stage, so the
committed extraction output stays reproducible (per the re-run constraint).

- **Finding 1 (High) — FIXED.** `run_classification` tracks per-row state (ok/failed/skipped) and
  writes `classification_parse_ok` / `classification_cache_hit` / `classification_error`. Failed AND
  skipped rows fall back to `action_category_pass1` and are NOT stamped (`classification_prompt_version`
  / `classification_run_at` left blank); only successful/cached rows are stamped. A partial output can
  no longer masquerade as fully classified.
- **Finding 2 (Medium) — FIXED.** `main` gates before overwriting the canonical parquet:
  `MIN_CLASSIFY_SUCCESS = 1.0` for full/pilot (fail on ANY failure; cached successes kept so
  `--stage classify` resumes), `MIN_SUCCESS_DEBUG` for `--sample`.
- **Finding 3 (Medium) — FIXED.** The prompt has an explicit PRECEDENCE block: temporary wind/solar
  testing (met towers, surveys) -> `temporary_resource_assessment`; `wind_onshore`/`solar` are
  PERMANENT generating facilities only; geothermal exploratory drilling -> `geothermal_exploration`.
- **Finding 4 (Low) — FIXED.** `classify_key` hashes `classification_tool_schema()` too, so an
  enum/required-field change invalidates stale cache entries even without a version bump.
- **Edge cases — ADDED to the prompt.** Standalone battery/storage -> other; solar+storage -> solar;
  distribution + substation/interconnection upgrades -> transmission_upgrade; a new line/circuit WITHIN
  an existing ROW -> transmission_upgrade (new greenfield ROW -> other); gen-tie precedence stated.
- **Not changed (out of scope):** the vestigial extraction `action_category` (Codex agreed removing it
  would force a paid re-extraction for little benefit). Extraction prompt, schema, and cache key untouched.

Validated offline ($0): py_compile, `--dry-run` (~$1.49 classify), and a mock end-to-end test covering
success-stamping, failure fallback to pass1, the no-summary skip, the success gate, and second-run
idempotency of `action_category_pass1`.

## Codex second-pass review (coding changes only, 2026-06-30)

Scope: reviewed commit `2436519` only, limited to Stage-2 classification code changes in
`03_enrich_llm.py` and `prompts.py`. I did not re-review downstream rewiring, report text, or
the resolution note prose.

### Findings

1. **High - `--stage both` can still overwrite the canonical parquet before the classification
   success gate fires.** The new gate is in the right place for `--stage classify`: it calls
   `run_classification()`, checks `st["n_ok"] / st["n_attempted"]`, and only then writes
   `clean_out` (`03_enrich_llm.py:400-409`). But in the default `--stage both` path,
   `run_extraction()` is called first (`03_enrich_llm.py:390-391`), and `run_extraction()` writes
   the analysis parquet to the same `clean_out` path before returning (`03_enrich_llm.py:186-192`).
   If classification later fails the new gate (`03_enrich_llm.py:404-408`), the message says
   `NOT writing {clean_out.name}`, but the extraction-only version has already been written. That
   can leave the canonical parquet without Stage-2 columns / corrected categories after a failed
   default run, and it can also regress a previously-classified canonical output if someone reruns
   `--stage both` and classification fails or the process is interrupted between extraction and
   classification. Recommendation: make the default `both` path atomic with respect to the final
   analysis parquet. Options: have `run_extraction()` return `clean_df` without writing `clean_out`
   when `do_classify` is true; write extraction-only output to a temporary/stage-specific path; or
   write the final classified parquet to a temp file and replace `clean_out` only after both stages
   pass.

2. **Low - the prompt semantics changed but `CLASSIFICATION_PROMPT_VERSION` stayed at v1.** The
   cache is safe because `classify_key()` now hashes the full prompt plus tool schema
   (`03_enrich_llm.py:76-79`), but the output audit column only records
   `CLASSIFICATION_PROMPT_VERSION` (`03_enrich_llm.py:299`, `03_enrich_llm.py:311`). The prompt now
   has materially different precedence and edge-case rules (`prompts.py:275-297`) under the same
   `d6_classify_prompt_v1` label. Since the real classify pass has not been run yet, this is easy
   to clean up before producing client-facing outputs: bump to `d6_classify_prompt_v2`, or store a
   prompt/schema hash in the output audit columns in addition to the human version label.

### Resolved From Prior Pass

- The stale-category failure path is substantially fixed inside `run_classification()`: failed and
  skipped rows fall back to `action_category_pass1`, are marked `classification_parse_ok = False`,
  and are not stamped with prompt version/run time (`03_enrich_llm.py:291-312`).
- The classification success gate now exists for attempted rows, with strict full/pilot behavior
  and looser `--sample` behavior (`03_enrich_llm.py:400-408`).
- The cache key now includes the classification tool schema (`03_enrich_llm.py:76-79`).
- The prompt overlap between wind met testing and temporary resource assessment is resolved by an
  explicit precedence rule (`prompts.py:291-295`).

### Verification Run

- `conda run -n nepa python -m py_compile prompts.py 03_enrich_llm.py` passed.
- `conda run -n nepa python 03_enrich_llm.py --stage classify --dry-run` passed and reported about
  `$1.49`.
- `conda run -n nepa python 03_enrich_llm.py --dry-run` passed and reported about `$18.31`
  extraction plus `$1.49` classification.
- Current `phase2/data/analysis/deliverable06/fonsi_enrichment.parquet` still has 452 rows and no
  Stage-2 audit columns, so the real classification output has not been materialized in this
  checkout.

## Resolution — second pass (author, 2026-06-30)

Both findings fixed. The Finding-1 change adds a write-gating flag to `run_extraction` that does NOT
touch the extraction computation/cache/raw output (extraction stays byte-reproducible) and is never
exercised by `--stage classify`.

- **Finding 1 (High) — FIXED.** `run_extraction(..., write_clean=True)`; in the default `--stage both`
  path `main` passes `write_clean=False`, so extraction no longer writes the analysis parquet —
  classify writes `clean_out` atomically only after its success gate passes. A failed `both` run now
  leaves the prior `clean_out` untouched (no extraction-only regression). `--stage classify` never
  calls `run_extraction`, so it is unaffected; `--stage extract` writes as before.
- **Finding 2 (Low) — FIXED.** Bumped to `d6_classify_prompt_v2`, and added `classification_config_sha`
  (16-char fingerprint of prompt template + tool schema) stamped per successful row, so the human
  label cannot silently drift from the prompt that produced a row.

Validated offline: py_compile, `--dry-run` (reports v2), and a mock test confirming
`classification_config_sha` stamps on ok rows and is blank on failed/skipped rows.
