# Technology timeline improvement suggestions


## Plans for improving Within-Project Sequencing

The current figure (02_geothermal.R:449) is limited to projects with ≥2 dated actions — and most geothermal actions are missing one or both timeline dates. Here's what's needed and what's possible:

Root cause
Timeline coverage for geothermal is low because:

BLM CEs for individual well permits often have no NOI/scoping → no initiation signal
Decision dates for CEs are rarely published in machine-readable form
The BERT extractor was trained on EIS/EA-heavy examples; CE-heavy geothermal actions are OOD
Plan A — Improve BERT timeline coverage for geothermal CEs (highest impact)
The most direct fix. Two levers:

Add geothermal-specific training examples to extract_timeline.py. BLM CE decisions use phrases like "Decision Record", "Finding of No Significant Impact", "approved" in combination with a date — collect 20–30 manually-confirmed (project_id, initiation_date, decision_date) pairs from the geothermal subset and add them to the BERT training set as high-weight examples.

Expand initiation patterns for well permit language: "application received", "permit to drill submitted", "well permit application", "Notice of Intent to Drill" (NID, distinct from NEPA NOI). These don't currently fire in extract_timeline.py's regex candidates.

Run with --bert-run after retraining. Check coverage with:


analysis %>% filter(project_is_geothermal) %>%
  summarise(pct_dated = mean(!is.na(bert_initiation_date_final) & !is.na(bert_decision_date_final)))
Plan B — LLM hybrid run on geothermal subset (medium effort)
Run the hybrid LLM timeline extractor specifically on geothermal actions where BERT returned no dates:


python code/extract/extract_timeline.py \
  --llm-run --hybrid --use-regex-cache \
  --filter project_is_geothermal=True \
  --sample 50 --model llama3.2:3b-instruct-q4_K_M \
  --output test_geothermal_llm.parquet
LLMs handle free-form CE decision language better than BERT. Validate on the 50-sample before full run.

Plan C — Better project key matching (low-hanging fruit, no data work)
The current normalize_geothermal_key() strips too many words (including location names), causing unrelated projects to merge. Replace with a Levenshtein/fuzzy match on the raw title after only light normalization (lowercase + punctuation). This won't increase date coverage but will reduce false merges in the sequence figure, making the sequenced sample more reliable.

Plan D — Enrich the sequence figure itself (visualization-side)
Even with limited dates, the figure can be improved:

Show geologic area / field name as a secondary grouping row label (e.g., "Coso", "Salton Sea", "Dixie Valley") — extract from title
Add a gap indicator for undated actions within a project (dotted line or hollow point showing "action with unknown dates exists here")
Sort by total project span rather than first start date, so the longest developments appear prominently
Facet by agency (BLM vs Forest Service) since BLM dominates drilling CEs and Forest Service tends to have longer reviews
Recommended order
Priority	Plan	Effort	Expected gain
1	A — BERT training examples	Medium	+30–50% date coverage
2	B — LLM hybrid on geothermal	Low	Fills gaps after A
3	C — Better key matching	Low	Cleaner figure immediately
4	D — Visualization enrichment	Low	Better story from existing data
Start with C (no data re-run needed) to get a cleaner figure now, then tackle A+B in the next extraction pass.