# Tier 5 adjudication record

`tier5_adjudication_record.csv` freezes the **raw** Tier 5 (Claude Haiku LLM fallback) verdicts for all 501 low-confidence D1 projects that reached Tier 5 — 495 successful adjudications (`rule_id = T5_llm`) plus 6 API failures (`rule_id = T5_llm_error`). Frozen 2026-07-20 from the pre-reconciliation parquet.

- Model: `claude-haiku-4-5-20251001`.
- `llm_primary` / `llm_secondary` are the LLM's **raw** ranking *before* hierarchy reconciliation — i.e. exactly what the model returned. The published parquet reorders these to obey `TRIGGER_HIERARCHY`.
- Columns: `project_id` (full UUID), `llm_primary`, `llm_secondary` (pipe-joined), `confidence`, `rule_id`, `evidence_text`, `llm_run_at`.
- To reproduce the published D1 classifications exactly — without re-calling the LLM — run `03_rerun_tier5.py --from-record`, which replays these verdicts deterministically and re-applies the same ingest-time hierarchy reconciliation.
