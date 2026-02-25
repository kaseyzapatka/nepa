# API Cost Estimates — Claude API Calls

**Model:** `claude-haiku-4-5-20251001`
**Pricing:** $1.00 / 1M input tokens · $5.00 / 1M output tokens
**Batch discount (50%):** $0.50 / 1M input · $2.50 / 1M output (if using Anthropic Batch API)

Ollama-based calls (extract_reviews.py, default timeline modes) are local and have no API cost — omitted.

---

## Summary Table

| Script | Mode | Est. calls | Est. input tokens | Est. output tokens | Est. cost |
|--------|------|-----------|-------------------|-------------------|-----------|
| `extract_gencap.py` | `--run llm` | ~1,100 | ~451K | ~61K | **~$0.75** |
| `extract_timeline.py` | `--llm-adj --provider claude` | ~1,326 | ~2,467K | ~146K | **~$3.20** |
| `extract_technology.py` | `--use-llm --provider anthropic` | ~750 | ~413K | ~53K | **~$0.68** |
| **Total** | | **~3,176** | **~3,331K** | **~260K** | **~$4.63** |

*Batch pricing (50% off output) reduces total to ~$4.00.*

---

## 1. Generation Capacity Adjudication — `extract_gencap.py --run llm`

**Purpose:** Resolve ambiguous cases where 2+ distinct power capacity values (e.g., 50 MW and 200 MW) were found in the same project's documents. The LLM picks the one that represents the proposed project.

**Trigger condition:** `project_gencap_candidate_count >= 2` after the regex document scan. Projects resolved via title or description in Pass 1 never reach this step.

**Prompt structure:**
```
NEPA {project_type} review. These are capacity values found by regex...
Project: {title}
Candidates:
[1] 50.0 MW — "context snippet up to 200 chars"
[2] 200.0 MW — "context snippet up to 200 chars"
Rules: [3 rules]
Return ONLY valid JSON: {"selected_index": ..., "confidence": ..., "reasoning": "..."}
```

| Parameter | Value |
|-----------|-------|
| max_tokens | 200 |
| temperature | 0.1 |

**Token estimates per call:**

| Component | Tokens (est.) |
|-----------|--------------|
| Static boilerplate (rules, JSON schema, instruction) | ~200 |
| Project title + type | ~20 |
| 2–4 candidates (value + unit + 200-char context each) | ~150–220 |
| **Total input** | **~370–440** |
| Output (JSON: index + confidence + one-sentence reasoning) | ~50–60 |

**Estimated call volume (full clean energy run):**

| Source | Total projects | Est. % with 2+ doc candidates | Est. calls |
|--------|---------------|-------------------------------|------------|
| CE     | ~19,400        | ~2% (most CE resolved via title/description) | ~390 |
| EA     | ~575           | ~45%                          | ~260 |
| EIS    | ~755           | ~60%                          | ~455 |
| **Total** | **~20,730** |                               | **~1,100** |

**Cost estimate (full run):**

| | Tokens | Cost |
|-|--------|------|
| Input | ~1,100 × 410 = **451,000 (~0.45M)** | $0.45 |
| Output | ~1,100 × 55 = **60,500 (~0.06M)** | $0.30 |
| **Total** | | **~$0.75** |

*With batch API (50% off output):* **~$0.60**

---

## 2. Timeline LLM Adjudication — `extract_timeline.py --llm-adj --provider claude`

**Purpose:** Post-BERT date adjudication for EA and EIS projects. BERT-classified date candidates (with context snippets) are sent to Claude to pick the single best initiation date and decision date.

**Trigger condition:** `project_gencap_candidate_count >= 2` after the BERT classification pass. Run with `--llm-adj --provider claude`.

**Prompt structure:** Project title + all BERT-classified date candidates (each with date, BERT label, document type, mention count, and context snippet) + decision mode constraint block + definitions and rules.

| Parameter | Value |
|-----------|-------|
| max_tokens | 200 |
| temperature | 0.1 |

**Token estimates per call:**

| Component | Tokens (est.) |
|-----------|--------------|
| Static boilerplate (definitions, rules, decision constraints) | ~400 |
| EA: up to 50 candidates × 80 tokens (date + context at 300 chars) | ~4,000 |
| EIS: up to 30 candidates × 65 tokens (date + context at 200 chars) | ~1,950 |
| **Total input per EA project** | **~2,200** (avg, ~20 candidates) |
| **Total input per EIS project** | **~1,600** (avg, ~15 candidates) |
| Output (JSON: 2 dates + 2 reasoning fields) | ~100–120 |

**Estimated call volume:**

| Source | Projects | Calls |
|--------|----------|-------|
| EA     | ~573     | ~573  |
| EIS    | ~753     | ~753  |
| **Total** | | **~1,326** |

**Cost estimate (full EA + EIS run):**

| | Tokens | Cost |
|-|--------|------|
| Input | 573 × 2,200 + 753 × 1,600 = **2,466,600 (~2.47M)** | $2.47 |
| Output | 1,326 × 110 = **145,860 (~0.15M)** | $0.73 |
| **Total** | | **~$3.20** |

*With batch API:* **~$2.84**

---

## 3. Transmission Line Length Adjudication — `extract_technology.py --use-llm --provider anthropic`

**Purpose:** When regex finds 2+ competing transmission line length values in a project's documents (e.g., "12 miles" and "47 miles"), Claude picks the one that represents the total length of the proposed line being built.

**Trigger condition:** Only triggered for transmission projects with 2+ non-trivial candidates (≥ 0.25 miles).

**Prompt structure:**
```
NEPA transmission line review. Pick the ONE candidate = total length of the proposed line.
Candidates: [up to 8, each with value + source text snippet up to 300 chars]
Rules: [5 rules about explicit length vs. partial crossings vs. location references]
Return ONLY valid JSON: {"selected_index": ..., "selected_length_miles": ..., "confidence": ..., "reasoning": "..."}
```

| Parameter | Value |
|-----------|-------|
| num_predict | 120 |
| temperature | 0.0 |

**Token estimates per call:**

| Component | Tokens (est.) |
|-----------|--------------|
| Static boilerplate (task description + 5 rules + JSON schema) | ~150 |
| Up to 8 candidates × ~65 tokens (value + 300-char context) | ~200–520 |
| **Total input** | **~400–670 (~550 avg)** |
| Output (JSON: index + length + confidence + one-sentence reasoning) | ~60–80 |

**Estimated call volume:**
- Clean energy projects with any transmission tag: ~7,100 (35% of 20,275)
- Estimated fraction with 2+ competing candidates: ~10–15%
- **Estimated calls: ~750**

**Cost estimate (full run):**

| | Tokens | Cost |
|-|--------|------|
| Input | ~750 × 550 = **412,500 (~0.41M)** | $0.41 |
| Output | ~750 × 70 = **52,500 (~0.05M)** | $0.27 |
| **Total** | | **~$0.68** |

*With batch API:* **~$0.54**

---

## Notes

- All three scripts use `claude-haiku-4-5-20251001`; no other Claude models are called.
- Call counts are **estimates** — actual counts depend on how many projects clear the 2+ candidate threshold after the regex pass. Run the regex step first and check `project_gencap_candidate_count` distributions to get a tighter number.
- `extract_reviews.py` LLM mode and the default `extract_timeline.py --llm-run` mode use local Ollama models (currently `llama3.2:3b-instruct-q4_K_M`) and have no API cost.
- Pricing as of February 2026. Verify at [platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing) before a large run.
