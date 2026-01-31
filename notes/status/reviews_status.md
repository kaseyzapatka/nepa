# Deliverable #2: Programmatic & Tiered Reviews

**Status**: First full extraction running in background
**Last Updated**: 2026-01-30
**Script**: `code/extract/extract_reviews.py`
**Output**: `data/analysis/projects_reviews.parquet`

## Deliverable Goal

Data on programmatic and tiered reviews: how many tiered reviews are there compared to total, and are they completed faster?

---

## Current Status

### Completed
- [x] Created `extract_reviews.py` with 3-tier extraction approach
- [x] Title-based detection for programmatic reviews
- [x] Regex extraction with false-positive filtering
- [x] LLM integration for ambiguous cases (optional flag)
- [x] Fixed bug: "tiering from PEIS" was incorrectly flagged as programmatic
- [x] Unit tests passing

### In Progress
- [ ] Full extraction running on 1,416 EA/EIS clean energy projects (background, `--no-llm`)
- [ ] Estimated runtime: 2-4 hours

### Next Steps
1. **Review results** from first extraction when complete
2. **Examine examples** - look at any programmatic/tiered reviews found
3. **Refine patterns** based on real examples (regex may need tuning)
4. **Second pass** with LLM enabled if needed for ambiguous cases
5. **Timeline integration** - merge with timeline data to answer "are tiered reviews faster?"

---

## Known Issues

### Issue 1: Rarity of Examples
**Problem**: Programmatic/tiered reviews are very rare, making testing difficult.
- Only **3 clean energy EA/EIS projects** have "programmatic" in title
- Random 10-project samples consistently miss these cases
- Tiered reviews estimated at only 5-15% of EA/EIS projects

**Solution**: Run full extraction first, then examine what we find rather than trying to sample.

### Issue 2: Performance/Speed
**Problem**: Each project requires loading pages from parquet with filters (~10 sec/project).
- Full run of 1,416 projects takes 2-4 hours
- Bottleneck is `pq.read_table(pages_path, filters=[...])` per project

**Potential optimization**: Pre-load all pages per source once, filter in memory.

### Issue 3: "Tier" False Positives
**Problem**: The word "tier" has many non-NEPA meanings.

| False Positive | Example |
|----------------|---------|
| EPA engine standards | "EPA Tier 4 engine" |
| Road classifications | "Tier 1 Roads and Primitive Roads" |
| Pricing systems | "tiered pricing structure" |
| Rankings | "first-tier", "top-tier" |

**Status**: Filtering working correctly - unit tests pass.

### Issue 4: Programmatic vs Tiered Distinction
**Problem**: A document might mention a PEIS in two different ways:
- "This Programmatic EIS analyzes..." → This IS a programmatic review
- "This EA tiers from the Solar PEIS..." → This TIERS FROM a programmatic review

**Status**: Fixed. Added check to exclude "tiering from" language from programmatic detection.

---

## Background

### What are Programmatic Reviews?

**Programmatic EIS/EA**: A broad environmental review that analyzes a class of similar actions (e.g., all solar projects on BLM land in California). These serve as "umbrella" documents that cover general environmental impacts.

Examples in data:
- "Solar Energy Development PEIS"
- "Wind Energy PEIS"
- "Geothermal PEIS"

### What are Tiered Reviews?

**Tiered Reviews**: Individual project-specific reviews that reference and build upon a programmatic review. They don't need to repeat analysis already done in the programmatic review, potentially making them faster.

Example language: *"This site-specific EA tiers from the 2012 Solar PEIS..."*

---

## Extraction Approach

### 3-Tier Strategy

| Tier | Method | When Used | Speed |
|------|--------|-----------|-------|
| 1 | Title detection | Always first | Instant |
| 2 | Regex with context | If title doesn't match | ~10 sec/project |
| 3 | LLM (Ollama) | Only if `--no-llm` not set AND medium-confidence matches | ~5 sec/call |

### Variables Created

| Variable | Type | Description |
|----------|------|-------------|
| `project_review_is_programmatic` | Boolean | TRUE if this project IS a programmatic review |
| `project_review_type` | Categorical | `programmatic`, `tiered`, `standard`, `unknown` |
| `project_review_tiers_from` | String | Name of the programmatic review being tiered from |
| `project_review_tiers_from_context` | String | Full context text showing the tiering relationship |
| `project_review_confidence` | Categorical | `high`, `medium`, `low` |
| `project_review_source` | String | Detection source: `title`, `text_regex`, `llm` |
| `project_review_match_text` | String | Actual matched text from document |

---

## Unit Test Results

```
=== Title Detection ===
✓ 'Programmatic Environmental Assessment...' -> programmatic=True
✓ 'New York Bight Programmatic EIS...' -> programmatic=True
✓ 'Regular Solar Farm Project...' -> programmatic=False
✓ 'Site-specific EA tiering from the Solar PEIS...' -> programmatic=False (BUG FIXED)
✓ 'Solar Energy Development PEIS...' -> programmatic=True

=== Text Extraction ===
✓ 'This EA tiers from the 2012 Solar Energy Development Programmatic EIS.'
    -> Match found, reference extracted: "2012 Solar Energy Development Programmatic EIS"
✓ 'Equipment will use EPA Tier 4 engines...' -> No match (false positive filtered)
✓ 'This is a standard environmental assessment...' -> No match (correct)
```

---

## Usage

```bash
# Test on 10 projects
python code/extract/extract_reviews.py --test

# Full extraction (EA + EIS, no LLM - faster first pass)
python code/extract/extract_reviews.py --run --no-llm

# Full extraction with LLM for ambiguous cases
python code/extract/extract_reviews.py --run

# Include CE projects (rarely have programmatic relationships)
python code/extract/extract_reviews.py --run --include-ce
```

### Check Results After Extraction

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/analysis/projects_reviews.parquet')
print(df['project_review_type'].value_counts())
print()
print('Tiered reviews:')
print(df[df['project_review_type']=='tiered'][['project_title','project_review_tiers_from']].head(10))
"
```

---

## Data Counts

| Category | Count |
|----------|-------|
| Total projects | 61,881 |
| Clean energy projects | 22,279 |
| Clean energy EA/EIS (current scope) | 1,416 |
| Projects with "programmatic" in title | 3 |
| Estimated tiered reviews | ~70-210 (5-15%) |

---

## Timeline Integration (TODO)

To answer "are tiered reviews completed faster":

1. Merge `projects_reviews.parquet` with timeline data
2. Compare duration between tiered vs standard reviews
3. Timeline extraction still being refined - revisit when ready

Key timeline variables:
- `llm_decision_date` - Final approval date
- `llm_application_date` - Start date
- Duration = decision_date - application_date

---

## Development Log

### 2026-01-30
- Created `extract_reviews.py` with 3-tier extraction approach
- Implemented title-based, regex, and LLM extraction tiers
- Added false positive filtering for EPA Tier standards, road tiers, etc.
- Fixed bug: "tiering from PEIS" incorrectly flagged as programmatic
- Unit tests passing
- **Started first full extraction** (background, `--no-llm`)
- **Next**: Review results when complete, examine found examples, refine patterns

---

## Files

- Script: `code/extract/extract_reviews.py`
- Output: `data/analysis/projects_reviews.parquet`
- This doc: `notes/status/reviews_status.md`

## Related Files

- `code/extract/extract_timeline.py` - Timeline extraction
- `code/extract/extract_gencap.py` - Generation capacity (similar pattern)
- `code/extract/extract_gencap_llm.py` - LLM extraction pattern reference
- `notes/project_overview.md` - Overall project deliverables
