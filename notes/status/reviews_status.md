# Deliverable #2: Programmatic & Tiered Reviews

**Status**: In Development
**Script**: `code/extract/extract_reviews.py`
**Output**: `data/analysis/projects_reviews.parquet`

## Deliverable Goal

Data on programmatic and tiered reviews: how many tiered reviews are there compared to total, and are they completed faster?

## Background

### What are Programmatic Reviews?

**Programmatic EIS/EA**: A broad environmental review that analyzes a class of similar actions (e.g., all solar projects on BLM land in California). These serve as "umbrella" documents that cover general environmental impacts.

Examples found in data:
- "Solar Energy Development PEIS"
- "Wind Energy PEIS"
- "Geothermal PEIS"

### What are Tiered Reviews?

**Tiered Reviews**: Individual project-specific reviews that reference and build upon a programmatic review. They don't need to repeat analysis already done in the programmatic review, potentially making them faster.

Example language: *"This site-specific EA tiers from the 2012 Solar PEIS..."*

## Extraction Approach

### 3-Tier Strategy

| Tier | Method | Confidence | Speed |
|------|--------|------------|-------|
| 1 | Title-based detection | High | Fast |
| 2 | Regex with context filtering | Medium-High | Medium |
| 3 | LLM (Ollama) for ambiguous cases | Variable | Slow |

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
| `project_review_pages_scanned` | Integer | Number of pages examined |
| `project_review_candidates_found` | Integer | Number of potential matches found |

## Current Scope

- **Included**: EA and EIS projects (these have programmatic/tiered relationships)
- **Excluded by default**: CE projects (rarely have these relationships)
- **Filter**: Clean energy projects only (matching deliverable scope)

## Usage

```bash
# Test on 10 projects
python code/extract/extract_reviews.py --test

# Sample run (50 projects)
python code/extract/extract_reviews.py --run --sample 50

# Full extraction (EA + EIS clean energy)
python code/extract/extract_reviews.py --run

# Include CE projects
python code/extract/extract_reviews.py --run --include-ce

# Disable LLM (regex only)
python code/extract/extract_reviews.py --run --no-llm
```

## Expected Prevalence

Based on sample analysis (50 clean energy projects):

| Review Type | Expected % |
|-------------|-----------|
| Programmatic | ~1-2% (projects that ARE programmatic reviews) |
| Tiered | ~5-15% of EA/EIS (projects tiering FROM programmatic) |
| Standard | ~85-95% (no programmatic relationship) |

## Key Findings from Sample Analysis

1. **37 clean energy projects** have "programmatic" explicitly in title
2. **Term "tier" has many false positives** - EPA Tier 1-4 engines, road classifications
3. **CEs rarely mention programmatic relationships** - minimal documentation
4. **EA/EIS documents frequently reference PEISs** but many are just citations, not tiering

## False Positive Handling

The script filters out these common false positives:
- EPA Tier 1/2/3/4 engine standards
- Road tier classifications
- Tiered pricing/rate systems
- First-tier/second-tier rankings

## Timeline Integration (TODO)

To answer "are tiered reviews completed faster", we need to combine this data with timeline data:

1. **Current**: `extract_timeline.py` extracts dates from documents
2. **Needed**: Merge `projects_reviews.parquet` with `projects_timeline.parquet`
3. **Analysis**: Compare duration (decision_date - start_date) between tiered vs standard reviews

### Timeline Variables to Use

From `extract_timeline.py`:
- `project_year` - Year of decision
- `llm_decision_date` - Final approval date
- `llm_application_date` - Start date
- Duration can be calculated as difference

**Note**: Timeline extraction is still being refined. Will revisit integration when timeline data is more complete.

## Development Log

### 2026-01-30
- Created `extract_reviews.py` with 3-tier extraction approach
- Implemented title-based, regex, and LLM extraction tiers
- Added false positive filtering for EPA Tier standards, etc.
- Created this status document
- Next: Run test extraction, validate results

## Files

- Script: `code/extract/extract_reviews.py`
- Output: `data/analysis/projects_reviews.parquet`
- This doc: `notes/reviews_status.md`

## Related Files

- `code/extract/extract_timeline.py` - Timeline extraction
- `code/extract/extract_gencap.py` - Generation capacity (similar pattern)
- `code/extract/extract_gencap_llm.py` - LLM extraction pattern reference
- `notes/project_overview.md` - Overall project deliverables
