# --------------------------
# PROGRAMMATIC & TIERED REVIEW EXTRACTION
# --------------------------
# Extract programmatic and tiered review information from NEPA documents
# Deliverable #2: How many tiered reviews are there compared to total,
# and are they completed faster?
#
# Strategy: Title-first, then regex with confidence scoring, LLM for ambiguous cases
#
# Usage:
#   python extract_reviews.py --test              # Test on 10 projects
#   python extract_reviews.py --run --sample 50   # Run on 50 projects
#   python extract_reviews.py --run               # Full extraction (EA + EIS)
#   python extract_reviews.py --run --include-ce  # Include CE projects

import re
import json
import pandas as pd
import requests
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --------------------------
# CONFIGURATION
# --------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b-instruct-q4_K_M"

# Confidence thresholds for LLM fallback
CONFIDENCE_HIGH = "high"      # No LLM needed
CONFIDENCE_MEDIUM = "medium"  # May need LLM verification
CONFIDENCE_LOW = "low"        # Needs LLM or manual review


# --------------------------
# REGEX PATTERNS FOR REVIEW DETECTION
# --------------------------

# Patterns indicating THIS project is a programmatic review
PROGRAMMATIC_TITLE_PATTERNS = [
    r'\bprogrammatic\b',
    r'\bprogram[\-\s]?wide\b',
    r'\bpeis\b',  # Programmatic EIS
    r'\bpea\b',   # Programmatic EA (careful - also matches other things)
]

# Patterns indicating this project TIERS FROM a programmatic review
# These are the key tiering patterns we want to extract
TIERING_PATTERNS = [
    # Direct tiering statements
    (r'(?:this|the)\s+(?:EA|EIS|environmental\s+(?:assessment|impact\s+statement))\s+(?:is\s+)?tier(?:s|ed|ing)\s+(?:to|from)\s+(?:the\s+)?(.{10,150}?)(?:\.|,|\n|$)', 'tiered_statement'),
    (r'tier(?:s|ed|ing)\s+(?:to|from)\s+(?:the\s+)?(.{10,150}?(?:PEIS|PEA|programmatic|program))(?:\.|,|\n|$)', 'tiering_to'),
    (r'(?:pursuant|according)\s+to\s+(?:the\s+)?(.{10,150}?(?:PEIS|PEA|programmatic))(?:\.|,|\n|$)', 'pursuant_to'),

    # Site-specific analysis tiering from programmatic
    (r'(?:site[\-\s]?specific|project[\-\s]?specific)\s+(?:EA|EIS|analysis)\s+(?:that\s+)?tier(?:s|ed|ing)\s+(?:to|from)\s+(?:the\s+)?(.{10,150}?)(?:\.|,|\n|$)', 'site_specific_tiering'),

    # References to programmatic reviews
    (r'(?:the\s+)?(\d{4}\s+.{10,100}?(?:PEIS|PEA|Programmatic\s+(?:EIS|EA)))(?:\s+(?:analyzed|addressed|covered))?.{0,50}?(?:tier|pursuant)', 'peis_reference'),
]

# Patterns that indicate a REFERENCE to programmatic review (not necessarily tiering)
PROGRAMMATIC_REFERENCE_PATTERNS = [
    r'(?:the\s+)?(\d{4}\s+.{5,80}?(?:PEIS|PEA))',
    r'(?:the\s+)?(.{5,80}?Programmatic\s+(?:EIS|EA|Environmental))',
    r'(?:Solar|Wind|Geothermal|Transmission)\s+(?:Energy\s+)?(?:PEIS|PEA|Programmatic)',
]

# FALSE POSITIVE PATTERNS - these should be excluded
FALSE_POSITIVE_PATTERNS = [
    r'\bEPA\s+Tier\s*[1-4]\b',           # EPA engine tiers
    r'\bTier\s*[1-4]\s+(?:engine|equipment|standard)\b',
    r'\b(?:first|second|third|top|bottom)[\-\s]?tier\b',  # Ranking tiers
    r'\bTier\s*[1-3]\s*:?\s*(?:Roads?|Primitive)\b',      # Road classifications
    r'\btiered\s+(?:pricing|rate|system|approach)\b',     # Non-NEPA tiering
]


# --------------------------
# DATA CLASSES
# --------------------------

@dataclass
class ReviewExtractionResult:
    """Result of review extraction for a single project."""
    project_id: str

    # Classification
    review_is_programmatic: bool
    review_type: str  # 'programmatic', 'tiered', 'standard', 'unknown'
    review_confidence: str  # 'high', 'medium', 'low'

    # Reference information (for tiered reviews)
    review_tiers_from: Optional[str]  # Name of programmatic review
    review_tiers_from_context: Optional[str]  # Full context text

    # Source tracking
    review_source: str  # 'title', 'text_regex', 'llm'
    review_match_text: Optional[str]  # The actual matched text

    # Metadata
    pages_scanned: int
    candidates_found: int

    def to_dict(self) -> dict:
        return {
            'project_id': self.project_id,
            'project_review_is_programmatic': self.review_is_programmatic,
            'project_review_type': self.review_type,
            'project_review_confidence': self.review_confidence,
            'project_review_tiers_from': self.review_tiers_from,
            'project_review_tiers_from_context': self.review_tiers_from_context,
            'project_review_source': self.review_source,
            'project_review_match_text': self.review_match_text,
            'project_review_pages_scanned': self.pages_scanned,
            'project_review_candidates_found': self.candidates_found,
        }


# --------------------------
# HELPER FUNCTIONS
# --------------------------

def is_false_positive(text: str) -> bool:
    """Check if text matches a false positive pattern."""
    for pattern in FALSE_POSITIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def clean_extracted_reference(ref: str) -> str:
    """Clean up an extracted programmatic review reference."""
    if not ref:
        return None

    # Remove leading/trailing whitespace and punctuation
    ref = ref.strip().strip('.,;:')

    # Remove common prefix words
    ref = re.sub(r'^(?:the|a|an)\s+', '', ref, flags=re.IGNORECASE)

    # Truncate if too long
    if len(ref) > 200:
        ref = ref[:200] + '...'

    return ref if ref else None


def extract_programmatic_reference(text: str, window: int = 200) -> Optional[str]:
    """
    Extract the name of a programmatic review from context.

    Args:
        text: Text containing reference to programmatic review
        window: Characters to search

    Returns:
        Extracted reference name or None
    """
    # Look for explicit PEIS/PEA names
    for pattern in PROGRAMMATIC_REFERENCE_PATTERNS:
        match = re.search(pattern, text[:window], re.IGNORECASE)
        if match:
            ref = match.group(1) if match.groups() else match.group(0)
            return clean_extracted_reference(ref)

    return None


# --------------------------
# TIER 1: TITLE-BASED DETECTION
# --------------------------

def check_title_for_programmatic(title: str) -> Tuple[bool, str]:
    """
    Check if project title indicates a programmatic review.

    Returns:
        (is_programmatic, confidence)
    """
    if not title:
        return False, CONFIDENCE_LOW

    title_lower = title.lower()

    # Exclude if title mentions "tiering from" - this is a tiered review, not programmatic
    if re.search(r'tier(?:s|ing|ed)?\s+(?:to|from)', title_lower):
        return False, CONFIDENCE_LOW

    # Strong indicators in title
    if 'programmatic' in title_lower:
        return True, CONFIDENCE_HIGH

    # PEIS/PEA in title (but be careful - could be referencing one)
    # Only count if PEIS appears prominently (not in "from the PEIS" context)
    if re.search(r'\bpeis\b', title_lower):
        # Exclude if it's in a "from the PEIS" context
        if not re.search(r'(?:from|pursuant\s+to)\s+(?:the\s+)?\w*\s*peis', title_lower):
            return True, CONFIDENCE_HIGH

    # PEA needs more context
    if re.search(r'\bpea\b', title_lower) and any(
        kw in title_lower for kw in ['environmental', 'assessment', 'program']
    ):
        if not re.search(r'(?:from|pursuant\s+to)\s+(?:the\s+)?\w*\s*pea', title_lower):
            return True, CONFIDENCE_MEDIUM

    return False, CONFIDENCE_LOW


# --------------------------
# TIER 2: REGEX-BASED EXTRACTION
# --------------------------

def extract_review_from_text(
    text: str,
    max_matches: int = 10
) -> list:
    """
    Extract review information from document text using regex.

    Args:
        text: Document text to search
        max_matches: Maximum matches to return

    Returns:
        List of dicts with match info
    """
    if not text or not isinstance(text, str):
        return []

    results = []
    seen_matches = set()

    # Check for tiering patterns
    for pattern, pattern_type in TIERING_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            full_match = match.group(0)

            # Skip false positives
            if is_false_positive(full_match):
                continue

            # Deduplicate
            match_key = full_match[:100]
            if match_key in seen_matches:
                continue
            seen_matches.add(match_key)

            # Extract the reference (group 1 if present)
            reference = None
            if match.groups():
                reference = clean_extracted_reference(match.group(1))

            # Get surrounding context
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].replace('\n', ' ')
            context = re.sub(r'\s+', ' ', context).strip()

            # Score confidence based on pattern type and context
            confidence = CONFIDENCE_MEDIUM
            if pattern_type in ['tiered_statement', 'tiering_to']:
                confidence = CONFIDENCE_HIGH
            elif pattern_type == 'pursuant_to':
                confidence = CONFIDENCE_MEDIUM

            results.append({
                'match': full_match,
                'pattern_type': pattern_type,
                'reference': reference,
                'context': context,
                'confidence': confidence,
                'position': match.start(),
            })

            if len(results) >= max_matches:
                break

    # Sort by confidence then position
    confidence_order = {CONFIDENCE_HIGH: 0, CONFIDENCE_MEDIUM: 1, CONFIDENCE_LOW: 2}
    results.sort(key=lambda x: (confidence_order.get(x['confidence'], 2), x['position']))

    return results


def check_text_for_programmatic(text: str) -> Tuple[bool, str, str]:
    """
    Check if text indicates this IS a programmatic review (not tiering from one).

    Returns:
        (is_programmatic, confidence, match_text)
    """
    if not text:
        return False, CONFIDENCE_LOW, None

    # Look for phrases indicating this document IS the programmatic review
    programmatic_indicators = [
        r'this\s+programmatic\s+(?:EIS|EA|environmental)',
        r'programmatic\s+(?:EIS|EA)\s+(?:is|was)\s+prepared',
        r'purpose\s+of\s+this\s+programmatic',
        r'this\s+(?:PEIS|PEA)\s+(?:analyzes|addresses|evaluates)',
    ]

    for pattern in programmatic_indicators:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return True, CONFIDENCE_HIGH, match.group(0)

    return False, CONFIDENCE_LOW, None


# --------------------------
# TIER 3: LLM EXTRACTION
# --------------------------

def build_review_prompt(candidates: list, project_title: str) -> str:
    """Build prompt for LLM to classify review type."""

    candidate_text = "\n".join([
        f"[{i+1}] ...{c['context']}..."
        for i, c in enumerate(candidates[:5])
    ])

    prompt = f"""Classify whether this NEPA project involves programmatic or tiered environmental review.

Project Title: {project_title}

Text excerpts mentioning programmatic/tiered review:
{candidate_text}

Instructions:
1. Determine if this project IS a programmatic review OR tiers FROM a programmatic review
2. If it tiers from a programmatic review, extract the name of that review
3. Return ONLY valid JSON

Return this exact JSON structure:
{{"is_programmatic": <true|false>, "review_type": "<programmatic|tiered|standard>", "tiers_from": "<name of programmatic review or null>", "confidence": "<high|medium|low>", "reasoning": "<brief explanation>"}}

If unclear or no programmatic relationship found, return:
{{"is_programmatic": false, "review_type": "standard", "tiers_from": null, "confidence": "low", "reasoning": "No clear programmatic relationship found"}}

JSON response:"""

    return prompt


def call_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: int = 120) -> Optional[str]:
    """Call Ollama API and return response text."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300,
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"Ollama API error: {e}")
        return None


def parse_llm_response(response: str) -> dict:
    """Parse LLM response into structured dict."""
    default = {
        "is_programmatic": False,
        "review_type": "unknown",
        "tiers_from": None,
        "confidence": "low",
        "reasoning": None,
        "parse_error": True
    }

    if not response:
        return default

    try:
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["parse_error"] = False
            return result
    except json.JSONDecodeError:
        pass

    return default


def extract_with_llm(
    candidates: list,
    project_title: str,
    model: str = DEFAULT_MODEL
) -> dict:
    """Use LLM to classify review type from candidate sentences."""

    if not candidates:
        return {
            "is_programmatic": False,
            "review_type": "standard",
            "tiers_from": None,
            "confidence": "low",
            "reasoning": "No candidates to analyze",
            "extraction_method": "no_candidates"
        }

    prompt = build_review_prompt(candidates, project_title)
    response = call_ollama(prompt, model=model)
    result = parse_llm_response(response)
    result["extraction_method"] = "llm"

    return result


# --------------------------
# PROJECT-LEVEL EXTRACTION
# --------------------------

def extract_review_for_project(
    project_id: str,
    project_title: str,
    documents_df: pd.DataFrame,
    pages_path: Path,
    model: str = DEFAULT_MODEL,
    max_pages: int = 30,
    use_llm: bool = True,
    verbose: bool = False
) -> ReviewExtractionResult:
    """
    Extract review information for a single project.

    Uses 3-tier approach:
    1. Title-based detection (fast, high confidence)
    2. Regex extraction with confidence scoring
    3. LLM for ambiguous cases (only if use_llm=True and confidence < high)

    Args:
        project_id: Project identifier
        project_title: Project name
        documents_df: DataFrame with document metadata
        pages_path: Path to pages.parquet file
        model: Ollama model for LLM tier
        max_pages: Maximum pages to scan
        use_llm: Whether to use LLM for ambiguous cases
        verbose: Print progress

    Returns:
        ReviewExtractionResult
    """
    import pyarrow.parquet as pq

    # Initialize result
    result = ReviewExtractionResult(
        project_id=project_id,
        review_is_programmatic=False,
        review_type='standard',
        review_confidence=CONFIDENCE_LOW,
        review_tiers_from=None,
        review_tiers_from_context=None,
        review_source='none',
        review_match_text=None,
        pages_scanned=0,
        candidates_found=0,
    )

    # ----- TIER 1: Title-based detection -----
    is_prog, title_conf = check_title_for_programmatic(project_title)
    if is_prog:
        result.review_is_programmatic = True
        result.review_type = 'programmatic'
        result.review_confidence = title_conf
        result.review_source = 'title'
        result.review_match_text = project_title
        if verbose:
            print(f"  Title match: programmatic ({title_conf})")
        return result

    # ----- TIER 2: Regex extraction from documents -----
    project_docs = documents_df[documents_df['project_id'] == project_id]

    if project_docs.empty:
        result.review_source = 'no_documents'
        return result

    # Prioritize main documents
    if 'main_document' in project_docs.columns:
        main_docs = project_docs[project_docs['main_document'] == 'YES']
        if not main_docs.empty:
            project_docs = main_docs

    doc_ids = project_docs['document_id'].tolist()

    # Read pages
    try:
        pages_table = pq.read_table(pages_path, filters=[('document_id', 'in', doc_ids)])
        pages_df = pages_table.to_pandas()
    except Exception as e:
        if verbose:
            print(f"  Error reading pages: {e}")
        result.review_source = 'error_reading_pages'
        return result

    if pages_df.empty:
        result.review_source = 'no_pages'
        return result

    # Sort by page number, focus on early pages
    pages_df = pages_df.sort_values('page_number')
    pages_to_check = min(max_pages, len(pages_df))

    all_candidates = []
    pages_scanned = 0

    for _, page in pages_df.head(pages_to_check).iterrows():
        pages_scanned += 1
        text = page.get('page_text', '') or ''

        # Check if this IS a programmatic document
        is_prog, prog_conf, prog_match = check_text_for_programmatic(text)
        if is_prog and prog_conf == CONFIDENCE_HIGH:
            result.review_is_programmatic = True
            result.review_type = 'programmatic'
            result.review_confidence = prog_conf
            result.review_source = 'text_regex'
            result.review_match_text = prog_match
            result.pages_scanned = pages_scanned
            if verbose:
                print(f"  Text match: programmatic ({prog_conf})")
            return result

        # Extract tiering candidates
        candidates = extract_review_from_text(text)
        all_candidates.extend(candidates)

        # If we have a high-confidence tiering match, use it
        high_conf = [c for c in candidates if c['confidence'] == CONFIDENCE_HIGH]
        if high_conf:
            best = high_conf[0]
            result.review_type = 'tiered'
            result.review_confidence = CONFIDENCE_HIGH
            result.review_tiers_from = best['reference']
            result.review_tiers_from_context = best['context']
            result.review_source = 'text_regex'
            result.review_match_text = best['match']
            result.pages_scanned = pages_scanned
            result.candidates_found = len(all_candidates)
            if verbose:
                print(f"  Text match: tiered from '{best['reference']}' ({CONFIDENCE_HIGH})")
            return result

    result.pages_scanned = pages_scanned
    result.candidates_found = len(all_candidates)

    # ----- TIER 3: LLM for ambiguous cases -----
    if all_candidates and use_llm:
        # Only use LLM if we have medium-confidence candidates
        medium_conf = [c for c in all_candidates if c['confidence'] == CONFIDENCE_MEDIUM]

        if medium_conf:
            if verbose:
                print(f"  Using LLM for {len(medium_conf)} medium-confidence candidates...")

            llm_result = extract_with_llm(medium_conf, project_title, model=model)

            if not llm_result.get('parse_error', True):
                if llm_result.get('is_programmatic'):
                    result.review_is_programmatic = True
                    result.review_type = 'programmatic'
                elif llm_result.get('review_type') == 'tiered':
                    result.review_type = 'tiered'
                    result.review_tiers_from = llm_result.get('tiers_from')
                    result.review_tiers_from_context = llm_result.get('reasoning')

                result.review_confidence = llm_result.get('confidence', CONFIDENCE_LOW)
                result.review_source = 'llm'
                result.review_match_text = medium_conf[0]['match'] if medium_conf else None

                if verbose:
                    print(f"  LLM result: {result.review_type} ({result.review_confidence})")

                return result

    # No clear programmatic/tiered relationship found
    result.review_type = 'standard'
    result.review_confidence = CONFIDENCE_HIGH if not all_candidates else CONFIDENCE_MEDIUM
    result.review_source = 'text_regex'

    return result


# --------------------------
# BATCH EXTRACTION
# --------------------------

def run_review_extraction(
    clean_energy_only: bool = True,
    include_ce: bool = False,
    sample_size: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    use_llm: bool = True,
    verbose: bool = True,
    output_path: Optional[str] = None,
    workers: int = 1,
) -> pd.DataFrame:
    """
    Run review extraction for multiple projects.

    Args:
        clean_energy_only: Only process clean energy projects
        include_ce: Include CE (Categorical Exclusion) projects
        sample_size: Limit to N projects (for testing)
        model: Ollama model for LLM tier
        use_llm: Whether to use LLM for ambiguous cases
        verbose: Print progress
        output_path: Custom output path
        workers: Number of parallel workers (1 = sequential)

    Returns:
        DataFrame with extraction results
    """
    print("\n=== Programmatic & Tiered Review Extraction ===")
    print(f"LLM: {'enabled' if use_llm else 'disabled'} (model: {model})")
    print(f"Include CE: {include_ce}")

    # Load projects
    projects_path = ANALYSIS_DIR / "projects_combined.parquet"
    if not projects_path.exists():
        print(f"Error: {projects_path} not found. Run extract_data.py first.")
        return None

    projects = pd.read_parquet(projects_path)
    print(f"Loaded {len(projects):,} total projects")

    # Filter by energy type
    if clean_energy_only:
        projects = projects[projects['project_energy_type'] == 'Clean']
        print(f"Filtered to {len(projects):,} clean energy projects")

    # Filter by dataset source (exclude CE by default)
    if not include_ce:
        projects = projects[projects['dataset_source'].isin(['EA', 'EIS'])]
        print(f"Filtered to {len(projects):,} EA/EIS projects (excluding CE)")

    if sample_size:
        projects = projects.sample(min(sample_size, len(projects)), random_state=42)
        print(f"Sampled {len(projects):,} projects")

    if projects.empty:
        print("No projects to process after filtering.")
        return None

    # Process by source
    import time
    results = []
    sources = list(projects['dataset_source'].unique())

    # Running counts
    n_programmatic_found = 0
    n_tiered_found = 0

    for source in sources:
        source_projects = projects[projects['dataset_source'] == source]
        total = len(source_projects)
        print(f"\n--- Processing {source} ({total} projects) ---")

        data_dir = PROCESSED_DIR / source.lower()

        docs_path = data_dir / "documents.parquet"
        pages_path = data_dir / "pages.parquet"

        documents_df = pd.read_parquet(docs_path)

        # Clean project_id in documents
        def extract_id(x):
            return x.get('value', '') if isinstance(x, dict) else x
        documents_df['project_id'] = documents_df['project_id'].apply(extract_id)

        start_time = time.time()

        for idx, (_, project) in enumerate(source_projects.iterrows()):
            project_id = project['project_id']
            project_title = project.get('project_title', '')

            result = extract_review_for_project(
                project_id=project_id,
                project_title=project_title,
                documents_df=documents_df,
                pages_path=pages_path,
                model=model,
                use_llm=use_llm,
                verbose=False,
            )

            result_dict = result.to_dict()
            result_dict['dataset_source'] = source
            results.append(result_dict)

            # Track counts
            if result.review_type == 'programmatic':
                n_programmatic_found += 1
            elif result.review_type == 'tiered':
                n_tiered_found += 1

            # Progress output every 10 projects
            if verbose and (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                remaining = (total - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx + 1}/{total}] {rate:.1f} proj/sec | "
                      f"~{remaining/60:.1f} min left | "
                      f"Found: {n_programmatic_found} prog, {n_tiered_found} tiered")

        # Source complete
        elapsed = time.time() - start_time
        print(f"  Completed {source} in {elapsed/60:.1f} min")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Merge with project metadata
    projects_with_reviews = projects.merge(
        results_df,
        on=['project_id', 'dataset_source'],
        how='left'
    )

    # Save
    if output_path:
        save_path = Path(output_path)
    else:
        save_path = ANALYSIS_DIR / "projects_reviews.parquet"

    projects_with_reviews.to_parquet(save_path)
    print(f"\nSaved to: {save_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"Total projects processed: {len(projects_with_reviews):,}")
    print(f"\nReview type distribution:")
    print(projects_with_reviews['project_review_type'].value_counts())
    print(f"\nConfidence distribution:")
    print(projects_with_reviews['project_review_confidence'].value_counts())
    print(f"\nSource distribution:")
    print(projects_with_reviews['project_review_source'].value_counts())

    # Count programmatic and tiered
    n_programmatic = (projects_with_reviews['project_review_type'] == 'programmatic').sum()
    n_tiered = (projects_with_reviews['project_review_type'] == 'tiered').sum()
    print(f"\nProgrammatic reviews: {n_programmatic:,}")
    print(f"Tiered reviews: {n_tiered:,}")

    if n_tiered > 0:
        # Show sample tiered reviews
        tiered = projects_with_reviews[projects_with_reviews['project_review_type'] == 'tiered']
        print(f"\nSample tiered reviews:")
        sample_cols = ['project_title', 'project_review_tiers_from', 'project_review_confidence']
        print(tiered[sample_cols].head(5).to_string())

    return projects_with_reviews


# --------------------------
# CLI
# --------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract programmatic and tiered review information from NEPA documents"
    )

    parser.add_argument('--run', action='store_true',
                        help='Run extraction')
    parser.add_argument('--test', action='store_true',
                        help='Test on 10 projects')
    parser.add_argument('--sample', type=int,
                        help='Sample N projects for testing')
    parser.add_argument('--include-ce', action='store_true',
                        help='Include CE (Categorical Exclusion) projects')
    parser.add_argument('--all-projects', action='store_true',
                        help='Process all projects, not just clean energy')
    parser.add_argument('--no-llm', action='store_true',
                        help='Disable LLM for ambiguous cases')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'Ollama model (default: {DEFAULT_MODEL})')
    parser.add_argument('--output', type=str,
                        help='Custom output path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.test:
        print("Running test on 10 projects...")
        results = run_review_extraction(
            clean_energy_only=True,
            include_ce=False,
            sample_size=10,
            model=args.model,
            use_llm=not args.no_llm,
            verbose=True,
        )

        if results is not None:
            print("\n=== Detailed Results ===")
            display_cols = [
                'project_title', 'project_review_type',
                'project_review_tiers_from', 'project_review_confidence',
                'project_review_source'
            ]
            # Truncate title for display
            results['project_title'] = results['project_title'].str[:50]
            print(results[display_cols].to_string())

    elif args.run:
        run_review_extraction(
            clean_energy_only=not args.all_projects,
            include_ce=args.include_ce,
            sample_size=args.sample,
            model=args.model,
            use_llm=not args.no_llm,
            verbose=args.verbose,
            output_path=args.output,
        )

    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  python extract_reviews.py --test              # Test on 10 projects")
        print("  python extract_reviews.py --run --sample 50   # Sample 50 projects")
        print("  python extract_reviews.py --run               # Full EA/EIS extraction")
        print("  python extract_reviews.py --run --include-ce  # Include CE projects")
        print("  python extract_reviews.py --run --no-llm      # Regex only (no LLM)")


if __name__ == "__main__":
    main()
