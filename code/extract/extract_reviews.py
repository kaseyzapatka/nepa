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
#   python extract_reviews.py --run --use-llm     # Enable LLM fallback on ambiguous cases

import re
import json
import pandas as pd
import requests
import duckdb
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

# Default page window to inspect for review signals.
DEFAULT_MAX_PAGES = 60


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

# Strong programmatic indicators in metadata/title/text
PROGRAMMATIC_STRONG_PATTERNS = [
    r'(?:draft|final|supplemental)\s+programmatic\s+environmental\s+(?:impact\s+statement|assessment)',
    r'programmatic\s+environmental\s+(?:impact\s+statement|assessment)',
    r'\b(?:dpeis|fpeis|speis|peis|pea)\b',
    r'this\s+programmatic\s+(?:eis|ea|environmental)',
    r'this\s+(?:peis|pea)\s+(?:analyzes|addresses|evaluates)',
]

# Medium-confidence synonyms that may be used for umbrella review documents
PROGRAMMATIC_MEDIUM_PATTERNS = [
]

# Stand-in terminology ("generic" / "tier 1"), enabled by default.
GENERIC_STANDIN_PATTERNS = [
    r'\bgeneric\s+(?:environmental\s+(?:impact\s+statement|assessment)|eis|ea)\b',
    r'\btier\s*(?:1|i|one)\s+(?:nepa\s+)?(?:review|environmental\s+(?:impact\s+statement|assessment)|eis|ea)\b',
    r'\b(?:environmental\s+(?:impact\s+statement|assessment)|eis|ea)\s+tier\s*(?:1|i|one)\b',
]

# Non-NEPA or ancillary uses of "programmatic" that should not classify a project
PROGRAMMATIC_EXCLUSION_PATTERNS = [
    r'programmatic\s+agreement',
    r'programmatic\s+biological\s+opinion',
    r'programmatic\s+consultation',
    r'programmatic\s+collaboration',
]

# Patterns indicating this project TIERS FROM a programmatic review
# These are the key tiering patterns we want to extract
TIERING_PATTERNS = [
    # Direct tiering statements
    (r'(?:this|the)\s+(?:EA|EIS|environmental\s+(?:assessment|impact\s+statement))\s+(?:is\s+)?tier(?:s|ed|ing)\s+(?:to|from)\s+(?:the\s+)?(.{10,150}?)(?:\.|,|\n|$)', 'tiered_statement'),
    (r'tier(?:s|ed|ing)\s+(?:to|from)\s+(?:the\s+)?(.{10,150}?(?:PEIS|PEA|programmatic|program))(?:\.|,|\n|$)', 'tiering_to'),
    (r'(?:pursuant|according)\s+to\s+(?:the\s+)?(.{10,150}?(?:PEIS|PEA|programmatic))(?:\.|,|\n|$)', 'pursuant_to'),
    (r'(?:incorporat(?:e|es|ed|ing)\s+by\s+reference|adopt(?:s|ed|ing)\s+the\s+analysis\s+in|build(?:s|ing)\s+upon)\s+(?:the\s+)?(.{10,150}?(?:PEIS|PEA|programmatic))(?:\.|,|\n|$)', 'reference_adoption'),

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
    review_source: str  # 'title', 'doc_metadata', 'text_regex', 'llm'
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


def normalize_text(text: str) -> str:
    """Normalize spacing in extracted text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', str(text)).strip()


def contains_programmatic_exclusion(text: str) -> bool:
    """Check whether text contains programmatic false-positive language."""
    text = normalize_text(text).lower()
    if not text:
        return False

    for pattern in PROGRAMMATIC_EXCLUSION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def normalize_project_id(value):
    """Unwrap project_id values that may be stored as dicts."""
    return value.get('value', '') if isinstance(value, dict) else value


def build_project_document_lookup(
    documents_df: pd.DataFrame,
    project_ids: set,
) -> Tuple[dict, pd.DataFrame]:
    """
    Build per-project document lookup and selected document_id map.

    If a project has one or more `main_document == YES` rows, only those
    rows are retained to match existing extraction behavior.
    """
    if documents_df is None or documents_df.empty or not project_ids:
        return {}, pd.DataFrame(columns=['project_id', 'document_id'])

    docs = documents_df.copy()
    docs['project_id'] = docs['project_id'].apply(normalize_project_id)
    docs = docs[docs['project_id'].isin(project_ids)].copy()
    if docs.empty:
        return {}, pd.DataFrame(columns=['project_id', 'document_id'])

    project_doc_lookup = {}
    selected_pairs = []

    for project_id, project_docs in docs.groupby('project_id', sort=False):
        selected_docs = project_docs
        if 'main_document' in selected_docs.columns:
            main_docs = selected_docs[selected_docs['main_document'].fillna('').str.upper() == 'YES']
            if not main_docs.empty:
                selected_docs = main_docs

        project_doc_lookup[project_id] = selected_docs.copy()
        selected_pairs.append(selected_docs[['project_id', 'document_id']])

    if selected_pairs:
        doc_pairs = pd.concat(selected_pairs, ignore_index=True).drop_duplicates()
    else:
        doc_pairs = pd.DataFrame(columns=['project_id', 'document_id'])

    return project_doc_lookup, doc_pairs


def load_project_pages_with_duckdb(
    pages_path: Path,
    document_pairs: pd.DataFrame,
    max_pages: int,
) -> dict:
    """
    Load top-N ordered pages per project using DuckDB for fast bulk retrieval.
    """
    if document_pairs is None or document_pairs.empty:
        return {}

    pages_path_sql = pages_path.as_posix().replace("'", "''")

    con = duckdb.connect()
    try:
        con.register('project_docs', document_pairs[['project_id', 'document_id']])
        query = f"""
        WITH joined AS (
            SELECT
                d.project_id,
                p.page_text,
                CAST(p.page_number AS VARCHAR) AS page_number,
                COALESCE(
                    TRY_CAST(regexp_extract(CAST(p.page_number AS VARCHAR), '(\\d+)', 1) AS INTEGER),
                    1000000000
                ) AS page_num
            FROM read_parquet('{pages_path_sql}') p
            INNER JOIN project_docs d USING (document_id)
        ),
        ranked AS (
            SELECT
                project_id,
                page_text,
                row_number() OVER (
                    PARTITION BY project_id
                    ORDER BY page_num, page_number
                ) AS rn
            FROM joined
        )
        SELECT project_id, page_text
        FROM ranked
        WHERE rn <= {int(max_pages)}
        ORDER BY project_id, rn
        """
        pages_df = con.execute(query).df()
    finally:
        con.close()

    if pages_df.empty:
        return {}

    pages_lookup = {}
    for project_id, group in pages_df.groupby('project_id', sort=False):
        pages_lookup[project_id] = [
            text if isinstance(text, str) else ''
            for text in group['page_text'].tolist()
        ]

    return pages_lookup


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

def check_title_for_programmatic(title: str, include_generic: bool = True) -> Tuple[bool, str]:
    """
    Check if project title indicates a programmatic review.

    Returns:
        (is_programmatic, confidence)
    """
    if not title:
        return False, CONFIDENCE_LOW

    title_clean = normalize_text(title)
    title_lower = title_clean.lower()

    # Exclude if title mentions "tiering from" - this is a tiered review, not programmatic
    if re.search(r'tier(?:s|ing|ed)?\s+(?:to|from)', title_lower):
        return False, CONFIDENCE_LOW

    # Exclude known non-review programmatic language
    if contains_programmatic_exclusion(title_lower):
        return False, CONFIDENCE_LOW

    # Strong phrase match
    for pattern in PROGRAMMATIC_STRONG_PATTERNS:
        if re.search(pattern, title_lower, re.IGNORECASE):
            # Prevent "from the PEIS/PEA" title references from being marked programmatic
            if re.search(r'(?:from|pursuant\s+to|tier(?:s|ing|ed)?\s+(?:to|from))\s+(?:the\s+)?\w*\s*(?:peis|pea)', title_lower):
                continue
            return True, CONFIDENCE_HIGH

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

    # Optional "Generic EIS/EA" stand-in terminology
    generic_patterns = GENERIC_STANDIN_PATTERNS if include_generic else []
    for pattern in generic_patterns:
        if re.search(pattern, title_lower, re.IGNORECASE):
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
            context = normalize_text(text[start:end])

            # Score confidence based on pattern type and context
            confidence = CONFIDENCE_MEDIUM
            if pattern_type in ['tiered_statement', 'tiering_to', 'site_specific_tiering']:
                confidence = CONFIDENCE_HIGH
            elif pattern_type in ['pursuant_to', 'reference_adoption']:
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


def check_text_for_programmatic(text: str, include_generic: bool = True) -> Tuple[bool, str, str]:
    """
    Check if text indicates this IS a programmatic review (not tiering from one).

    Returns:
        (is_programmatic, confidence, match_text)
    """
    if not text:
        return False, CONFIDENCE_LOW, None

    text_clean = normalize_text(text)
    if contains_programmatic_exclusion(text_clean):
        return False, CONFIDENCE_LOW, None

    # Look for phrases indicating this document IS the umbrella review
    strong_indicators = [
        r'this\s+programmatic\s+(?:eis|ea|environmental)',
        r'programmatic\s+(?:eis|ea)\s+(?:is|was)\s+prepared',
        r'purpose\s+of\s+this\s+programmatic',
        r'this\s+(?:peis|pea)\s+(?:analyzes|addresses|evaluates)',
    ]

    for pattern in strong_indicators:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            return True, CONFIDENCE_HIGH, match.group(0)

    medium_indicators = GENERIC_STANDIN_PATTERNS if include_generic else []

    for pattern in medium_indicators:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            return True, CONFIDENCE_MEDIUM, match.group(0)

    return False, CONFIDENCE_LOW, None


def check_document_metadata_for_programmatic(
    project_docs: pd.DataFrame,
    include_generic: bool = True
) -> Tuple[bool, str, Optional[str]]:
    """
    Check file/document metadata for programmatic indicators.

    This helps recover true positives when OCR text is weak on early pages.
    """
    if project_docs is None or project_docs.empty:
        return False, CONFIDENCE_LOW, None

    # Prefer main documents first when available.
    docs = project_docs.copy()
    if 'main_document' in docs.columns:
        docs['_main_first'] = (docs['main_document'].fillna('').str.upper() == 'YES').astype(int)
        docs = docs.sort_values('_main_first', ascending=False)

    medium_match = None
    for _, row in docs.iterrows():
        for field in ['document_title', 'file_name']:
            value = normalize_text(row.get(field, ''))
            if not value:
                continue

            value_lower = value.lower()
            if contains_programmatic_exclusion(value_lower):
                continue

            for pattern in PROGRAMMATIC_STRONG_PATTERNS:
                if re.search(pattern, value_lower, re.IGNORECASE):
                    if re.search(r'(?:from|pursuant\s+to|tier(?:s|ing|ed)?\s+(?:to|from))\s+(?:the\s+)?\w*\s*(?:peis|pea)', value_lower):
                        continue
                    return True, CONFIDENCE_HIGH, value

            generic_patterns = GENERIC_STANDIN_PATTERNS if include_generic else []
            for pattern in generic_patterns:
                if re.search(pattern, value_lower, re.IGNORECASE):
                    medium_match = value

    if medium_match:
        return True, CONFIDENCE_MEDIUM, medium_match

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
    project_docs: Optional[pd.DataFrame],
    project_pages: Optional[list],
    model: str = DEFAULT_MODEL,
    max_pages: int = DEFAULT_MAX_PAGES,
    use_llm: bool = False,
    include_generic: bool = True,
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
        project_docs: Document metadata for this project
        project_pages: Ordered page text for this project
        model: Ollama model for LLM tier
        max_pages: Maximum pages to scan
        use_llm: Whether to use LLM for ambiguous cases
        verbose: Print progress

    Returns:
        ReviewExtractionResult
    """
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
    is_prog, title_conf = check_title_for_programmatic(project_title, include_generic=include_generic)
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
    if project_docs is None or project_docs.empty:
        result.review_source = 'no_documents'
        return result

    # Metadata fallback before page-level OCR search
    meta_is_prog, meta_conf, meta_match = check_document_metadata_for_programmatic(
        project_docs,
        include_generic=include_generic,
    )
    if meta_is_prog and meta_conf == CONFIDENCE_HIGH:
        result.review_is_programmatic = True
        result.review_type = 'programmatic'
        result.review_confidence = meta_conf
        result.review_source = 'doc_metadata'
        result.review_match_text = meta_match
        return result

    page_texts = project_pages or []
    if not page_texts:
        result.review_source = 'no_pages'
        return result

    pages_to_check = min(max_pages, len(page_texts))

    all_candidates = []
    pages_scanned = 0

    for text in page_texts[:pages_to_check]:
        pages_scanned += 1

        # Check if this IS a programmatic document
        is_prog, prog_conf, prog_match = check_text_for_programmatic(
            text,
            include_generic=include_generic,
        )
        if is_prog and prog_conf in [CONFIDENCE_HIGH, CONFIDENCE_MEDIUM]:
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

    # Medium-confidence metadata fallback after text scan.
    if meta_is_prog and meta_conf == CONFIDENCE_MEDIUM:
        result.review_is_programmatic = True
        result.review_type = 'programmatic'
        result.review_confidence = meta_conf
        result.review_source = 'doc_metadata'
        result.review_match_text = meta_match
        return result

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
    sample_size: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    use_llm: bool = False,
    verbose: bool = True,
    output_path: Optional[str] = None,
    workers: int = 1,
    max_pages: int = DEFAULT_MAX_PAGES,
    include_generic: bool = True,
) -> pd.DataFrame:
    """
    Run review extraction for multiple projects.

    Args:
        sample_size: Limit to N projects (for testing)
        model: Ollama model for LLM tier
        use_llm: Whether to use LLM for ambiguous cases
        verbose: Print progress
        output_path: Custom output path
        workers: Number of parallel workers (1 = sequential)
        max_pages: Maximum pages to inspect per project
        include_generic: Whether to treat "Generic" / "Tier 1 NEPA review" phrases as stand-ins for programmatic review

    Returns:
        DataFrame with extraction results
    """
    print("\n=== Programmatic & Tiered Review Extraction ===")
    print(f"LLM: {'enabled' if use_llm else 'disabled'} (model: {model})")
    print("Scope: Clean energy EA/EIS projects")
    print(f"Include Generic/Tier 1 stand-ins: {include_generic}")

    # Load projects
    projects_path = ANALYSIS_DIR / "projects_combined.parquet"
    if not projects_path.exists():
        print(f"Error: {projects_path} not found. Run extract_data.py first.")
        return None

    projects = pd.read_parquet(projects_path)
    print(f"Loaded {len(projects):,} total projects")

    # Fixed scope: clean energy EA/EIS projects only
    projects = projects[projects['project_energy_type'] == 'Clean']
    print(f"Filtered to {len(projects):,} clean energy projects")
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

        start_time = time.time()
        project_inputs = [
            (project['project_id'], project.get('project_title', ''))
            for _, project in source_projects.iterrows()
        ]
        project_ids = {project_id for project_id, _ in project_inputs}

        # Build source-level doc/page lookups once, then classify projects from memory.
        doc_columns = ['project_id', 'document_id', 'document_title', 'file_name', 'main_document']
        documents_df = pd.read_parquet(docs_path, columns=doc_columns)
        project_docs_lookup, document_pairs = build_project_document_lookup(
            documents_df=documents_df,
            project_ids=project_ids,
        )
        project_pages_lookup = load_project_pages_with_duckdb(
            pages_path=pages_path,
            document_pairs=document_pairs,
            max_pages=max_pages,
        )

        if verbose:
            elapsed_prep = time.time() - start_time
            n_docs = len(document_pairs)
            n_projects_with_pages = len(project_pages_lookup)
            print(f"  Prepared source caches in {elapsed_prep:.1f}s | "
                  f"{n_docs:,} docs | {n_projects_with_pages:,} projects with pages")

        if workers and workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_project = {
                    executor.submit(
                        extract_review_for_project,
                        project_id=project_id,
                        project_title=project_title,
                        project_docs=project_docs_lookup.get(project_id),
                        project_pages=project_pages_lookup.get(project_id, []),
                        model=model,
                        max_pages=max_pages,
                        use_llm=use_llm,
                        include_generic=include_generic,
                        verbose=False,
                    ): (project_id, project_title)
                    for project_id, project_title in project_inputs
                }

                completed = 0
                for future in as_completed(future_to_project):
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as e:
                        project_id, _ = future_to_project[future]
                        if verbose:
                            print(f"  Error processing {project_id}: {e}")
                        result = ReviewExtractionResult(
                            project_id=project_id,
                            review_is_programmatic=False,
                            review_type='unknown',
                            review_confidence=CONFIDENCE_LOW,
                            review_tiers_from=None,
                            review_tiers_from_context=None,
                            review_source='error',
                            review_match_text=None,
                            pages_scanned=0,
                            candidates_found=0,
                        )

                    result_dict = result.to_dict()
                    result_dict['dataset_source'] = source
                    results.append(result_dict)

                    if result.review_type == 'programmatic':
                        n_programmatic_found += 1
                    elif result.review_type == 'tiered':
                        n_tiered_found += 1

                    if verbose and completed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = (total - completed) / rate if rate > 0 else 0
                        print(f"  [{completed}/{total}] {rate:.1f} proj/sec | "
                              f"~{remaining/60:.1f} min left | "
                              f"Found: {n_programmatic_found} prog, {n_tiered_found} tiered")
        else:
            for idx, (project_id, project_title) in enumerate(project_inputs):
                result = extract_review_for_project(
                    project_id=project_id,
                    project_title=project_title,
                    project_docs=project_docs_lookup.get(project_id),
                    project_pages=project_pages_lookup.get(project_id, []),
                    model=model,
                    max_pages=max_pages,
                    use_llm=use_llm,
                    include_generic=include_generic,
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
    parser.add_argument('--use-llm', action='store_true',
                        help='Enable LLM fallback for ambiguous cases (default: off)')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'Ollama model (default: {DEFAULT_MODEL})')
    parser.add_argument('--output', type=str,
                        help='Custom output path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--max-pages', type=int, default=DEFAULT_MAX_PAGES,
                        help=f'Max pages to scan per project (default: {DEFAULT_MAX_PAGES})')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of concurrent workers for project extraction (default: 1)')

    args = parser.parse_args()

    if args.test:
        print("Running test on 10 projects...")
        test_output = ANALYSIS_DIR / "projects_reviews_test.parquet"
        results = run_review_extraction(
            sample_size=10,
            model=args.model,
            use_llm=args.use_llm,
            verbose=True,
            max_pages=args.max_pages,
            output_path=str(test_output),
            workers=args.workers,
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
            sample_size=args.sample,
            model=args.model,
            use_llm=args.use_llm,
            verbose=args.verbose,
            output_path=args.output,
            max_pages=args.max_pages,
            workers=args.workers,
        )

    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  python extract_reviews.py --test              # Test on 10 projects")
        print("  python extract_reviews.py --run --sample 50   # Sample 50 projects")
        print("  python extract_reviews.py --run               # Full EA/EIS extraction")
        print("  python extract_reviews.py --run --use-llm     # Enable LLM fallback")


if __name__ == "__main__":
    main()
