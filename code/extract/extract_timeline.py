# --------------------------
# TIMELINE EXTRACTION
# --------------------------
# Extract dates from document text to construct project timelines
# Strategy: Regex first to find all dates, then order them chronologically

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------
# FILE PATHS
# --------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"


# --------------------------
# DATE PATTERNS
# --------------------------

# Month names for pattern matching
MONTHS = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
MONTHS_SHORT = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'

# Date patterns (ordered by specificity)
DATE_PATTERNS = [
    # "January 15, 2024" or "January 15 2024"
    (rf'({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})', 'MDY_full'),
    # "Jan 15, 2024" or "Jan. 15, 2024"
    (rf'({MONTHS_SHORT})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})', 'MDY_short'),
    # "15 January 2024"
    (rf'(\d{{1,2}})\s+({MONTHS})\s+(\d{{4}})', 'DMY_full'),
    # "01/15/2024" or "1/15/2024"
    (r'(\d{1,2})/(\d{1,2})/(\d{4})', 'numeric_slash'),
    # "01-15-2024" or "2024-01-15" (ISO format)
    (r'(\d{4})-(\d{1,2})-(\d{1,2})', 'ISO'),
    (r'(\d{1,2})-(\d{1,2})-(\d{4})', 'numeric_dash'),
    # "January 2024" (month-year only)
    (rf'({MONTHS})\s+(\d{{4}})', 'MY_full'),
    # "Jan 2024"
    (rf'({MONTHS_SHORT})\.?\s+(\d{{4}})', 'MY_short'),
]

# Context keywords that indicate what type of date this is
DATE_CONTEXT_KEYWORDS = {
    'start': ['commenced', 'initiated', 'began', 'starting', 'start date', 'initiated on'],
    'submission': ['submitted', 'filed', 'received', 'application date'],
    'notice': ['notice of intent', 'NOI', 'published', 'federal register'],
    'draft': ['draft', 'DEIS', 'DEA'],
    'final': ['final', 'FEIS', 'FEA'],
    'decision': ['decision', 'ROD', 'record of decision', 'FONSI', 'finding of no significant impact',
                 'approved', 'signed', 'issued'],
    'comment': ['comment period', 'public comment', 'comments due'],
    'scoping': ['scoping', 'scoping period'],
}

# Context keywords that indicate a date should be EXCLUDED (law/statute years, references)
DATE_EXCLUSION_KEYWORDS = [
    # Law/statute references
    'act of', 'act (', 'policy act', 'preservation act', 'conservation act',
    'management act', 'protection act', 'improvement act', 'reform act',
    'recovery act', 'species act', 'water act', 'air act', 'lands act',
    'flpma', 'nepa', 'nhpa', 'anilca', 'cercla', 'rcra', 'esa', 'cwa', 'caa',
    'statute', 'u.s.c.', 'usc', 'public law', 'p.l.', 'amended in',
    # Bibliographic references (citations)
    'accessed', 'retrieved', 'available at', 'http://', 'https://', 'www.',
    'et al.', 'et al,', 'eds.', 'editor', 'vol.', 'volume', 'pp.', 'pages',
    'journal', 'publication', 'proceedings', 'report no.', 'technical report',
    'isbn', 'issn', 'doi:', 'reference', 'cited', 'bibliography',
]

# Regex patterns that indicate a date is in a citation/reference (Author. Year. format)
CITATION_PATTERNS = [
    r'\b[A-Z][a-z]+\.\s*\d{4}\.',  # "Smith. 2005." or "BLM. 2005."
    r'\b[A-Z]{2,}\.\s*\d{4}\.',     # "EPA. 2010." "USFWS. 2015."
    r'\(\d{4}\)',                    # "(2005)" - parenthetical citations
    r'\d{4}[a-z]?\)',               # "2005a)" - lettered citations
]

# Month name to number mapping
MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
}


def parse_date_match(match, pattern_type):
    """
    Parse a regex match into a datetime object.

    Returns:
        datetime or None
    """
    try:
        groups = match.groups()

        if pattern_type == 'MDY_full' or pattern_type == 'MDY_short':
            month_str, day, year = groups
            month = MONTH_MAP.get(month_str.lower())
            return datetime(int(year), month, int(day))

        elif pattern_type == 'DMY_full':
            day, month_str, year = groups
            month = MONTH_MAP.get(month_str.lower())
            return datetime(int(year), month, int(day))

        elif pattern_type == 'numeric_slash':
            month, day, year = groups
            return datetime(int(year), int(month), int(day))

        elif pattern_type == 'ISO':
            year, month, day = groups
            return datetime(int(year), int(month), int(day))

        elif pattern_type == 'numeric_dash':
            month, day, year = groups
            return datetime(int(year), int(month), int(day))

        elif pattern_type == 'MY_full' or pattern_type == 'MY_short':
            month_str, year = groups
            month = MONTH_MAP.get(month_str.lower())
            return datetime(int(year), month, 1)  # Default to first of month

    except (ValueError, TypeError, KeyError):
        return None

    return None


def should_exclude_date(text, match_start, match_end, window=50):
    """
    Check if a date should be excluded (e.g., it's a law/statute year or citation).

    Args:
        text: Full text
        match_start: Start position of date match
        match_end: End position of date match
        window: Characters to look before/after

    Returns:
        bool: True if date should be excluded
    """
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    context = text[start:end]
    context_lower = context.lower()

    # Check keyword exclusions
    for keyword in DATE_EXCLUSION_KEYWORDS:
        if keyword in context_lower:
            return True

    # Check citation patterns (case-sensitive for Author. Year. patterns)
    for pattern in CITATION_PATTERNS:
        if re.search(pattern, context):
            return True

    return False


def get_date_context(text, match_start, match_end, window=100):
    """
    Extract context around a date match to determine what type of date it is.

    Args:
        text: Full text
        match_start: Start position of date match
        match_end: End position of date match
        window: Characters to look before/after

    Returns:
        str: Context type ('start', 'decision', 'unknown', etc.)
    """
    # Get surrounding text
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    context = text[start:end].lower()

    # Check for context keywords
    for context_type, keywords in DATE_CONTEXT_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in context:
                return context_type

    return 'unknown'


def extract_dates_from_text(text):
    """
    Extract all dates from a text string with their context.

    Args:
        text: String containing document text

    Returns:
        list of dicts: [{'date': datetime, 'context': str, 'match': str}, ...]
    """
    if not text or not isinstance(text, str):
        return []

    results = []
    seen_dates = set()

    for pattern, pattern_type in DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            date_obj = parse_date_match(match, pattern_type)

            if date_obj is None:
                continue

            # Skip dates outside reasonable range (1990-2030) - narrowed to avoid old law years
            if date_obj.year < 1990 or date_obj.year > 2030:
                continue

            # Skip dates that appear to be law/statute years
            if should_exclude_date(text, match.start(), match.end()):
                continue

            # Deduplicate by date
            date_key = date_obj.strftime('%Y-%m-%d')
            if date_key in seen_dates:
                continue
            seen_dates.add(date_key)

            # Get context
            context = get_date_context(text, match.start(), match.end())

            results.append({
                'date': date_obj,
                'date_str': date_key,
                'context': context,
                'match': match.group(0)
            })

    # Sort by date
    results.sort(key=lambda x: x['date'])

    return results


# Document type category mapping (same as in extract_data.py)
DOCUMENT_TYPE_CATEGORIES = {
    'decision': ['ROD', 'FONSI', 'CE'],  # Decision documents - primary source for timelines
    'final': ['FEIS', 'EA'],              # Final documents (EA can be final)
    'draft': ['DEIS', 'DEA'],             # Draft documents
    'other': ['OTHER', ''],               # Other/unknown documents
}


def classify_document_type(doc_type):
    """Classify a document_type into a category."""
    if pd.isna(doc_type) or doc_type == '':
        return 'other'
    doc_type_upper = str(doc_type).upper().strip()
    for category, types in DOCUMENT_TYPE_CATEGORIES.items():
        if doc_type_upper in types:
            return category
    return 'other'


def build_project_timeline(project_id, pages_df, documents_df, decision_docs_only=True):
    """
    Build a timeline for a single project by extracting dates from documents.

    Args:
        project_id: Project ID string
        pages_df: DataFrame with page text
        documents_df: DataFrame with document metadata
        decision_docs_only: If True, prioritize decision documents (ROD, FONSI, CE).
                           Falls back to final/draft/other if no decision docs found.

    Returns:
        dict with timeline information
    """
    # Get documents for this project
    project_docs = documents_df[documents_df['project_id'] == project_id].copy()

    if project_docs.empty:
        return {
            'project_id': project_id,
            'project_dates': [],
            'project_date_earliest': None,
            'project_date_latest': None,
            'project_duration_days': None,
            'project_year': None,
        }

    # Add document type category if not present
    if 'document_type_category' not in project_docs.columns:
        project_docs['document_type_category'] = project_docs['document_type'].apply(classify_document_type)

    all_dates = []
    docs_used = []

    # Prioritize document types: decision > final > draft > other
    if decision_docs_only:
        priority_order = ['decision', 'final', 'draft', 'other']
        for doc_category in priority_order:
            category_docs = project_docs[project_docs['document_type_category'] == doc_category]
            if not category_docs.empty:
                # Use this category
                doc_ids = category_docs['document_id'].tolist()
                docs_used = doc_category
                break
        else:
            # Fallback to all docs
            doc_ids = project_docs['document_id'].tolist()
            docs_used = 'all'
    else:
        # Process all documents
        doc_ids = project_docs['document_id'].tolist()
        docs_used = 'all'

    project_pages = pages_df[pages_df['document_id'].isin(doc_ids)]

    for _, page in project_pages.iterrows():
        dates = extract_dates_from_text(page.get('page_text', ''))
        all_dates.extend(dates)

    if not all_dates:
        return {
            'project_id': project_id,
            'project_dates': [],
            'project_date_earliest': None,
            'project_date_latest': None,
            'project_duration_days': None,
            'project_year': None,
        }

    # Deduplicate and sort
    unique_dates = {}
    for d in all_dates:
        key = d['date_str']
        if key not in unique_dates:
            unique_dates[key] = d
        elif d['context'] != 'unknown' and unique_dates[key]['context'] == 'unknown':
            unique_dates[key] = d  # Prefer dates with known context

    sorted_dates = sorted(unique_dates.values(), key=lambda x: x['date'])

    # Extract key dates
    earliest = sorted_dates[0]['date'] if sorted_dates else None
    latest = sorted_dates[-1]['date'] if sorted_dates else None

    duration = None
    if earliest and latest:
        duration = (latest - earliest).days

    # Try to find decision date
    decision_dates = [d for d in sorted_dates if d['context'] == 'decision']
    decision_date = decision_dates[-1]['date'] if decision_dates else None

    # Determine project year (use decision date if available, else latest)
    project_year = None
    if decision_date:
        project_year = decision_date.year
    elif latest:
        project_year = latest.year

    # Flag projects that may need LLM review
    # Criteria: long duration (>10 years), no decision date found, or many dates with unknown context
    needs_llm_review = False
    review_reasons = []

    if duration and duration > 3650:  # >10 years suggests false positives
        needs_llm_review = True
        review_reasons.append("long_duration")

    if not decision_date and len(sorted_dates) > 0:
        needs_llm_review = True
        review_reasons.append("no_decision_date")

    unknown_count = sum(1 for d in sorted_dates if d['context'] == 'unknown')
    if len(sorted_dates) > 0 and unknown_count / len(sorted_dates) > 0.7:
        needs_llm_review = True
        review_reasons.append("mostly_unknown_context")

    return {
        'project_id': project_id,
        'project_dates': [d['date_str'] for d in sorted_dates[:20]],  # Keep first 20
        'project_date_contexts': [d['context'] for d in sorted_dates[:20]],  # Context for each date
        'project_date_earliest': earliest.strftime('%Y-%m-%d') if earliest else None,
        'project_date_latest': latest.strftime('%Y-%m-%d') if latest else None,
        'project_date_decision': decision_date.strftime('%Y-%m-%d') if decision_date else None,
        'project_duration_days': duration,
        'project_year': project_year,
        'project_timeline_needs_review': needs_llm_review,
        'project_timeline_review_reasons': review_reasons,
        'project_timeline_doc_source': docs_used,  # Which doc type was used for extraction
    }


def run_timeline_extraction(sample_size=None, clean_energy_only=False, decision_docs_only=True):
    """
    Run timeline extraction for all projects.

    Args:
        sample_size: If set, only process this many projects
        clean_energy_only: If True, only process clean energy projects
        decision_docs_only: If True, prioritize decision documents for extraction

    Outputs:
        data/analysis/projects_timeline.parquet
    """
    print("\n=== Timeline Extraction ===")

    # Load projects
    projects_path = ANALYSIS_DIR / "projects_combined.parquet"
    if not projects_path.exists():
        print(f"Error: {projects_path} not found. Run extract_data.py first.")
        return

    projects = pd.read_parquet(projects_path)
    print(f"Loaded {len(projects):,} projects")

    # Filter if requested
    if clean_energy_only:
        projects = projects[projects['project_energy_type'] == 'Clean']
        print(f"Filtered to {len(projects):,} clean energy projects")

    if sample_size:
        projects = projects.head(sample_size)
        print(f"Sampling {len(projects):,} projects")

    # Process by source dataset
    results = []

    for source in projects['dataset_source'].unique():
        print(f"\nProcessing {source} projects...")

        source_projects = projects[projects['dataset_source'] == source]
        data_dir = PROCESSED_DIR / source.lower()

        pages_df = pd.read_parquet(data_dir / "pages.parquet")
        documents_df = pd.read_parquet(data_dir / "documents.parquet")

        # Clean project_id in documents
        def extract_id(x):
            if isinstance(x, dict):
                return x.get('value', '')
            return x

        documents_df['project_id'] = documents_df['project_id'].apply(extract_id)

        # Process each project
        for idx, (_, project) in enumerate(source_projects.iterrows()):
            if idx % 100 == 0:
                print(f"  Processing project {idx + 1}/{len(source_projects)}...")

            project_id = project['project_id']
            timeline = build_project_timeline(project_id, pages_df, documents_df, decision_docs_only)
            results.append(timeline)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Convert list columns to JSON for parquet
    import json
    results_df['project_dates'] = results_df['project_dates'].apply(json.dumps)
    results_df['project_date_contexts'] = results_df['project_date_contexts'].apply(json.dumps)
    results_df['project_timeline_review_reasons'] = results_df['project_timeline_review_reasons'].apply(json.dumps)

    # Merge back to projects
    projects_with_timeline = projects.merge(results_df, on='project_id', how='left')

    # Save
    output_path = ANALYSIS_DIR / "projects_timeline.parquet"
    projects_with_timeline.to_parquet(output_path)
    print(f"\nSaved to: {output_path}")

    # Summary
    has_dates = projects_with_timeline['project_date_earliest'].notna()
    has_year = projects_with_timeline['project_year'].notna()
    needs_review = projects_with_timeline['project_timeline_needs_review'] == True
    print(f"\nProjects with dates extracted: {has_dates.sum():,} ({has_dates.mean() * 100:.1f}%)")
    print(f"Projects with year determined: {has_year.sum():,} ({has_year.mean() * 100:.1f}%)")
    print(f"Projects flagged for LLM review: {needs_review.sum():,} ({needs_review.mean() * 100:.1f}%)")

    if has_year.sum() > 0:
        print(f"\nYear distribution:")
        print(projects_with_timeline['project_year'].value_counts().sort_index().tail(10))

    return projects_with_timeline


# --------------------------
# LLM EXTRACTION (for validation/gaps)
# --------------------------

def create_llm_prompt(page_texts, max_chars=4000):
    """
    Create a prompt for LLM timeline extraction.

    Args:
        page_texts: List of page text strings
        max_chars: Maximum characters to include

    Returns:
        str: Prompt for LLM
    """
    combined_text = "\n\n".join(page_texts)
    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars] + "..."

    prompt = f"""Extract all dates mentioned in this NEPA document and identify what each date refers to.

Document text:
{combined_text}

Please extract dates in this JSON format:
{{
  "dates": [
    {{"date": "YYYY-MM-DD", "type": "notice|draft|final|decision|comment|other", "description": "brief description"}}
  ],
  "project_start_date": "YYYY-MM-DD or null",
  "decision_date": "YYYY-MM-DD or null",
  "duration_days": number or null
}}

Only include dates you are confident about. Use null if uncertain."""

    return prompt


def extract_with_ollama(project_id, pages_df, documents_df, model="llama3.2"):
    """
    Extract timeline using local Ollama LLM (free).

    Requires: ollama installed and running locally
    Install: https://ollama.ai/
    Run: ollama pull llama3.2

    Args:
        project_id: Project ID
        pages_df: Pages dataframe
        documents_df: Documents dataframe
        model: Ollama model name

    Returns:
        dict with extracted timeline
    """
    try:
        import requests
    except ImportError:
        print("requests library required for Ollama")
        return None

    # Get main document pages
    project_docs = documents_df[documents_df['project_id'] == project_id]
    main_docs = project_docs[project_docs['main_document'] == 'YES']

    if main_docs.empty:
        main_docs = project_docs.head(1)

    doc_ids = main_docs['document_id'].tolist()
    pages = pages_df[pages_df['document_id'].isin(doc_ids)]

    # Get first few pages
    page_texts = pages.head(10)['page_text'].tolist()

    prompt = create_llm_prompt(page_texts)

    # Call Ollama API
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            print(f"Ollama error: {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        print("Ollama not running. Start with: ollama serve")
        return None
    except Exception as e:
        print(f"Ollama error: {e}")
        return None


# --------------------------
# TESTING
# --------------------------

if __name__ == "__main__":
    # Test date extraction
    test_texts = [
        "The project was approved on January 15, 2024.",
        "Notice of Intent published in the Federal Register on 03/22/2023.",
        "The Draft EIS was released in March 2022.",
        "Decision Record signed December 1, 2023.",
        "Comment period ends Jan. 30, 2024.",
        "Project commenced on 2021-06-15 (ISO format).",
        "No dates in this text.",
    ]

    print("Testing date extraction patterns...")
    for text in test_texts:
        results = extract_dates_from_text(text)
        print(f"\nText: {text}")
        for r in results:
            print(f"  {r['date_str']} ({r['context']}): {r['match']}")

    print("\n" + "=" * 50)
    print("\nTo run full extraction:")
    print("  python extract_timeline.py --run")
    print("\nFor a test sample:")
    print("  python extract_timeline.py --run --sample 100")
    print("\nFor clean energy only:")
    print("  python extract_timeline.py --run --clean-energy")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Run full extraction')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--clean-energy', action='store_true', help='Process clean energy only')
    parser.add_argument('--all-docs', action='store_true', help='Use all documents (not just decision docs)')

    args = parser.parse_args()

    if args.run:
        run_timeline_extraction(
            sample_size=args.sample,
            clean_energy_only=args.clean_energy,
            decision_docs_only=not args.all_docs
        )
