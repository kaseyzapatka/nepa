# --------------------------
# TIMELINE EXTRACTION
# --------------------------
# Extract dates from document text to construct project timelines
# Strategy: Regex first to find all dates, then order them chronologically

import re
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
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
    # "01/15/2024" or "1/15/2024" (4-digit year)
    (r'(\d{1,2})/(\d{1,2})/(\d{4})', 'numeric_slash'),
    # "01/15/24" or "1/15/24" (2-digit year)
    (r'(\d{1,2})/(\d{1,2})/(\d{2})\b', 'numeric_slash_2y'),
    # "2024-01-15" (ISO format)
    (r'(\d{4})-(\d{1,2})-(\d{1,2})', 'ISO'),
    # "01-15-2024" (dash with 4-digit year)
    (r'(\d{1,2})-(\d{1,2})-(\d{4})', 'numeric_dash'),
    # "2024.01.15" (digital signature format)
    (r'(\d{4})\.(\d{2})\.(\d{2})', 'digital_sig'),
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
# NOTE: These must be specific enough to avoid excluding valid document dates
# Avoid generic terms like "nepa" which would exclude "NEPA Compliance Officer Date:"
DATE_EXCLUSION_KEYWORDS = [
    # Law/statute references - use specific phrases, not just acronyms
    'act of 19', 'act of 20',  # "Act of 1969", "Act of 2000"
    'act (19', 'act (20',      # "Act (1969)"
    'policy act', 'preservation act', 'conservation act',
    'management act', 'protection act', 'improvement act', 'reform act',
    'recovery act', 'species act', 'water act', 'air act', 'lands act',
    'statute', 'u.s.c.', 'usc', 'public law', 'p.l.', 'amended in',
    # Bibliographic references (citations)
    'accessed on', 'retrieved on', 'available at',
    'et al.', 'et al,', 'eds.', 'editor', 'vol.', 'volume', 'pp.', 'pages',
    'journal', 'proceedings', 'report no.', 'technical report',
    'isbn', 'issn', 'doi:',
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

        elif pattern_type == 'numeric_slash_2y':
            # 2-digit year: assume 2000s for 00-30, 1900s for 31-99
            month, day, year_2d = groups
            year_int = int(year_2d)
            year = 2000 + year_int if year_int <= 30 else 1900 + year_int
            return datetime(year, int(month), int(day))

        elif pattern_type == 'ISO':
            year, month, day = groups
            return datetime(int(year), int(month), int(day))

        elif pattern_type == 'numeric_dash':
            month, day, year = groups
            return datetime(int(year), int(month), int(day))

        elif pattern_type == 'digital_sig':
            # Digital signature format: YYYY.MM.DD
            year, month, day = groups
            return datetime(int(year), int(month), int(day))

        elif pattern_type == 'MY_full' or pattern_type == 'MY_short':
            month_str, year = groups
            month = MONTH_MAP.get(month_str.lower())
            return datetime(int(year), month, 1)  # Default to first of month

    except (ValueError, TypeError, KeyError):
        return None

    return None


def extract_dates_from_filename(file_name: str) -> list:
    """
    Extract date strings from a filename.

    Returns:
        List of dicts: [{'date': 'YYYY-MM-DD', 'match': 'Feb 27 2023'}, ...]
    """
    if not file_name or not isinstance(file_name, str):
        return []

    variants = [
        file_name,
        file_name.replace('_', '-'),
        file_name.replace('_', '/'),
        re.sub(r'[_-]+', ' ', file_name),
    ]

    results = []
    seen = set()

    # Extra filename-specific pattern: YYYY/MM/DD
    ymd_slash = re.compile(r'(\d{4})/(\d{1,2})/(\d{1,2})')

    for text in variants:
        # DATE_PATTERNS handle month names, MDY, DMY, ISO, numeric slash/dash
        for pattern, pattern_type in DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date_obj = parse_date_match(match, pattern_type)
                if date_obj is None:
                    continue
                date_str = date_obj.strftime('%Y-%m-%d')
                key = (date_str, match.group(0))
                if key in seen:
                    continue
                seen.add(key)
                results.append({'date': date_str, 'match': match.group(0)})

        # YYYY/MM/DD variant
        for match in ymd_slash.finditer(text):
            year, month, day = match.groups()
            try:
                date_obj = datetime(int(year), int(month), int(day))
            except ValueError:
                continue
            date_str = date_obj.strftime('%Y-%m-%d')
            key = (date_str, match.group(0))
            if key in seen:
                continue
            seen.add(key)
            results.append({'date': date_str, 'match': match.group(0)})

    # Sort by date string
    results.sort(key=lambda x: x['date'])
    return results


def build_file_name_date_map(documents_df: pd.DataFrame, main_docs_only: bool = True) -> dict:
    """
    Build a map of project_id -> JSON list of filename date matches.
    """
    if documents_df is None or documents_df.empty:
        return {}

    df = documents_df.copy()

    def extract_id(x):
        if isinstance(x, dict):
            return x.get('value', '')
        return x

    if 'project_id' in df.columns:
        df['project_id'] = df['project_id'].apply(extract_id)

    if main_docs_only and 'main_document' in df.columns:
        df = df[df['main_document'] == 'YES']

    project_map = {}

    for _, row in df.iterrows():
        project_id = row.get('project_id')
        file_name = row.get('file_name')
        if not project_id or not file_name:
            continue
        matches = extract_dates_from_filename(file_name)
        if not matches:
            continue
        for m in matches:
            m['file_name'] = file_name
            if 'document_id' in row:
                m['document_id'] = row.get('document_id')
        project_map.setdefault(project_id, []).extend(matches)

    # Deduplicate per project by (date, match, file_name)
    deduped = {}
    for project_id, items in project_map.items():
        seen = set()
        out = []
        for item in items:
            key = (item.get('date'), item.get('match'), item.get('file_name'))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        deduped[project_id] = json.dumps(out)

    return deduped

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

    # Count documents and main documents for this project
    document_count = len(project_docs)
    main_document_count = 0
    if not project_docs.empty and 'main_document' in project_docs.columns:
        main_document_count = (project_docs['main_document'] == 'YES').sum()

    if project_docs.empty:
        return {
            'project_id': project_id,
            'project_dates': [],
            'project_date_earliest': None,
            'project_date_latest': None,
            'project_duration_days': None,
            'project_year': None,
            'project_document_count': document_count,
            'project_main_document_count': main_document_count,
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
            'project_document_count': document_count,
            'project_main_document_count': main_document_count,
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
        'project_document_count': document_count,
        'project_main_document_count': main_document_count,
    }


def run_timeline_extraction(sample_size=None, clean_energy_only=False, ce_only=False,
                            decision_docs_only=True, main_docs_only=False):
    """
    Run timeline extraction for all projects.

    Args:
        sample_size: If set, only process this many projects
        clean_energy_only: If True, only process clean energy projects
        ce_only: If True, only process CE (Categorical Exclusion) projects
        decision_docs_only: If True, prioritize decision documents for extraction
        main_docs_only: If True, only read pages from main_document == 'YES' documents

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

    # Filter by dataset source (CE only)
    if ce_only:
        projects = projects[projects['dataset_source'] == 'CE']
        print(f"Filtered to {len(projects):,} CE projects")

    # Filter by energy type
    if clean_energy_only:
        projects = projects[projects['project_energy_type'] == 'Clean']
        print(f"Filtered to {len(projects):,} clean energy projects")

    if sample_size:
        projects = projects.head(sample_size)
        print(f"Sampling {len(projects):,} projects")

    if main_docs_only:
        print("Will only read pages from main_document == 'YES'")

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

        # Precompute filename date matches for this source
        file_name_dates_map = build_file_name_date_map(
            documents_df,
            main_docs_only=main_docs_only
        )

        # Filter to main documents only if requested
        if main_docs_only:
            main_doc_ids = documents_df[documents_df['main_document'] == 'YES']['document_id'].tolist()
            pages_df = pages_df[pages_df['document_id'].isin(main_doc_ids)]
            print(f"    Filtered to {len(pages_df):,} pages from main documents")

        # Process each project
        for idx, (_, project) in enumerate(source_projects.iterrows()):
            if idx % 100 == 0:
                print(f"  Processing project {idx + 1}/{len(source_projects)}...")

            project_id = project['project_id']
            timeline = build_project_timeline(project_id, pages_df, documents_df, decision_docs_only)
            timeline['project_file_name_dates'] = file_name_dates_map.get(project_id, '[]')
            results.append(timeline)

    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df['run_timestamp'] = pd.Timestamp.utcnow().isoformat()
    results_df['run_timestamp'] = pd.Timestamp.utcnow().isoformat()

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
# LLM EXTRACTION (with preprocessing)
# --------------------------

# Import preprocessing module
from preprocess_documents import preprocess_for_timeline

# Default Ollama model - llama3.2 is fast and accurate for structured extraction
DEFAULT_LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"

# Context window for date extraction (chars before/after each date)
DATE_CONTEXT_WINDOW = 80

# Default cache path for hybrid regex candidates
REGEX_CACHE_PATH = ANALYSIS_DIR / "regex_candidates.parquet"


# --------------------------
# BERT CLASSIFIER APPROACH
# --------------------------
# Uses weak supervision (pattern-based auto-labeling) to train a fast classifier

# Model paths
BERT_MODEL_DIR = BASE_DIR / "models" / "timeline_classifier"
BERT_TRAINING_DATA_PATH = ANALYSIS_DIR / "bert_training_data.parquet"

# Auto-labeling patterns (ranked confidence rules)
DECISION_PATTERNS_STRONG = [
    r'digitally signed by',
    r'signed by\s+\w+',
    r'signature of',
    r'/s/\s*\w+',  # Digital signature format
    r'concur.*nepa compliance officer',
    r'nepa compliance officer.*concur',
    r'nepa compliance officer.*date',
    r'nepa compliance officer',
    r'authoriz(ing|ed) official',
    r'authority and approval',
    r'NCO Determination:',
    r'determination and approval',
    r'categorical exclusion determination',
    r'categorical exclusion (determination|approval)',
    r'decision memo',
    r'decision memorandum',
    r'decision record',
    r'authorizing official.*date',
    r'field.*manager.*signature',
    r'field office manager determination',
    r'field manager (acting)?',
    r'assistant field manager',
    r'approved.*signature',
    r'fonsi.*signed',
    r'rod.*signed',
    r'decision.*signed',
    r'approval date',
    r'date of approval',
    r'ce determination date',
]

DECISION_PATTERNS_MED = [
    r'digitally signed by',
    r'date determined',
    r'field office manager',
    r'authorizing official',
    r'final approval',
]

DECISION_PATTERNS_WEAK = [
    r'determination',
    r'approval',
]

INITIATION_PATTERNS_STRONG = [
    r'scoping meeting',
    r'scoping period',
    r'notice of intent',
    r'submitted notice of intent',
    r'\bnoi\b',
    r'\bnoi\b.*publish',
    r'noi published',
    r'doe initiator signature',
    r'application received',
    r'application submitted',
    r'consultation initiated',
    r'initiation of consultation',
    r'initiated on',
    r'right[- ]of[- ]way application',
    r'row application',
    r'\brow\b application',
    r'submitted (a )?(completed )?right[- ]of[- ]way application',
    r'blm received (a|the) (row )?application',
    r'initiator signature',
    r'designation form',
    r'doe initiator signature',
]

INITIATION_PATTERNS_MED = [
    r'renewal application received',
    r'by renewal application received',
    r'project proposed',
    r'proposed action',
    r'designation',
    r'nepa process started',
    r'nepa review began',
    r'proposal submitted',
    r'request received',
    r'review was initiated',
    r'(distribution|review) (was )?initiated',
]

INITIATION_PATTERNS_WEAK = [
    r'may require (a )?new nepa determinat',
    r'may be categorically excluded',
    r'categorically excluded from further nepa review',
]

REVIEW_PATTERNS_STRONG = [
    r'review completed',
    r'interim review',
    r'decision in principle',
    r'phase\s+\d+\s+was approved',
    r'phase\s+\d+\s+approved',
    r'approved by\s+\w+.*(review|phase)',
    r'realty specialist',
    r'wildlife biologist',
    r'scientist',
    r'fisheries/?wildlife biologist',
    r'recreation (planner|specialist)',
    r'outdoor recreation planner',
    r'natural resource specialist',
    r'environmental coordinator',
    r'planning & environmental coordinator',
    r'planning and environmental coordinator',
    r'environmental coordinator',
    r'environmental specialist',
    r'botanist',
    r'archaeologist',
    r'archeologist',
    r'district archaeologist',
    r'cultural resource(s)? specialist',
    r'project officer',
    r'shpo.*concur',
    r'environmental clearance memorandum',
    r'yes\s+no\s+reviewer/title\s+initials\s*&\s*date',
    r'yes\s+no\s+reviewer',
    r'reviewer/title\s+initials\s*&\s*date',
    r'nepa review completed',
    r'memorandum of agreement',
    r'\bmoa\b',
    r'section 106',
]

REVIEW_PATTERNS_STRONG_CASE_SENSITIVE = [
    r'\bSubject Matter Expert\b',
    r'\bExpert\b',
    r'\bNEPA-SME\b',
    r'\bNEPA SME\b',
    r'\bSME\b',
]

REVIEW_PATTERNS_MED = [
    r'review memo',
    r'review memorandum',
    r'reviewed by',
]

REVIEW_PATTERNS_WEAK = [
    r'specialist',
]

# Expiration cues (for future dates, e.g., CE expiration)
EXPIRATION_PATTERNS_STRONG = [
    r're-?authoriz\w*.*until',
    r'categorical exclusion expires',
    r'\bexpire\b',
    r'\bexpir\w*\s+on\b',
    r'expiration date would',
    r'for a term of',
    r'expiration date\s*[:]',
    r'expiration date of',
    r'valid until',
]

# Background/plan reference cues (treat as historical/other)
HISTORICAL_CONTEXT_PATTERNS = [
    r'resource management plan',
    r'\brmp\b',
    r'conformance with the applicable lup',
    r'land use plan',
    r'plan maintenance action',
    r'record of decision',
    r'was assigned to',
    r'assigned back to',
    r'lease was issued',
    r'communication site was established',
]

# Hard exclusions for initiation candidates
INITIATION_EXCLUSION_PATTERNS = [
    r'program specific guidance',
    r'prepared in accordance with .* guidance',
    r'resource management plan',
    r'\brmp\b',
    r'land use plan',
    r'conformance with the applicable lup',
    r'record of decision',
    r'plan maintenance action',
    r'memorandum of agreement',
    r'\bmoa\b',
    r'section 106',
    r'shpo.*concur',
    r'map created',
    r'map prepared',
    r'form approved',
    r'previous editions obsolete',
    r'reviewer/title\s+initials\s*&\s*date',
    r'yes\s+no\s+reviewer',
    r'specialist signature',
]

MEMO_DATE_PATTERNS = [
    r'\bDATE:\b',
    r'\bmemorandum\b',
]

DECISION_BOILERPLATE_PATTERNS = [
    r'previous editions obsolete',
    r'form approved',
    r'forms mgmt',
    r'netl f \d+',
    r'doe f \d+',
    r'revised:',
    r'reviewed:',
]

OTHER_PATTERNS_STRONG = [
    r'map created',
    r'map prepared',
    r'revised \d{4}',
    r'prepared by.*\d{4}',
    r'\d+\s*cfr\s*\d+',
    r'\d+\s*u\.?s\.?c',
    r'\d+\s*fr\s*\d+',
    r'federal register',
    r'public law',
    r'act of \d{4}',
    r'program specific guidance',
    r'prepared in accordance with .* guidance',
]


def auto_label_context(context: str) -> str:
    """
    Auto-label a date context using pattern matching (weak supervision).

    Args:
        context: The text context around a date

    Returns:
        'decision', 'initiation', 'other', or None if uncertain
    """
    context_lower = context.lower()

    # Check decision patterns first (highest priority)
    for pattern in DECISION_PATTERNS_STRONG:
        if re.search(pattern, context_lower):
            return 'decision'
    for pattern in DECISION_PATTERNS_MED:
        if re.search(pattern, context_lower):
            return 'decision'
    for pattern in DECISION_PATTERNS_WEAK:
        if re.search(pattern, context_lower):
            return 'decision'

    # Check review patterns (treat as initiation-negative)
    for pattern in REVIEW_PATTERNS_STRONG:
        if re.search(pattern, context_lower):
            return 'review'
    for pattern in REVIEW_PATTERNS_STRONG_CASE_SENSITIVE:
        if re.search(pattern, context):
            return 'review'
    for pattern in REVIEW_PATTERNS_MED:
        if re.search(pattern, context_lower):
            return 'review'
    for pattern in REVIEW_PATTERNS_WEAK:
        if re.search(pattern, context_lower):
            return 'review'

    # Check initiation patterns
    for pattern in INITIATION_PATTERNS_STRONG:
        if re.search(pattern, context_lower):
            return 'initiation'
    for pattern in INITIATION_PATTERNS_MED:
        if re.search(pattern, context_lower):
            return 'initiation'
    for pattern in INITIATION_PATTERNS_WEAK:
        if re.search(pattern, context_lower):
            return 'initiation'

    # Check other patterns (negative examples)
    for pattern in OTHER_PATTERNS_STRONG:
        if re.search(pattern, context_lower):
            return 'other'

    # Return None for ambiguous cases (don't include in training)
    return None


def generate_bert_training_data(
    sample_size: int = None,
    output_path: Path = BERT_TRAINING_DATA_PATH,
    min_samples_per_class: int = 100,
):
    """
    Generate training data for BERT classifier using weak supervision.

    Uses pattern-based auto-labeling on all regex-extracted date contexts.

    Args:
        sample_size: If set, only process this many projects
        output_path: Where to save the training data
        min_samples_per_class: Minimum samples needed per class

    Returns:
        DataFrame with labeled training examples
    """
    import time

    print("\n=== Generating BERT Training Data (Weak Supervision) ===")

    # Check if regex cache exists
    if not REGEX_CACHE_PATH.exists():
        print(f"Regex cache not found at {REGEX_CACHE_PATH}")
        print("Run --regex-prep first to build the cache, or building now...")
        run_regex_prep()

    # Load regex cache
    print("Loading regex cache...")
    cache_df = pd.read_parquet(REGEX_CACHE_PATH)
    print(f"Loaded {len(cache_df):,} date contexts")

    if sample_size:
        # Sample by project to maintain diversity
        project_ids = cache_df['project_id'].unique()
        if len(project_ids) > sample_size:
            project_ids = pd.Series(project_ids).sample(n=sample_size, random_state=42).tolist()
            cache_df = cache_df[cache_df['project_id'].isin(project_ids)]
        print(f"Sampled to {len(cache_df):,} contexts from {len(project_ids)} projects")

    # Auto-label each context
    print("Auto-labeling contexts...")
    start = time.time()

    labeled_data = []
    label_counts = {'decision': 0, 'initiation': 0, 'review': 0, 'other': 0, 'unlabeled': 0}

    for _, row in cache_df.iterrows():
        context = row.get('context', '')
        if not context or len(context) < 10:
            continue

        label = auto_label_context(context)

        if label:
            labeled_data.append({
                'context': context,
                'label': label,
                'date': row.get('date'),
                'project_id': row.get('project_id'),
            })
            label_counts[label] += 1
        else:
            label_counts['unlabeled'] += 1

    elapsed = time.time() - start
    print(f"Labeled {len(labeled_data):,} contexts in {elapsed:.1f}s")
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count:,}")

    # Check minimum samples
    for label in ['decision', 'initiation', 'review', 'other']:
        if label_counts[label] < min_samples_per_class:
            print(f"\nWARNING: Only {label_counts[label]} samples for '{label}' (min: {min_samples_per_class})")

    # Create DataFrame
    training_df = pd.DataFrame(labeled_data)
    training_df['run_timestamp'] = pd.Timestamp.utcnow().isoformat()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(output_path)
    print(f"\nSaved training data to: {output_path}")

    return training_df


def train_bert_classifier(
    training_data_path: Path = BERT_TRAINING_DATA_PATH,
    model_name: str = "distilbert-base-uncased",
    output_dir: Path = BERT_MODEL_DIR,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    test_split: float = 0.1,
):
    """
    Train a BERT classifier on the auto-labeled data.

    Args:
        training_data_path: Path to training data parquet
        model_name: Hugging Face model name
        output_dir: Where to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        max_length: Max token length
        test_split: Fraction to hold out for validation

    Returns:
        Trained model and tokenizer
    """
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
        )
        from datasets import Dataset
        import numpy as np
    except ImportError:
        print("ERROR: transformers and datasets libraries required.")
        print("Install with: pip install transformers datasets torch")
        return None, None

    print(f"\n=== Training BERT Classifier ===")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    # Load training data
    if not training_data_path.exists():
        print(f"Training data not found at {training_data_path}")
        print("Run --bert-generate first to create training data.")
        return None, None

    df = pd.read_parquet(training_data_path)
    print(f"Loaded {len(df):,} training examples")

    # Create label mapping
    label2id = {'decision': 0, 'initiation': 1, 'review': 2, 'other': 3}
    id2label = {v: k for k, v in label2id.items()}

    df['label_id'] = df['label'].map(label2id)

    # Check class balance
    print("\nClass distribution:")
    print(df['label'].value_counts())

    # Split into train/test
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * (1 - test_split))
    train_df = df_shuffled[:split_idx]
    test_df = df_shuffled[split_idx:]

    print(f"\nTrain: {len(train_df):,}, Test: {len(test_df):,}")

    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[['context', 'label_id']])
    test_dataset = Dataset.from_pandas(test_df[['context', 'label_id']])

    # Load tokenizer and model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples['context'],
            truncation=True,
            max_length=max_length,
            padding=False,  # Let data collator handle padding
        )

    print("Tokenizing...")
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=['context'])
    test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=['context'])

    # Rename label column
    train_dataset = train_dataset.rename_column('label_id', 'labels')
    test_dataset = test_dataset.rename_column('label_id', 'labels')

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Class weights (handle imbalance)
    import numpy as np
    import torch
    label_counts = df['label_id'].value_counts().sort_index()
    total = label_counts.sum()
    # Inverse frequency weights
    class_weights = total / (label_counts * len(label_counts))
    class_weights = torch.tensor(class_weights.values, dtype=torch.float)
    print("\nClass weights:")
    for lbl, w in zip(label2id.keys(), class_weights.tolist()):
        print(f"  {lbl}: {w:.3f}")

    # Metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}

    # Training arguments
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="no",  # Skip eval during training (numpy issue on Apple Silicon)
        save_strategy="epoch",
        load_best_model_at_end=False,
    )

    # Trainer with weighted loss (compatible with older transformers)
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nTraining...")
    trainer.train()

    # Skip evaluation due to numpy compatibility issue on Apple Silicon
    # The model is still trained and saved correctly
    print("\nSkipping evaluation (numpy compatibility issue on Apple Silicon)")

    # Save
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save label mapping
    import json
    with open(output_dir / "label_mapping.json", 'w') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f)

    print("Training complete!")

    return model, tokenizer


class BertDateClassifier:
    """
    Fast BERT-based classifier for date contexts.

    Usage:
        classifier = BertDateClassifier()
        classifier.load()
        results = classifier.classify(["context1", "context2"])
    """

    def __init__(self, model_dir: Path = BERT_MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None
        self.device = None

    def load(self):
        """Load the trained model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            raise ImportError("transformers and torch required. Install with: pip install transformers torch")

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model not found at {self.model_dir}. Run --bert-train first.")

        print(f"Loading BERT classifier from {self.model_dir}...")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))

        # Load label mapping
        import json
        with open(self.model_dir / "label_mapping.json") as f:
            mapping = json.load(f)
            self.label2id = mapping['label2id']
            self.id2label = {int(k): v for k, v in mapping['id2label'].items()}

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on {self.device}")

    def classify(self, contexts: list, batch_size: int = 32) -> list:
        """
        Classify a list of contexts.

        Args:
            contexts: List of context strings
            batch_size: Batch size for inference

        Returns:
            List of dicts with 'label' and 'confidence'
        """
        import torch

        if self.model is None:
            self.load()

        results = []

        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                predictions = torch.argmax(probs, dim=1)
                confidences = probs.max(dim=1).values

            # Convert to results
            for pred, conf in zip(predictions.cpu().numpy(), confidences.cpu().numpy()):
                results.append({
                    'label': self.id2label[int(pred)],
                    'confidence': float(conf),
                })

        return results

    def classify_single(self, context: str) -> dict:
        """Classify a single context."""
        return self.classify([context])[0]


def _is_decision_boilerplate(context: str) -> bool:
    return _has_any_regex(context, DECISION_BOILERPLATE_PATTERNS)


def _decision_strength(context: str) -> int:
    if not context:
        return 0
    if _has_any_regex(context, DECISION_PATTERNS_STRONG):
        return 3
    if _has_any_regex(context, DECISION_PATTERNS_MED):
        return 2
    if _has_any_regex(context, DECISION_PATTERNS_WEAK):
        return 1
    if _has_any(context, DECISION_CUES):
        return 1
    return 0


def _has_strong_review_cue(context: str) -> bool:
    return (
        _has_any_regex(context, REVIEW_PATTERNS_STRONG)
        or _has_any_regex_case_sensitive(context, REVIEW_PATTERNS_STRONG_CASE_SENSITIVE)
    )


def _has_strong_decision_cue(context: str) -> bool:
    return _has_any_regex(context, DECISION_PATTERNS_STRONG)


def _has_reviewer_checkbox(context: str) -> bool:
    return _has_any_regex(context, [
        r'yes\s+no\s+reviewer/title\s+initials\s*&\s*date',
        r'yes\s+no\s+reviewer',
        r'reviewer/title\s+initials\s*&\s*date',
    ])


def _is_expiration_candidate(date_str: str, context: str) -> bool:
    if not date_str or not context:
        return False
    # Only bucket future dates after 2025-12-31
    if date_str <= "2025-12-31":
        return False
    return _has_any_regex(context, EXPIRATION_PATTERNS_STRONG)


def _is_historical_by_year(date_str: str) -> bool:
    if not date_str:
        return False
    # Any date before 2000 is historical
    return date_str < "2000-01-01"


def _has_historical_context(context: str) -> bool:
    return _has_any_regex(context, HISTORICAL_CONTEXT_PATTERNS)


def _apply_historical_gap_rule(classified_dates: list, gap_days: int = 730, enable: bool = True) -> None:
    """
    If the earliest date is more than gap_days before the next earliest date,
    re-label the earliest as historical.
    """
    if not enable:
        return
    if not classified_dates or len(classified_dates) < 2:
        return
    # Sort by date (ISO strings safe for lexicographic order)
    sorted_dates = sorted(classified_dates, key=lambda x: x.get('date') or "")
    d0 = sorted_dates[0].get('date')
    d1 = sorted_dates[1].get('date')
    if not d0 or not d1:
        return
    try:
        from datetime import datetime
        dt0 = datetime.strptime(d0, "%Y-%m-%d")
        dt1 = datetime.strptime(d1, "%Y-%m-%d")
    except Exception:
        return
    if (dt1 - dt0).days > gap_days:
        sorted_dates[0]['type'] = 'historical'


def _select_best_decision(decision_dates: list) -> dict:
    if not decision_dates:
        return None

    for d in decision_dates:
        ctx = d.get('context', '')
        d['_decision_strength'] = _decision_strength(ctx)
        d['_boilerplate_penalty'] = -2 if _is_decision_boilerplate(ctx) else 0
        d['_confidence'] = d.get('bert_confidence', d.get('confidence_score', 0))
        d['_score'] = d['_decision_strength'] + d['_boilerplate_penalty'] + (2 * d['_confidence'])

    decision_dates.sort(key=lambda x: (x['_score'], x['date']), reverse=True)
    best = decision_dates[0]

    if len(decision_dates) > 1:
        runner_up = decision_dates[1]
        score_gap = abs(best['_score'] - runner_up['_score'])
        pos_gap = abs((best.get('position_pct') or 0) - (runner_up.get('position_pct') or 0))

        # Only use header/footer position when cue strength is similar but positions disagree
        if score_gap < 0.3 and pos_gap >= 20:
            best = max([best, runner_up], key=lambda x: x.get('position_pct') or 0)

    return best


def _initiation_strength(context: str) -> int:
    if not context:
        return 0
    # Strong signals
    if _has_any_regex(context, [r'doe initiator signature', r'initiator signature']):
        return 3
    if _has_any_regex(context, [r'application received', r'application submitted', r'notice of intent', r'\bnoi\b', r'scoping']):
        return 2
    if _has_any_regex(context, [r'right[- ]of[- ]way application', r'\brow\b application', r'proposal submitted', r'request received', r'deemed the application complete']):
        return 2
    if _has_any_regex(context, [r'proposed action', r'project proposed', r're-submitted', r'amended and re-submitted']):
        return 1
    return 0


def _is_initiation_excluded(context: str) -> bool:
    return _has_any_regex(context, INITIATION_EXCLUSION_PATTERNS)


def _select_best_initiation(initiation_dates: list, decision_date: str = None) -> dict:
    """
    Select the best initiation date using cue strength, confidence, and sanity checks.
    """
    if not initiation_dates:
        return None

    candidates = []
    for d in initiation_dates:
        ctx = d.get('context', '')
        if _is_initiation_excluded(ctx):
            continue
        score = _initiation_strength(ctx)
        # Penalize if initiation after decision
        if decision_date and d.get('date') and d.get('date') > decision_date:
            score -= 3
        d['_init_score'] = score
        d['_confidence'] = d.get('bert_confidence', d.get('confidence_score', 0))
        d['_score'] = d['_init_score'] + d['_confidence']
        candidates.append(d)

    if not candidates:
        return None

    # Prefer higher score, then earliest date
    candidates.sort(key=lambda x: (x['_score'], x['date']), reverse=True)
    best = candidates[0]

    # If scores are tied, take earliest
    top_score = best['_score']
    tied = [c for c in candidates if abs(c['_score'] - top_score) < 1e-6]
    if len(tied) > 1:
        tied.sort(key=lambda x: x['date'])
        best = tied[0]

    return best


def extract_with_bert(
    dates_with_context: list,
    classifier: BertDateClassifier = None,
    apply_historical_gap_rule: bool = True,
) -> dict:
    """
    Classify pre-extracted dates using BERT classifier.

    This replaces the LLM call in the hybrid approach with a fast BERT inference.

    Args:
        dates_with_context: List from extract_dates_with_context()
        classifier: Pre-loaded BertDateClassifier (optional, will load if None)

    Returns:
        Dict with classified dates and summary fields (same format as LLM approach)
    """
    import json

    result = {
        'approach': 'bert_classifier',
        'dates_json': '[]',
        'n_dates_found': len(dates_with_context),
        'decision_date': None,
        'decision_date_source': None,
        'decision_confidence': None,
        'decision_date_final': None,
        'earliest_review_date': None,
        'latest_review_date': None,
        'n_review_dates': 0,
        'n_specialist_reviews': 0,
        'application_date': None,
        'inferred_application_date': None,
        'initiation_date_final': None,
        'earliest_historical_date': None,
        'n_historical_dates': 0,
        'expiration_date': None,
        'error': None,
    }

    if not dates_with_context:
        result['error'] = 'no_dates_found_by_regex'
        return result

    # Load classifier if not provided
    if classifier is None:
        try:
            classifier = BertDateClassifier()
            classifier.load()
        except Exception as e:
            result['error'] = f'bert_load_error: {str(e)}'
            return result

    # Extract contexts (classification uses original; cleaned saved for review)
    original_contexts = [d.get('context', '') for d in dates_with_context]
    cleaned_contexts = []
    cleaned_flags = []
    for ctx in original_contexts:
        cleaned, flag = _clean_context(ctx)
        cleaned_contexts.append(cleaned)
        cleaned_flags.append(flag)

    # Classify
    try:
        classifications = classifier.classify(original_contexts)
    except Exception as e:
        result['error'] = f'bert_classify_error: {str(e)}'
        return result

    # Build classified dates list
    classified_dates = []
    for date_info, classification, orig_ctx, clean_ctx, clean_flag in zip(
        dates_with_context, classifications, original_contexts, cleaned_contexts, cleaned_flags
    ):
        context_text = orig_ctx
        context_for_rules = orig_ctx
        forced_type = None
        if _is_decision_boilerplate(context_for_rules):
            forced_type = 'other'
        if _has_any_regex(context_for_rules, [r'initiator signature', r'\binitiator\b']):
            forced_type = 'initiation'
        if _has_any_regex(context_for_rules, [r'nepa compliance officer']):
            forced_type = 'decision'
        label = forced_type or classification['label']

        # Guardrail: strong decision/review cues cannot be initiation
        if label == 'initiation':
            if _has_strong_decision_cue(context_for_rules) and not _has_any_regex(context_for_rules, [r'initiator signature', r'\binitiator\b']):
                label = 'decision'
            elif _has_strong_review_cue(context_for_rules):
                label = 'review'

        # Guardrail: reviewer checkbox lines are review, not decision
        if label == 'decision' and _has_reviewer_checkbox(context_for_rules):
            label = 'review'

        # Guardrail: specialist signature lines are review, not decision
        if label == 'decision' and _has_any_regex(context_for_rules, [r'specialist signature', r'\bspecialist\b']):
            label = 'review'

        # Bucket future expiration dates (post-2025-12-31)
        if _is_expiration_candidate(date_info.get('date'), context_for_rules):
            label = 'expiration'

        # Bucket historical dates (< 2000)
        if _is_historical_by_year(date_info.get('date')):
            label = 'historical'
        elif _has_historical_context(context_for_rules):
            label = 'historical'
        classified_dates.append({
            'date': date_info['date'],
            'type': label,
            'source': context_text,
            'context': context_text,
            'context_cleaned': clean_ctx,
            'context_cleaned_flag': clean_flag,
            'confidence': 'high' if classification['confidence'] >= 0.8 else 'medium',
            'bert_confidence': classification['confidence'],
            'position_pct': date_info.get('position_pct'),
        })

    # Context-level dedupe: keep highest-confidence entry per (date, type)
    deduped = {}
    for d in classified_dates:
        key = (d['date'], d['type'])
        score = d.get('bert_confidence', 0)
        if key not in deduped or score > deduped[key].get('bert_confidence', 0):
            deduped[key] = d
    classified_dates = list(deduped.values())

    # Historical gap rule (after dedupe)
    _apply_historical_gap_rule(classified_dates, gap_days=730, enable=apply_historical_gap_rule)

    output_dates = [
        {k: v for k, v in d.items() if k not in ('context', '_decision_strength', '_boilerplate_penalty', '_confidence', '_score')}
        for d in classified_dates
    ]
    result['n_dates_found'] = len(classified_dates)

    # Extract summary fields
    decision_dates = [d for d in classified_dates if d['type'] == 'decision']
    initiation_dates = [d for d in classified_dates if d['type'] == 'initiation']
    review_dates = [d for d in classified_dates if d['type'] == 'review']
    expiration_dates = [d for d in classified_dates if d['type'] == 'expiration']
    historical_dates = [d for d in classified_dates if d['type'] == 'historical']

    # If no decision cues exist anywhere, treat memo DATE lines as review
    has_decision_cues = any(
        _has_any(d.get('context', ''), DECISION_CUES) or _has_any_regex(d.get('context', ''), DECISION_PATTERNS_STRONG)
        for d in classified_dates
    )
    if not has_decision_cues:
        for d in classified_dates:
            if d['type'] == 'other' and _has_any_regex(d.get('context', ''), MEMO_DATE_PATTERNS):
                d['type'] = 'review'

        review_dates = [d for d in classified_dates if d['type'] == 'review']

    # Decision date (cue strength, then confidence, then date)
    if decision_dates:
        best = _select_best_decision(decision_dates)
        if best:
            result['decision_date'] = best['date']
            result['decision_date_source'] = best['source']
            result['decision_confidence'] = best['confidence']
            result['decision_date_final'] = best['date']

    # Review dates
    if review_dates:
        review_dates.sort(key=lambda x: x['date'])
        result['earliest_review_date'] = review_dates[0]['date']
        result['latest_review_date'] = review_dates[-1]['date']
        result['n_review_dates'] = len(review_dates)

    # Initiation/application date (earliest)
    if initiation_dates:
        initiation_dates.sort(key=lambda x: x['date'])
        result['application_date'] = initiation_dates[0]['date']
        result['inferred_application_date'] = initiation_dates[0]['date']

        best_init = _select_best_initiation(initiation_dates, decision_date=result['decision_date'])
        if best_init:
            result['initiation_date_final'] = best_init['date']
    elif review_dates:
        result['inferred_application_date'] = result['earliest_review_date']
        result['initiation_date_final'] = result['earliest_review_date']

    # Expiration date (earliest)
    if expiration_dates:
        expiration_dates.sort(key=lambda x: x['date'])
        result['expiration_date'] = expiration_dates[0]['date']

    # Historical date (earliest + count)
    if historical_dates:
        historical_dates.sort(key=lambda x: x['date'])
        result['earliest_historical_date'] = historical_dates[0]['date']
        result['n_historical_dates'] = len(historical_dates)

    # Append final picks into the raw list with flags
    final_rows = []
    if result.get('decision_date_final'):
        final_rows.append({
            'date': result['decision_date_final'],
            'type': 'decision_final',
            'source': result.get('decision_date_source'),
            'final_flag': True,
            'final_imputed': False,
        })
    if result.get('initiation_date_final'):
        # imputed if it came from review fallback or differs from explicit initiation
        final_imputed = False
        if not initiation_dates and review_dates:
            final_imputed = True
        final_rows.append({
            'date': result['initiation_date_final'],
            'type': 'initiation_final',
            'source': None,
            'final_flag': True,
            'final_imputed': final_imputed,
        })

    output_dates.extend(final_rows)
    result['dates_json'] = json.dumps(output_dates)

    return result


def run_bert_timeline_extraction(
    sample_size: int = None,
    clean_energy_only: bool = True,
    ce_only: bool = True,
    main_docs_only: bool = True,
    output_file: str = None,
    use_regex_cache: bool = True,
    workers: int = 4,
):
    """
    Run BERT-based timeline extraction.

    Much faster than LLM approach (~50-100x speedup).

    Args:
        sample_size: If set, only process this many projects
        output_file: Custom output filename
        use_regex_cache: Use precomputed regex cache (recommended)
        workers: Number of parallel workers (less important for BERT)
    """
    import time

    print("\n=== BERT Timeline Extraction ===")

    # Load classifier
    try:
        classifier = BertDateClassifier()
        classifier.load()
    except Exception as e:
        print(f"ERROR: Could not load BERT classifier: {e}")
        print("Run --bert-train first to train the model.")
        return None

    # Load projects
    projects_path = ANALYSIS_DIR / "projects_combined.parquet"
    if not projects_path.exists():
        print(f"Error: {projects_path} not found.")
        return None

    projects = pd.read_parquet(projects_path)
    print(f"Loaded {len(projects):,} projects")

    if ce_only:
        projects = projects[projects['dataset_source'] == 'CE']
        print(f"Filtered to {len(projects):,} CE projects")

    if clean_energy_only:
        projects = projects[projects['project_energy_type'] == 'Clean']
        print(f"Filtered to {len(projects):,} clean energy projects")

    if sample_size:
        projects = projects.sample(n=min(sample_size, len(projects)), random_state=42)
        print(f"Sampled {len(projects):,} projects")

    if projects.empty:
        print("No projects to process.")
        return None

    # Load documents for filename date extraction (CE only)
    file_name_dates_map = {}
    if ce_only:
        data_dir = PROCESSED_DIR / "ce"
        documents_df = pd.read_parquet(data_dir / "documents.parquet")

        def extract_id(x):
            if isinstance(x, dict):
                return x.get('value', '')
            return x

        documents_df['project_id'] = documents_df['project_id'].apply(extract_id)
        file_name_dates_map = build_file_name_date_map(
            documents_df,
            main_docs_only=main_docs_only
        )

    # Load regex cache
    if use_regex_cache:
        if not REGEX_CACHE_PATH.exists():
            print(f"Regex cache not found. Run --regex-prep first.")
            return None
        regex_cache_df = pd.read_parquet(REGEX_CACHE_PATH)
        print(f"Loaded regex cache: {len(regex_cache_df):,} rows")
    else:
        print("ERROR: BERT approach requires regex cache. Run --regex-prep first.")
        return None

    # Process projects
    results = []
    total = len(projects)
    start_time = time.time()

    print(f"\nProcessing {total} projects with BERT...")

    for idx, (_, project) in enumerate(projects.iterrows(), 1):
        project_id = project['project_id']

        # Get cached dates for this project
        project_dates = regex_cache_df[regex_cache_df['project_id'] == project_id]
        dates_with_context = project_dates[[
            'date', 'match', 'context', 'position', 'position_pct'
        ]].to_dict(orient='records')

        # Classify with BERT (apply historical gap rule only for CE projects)
        apply_gap_rule = project.get('dataset_source') == 'CE'
        bert_result = extract_with_bert(
            dates_with_context,
            classifier,
            apply_historical_gap_rule=apply_gap_rule
        )

        # Build result row
        result = {
            'project_id': project_id,
            'bert_dates_json': bert_result.get('dates_json', '[]'),
            'bert_n_dates_found': bert_result.get('n_dates_found', 0),
            'bert_decision_date': bert_result.get('decision_date'),
            'bert_decision_date_source': bert_result.get('decision_date_source'),
            'bert_decision_confidence': bert_result.get('decision_confidence'),
            'bert_decision_date_final': bert_result.get('decision_date_final'),
            'bert_earliest_review_date': bert_result.get('earliest_review_date'),
            'bert_latest_review_date': bert_result.get('latest_review_date'),
            'bert_n_review_dates': bert_result.get('n_review_dates', 0),
            'bert_application_date': bert_result.get('application_date'),
            'bert_inferred_application_date': bert_result.get('inferred_application_date'),
            'bert_initiation_date_final': bert_result.get('initiation_date_final'),
            'bert_error': bert_result.get('error'),
            'project_file_name_dates': file_name_dates_map.get(project_id, '[]'),
        }
        results.append(result)

        if idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (total - idx) / rate if rate > 0 else 0
            print(f"  [{idx}/{total}] {rate:.1f} projects/sec, ~{remaining:.0f}s remaining")

    elapsed_total = time.time() - start_time
    print(f"\nCompleted {total} projects in {elapsed_total:.1f}s")
    print(f"Average: {elapsed_total/total*1000:.1f}ms/project")

    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df['run_timestamp'] = pd.Timestamp.utcnow().isoformat()
    projects_with_bert = projects.merge(results_df, on='project_id', how='left')

    # Save
    if output_file:
        output_path = ANALYSIS_DIR / output_file
    else:
        output_path = ANALYSIS_DIR / "projects_timeline_bert.parquet"

    projects_with_bert.to_parquet(output_path)
    print(f"\nSaved to: {output_path}")

    # Summary
    has_decision = projects_with_bert['bert_decision_date'].notna()
    has_application = projects_with_bert['bert_application_date'].notna()
    has_error = projects_with_bert['bert_error'].notna()

    print(f"\n=== Results Summary ===")
    print(f"Projects with decision date: {has_decision.sum():,} ({100*has_decision.mean():.1f}%)")
    print(f"Projects with application date: {has_application.sum():,} ({100*has_application.mean():.1f}%)")
    print(f"Projects with errors: {has_error.sum():,} ({100*has_error.mean():.1f}%)")

    return projects_with_bert


# --------------------------
# HYBRID REGEX + LLM APPROACH
# --------------------------

# Hybrid cue lists (over-inclusive by design)
DECISION_CUES = [
    'signed', 'signature', 'digitally signed', 'approved', 'approval',
    'determination', 'decision', 'decision memorandum', 'final approval',
    'authorizing official', 'field manager', 'field office manager',
    'field office manager determination', 'nepa compliance officer',
    'environmental coordinator', 'concur', 'date determined', 'initiator signature',
]

INITIATION_CUES = [
    'initiated', 'initiate', 'consultation', 'consulted', 'scoping',
    'notice of intent', 'noi', 'submitted', 'submission', 'application received',
    'received', 'request received', 'proposal submitted', 'prepared and submitted',
    'comment period', '30-day comment period', 'request for', 'right-of-way application',
    'row application', 'right of way application',
    'document creation', 'date created', 'date prepared', 'prepared', 'drafted',
    'revised', 'reviewed',
]

REVIEW_CUES = [
    'review completed', 'interim', 'phase', 'decision in principle',
    'review memo', 'review memorandum', 'reviewed by', 'nepa review',
]

# Strong exclusions (keep tight to avoid discarding useful start/creation dates)
EXCLUSION_PATTERNS = [
    r'\b\d+\s*FR\s*\d+\b',          # Federal Register citations
    r'\b\d+\s*CFR\s*\d+\b',         # CFR citations
    r'\b\d+\s*U\.?S\.?C\.?\b',      # USC citations
    r'\bFederal Register\b',
    r'https?://',
    r'\bOMB Control\b',
    r'\bPaperwork Reduction\b',
]


def _sentence_spans(text: str) -> list:
    """
    Split text into sentence-like spans with start/end indices.
    Uses punctuation and line breaks as boundaries.
    """
    spans = []
    for m in re.finditer(r'.*?(?:[.!?]+|\n{2,}|\n|$)', text, flags=re.DOTALL):
        s = m.group(0)
        if s and s.strip():
            spans.append((m.start(), m.end(), s.strip()))
    return spans


def _expand_context(spans: list, idx: int, min_chars: int = 100) -> str:
    """
    Expand context to meet a minimum character length using adjacent sentences.
    """
    if not spans:
        return ''
    start_idx = end_idx = idx
    context = spans[idx][2]
    while len(context) < min_chars and (start_idx > 0 or end_idx < len(spans) - 1):
        if start_idx > 0:
            start_idx -= 1
            context = f"{spans[start_idx][2]} {context}"
        if len(context) >= min_chars:
            break
        if end_idx < len(spans) - 1:
            end_idx += 1
            context = f"{context} {spans[end_idx][2]}"
    return re.sub(r'\s+', ' ', context).strip()


def _has_any(text: str, cues: list) -> bool:
    text_lower = text.lower()
    return any(cue in text_lower for cue in cues)

def _has_any_regex(text: str, patterns: list) -> bool:
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)


def _has_any_regex_case_sensitive(text: str, patterns: list) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)

def _min_context_chars_for_sentence(sent_text: str) -> int:
    # Use tighter context for strong signature cues; expand for weaker review/initiation cues.
    if _has_any_regex(sent_text, DECISION_PATTERNS_STRONG):
        return 60
    if _has_any(sent_text, INITIATION_CUES):
        return 140
    if _has_any(sent_text, REVIEW_CUES):
        return 160
    return 100


def _is_signature_block(context: str) -> bool:
    return _has_any_regex(context, [
        r'/s/\s*\w+',
        r'\bsigned\b',
        r'\bsignature\b',
        r'\bdate\s*:',
    ])


def _expand_for_decision_cues(spans: list, sent_idx: int, context: str) -> str:
    """
    If a date is near boilerplate text but an adjacent sentence has decision cues,
    merge the adjacent sentence to capture the signature context.
    """
    if not spans or sent_idx is None:
        return context
    if not _is_decision_boilerplate(context) and not _is_signature_block(context):
        return context

    merged = context

    def _should_pull(sent: str) -> bool:
        if not sent:
            return False
        return _has_any_regex(
            sent,
            [
                r'initiator signature',
                r'nepa compliance officer',
                r'concur',
                r'authorizing official',
                r'field manager',
            ],
        )

    # Pull adjacent lines with decision cues
    if sent_idx + 1 < len(spans):
        next_sent = spans[sent_idx + 1][2]
        if _has_any(next_sent, DECISION_CUES) or _has_any_regex(next_sent, DECISION_PATTERNS_STRONG):
            merged = f"{merged} {next_sent}"
        elif _should_pull(next_sent):
            merged = f"{merged} {next_sent}"
    if sent_idx - 1 >= 0:
        prev_sent = spans[sent_idx - 1][2]
        if _has_any(prev_sent, DECISION_CUES) or _has_any_regex(prev_sent, DECISION_PATTERNS_STRONG):
            merged = f"{prev_sent} {merged}"
        elif _should_pull(prev_sent):
            merged = f"{prev_sent} {merged}"

    # Pull up to two prior lines if this looks like a signature/date block
    if _is_signature_block(context) and sent_idx - 2 >= 0:
        prev2 = spans[sent_idx - 2][2]
        if _should_pull(prev2):
            merged = f"{prev2} {merged}"

    return re.sub(r'\\s+', ' ', merged).strip()


def _clean_context(context: str) -> tuple:
    """
    Strip boilerplate lines from context while preserving original text.
    Returns (cleaned_text, cleaned_flag).
    """
    if not context:
        return "", False

    original = context
    # Line-level removal for robust boilerplate stripping
    lines = re.split(r'[\\n\\r]+', original)
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # If the Recovery Act clause is on the same line as a signature,
        # drop only the clause and keep the signature tail.
        if re.search(r'Recovery Act', line_stripped, flags=re.IGNORECASE):
            if re.search(r'Approved by|NEPA Compliance Officer|Authorizing Official|Field Office Manager|Signature|Signed by', line_stripped, flags=re.IGNORECASE):
                split = re.split(r'Recovery Act\\)?\\s*', line_stripped, flags=re.IGNORECASE, maxsplit=1)
                if len(split) > 1 and split[1].strip():
                    line_stripped = split[1].strip()
                else:
                    continue
            else:
                continue
        if re.search(r'\\bDOE F \\d+[\\w\\.\\-]*', line_stripped, flags=re.IGNORECASE):
            continue
        if re.search(r'\\bNETL F \\d+[\\w\\.\\-]*', line_stripped, flags=re.IGNORECASE):
            continue
        if re.search(r'Previous Editions Obsolete', line_stripped, flags=re.IGNORECASE):
            continue
        if re.search(r'Electronic Form Approved by Forms Mgmt\\.', line_stripped, flags=re.IGNORECASE):
            continue
        if re.search(r'Paperwork Reduction Act', line_stripped, flags=re.IGNORECASE):
            continue
        if re.search(r'OMB Control', line_stripped, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line_stripped)

    cleaned = ' '.join(cleaned_lines)
    cleaned = re.sub(r'\\s+', ' ', cleaned).strip()
    return cleaned, cleaned != original


def _is_excluded_context(text: str) -> bool:
    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def extract_dates_with_context(
    text: str,
    context_window: int = DATE_CONTEXT_WINDOW,
    min_context_chars: int = 80,
    keep_all_candidates: bool = False,
    dedupe_by_date: bool = True,
) -> list:
    """
    Extract candidate dates from text using regex, with sentence-based context.

    This is the first step of the hybrid approach:
    1. Regex finds all dates (fast, reliable)
    2. Context is captured for LLM classification (sentence-based, expanded if short)
    3. Only initiation/decision candidates are kept

    Args:
        text: Full document text
        context_window: Fallback window if sentence detection fails
        min_context_chars: Minimum context length (expand with adjacent sentences)

    Returns:
        List of dicts: [{'date': 'YYYY-MM-DD', 'match': '01/15/2024', 'context': '...'}, ...]
    """
    results = []
    seen_dates = set()
    seen_contexts = set()

    # Build sentence spans once for context extraction
    spans = _sentence_spans(text)

    # Map sentence index to date matches
    sentence_matches = [[] for _ in spans]

    for pattern, pattern_type in DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            date_obj = parse_date_match(match, pattern_type)

            if date_obj is None:
                continue

            # Filter to reasonable year range (but allow historical for classification)
            if date_obj.year < 1950 or date_obj.year > 2030:
                continue

            # Skip obvious law/statute years
            if should_exclude_date(text, match.start(), match.end(), window=30):
                continue

            date_str = date_obj.strftime('%Y-%m-%d')

            # Find sentence index containing this match
            sent_idx = None
            for i, (s_start, s_end, _) in enumerate(spans):
                if s_start <= match.start() < s_end:
                    sent_idx = i
                    break

            if sent_idx is None:
                # Fallback to window if no sentence found
                start = max(0, match.start() - context_window)
                end = min(len(text), match.end() + context_window)
                context = re.sub(r'\s+', ' ', text[start:end]).strip()
            else:
                min_chars = _min_context_chars_for_sentence(spans[sent_idx][2])
                context = _expand_context(spans, sent_idx, min_chars=min_chars)
                context = _expand_for_decision_cues(spans, sent_idx, context)

            # Exclusion filters (FR/CFR/USC/URLs/OMB)
            if _is_excluded_context(context):
                continue

            sentence_matches[sent_idx].append({
                'date_str': date_str,
                'match': match.group(),
                'position': match.start(),
            })

            # Candidate filter: keep if context has decision/initiation/review cues
            if not keep_all_candidates:
                if not (_has_any(context, DECISION_CUES)
                        or _has_any(context, INITIATION_CUES)
                        or _has_any(context, REVIEW_CUES)):
                    continue

            # Deduplicate by context to avoid duplicate signature sentences
            context_key = re.sub(r'\s+', ' ', context).strip().lower()
            if context_key in seen_contexts:
                continue

            # Deduplicate by date string (optional)
            if dedupe_by_date:
                if date_str in seen_dates:
                    continue
                seen_dates.add(date_str)
            seen_contexts.add(context_key)

            results.append({
                'date': date_str,
                'match': match.group(),
                'context': context,
                'position': match.start(),
                'position_pct': match.start() / len(text) * 100 if len(text) > 0 else 0,
            })

    # Link initiation cue in sentence without date to date in next sentence
    for i, (_, _, sent_text) in enumerate(spans):
        if not sent_text:
            continue
        if not _has_any(sent_text, INITIATION_CUES):
            continue
        if sentence_matches[i]:
            continue  # already has dates in this sentence
        if i + 1 >= len(spans):
            continue
        if not sentence_matches[i + 1]:
            continue

        # Combine initiation sentence with next sentence context
        combined = f"{sent_text} {spans[i + 1][2]}"
        combined = re.sub(r'\s+', ' ', combined).strip()
        if _is_excluded_context(combined):
            continue
        combined_key = combined.lower()

        for m in sentence_matches[i + 1]:
            if m['date_str'] in seen_dates:
                continue
            if combined_key in seen_contexts:
                continue
            seen_dates.add(m['date_str'])
            seen_contexts.add(combined_key)
            results.append({
                'date': m['date_str'],
                'match': m['match'],
                'context': combined,
                'position': m['position'],
                'position_pct': m['position'] / len(text) * 100 if len(text) > 0 else 0,
            })

    # Sort by date
    results.sort(key=lambda x: x['date'])

    return results


def create_classification_prompt(dates_with_context: list) -> str:
    """
    Create a streamlined prompt for LLM to classify pre-extracted dates.

    This is much shorter than the full-document prompt because:
    - Dates are already found by regex
    - LLM only needs to classify based on context

    Args:
        dates_with_context: List from extract_dates_with_context()

    Returns:
        Prompt string for LLM
    """
    if not dates_with_context:
        return ""

    lines = [
        "Classify each date extracted from a NEPA Categorical Exclusion document.",
        "",
        "Only classify the following date types:",
        "- decision: Final approval signature (Field Manager, NEPA Compliance Officer, Authorizing Official)",
        "- initiation: Start of the review or consultation (e.g., initiated consultation, application received, NOI, scoping)",
        "- review: Interim review or phase approval (not final decision, not initiation)",
        "- other: Anything else",
        "",
        "DATES TO CLASSIFY:",
        ""
    ]

    for i, d in enumerate(dates_with_context, 1):
        lines.append(f'{i}. {d["match"]} ({d["date"]})')
        lines.append(f'   Context: "...{d["context"]}..."')
        lines.append("")

    lines.extend([
        "Return JSON only:",
        '{"classifications": [',
        '  {"date": "YYYY-MM-DD", "type": "decision|initiation|review|other", "reason": "brief explanation"},',
        '  ...',
        ']}'
    ])

    return "\n".join(lines)


def parse_classification_response(response_text: str, dates_with_context: list) -> dict:
    """
    Parse the LLM classification response.

    Args:
        response_text: Raw LLM response
        dates_with_context: Original dates list (for fallback)

    Returns:
        Dict with classified dates and summary fields
    """
    import json

    result = {
        'dates_json': '[]',
        'n_dates_found': len(dates_with_context),
        'decision_date': None,
        'decision_date_source': None,
        'decision_confidence': None,
        'earliest_review_date': None,
        'latest_review_date': None,
        'n_review_dates': 0,
        'application_date': None,
        'inferred_application_date': None,
        'earliest_historical_date': None,
        'n_historical_dates': 0,
        'expiration_date': None,
        'parse_error': None,
        'raw_response': response_text[:500] if response_text else None,
    }

    if not response_text:
        result['parse_error'] = 'empty_response'
        return result

    try:
        # Find JSON in response
        json_match = re.search(r'\{[^{}]*"classifications"\s*:\s*\[[^\]]*\][^{}]*\}', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if json_match:
            parsed = json.loads(json_match.group())
            classifications = parsed.get('classifications', [])

            # Build classified dates list
            context_map = {}
            for orig in dates_with_context:
                if orig.get('date') not in context_map:
                    context_map[orig.get('date')] = {
                        'context': orig.get('context', ''),
                        'position_pct': orig.get('position_pct'),
                    }

            classified_dates = []
            for c in classifications:
                if not isinstance(c, dict):
                    continue

                date_str = normalize_date_string(c.get('date'))
                if date_str:
                    ctx_info = context_map.get(date_str, {})
                    classified_dates.append({
                        'date': date_str,
                        'type': c.get('type', 'other'),
                        'source': str(c.get('reason', ''))[:200],
                        'context': ctx_info.get('context', ''),
                        'position_pct': ctx_info.get('position_pct'),
                        'confidence': 'high',  # LLM classified it
                        'confidence_score': 1.0,
                    })

            # If LLM didn't return all dates, add unclassified ones
            classified_date_strs = {d['date'] for d in classified_dates}
            for orig in dates_with_context:
                if orig['date'] not in classified_date_strs:
                    classified_dates.append({
                        'date': orig['date'],
                        'type': 'other',
                        'source': 'not classified by LLM',
                        'context': orig.get('context', ''),
                        'position_pct': orig.get('position_pct'),
                        'confidence': 'low',
                        'confidence_score': 0.2,
                    })

            output_dates = [
                {k: v for k, v in d.items() if k not in ('context', '_decision_strength', '_boilerplate_penalty',
                                                        '_confidence', '_score')}
                for d in classified_dates
            ]
            result['dates_json'] = json.dumps(output_dates)
            result['n_dates_found'] = len(classified_dates)

            # Extract summary fields by type (hybrid focuses on decision/initiation)
            decision_dates = [d for d in classified_dates if d['type'] == 'decision']
            initiation_types = {'initiation', 'application', 'start'}
            application_dates = [d for d in classified_dates if d['type'] in initiation_types]
            review_types = {'review', 'specialist_review'}
            review_dates = [d for d in classified_dates if d['type'] in review_types]
            historical_dates = [d for d in classified_dates if d['type'] == 'historical']
            expiration_dates = [d for d in classified_dates if d['type'] == 'expiration']

            # Decision date (cue strength, then confidence, then date)
            if decision_dates:
                best = _select_best_decision(decision_dates)
                if best:
                    result['decision_date'] = best['date']
                    result['decision_date_source'] = best['source']
                    result['decision_confidence'] = best['confidence']

            # Review dates
            if review_dates:
                review_dates.sort(key=lambda x: x['date'])
                result['earliest_review_date'] = review_dates[0]['date']
                result['latest_review_date'] = review_dates[-1]['date']
                result['n_review_dates'] = len(review_dates)
                result['n_specialist_reviews'] = len(review_dates)

            # Initiation/application date
            if application_dates:
                application_dates.sort(key=lambda x: x['date'])
                result['application_date'] = application_dates[0]['date']

            # Inferred application date
            if result['earliest_review_date'] and not result['application_date']:
                result['inferred_application_date'] = result['earliest_review_date']
            elif result['application_date']:
                result['inferred_application_date'] = result['application_date']

            # Historical dates
            if historical_dates:
                historical_dates.sort(key=lambda x: x['date'])
                result['earliest_historical_date'] = historical_dates[0]['date']
                result['n_historical_dates'] = len(historical_dates)

            # Expiration date
            if expiration_dates:
                expiration_dates.sort(key=lambda x: x['date'])
                result['expiration_date'] = expiration_dates[-1]['date']

        else:
            result['parse_error'] = 'no_json_found'

    except json.JSONDecodeError as e:
        result['parse_error'] = f'json_decode_error: {str(e)}'
    except Exception as e:
        result['parse_error'] = f'parse_error: {str(e)}'

    return result


def extract_with_hybrid_approach(
    text: str,
    model: str = DEFAULT_LLM_MODEL,
    timeout: int = 120,
    context_window: int = DATE_CONTEXT_WINDOW,
    dates_with_context: list = None,
) -> dict:
    """
    Extract timeline using hybrid regex + LLM approach.

    1. Regex extracts all dates with context (fast)
    2. LLM classifies each date based on context (accurate)

    This is more efficient than the full-document approach:
    - ~80% fewer tokens sent to LLM
    - Clearer task for LLM (just classify, don't find)
    - Regex handles date finding (what it's good at)
    - LLM handles understanding (what it's good at)

    Args:
        text: Full document text
        model: Ollama model name
        timeout: Request timeout in seconds
        context_window: Chars to capture around each date

    Returns:
        Dict with classified dates and summary fields
    """
    import requests

    result = {
        'llm_model': model,
        'original_chars': len(text),
        'approach': 'hybrid_regex_llm',
        'error': None,
    }

    # Step 1: Extract dates with context using regex (or use precomputed cache)
    if dates_with_context is None:
        dates_with_context = extract_dates_with_context(text, context_window)
    result['n_dates_regex'] = len(dates_with_context)

    if not dates_with_context:
        result['error'] = 'no_dates_found_by_regex'
        result['dates_json'] = '[]'
        result['n_dates_found'] = 0
        return result

    # Step 2: Create classification prompt
    prompt = create_classification_prompt(dates_with_context)
    result['prompt_chars'] = len(prompt)

    # Step 3: Call LLM for classification
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'num_predict': 256,  # Shorter response needed
                    }
            },
            timeout=timeout
        )

        if response.status_code == 200:
            llm_response = response.json().get('response', '')

            # Parse classification response
            parsed = parse_classification_response(llm_response, dates_with_context)
            result.update(parsed)

            # Add timing info
            result['llm_eval_duration_ms'] = response.json().get('eval_duration', 0) / 1e6

        else:
            result['error'] = f'ollama_http_error: {response.status_code}'

    except requests.exceptions.ConnectionError:
        result['error'] = 'ollama_not_running'
    except requests.exceptions.Timeout:
        result['error'] = 'ollama_timeout'
    except Exception as e:
        result['error'] = f'ollama_error: {str(e)}'

    return result


def create_llm_prompt(processed_text: str) -> str:
    """
    Create a prompt for LLM timeline extraction - extracts ALL dates with context.

    Args:
        processed_text: Preprocessed document text (from preprocess_for_timeline)

    Returns:
        str: Prompt for LLM
    """
    prompt = f"""You are extracting ALL dates from a NEPA (National Environmental Policy Act) Categorical Exclusion document.

DOCUMENT TEXT (key sections extracted):
---
{processed_text}
---

Extract EVERY date you find in the document. For each date, identify:

1. DATE TYPES:
   - "decision": Final approval by authorizing official (Field Manager, NEPA Compliance Officer, etc.)
   - "specialist_review": Individual specialist sign-offs (wildlife biologist, archaeologist, realty specialist, etc.)
   - "application": When proposal/application was submitted or received
   - "historical": Prior actions, original permit/ROW issuances from past years
   - "expiration": End dates for permits, authorizations, or comment periods
   - "effective": When an action becomes effective
   - "other": Any other dates that don't fit above categories

2. For each date provide:
   - The date in YYYY-MM-DD format
   - The type from the list above
   - A brief quote (10-20 words) showing context
   - Confidence: high (clearly stated), medium (inferred), low (uncertain)

Return JSON only:
{{
  "dates": [
    {{"date": "YYYY-MM-DD", "type": "decision", "source": "Field Manager signed on...", "confidence": "high"}},
    {{"date": "YYYY-MM-DD", "type": "specialist_review", "source": "Wildlife biologist approved...", "confidence": "high"}},
    ...
  ]
}}

Important:
- Include ALL dates found, even historical ones from prior years
- The decision date is usually the Field Manager or Authorizing Official signature date
- Specialist review dates help establish when the review process occurred
- If no dates found, return {{"dates": []}}
- Return ONLY the JSON object, no other text."""

    return prompt


def normalize_date_string(val: str) -> str:
    """
    Normalize a date string to YYYY-MM-DD format.

    Args:
        val: Date string in various formats

    Returns:
        Normalized date string or None if invalid
    """
    if not val or val == 'null' or str(val).lower() == 'none':
        return None

    val_str = str(val).strip()

    # Try various date formats
    date_formats = [
        ('%Y-%m-%d', r'^\d{4}-\d{2}-\d{2}$'),      # 2024-03-20
        ('%m/%d/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),  # 03/20/2024
        ('%m/%d/%y', r'^\d{1,2}/\d{1,2}/\d{2}$'),  # 03/20/24
        ('%m-%d-%Y', r'^\d{1,2}-\d{1,2}-\d{4}$'),  # 03-20-2024
        ('%Y.%m.%d', r'^\d{4}\.\d{2}\.\d{2}$'),    # 2024.03.20
    ]

    for fmt, pattern in date_formats:
        if re.match(pattern, val_str):
            try:
                parsed_date = datetime.strptime(val_str, fmt)
                # Validate year range (allow historical dates for this workflow)
                if 1950 <= parsed_date.year <= 2030:
                    return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue

    return None


def parse_llm_response(response_text: str) -> dict:
    """
    Parse the LLM response to extract structured multi-date data.

    Args:
        response_text: Raw text response from LLM

    Returns:
        dict with parsed fields including all dates and summary
    """
    import json

    # Initialize result with all fields
    result = {
        # All dates as JSON array
        'dates_json': '[]',
        'n_dates_found': 0,

        # Summary fields (extracted from dates array)
        'decision_date': None,
        'decision_date_source': None,
        'decision_confidence': None,

        'earliest_review_date': None,
        'latest_review_date': None,
        'n_review_dates': 0,
        'n_specialist_reviews': 0,

        'application_date': None,
        'inferred_application_date': None,  # Earliest review if no explicit application

        'earliest_historical_date': None,
        'n_historical_dates': 0,

        'expiration_date': None,

        # Metadata
        'parse_error': None,
        'raw_response': response_text[:500] if response_text else None,
    }

    if not response_text:
        result['parse_error'] = 'empty_response'
        return result

    try:
        # Try to find JSON in the response - need to match nested structure
        # Look for {"dates": [...]} pattern
        json_match = re.search(r'\{[^{}]*"dates"\s*:\s*\[[^\]]*\][^{}]*\}', response_text, re.DOTALL)
        if not json_match:
            # Fallback: try to find any JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)

            dates_list = parsed.get('dates', [])
            if not isinstance(dates_list, list):
                dates_list = []

            # Normalize all dates and filter valid ones
            valid_dates = []
            for d in dates_list:
                if not isinstance(d, dict):
                    continue

                normalized = normalize_date_string(d.get('date'))
                if normalized:
                    valid_dates.append({
                        'date': normalized,
                        'type': d.get('type', 'other'),
                        'source': str(d.get('source', ''))[:200],  # Truncate long sources
                        'confidence': d.get('confidence', 'medium'),
                    })

            # Store all dates as JSON
            result['dates_json'] = json.dumps(valid_dates)
            result['n_dates_found'] = len(valid_dates)

            # Extract summary fields by type
            decision_dates = [d for d in valid_dates if d['type'] == 'decision']
            review_dates = [d for d in valid_dates if d['type'] in {'review', 'specialist_review'}]
            application_dates = [d for d in valid_dates if d['type'] == 'application']
            historical_dates = [d for d in valid_dates if d['type'] == 'historical']
            expiration_dates = [d for d in valid_dates if d['type'] == 'expiration']

            # Decision date (use latest if multiple)
            if decision_dates:
                decision_dates.sort(key=lambda x: x['date'])
                latest_decision = decision_dates[-1]
                result['decision_date'] = latest_decision['date']
                result['decision_date_source'] = latest_decision['source']
                result['decision_confidence'] = latest_decision['confidence']

            # Specialist review dates
            if review_dates:
                review_dates.sort(key=lambda x: x['date'])
                result['earliest_review_date'] = review_dates[0]['date']
                result['latest_review_date'] = review_dates[-1]['date']
                result['n_review_dates'] = len(review_dates)
                result['n_specialist_reviews'] = len(review_dates)

            # Application date
            if application_dates:
                application_dates.sort(key=lambda x: x['date'])
                result['application_date'] = application_dates[0]['date']

            # Inferred application date (earliest review if no explicit application)
            if result['earliest_review_date'] and not result['application_date']:
                result['inferred_application_date'] = result['earliest_review_date']
            elif result['application_date']:
                result['inferred_application_date'] = result['application_date']

            # Historical dates
            if historical_dates:
                historical_dates.sort(key=lambda x: x['date'])
                result['earliest_historical_date'] = historical_dates[0]['date']
                result['n_historical_dates'] = len(historical_dates)

            # Expiration date (use latest if multiple)
            if expiration_dates:
                expiration_dates.sort(key=lambda x: x['date'])
                result['expiration_date'] = expiration_dates[-1]['date']

        else:
            result['parse_error'] = 'no_json_found'

    except json.JSONDecodeError as e:
        result['parse_error'] = f'json_decode_error: {str(e)}'
    except Exception as e:
        result['parse_error'] = f'parse_error: {str(e)}'

    return result


def extract_with_ollama(
    text: str,
    model: str = DEFAULT_LLM_MODEL,
    timeout: int = 240,
    preprocess: bool = True,
) -> dict:
    """
    Extract timeline using local Ollama LLM with preprocessing.

    Args:
        text: Document text (will be preprocessed if preprocess=True)
        model: Ollama model name
        timeout: Request timeout in seconds
        preprocess: Whether to preprocess text first

    Returns:
        dict with extracted timeline and metadata (all dates + summary fields)
    """
    import requests

    result = {
        # Preprocessing metadata
        'llm_model': model,
        'original_chars': len(text),
        'processed_chars': 0,
        'reduction_pct': 0,
        'error': None,
    }

    # Preprocess if requested
    if preprocess:
        preprocess_result = preprocess_for_timeline(text)
        processed_text = preprocess_result.processed_text
        result['processed_chars'] = len(processed_text)
        result['reduction_pct'] = preprocess_result.reduction_pct
        result['keywords_found'] = preprocess_result.keywords_found
    else:
        processed_text = text[:8000]  # Truncate if not preprocessing
        result['processed_chars'] = len(processed_text)

    # Create prompt
    prompt = create_llm_prompt(processed_text)
    result['prompt_chars'] = len(prompt)

    # Call Ollama API
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,  # Low temperature for consistent extraction
                    'num_predict': 1024,  # Increased for multi-date response
                }
            },
            timeout=timeout
        )

        if response.status_code == 200:
            response_data = response.json()
            llm_response = response_data.get('response', '')

            # Parse the response (now returns all dates + summary)
            parsed = parse_llm_response(llm_response)
            result.update(parsed)

            # Add timing info if available
            result['llm_eval_duration_ms'] = response_data.get('eval_duration', 0) / 1e6

        else:
            result['error'] = f'ollama_http_error: {response.status_code}'

    except requests.exceptions.ConnectionError:
        result['error'] = 'ollama_not_running'
    except requests.exceptions.Timeout:
        result['error'] = 'ollama_timeout'
    except Exception as e:
        result['error'] = f'ollama_error: {str(e)}'

    return result


def build_project_timeline_llm(
    project_id: str,
    pages_df: pd.DataFrame,
    documents_df: pd.DataFrame,
    model: str = DEFAULT_LLM_MODEL,
    use_hybrid: bool = False,
    timeout: int = 120,
    dates_with_context: list = None,
) -> dict:
    """
    Build a timeline for a single project using LLM extraction.

    Extracts ALL dates with context and provides summary fields.

    Args:
        project_id: Project ID string
        pages_df: DataFrame with page text (already filtered to main docs if needed)
        documents_df: DataFrame with document metadata
        model: Ollama model name

    Returns:
        dict with timeline information including all dates and summary fields
    """
    # Get documents for this project
    project_docs = documents_df[documents_df['project_id'] == project_id].copy()

    # Count documents
    document_count = len(project_docs)
    main_document_count = 0
    if not project_docs.empty and 'main_document' in project_docs.columns:
        main_document_count = (project_docs['main_document'] == 'YES').sum()

    result = {
        'project_id': project_id,
        'project_document_count': document_count,
        'project_main_document_count': main_document_count,

        # All dates as JSON
        'llm_dates_json': '[]',
        'llm_n_dates_found': 0,

        # Decision date fields
        'llm_decision_date': None,
        'llm_decision_date_source': None,
        'llm_decision_confidence': None,

        # Review fields
        'llm_earliest_review_date': None,
        'llm_latest_review_date': None,
        'llm_n_review_dates': 0,
        'llm_n_specialist_reviews': 0,

        # Application date fields
        'llm_application_date': None,
        'llm_inferred_application_date': None,

        # Historical dates
        'llm_earliest_historical_date': None,
        'llm_n_historical_dates': 0,

        # Expiration
        'llm_expiration_date': None,

        # Metadata
        'llm_model': model,
        'llm_approach': 'hybrid' if use_hybrid else 'full_document',
        'llm_error': None,
        'llm_processed_chars': 0,
        'llm_reduction_pct': 0,
        'llm_raw_response': None,
    }

    if project_docs.empty:
        result['llm_error'] = 'no_documents'
        return result

    # If hybrid with cached dates, skip loading page text
    if use_hybrid and dates_with_context is not None:
        all_text = ""
        result['total_chars'] = 0
    else:
        # Get page text for this project
        doc_ids = project_docs['document_id'].tolist()
        project_pages = pages_df[pages_df['document_id'].isin(doc_ids)]

        if project_pages.empty:
            result['llm_error'] = 'no_pages'
            return result

        # Combine all page text
        all_text = "\n\n".join(project_pages['page_text'].dropna().tolist())
        result['total_chars'] = len(all_text)

        if not all_text.strip():
            result['llm_error'] = 'empty_text'
            return result

    # Extract with LLM (hybrid or full-document approach)
    if use_hybrid:
        llm_result = extract_with_hybrid_approach(
            all_text,
            model=model,
            timeout=timeout,
            dates_with_context=dates_with_context,
        )
    else:
        llm_result = extract_with_ollama(all_text, model=model, timeout=timeout)

    # Merge all results
    result['llm_dates_json'] = llm_result.get('dates_json', '[]')
    result['llm_n_dates_found'] = llm_result.get('n_dates_found', 0)

    result['llm_decision_date'] = llm_result.get('decision_date')
    result['llm_decision_date_source'] = llm_result.get('decision_date_source')
    result['llm_decision_confidence'] = llm_result.get('decision_confidence')

    result['llm_earliest_review_date'] = llm_result.get('earliest_review_date')
    result['llm_latest_review_date'] = llm_result.get('latest_review_date')
    result['llm_n_review_dates'] = llm_result.get('n_review_dates', llm_result.get('n_specialist_reviews', 0))
    result['llm_n_specialist_reviews'] = llm_result.get('n_specialist_reviews', result['llm_n_review_dates'])

    result['llm_application_date'] = llm_result.get('application_date')
    result['llm_inferred_application_date'] = llm_result.get('inferred_application_date')

    result['llm_earliest_historical_date'] = llm_result.get('earliest_historical_date')
    result['llm_n_historical_dates'] = llm_result.get('n_historical_dates', 0)

    result['llm_expiration_date'] = llm_result.get('expiration_date')

    result['llm_error'] = llm_result.get('error')
    result['llm_processed_chars'] = llm_result.get('processed_chars', 0)
    result['llm_prompt_chars'] = llm_result.get('prompt_chars', 0)
    result['llm_reduction_pct'] = llm_result.get('reduction_pct', 0)
    result['llm_raw_response'] = llm_result.get('raw_response')

    return result


def run_llm_timeline_extraction(
    sample_size: int = None,
    clean_energy_only: bool = True,
    ce_only: bool = True,
    main_docs_only: bool = True,
    model: str = DEFAULT_LLM_MODEL,
    output_file: str = None,
    use_hybrid: bool = False,
    timeout: int = 120,
    use_regex_cache: bool = False,
    regex_cache_path: Path = REGEX_CACHE_PATH,
    workers: int = 4,
):
    """
    Run LLM-based timeline extraction for CE clean energy projects.

    Args:
        sample_size: If set, only process this many projects
        clean_energy_only: If True, only process clean energy projects
        ce_only: If True, only process CE (Categorical Exclusion) projects
        main_docs_only: If True, only read pages from main_document == 'YES' documents
        model: Ollama model name
        output_file: Custom output filename (default: projects_timeline_llm.parquet)
        use_hybrid: If True, use hybrid regex+LLM approach (faster)
        timeout: Timeout in seconds per document

    Returns:
        DataFrame with results
    """
    approach = "hybrid regex+LLM" if use_hybrid else "full-document"
    print(f"\n=== LLM Timeline Extraction ({approach}) ===")
    print(f"Model: {model}")
    print(f"Timeout: {timeout}s")

    # Load projects
    projects_path = ANALYSIS_DIR / "projects_combined.parquet"
    if not projects_path.exists():
        print(f"Error: {projects_path} not found. Run extract_data.py first.")
        return None

    projects = pd.read_parquet(projects_path)
    print(f"Loaded {len(projects):,} projects")

    # Filter by dataset source (CE only)
    if ce_only:
        projects = projects[projects['dataset_source'] == 'CE']
        print(f"Filtered to {len(projects):,} CE projects")

    # Filter by energy type
    if clean_energy_only:
        projects = projects[projects['project_energy_type'] == 'Clean']
        print(f"Filtered to {len(projects):,} clean energy projects")

    if sample_size:
        projects = projects.sample(n=min(sample_size, len(projects)), random_state=42)
        print(f"Sampled {len(projects):,} projects")

    if projects.empty:
        print("No projects to process after filtering.")
        return None

    # Load CE data
    print("\nLoading CE document data...")
    data_dir = PROCESSED_DIR / "ce"
    pages_df = pd.read_parquet(data_dir / "pages.parquet")
    documents_df = pd.read_parquet(data_dir / "documents.parquet")

    # Clean project_id in documents
    def extract_id(x):
        if isinstance(x, dict):
            return x.get('value', '')
        return x

    documents_df['project_id'] = documents_df['project_id'].apply(extract_id)

    # Precompute filename date matches
    file_name_dates_map = build_file_name_date_map(
        documents_df,
        main_docs_only=main_docs_only
    )

    # Filter to main documents only if requested
    if main_docs_only:
        main_doc_ids = documents_df[documents_df['main_document'] == 'YES']['document_id'].tolist()
        pages_df = pages_df[pages_df['document_id'].isin(main_doc_ids)]
        print(f"Filtered to {len(pages_df):,} pages from main documents")

    regex_cache_df = None
    if use_hybrid and use_regex_cache:
        if not regex_cache_path.exists():
            print(f"Error: regex cache not found at {regex_cache_path}. Run --regex-prep first.")
            return None
        regex_cache_df = pd.read_parquet(regex_cache_path)
        print(f"Loaded regex cache: {len(regex_cache_df):,} rows")

    # Process each project
    results = []
    total = len(projects)

    print(f"\nProcessing {total} projects with LLM...")
    print("(This may take a while)\n")

    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    start_time = time.time()

    def _run_one(project_id: str):
        if use_hybrid and use_regex_cache and regex_cache_df is not None:
            cached_dates = regex_cache_df[regex_cache_df['project_id'] == project_id]
            dates_with_context = cached_dates[[
                'date', 'match', 'context', 'position', 'position_pct'
            ]].to_dict(orient='records')
            result = build_project_timeline_llm(
                project_id, pages_df, documents_df, model=model,
                use_hybrid=True, timeout=timeout, dates_with_context=dates_with_context
            )
            result['project_file_name_dates'] = file_name_dates_map.get(project_id, '[]')
            return result
        result = build_project_timeline_llm(
            project_id, pages_df, documents_df, model=model,
            use_hybrid=use_hybrid, timeout=timeout
        )
        result['project_file_name_dates'] = file_name_dates_map.get(project_id, '[]')
        return result

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for _, project in projects.iterrows():
                futures.append(executor.submit(_run_one, project['project_id']))

            for idx, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if idx % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = idx / elapsed
                    remaining = (total - idx) / rate if rate > 0 else 0
                    print(f"  [{idx}/{total}] {rate:.1f} projects/sec, ~{remaining/60:.1f} min remaining")
    else:
        for idx, (_, project) in enumerate(projects.iterrows(), 1):
            project_id = project['project_id']
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (total - idx) / rate if rate > 0 else 0
                print(f"  [{idx}/{total}] {rate:.1f} projects/sec, ~{remaining/60:.1f} min remaining")
            results.append(_run_one(project_id))

    elapsed_total = time.time() - start_time
    print(f"\nCompleted {total} projects in {elapsed_total/60:.1f} minutes")
    print(f"Average: {elapsed_total/total:.2f} sec/project")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Merge with project metadata
    projects_with_llm = projects.merge(results_df, on='project_id', how='left')

    # Save
    if output_file:
        output_path = ANALYSIS_DIR / output_file
    else:
        output_path = ANALYSIS_DIR / "projects_timeline_llm.parquet"

    projects_with_llm.to_parquet(output_path)
    print(f"\nSaved to: {output_path}")

    # Summary
    has_decision = projects_with_llm['llm_decision_date'].notna()
    has_application = projects_with_llm['llm_application_date'].notna()
    has_inferred_app = projects_with_llm['llm_inferred_application_date'].notna()
    has_reviews = projects_with_llm['llm_n_specialist_reviews'] > 0
    high_conf = projects_with_llm['llm_decision_confidence'] == 'high'
    has_error = projects_with_llm['llm_error'].notna()

    print(f"\n=== Results Summary ===")
    print(f"Projects with decision date: {has_decision.sum():,} ({100*has_decision.mean():.1f}%)")
    print(f"Projects with explicit application date: {has_application.sum():,} ({100*has_application.mean():.1f}%)")
    print(f"Projects with inferred application date: {has_inferred_app.sum():,} ({100*has_inferred_app.mean():.1f}%)")
    print(f"Projects with specialist reviews: {has_reviews.sum():,} ({100*has_reviews.mean():.1f}%)")
    print(f"High confidence decisions: {high_conf.sum():,} ({100*high_conf.mean():.1f}%)")
    print(f"Projects with errors: {has_error.sum():,} ({100*has_error.mean():.1f}%)")

    # Date count distribution
    print(f"\nDates found per project:")
    print(projects_with_llm['llm_n_dates_found'].describe())

    if has_error.sum() > 0:
        print(f"\nError breakdown:")
        print(projects_with_llm[has_error]['llm_error'].value_counts())

    return projects_with_llm


def run_regex_prep(
    sample_size: int = None,
    clean_energy_only: bool = True,
    ce_only: bool = True,
    main_docs_only: bool = True,
    output_file: str = None,
    keep_all_candidates: bool = True,
):
    """
    Precompute regex candidate dates with context for hybrid LLM runs.

    Saves a single cache file to data/analysis for reuse across runs.
    """
    print("\n=== Regex Candidate Preprocessing ===")

    # Load projects
    projects_path = ANALYSIS_DIR / "projects_combined.parquet"
    if not projects_path.exists():
        print(f"Error: {projects_path} not found. Run extract_data.py first.")
        return None

    projects = pd.read_parquet(projects_path)
    print(f"Loaded {len(projects):,} projects")

    # Filter by dataset source (CE only)
    if ce_only:
        projects = projects[projects['dataset_source'] == 'CE']
        print(f"Filtered to {len(projects):,} CE projects")

    # Filter by energy type
    if clean_energy_only:
        projects = projects[projects['project_energy_type'] == 'Clean']
        print(f"Filtered to {len(projects):,} clean energy projects")

    if sample_size:
        projects = projects.sample(n=min(sample_size, len(projects)), random_state=42)
        print(f"Sampled {len(projects):,} projects")

    if projects.empty:
        print("No projects to process after filtering.")
        return None

    # Load CE data
    print("\nLoading CE document data...")
    data_dir = PROCESSED_DIR / "ce"
    pages_df = pd.read_parquet(data_dir / "pages.parquet")
    documents_df = pd.read_parquet(data_dir / "documents.parquet")

    # Clean project_id in documents
    def extract_id(x):
        if isinstance(x, dict):
            return x.get('value', '')
        return x

    documents_df['project_id'] = documents_df['project_id'].apply(extract_id)

    # Filter to main documents only if requested
    if main_docs_only:
        main_doc_ids = documents_df[documents_df['main_document'] == 'YES']['document_id'].tolist()
        pages_df = pages_df[pages_df['document_id'].isin(main_doc_ids)]
        print(f"Filtered to {len(pages_df):,} pages from main documents")

    results = []
    total = len(projects)

    print(f"\nProcessing {total} projects for regex candidates...")

    for idx, (_, project) in enumerate(projects.iterrows()):
        if idx % 100 == 0:
            print(f"  Processing project {idx + 1}/{total}...")

        project_id = project['project_id']
        doc_ids = documents_df[documents_df['project_id'] == project_id]['document_id'].tolist()
        if not doc_ids:
            continue

        project_pages = pages_df[pages_df['document_id'].isin(doc_ids)]
        if project_pages.empty:
            continue

        all_text = "\n\n".join(project_pages['page_text'].dropna().tolist())
        if not all_text.strip():
            continue

        dates_with_context = extract_dates_with_context(
            all_text,
            keep_all_candidates=keep_all_candidates,
            dedupe_by_date=not keep_all_candidates,
        )
        for d in dates_with_context:
            results.append({
                'project_id': project_id,
                'date': d.get('date'),
                'match': d.get('match'),
                'context': d.get('context'),
                'position': d.get('position'),
                'position_pct': d.get('position_pct'),
            })

    results_df = pd.DataFrame(results)
    # Add run timestamp for provenance
    results_df['run_timestamp'] = pd.Timestamp.utcnow().isoformat()

    if output_file:
        output_path = ANALYSIS_DIR / output_file
    else:
        output_path = REGEX_CACHE_PATH

    results_df.to_parquet(output_path)
    print(f"\nSaved regex cache to: {output_path}")
    print(f"Total candidate rows: {len(results_df):,}")

    return results_df


def test_llm_extraction_sample(n_samples: int = 5, model: str = DEFAULT_LLM_MODEL):
    """
    Test LLM extraction on a small sample with detailed output.

    Args:
        n_samples: Number of projects to test
        model: Ollama model name

    Returns:
        DataFrame with detailed results
    """
    import json as json_module
    import time

    print(f"\n=== Testing LLM Extraction ({n_samples} samples) ===")
    print(f"Model: {model}\n")

    # Load data
    projects = pd.read_parquet(ANALYSIS_DIR / "projects_combined.parquet")

    # Filter to CE + Clean energy
    projects = projects[
        (projects['dataset_source'] == 'CE') &
        (projects['project_energy_type'] == 'Clean')
    ].sample(n=n_samples, random_state=42)

    print(f"Selected {len(projects)} CE clean energy projects\n")

    # Load CE data
    pages_df = pd.read_parquet(PROCESSED_DIR / "ce" / "pages.parquet")
    documents_df = pd.read_parquet(PROCESSED_DIR / "ce" / "documents.parquet")

    def extract_id(x):
        return x.get('value', '') if isinstance(x, dict) else x

    documents_df['project_id'] = documents_df['project_id'].apply(extract_id)

    # Filter to main docs
    main_doc_ids = documents_df[documents_df['main_document'] == 'YES']['document_id'].tolist()
    pages_df = pages_df[pages_df['document_id'].isin(main_doc_ids)]

    results = []

    for idx, (_, project) in enumerate(projects.iterrows(), 1):
        project_id = project['project_id']
        project_title = project.get('project_title', 'N/A')[:60]

        print(f"{'='*70}")
        print(f"[{idx}/{n_samples}] Project: {project_id}")
        print(f"Title: {project_title}...")
        print(f"{'='*70}")

        # Get document info
        project_docs = documents_df[documents_df['project_id'] == project_id]
        doc_ids = project_docs['document_id'].tolist()
        project_pages = pages_df[pages_df['document_id'].isin(doc_ids)]

        if project_pages.empty:
            print("  No pages found for this project\n")
            continue

        all_text = "\n\n".join(project_pages['page_text'].dropna().tolist())
        print(f"  Total text: {len(all_text):,} chars")

        # Preprocess
        preprocess_result = preprocess_for_timeline(all_text)
        print(f"  After preprocessing: {len(preprocess_result.processed_text):,} chars ({preprocess_result.reduction_pct:.0f}% reduction)")
        print(f"  Keywords found: {preprocess_result.keywords_found}")

        # Extract with LLM
        print(f"\n  Calling {model}...")
        start = time.time()
        llm_result = extract_with_ollama(all_text, model=model)
        elapsed = time.time() - start
        print(f"  LLM response time: {elapsed:.1f}s")

        # Show results
        print(f"\n  RESULTS:")
        print(f"    Total dates found: {llm_result.get('n_dates_found', 0)}")
        print(f"\n    DECISION:")
        print(f"      Date:       {llm_result.get('decision_date')}")
        print(f"      Confidence: {llm_result.get('decision_confidence')}")
        print(f"      Source:     {str(llm_result.get('decision_date_source', 'N/A'))[:80]}...")

        print(f"\n    SPECIALIST REVIEWS:")
        print(f"      Count:    {llm_result.get('n_specialist_reviews', 0)}")
        print(f"      Earliest: {llm_result.get('earliest_review_date')}")
        print(f"      Latest:   {llm_result.get('latest_review_date')}")

        print(f"\n    APPLICATION:")
        print(f"      Explicit: {llm_result.get('application_date')}")
        print(f"      Inferred: {llm_result.get('inferred_application_date')}")

        print(f"\n    HISTORICAL:")
        print(f"      Count:    {llm_result.get('n_historical_dates', 0)}")
        print(f"      Earliest: {llm_result.get('earliest_historical_date')}")

        print(f"\n    EXPIRATION: {llm_result.get('expiration_date')}")

        if llm_result.get('error'):
            print(f"\n    ERROR: {llm_result.get('error')}")

        # Show all dates
        dates_json = llm_result.get('dates_json', '[]')
        try:
            dates_list = json_module.loads(dates_json)
            if dates_list:
                print(f"\n    ALL DATES EXTRACTED:")
                for d in dates_list:
                    print(f"      {d['date']} [{d['type']}] - {d['source'][:50]}...")
        except Exception:
            pass

        results.append({
            'project_id': project_id,
            'project_title': project_title,
            'total_chars': len(all_text),
            'processed_chars': len(preprocess_result.processed_text),
            'n_dates_found': llm_result.get('n_dates_found', 0),
            'decision_date': llm_result.get('decision_date'),
            'decision_confidence': llm_result.get('decision_confidence'),
            'earliest_review': llm_result.get('earliest_review_date'),
            'n_reviews': llm_result.get('n_specialist_reviews', 0),
            'inferred_app_date': llm_result.get('inferred_application_date'),
            'error': llm_result.get('error'),
            'response_time_sec': elapsed,
        })

        print()

    # Summary table
    results_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(results_df[[
        'project_id', 'n_dates_found', 'decision_date',
        'decision_confidence', 'n_reviews', 'response_time_sec'
    ]].to_string())

    return results_df


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
    print("\nREGEX EXTRACTION:")
    print("  python extract_timeline.py --run")
    print("  python extract_timeline.py --run --sample 100")
    print("  python extract_timeline.py --run --ce-only --clean-energy --main-docs-only")
    print("\nLLM EXTRACTION (requires Ollama running):")
    print("  python extract_timeline.py --llm-test 5          # Test on 5 samples")
    print("  python extract_timeline.py --llm-run --sample 50 # Run on 50 projects")
    print("  python extract_timeline.py --llm-run             # Run on all CE clean energy")

    import argparse
    parser = argparse.ArgumentParser(
        description="Extract timeline dates from NEPA documents using regex or LLM"
    )

    # Regex extraction options
    parser.add_argument('--run', action='store_true', help='Run regex-based extraction')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--clean-energy', action='store_true', help='Process clean energy only')
    parser.add_argument('--ce-only', action='store_true', help='Process CE (Categorical Exclusion) projects only')
    parser.add_argument('--main-docs-only', action='store_true', help='Only read pages from main_document == YES')
    parser.add_argument('--all-docs', action='store_true', help='Use all documents (not just decision docs)')

    # LLM extraction options
    parser.add_argument('--llm-test', type=int, metavar='N',
                        help='Test LLM extraction on N samples with detailed output')
    parser.add_argument('--llm-run', action='store_true',
                        help='Run LLM extraction (CE + clean energy + main docs by default)')
    parser.add_argument('--model', type=str, default=DEFAULT_LLM_MODEL,
                        help=f'Ollama model to use (default: {DEFAULT_LLM_MODEL})')
    parser.add_argument('--output', type=str,
                        help='Custom output filename for LLM results')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Timeout in seconds per document (default: 120, use 300+ for larger models)')
    parser.add_argument('--project-id', type=str,
                        help='Run extraction on a single project by ID')
    parser.add_argument('--hybrid', action='store_true',
                        help='Use hybrid regex+LLM approach (faster, recommended)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel LLM workers (default: 4)')
    parser.add_argument('--regex-prep', action='store_true',
                        help='Precompute regex candidate cache for hybrid LLM')
    parser.add_argument('--regex-filtered', action='store_true',
                        help='Apply cue filtering when building regex cache (legacy behavior)')
    parser.add_argument('--use-regex-cache', action='store_true',
                        help='Use precomputed regex cache for hybrid LLM')
    parser.add_argument('--regex-cache', type=str,
                        help='Custom regex cache filename (data/analysis/...)')

    # BERT classifier options
    parser.add_argument('--bert-generate', action='store_true',
                        help='Generate training data for BERT using weak supervision')
    parser.add_argument('--bert-train', action='store_true',
                        help='Train BERT classifier on auto-labeled data')
    parser.add_argument('--bert-run', action='store_true',
                        help='Run BERT-based timeline extraction (fast)')
    parser.add_argument('--bert-model', type=str, default='distilbert-base-uncased',
                        help='Hugging Face model for BERT (default: distilbert-base-uncased)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs for BERT (default: 3)')

    args = parser.parse_args()

    # Run regex extraction
    if args.run:
        run_timeline_extraction(
            sample_size=args.sample,
            clean_energy_only=args.clean_energy,
            ce_only=args.ce_only,
            decision_docs_only=not args.all_docs,
            main_docs_only=args.main_docs_only
        )

    # Test LLM extraction
    elif args.llm_test:
        test_llm_extraction_sample(n_samples=args.llm_test, model=args.model)

    # Precompute regex candidates
    elif args.regex_prep:
        run_regex_prep(
            sample_size=args.sample,
            clean_energy_only=True,
            ce_only=True,
            main_docs_only=True,
            output_file=args.regex_cache,
            keep_all_candidates=not args.regex_filtered,
        )
        sys.exit(0)

    # Run LLM extraction on single project
    elif args.project_id:
        import json as json_module
        import time

        approach = "hybrid regex+LLM" if args.hybrid else "full-document LLM"
        print(f"\n=== Single Project Extraction ({approach}) ===")
        print(f"Project ID: {args.project_id}")
        print(f"Model: {args.model}")
        print(f"Timeout: {args.timeout}s")

        # Load CE data
        pages_df = pd.read_parquet(PROCESSED_DIR / "ce" / "pages.parquet")
        documents_df = pd.read_parquet(PROCESSED_DIR / "ce" / "documents.parquet")

        # Clean project_id
        documents_df['project_id'] = documents_df['project_id'].apply(
            lambda x: x.get('value', '') if isinstance(x, dict) else x
        )

        # Filter to main docs for this project
        project_docs = documents_df[documents_df['project_id'] == args.project_id]
        if project_docs.empty:
            print(f"ERROR: No documents found for project {args.project_id}")
            sys.exit(1)

        main_docs = project_docs[project_docs['main_document'] == 'YES']
        doc_ids = main_docs['document_id'].tolist() if not main_docs.empty else project_docs['document_id'].tolist()

        print(f"Documents found: {len(project_docs)} (main: {len(main_docs)})")

        # Get page text
        project_pages = pages_df[pages_df['document_id'].isin(doc_ids)]
        all_text = "\n\n".join(project_pages['page_text'].dropna().tolist())
        print(f"Total text: {len(all_text):,} chars")

        # Extract using selected approach
        print(f"\nCalling {args.model}...")
        start = time.time()

        if args.hybrid:
            # Show what regex finds first
            if args.use_regex_cache:
                cache_path = ANALYSIS_DIR / args.regex_cache if args.regex_cache else REGEX_CACHE_PATH
                if cache_path.exists():
                    cache_df = pd.read_parquet(cache_path)
                    dates_found = cache_df[cache_df['project_id'] == args.project_id][[
                        'date', 'match', 'context', 'position', 'position_pct'
                    ]].to_dict(orient='records')
                else:
                    print(f"WARNING: regex cache not found at {cache_path}, falling back to live extraction.")
                    dates_found = extract_dates_with_context(all_text)
            else:
                dates_found = extract_dates_with_context(all_text)
            print(f"Regex found {len(dates_found)} dates")
            for d in dates_found:
                print(f"  {d['date']} ({d['match']}): ...{d['context'][:80]}...")

            result = extract_with_hybrid_approach(
                all_text,
                model=args.model,
                timeout=args.timeout,
                dates_with_context=dates_found,
            )
            print(f"\nPrompt size: {result.get('prompt_chars', 0):,} chars (vs ~5,000 for full-document)")
        else:
            result = extract_with_ollama(all_text, model=args.model, timeout=args.timeout)

        elapsed = time.time() - start
        print(f"Response time: {elapsed:.1f}s")

        # Display results
        print(f"\n{'='*60}")
        print("RESULTS")
        print('='*60)

        if result.get('error'):
            print(f"ERROR: {result['error']}")
        else:
            print(f"Dates found: {result.get('n_dates_found', 0)}")
            print("\nDECISION:")
            print(f"  Date: {result.get('decision_date')}")
            print(f"  Confidence: {result.get('decision_confidence')}")
            print(f"  Source: {str(result.get('decision_date_source', 'N/A'))[:80]}")

            print("\nSPECIALIST REVIEWS:")
            print(f"  Count: {result.get('n_specialist_reviews', 0)}")
            print(f"  Earliest: {result.get('earliest_review_date')}")
            print(f"  Latest: {result.get('latest_review_date')}")

            print("\nAPPLICATION:")
            print(f"  Explicit: {result.get('application_date')}")
            print(f"  Inferred: {result.get('inferred_application_date')}")

            print(f"\nHISTORICAL: {result.get('n_historical_dates', 0)} dates")
            print(f"EXPIRATION: {result.get('expiration_date')}")

            # Show all dates
            dates_json = result.get('dates_json', '[]')
            try:
                dates = json_module.loads(dates_json)
                if dates:
                    print("\nALL DATES CLASSIFIED:")
                    for d in dates:
                        print(f"  {d['date']} [{d['type']}] - {d.get('source', 'N/A')[:60]}...")
            except json_module.JSONDecodeError:
                pass

            print("\nRAW RESPONSE:")
            print(result.get('raw_response', 'N/A')[:500])

    # Run LLM extraction
    elif args.llm_run:
        run_llm_timeline_extraction(
            sample_size=args.sample,
            clean_energy_only=True,  # Default to clean energy
            ce_only=True,            # Default to CE only
            main_docs_only=True,     # Default to main docs only
            model=args.model,
            output_file=args.output,
            use_hybrid=args.hybrid,
            timeout=args.timeout,
            use_regex_cache=args.use_regex_cache,
            regex_cache_path=ANALYSIS_DIR / args.regex_cache if args.regex_cache else REGEX_CACHE_PATH,
            workers=args.workers,
        )

    # Generate BERT training data
    elif args.bert_generate:
        generate_bert_training_data(
            sample_size=args.sample,
            output_path=BERT_TRAINING_DATA_PATH,
        )

    # Train BERT classifier
    elif args.bert_train:
        train_bert_classifier(
            training_data_path=BERT_TRAINING_DATA_PATH,
            model_name=args.bert_model,
            output_dir=BERT_MODEL_DIR,
            epochs=args.epochs,
        )

    # Run BERT extraction
    elif args.bert_run:
        run_bert_timeline_extraction(
            sample_size=args.sample,
            clean_energy_only=True,
            ce_only=True,
            main_docs_only=True,
            output_file=args.output,
            use_regex_cache=True,
        )
