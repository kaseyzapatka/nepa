# --------------------------
# TIMELINE EXTRACTION
# --------------------------
# Extract dates from document text to construct project timelines
# Strategy: Regex first to find all dates, then order them chronologically

import re
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
# LLM EXTRACTION (with preprocessing)
# --------------------------

# Import preprocessing module
from preprocess_documents import preprocess_for_timeline

# Default Ollama model - llama3.2 is fast and accurate for structured extraction
DEFAULT_LLM_MODEL = "llama3.2:latest"


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
            review_dates = [d for d in valid_dates if d['type'] == 'specialist_review']
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

        # Specialist review fields
        'llm_earliest_review_date': None,
        'llm_latest_review_date': None,
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
        'llm_error': None,
        'llm_processed_chars': 0,
        'llm_reduction_pct': 0,
        'llm_raw_response': None,
    }

    if project_docs.empty:
        result['llm_error'] = 'no_documents'
        return result

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

    # Extract with LLM
    llm_result = extract_with_ollama(all_text, model=model)

    # Merge all results
    result['llm_dates_json'] = llm_result.get('dates_json', '[]')
    result['llm_n_dates_found'] = llm_result.get('n_dates_found', 0)

    result['llm_decision_date'] = llm_result.get('decision_date')
    result['llm_decision_date_source'] = llm_result.get('decision_date_source')
    result['llm_decision_confidence'] = llm_result.get('decision_confidence')

    result['llm_earliest_review_date'] = llm_result.get('earliest_review_date')
    result['llm_latest_review_date'] = llm_result.get('latest_review_date')
    result['llm_n_specialist_reviews'] = llm_result.get('n_specialist_reviews', 0)

    result['llm_application_date'] = llm_result.get('application_date')
    result['llm_inferred_application_date'] = llm_result.get('inferred_application_date')

    result['llm_earliest_historical_date'] = llm_result.get('earliest_historical_date')
    result['llm_n_historical_dates'] = llm_result.get('n_historical_dates', 0)

    result['llm_expiration_date'] = llm_result.get('expiration_date')

    result['llm_error'] = llm_result.get('error')
    result['llm_processed_chars'] = llm_result.get('processed_chars', 0)
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

    Returns:
        DataFrame with results
    """
    print("\n=== LLM Timeline Extraction ===")
    print(f"Model: {model}")

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

    # Process each project
    results = []
    total = len(projects)

    print(f"\nProcessing {total} projects with LLM...")
    print("(This may take a while - ~2-5 seconds per project)\n")

    import time
    start_time = time.time()

    for idx, (_, project) in enumerate(projects.iterrows()):
        project_id = project['project_id']

        # Progress update
        if idx > 0 and idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (total - idx) / rate if rate > 0 else 0
            print(f"  [{idx}/{total}] {rate:.1f} projects/sec, ~{remaining/60:.1f} min remaining")

        timeline = build_project_timeline_llm(
            project_id, pages_df, documents_df, model=model
        )
        results.append(timeline)

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

    # Run LLM extraction on single project
    elif args.project_id:
        import json as json_module
        import time

        print(f"\n=== Single Project LLM Extraction ===")
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

        # Extract with custom timeout
        print(f"\nCalling {args.model}...")
        start = time.time()
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
                    print("\nALL DATES EXTRACTED:")
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
        )
