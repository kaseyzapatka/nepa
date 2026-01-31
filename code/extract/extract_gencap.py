# --------------------------
# GENERATION CAPACITY EXTRACTION
# --------------------------
# Extract generation capacity values from document text
# Strategy: Regex first, LLM for gaps (clean energy projects only)

import re
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import GENCAP_UNITS


# --------------------------
# FILE PATHS
# --------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"


# --------------------------
# REGEX PATTERNS FOR GENERATION CAPACITY
# --------------------------

# Pattern components
NUMBER_CORE = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?'
RANGE_PATTERN = rf'({NUMBER_CORE})(?:\s*(?:-|–|—|to)\s*({NUMBER_CORE}))?'
PREFIX_PATTERN = r'(?:about|approx(?:\.|imately)?|approximately|up to|~)\s*'

# Unit patterns (case-insensitive)
POWER_UNIT_PATTERNS = [
    r'MW', r'MWac', r'MWdc', r'MWe', r'MWt', r'MWth', r'MWp',
    r'GW', r'GWe', r'kW', r'kWe', r'kWac', r'kWdc', r'GWac', r'GWdc',
    r'megawatt(?:-?\s*electric)?s?', r'megawatt(?:-?\s*thermal)?s?',
    r'gigawatt(?:-?\s*electric)?s?', r'kilowatt(?:-?\s*electric)?s?'
]

ENERGY_UNIT_PATTERNS = [
    r'MWh', r'GWh', r'kWh',
    r'megawatt-?\s*hours?', r'gigawatt-?\s*hours?', r'kilowatt-?\s*hours?'
]

UNIT_PATTERN = rf'({"|".join(POWER_UNIT_PATTERNS)}|{"|".join(ENERGY_UNIT_PATTERNS)})'

# Combined patterns for capacity extraction
CAPACITY_PATTERNS = [
    # "50 MW", "1.5 GW", "500 kW", "1kWe"
    rf'(?:{PREFIX_PATTERN})?{RANGE_PATTERN}\s*{UNIT_PATTERN}',
    # "capacity of 50 MW", "generating 100 megawatts", "nameplate 200 MW"
    rf'(?:capacity|generating|generate|produces?|output|nameplate|rated|net)\s+(?:of\s+)?(?:{PREFIX_PATTERN})?{RANGE_PATTERN}\s*{UNIT_PATTERN}',
    # "50-MW facility", "100-megawatt project", "A 100-megawatt solar facility"
    rf'(?:{PREFIX_PATTERN})?{RANGE_PATTERN}\s*-?\s*{UNIT_PATTERN}\s+(?:facility|project|plant|farm|array|solar|wind|power|battery|storage)',
    # "a 100-megawatt facility" (article + number-unit + noun)
    rf'[Aa]\s+(?:{PREFIX_PATTERN})?{RANGE_PATTERN}\s*-?\s*{UNIT_PATTERN}\s+\w+',
]

CONTEXT_WORDS = {
    'project', 'proposed', 'facility', 'plant', 'farm', 'array', 'system',
    'nameplate', 'rated', 'net', 'capacity', 'would', 'will', 'generate',
}

HISTORICAL_WORDS = {
    'existing', 'previous', 'previously', 'former', 'historical', 'nearby',
    'adjacent', 'another', 'other', 'currently', 'prior', 'legacy',
}

AMBIGUOUS_WORDS = {
    'similar', 'comparable', 'reference', 'example',
}


# --------------------------
# HELPERS
# --------------------------

def normalize_unit(unit_str):
    """Normalize unit string to standard form."""
    unit_lower = re.sub(r'\s+', ' ', unit_str.lower().strip())
    return GENCAP_UNITS.get(unit_lower, unit_str)


def classify_unit(unit_str):
    """Classify unit as power or energy."""
    if unit_str in {'GW', 'MW', 'kW'}:
        return 'power'
    if unit_str in {'GWh', 'MWh', 'kWh'}:
        return 'energy'
    return None


def parse_number(num_str):
    """Parse number string (handles commas)."""
    try:
        return float(num_str.replace(',', ''))
    except (ValueError, AttributeError):
        return None


def is_invalid_match(match_text):
    """Filter out non-capacity usages like MW-year or $/MW."""
    if not match_text:
        return False
    lower = match_text.lower()
    invalid_tokens = [
        'mw-year', 'mw yr', 'mw/yr', 'mwy',
        '$/mw', '$ /mw', 'per mw', 'mw per',
    ]
    return any(tok in lower for tok in invalid_tokens)


def is_initials_date_context(context_text: str) -> bool:
    """Detect initials/date lists that can trigger false MW matches."""
    if not context_text:
        return False
    text = context_text.lower()
    # Common patterns like "MW, 5/21/15" or "initials/date"
    initials_date = re.compile(r'\b[a-z]{1,3}\b,?\s*\d{1,2}/\d{1,2}/\d{2,4}')
    date_near_unit = re.compile(r'\b(?:mw|kw|gw)\b[^\\n]{0,30}\b\\d{1,2}/\\d{1,2}/\\d{2,4}')
    if 'initials/date' in text:
        return True
    return bool(initials_date.search(text) or date_near_unit.search(text))


def score_confidence(context):
    """Score confidence based on local context."""
    if not context:
        return 'low'
    text = context.lower()
    score = 0
    if any(w in text for w in CONTEXT_WORDS):
        score += 2
    if any(w in text for w in HISTORICAL_WORDS):
        score -= 2
    if any(w in text for w in AMBIGUOUS_WORDS):
        score -= 1
    if score >= 2:
        return 'high'
    if score >= 0:
        return 'medium'
    return 'low'


# --------------------------
# EXTRACTION
# --------------------------

def extract_capacity_from_text(text, source='document'):
    """
    Extract generation capacity from a text string.

    Args:
        text: String containing document text

    Returns:
        list of dicts: [{'value': float, 'unit': str, 'match': str, 'unit_type': str,
                         'context': str, 'confidence': str}, ...]
    """
    if not text or not isinstance(text, str):
        return []

    results = []
    seen_matches = set()

    for pattern in CAPACITY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            full_match = match.group(0)
            if full_match in seen_matches:
                continue
            seen_matches.add(full_match)

            groups = match.groups()
            if len(groups) < 3:
                continue

            num_str = groups[0]
            num_str_2 = groups[1]
            unit_str = groups[2]

            if is_invalid_match(full_match):
                continue

            value_1 = parse_number(num_str)
            value_2 = parse_number(num_str_2) if num_str_2 else None
            if value_1 is None and value_2 is None:
                continue

            value = max(v for v in [value_1, value_2] if v is not None)
            unit = normalize_unit(unit_str)
            unit_type = classify_unit(unit)
            if not unit_type:
                continue

                context_start = max(0, match.start() - 80)
                context_end = min(len(text), match.end() + 80)
                context = text[context_start:context_end].replace('\n', ' ')
                if is_initials_date_context(context):
                    continue
            confidence = 'high' if source == 'title' else score_confidence(context)

            results.append({
                'value': value,
                'unit': unit,
                'unit_type': unit_type,
                'match': full_match,
                'context': context,
                'confidence': confidence,
            })

    return results


def get_primary_capacity(capacities, unit_type):
    """
    Select the primary capacity from a list of extracted capacities.

    Prefers GW over MW over kW for power; GWh over MWh over kWh for energy.

    Args:
        capacities: list of capacity dicts

    Returns:
        dict or None: capacity dict
    """
    if not capacities:
        return None

    filtered = [c for c in capacities if c['unit_type'] == unit_type]
    if not filtered:
        return None

    unit_priority = {'GW': 3, 'MW': 2, 'kW': 1} if unit_type == 'power' else {'GWh': 3, 'MWh': 2, 'kWh': 1}

    best = None
    best_priority = -1

    for cap in filtered:
        priority = unit_priority.get(cap['unit'], 0)
        if priority > best_priority:
            best = cap
            best_priority = priority
        elif priority == best_priority and best and cap['value'] > best['value']:
            best = cap

    return best


def extract_project_capacity(project_id, project_title, project_type, pages_df, documents_df):
    """
    Extract generation capacity for a single project.

    Processes main documents first, then other documents if needed.

    Args:
        project_id: Project ID string
        project_title: Project title string
        project_type: Project type string/list
        pages_df: DataFrame with page text
        documents_df: DataFrame with document metadata

    Returns:
        dict with power/energy values and metadata
    """
    # Title-first extraction
    title_caps = extract_capacity_from_text(project_title, source='title')
    title_power = get_primary_capacity(title_caps, unit_type='power')
    title_energy = get_primary_capacity(title_caps, unit_type='energy')
    if title_power or title_energy:
        primary = title_power or title_energy
        return {
            'project_gencap_value': title_power['value'] if title_power else None,
            'project_gencap_unit': title_power['unit'] if title_power else None,
            'project_gencap_energy_value': title_energy['value'] if title_energy else None,
            'project_gencap_energy_unit': title_energy['unit'] if title_energy else None,
            'project_gencap_matches': [c['match'] for c in title_caps if c['unit_type'] == 'power'][:5],
            'project_gencap_energy_matches': [c['match'] for c in title_caps if c['unit_type'] == 'energy'][:5],
            'project_gencap_source': 'title',
            'project_gencap_confidence': 'high',
            'project_gencap_context': primary['context'] if primary else None,
        }

    # Get documents for this project
    project_docs = documents_df[documents_df['project_id'] == project_id]

    if project_docs.empty:
        return {
            'project_gencap_value': None,
            'project_gencap_unit': None,
            'project_gencap_energy_value': None,
            'project_gencap_energy_unit': None,
            'project_gencap_matches': [],
            'project_gencap_energy_matches': [],
            'project_gencap_source': 'no_documents',
            'project_gencap_confidence': 'low',
            'project_gencap_context': None,
        }

    # Process main documents first
    main_docs = project_docs[project_docs['main_document'] == 'YES']
    other_docs = project_docs[project_docs['main_document'] != 'YES']

    all_capacities = []

    for docs in [main_docs, other_docs]:
        if not docs.empty:
            doc_ids = docs['document_id'].tolist()
            project_pages = pages_df[pages_df['document_id'].isin(doc_ids)]

            for _, page in project_pages.iterrows():
                capacities = extract_capacity_from_text(page.get('page_text', ''), source='document')
                all_capacities.extend(capacities)

            if all_capacities and docs is main_docs:
                break

    primary_power = get_primary_capacity(all_capacities, unit_type='power')
    primary_energy = get_primary_capacity(all_capacities, unit_type='energy')
    primary = primary_power or primary_energy

    return {
        'project_gencap_value': primary_power['value'] if primary_power else None,
        'project_gencap_unit': primary_power['unit'] if primary_power else None,
        'project_gencap_energy_value': primary_energy['value'] if primary_energy else None,
        'project_gencap_energy_unit': primary_energy['unit'] if primary_energy else None,
        'project_gencap_matches': [c['match'] for c in all_capacities if c['unit_type'] == 'power'][:5],
        'project_gencap_energy_matches': [c['match'] for c in all_capacities if c['unit_type'] == 'energy'][:5],
        'project_gencap_source': 'document' if primary else 'none',
        'project_gencap_confidence': primary['confidence'] if primary else 'low',
        'project_gencap_context': primary['context'] if primary else None,
    }


# --------------------------
# RUNNER
# --------------------------

def run_capacity_extraction(clean_energy_only=True, sample_size=None, source=None, output_path=None, parallel_workers=0):
    """
    Run generation capacity extraction for projects.

    Args:
        clean_energy_only: If True, only process clean energy projects
        sample_size: If set, only process this many projects (for testing)

    Outputs:
        data/analysis/projects_gencap.parquet
    """
    print("\n=== Generation Capacity Extraction ===")

    projects_path = ANALYSIS_DIR / "projects_combined.parquet"
    if not projects_path.exists():
        print(f"Error: {projects_path} not found. Run extract_data.py first.")
        return

    projects = pd.read_parquet(projects_path)
    print(f"Loaded {len(projects):,} projects")

    if clean_energy_only:
        projects = projects[projects['project_energy_type'] == 'Clean']
        print(f"Filtered to {len(projects):,} clean energy projects")

    if sample_size:
        projects = projects.head(sample_size)
        print(f"Sampling {len(projects):,} projects")

    if source is None and parallel_workers and parallel_workers > 1:
        return _run_parallel_sources(projects, clean_energy_only, sample_size, parallel_workers, output_path)

    if source:
        projects = projects[projects['dataset_source'] == source]

    results = []

    sources = [source] if source else list(projects['dataset_source'].unique())
    for src in sources:
        print(f"\nProcessing {src} projects...")

        source_projects = projects if source else projects[projects['dataset_source'] == src]
        data_dir = PROCESSED_DIR / src.lower()

        pages_df = pd.read_parquet(data_dir / "pages.parquet")
        documents_df = pd.read_parquet(data_dir / "documents.parquet")

        def extract_id(x):
            if isinstance(x, dict):
                return x.get('value', '')
            return x

        documents_df['project_id'] = documents_df['project_id'].apply(extract_id)

        for idx, (_, project) in enumerate(source_projects.iterrows()):
            if idx % 100 == 0:
                print(f"  Processing project {idx + 1}/{len(source_projects)}...")

            project_id = project['project_id']
            capacity = extract_project_capacity(
                project_id=project_id,
                project_title=project.get('project_title', ''),
                project_type=project.get('project_type', ''),
                pages_df=pages_df,
                documents_df=documents_df,
            )

            results.append({
                'project_id': project_id,
                'dataset_source': src,
                **capacity
            })

    results_df = pd.DataFrame(results)

    projects_with_cap = projects.merge(results_df, on=['project_id', 'dataset_source'], how='left')

    if output_path:
        save_path = Path(output_path)
    else:
        if source:
            save_path = ANALYSIS_DIR / f"projects_gencap_{source.lower()}.parquet"
        else:
            save_path = ANALYSIS_DIR / "projects_gencap.parquet"

    projects_with_cap.to_parquet(save_path)
    print(f"\nSaved to: {save_path}")

    has_cap = projects_with_cap['project_gencap_value'].notna()
    title_hits = (projects_with_cap['project_gencap_source'] == 'title').sum()
    doc_hits = (projects_with_cap['project_gencap_source'] == 'document').sum()
    print(f"\nProjects with capacity extracted (power): {has_cap.sum():,} ({has_cap.mean() * 100:.1f}%)")
    print(f"  Title matches: {title_hits:,}")
    print(f"  Document matches: {doc_hits:,}")

    if 'project_gencap_confidence' in projects_with_cap.columns:
        conf_counts = projects_with_cap['project_gencap_confidence'].value_counts(dropna=False).to_dict()
        print(f"  Confidence counts: {conf_counts}")

    return projects_with_cap


def _parallel_worker(args):
    """Worker for parallel source extraction."""
    src, clean_energy_only, sample_size = args
    tmp_path = ANALYSIS_DIR / f"projects_gencap_{src.lower()}_tmp.parquet"
    run_capacity_extraction(
        clean_energy_only=clean_energy_only,
        sample_size=sample_size,
        source=src,
        output_path=str(tmp_path),
        parallel_workers=0,
    )
    return str(tmp_path)


def _run_parallel_sources(projects, clean_energy_only, sample_size, parallel_workers, output_path):
    """Run per-source extraction in parallel and combine outputs."""
    from multiprocessing import get_context

    sources = list(projects['dataset_source'].unique())
    if not sources:
        print("No sources found to process.")
        return None

    tmp_paths = []
    ctx = get_context("spawn")

    with ctx.Pool(processes=min(parallel_workers, len(sources))) as pool:
        for p in pool.map(_parallel_worker, [(s, clean_energy_only, sample_size) for s in sources]):
            tmp_paths.append(Path(p))

    parts = [pd.read_parquet(p) for p in tmp_paths if p.exists()]
    if not parts:
        print("No outputs created in parallel run.")
        return None

    combined = pd.concat(parts, ignore_index=True)
    if output_path:
        save_path = Path(output_path)
    else:
        save_path = ANALYSIS_DIR / "projects_gencap.parquet"
    combined.to_parquet(save_path)
    print(f"\nSaved combined output to: {save_path}")

    # Clean up temp files
    for p in tmp_paths:
        if p.exists():
            p.unlink()

    return combined


# --------------------------
# TESTING
# --------------------------

if __name__ == "__main__":
    test_texts = [
        "The project will generate 50 MW of electricity.",
        "A 100-megawatt solar facility",
        "capacity of 1,500 MW",
        "The wind farm produces 2.5 GW annually.",
        "Storage capacity of 500 MWh",
        "Project 1kWe demonstration unit",
        "No capacity mentioned here.",
    ]

    print("Testing capacity extraction patterns...")
    for text in test_texts:
        results = extract_capacity_from_text(text)
        print(f"\nText: {text}")
        print(f"  Found: {results}")

    print("\n" + "=" * 50)
    print("\nTo run full extraction, use:")
    print("  python extract_gencap.py --run")
    print("\nFor a test sample:")
    print("  python extract_gencap.py --run --sample 100")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Run full extraction')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--all', action='store_true', help='Process all projects, not just clean energy')
    parser.add_argument('--source', choices=['ce', 'ea', 'eis'], help='Process a single dataset source')
    parser.add_argument('--output', type=str, help='Output file path (parquet)')
    parser.add_argument('--parallel', type=int, default=0, help='Run CE/EA/EIS in parallel with N workers')

    args = parser.parse_args()

    if args.run:
        run_capacity_extraction(
            clean_energy_only=not args.all,
            sample_size=args.sample,
            source=args.source.upper() if args.source else None,
            output_path=args.output,
            parallel_workers=args.parallel,
        )
