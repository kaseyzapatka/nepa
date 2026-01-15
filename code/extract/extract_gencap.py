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
NUMBER_PATTERN = r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)'

# Unit patterns (case-insensitive)
UNIT_PATTERNS = [
    r'MW',
    r'megawatts?',
    r'GW',
    r'gigawatts?',
    r'kW',
    r'kilowatts?',
    r'MWh',
    r'kWh',
    r'GWh',
]

# Combined patterns for capacity extraction
CAPACITY_PATTERNS = [
    # "50 MW", "1.5 GW", "500 kW"
    rf'{NUMBER_PATTERN}\s*({"|".join(UNIT_PATTERNS)})',
    # "capacity of 50 MW", "generating 100 megawatts"
    rf'(?:capacity|generating|generate|produces?|output)\s+(?:of\s+)?{NUMBER_PATTERN}\s*({"|".join(UNIT_PATTERNS)})',
    # "50-MW facility", "100-megawatt project", "A 100-megawatt solar facility"
    rf'{NUMBER_PATTERN}\s*-?\s*({"|".join(UNIT_PATTERNS)})\s+(?:facility|project|plant|farm|array|solar|wind|power)',
    # "a 100-megawatt facility" (article + number-unit + noun)
    rf'[Aa]\s+{NUMBER_PATTERN}\s*-?\s*({"|".join(UNIT_PATTERNS)})\s+\w+',
]


def normalize_unit(unit_str):
    """Normalize unit string to standard form."""
    unit_lower = unit_str.lower().strip()
    return GENCAP_UNITS.get(unit_lower, unit_str)


def parse_number(num_str):
    """Parse number string (handles commas)."""
    try:
        return float(num_str.replace(',', ''))
    except (ValueError, AttributeError):
        return None


def extract_capacity_from_text(text):
    """
    Extract generation capacity from a text string.

    Args:
        text: String containing document text

    Returns:
        list of dicts: [{'value': float, 'unit': str, 'match': str}, ...]
    """
    if not text or not isinstance(text, str):
        return []

    results = []
    seen_matches = set()

    for pattern in CAPACITY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Get the full match for deduplication
            full_match = match.group(0)
            if full_match in seen_matches:
                continue
            seen_matches.add(full_match)

            # Extract number and unit
            groups = match.groups()
            if len(groups) >= 2:
                num_str = groups[0]
                unit_str = groups[1]

                value = parse_number(num_str)
                unit = normalize_unit(unit_str)

                if value is not None:
                    results.append({
                        'value': value,
                        'unit': unit,
                        'match': full_match
                    })

    return results


def get_primary_capacity(capacities):
    """
    Select the primary capacity from a list of extracted capacities.

    Prefers MW over kW, takes largest value of same unit type.

    Args:
        capacities: list of capacity dicts

    Returns:
        dict or None: {'value': float, 'unit': str}
    """
    if not capacities:
        return None

    # Group by base unit (MW preferred over kW)
    unit_priority = {'GW': 3, 'MW': 2, 'kW': 1, 'GWh': 3, 'MWh': 2, 'kWh': 1}

    best = None
    best_priority = -1

    for cap in capacities:
        priority = unit_priority.get(cap['unit'], 0)
        if priority > best_priority:
            best = cap
            best_priority = priority
        elif priority == best_priority and best and cap['value'] > best['value']:
            best = cap

    if best:
        return {'value': best['value'], 'unit': best['unit']}
    return None


def extract_project_capacity(project_id, pages_df, documents_df):
    """
    Extract generation capacity for a single project.

    Processes main documents first, then other documents if needed.

    Args:
        project_id: Project ID string
        pages_df: DataFrame with page text
        documents_df: DataFrame with document metadata

    Returns:
        dict: {
            'project_gencap_value': float or None,
            'project_gencap_unit': str or None,
            'project_gencap_matches': list of match strings
        }
    """
    # Get documents for this project
    project_docs = documents_df[documents_df['project_id'] == project_id]

    if project_docs.empty:
        return {
            'project_gencap_value': None,
            'project_gencap_unit': None,
            'project_gencap_matches': []
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
                capacities = extract_capacity_from_text(page.get('page_text', ''))
                all_capacities.extend(capacities)

            # If we found capacity in main docs, stop
            if all_capacities and docs is main_docs:
                break

    # Get primary capacity
    primary = get_primary_capacity(all_capacities)

    return {
        'project_gencap_value': primary['value'] if primary else None,
        'project_gencap_unit': primary['unit'] if primary else None,
        'project_gencap_matches': [c['match'] for c in all_capacities[:5]]  # Keep first 5 matches
    }


def run_capacity_extraction(clean_energy_only=True, sample_size=None):
    """
    Run generation capacity extraction for projects.

    Args:
        clean_energy_only: If True, only process clean energy projects
        sample_size: If set, only process this many projects (for testing)

    Outputs:
        data/analysis/projects_gencap.parquet
    """
    print("\n=== Generation Capacity Extraction ===")

    # Load projects
    projects_path = ANALYSIS_DIR / "projects_combined.parquet"
    if not projects_path.exists():
        print(f"Error: {projects_path} not found. Run extract_data.py first.")
        return

    projects = pd.read_parquet(projects_path)
    print(f"Loaded {len(projects):,} projects")

    # Filter to clean energy if requested
    if clean_energy_only:
        projects = projects[projects['project_energy_type'] == 'Clean']
        print(f"Filtered to {len(projects):,} clean energy projects")

    # Sample if requested
    if sample_size:
        projects = projects.head(sample_size)
        print(f"Sampling {len(projects):,} projects")

    # Load pages and documents for each dataset source
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
            capacity = extract_project_capacity(project_id, pages_df, documents_df)

            results.append({
                'project_id': project_id,
                **capacity
            })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Merge back to projects
    projects_with_cap = projects.merge(results_df, on='project_id', how='left')

    # Save
    output_path = ANALYSIS_DIR / "projects_gencap.parquet"
    projects_with_cap.to_parquet(output_path)
    print(f"\nSaved to: {output_path}")

    # Summary
    has_cap = projects_with_cap['project_gencap_value'].notna()
    print(f"\nProjects with capacity extracted: {has_cap.sum():,} ({has_cap.mean() * 100:.1f}%)")

    return projects_with_cap


# --------------------------
# TESTING
# --------------------------

if __name__ == "__main__":
    # Test regex extraction
    test_texts = [
        "The project will generate 50 MW of electricity.",
        "A 100-megawatt solar facility",
        "capacity of 1,500 MW",
        "The wind farm produces 2.5 GW annually.",
        "Storage capacity of 500 MWh",
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

    args = parser.parse_args()

    if args.run:
        run_capacity_extraction(
            clean_energy_only=not args.all,
            sample_size=args.sample
        )
