# --------------------------
# GENERATION CAPACITY EXTRACTION
# --------------------------
# Extract generation capacity values from document text.
# Strategy: Regex first (title → description → document pages).
#           LLM adjudication for ambiguous multi-candidate cases.
#
# Usage:
#   python extract_gencap.py --run regex               # regex extraction on all sources
#   python extract_gencap.py --run regex --parallel 3  # regex in parallel
#   python extract_gencap.py --run llm --workers 4     # LLM adjudication + merge
#   python extract_gencap.py --run llm --sample 10     # LLM test sample
#   python extract_gencap.py --self-test               # test regex patterns

import re
import json
import ast
import time
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Match longer/more-specific units first (prevents MW matching inside MWh/MWac).
UNIT_PATTERNS = sorted(POWER_UNIT_PATTERNS + ENERGY_UNIT_PATTERNS, key=len, reverse=True)
UNIT_PATTERN = rf'({"|".join(UNIT_PATTERNS)})'

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
# LLM CONFIGURATION
# --------------------------

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
POWER_UNITS = {"GW", "MW", "kW"}


# --------------------------
# SHARED HELPERS (REGEX)
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


def value_to_text(value):
    """Convert list/JSON/scalar values to a normalized plain-text string."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(v) for v in value if str(v).strip())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, (list, tuple)):
                        return " ".join(str(v) for v in parsed if str(v).strip())
                except Exception:
                    pass
        return text
    return str(value)


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
    date_near_unit = re.compile(r'\b(?:mw|kw|gw)\b[^\n]{0,30}\b\d{1,2}/\d{1,2}/\d{2,4}')
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


def value_to_list(value) -> list:
    """Convert list/JSON/scalar values to a normalized list of strings."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, (list, tuple)):
                        return [str(v) for v in parsed if str(v).strip()]
                except Exception:
                    pass
        return [text]
    return [str(value)]


def normalize_power_unit(unit):
    """Normalize common power unit variants to GW/MW/kW."""
    if unit is None or (isinstance(unit, float) and pd.isna(unit)):
        return None
    text = str(unit).strip().lower().replace(" ", "")
    mapping = {
        "gw": "GW", "gwe": "GW", "gwac": "GW", "gwdc": "GW",
        "gigawatt": "GW", "gigawatts": "GW",
        "mw": "MW", "mwe": "MW", "mwt": "MW", "mwth": "MW",
        "mwac": "MW", "mwdc": "MW", "mwp": "MW",
        "megawatt": "MW", "megawatts": "MW",
        "kw": "kW", "kwe": "kW", "kwac": "kW", "kwdc": "kW",
        "kilowatt": "kW", "kilowatts": "kW",
    }
    return mapping.get(text, str(unit).strip())


def _extract_power_pairs_from_matches(match_value) -> set:
    """Extract (value, unit) pairs from regex match text for power units only."""
    pattern = re.compile(
        r'(\d[\d,\.]*)\s*(?:-|–|—)?\s*'
        r'(MWac|MWdc|MWe|MWt|MWth|MWp|GWac|GWdc|kWe|kWac|kWdc|MW|GW|kW|'
        r'megawatts?|gigawatts?|kilowatts?)',
        re.IGNORECASE,
    )
    pairs = set()
    for text in value_to_list(match_value):
        for m in pattern.finditer(str(text)):
            value_str, unit_str = m.group(1), m.group(2)
            try:
                value = float(value_str.replace(",", ""))
            except ValueError:
                continue
            unit = normalize_power_unit(unit_str)
            if unit in POWER_UNITS:
                pairs.add((round(value, 6), unit))
    return pairs


def _llm_selection_in_regex_matches(llm_value, llm_unit, regex_matches) -> bool:
    """Check if LLM-selected value/unit appears in regex candidate match list."""
    try:
        value = float(llm_value)
    except (TypeError, ValueError):
        return False
    unit = normalize_power_unit(llm_unit)
    if unit not in POWER_UNITS:
        return False
    pairs = _extract_power_pairs_from_matches(regex_matches)
    return (round(value, 6), unit) in pairs


def is_non_power_project(project_type: str) -> bool:
    """Check if project uses non-power metrics (e.g., pipelines use volume).

    Only returns True if the project is PURELY a pipeline project,
    not if it includes pipeline along with power generation types.
    """
    if not project_type:
        return False
    pt_lower = project_type.lower()
    power_keywords = ['solar', 'wind', 'nuclear', 'geothermal', 'hydropower',
                      'hydrokinetic', 'biomass', 'energy production', 'energy storage']
    has_power_type = any(kw in pt_lower for kw in power_keywords)
    if has_power_type:
        return False
    return 'pipeline' in pt_lower and 'solar' not in pt_lower and 'wind' not in pt_lower


# --------------------------
# PAGE LOADING (DUCKDB)
# --------------------------

def load_project_pages_with_duckdb(
    pages_path: Path,
    document_pairs: pd.DataFrame,
    max_pages: Optional[int],
) -> dict:
    """
    Load top-N ordered pages per project using DuckDB for fast bulk retrieval.

    Args:
        pages_path: Path to pages.parquet
        document_pairs: DataFrame with ['project_id', 'document_id'] and optional 'doc_rank'
        max_pages: Max pages to keep per project after ordering.
            If None, returns all pages for each project.

    Returns:
        dict: {project_id: [page_text, ...]}
    """
    if document_pairs is None or document_pairs.empty:
        return {}

    pairs = document_pairs.copy()
    required = {"project_id", "document_id"}
    if not required.issubset(pairs.columns):
        raise ValueError("document_pairs must include project_id and document_id")

    if "doc_rank" not in pairs.columns:
        pairs["doc_rank"] = 0

    pairs = pairs[["project_id", "document_id", "doc_rank"]].drop_duplicates()
    if pairs.empty:
        return {}

    pages_path_sql = pages_path.as_posix().replace("'", "''")

    con = duckdb.connect()
    try:
        con.register("project_docs", pairs)
        where_clause = ""
        if max_pages is not None:
            where_clause = f"WHERE rn <= {int(max_pages)}"

        query = f"""
        WITH joined AS (
            SELECT
                d.project_id,
                d.doc_rank,
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
                    ORDER BY doc_rank, page_num, page_number
                ) AS rn
            FROM joined
        )
        SELECT project_id, page_text
        FROM ranked
        {where_clause}
        ORDER BY project_id, rn
        """
        pages_df = con.execute(query).df()
    finally:
        con.close()

    if pages_df.empty:
        return {}

    lookup = {}
    for project_id, group in pages_df.groupby("project_id", sort=False):
        lookup[project_id] = [
            text if isinstance(text, str) else ""
            for text in group["page_text"].tolist()
        ]
    return lookup


def build_regex_document_pairs(documents_df: pd.DataFrame, project_ids: set) -> pd.DataFrame:
    """Build document pairs for regex page loading (main docs first)."""
    if not project_ids:
        return pd.DataFrame(columns=["project_id", "document_id", "doc_rank"])

    docs = documents_df[documents_df["project_id"].isin(project_ids)].copy()
    if docs.empty:
        return pd.DataFrame(columns=["project_id", "document_id", "doc_rank"])

    main_series = docs["main_document"] if "main_document" in docs.columns else pd.Series("", index=docs.index)
    docs["_main_priority"] = np.where(main_series.fillna("").astype(str).str.upper() == "YES", 0, 1)
    docs = docs.sort_values(["project_id", "_main_priority", "document_id"], kind="stable")
    docs = docs.drop_duplicates(subset=["project_id", "document_id"], keep="first")
    docs["doc_rank"] = docs.groupby("project_id").cumcount()
    return docs[["project_id", "document_id", "doc_rank"]]



# --------------------------
# REGEX EXTRACTION
# --------------------------

def extract_capacity_from_text(text, source='document'):
    """
    Extract generation capacity from a text string.

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
            if source != 'title' and is_initials_date_context(context):
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


def count_distinct_capacities(capacities, unit_type):
    """Count distinct value+unit candidates for a given unit type."""
    if not capacities:
        return 0
    values = set()
    for cap in capacities:
        if cap.get('unit_type') != unit_type:
            continue
        value = cap.get('value')
        unit = cap.get('unit')
        if value is None or unit is None:
            continue
        values.add((round(float(value), 6), unit))
    return len(values)


def _empty_capacity_result(source: str = "none") -> dict:
    """Build an empty capacity result with a specific source label."""
    return {
        "project_gencap_value": None,
        "project_gencap_unit": None,
        "project_gencap_energy_value": None,
        "project_gencap_energy_unit": None,
        "project_gencap_matches": [],
        "project_gencap_energy_matches": [],
        "project_gencap_candidate_count": 0,
        "project_gencap_energy_candidate_count": 0,
        "project_gencap_source": source,
        "project_gencap_confidence": "low",
        "project_gencap_context": None,
        "project_gencap_candidates_json": [],
    }


def extract_project_capacity_title_description(project_title, project_description):
    """
    Extract capacity from title then description only.

    Returns:
        dict with extraction columns, or None if neither field has a match.
    """
    title_caps = extract_capacity_from_text(project_title, source="title")
    title_power = get_primary_capacity(title_caps, unit_type="power")
    title_energy = get_primary_capacity(title_caps, unit_type="energy")
    if title_power or title_energy:
        primary = title_power or title_energy
        return {
            "project_gencap_value": title_power["value"] if title_power else None,
            "project_gencap_unit": title_power["unit"] if title_power else None,
            "project_gencap_energy_value": title_energy["value"] if title_energy else None,
            "project_gencap_energy_unit": title_energy["unit"] if title_energy else None,
            "project_gencap_matches": [c["match"] for c in title_caps if c["unit_type"] == "power"][:5],
            "project_gencap_energy_matches": [c["match"] for c in title_caps if c["unit_type"] == "energy"][:5],
            "project_gencap_candidate_count": count_distinct_capacities(title_caps, unit_type="power"),
            "project_gencap_energy_candidate_count": count_distinct_capacities(title_caps, unit_type="energy"),
            "project_gencap_source": "title",
            "project_gencap_confidence": "high",
            "project_gencap_context": primary["context"] if primary else None,
            "project_gencap_candidates_json": [
                {"id": i + 1, "value": c["value"], "unit": c["unit"], "context": c["context"]}
                for i, c in enumerate(
                    [x for x in title_caps if x["unit_type"] == "power" and x.get("context")][:5]
                )
            ],
        }

    description_text = value_to_text(project_description)
    desc_caps = extract_capacity_from_text(description_text, source="description")
    desc_power = get_primary_capacity(desc_caps, unit_type="power")
    desc_energy = get_primary_capacity(desc_caps, unit_type="energy")
    if desc_power or desc_energy:
        primary = desc_power or desc_energy
        return {
            "project_gencap_value": desc_power["value"] if desc_power else None,
            "project_gencap_unit": desc_power["unit"] if desc_power else None,
            "project_gencap_energy_value": desc_energy["value"] if desc_energy else None,
            "project_gencap_energy_unit": desc_energy["unit"] if desc_energy else None,
            "project_gencap_matches": [c["match"] for c in desc_caps if c["unit_type"] == "power"][:5],
            "project_gencap_energy_matches": [c["match"] for c in desc_caps if c["unit_type"] == "energy"][:5],
            "project_gencap_candidate_count": count_distinct_capacities(desc_caps, unit_type="power"),
            "project_gencap_energy_candidate_count": count_distinct_capacities(desc_caps, unit_type="energy"),
            "project_gencap_source": "description",
            "project_gencap_confidence": "high",
            "project_gencap_context": primary["context"] if primary else None,
            "project_gencap_candidates_json": [
                {"id": i + 1, "value": c["value"], "unit": c["unit"], "context": c["context"]}
                for i, c in enumerate(
                    [x for x in desc_caps if x["unit_type"] == "power" and x.get("context")][:5]
                )
            ],
        }

    return None


def extract_project_capacity_from_pages(
    project_id,
    project_title,
    project_description,
    project_type,
    pages_lookup,
    documents_df,
):
    """
    Extract generation capacity from preloaded document pages for one project.

    Pages are expected to be in priority order (main docs first, then others).
    """
    _ = project_title, project_description, project_type  # kept for interface parity

    project_docs = documents_df[documents_df["project_id"] == project_id]
    if project_docs.empty:
        return _empty_capacity_result(source="no_documents")

    pages = pages_lookup.get(project_id, [])
    if not pages:
        return _empty_capacity_result(source="none")

    all_capacities = []
    for page_text in pages:
        capacities = extract_capacity_from_text(page_text, source="document")
        if capacities:
            all_capacities.extend(capacities)

    primary_power = get_primary_capacity(all_capacities, unit_type="power")
    primary_energy = get_primary_capacity(all_capacities, unit_type="energy")
    primary = primary_power or primary_energy

    return {
        "project_gencap_value": primary_power["value"] if primary_power else None,
        "project_gencap_unit": primary_power["unit"] if primary_power else None,
        "project_gencap_energy_value": primary_energy["value"] if primary_energy else None,
        "project_gencap_energy_unit": primary_energy["unit"] if primary_energy else None,
        "project_gencap_matches": [c["match"] for c in all_capacities if c["unit_type"] == "power"][:5],
        "project_gencap_energy_matches": [c["match"] for c in all_capacities if c["unit_type"] == "energy"][:5],
        "project_gencap_candidate_count": count_distinct_capacities(all_capacities, unit_type="power"),
        "project_gencap_energy_candidate_count": count_distinct_capacities(all_capacities, unit_type="energy"),
        "project_gencap_source": "document" if primary else "none",
        "project_gencap_confidence": primary["confidence"] if primary else "low",
        "project_gencap_context": primary["context"] if primary else None,
        "project_gencap_candidates_json": [
            {"id": i + 1, "value": c["value"], "unit": c["unit"], "context": c["context"]}
            for i, c in enumerate(
                [x for x in all_capacities if x["unit_type"] == "power" and x.get("context")][:5]
            )
        ],
    }


# --------------------------
# REGEX RUNNER
# --------------------------

def run_capacity_extraction(
    clean_energy_only=True,
    sample_size=None,
    source=None,
    output_path=None,
    parallel_workers=0,
    regex_page_cap: Optional[int] = 50,
    regex_fallback_all_pages: bool = True,
):
    """
    Run generation capacity regex extraction for all projects.

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

    if source is None and parallel_workers and parallel_workers > 1:
        return _run_parallel_sources(
            projects,
            clean_energy_only,
            sample_size,
            parallel_workers,
            output_path,
            regex_page_cap=regex_page_cap,
            regex_fallback_all_pages=regex_fallback_all_pages,
        )

    if source:
        projects = projects[projects['dataset_source'] == source]

    if sample_size:
        projects = projects.head(sample_size)
        print(f"Sampling {len(projects):,} projects")

    results = []
    page_cap = regex_page_cap if (regex_page_cap is not None and regex_page_cap > 0) else None

    sources = [source] if source else list(projects['dataset_source'].unique())
    for src in sources:
        print(f"\nProcessing {src} projects...")

        source_projects = projects if source else projects[projects['dataset_source'] == src]
        data_dir = PROCESSED_DIR / src.lower()

        documents_df = pd.read_parquet(data_dir / "documents.parquet")
        pages_path = data_dir / "pages.parquet"

        def extract_id(x):
            if isinstance(x, dict):
                return x.get('value', '')
            return x

        documents_df['project_id'] = documents_df['project_id'].apply(extract_id)

        # Pass 1: title/description only (no page I/O)
        source_results = []
        doc_needed = []
        for idx, (_, project) in enumerate(source_projects.iterrows()):
            if idx % 100 == 0:
                print(f"  Pass 1 (title/description) project {idx + 1}/{len(source_projects)}...")

            project_id = project['project_id']
            capacity = extract_project_capacity_title_description(
                project_title=project.get('project_title', ''),
                project_description=project.get('project_description', ''),
            )
            if capacity is not None:
                source_results.append({
                    "project_id": project_id,
                    "dataset_source": src,
                    **capacity,
                })
            else:
                doc_needed.append(project)

        # Pass 2: bulk DuckDB page load for projects still needing document scan.
        if doc_needed:
            print(f"  Pass 2 (DuckDB pages): {len(doc_needed):,} projects")
            doc_needed_ids = {p["project_id"] for p in doc_needed}

            docs_needed = documents_df[documents_df["project_id"].isin(doc_needed_ids)].copy()
            docs_present_ids = set(docs_needed["project_id"].unique())
            main_series = docs_needed["main_document"] if "main_document" in docs_needed.columns else pd.Series("", index=docs_needed.index)
            main_mask = main_series.fillna("").astype(str).str.upper() == "YES"

            # Stage A: scan main documents first (matches legacy behavior)
            main_pairs = build_regex_document_pairs(docs_needed[main_mask], doc_needed_ids)
            main_lookup = load_project_pages_with_duckdb(
                pages_path=pages_path,
                document_pairs=main_pairs,
                max_pages=page_cap,
            )

            fallback_to_other = []
            doc_stage_results = {}
            for idx, project in enumerate(doc_needed):
                if idx % 100 == 0:
                    print(f"  Pass 2A (main docs) project {idx + 1}/{len(doc_needed)}...")

                project_id = project["project_id"]
                if project_id not in docs_present_ids:
                    doc_stage_results[project_id] = _empty_capacity_result(source="no_documents")
                    continue

                main_pages = main_lookup.get(project_id, [])
                main_capacity = extract_project_capacity_from_pages(
                    project_id=project_id,
                    project_title=project.get("project_title", ""),
                    project_description=project.get("project_description", ""),
                    project_type=project.get("project_type", ""),
                    pages_lookup={project_id: main_pages},
                    documents_df=documents_df,
                )

                if main_capacity.get("project_gencap_source") == "document":
                    doc_stage_results[project_id] = main_capacity
                else:
                    fallback_to_other.append(project)

            # Stage B: only for projects with no main-document hit, scan other docs.
            if fallback_to_other:
                other_ids = {p["project_id"] for p in fallback_to_other}
                other_docs = docs_needed[docs_needed["project_id"].isin(other_ids)].copy()
                if "main_document" in other_docs.columns:
                    other_docs = other_docs[other_docs["main_document"].fillna("").astype(str).str.upper() != "YES"]

                other_pairs = build_regex_document_pairs(other_docs, other_ids)
                other_lookup = load_project_pages_with_duckdb(
                    pages_path=pages_path,
                    document_pairs=other_pairs,
                    max_pages=page_cap,
                )

                for idx, project in enumerate(fallback_to_other):
                    if idx % 100 == 0:
                        print(f"  Pass 2B (other docs) project {idx + 1}/{len(fallback_to_other)}...")

                    project_id = project["project_id"]
                    capacity = extract_project_capacity_from_pages(
                        project_id=project_id,
                        project_title=project.get("project_title", ""),
                        project_description=project.get("project_description", ""),
                        project_type=project.get("project_type", ""),
                        pages_lookup=other_lookup,
                        documents_df=documents_df,
                    )
                    doc_stage_results[project_id] = capacity

            # Stage C: for remaining no-hit projects, rescan all pages to recover parity.
            if regex_fallback_all_pages and page_cap is not None:
                unresolved = [
                    p for p in doc_needed
                    if p["project_id"] in docs_present_ids
                    and doc_stage_results.get(p["project_id"], {}).get("project_gencap_source") != "document"
                ]
                if unresolved:
                    print(f"  Pass 2C (all pages fallback): {len(unresolved):,} projects")
                    unresolved_ids = {p["project_id"] for p in unresolved}
                    # C1: full main-doc scan
                    full_main_pairs = build_regex_document_pairs(docs_needed[main_mask], unresolved_ids)
                    full_main_lookup = load_project_pages_with_duckdb(
                        pages_path=pages_path,
                        document_pairs=full_main_pairs,
                        max_pages=None,
                    )

                    unresolved_after_main = []
                    for idx, project in enumerate(unresolved):
                        if idx % 100 == 0:
                            print(f"  Pass 2C1 (main docs full) project {idx + 1}/{len(unresolved)}...")

                        project_id = project["project_id"]
                        capacity_main = extract_project_capacity_from_pages(
                            project_id=project_id,
                            project_title=project.get("project_title", ""),
                            project_description=project.get("project_description", ""),
                            project_type=project.get("project_type", ""),
                            pages_lookup={project_id: full_main_lookup.get(project_id, [])},
                            documents_df=documents_df,
                        )
                        if capacity_main.get("project_gencap_source") == "document":
                            doc_stage_results[project_id] = capacity_main
                        else:
                            unresolved_after_main.append(project)

                    # C2: for still-unresolved rows, full other-doc scan.
                    if unresolved_after_main:
                        unresolved_other_ids = {p["project_id"] for p in unresolved_after_main}
                        unresolved_other_docs = docs_needed[docs_needed["project_id"].isin(unresolved_other_ids)].copy()
                        if "main_document" in unresolved_other_docs.columns:
                            unresolved_other_docs = unresolved_other_docs[
                                unresolved_other_docs["main_document"].fillna("").astype(str).str.upper() != "YES"
                            ]

                        full_other_pairs = build_regex_document_pairs(unresolved_other_docs, unresolved_other_ids)
                        full_other_lookup = load_project_pages_with_duckdb(
                            pages_path=pages_path,
                            document_pairs=full_other_pairs,
                            max_pages=None,
                        )

                        for idx, project in enumerate(unresolved_after_main):
                            if idx % 100 == 0:
                                print(f"  Pass 2C2 (other docs full) project {idx + 1}/{len(unresolved_after_main)}...")

                            project_id = project["project_id"]
                            capacity_other = extract_project_capacity_from_pages(
                                project_id=project_id,
                                project_title=project.get("project_title", ""),
                                project_description=project.get("project_description", ""),
                                project_type=project.get("project_type", ""),
                                pages_lookup={project_id: full_other_lookup.get(project_id, [])},
                                documents_df=documents_df,
                            )
                            doc_stage_results[project_id] = capacity_other

            for project in doc_needed:
                project_id = project["project_id"]
                capacity = doc_stage_results.get(project_id, _empty_capacity_result(source="none"))
                source_results.append({
                    "project_id": project_id,
                    "dataset_source": src,
                    **capacity,
                })

        results.extend(source_results)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        results_df = pd.DataFrame(columns=['project_id', 'dataset_source'])

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
    src, clean_energy_only, sample_size, regex_page_cap, regex_fallback_all_pages = args
    tmp_path = ANALYSIS_DIR / f"projects_gencap_{src.lower()}_tmp.parquet"
    run_capacity_extraction(
        clean_energy_only=clean_energy_only,
        sample_size=sample_size,
        source=src,
        output_path=str(tmp_path),
        parallel_workers=0,
        regex_page_cap=regex_page_cap,
        regex_fallback_all_pages=regex_fallback_all_pages,
    )
    return str(tmp_path)


def _run_parallel_sources(
    projects,
    clean_energy_only,
    sample_size,
    parallel_workers,
    output_path,
    regex_page_cap: Optional[int] = 50,
    regex_fallback_all_pages: bool = True,
):
    """Run per-source extraction in parallel and combine outputs."""
    from multiprocessing import get_context

    sources = list(projects['dataset_source'].unique())
    if not sources:
        print("No sources found to process.")
        return None

    tmp_paths = []
    ctx = get_context("spawn")

    with ctx.Pool(processes=min(parallel_workers, len(sources))) as pool:
        worker_args = [
            (s, clean_energy_only, sample_size, regex_page_cap, regex_fallback_all_pages)
            for s in sources
        ]
        for p in pool.map(_parallel_worker, worker_args):
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

    for p in tmp_paths:
        if p.exists():
            p.unlink()

    return combined


def has_number_with_unit(text: str) -> bool:
    """Check if text has a number followed by a power unit."""
    if not text:
        return False
    pattern = (
        r'\d[\d,\.]*\s*(?:-|–|—)?\s*'
        r'(?:MWh|GWh|kWh|MWac|MWdc|MWe|MWt|MWth|MWp|GWac|GWdc|kWe|kWac|kWdc|MW|GW|kW|'
        r'megawatt(?:-?\s*hours?)?|gigawatt(?:-?\s*hours?)?|kilowatt(?:-?\s*hours?)?)'
    )
    return bool(re.search(pattern, text, re.IGNORECASE))


def _normalize_unit_llm(unit: str) -> str:
    """Normalize common unit strings to standard form (including energy units)."""
    u = unit.lower().strip()
    mapping = {
        'mw': 'MW', 'mwe': 'MW', 'mwt': 'MW', 'megawatt': 'MW', 'megawatts': 'MW',
        'gw': 'GW', 'gwe': 'GW', 'gigawatt': 'GW', 'gigawatts': 'GW',
        'kw': 'kW', 'kwe': 'kW', 'kilowatt': 'kW', 'kilowatts': 'kW',
        'mwh': 'MWh', 'gwh': 'GWh', 'kwh': 'kWh',
        'megawatt-hour': 'MWh', 'megawatt-hours': 'MWh',
        'gigawatt-hour': 'GWh', 'gigawatt-hours': 'GWh',
        'kilowatt-hour': 'kWh', 'kilowatt-hours': 'kWh',
    }
    return mapping.get(u, unit)


def _fallback_extract_from_candidates(sentences: list) -> dict:
    """Fallback extraction: pick max numeric capacity from candidate sentences."""
    if not sentences:
        return {"capacity_value": None, "capacity_unit": None, "confidence": "low", "source_quote": None}

    pattern = re.compile(
        r'(\d[\d,\.]*)\s*'
        r'(MWh|GWh|kWh|MWac|MWdc|MWe|MWt|MWth|MWp|GWac|GWdc|kWe|kWac|kWdc|MW|GW|kW|'
        r'megawatt(?:-?\s*hours?)?|gigawatt(?:-?\s*hours?)?|kilowatt(?:-?\s*hours?)?)',
        re.IGNORECASE,
    )

    matches = []
    for s in sentences:
        for m in pattern.finditer(s):
            val_str, unit_str = m.group(1), m.group(2)
            try:
                val = float(val_str.replace(',', ''))
            except ValueError:
                continue
            unit = _normalize_unit_llm(unit_str)
            matches.append((val, unit, m.group(0), s))

    if not matches:
        return {"capacity_value": None, "capacity_unit": None, "confidence": "low", "source_quote": None}

    power_units = {'GW', 'MW', 'kW'}
    energy_units = {'GWh', 'MWh', 'kWh'}

    def to_base(val, unit):
        if unit == 'GW':
            return val * 1000
        if unit == 'kW':
            return val / 1000
        if unit == 'GWh':
            return val * 1000
        if unit == 'kWh':
            return val / 1000
        return val

    power = [m for m in matches if m[1] in power_units]
    energy = [m for m in matches if m[1] in energy_units]
    pool = power if power else energy

    best = max(pool, key=lambda m: to_base(m[0], m[1]))
    value, unit, quote, _sentence = best
    return {
        "capacity_value": value,
        "capacity_unit": unit,
        "confidence": "medium",
        "source_quote": quote,
    }


# --------------------------
# LLM EXTRACTION
# --------------------------

def build_extraction_prompt(candidates: list, project_title: str, project_type: str) -> str:
    """Build adjudication prompt for LLM to select among structured capacity candidates."""
    lines = []
    for c in candidates:
        lines.append(f'[{c["id"]}] {c["value"]} {c["unit"]} — "{str(c.get("context", ""))[:200]}"')
    candidates_text = "\n".join(lines)

    return f"""NEPA {project_type} review. These are capacity values found by regex in the document. Pick the ONE that is the proposed project's generation capacity.

Project: {project_title}

Candidates:
{candidates_text}

Rules:
1. Pick the capacity of the PROPOSED PROJECT being reviewed — not comparisons, existing infrastructure, or neighboring projects
2. Prefer candidates whose context uses words like "proposed", "nameplate", "rated", or "will generate"
3. Ignore candidates describing existing systems, regional totals, or reference projects

Return ONLY valid JSON:
{{"selected_index": <1-based int or null>, "confidence": "<high|medium|low>", "reasoning": "<one sentence max 80 chars>"}}

If no candidate clearly represents the proposed project capacity, return:
{{"selected_index": null, "confidence": "low", "reasoning": "no clear project capacity candidate"}}

JSON:"""


def call_claude_api(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
    max_retries: int = 3,
) -> Optional[str]:
    """Call Claude (Anthropic SDK) and return response text."""
    try:
        import anthropic
    except Exception as e:
        return f"__LLM_ERROR__:ImportError:{e}"

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "__LLM_ERROR__:APIError:ANTHROPIC_API_KEY not set"

    try:
        client = anthropic.Anthropic(timeout=timeout)
    except Exception:
        client = anthropic.Anthropic()

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = []
            for block in getattr(msg, "content", []):
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts).strip()
        except anthropic.RateLimitError as e:
            retry_after = min(60, 2 ** attempt * 2)
            response = getattr(e, "response", None)
            headers = getattr(response, "headers", {}) if response is not None else {}
            if headers:
                val = headers.get("retry-after") or headers.get("Retry-After")
                if val:
                    try:
                        retry_after = max(1, int(float(val)))
                    except (TypeError, ValueError):
                        pass
            time.sleep(retry_after)
            continue
        except anthropic.APITimeoutError as e:
            return f"__LLM_ERROR__:APITimeoutError:{e}"
        except anthropic.APIError as e:
            return f"__LLM_ERROR__:APIError:{e}"
        except Exception as e:
            return f"__LLM_ERROR__:Exception:{e}"

    return "__LLM_ERROR__:RateLimitError:rate_limit_exhausted"


def parse_llm_response(response: str) -> dict:
    """Parse LLM response for candidate selection adjudication."""
    empty = {"selected_index": None, "confidence": "low", "reasoning": None, "parse_error": True}
    if not response:
        return empty
    if isinstance(response, str) and response.startswith("__LLM_ERROR__"):
        return {**empty, "llm_error": response}
    try:
        m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if m:
            result = json.loads(m.group())
            result.setdefault("selected_index", None)
            result.setdefault("confidence", "low")
            result.setdefault("reasoning", None)
            result["parse_error"] = False
            try:
                if result["selected_index"] is not None:
                    result["selected_index"] = int(result["selected_index"])
            except (TypeError, ValueError):
                result["selected_index"] = None
            return result
    except json.JSONDecodeError:
        pass
    return empty


def extract_capacity_with_llm(candidates: list, project_title: str, project_type: str,
                               model: str = DEFAULT_MODEL) -> dict:
    """Use LLM to adjudicate among structured regex capacity candidates."""
    if not candidates:
        return {
            "capacity_value": None,
            "capacity_unit": None,
            "confidence": "low",
            "source_quote": None,
            "extraction_method": "no_candidates",
            "parse_error": False,
            "llm_error": None,
            "llm_selected_candidate_id": None,
            "llm_reasoning": None,
            "llm_selection_mode": "none",
            "num_candidates": 0,
        }

    prompt = build_extraction_prompt(candidates, project_title, project_type)
    response = call_claude_api(prompt, model=model)
    result = parse_llm_response(response)

    base = {
        "llm_selected_candidate_id": None,
        "llm_reasoning": result.get("reasoning"),
        "llm_selection_mode": "none",
        "extraction_method": "llm",
        "num_candidates": len(candidates),
        "parse_error": bool(result.get("parse_error", False)),
        "llm_error": result.get("llm_error"),
    }

    if result.get("llm_error"):
        err = str(result["llm_error"])
        timeout_tokens = ("ReadTimeout", "APITimeoutError", "claude_timeout", "timeout")
        base["extraction_method"] = "llm_timeout" if any(t in err for t in timeout_tokens) else "llm_error"
        return {
            **base,
            "capacity_value": None,
            "capacity_unit": None,
            "confidence": "low",
            "source_quote": None,
        }

    idx = result.get("selected_index")
    if idx is not None and 1 <= idx <= len(candidates):
        chosen = candidates[idx - 1]
        base["llm_selected_candidate_id"] = idx
        base["llm_selection_mode"] = "single"
        return {
            **base,
            "capacity_value": chosen.get("value"),
            "capacity_unit": chosen.get("unit"),
            "confidence": result.get("confidence", "medium"),
            "source_quote": str(chosen.get("context", ""))[:200] if chosen.get("context") else None,
        }

    fallback_sentences = [str(c.get("context")) for c in candidates if c.get("context")]
    fallback = _fallback_extract_from_candidates(fallback_sentences)
    if fallback.get("capacity_value") is not None:
        base["extraction_method"] = "fallback_from_candidates"
        return {**base, **fallback}

    return {
        **base,
        "capacity_value": None,
        "capacity_unit": None,
        "confidence": "low",
        "source_quote": None,
        "extraction_method": "llm_no_selection",
    }


# --------------------------
# LLM PROJECT-LEVEL EXTRACTION
# --------------------------

def extract_capacity_for_project(
    project_id: str,
    project_title: str,
    project_type: str,
    candidates: Optional[list] = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> dict:
    """
    Extract generation capacity for a single project using the LLM.

    Receives pre-extracted structured candidates from the regex pass
    (project_gencap_candidates_json). No page I/O is performed.

    Args:
        project_id: Project identifier
        project_title: Project name (for context in LLM prompt)
        project_type: Project type (e.g., 'solar', 'wind')
        candidates: Candidate dicts from regex extraction
            (project_gencap_candidates_json), each containing id/value/unit/context.
        model: Claude model to use
        verbose: Print progress

    Returns:
        dict with capacity_value, capacity_unit, confidence, source_quote, etc.
    """
    result = {
        "project_id": project_id,
        "project_title": project_title,
        "project_type": project_type,
        "capacity_value": None,
        "capacity_unit": None,
        "confidence": "low",
        "source_quote": None,
        "extraction_method": None,
        "candidates_found": 0,
        "llm_selected_candidate_id": None,
        "llm_reasoning": None,
        "llm_selection_mode": None,
    }

    if is_non_power_project(project_type):
        result["extraction_method"] = "skipped_non_power"
        result["note"] = "Project type uses non-power metrics (volume, not MW)"
        return result

    # Filter to candidates that contain a numeric capacity value
    normalized_candidates = []
    for i, c in enumerate(candidates or [], start=1):
        if not isinstance(c, dict):
            continue
        item = dict(c)
        try:
            item["id"] = int(item.get("id")) if item.get("id") is not None else i
        except (TypeError, ValueError):
            item["id"] = i
        normalized_candidates.append(item)

    valid_candidates = [
        c for c in normalized_candidates
        if c.get("context") and has_number_with_unit(str(c.get("context", "")))
    ]
    result["candidates_found"] = len(valid_candidates)

    if not valid_candidates:
        result["extraction_method"] = "no_candidates"
        return result

    if verbose:
        print(f"  {len(valid_candidates)} structured candidates from regex pass")

    llm_result = extract_capacity_with_llm(valid_candidates, project_title, project_type, model=model)
    result.update(llm_result)

    return result


# --------------------------
# LLM BATCH EXTRACTION
# --------------------------


def extract_capacity_for_projects(
    source: str = 'eis',
    clean_energy_only: bool = True,
    sample_size: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
    regex_results_path: Optional[str] = None,
    only_low_medium: bool = False,
    ambiguous_only: bool = True,
    workers: int = 4,
    require_regex_capacity: bool = False,
    project_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Run LLM extraction for multiple projects within a single source.

    Args:
        source: Dataset source ('eis', 'ea', 'ce')
        ambiguous_only: Only process projects with 2+ distinct regex candidates (default True)
    """
    print(f"\n=== LLM Capacity Extraction ({source.upper()}) ===")
    print(f"Model: {model}")

    if regex_results_path:
        regex_path = Path(regex_results_path)
        if not regex_path.exists():
            raise FileNotFoundError(f"Regex results not found: {regex_path}")
        projects = pd.read_parquet(regex_path)
    else:
        projects = pd.read_parquet(ANALYSIS_DIR / "projects_combined.parquet")

    projects = projects[projects['dataset_source'] == source.upper()]

    if clean_energy_only and 'project_energy_type' in projects.columns:
        projects = projects[projects['project_energy_type'] == 'Clean']

    if ambiguous_only and 'project_gencap_candidate_count' in projects.columns:
        projects = projects.copy()
        projects['project_gencap_candidate_count'] = pd.to_numeric(
            projects['project_gencap_candidate_count'], errors='coerce'
        ).fillna(0)
        projects = projects[projects['project_gencap_candidate_count'] >= 2]
        print(f"Ambiguous-only filter (2+ distinct regex power candidates): {len(projects):,} projects")
    elif ambiguous_only:
        print("Ambiguous-only filter requested, but project_gencap_candidate_count not found; skipping.")

    if only_low_medium and 'project_gencap_confidence' in projects.columns:
        conf = projects['project_gencap_confidence'].fillna('low').astype(str).str.lower()
        projects = projects[conf.isin(['low', 'medium'])]

    if only_low_medium and 'project_gencap_source' in projects.columns:
        projects = projects[~projects['project_gencap_source'].isin(['title', 'skipped_transmission_only'])]

    if require_regex_capacity and 'project_gencap_value' in projects.columns:
        projects = projects[projects['project_gencap_value'].notna()]

    if project_id:
        projects = projects[projects['project_id'] == project_id]

    print(f"Projects to process: {len(projects):,}")

    if sample_size:
        projects = projects.sample(min(sample_size, len(projects)), random_state=42)
        print(f"Sampled: {len(projects)}")

    ctx_col = "project_gencap_candidates_json"
    cols = ["project_id", "project_title", "project_type"]
    if ctx_col in projects.columns:
        cols.append(ctx_col)
    else:
        print(f"WARNING: {ctx_col} not found in regex output. Re-run --run regex first.")
    project_records = projects[cols].to_dict("records")

    def run_one(project_row):
        pid = project_row["project_id"]
        raw = project_row.get(ctx_col, [])
        if isinstance(raw, str):
            try:
                candidates = json.loads(raw)
            except Exception:
                candidates = []
        elif isinstance(raw, (list, tuple, np.ndarray)):
            candidates = list(raw)
        else:
            candidates = []

        normalized_candidates = []
        for item in candidates:
            if isinstance(item, dict):
                normalized_candidates.append(item)
            elif isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, dict):
                        normalized_candidates.append(parsed)
                except Exception:
                    continue
        return extract_capacity_for_project(
            project_id=pid,
            project_title=project_row["project_title"],
            project_type=project_row["project_type"],
            candidates=normalized_candidates,
            model=model,
            verbose=False,
        )

    results = []
    if workers and workers > 1 and project_records:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, project): project for project in project_records}
            for idx, future in enumerate(as_completed(futures), start=1):
                if verbose and idx % 10 == 0:
                    print(f"\nProcessed {idx}/{len(project_records)} projects...")
                try:
                    results.append(future.result())
                except Exception as e:
                    project = futures[future]
                    results.append({
                        "project_id": project.get("project_id"),
                        "project_title": project.get("project_title"),
                        "project_type": project.get("project_type"),
                        "capacity_value": None,
                        "capacity_unit": None,
                        "confidence": "low",
                        "source_quote": None,
                        "extraction_method": "llm_error",
                        "candidates_found": 0,
                        "llm_selected_candidate_id": None,
                        "llm_reasoning": None,
                        "llm_selection_mode": None,
                        "llm_error": f"__LLM_ERROR__:Exception:{e}",
                    })
    else:
        for idx, project in enumerate(project_records):
            if verbose and idx % 10 == 0:
                print(f"\nProcessing {idx + 1}/{len(project_records)}: {project['project_title'][:50]}...")
            result = run_one(project)
            results.append(result)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        results_df = pd.DataFrame(columns=[
            "project_id", "project_title", "project_type",
            "capacity_value", "capacity_unit", "confidence", "source_quote",
            "extraction_method", "candidates_found",
            "num_candidates", "parse_error", "llm_error",
            "llm_selected_candidate_id", "llm_reasoning", "llm_selection_mode",
        ])
    results_df["dataset_source"] = source.upper()
    run_timestamp_utc = pd.Timestamp.utcnow().isoformat()
    results_df["llm_run_completed_at_utc"] = run_timestamp_utc
    results_df["llm_model_used"] = model
    results_df["llm_trigger_mode"] = "regex_multi_candidate" if ambiguous_only else "candidate_json"

    print("\n=== LLM Summary ===")
    has_capacity = results_df['capacity_value'].notna()
    print(f"Projects with capacity extracted: {has_capacity.sum()} / {len(results_df)} ({has_capacity.mean()*100:.1f}%)")
    print(f"Extraction methods: {results_df['extraction_method'].value_counts().to_dict()}")

    return results_df


# --------------------------
# LLM MERGE
# --------------------------

def merge_llm_results_into_regex(
    regex_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    source: str,
    run_timestamp_utc: str,
    model: str,
    trigger_mode: str,
) -> pd.DataFrame:
    """
    Merge LLM adjudication results into regex dataset and compute final capacity fields.

    Final capacity = regex output, overridden by validated LLM selections.
    """
    merged = regex_df.copy()
    source = source.upper()
    source_mask = merged["dataset_source"].astype(str).str.upper() == source

    for col in [
        "project_gencap_llm_run_completed_at_utc",
        "project_gencap_llm_model_used",
        "project_gencap_llm_trigger_mode",
    ]:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged.loc[source_mask, "project_gencap_llm_run_completed_at_utc"] = run_timestamp_utc
    merged.loc[source_mask, "project_gencap_llm_model_used"] = model
    merged.loc[source_mask, "project_gencap_llm_trigger_mode"] = trigger_mode

    llm_cols = [
        "llm_capacity_value", "llm_capacity_unit", "llm_capacity_unit_norm",
        "llm_confidence", "llm_source_quote", "llm_extraction_method",
        "llm_candidates_found", "llm_num_candidates",
        "llm_parse_error", "llm_error", "llm_run_completed_at_utc",
        "llm_model_used", "llm_trigger_mode",
        "llm_selected_candidate_id", "llm_reasoning", "llm_selection_mode",
        "project_gencap_llm_triggered",
        "project_gencap_llm_selected_from_regex_candidates",
        "project_gencap_llm_selection_logic",
        "project_gencap_llm_selected_value",
        "project_gencap_llm_selected_unit",
        "project_gencap_llm_selected_quote",
        "project_gencap_llm_selected_candidate_id",
        "project_gencap_llm_reasoning",
        "project_gencap_llm_selection_mode",
    ]
    for col in llm_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged.loc[source_mask, llm_cols] = pd.NA
    merged.loc[source_mask, "project_gencap_llm_triggered"] = False

    if llm_df is not None and not llm_df.empty:
        llm_src = llm_df.copy()
        if "dataset_source" in llm_src.columns:
            llm_src["dataset_source"] = llm_src["dataset_source"].astype(str).str.upper()
        else:
            llm_src["dataset_source"] = source
        llm_src = llm_src[llm_src["dataset_source"] == source].copy()
        if not llm_src.empty:
            if {"project_id", "dataset_source"}.issubset(llm_src.columns):
                llm_src = llm_src.drop_duplicates(subset=["project_id", "dataset_source"], keep="last")

            llm_src["llm_capacity_unit_norm"] = llm_src["capacity_unit"].apply(normalize_power_unit)
            llm_src["__key"] = llm_src["project_id"].astype(str) + "|" + llm_src["dataset_source"].astype(str)
            llm_keyed = llm_src.set_index("__key")

            merged["__key"] = merged["project_id"].astype(str) + "|" + merged["dataset_source"].astype(str)
            source_keys = set(llm_keyed.index.tolist())
            merged.loc[source_mask & merged["__key"].isin(source_keys), "project_gencap_llm_triggered"] = True

            map_pairs = [
                ("capacity_value", "llm_capacity_value"),
                ("capacity_unit", "llm_capacity_unit"),
                ("llm_capacity_unit_norm", "llm_capacity_unit_norm"),
                ("confidence", "llm_confidence"),
                ("source_quote", "llm_source_quote"),
                ("extraction_method", "llm_extraction_method"),
                ("candidates_found", "llm_candidates_found"),
                ("num_candidates", "llm_num_candidates"),
                ("parse_error", "llm_parse_error"),
                ("llm_error", "llm_error"),
                ("llm_run_completed_at_utc", "llm_run_completed_at_utc"),
                ("llm_model_used", "llm_model_used"),
                ("llm_trigger_mode", "llm_trigger_mode"),
                ("llm_selected_candidate_id", "llm_selected_candidate_id"),
                ("llm_reasoning", "llm_reasoning"),
                ("llm_selection_mode", "llm_selection_mode"),
            ]

            for src_col, dst_col in map_pairs:
                if src_col in llm_keyed.columns:
                    merged.loc[source_mask, dst_col] = merged.loc[source_mask, "__key"].map(llm_keyed[src_col])

            if "project_gencap_matches" in merged.columns:
                merged.loc[source_mask, "project_gencap_llm_selected_from_regex_candidates"] = (
                    merged.loc[source_mask].apply(
                        lambda r: _llm_selection_in_regex_matches(
                            r.get("llm_capacity_value"),
                            r.get("llm_capacity_unit_norm"),
                            r.get("project_gencap_matches"),
                        ),
                        axis=1,
                    )
                )
            else:
                merged.loc[source_mask, "project_gencap_llm_selected_from_regex_candidates"] = False

            merged.loc[source_mask, "project_gencap_llm_selected_value"] = merged.loc[source_mask, "llm_capacity_value"]
            merged.loc[source_mask, "project_gencap_llm_selected_unit"] = merged.loc[source_mask, "llm_capacity_unit_norm"]
            merged.loc[source_mask, "project_gencap_llm_selected_quote"] = merged.loc[source_mask, "llm_source_quote"]
            merged.loc[source_mask, "project_gencap_llm_selected_candidate_id"] = merged.loc[source_mask, "llm_selected_candidate_id"]
            merged.loc[source_mask, "project_gencap_llm_reasoning"] = merged.loc[source_mask, "llm_reasoning"]
            merged.loc[source_mask, "project_gencap_llm_selection_mode"] = merged.loc[source_mask, "llm_selection_mode"]

            merged.drop(columns=["__key"], inplace=True, errors="ignore")

    llm_value_num = pd.to_numeric(merged.get("llm_capacity_value"), errors="coerce")
    llm_extraction_method = merged.get("llm_extraction_method").fillna("").astype(str)

    merged["llm_is_valid_power"] = (
        llm_value_num.notna()
        & (llm_value_num > 0)
        & merged.get("llm_capacity_unit_norm").isin(POWER_UNITS)
    )
    merged["llm_is_rejected_method"] = llm_extraction_method.isin(
        ["no_candidates", "llm_no_selection", "llm_error", "llm_timeout"]
    )
    merged["llm_should_override_regex"] = merged["llm_is_valid_power"] & ~merged["llm_is_rejected_method"]

    for col, fallback in [
        ("project_gencap_final_value", "project_gencap_value"),
        ("project_gencap_final_unit", "project_gencap_unit"),
        ("project_gencap_final_source", "project_gencap_source"),
        ("project_gencap_final_confidence", "project_gencap_confidence"),
        ("project_gencap_final_quote", "project_gencap_context"),
    ]:
        if col not in merged.columns:
            merged[col] = merged.get(fallback)
        merged.loc[source_mask, col] = merged.loc[source_mask, fallback]

    llm_override_mask = source_mask & merged["llm_should_override_regex"].eq(True)
    merged.loc[llm_override_mask, "project_gencap_final_value"] = llm_value_num[llm_override_mask]
    merged.loc[llm_override_mask, "project_gencap_final_unit"] = merged.loc[llm_override_mask, "llm_capacity_unit_norm"]
    merged.loc[llm_override_mask, "project_gencap_final_source"] = merged.loc[llm_override_mask, "llm_extraction_method"]
    merged.loc[llm_override_mask, "project_gencap_final_confidence"] = merged.loc[llm_override_mask, "llm_confidence"]
    merged.loc[llm_override_mask, "project_gencap_final_quote"] = merged.loc[llm_override_mask, "llm_source_quote"]

    if "llm_merge_decision" not in merged.columns:
        merged["llm_merge_decision"] = pd.NA
    merged.loc[source_mask, "llm_merge_decision"] = "regex_no_llm"
    merged.loc[source_mask & merged["llm_capacity_value"].notna() & ~llm_override_mask, "llm_merge_decision"] = "regex_invalid_or_rejected_llm"
    merged.loc[llm_override_mask & merged["project_gencap_value"].notna(), "llm_merge_decision"] = "llm_override_regex"
    merged.loc[llm_override_mask & merged["project_gencap_value"].isna(), "llm_merge_decision"] = "llm_only_fill"
    merged.loc[
        source_mask & merged["project_gencap_final_value"].isna() & merged["llm_capacity_value"].isna(),
        "llm_merge_decision",
    ] = "no_capacity"

    # Human-readable audit trail
    llm_triggered = merged["project_gencap_llm_triggered"].eq(True)
    llm_valid = merged["llm_is_valid_power"].eq(True)
    llm_rejected = merged["llm_is_rejected_method"].eq(True)
    llm_selected_regex = merged["project_gencap_llm_selected_from_regex_candidates"].eq(True)

    merged.loc[source_mask, "project_gencap_llm_selection_logic"] = "not_triggered"
    merged.loc[source_mask & llm_triggered, "project_gencap_llm_selection_logic"] = "triggered_no_selection"
    merged.loc[
        source_mask & llm_triggered & llm_valid & llm_rejected,
        "project_gencap_llm_selection_logic",
    ] = "selected_valid_power_rejected_by_method"
    merged.loc[
        source_mask & llm_triggered & llm_valid & ~llm_rejected & llm_selected_regex,
        "project_gencap_llm_selection_logic",
    ] = "selected_regex_candidate"
    merged.loc[
        source_mask & llm_triggered & llm_valid & ~llm_rejected & ~llm_selected_regex,
        "project_gencap_llm_selection_logic",
    ] = "selected_non_regex_candidate"
    merged.loc[
        source_mask & llm_triggered & ~llm_valid,
        "project_gencap_llm_selection_logic",
    ] = "selected_invalid_or_non_power"

    return merged


def resolve_regex_results_path(regex_results_path: Optional[str], source: str) -> Path:
    """Resolve input regex results path with sensible defaults."""
    if regex_results_path:
        return Path(regex_results_path)
    candidates = [
        ANALYSIS_DIR / f"projects_gencap_{source.lower()}.parquet",
        ANALYSIS_DIR / "projects_gencap.parquet",
        ANALYSIS_DIR / "projects_gencap_flagged.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    return ANALYSIS_DIR / "projects_gencap.parquet"


def run_llm_merge_pipeline(
    source: str = "eis",
    clean_energy_only: bool = True,
    sample_size: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
    regex_results_path: Optional[str] = None,
    only_low_medium: bool = False,
    ambiguous_only: bool = True,
    workers: int = 4,
    require_regex_capacity: bool = False,
    project_id: Optional[str] = None,
    output_path: Optional[str] = None,
    llm_output_path: Optional[str] = None,
) -> tuple:
    """
    Run LLM adjudication and immediately merge results into regex output dataset.

    Returns:
        tuple: (llm_results_df, merged_df, merged_output_path)
    """
    source = source.lower()
    regex_path = resolve_regex_results_path(regex_results_path, source)
    if not regex_path.exists():
        raise FileNotFoundError(f"Regex results not found: {regex_path}")

    regex_df = pd.read_parquet(regex_path)
    llm_results = extract_capacity_for_projects(
        source=source,
        clean_energy_only=clean_energy_only,
        sample_size=sample_size,
        model=model,
        verbose=verbose,
        regex_results_path=str(regex_path),
        only_low_medium=only_low_medium,
        ambiguous_only=ambiguous_only,
        workers=workers,
        require_regex_capacity=require_regex_capacity,
        project_id=project_id,
    )

    if not llm_results.empty and "llm_run_completed_at_utc" in llm_results.columns:
        run_timestamp_utc = str(llm_results["llm_run_completed_at_utc"].iloc[0])
    else:
        run_timestamp_utc = pd.Timestamp.utcnow().isoformat()
    trigger_mode = "regex_multi_candidate" if ambiguous_only else "candidate_json"
    merged = merge_llm_results_into_regex(
        regex_df=regex_df,
        llm_df=llm_results,
        source=source,
        run_timestamp_utc=run_timestamp_utc,
        model=model,
        trigger_mode=trigger_mode,
    )

    save_path = Path(output_path) if output_path else regex_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(save_path, index=False)

    llm_save_path = (
        Path(llm_output_path) if llm_output_path
        else (ANALYSIS_DIR / f"gencap_{source.lower()}_llm.parquet")
    )
    llm_save_path.parent.mkdir(parents=True, exist_ok=True)
    llm_results.to_parquet(llm_save_path, index=False)

    print(f"\nSaved merged dataset: {save_path}")
    print(f"Saved raw LLM output: {llm_save_path}")
    source_mask = merged["dataset_source"].astype(str).str.upper() == source.upper()
    override_count = int((source_mask & (merged["llm_merge_decision"] == "llm_override_regex")).sum())
    fill_count = int((source_mask & (merged["llm_merge_decision"] == "llm_only_fill")).sum())
    print(f"LLM final selection updates ({source.upper()}): overrides={override_count:,}, llm_only_fills={fill_count:,}")

    return llm_results, merged, save_path


# --------------------------
# CLI
# --------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generation capacity extraction: regex and LLM adjudication.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_gencap.py --run regex                      # regex extraction (all sources)
  python extract_gencap.py --run regex --parallel 3         # regex in parallel
  python extract_gencap.py --run regex --sample 100         # regex test sample
  python extract_gencap.py --run llm --workers 4            # LLM adjudication + merge
  python extract_gencap.py --run llm --sample 10            # LLM test run
  python extract_gencap.py --run llm --include-non-ambiguous --workers 4
  python extract_gencap.py --self-test                      # test regex patterns
""",
    )
    parser.add_argument('--run', choices=['regex', 'llm'], metavar='{regex,llm}',
                        help='regex: run regex extraction on all sources; llm: run LLM adjudication + merge on all sources')
    parser.add_argument('--self-test', action='store_true', help='Run built-in regex test cases and exit')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--all', action='store_true', help='Process all projects, not just clean energy')
    parser.add_argument('--input', type=str,
                        help='Input regex parquet for llm mode (default: data/analysis/projects_gencap.parquet)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'LLM model for --run llm mode (default: {DEFAULT_MODEL})')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel workers for --run llm mode (default: 4)')
    parser.add_argument('--include-non-ambiguous', action='store_true',
                        help='In --run llm mode, include projects with <2 regex candidates')
    parser.add_argument('--only-low-medium', action='store_true',
                        help='In --run llm mode, further limit to low/medium regex confidence')
    parser.add_argument('--require-regex-capacity', action='store_true',
                        help='In --run llm mode, only process rows with regex capacity values')
    parser.add_argument('--project-id', type=str,
                        help='In --run llm mode, run a single project_id')
    parser.add_argument('--llm-output', type=str,
                        help='Optional raw LLM output path (default: data/analysis/gencap_{source}_llm.parquet)')
    parser.add_argument('--output', type=str, help='Output file path (parquet)')
    parser.add_argument('--parallel', type=int, default=0,
                        help='Run CE/EA/EIS in parallel with N workers (--run regex only)')
    parser.add_argument('--regex-page-cap', type=int, default=50,
                        help='Fast regex pass cap: max pages per project for doc scans (default: 50; <=0 means all pages)')
    parser.add_argument('--no-regex-fallback-all-pages', action='store_true',
                        help='Disable regex fallback that rescans unresolved projects with all pages')

    args = parser.parse_args()

    if args.self_test:
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

    elif args.run == 'llm':
        input_path = Path(args.input) if args.input else (ANALYSIS_DIR / "projects_gencap.parquet")
        output_path = Path(args.output) if args.output else input_path

        sources = ["CE", "EA", "EIS"]
        current_input = input_path

        for src in sources:
            print(f"\n=== LLM adjudication + merge ({src}) ===")

            _, merged_df, saved_path = run_llm_merge_pipeline(
                source=src.lower(),
                clean_energy_only=not args.all,
                sample_size=args.sample,
                model=args.model,
                verbose=True,
                regex_results_path=str(current_input),
                only_low_medium=args.only_low_medium,
                ambiguous_only=not args.include_non_ambiguous,
                workers=args.workers,
                require_regex_capacity=args.require_regex_capacity,
                project_id=args.project_id,
                output_path=str(output_path),
                llm_output_path=args.llm_output,
            )

            source_mask = merged_df["dataset_source"].astype(str).str.upper() == src
            final_count = int((source_mask & merged_df["project_gencap_final_value"].notna()).sum())
            print(f"{src} rows with final capacity: {final_count:,}")
            current_input = saved_path

        print(f"\nFinal merged output: {output_path}")

    elif args.run == 'regex':
        run_capacity_extraction(
            clean_energy_only=not args.all,
            sample_size=args.sample,
            source=None,
            output_path=args.output,
            parallel_workers=args.parallel,
            regex_page_cap=args.regex_page_cap,
            regex_fallback_all_pages=not args.no_regex_fallback_all_pages,
        )

    else:
        parser.print_help()
