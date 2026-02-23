# --------------------------
# GENERATION CAPACITY EXTRACTION WITH LLM
# --------------------------
# Strategy: Filter sentences with capacity terms, then use local LLM (Ollama) to extract
# structured capacity information.
#
# Usage:
#   As module:
#     from extract_gencap_llm import extract_capacity_for_projects
#     results = extract_capacity_for_projects(project_ids, source='eis')
#
#   As CLI:
#     python extract_gencap_llm.py --source eis --sample 10
#     python extract_gencap_llm.py --source eis --run

import re
import json
import ast
import pandas as pd
import requests
from pathlib import Path
from typing import Optional
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
POWER_UNITS = {"GW", "MW", "kW"}

# Multiprocessing worker globals
_WORKER_DOCS = None
_WORKER_PAGES_PATH = None
_WORKER_MODEL = None
_WORKER_VERBOSE = False

# --------------------------
# CAPACITY SEARCH TERMS BY PROJECT TYPE
# --------------------------

# Base terms for all energy projects
BASE_CAPACITY_TERMS = {
    # Power units
    'mw', 'gw', 'kw', 'mwh', 'gwh', 'kwh',
    'megawatt', 'megawatts', 'gigawatt', 'gigawatts',
    'kilowatt', 'kilowatts',
    'megawatt-hour', 'megawatt-hours', 'kilowatt-hour', 'kilowatt-hours',
    # Context words
    'nameplate', 'capacity', 'generate', 'generates', 'generating',
    'generation', 'output', 'rated', 'produce', 'produces', 'producing',
}

# Additional terms by project type
PROJECT_TYPE_TERMS = {
    'solar': {'ac', 'dc', 'photovoltaic', 'pv', 'panel', 'array'},
    'wind': {'turbine', 'rotor', 'nacelle'},
    'nuclear': {'mwe', 'megawatt-electric', 'reactor', 'thermal', 'mwt'},
    'geothermal': {'binary', 'flash', 'steam', 'wellhead'},
    'hydropower': {'dam', 'turbine', 'head', 'flow'},
    'hydrokinetic': {'tidal', 'wave', 'current'},
    'biomass': {'btu', 'british thermal', 'boiler', 'combustion'},
    'storage': {'battery', 'storage', 'discharge', 'duration'},
    'transmission': {'kv', 'kilovolt', 'volt', 'voltage', 'transfer'},
    'carbon capture': {'co2', 'capture', 'sequestration', 'tons', 'mwe'},
}

# Project types where capacity is NOT measured in MW/GW/kW
NON_POWER_PROJECT_TYPES = {
    'pipeline': {'barrel', 'barrels', 'mcf', 'cubic feet', 'cubic meter',
                 'dekatherm', 'diameter', 'throughput'},
}


def get_terms_for_project_type(project_type: str) -> set:
    """Get relevant capacity terms based on project type."""
    if not project_type or not isinstance(project_type, str):
        return BASE_CAPACITY_TERMS

    pt_lower = project_type.lower()
    terms = BASE_CAPACITY_TERMS.copy()

    # Add type-specific terms
    for key, type_terms in PROJECT_TYPE_TERMS.items():
        if key in pt_lower:
            terms.update(type_terms)

    return terms


def is_non_power_project(project_type: str) -> bool:
    """Check if project uses non-power metrics (e.g., pipelines use volume).

    Only returns True if the project is PURELY a pipeline project,
    not if it includes pipeline along with power generation types.
    """
    if not project_type:
        return False
    pt_lower = project_type.lower()

    # If project has any power generation type, it's not a non-power project
    power_keywords = ['solar', 'wind', 'nuclear', 'geothermal', 'hydropower',
                      'hydrokinetic', 'biomass', 'energy production', 'energy storage']
    has_power_type = any(kw in pt_lower for kw in power_keywords)

    # Only mark as non-power if it's ONLY a pipeline project
    if has_power_type:
        return False

    return 'pipeline' in pt_lower and 'solar' not in pt_lower and 'wind' not in pt_lower


def extract_numeric_page_number(page_number) -> int:
    """Extract a numeric page token for stable ordering of mixed page labels."""
    if page_number is None or pd.isna(page_number):
        return 10**9
    match = re.search(r'(\d+)', str(page_number))
    if match:
        return int(match.group(1))
    return 10**9


def parse_match_count(value) -> int:
    """Count regex matches stored as list/array/JSON string."""
    if value is None:
        return 0
    if isinstance(value, float) and pd.isna(value):
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    # pyarrow-backed arrays can arrive as ndarray-like objects.
    if hasattr(value, "__len__") and not isinstance(value, str):
        try:
            return len(value)
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        if text.startswith("[") and text.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    if isinstance(parsed, (list, tuple)):
                        return len(parsed)
                except Exception:
                    pass
    return 0


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
        "gw": "GW",
        "gwe": "GW",
        "gwac": "GW",
        "gwdc": "GW",
        "gigawatt": "GW",
        "gigawatts": "GW",
        "mw": "MW",
        "mwe": "MW",
        "mwt": "MW",
        "mwth": "MW",
        "mwac": "MW",
        "mwdc": "MW",
        "mwp": "MW",
        "megawatt": "MW",
        "megawatts": "MW",
        "kw": "kW",
        "kwe": "kW",
        "kwac": "kW",
        "kwdc": "kW",
        "kilowatt": "kW",
        "kilowatts": "kW",
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


# --------------------------
# SENTENCE FILTERING
# --------------------------

def extract_words(text: str) -> set:
    """Extract lowercase words from text."""
    if not text:
        return set()
    return set(re.findall(r'\b[a-z]+\b', text.lower()))


def has_capacity_terms(text: str, terms: set) -> bool:
    """Check if text contains any capacity-related terms."""
    if not text:
        return False
    words = extract_words(text)
    return bool(words & terms)


def has_number_with_unit(text: str) -> bool:
    """Check if text has a number followed by a power unit."""
    if not text:
        return False
    # Match patterns like "50 MW", "1,500 megawatts", "2.5 GW", "15-MW".
    # Longer/more-specific units come first to avoid partial matches (MW in MWh).
    pattern = (
        r'\d[\d,\.]*\s*(?:-|–|—)?\s*'
        r'(?:MWh|GWh|kWh|MWac|MWdc|MWe|MWt|MWth|MWp|GWac|GWdc|kWe|kWac|kWdc|MW|GW|kW|'
        r'megawatt(?:-?\s*hours?)?|gigawatt(?:-?\s*hours?)?|kilowatt(?:-?\s*hours?)?)'
    )
    return bool(re.search(pattern, text, re.IGNORECASE))


def _normalize_unit_llm(unit: str) -> str:
    """Normalize common unit strings to standard form."""
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

    # Prioritize power units over energy, then choose highest value (normalized to MW or MWh)
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


def extract_candidate_sentences(text: str, terms: set, max_sentences: int = 10) -> list:
    """
    Extract sentences that likely contain capacity information.

    Returns list of sentences, sorted by relevance score.
    """
    if not text or not isinstance(text, str):
        return []

    # Split into sentences (handle various delimiters)
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)

    candidates = []
    for sent in sentences:
        sent = sent.strip()
        # More lenient length limits
        if len(sent) < 20:
            continue
        if len(sent) > 600 and not has_number_with_unit(sent):
            continue

        # Score the sentence
        score = 0

        # Strong signal: has number + unit (MW, GW, kW, megawatt, etc.)
        if has_number_with_unit(sent):
            score += 5  # Key signal
        elif has_capacity_terms(sent, terms):
            score += 1  # Weak signal without number
        else:
            continue  # Skip if no relevant terms

        # Keep long list-style sentences if they contain numeric units
        if len(sent) > 600 and has_number_with_unit(sent):
            score = max(score, 3)

        # Bonus for project-related context
        context_words = {'project', 'proposed', 'facility', 'plant', 'farm', 'array',
                        'system', 'generate', 'capacity', 'nameplate', 'rated'}
        if extract_words(sent) & context_words:
            score += 2

        candidates.append((sent, score))

    # Sort by score descending, take top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    # Lowered threshold: score >= 1 (just need SOME signal)
    return [sent for sent, score in candidates[:max_sentences] if score >= 1]


# --------------------------
# LLM EXTRACTION
# --------------------------

def build_extraction_prompt(sentences: list, project_title: str, project_type: str) -> str:
    """Build prompt for LLM to extract capacity."""

    sentences_text = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences[:5]))

    prompt = f"""Extract the proposed project's generation capacity from these text excerpts.

Project Title: {project_title}
Project Type: {project_type}

Text excerpts:
{sentences_text}

Instructions:
1. Find the PRIMARY generation capacity of the proposed project (not alternatives, not comparisons to other projects)
2. If multiple values exist, prefer "nameplate" or "rated" capacity
3. If a range is given (e.g., "50-100 MW"), use the higher value
4. The source_quote MUST include the numeric value and unit exactly as shown in the text
5. Return ONLY valid JSON, no other text

Return this exact JSON structure:
{{"capacity_value": <number or null>, "capacity_unit": "<MW|GW|kW|MWh|GWh|kWh or null>", "confidence": "<high|medium|low>", "source_quote": "<exact quote or null>"}}

If no project capacity is clearly stated OR you cannot provide a source_quote with the numeric value and unit, return:
{{"capacity_value": null, "capacity_unit": null, "confidence": "low", "source_quote": null}}

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
                    "temperature": 0.1,  # Low temperature for consistent extraction
                    "num_predict": 200,  # Limit response length
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"Ollama API error: {e}")
        return f"__OLLAMA_ERROR__:{type(e).__name__}:{e}"


def parse_llm_response(response: str) -> dict:
    """Parse LLM response into structured dict."""
    if not response:
        return {"capacity_value": None, "capacity_unit": None, "confidence": "low", "source_quote": None, "parse_error": True}

    if isinstance(response, str) and response.startswith("__OLLAMA_ERROR__"):
        return {"capacity_value": None, "capacity_unit": None, "confidence": "low", "source_quote": None, "parse_error": True, "llm_error": response}

    # Try to extract JSON from response
    try:
        # Look for JSON object in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Validate expected fields
            result.setdefault("capacity_value", None)
            result.setdefault("capacity_unit", None)
            result.setdefault("confidence", "low")
            result.setdefault("source_quote", None)
            result["parse_error"] = False
            return result
    except json.JSONDecodeError:
        pass

    return {"capacity_value": None, "capacity_unit": None, "confidence": "low", "source_quote": None, "parse_error": True}


def extract_capacity_with_llm(sentences: list, project_title: str, project_type: str,
                              model: str = DEFAULT_MODEL) -> dict:
    """Use LLM to extract capacity from candidate sentences."""
    if not sentences:
        return {"capacity_value": None, "capacity_unit": None, "confidence": "low",
                "source_quote": None, "extraction_method": "no_candidates"}

    # Reject if no numeric capacity patterns appear in candidates
    if not any(has_number_with_unit(s) for s in sentences):
        return {"capacity_value": None, "capacity_unit": None, "confidence": "low",
                "source_quote": None, "extraction_method": "no_numeric_candidates"}

    prompt = build_extraction_prompt(sentences, project_title, project_type)
    response = call_ollama(prompt, model=model)
    result = parse_llm_response(response)
    result["extraction_method"] = "llm"
    result["num_candidates"] = len(sentences)

    if result.get("llm_error"):
        result["extraction_method"] = "llm_timeout" if "ReadTimeout" in result["llm_error"] else "llm_error"
        return result

    # Enforce source_quote with numeric value + unit
    quote = result.get("source_quote")
    if result.get("capacity_value") is not None:
        if not quote or not has_number_with_unit(quote):
            fallback = _fallback_extract_from_candidates(sentences)
            if fallback.get("capacity_value") is not None:
                fallback["extraction_method"] = "fallback_from_candidates"
                fallback["num_candidates"] = len(sentences)
                return fallback
            return {
                "capacity_value": None,
                "capacity_unit": None,
                "confidence": "low",
                "source_quote": None,
                "extraction_method": "llm_rejected_no_quote",
                "num_candidates": len(sentences)
            }

    if result.get("capacity_value") is None:
        fallback = _fallback_extract_from_candidates(sentences)
        if fallback.get("capacity_value") is not None:
            fallback["extraction_method"] = "fallback_from_candidates"
            fallback["num_candidates"] = len(sentences)
            return fallback

    return result


# --------------------------
# PROJECT-LEVEL EXTRACTION
# --------------------------

def extract_capacity_for_project(
    project_id: str,
    project_title: str,
    project_type: str,
    documents_df: pd.DataFrame,
    pages_path: Path,
    model: str = DEFAULT_MODEL,
    max_pages: int = 25,
    verbose: bool = False
) -> dict:
    """
    Extract generation capacity for a single project.

    Args:
        project_id: Project identifier
        project_title: Project name (for context in LLM prompt)
        project_type: Project type (e.g., 'solar', 'wind')
        documents_df: DataFrame with document metadata (must have project_id column)
        pages_path: Path to pages.parquet file
        model: Ollama model to use
        max_pages: Maximum pages to scan per project
        verbose: Print progress

    Returns:
        dict with capacity_value, capacity_unit, confidence, source_quote, etc.
    """
    import pyarrow.parquet as pq

    result = {
        "project_id": project_id,
        "project_title": project_title,
        "project_type": project_type,
        "capacity_value": None,
        "capacity_unit": None,
        "confidence": "low",
        "source_quote": None,
        "extraction_method": None,
        "pages_scanned": 0,
        "candidates_found": 0,
    }

    # Check if this is a non-power project (e.g., pipeline)
    if is_non_power_project(project_type):
        result["extraction_method"] = "skipped_non_power"
        result["note"] = "Project type uses non-power metrics (volume, not MW)"
        return result

    # Get documents for this project
    project_docs = documents_df[documents_df['project_id'] == project_id]
    if project_docs.empty:
        result["extraction_method"] = "no_documents"
        return result

    # Prioritize by document type first (FEIS/DEIS have capacity info), then by main_document
    # ROD often just references the capacity, not the full description
    doc_type_priority = {'FEIS': 1, 'DEIS': 2, 'EA': 3, 'DEA': 4, 'FONSI': 5, 'ROD': 6, 'CE': 7, 'OTHER': 8, '': 5}
    project_docs = project_docs.copy()
    project_docs['_type_priority'] = project_docs['document_type'].map(lambda x: doc_type_priority.get(x, 10))
    project_docs['_main_priority'] = project_docs['main_document'].map({'YES': 0, 'NO': 1, '': 2})
    # Also prefer larger documents (more likely to have detail)
    def get_size_priority(x):
        try:
            pages = int(x) if pd.notna(x) else 0
            return 0 if pages > 50 else 1
        except (ValueError, TypeError):
            return 1
    project_docs['_size_priority'] = project_docs['total_pages'].apply(get_size_priority)
    project_docs = project_docs.sort_values(['_type_priority', '_main_priority', '_size_priority'])

    # Get capacity terms for this project type
    terms = get_terms_for_project_type(project_type)

    # Collect candidate sentences across documents
    all_candidates = []
    pages_scanned = 0
    docs_scanned = 0
    max_docs = 3  # Check up to 3 documents

    for _, doc in project_docs.iterrows():
        if pages_scanned >= max_pages or docs_scanned >= max_docs:
            break
        if len(all_candidates) >= 15:  # Found enough
            break

        doc_id = doc['document_id']
        docs_scanned += 1

        # Read pages for this document
        try:
            pages_table = pq.read_table(pages_path, filters=[('document_id', '=', doc_id)])
            doc_pages = pages_table.to_pandas()
        except Exception as e:
            if verbose:
                print(f"  Error reading pages for doc {doc_id}: {e}")
            continue

        # Sort by numeric page value first, then raw label for deterministic order.
        doc_pages['_page_number_num'] = doc_pages['page_number'].apply(extract_numeric_page_number)
        doc_pages = doc_pages.sort_values(['_page_number_num', 'page_number'])
        pages_to_check = min(50, len(doc_pages), max_pages - pages_scanned)

        for _, page in doc_pages.head(pages_to_check).iterrows():
            pages_scanned += 1
            text = page.get('page_text', '')

            candidates = extract_candidate_sentences(text, terms, max_sentences=5)
            all_candidates.extend(candidates)

            # If we have enough candidates, stop scanning this doc
            if len(all_candidates) >= 15:
                break

    result["pages_scanned"] = pages_scanned
    result["docs_scanned"] = docs_scanned
    result["candidates_found"] = len(all_candidates)

    if verbose:
        print(f"  Scanned {pages_scanned} pages, found {len(all_candidates)} candidate sentences")

    # Use LLM to extract capacity
    if all_candidates:
        llm_result = extract_capacity_with_llm(all_candidates, project_title, project_type, model=model)
        result.update(llm_result)
    else:
        result["extraction_method"] = "no_candidates"

    return result


# --------------------------
# BATCH EXTRACTION
# --------------------------

def _init_worker(docs_df, pages_path, model, verbose):
    """Initialize worker globals for multiprocessing."""
    global _WORKER_DOCS, _WORKER_PAGES_PATH, _WORKER_MODEL, _WORKER_VERBOSE
    _WORKER_DOCS = docs_df
    _WORKER_PAGES_PATH = pages_path
    _WORKER_MODEL = model
    _WORKER_VERBOSE = verbose


def _process_project_row(project):
    """Process a single project row (for multiprocessing)."""
    return extract_capacity_for_project(
        project_id=project['project_id'],
        project_title=project['project_title'],
        project_type=project['project_type'],
        documents_df=_WORKER_DOCS,
        pages_path=_WORKER_PAGES_PATH,
        model=_WORKER_MODEL,
        verbose=_WORKER_VERBOSE
    )

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
    Extract generation capacity for multiple projects.

    Args:
        source: Dataset source ('eis', 'ea', 'ce')
        clean_energy_only: Only process clean energy projects
        sample_size: Limit to N projects (for testing)
        model: Ollama model to use
        verbose: Print progress

    Returns:
        DataFrame with extraction results
    """
    print(f"\n=== Generation Capacity Extraction ({source.upper()}) ===")
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
    elif ambiguous_only and 'project_gencap_matches' in projects.columns:
        projects = projects.copy()
        projects['project_gencap_match_count'] = projects['project_gencap_matches'].apply(parse_match_count)
        projects = projects[projects['project_gencap_match_count'] >= 2]
        print(f"Ambiguous-only filter (2+ regex power matches): {len(projects):,} projects")
    elif ambiguous_only:
        print("Ambiguous-only filter requested, but project_gencap_matches not found; skipping ambiguity filter.")

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

    # Load documents
    docs_path = PROCESSED_DIR / source.lower() / "documents.parquet"
    documents_df = pd.read_parquet(docs_path)

    # Clean project_id in documents
    def extract_id(x):
        return x.get('value', '') if isinstance(x, dict) else x
    documents_df['project_id'] = documents_df['project_id'].apply(extract_id)

    pages_path = PROCESSED_DIR / source.lower() / "pages.parquet"

    project_records = projects[['project_id', 'project_title', 'project_type']].to_dict('records')

    results = []
    if workers and workers > 1:
        from multiprocessing import get_context
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(documents_df, pages_path, model, False),
        ) as pool:
            for idx, result in enumerate(pool.imap_unordered(_process_project_row, project_records, chunksize=1)):
                if verbose and idx % 10 == 0:
                    print(f"\nProcessed {idx + 1}/{len(project_records)} projects...")
                results.append(result)
    else:
        for idx, project in enumerate(project_records):
            if verbose and idx % 10 == 0:
                print(f"\nProcessing {idx + 1}/{len(project_records)}: {project['project_title'][:50]}...")

            result = extract_capacity_for_project(
                project_id=project['project_id'],
                project_title=project['project_title'],
                project_type=project['project_type'],
                documents_df=documents_df,
                pages_path=pages_path,
                model=model,
                verbose=verbose
            )
            results.append(result)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        results_df = pd.DataFrame(columns=[
            "project_id",
            "project_title",
            "project_type",
            "capacity_value",
            "capacity_unit",
            "confidence",
            "source_quote",
            "extraction_method",
            "pages_scanned",
            "candidates_found",
            "num_candidates",
            "parse_error",
            "llm_error",
        ])
    results_df["dataset_source"] = source.upper()
    run_timestamp_utc = pd.Timestamp.utcnow().isoformat()
    results_df["llm_run_completed_at_utc"] = run_timestamp_utc
    results_df["llm_model_used"] = model
    results_df["llm_trigger_mode"] = "regex_multi_candidate" if ambiguous_only else "candidate_sentences"

    # Summary
    print("\n=== Summary ===")
    has_capacity = results_df['capacity_value'].notna()
    print(f"Projects with capacity extracted: {has_capacity.sum()} / {len(results_df)} ({has_capacity.mean()*100:.1f}%)")
    print(f"Extraction methods: {results_df['extraction_method'].value_counts().to_dict()}")
    if 'pages_scanned' in results_df.columns:
        print(f"Average pages scanned: {results_df['pages_scanned'].mean():.1f}")
    if 'num_candidates' in results_df.columns:
        print(f"Average candidate sentences: {results_df['num_candidates'].mean():.1f}")

    return results_df


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

    The final capacity represents regex output overridden by validated LLM selections.
    """
    merged = regex_df.copy()
    source = source.upper()
    source_mask = merged["dataset_source"].astype(str).str.upper() == source

    # Ensure metadata columns exist and stamp this source run.
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
        "llm_capacity_value",
        "llm_capacity_unit",
        "llm_capacity_unit_norm",
        "llm_confidence",
        "llm_source_quote",
        "llm_extraction_method",
        "llm_pages_scanned",
        "llm_candidates_found",
        "llm_num_candidates",
        "llm_parse_error",
        "llm_error",
        "llm_run_completed_at_utc",
        "llm_model_used",
        "llm_trigger_mode",
        "project_gencap_llm_triggered",
        "project_gencap_llm_selected_from_regex_candidates",
        "project_gencap_llm_selection_logic",
        "project_gencap_llm_selected_value",
        "project_gencap_llm_selected_unit",
        "project_gencap_llm_selected_quote",
    ]
    for col in llm_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    # Clear prior LLM annotations for this source before applying current run.
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
                ("pages_scanned", "llm_pages_scanned"),
                ("candidates_found", "llm_candidates_found"),
                ("num_candidates", "llm_num_candidates"),
                ("parse_error", "llm_parse_error"),
                ("llm_error", "llm_error"),
                ("llm_run_completed_at_utc", "llm_run_completed_at_utc"),
                ("llm_model_used", "llm_model_used"),
                ("llm_trigger_mode", "llm_trigger_mode"),
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

            merged.drop(columns=["__key"], inplace=True, errors="ignore")

    llm_value_num = pd.to_numeric(merged.get("llm_capacity_value"), errors="coerce")
    llm_extraction_method = merged.get("llm_extraction_method").fillna("").astype(str)

    merged["llm_is_valid_power"] = (
        llm_value_num.notna()
        & (llm_value_num > 0)
        & merged.get("llm_capacity_unit_norm").isin(POWER_UNITS)
    )
    merged["llm_is_rejected_method"] = llm_extraction_method.isin(
        ["no_candidates", "no_numeric_candidates", "llm_rejected_no_quote", "llm_error", "llm_timeout"]
    )
    merged["llm_should_override_regex"] = merged["llm_is_valid_power"] & ~merged["llm_is_rejected_method"]

    # Final capacity defaults to regex values.
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

    llm_override_mask = source_mask & (merged["llm_should_override_regex"] == True)
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

    # Human-readable audit trail of how/why LLM selected (or did not select) a final value.
    llm_triggered = merged["project_gencap_llm_triggered"] == True
    llm_valid = merged["llm_is_valid_power"] == True
    llm_rejected = merged["llm_is_rejected_method"] == True
    llm_selected_regex = merged["project_gencap_llm_selected_from_regex_candidates"] == True

    merged.loc[source_mask, "project_gencap_llm_selection_logic"] = "not_triggered"
    merged.loc[source_mask & llm_triggered, "project_gencap_llm_selection_logic"] = "triggered_no_selection"
    merged.loc[
        source_mask
        & llm_triggered
        & llm_valid
        & llm_rejected,
        "project_gencap_llm_selection_logic",
    ] = "selected_valid_power_rejected_by_method"
    merged.loc[
        source_mask
        & llm_triggered
        & llm_valid
        & ~llm_rejected
        & llm_selected_regex,
        "project_gencap_llm_selection_logic",
    ] = "selected_regex_candidate"
    merged.loc[
        source_mask
        & llm_triggered
        & llm_valid
        & ~llm_rejected
        & ~llm_selected_regex,
        "project_gencap_llm_selection_logic",
    ] = "selected_non_regex_candidate"
    merged.loc[
        source_mask
        & llm_triggered
        & ~llm_valid,
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
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
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
    trigger_mode = "regex_multi_candidate" if ambiguous_only else "candidate_sentences"
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

    llm_save_path = Path(llm_output_path) if llm_output_path else (ANALYSIS_DIR / f"gencap_{source.lower()}_llm.parquet")
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
# CLI INTERFACE
# --------------------------

def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract generation capacity from NEPA documents using LLM"
    )
    parser.add_argument('--source', choices=['eis', 'ea', 'ce'], default='eis',
                        help='Dataset source (default: eis)')
    parser.add_argument('--sample', type=int, help='Sample N projects (for testing)')
    parser.add_argument('--all-projects', action='store_true',
                        help='Process all projects, not just clean energy')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'Ollama model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--regex-results', type=str,
                        help='Path to regex results (default: projects_gencap_{source}.parquet or projects_gencap.parquet)')
    parser.add_argument('--include-high', action='store_true',
                        help='Include high-confidence regex cases (legacy; no effect unless --only-low-medium is set)')
    parser.add_argument('--only-low-medium', action='store_true',
                        help='Also filter to low/medium confidence regex cases')
    parser.add_argument('--include-non-ambiguous', action='store_true',
                        help='Include projects without regex ambiguity (default is ambiguous-only: 2+ regex matches)')
    parser.add_argument('--require-regex-capacity', action='store_true',
                        help='Only sample projects with regex capacity values')
    parser.add_argument('--project-id', type=str,
                        help='Run extraction for a single project_id')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--run', action='store_true', help='Run extraction')
    parser.add_argument('--test', action='store_true', help='Test on 3 projects')
    parser.add_argument('--output', type=str,
                        help='Merged output path (default: overwrite regex input)')
    parser.add_argument('--llm-output', type=str,
                        help='Raw LLM output path (default: data/analysis/gencap_{source}_llm.parquet)')

    args = parser.parse_args()

    if args.test:
        print("Running test extraction on 10 projects...")
        results = extract_capacity_for_projects(
            source=args.source,
            sample_size=10,
            model=args.model,
            verbose=True,
            regex_results_path=args.regex_results,
            only_low_medium=args.only_low_medium and not args.include_high,
            ambiguous_only=not args.include_non_ambiguous,
            require_regex_capacity=args.require_regex_capacity,
            project_id=args.project_id,
            workers=args.workers
        )
        print("\n=== Results ===")
        print(results[['project_title', 'capacity_value', 'capacity_unit', 'confidence', 'source_quote']].to_string())

    elif args.run:
        _llm_results, merged, merged_path = run_llm_merge_pipeline(
            source=args.source,
            clean_energy_only=not args.all_projects,
            sample_size=args.sample,
            model=args.model,
            verbose=True,
            regex_results_path=args.regex_results,
            only_low_medium=args.only_low_medium and not args.include_high,
            ambiguous_only=not args.include_non_ambiguous,
            require_regex_capacity=args.require_regex_capacity,
            project_id=args.project_id,
            workers=args.workers,
            output_path=args.output,
            llm_output_path=args.llm_output,
        )
        source_mask = merged["dataset_source"].astype(str).str.upper() == args.source.upper()
        with_final = int((source_mask & merged["project_gencap_final_value"].notna()).sum())
        print(f"Rows with final capacity after merge ({args.source.upper()}): {with_final:,}")
        print(f"Merged output path: {merged_path}")

    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  python extract_gencap_llm.py --test                    # Quick test on 10 projects")
        print("  python extract_gencap_llm.py --source eis --sample 20  # Test on 20 EIS projects")
        print("  python extract_gencap_llm.py --source eis --run        # Full EIS extraction + merge")
        print("  python extract_gencap_llm.py --source eis --run --regex-results data/analysis/projects_gencap.parquet --workers 4")


if __name__ == "__main__":
    main()
