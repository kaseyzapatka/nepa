# --------------------------
# NEPA DATA EXTRACTION PIPELINE
# --------------------------
# Main script to extract, clean, and combine NEPA datasets
# Outputs analysis-ready parquet files

# --------------------------
# ENVIRONMENT SETUP
# --------------------------
#
# 1. HUGGING FACE SETUP
# --------------------------
# To access data from huggingface api, make sure there is a huggingface token installed on
# the local computer and connected to a huggingface account. Running the code below in the
# terminal will prompt entering the token:
#     hf auth login
#
# 2. LIBRARIES AND WORKING ENVIRONMENTS
# --------------------------
# Once connection to huggingface is established, the correct libraries and a working environment
# needs to be established. This can be accomplished running the code below in the terminal:
#     source [[filepath]]/setup_textanalysis.sh

# --------------------------
# LIBRARIES
# --------------------------
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# datasets is only needed for raw extraction from HuggingFace
# Import lazily to allow analysis mode to work without it
def load_dataset(*args, **kwargs):
    from datasets import load_dataset as _load_dataset
    return _load_dataset(*args, **kwargs)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract.classify_energy import add_energy_columns
from extract.parse_location import add_location_columns
import json

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)


# --------------------------
# FILE PATHS (RELATIVE TO THIS FILE)
# --------------------------

# this file: project/code/extract/extract_data.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # goes from extract/ -> code/ -> project/

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Notes directory for filter files
NOTES_DIR = BASE_DIR / "notes"


# --------------------------
# DEPARTMENT CLASSIFICATION
# --------------------------

def classify_department(lead_agency):
    """
    Classify lead agency into a department-level grouping.

    This mirrors the R code logic in deliverable1/02_agency.R.
    Collapses detailed agency names into high-level departments.

    Args:
        lead_agency: Lead agency string (may contain multiple values as JSON array or numpy array)

    Returns:
        str: Department name or "Other / Unclassified"
    """
    # Handle None
    if lead_agency is None:
        return "Other / Unclassified"

    # Handle numpy arrays - use first value
    if isinstance(lead_agency, np.ndarray):
        if len(lead_agency) == 0:
            return "Other / Unclassified"
        lead_agency = lead_agency[0]

    # Handle lists - use first value
    if isinstance(lead_agency, list):
        if len(lead_agency) == 0:
            return "Other / Unclassified"
        lead_agency = lead_agency[0]

    # Handle NaN
    if isinstance(lead_agency, float) and np.isnan(lead_agency):
        return "Other / Unclassified"

    # Handle JSON-encoded arrays - parse and use first value for classification
    if isinstance(lead_agency, str):
        if lead_agency.startswith('['):
            try:
                agencies = json.loads(lead_agency)
                if isinstance(agencies, list) and agencies:
                    lead_agency = agencies[0]  # Use first agency for classification
            except json.JSONDecodeError:
                pass

    # Handle empty strings
    if not lead_agency:
        return "Other / Unclassified"

    agency_str = str(lead_agency)

    # Department mappings (order matters - first match wins)
    if agency_str.startswith("Department of Energy"):
        return "Department of Energy"
    if agency_str.startswith("Department of the Interior"):
        return "Department of the Interior"
    if agency_str.startswith("Department of Agriculture"):
        return "Department of Agriculture"
    if agency_str.startswith("Department of Defense"):
        return "Department of Defense"
    if agency_str.startswith("Department of Homeland Security"):
        return "Department of Homeland Security"
    if agency_str.startswith("Department of Transportation"):
        return "Department of Transportation"
    if agency_str.startswith("Department of Health and Human Services"):
        return "Department of Health and Human Services"
    if agency_str.startswith("Department of Housing and Urban Development"):
        return "Department of Housing and Urban Development"
    if agency_str.startswith("Department of Commerce"):
        return "Department of Commerce"
    if agency_str.startswith("Department of State"):
        return "Department of State"
    if agency_str.startswith("Department of Justice"):
        return "Department of Justice"
    if agency_str.startswith("Department of Veterans Affairs"):
        return "Department of Veterans Affairs"
    if agency_str.startswith("Department of the Treasury"):
        return "Department of the Treasury"
    if agency_str.startswith("Major Independent Agencies"):
        return "Major Independent Agencies"
    if agency_str.startswith("Other Independent Agencies"):
        return "Other Independent Agencies"
    if agency_str.startswith("General Services Administration"):
        return "General Services Administration"
    if agency_str == "Legislative Branch":
        return "Legislative Branch"
    if agency_str == "International Assistance Programs":
        return "International Assistance Programs"

    return "Other / Unclassified"


def add_department_column(df):
    """
    Add project_department column to DataFrame.

    Args:
        df: DataFrame with 'lead_agency' column

    Returns:
        DataFrame with 'project_department' column added
    """
    df = df.copy()
    df['project_department'] = df['lead_agency'].apply(classify_department)
    return df


# --------------------------
# MULTI-VALUE FLAGS
# --------------------------

def parse_json_list(value):
    """
    Parse a JSON-encoded list or return the value as-is.

    Args:
        value: String (possibly JSON array), list, or other value

    Returns:
        list: Parsed list of values
    """
    if value is None:
        return []
    if isinstance(value, (list, np.ndarray)):
        return list(value)
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        if value.startswith('['):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        # Single value
        if value.strip():
            return [value]
    return []


def count_values(value):
    """
    Count the number of values in a field.

    Args:
        value: String (possibly JSON array), list, or other value

    Returns:
        int: Number of values
    """
    parsed = parse_json_list(value)
    return len(parsed)


def has_multiple_values(value):
    """
    Check if a field has multiple values.

    Args:
        value: String (possibly JSON array), list, or other value

    Returns:
        bool: True if multiple values exist
    """
    return count_values(value) > 1


def add_multi_value_flags(df):
    """
    Add flags for projects that span multiple states, counties, or departments.

    Adds columns:
    - project_multi_state: Boolean, True if project spans multiple states
    - project_multi_county: Boolean, True if project spans multiple counties
    - project_multi_department: Boolean, True if project has multiple lead agencies

    Args:
        df: DataFrame with project_state, project_county, and lead_agency columns

    Returns:
        DataFrame with multi-value flag columns added
    """
    df = df.copy()

    # Multi-state flag: check if project_state has multiple values
    df['project_multi_state'] = df['project_state'].apply(has_multiple_values)

    # Multi-county flag: check if project_county has multiple values
    df['project_multi_county'] = df['project_county'].apply(has_multiple_values)

    # Multi-department flag: check if lead_agency has multiple values
    # (Note: only ~40 of 61,881 projects have multiple agencies per R code)
    df['project_multi_department'] = df['lead_agency'].apply(has_multiple_values)

    return df


# --------------------------
# MILITARY & NUCLEAR WASTE FILTERS
# --------------------------
#
# FILTERING LOGIC SUMMARY:
# ========================
# The following filters reclassify projects from "Clean" to "Other" energy type:
#
# 1. UTILITIES + NON-ENERGY EXCLUSIONS (in classify_energy.py)
#    - Projects with ONLY "Utilities" + one of: Broadband, Waste Management,
#      Land Development (Housing, Other, Urban)
#    - Rationale: These are likely telecommunications or development projects
#      with utility connections, not clean energy projects
#
# 2. MILITARY / DEFENSE SITE EXCLUSIONS
#    - Projects with both "Military and Defense" AND "Nuclear" project_type tags
#    - Loaded from: notes/military_project_ids_to_filter.csv
#    - Rationale: Defense-related nuclear projects are not civilian clean energy
#
# 3. NUCLEAR WASTE MANAGEMENT EXCLUSIONS
#    - Projects with both "Waste Management" AND "Nuclear" project_type tags
#    - AND associated with NNSA, Office of Legacy Management, or Office of
#      Environmental Management (based on lead_agency, project_sponsor, OR project_title)
#    - Exclusion terms loaded from: notes/agencies_to_be_excluded.txt
#    - Rationale: Nuclear waste cleanup/storage is not clean energy production
#    - Title-based filtering added to catch projects where sponsor field is missing
#      (e.g., "Hanford Site" appearing only in the title)
#
# The exclusion terms file (agencies_to_be_excluded.txt) includes:
#    - NNSA sites: Los Alamos, Sandia, Livermore, Pantex, Y-12, Savannah River, etc.
#    - Legacy Management: Albuquerque Operations Office, Grand Junction Office
#    - Environmental Management: Hanford, Richland, Oak Ridge, WIPP, etc.


def load_exclusion_terms():
    """
    Load exclusion terms from notes/agencies_to_be_excluded.txt.

    These terms identify projects associated with NNSA, Office of Legacy Management,
    or Office of Environmental Management that should not be classified as clean energy
    when combined with Nuclear + Waste Management tags.

    Returns:
        list: List of exclusion term strings (lowercase, stripped)
    """
    txt_path = NOTES_DIR / "agencies_to_be_excluded.txt"
    if not txt_path.exists():
        print(f"  Warning: {txt_path} not found, using default patterns")
        return []

    try:
        terms = []
        with open(txt_path, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                line = line.strip()
                if line and not line.startswith('#'):
                    terms.append(line.lower())
        return terms
    except Exception as e:
        print(f"  Warning: Error loading exclusion terms: {e}")
        return []


def load_military_project_ids():
    """
    Load list of military project IDs that should be filtered from clean energy.

    These are projects with both "Military and Defense" AND "Nuclear" tags,
    which are defense-related nuclear projects, not clean energy.

    Returns:
        set: Set of project IDs to filter
    """
    csv_path = NOTES_DIR / "military_project_ids_to_filter.csv"
    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found")
        return set()

    try:
        df = pd.read_csv(csv_path)
        return set(df['project_id'].dropna().astype(str))
    except Exception as e:
        print(f"  Warning: Error loading military project IDs: {e}")
        return set()


def is_nuclear_waste_project(project_type, project_sponsor, lead_agency, project_title,
                              exclusion_terms=None):
    """
    Check if a project is a nuclear waste project that should not be classified as clean energy.

    Identifies projects that have both "Waste Management" AND "Nuclear" tags,
    AND are associated with NNSA, Legacy Management, or Environmental Management
    based on lead_agency, project_sponsor, OR project_title.

    Title-based filtering catches cases where projects slipped through because
    the sponsor field was missing (e.g., "Hanford Site" appearing only in the title).

    Args:
        project_type: Project type string (may be JSON array or numpy array)
        project_sponsor: Project sponsor string
        lead_agency: Lead agency string
        project_title: Project title string
        exclusion_terms: List of exclusion terms (loaded from agencies_to_be_excluded.txt)
                        If None, will load from file.

    Returns:
        bool: True if this is a nuclear waste project to filter
    """
    # Parse project_type - handle None, NaN, arrays, and strings
    if project_type is None:
        types_str = ""
    elif isinstance(project_type, (list, np.ndarray)):
        types_str = " ".join(str(t) for t in project_type).lower()
    elif isinstance(project_type, float) and np.isnan(project_type):
        types_str = ""
    else:
        types_str = str(project_type).lower()

    # Must have both "Waste Management" AND "Nuclear" tags
    has_waste_management = "waste management" in types_str
    has_nuclear = "nuclear" in types_str

    if not (has_waste_management and has_nuclear):
        return False

    # Convert fields to strings for pattern matching
    # Handle None, NaN, arrays, and strings
    def to_lower_str(val):
        if val is None:
            return ""
        if isinstance(val, (list, np.ndarray)):
            return " ".join(str(v) for v in val).lower()
        if isinstance(val, float) and np.isnan(val):
            return ""
        return str(val).lower()

    sponsor_str = to_lower_str(project_sponsor)
    agency_str = to_lower_str(lead_agency)
    title_str = to_lower_str(project_title)

    # Use provided exclusion terms or load from file
    if exclusion_terms is None:
        exclusion_terms = load_exclusion_terms()

    # If exclusion terms loaded successfully, use them
    if exclusion_terms:
        # Check if any exclusion term appears in sponsor, agency, OR title
        for term in exclusion_terms:
            if term in sponsor_str or term in agency_str or term in title_str:
                return True
    else:
        # Fallback to hardcoded patterns if file not available
        fallback_patterns = [
            "nnsa", "national nuclear security administration",
            "kansas city", "livermore", "lawrence livermore", "los alamos",
            "naval nuclear", "nevada", "pantex", "sandia", "savannah river", "y-12",
            "office of legacy management", "legacy management",
            "albuquerque operations office", "grand junction office",
            "office of environmental management", "environmental management",
            "hanford", "richland", "office of river protection",
            "paducah", "portsmouth", "oak ridge", "waste isolation pilot plant", "carlsbad",
        ]
        for pattern in fallback_patterns:
            if pattern in sponsor_str or pattern in agency_str or pattern in title_str:
                return True

    # Also check for specific lead agency matches (always check these)
    agency_specific = [
        "national nuclear security administration",
        "office of legacy management",
        "office of environmental management",
    ]
    for pattern in agency_specific:
        if pattern in agency_str:
            return True

    return False


def apply_energy_type_filters(df):
    """
    Apply military and nuclear waste filters to reclassify projects as "Other".

    Projects that were initially classified as "Clean" but match these filters
    will be reclassified as "Other" since they are not clean energy projects.

    Adds columns:
    - project_is_military_nuclear: Boolean flag for military nuclear projects
    - project_is_nuclear_waste: Boolean flag for nuclear waste projects

    Modifies:
    - project_energy_type: Updates from "Clean" to "Other" for filtered projects
    - project_energy_type_strict: Updates from "Clean" to "Other" for filtered projects

    Args:
        df: DataFrame with project_id, project_type, project_sponsor, lead_agency,
            project_title, project_energy_type, and project_energy_type_strict columns

    Returns:
        DataFrame with filters applied
    """
    df = df.copy()

    # Load military project IDs
    print("  Loading military project filter...")
    military_ids = load_military_project_ids()
    print(f"    Found {len(military_ids)} military project IDs to filter")

    # Flag military nuclear projects
    df['project_is_military_nuclear'] = df['project_id'].astype(str).isin(military_ids)

    # Load exclusion terms once for nuclear waste filtering
    print("  Loading nuclear waste exclusion terms...")
    exclusion_terms = load_exclusion_terms()
    print(f"    Loaded {len(exclusion_terms)} exclusion terms from agencies_to_be_excluded.txt")

    # Flag nuclear waste projects
    # Checks project_type for Nuclear + Waste Management tags, then checks
    # lead_agency, project_sponsor, AND project_title for exclusion terms
    print("  Identifying nuclear waste projects...")
    df['project_is_nuclear_waste'] = df.apply(
        lambda row: is_nuclear_waste_project(
            row.get('project_type'),
            row.get('project_sponsor'),
            row.get('lead_agency'),
            row.get('project_title'),
            exclusion_terms
        ),
        axis=1
    )

    nuclear_waste_count = df['project_is_nuclear_waste'].sum()
    print(f"    Found {nuclear_waste_count} nuclear waste projects to filter")

    # Reclassify filtered projects from "Clean" to "Other"
    filter_mask = df['project_is_military_nuclear'] | df['project_is_nuclear_waste']
    clean_mask = df['project_energy_type'] == 'Clean'
    reclassify_mask = filter_mask & clean_mask

    reclassified_count = reclassify_mask.sum()
    print(f"  Reclassifying {reclassified_count} projects from 'Clean' to 'Other'")

    df.loc[reclassify_mask, 'project_energy_type'] = 'Other'
    df.loc[reclassify_mask, 'project_energy_type_strict'] = 'Other'

    return df


# --------------------------
# EXTRACTION FUNCTIONS
# --------------------------

def load_nepa_dataset(dataset_type):
    """
    Load NEPA dataset from HuggingFace.

    Args:
        dataset_type: "EA", "EIS", or "CE"

    Returns:
        HuggingFace dataset
    """
    ds = load_dataset(
        "PNNL/NEPATEC2.0",
        data_files=f"{dataset_type.upper()}/*/*.jsonl",
        split="train"
    )
    return ds


def extract_projects(ds):
    """Extract project-level metadata from dataset."""
    projects = []
    for item in ds:
        p = item["project"]
        projects.append({
            "project_id": p.get("project_ID"),
            "project_title": p.get("project_title", {}).get("value", ""),
            "project_sector": p.get("project_sector", {}).get("value", ""),
            "project_type": p.get("project_type", {}).get("value", ""),
            "project_description": p.get("project_description", {}).get("value", ""),
            "project_sponsor": p.get("project_sponsor", {}).get("value", ""),
            "project_location": p.get("location", {}).get("value", "")
        })
    return pd.DataFrame(projects)


def extract_processes(ds):
    """Extract process-level metadata from dataset."""
    rows = []
    for item in ds:
        process = item["process"]
        rows.append({
            "project_id": item["project"]["project_ID"],
            "process_family": process.get("process_family", {}).get("value", ""),
            "process_type": process.get("process_type", {}).get("value", ""),
            "lead_agency": process.get("lead_agency", {}).get("value", "")
        })
    return pd.DataFrame(rows)


def extract_documents(ds):
    """Extract document-level metadata from dataset."""
    docs = []
    for item in ds:
        pid = item["project"]["project_ID"]
        for doc in item["documents"]:
            meta = doc.get("metadata", {})
            doc_meta = meta.get("document_metadata", {})
            file_meta = meta.get("file_metadata", {})

            docs.append({
                "project_id": pid,
                "document_id": doc_meta.get("document_ID", {}).get("value", ""),
                "document_type": doc_meta.get("document_type", {}).get("value", ""),
                "document_title": doc_meta.get("document_title", {}).get("value", ""),
                "prepared_by": doc_meta.get("prepared_by", {}).get("value", ""),
                "ce_category": doc_meta.get("ce_category", {}).get("value", ""),
                "file_id": file_meta.get("file_ID", {}).get("value", ""),
                "file_name": file_meta.get("file_name", {}).get("value", ""),
                "total_pages": file_meta.get("total_pages", {}).get("value", ""),
                "main_document": file_meta.get("main_document", {}).get("value", "")
            })
    return pd.DataFrame(docs)


def extract_pages(ds):
    """Extract page-level data from dataset."""
    pages = []
    for item in ds:
        pid = item["project"]["project_ID"]
        for doc in item["documents"]:
            doc_meta = doc["metadata"]["document_metadata"]
            doc_id = doc_meta["document_ID"]["value"]
            for pg in doc.get("pages", []):
                pages.append({
                    "document_id": doc_id,
                    "page_number": pg.get("page number"),
                    "page_text": pg.get("page text", "")
                })
    return pd.DataFrame(pages)


# --------------------------
# RAW EXTRACTION PIPELINE
# --------------------------

def run_raw_extraction(dataset_type):
    """
    Extract raw data from HuggingFace and save to processed parquet files.

    Args:
        dataset_type: "EA", "EIS", or "CE"

    Creates:
        data/processed/{dataset_type}/projects.parquet
                                      processes.parquet
                                      documents.parquet
                                      pages.parquet
    """
    print(f"\n=== Processing {dataset_type} dataset ===")

    out_dir = PROCESSED_DIR / dataset_type.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_nepa_dataset(dataset_type)

    projects_df = extract_projects(ds)
    processes_df = extract_processes(ds)
    documents_df = extract_documents(ds)
    pages_df = extract_pages(ds)

    projects_df.to_parquet(out_dir / "projects.parquet")
    processes_df.to_parquet(out_dir / "processes.parquet")
    documents_df.to_parquet(out_dir / "documents.parquet")
    pages_df.to_parquet(out_dir / "pages.parquet")

    print(f"  Saved to {out_dir}")
    print(f"  Projects: {len(projects_df):,}")
    print(f"  Documents: {len(documents_df):,}")
    print(f"  Pages: {len(pages_df):,}")


# --------------------------
# ANALYSIS DATA PIPELINE
# --------------------------

def load_processed_data(dataset_type):
    """Load previously extracted parquet files."""
    data_dir = PROCESSED_DIR / dataset_type.lower()

    return {
        'projects': pd.read_parquet(data_dir / "projects.parquet"),
        'processes': pd.read_parquet(data_dir / "processes.parquet"),
        'documents': pd.read_parquet(data_dir / "documents.parquet"),
    }


def clean_project_id(df):
    """
    Clean project_id column - extract string value from dict format.

    The raw data has project_id as {'value': 'xxx'}, convert to just 'xxx'.
    """
    df = df.copy()

    def extract_id(x):
        if isinstance(x, dict):
            return x.get('value', '')
        return x

    df['project_id'] = df['project_id'].apply(extract_id)
    return df


def convert_complex_columns_for_parquet(df):
    """
    Convert complex columns (lists, arrays, dicts) to JSON strings for parquet compatibility.

    Parquet has trouble with mixed-type object columns containing numpy arrays.
    Converting to JSON strings ensures consistent serialization.
    """
    import json
    df = df.copy()

    def to_json_safe(x):
        if x is None:
            return None
        if isinstance(x, (list, np.ndarray)):
            # Convert numpy array to list for JSON serialization
            if isinstance(x, np.ndarray):
                x = x.tolist()
            return json.dumps(x)
        if isinstance(x, dict):
            return json.dumps(x)
        return x

    # Columns that may contain complex types
    complex_cols = [
        'project_sector', 'project_type', 'project_description',
        'project_sponsor', 'project_location', 'lead_agency',
        'project_state', 'project_county'
    ]

    for col in complex_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_json_safe)

    return df


def create_combined_projects():
    """
    Create a combined projects dataset with all derived variables.

    Combines EA, EIS, CE datasets and adds:
    - process_type column (EA, EIS, CE)
    - Energy classification columns (project_energy_type, project_type_count, etc.)
    - Military/nuclear waste filters (project_is_military_nuclear, project_is_nuclear_waste)
    - Location columns (project_state, project_county, project_lat, project_lon)
    - Department classification (project_department)
    - Multi-value flags (project_multi_state, project_multi_county, project_multi_department)
    - Document flags (project_has_decision_doc, project_has_final_doc, etc.)

    Outputs:
        data/analysis/projects_combined.parquet
    """
    print("\n=== Creating combined projects dataset ===")

    all_projects = []

    for dataset_type in ["ea", "eis", "ce"]:
        print(f"\nProcessing {dataset_type.upper()}...")

        data = load_processed_data(dataset_type)
        projects = data['projects']
        processes = data['processes']

        # Clean project_id format
        projects = clean_project_id(projects)
        processes = clean_project_id(processes)

        # Merge to get process_type and lead_agency
        projects = projects.merge(
            processes[['project_id', 'process_type', 'lead_agency']],
            on='project_id',
            how='left'
        )

        # Add source dataset indicator
        projects['dataset_source'] = dataset_type.upper()

        # Standardize process_type values
        process_type_map = {
            'Environmental Assessment (EA)': 'EA',
            'Environmental Impact Statement (EIS)': 'EIS',
            'Categorical Exclusion': 'CE',
        }
        projects['process_type'] = projects['process_type'].map(
            lambda x: process_type_map.get(x, dataset_type.upper())
        )

        print(f"  Loaded {len(projects):,} projects")
        all_projects.append(projects)

    # Combine all datasets
    combined = pd.concat(all_projects, ignore_index=True)
    print(f"\nTotal combined projects: {len(combined):,}")

    # Add energy classification columns
    print("Adding energy classification...")
    combined = add_energy_columns(combined)

    # Apply military and nuclear waste filters
    print("Applying military/nuclear waste filters...")
    combined = apply_energy_type_filters(combined)

    # Add location columns
    print("Adding location columns...")
    combined = add_location_columns(combined)

    # Add department classification
    print("Adding department classification...")
    combined = add_department_column(combined)

    # Add multi-value flags (multi-state, multi-county, multi-department)
    print("Adding multi-value flags...")
    combined = add_multi_value_flags(combined)

    # Add document flags (requires loading documents)
    print("Adding document type flags...")
    all_documents = []
    for dataset_type in ["ea", "eis", "ce"]:
        data = load_processed_data(dataset_type)
        documents = clean_project_id(data['documents'])
        all_documents.append(documents)
    documents_combined = pd.concat(all_documents, ignore_index=True)
    doc_flags = get_project_document_flags(documents_combined)
    combined = combined.merge(doc_flags, on='project_id', how='left')

    # Fill missing flags with False/0
    combined['project_has_decision_doc'] = combined['project_has_decision_doc'].fillna(False)
    combined['project_has_final_doc'] = combined['project_has_final_doc'].fillna(False)
    combined['project_has_draft_doc'] = combined['project_has_draft_doc'].fillna(False)
    combined['project_doc_count'] = combined['project_doc_count'].fillna(0).astype(int)

    # Convert complex columns for parquet compatibility
    print("Converting complex columns for parquet...")
    combined = convert_complex_columns_for_parquet(combined)

    # Save to analysis directory
    output_path = ANALYSIS_DIR / "projects_combined.parquet"
    combined.to_parquet(output_path)
    print(f"\nSaved to: {output_path}")

    # Print summary stats
    print("\n=== Summary Statistics ===")
    print(f"Total projects: {len(combined):,}")
    print(f"\nBy process type:")
    print(combined['process_type'].value_counts())
    print(f"\nBy energy type:")
    print(combined['project_energy_type'].value_counts())
    print(f"\nProjects flagged for review: {combined['project_energy_type_questions'].sum():,}")
    print(f"\nFiltered projects:")
    print(f"  Military nuclear projects: {combined['project_is_military_nuclear'].sum():,}")
    print(f"  Nuclear waste projects: {combined['project_is_nuclear_waste'].sum():,}")
    print(f"\nMulti-value projects:")
    print(f"  Multi-state: {combined['project_multi_state'].sum():,}")
    print(f"  Multi-county: {combined['project_multi_county'].sum():,}")
    print(f"  Multi-department: {combined['project_multi_department'].sum():,}")
    print(f"\nBy department:")
    print(combined['project_department'].value_counts())
    print(f"\nDocument availability:")
    print(f"  Projects with decision docs: {combined['project_has_decision_doc'].sum():,}")
    print(f"  Projects with final docs: {combined['project_has_final_doc'].sum():,}")
    print(f"  Projects with draft docs: {combined['project_has_draft_doc'].sum():,}")

    return combined


def create_combined_processes():
    """
    Create a combined processes dataset.

    Outputs:
        data/analysis/processes_combined.parquet
    """
    print("\n=== Creating combined processes dataset ===")

    all_processes = []

    for dataset_type in ["ea", "eis", "ce"]:
        data = load_processed_data(dataset_type)
        processes = clean_project_id(data['processes'])
        processes['dataset_source'] = dataset_type.upper()
        all_processes.append(processes)

    combined = pd.concat(all_processes, ignore_index=True)

    output_path = ANALYSIS_DIR / "processes_combined.parquet"
    combined.to_parquet(output_path)
    print(f"Saved {len(combined):,} processes to: {output_path}")

    return combined


# --------------------------
# DOCUMENT TYPE CLASSIFICATION
# --------------------------

# Document type categories for filtering timeline extraction
DOCUMENT_TYPE_CATEGORIES = {
    'decision': ['ROD', 'FONSI', 'CE'],  # Decision documents - primary source for timelines
    'final': ['FEIS', 'EA'],              # Final documents (EA can be final)
    'draft': ['DEIS', 'DEA'],             # Draft documents
    'other': ['OTHER', ''],               # Other/unknown documents
}


def classify_document_type(doc_type):
    """
    Classify a document_type into a category.

    Args:
        doc_type: The document_type value (e.g., 'ROD', 'FEIS', 'DEIS')

    Returns:
        str: Category name ('decision', 'final', 'draft', 'other')
    """
    if pd.isna(doc_type) or doc_type == '':
        return 'other'

    doc_type_upper = str(doc_type).upper().strip()

    for category, types in DOCUMENT_TYPE_CATEGORIES.items():
        if doc_type_upper in types:
            return category

    return 'other'


def add_document_type_category(df):
    """
    Add document_type_category column to documents dataframe.

    Args:
        df: Documents dataframe with document_type column

    Returns:
        DataFrame with document_type_category column added
    """
    df = df.copy()
    df['document_type_category'] = df['document_type'].apply(classify_document_type)
    return df


def get_project_document_flags(documents_df):
    """
    Create project-level flags for document types.

    Args:
        documents_df: Documents dataframe with project_id and document_type columns

    Returns:
        DataFrame with project_id and document flag columns
    """
    # Ensure document_type_category exists
    if 'document_type_category' not in documents_df.columns:
        documents_df = add_document_type_category(documents_df)

    # Group by project and check for each category
    project_flags = documents_df.groupby('project_id').agg(
        project_has_decision_doc=('document_type_category', lambda x: 'decision' in x.values),
        project_has_final_doc=('document_type_category', lambda x: 'final' in x.values),
        project_has_draft_doc=('document_type_category', lambda x: 'draft' in x.values),
        project_doc_count=('document_id', 'count'),
    ).reset_index()

    return project_flags


def convert_documents_for_parquet(df):
    """Convert complex columns in documents dataframe for parquet compatibility."""
    import json
    df = df.copy()

    def to_json_safe(x):
        if x is None:
            return None
        if isinstance(x, (list, np.ndarray)):
            if isinstance(x, np.ndarray):
                x = x.tolist()
            return json.dumps(x)
        if isinstance(x, dict):
            return json.dumps(x)
        return x

    # Document columns that may contain complex types
    complex_cols = ['prepared_by', 'ce_category']

    for col in complex_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_json_safe)

    return df


def create_combined_documents():
    """
    Create a combined documents dataset.

    Outputs:
        data/analysis/documents_combined.parquet
    """
    print("\n=== Creating combined documents dataset ===")

    all_documents = []

    for dataset_type in ["ea", "eis", "ce"]:
        data = load_processed_data(dataset_type)
        documents = clean_project_id(data['documents'])
        documents['dataset_source'] = dataset_type.upper()
        all_documents.append(documents)

    combined = pd.concat(all_documents, ignore_index=True)

    # Add document type category
    print("Adding document type categories...")
    combined = add_document_type_category(combined)

    # Convert complex columns for parquet compatibility
    combined = convert_documents_for_parquet(combined)

    output_path = ANALYSIS_DIR / "documents_combined.parquet"
    combined.to_parquet(output_path)
    print(f"Saved {len(combined):,} documents to: {output_path}")

    # Print document type category stats
    print("\nDocument type categories:")
    print(combined['document_type_category'].value_counts())

    return combined


# --------------------------
# MAIN ENTRY POINTS
# --------------------------

def run_full_extraction():
    """
    Run full extraction from HuggingFace (slow, requires HF token).
    Use this only when you need to refresh the raw data.
    """
    for dataset in ["EA", "EIS", "CE"]:
        run_raw_extraction(dataset)
    print("\nAll datasets extracted.")


def run_analysis_pipeline():
    """
    Create analysis-ready datasets from existing processed data.
    This is the main entry point for creating datasets for deliverables.
    """
    create_combined_projects()
    create_combined_processes()
    create_combined_documents()
    print("\nAnalysis pipeline complete.")


# --------------------------
# MAIN
# --------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NEPA Data Extraction Pipeline")
    parser.add_argument(
        "--mode",
        choices=["extract", "analysis", "all"],
        default="analysis",
        help="extract: Download from HuggingFace; analysis: Create combined datasets; all: Both"
    )

    args = parser.parse_args()

    if args.mode == "extract":
        run_full_extraction()
    elif args.mode == "analysis":
        run_analysis_pipeline()
    elif args.mode == "all":
        run_full_extraction()
        run_analysis_pipeline()
