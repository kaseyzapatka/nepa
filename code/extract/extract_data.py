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
    - Energy classification columns
    - Location columns

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

    # Add location columns
    print("Adding location columns...")
    combined = add_location_columns(combined)

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
