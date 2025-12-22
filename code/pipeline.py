# --------------------------
# ENVIRONMENT SETUP
# --------------------------

#
# 1. HUGGING FACE SETUP
# --------------------------
# To access data from huggingface api, make sure there is a huggingface token installed on 
# the local computer and connected to a huggingface account. Running the code below in the 
# terminal will print entering the token: 
    
    # hf auth login 

#
# 2. LIBRARIES AND WORKING ENVIRONMENTS
# --------------------------
# Once connection to huggingface is established, the correct libraries and a working environment
# needs to be established. This can be accomplished runnning the code below in the terminal
# to accomplish this:

    # source [[filepath]]/setup_textanalysis.sh

# --------------------------
# LIBRARIES
# --------------------------
from datasets import load_dataset
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)


# --------------------------
# FILE PATHS (RELATIVE TO THIS FILE)
# --------------------------

# this file: project/code/pipeline.py
BASE_DIR = Path(__file__).resolve().parent.parent   # goes from code/ → project/

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------
# FUNCTIONS
# --------------------------

def load_nepa_dataset(dataset_type):
    """
    dataset_type ∈ {"EA", "EIS", "CE"}
    Returns HuggingFace dataset.
    """
    ds = load_dataset(
        "PNNL/NEPATEC2.0",
        data_files=f"{dataset_type.upper()}/*/*.jsonl",
        split="train"
    )
    return ds


def extract_projects(ds):
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


def run_pipeline(dataset_type):
    """
    dataset_type = "EA", "EIS", or "CE"
    Creates:
      data/processed/{dataset_type}/projects.parquet
                                   processes.parquet
                                   documents.parquet
                                   pages.parquet
    """
    print(f"\n=== Processing {dataset_type} dataset ===")

    out_dir = PROCESSED_DIR / dataset_type
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

    print(f"✔ Finished {dataset_type}")


# --------------------------
# MAIN RUN
# --------------------------

if __name__ == "__main__":
    for dataset in ["EA", "EIS", "CE"]:
        run_pipeline(dataset)

    print("\nAll datasets processed.")
