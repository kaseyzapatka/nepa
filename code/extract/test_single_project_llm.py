#!/usr/bin/env python3
"""
Test LLM timeline extraction on a single project.
Use this to manually validate regex results vs LLM extraction.

Usage:
    # Test by project title (partial match)
    python test_single_project_llm.py --title "Coldfoot"

    # Test by project ID
    python test_single_project_llm.py --id "6e9fc6608e30977c74305d2a98628a13"

    # Use a different model
    python test_single_project_llm.py --title "Coldfoot" --model qwen2:7b

    # Just show regex results (no LLM)
    python test_single_project_llm.py --title "Coldfoot" --regex-only
"""

import pandas as pd
import json
import requests
import sys
from pathlib import Path
import argparse
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"

DEFAULT_MODEL = "llama3.2:latest"


# Document type category mapping
DOCUMENT_TYPE_CATEGORIES = {
    'decision': ['ROD', 'FONSI', 'CE'],
    'final': ['FEIS', 'EA'],
    'draft': ['DEIS', 'DEA'],
    'other': ['OTHER', ''],
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


def get_project_pages(project_id, source, max_pages=15, decision_docs_only=True):
    """Load pages for a project, prioritizing decision documents."""
    data_dir = PROCESSED_DIR / source.lower()

    docs_df = pd.read_parquet(data_dir / "documents.parquet")
    pages_df = pd.read_parquet(data_dir / "pages.parquet")

    def extract_id(x):
        return x.get('value', '') if isinstance(x, dict) else x

    docs_df['project_id'] = docs_df['project_id'].apply(extract_id)
    docs_df['document_type_category'] = docs_df['document_type'].apply(classify_document_type)

    project_docs = docs_df[docs_df['project_id'] == project_id]

    # Prioritize document types: decision > final > draft > other
    doc_ids = []
    docs_used = 'none'
    if decision_docs_only:
        for category in ['decision', 'final', 'draft', 'other']:
            category_docs = project_docs[project_docs['document_type_category'] == category]
            if not category_docs.empty:
                doc_ids = category_docs['document_id'].tolist()
                docs_used = category
                break
    else:
        # Use main documents or all documents
        main_docs = project_docs[project_docs['main_document'] == 'YES']
        if main_docs.empty:
            main_docs = project_docs
        doc_ids = main_docs['document_id'].tolist()
        docs_used = 'all'

    project_pages = pages_df[pages_df['document_id'].isin(doc_ids)]

    print(f"  Document source: {docs_used} ({len(doc_ids)} documents)")

    return project_pages.head(max_pages)['page_text'].tolist()


def call_ollama(prompt, model=DEFAULT_MODEL, timeout=180):
    """Call Ollama API."""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.1}
            },
            timeout=timeout
        )

        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            print(f"Ollama error: {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        print("\nERROR: Ollama is not running!")
        print("Start it with: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print(f"\nTimeout after {timeout}s - try a smaller model")
        return None


def create_prompt(page_texts, max_chars=8000):
    """Create extraction prompt."""
    combined = "\n\n---PAGE---\n\n".join(page_texts)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n\n[...truncated...]"

    return f"""Analyze this NEPA document and extract the project timeline.

IMPORTANT: Only extract dates for THIS specific project. Ignore:
- Law/act years (e.g., "Clean Air Act of 1970")
- Reference dates from other projects
- General regulation dates

Document:
{combined}

Return JSON:
{{
  "dates": [
    {{"date": "YYYY-MM-DD", "type": "notice|draft|final|decision|comment|scoping|other", "context": "quote"}}
  ],
  "project_start_date": "YYYY-MM-DD",
  "decision_date": "YYYY-MM-DD",
  "project_year": YYYY
}}

Return ONLY JSON, no other text."""


def main():
    parser = argparse.ArgumentParser(description="Test LLM timeline extraction on a single project")
    parser.add_argument('--title', type=str, help='Project title (partial match)')
    parser.add_argument('--id', type=str, help='Project ID')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help=f'Ollama model (default: {DEFAULT_MODEL})')
    parser.add_argument('--regex-only', action='store_true', help='Only show regex results, skip LLM')

    args = parser.parse_args()

    if not args.title and not args.id:
        parser.print_help()
        print("\nExample: python test_single_project_llm.py --title 'Coldfoot'")
        return

    # Load timeline data
    timeline_path = ANALYSIS_DIR / "projects_timeline.parquet"
    if not timeline_path.exists():
        print(f"ERROR: {timeline_path} not found")
        print("Run extract_timeline.py first")
        return

    projects = pd.read_parquet(timeline_path)

    # Find project
    if args.id:
        matches = projects[projects['project_id'] == args.id]
    else:
        matches = projects[projects['project_title'].str.contains(args.title, case=False, na=False)]

    if matches.empty:
        print(f"No project found matching: {args.title or args.id}")
        return

    project = matches.iloc[0]

    print("=" * 80)
    print("PROJECT INFO")
    print("=" * 80)
    print(f"Title: {project['project_title']}")
    print(f"ID: {project['project_id']}")
    print(f"Process Type: {project['process_type']}")
    print(f"Dataset: {project['dataset_source']}")
    print(f"Energy Type: {project['project_energy_type']}")

    print("\n" + "=" * 80)
    print("REGEX EXTRACTION RESULTS")
    print("=" * 80)
    print(f"Earliest: {project['project_date_earliest']}")
    print(f"Latest: {project['project_date_latest']}")
    print(f"Decision: {project['project_date_decision']}")
    print(f"Duration: {project['project_duration_days']} days")
    print(f"Year: {project['project_year']}")
    print(f"Needs Review: {project['project_timeline_needs_review']}")
    print(f"Review Reasons: {project['project_timeline_review_reasons']}")

    dates = json.loads(project['project_dates']) if pd.notna(project['project_dates']) else []
    contexts = json.loads(project['project_date_contexts']) if pd.notna(project['project_date_contexts']) else []

    print(f"\nDates extracted ({len(dates)}):")
    for d, c in zip(dates, contexts):
        print(f"  {d} ({c})")

    if args.regex_only:
        return

    # LLM extraction
    print("\n" + "=" * 80)
    print(f"LLM EXTRACTION (model: {args.model})")
    print("=" * 80)

    print("\nLoading document pages...")
    page_texts = get_project_pages(project['project_id'], project['dataset_source'])

    if not page_texts:
        print("No pages found!")
        return

    print(f"Loaded {len(page_texts)} pages")
    print(f"Calling Ollama ({args.model})... this may take 30-120 seconds...")

    prompt = create_prompt(page_texts)
    start = time.time()
    response = call_ollama(prompt, model=args.model)
    elapsed = time.time() - start

    if response:
        print(f"\nCompleted in {elapsed:.1f}s")
        print("\n--- LLM Raw Response ---")
        print(response[:2000])
        if len(response) > 2000:
            print("...[truncated]")

        # Try to parse JSON
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                print("\n--- Parsed LLM Results ---")
                print(f"Project Year: {parsed.get('project_year')}")
                print(f"Decision Date: {parsed.get('decision_date')}")
                print(f"Start Date: {parsed.get('project_start_date')}")
                print(f"Dates found: {len(parsed.get('dates', []))}")
        except:
            print("\nCould not parse JSON from response")
    else:
        print("\nNo response from LLM")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Regex Year: {project['project_year']}")
    if response:
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            parsed = json.loads(response[start:end])
            print(f"LLM Year: {parsed.get('project_year')}")
            match = project['project_year'] == parsed.get('project_year')
            print(f"Match: {'YES' if match else 'NO'}")
        except:
            print("LLM Year: Could not parse")


if __name__ == "__main__":
    main()
