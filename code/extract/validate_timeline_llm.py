# --------------------------
# TIMELINE VALIDATION WITH OLLAMA
# --------------------------
# Use local LLM to validate regex-extracted dates on a sample of projects
# Compares regex results vs LLM extraction to measure accuracy

import pandas as pd
import json
import requests
import random
from pathlib import Path
from datetime import datetime
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------
# FILE PATHS
# --------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"


# --------------------------
# OLLAMA CONFIGURATION
# --------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:latest"  # Faster than qwen2:7b; alternatives: qwen2:7b (better JSON)


def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def list_ollama_models():
    """List available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [m['name'] for m in models]
    except:
        pass
    return []


def create_extraction_prompt(page_texts, max_chars=6000):
    """Create prompt for LLM date extraction."""
    combined_text = "\n\n---PAGE BREAK---\n\n".join(page_texts)
    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars] + "\n\n[...text truncated...]"

    prompt = f"""You are analyzing a NEPA (National Environmental Policy Act) document. Extract all PROJECT-RELATED dates mentioned.

IMPORTANT:
- ONLY extract dates that relate to THIS specific project's timeline
- DO NOT include years from law names (e.g., "Clean Air Act of 1970" - the 1970 is NOT a project date)
- DO NOT include reference dates from other projects or general regulations

Document excerpt:
{combined_text}

Return a JSON object with this exact structure:
{{
  "dates_found": [
    {{"date": "YYYY-MM-DD", "type": "decision|draft|final|notice|comment|scoping|other", "context": "brief quote or description"}}
  ],
  "earliest_project_date": "YYYY-MM-DD or null",
  "decision_date": "YYYY-MM-DD or null",
  "project_year": YYYY or null
}}

If you find a month-year without a day, use the 1st of that month.
Only include dates you are confident are project-related. Return ONLY valid JSON, no other text."""

    return prompt


def extract_with_ollama(page_texts, model=DEFAULT_MODEL, timeout=120):
    """
    Extract dates using Ollama LLM.

    Args:
        page_texts: List of page text strings
        model: Ollama model name
        timeout: Request timeout in seconds

    Returns:
        dict: Parsed JSON response or None on error
    """
    prompt = create_extraction_prompt(page_texts)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,  # Low temperature for consistent output
                }
            },
            timeout=timeout
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')

            # Try to parse JSON from response
            # Sometimes LLM adds text before/after JSON
            try:
                # Find JSON in response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                print(f"  Warning: Could not parse JSON from response")
                return {'raw_response': response_text}

        else:
            print(f"  Ollama error: {response.status_code}")
            return None

    except requests.exceptions.Timeout:
        print(f"  Timeout after {timeout}s")
        return None
    except requests.exceptions.ConnectionError:
        print("  Ollama not running. Start with: ollama serve")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def get_project_pages(project_id, source, max_pages=10):
    """Load pages for a project from processed parquet files."""
    data_dir = PROCESSED_DIR / source.lower()

    docs_df = pd.read_parquet(data_dir / "documents.parquet")
    pages_df = pd.read_parquet(data_dir / "pages.parquet")

    # Clean project_id
    def extract_id(x):
        if isinstance(x, dict):
            return x.get('value', '')
        return x

    docs_df['project_id'] = docs_df['project_id'].apply(extract_id)

    # Get documents for this project (prefer main documents)
    project_docs = docs_df[docs_df['project_id'] == project_id]
    main_docs = project_docs[project_docs['main_document'] == 'YES']

    if main_docs.empty:
        main_docs = project_docs

    # Get pages
    doc_ids = main_docs['document_id'].tolist()
    project_pages = pages_df[pages_df['document_id'].isin(doc_ids)]

    # Return first N pages
    return project_pages.head(max_pages)['page_text'].tolist()


def compare_results(regex_result, llm_result):
    """
    Compare regex extraction with LLM extraction.

    Returns:
        dict: Comparison metrics
    """
    comparison = {
        'regex_earliest': regex_result.get('project_date_earliest'),
        'regex_decision': regex_result.get('project_date_decision'),
        'regex_year': regex_result.get('project_year'),
        'llm_earliest': None,
        'llm_decision': None,
        'llm_year': None,
        'match_earliest': False,
        'match_decision': False,
        'match_year': False,
    }

    if llm_result and isinstance(llm_result, dict):
        comparison['llm_earliest'] = llm_result.get('earliest_project_date')
        comparison['llm_decision'] = llm_result.get('decision_date')
        comparison['llm_year'] = llm_result.get('project_year')

        # Check matches
        if comparison['regex_earliest'] and comparison['llm_earliest']:
            comparison['match_earliest'] = comparison['regex_earliest'] == comparison['llm_earliest']

        if comparison['regex_decision'] and comparison['llm_decision']:
            comparison['match_decision'] = comparison['regex_decision'] == comparison['llm_decision']

        if comparison['regex_year'] and comparison['llm_year']:
            comparison['match_year'] = comparison['regex_year'] == comparison['llm_year']

    return comparison


def run_validation(sample_size=50, model=DEFAULT_MODEL, random_seed=42):
    """
    Run validation comparing regex vs LLM date extraction.

    Args:
        sample_size: Number of projects to validate
        model: Ollama model to use
        random_seed: For reproducible sampling

    Outputs:
        Prints validation results and saves to data/analysis/timeline_validation.csv
    """
    print("=== Timeline Validation: Regex vs LLM ===\n")

    # Check Ollama
    if not check_ollama_running():
        print("ERROR: Ollama is not running.")
        print("Start it with: ollama serve")
        print("\nAvailable once started, check models with: ollama list")
        return

    available_models = list_ollama_models()
    print(f"Ollama is running. Available models: {available_models}")

    if model not in [m.split(':')[0] for m in available_models] and model not in available_models:
        print(f"\nWARNING: Model '{model}' may not be available.")
        print(f"Pull it with: ollama pull {model}")

    # Load regex results
    timeline_path = ANALYSIS_DIR / "projects_timeline.parquet"
    if not timeline_path.exists():
        print(f"\nERROR: {timeline_path} not found.")
        print("Run extract_timeline.py first.")
        return

    projects = pd.read_parquet(timeline_path)
    print(f"\nLoaded {len(projects):,} projects with regex timeline data")

    # Sample projects (prefer those with dates extracted)
    has_dates = projects[projects['project_date_earliest'].notna()]
    print(f"Projects with dates: {len(has_dates):,}")

    random.seed(random_seed)
    if len(has_dates) > sample_size:
        sample_indices = random.sample(list(has_dates.index), sample_size)
        sample = projects.loc[sample_indices]
    else:
        sample = has_dates.head(sample_size)

    print(f"Validating {len(sample)} projects with model: {model}")
    print("\nStarting validation (this may take a few minutes)...\n")

    # Run validation
    results = []
    for idx, (_, row) in enumerate(sample.iterrows()):
        project_id = row['project_id']
        project_title = row['project_title'][:40] if row['project_title'] else 'Untitled'
        source = row['dataset_source']

        print(f"[{idx+1}/{len(sample)}] {project_title}...")

        # Get pages
        page_texts = get_project_pages(project_id, source)

        if not page_texts:
            print(f"  No pages found, skipping")
            continue

        # Extract with LLM
        start_time = time.time()
        llm_result = extract_with_ollama(page_texts, model=model)
        elapsed = time.time() - start_time

        # Compare
        regex_result = {
            'project_date_earliest': row['project_date_earliest'],
            'project_date_decision': row['project_date_decision'],
            'project_year': row['project_year'],
        }

        comparison = compare_results(regex_result, llm_result)
        comparison['project_id'] = project_id
        comparison['project_title'] = project_title
        comparison['llm_time_seconds'] = elapsed
        comparison['llm_raw'] = json.dumps(llm_result) if llm_result else None

        results.append(comparison)

        # Print quick result
        year_match = "✓" if comparison['match_year'] else "✗"
        print(f"  Year: regex={comparison['regex_year']} vs llm={comparison['llm_year']} {year_match} ({elapsed:.1f}s)")

    # Summary
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 50)
    print("=== VALIDATION SUMMARY ===\n")

    total = len(results_df)
    if total == 0:
        print("No results to summarize")
        return

    year_matches = results_df['match_year'].sum()
    earliest_matches = results_df['match_earliest'].sum()
    decision_matches = results_df['match_decision'].sum()

    # Count where both have values
    both_have_year = ((results_df['regex_year'].notna()) & (results_df['llm_year'].notna())).sum()
    both_have_earliest = ((results_df['regex_earliest'].notna()) & (results_df['llm_earliest'].notna())).sum()
    both_have_decision = ((results_df['regex_decision'].notna()) & (results_df['llm_decision'].notna())).sum()

    print(f"Total projects validated: {total}")
    print(f"\nYear match rate: {year_matches}/{both_have_year} ({100*year_matches/max(1,both_have_year):.1f}%)")
    print(f"Earliest date match rate: {earliest_matches}/{both_have_earliest} ({100*earliest_matches/max(1,both_have_earliest):.1f}%)")
    print(f"Decision date match rate: {decision_matches}/{both_have_decision} ({100*decision_matches/max(1,both_have_decision):.1f}%)")

    avg_time = results_df['llm_time_seconds'].mean()
    print(f"\nAverage LLM extraction time: {avg_time:.1f}s per project")

    # Save results
    output_path = ANALYSIS_DIR / "timeline_validation.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    return results_df


# --------------------------
# MAIN
# --------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate timeline extraction with Ollama LLM")
    parser.add_argument('--sample', type=int, default=50, help='Number of projects to validate')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Ollama model to use')
    parser.add_argument('--check', action='store_true', help='Just check Ollama status and available models')

    args = parser.parse_args()

    if args.check:
        if check_ollama_running():
            print("Ollama is running!")
            print(f"Available models: {list_ollama_models()}")
        else:
            print("Ollama is NOT running.")
            print("Start with: ollama serve")
    else:
        run_validation(sample_size=args.sample, model=args.model)
