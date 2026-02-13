"""
Phase 2, Deliverable 2: "Significant Impact" Factors — Proof of Concept
========================================================================
Goal: Can we identify significance determinations by resource area
      from EIS and EA documents?

Strategy: Scan pages in batches (the EIS parquet is 6 GB) and check:
  1. How many projects contain significance language at all?
  2. Can we pair significance language with resource areas on the same page?
  3. Can we extract structured (resource_area, determination) pairs?

This is a feasibility check, not a final pipeline.
"""

import pyarrow.parquet as pq
import pandas as pd
import re
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"

# ============================================================
# PATTERNS
# ============================================================

# Significance determination phrases
SIGNIFICANCE_PATTERNS = {
    "significant_and_unavoidable": re.compile(
        r"significant\s+and\s+unavoidable", re.I
    ),
    "less_than_significant": re.compile(
        r"less\s+than\s+significant", re.I
    ),
    "less_than_significant_with_mitigation": re.compile(
        r"(?:mitigat\w+\s+to|reduced\s+to)\s+(?:a\s+)?less\s+than\s+significant", re.I
    ),
    "no_significant_impact": re.compile(
        r"no\s+significant\s+(?:adverse\s+)?(?:impact|effect)", re.I
    ),
    "significant_impact": re.compile(
        r"(?<!no\s)(?<!no\s\s)significant\s+(?:adverse\s+)?(?:impact|effect)", re.I
    ),
}

# Resource area keywords
RESOURCE_AREAS = {
    "air_quality": re.compile(r"\b(?:air quality|air resources|emissions)\b", re.I),
    "water": re.compile(r"\b(?:water resources|water quality|hydrology|groundwater|surface water|floodplain)\b", re.I),
    "biological": re.compile(r"\b(?:biological resources|wildlife|vegetation|habitat|threatened species|endangered species|special.?status species)\b", re.I),
    "cultural": re.compile(r"\b(?:cultural resources|historic properties|archaeological|tribal|traditional cultural)\b", re.I),
    "visual": re.compile(r"\b(?:visual resources|aesthetics|visual impact|scenic)\b", re.I),
    "noise": re.compile(r"\b(?:noise|sound levels|decibel)\b", re.I),
    "soils_geology": re.compile(r"\b(?:soils|geology|geologic|erosion|seismic)\b", re.I),
    "socioeconomic": re.compile(r"\b(?:socioeconomic|environmental justice|disproportionate)\b", re.I),
    "transportation": re.compile(r"\b(?:transportation|traffic|road)\b", re.I),
    "land_use": re.compile(r"\b(?:land use|recreation|public lands|zoning)\b", re.I),
    "climate_ghg": re.compile(r"\b(?:climate change|greenhouse gas|GHG|carbon)\b", re.I),
    "public_health": re.compile(r"\b(?:public health|hazardous materials|contamination|human health)\b", re.I),
}


def scan_pages(source: str, max_pages: int = 200_000):
    """Scan pages from a parquet file and collect significance stats."""
    pages_path = DATA_DIR / "processed" / source / "pages.parquet"
    docs = pd.read_parquet(
        DATA_DIR / "analysis" / "documents_combined.parquet",
        columns=["document_id", "project_id"],
    )
    doc_to_project = dict(zip(docs["document_id"], docs["project_id"]))

    pf = pq.ParquetFile(pages_path)

    # Track project-level hits
    project_any_sig = set()
    project_by_pattern = defaultdict(set)
    project_resource_sig = defaultdict(lambda: defaultdict(set))  # project -> resource -> set of determinations
    all_projects = set()
    pages_scanned = 0

    # Collect structured examples
    structured_examples = []

    for batch in pf.iter_batches(batch_size=5000):
        df = batch.to_pandas()
        pages_scanned += len(df)

        for _, row in df.iterrows():
            proj_id = doc_to_project.get(row["document_id"])
            if not proj_id:
                continue
            all_projects.add(proj_id)
            text = str(row["page_text"])

            # Check each significance pattern
            for pat_name, pat in SIGNIFICANCE_PATTERNS.items():
                if pat.search(text):
                    project_any_sig.add(proj_id)
                    project_by_pattern[pat_name].add(proj_id)

                    # Try to pair with resource areas in same sentence
                    sentences = re.findall(
                        r"[^.]*" + pat.pattern + r"[^.]*\.", text, re.I
                    )
                    for sent in sentences[:3]:
                        for res_name, res_pat in RESOURCE_AREAS.items():
                            if res_pat.search(sent):
                                project_resource_sig[proj_id][res_name].add(pat_name)
                                if len(structured_examples) < 30:
                                    structured_examples.append({
                                        "project_id": proj_id,
                                        "resource": res_name,
                                        "determination": pat_name,
                                        "sentence": sent.strip()[:300],
                                    })

        if pages_scanned >= max_pages:
            break

    return {
        "source": source,
        "pages_scanned": pages_scanned,
        "total_projects": len(all_projects),
        "projects_any_sig": len(project_any_sig),
        "by_pattern": {k: len(v) for k, v in project_by_pattern.items()},
        "projects_with_resource_pairs": len(project_resource_sig),
        "resource_sig_map": dict(project_resource_sig),
        "examples": structured_examples,
    }


def print_results(results):
    """Print scan results."""
    src = results["source"].upper()
    total = results["total_projects"]
    print(f"\n{'='*60}")
    print(f"  {src}: Scanned {results['pages_scanned']:,} pages across {total} projects")
    print(f"{'='*60}")

    print(f"\n--- Significance Language Coverage ---")
    print(f"  Any significance language: {results['projects_any_sig']} / {total} "
          f"({results['projects_any_sig']/total*100:.1f}%)")
    for pat_name, count in sorted(results["by_pattern"].items(), key=lambda x: -x[1]):
        print(f"  {pat_name}: {count} / {total} ({count/total*100:.1f}%)")

    n_paired = results["projects_with_resource_pairs"]
    print(f"\n--- Resource Area Pairing ---")
    print(f"  Projects with (resource, determination) pairs: {n_paired} / {total} "
          f"({n_paired/total*100:.1f}%)")

    # Aggregate resource area coverage
    resource_counts = defaultdict(int)
    for proj_id, resources in results["resource_sig_map"].items():
        for res in resources:
            resource_counts[res] += 1

    if resource_counts:
        print(f"\n  Resource areas found with significance language:")
        for res, count in sorted(resource_counts.items(), key=lambda x: -x[1]):
            print(f"    {res}: {count} projects")

    print(f"\n--- Structured Examples ---")
    for ex in results["examples"][:8]:
        print(f"  [{ex['resource']}] {ex['determination']}")
        print(f"    >> {ex['sentence'][:250]}")
        print()


if __name__ == "__main__":
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000

    # Scan EIS (the primary target for significance determinations)
    print("Scanning EIS pages...")
    eis_results = scan_pages("eis", max_pages=max_pages)
    print_results(eis_results)

    # Scan EA (for FONSI / mitigated FONSI patterns)
    print("\nScanning EA pages...")
    ea_results = scan_pages("ea", max_pages=max_pages)
    print_results(ea_results)

    # Verdict
    print(f"\n{'='*60}")
    print("  POC VERDICT")
    print(f"{'='*60}")

    eis_sig_pct = eis_results["projects_any_sig"] / eis_results["total_projects"] * 100
    eis_pair_pct = eis_results["projects_with_resource_pairs"] / eis_results["total_projects"] * 100
    ea_sig_pct = ea_results["projects_any_sig"] / ea_results["total_projects"] * 100
    ea_pair_pct = ea_results["projects_with_resource_pairs"] / ea_results["total_projects"] * 100

    print(f"\n  EIS: {eis_sig_pct:.1f}% have significance language, "
          f"{eis_pair_pct:.1f}% have resource+significance pairs")
    print(f"  EA:  {ea_sig_pct:.1f}% have significance language, "
          f"{ea_pair_pct:.1f}% have resource+significance pairs")

    print(f"\n  Regex alone can DETECT significance language in most projects.")
    print(f"  Regex can PAIR resource areas with determinations for many projects.")
    print(f"  The gap: regex can't distinguish the FINAL determination from")
    print(f"  discussion text (e.g., 'impacts would be significant' in alternatives")
    print(f"  analysis vs. the actual finding). LLM needed for that disambiguation.")
    print(f"{'='*60}")
