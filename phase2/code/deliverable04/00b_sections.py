"""
D4 wrapper for build_document_sections.py.

Ensures document_sections.parquet covers the full CE/EA/EIS corpus (not clean-only)
and is not stale relative to the source pages data. Writes D4-specific section QA
diagnostics used by retrieval scoring.

Usage:
    python 00b_sections.py [--check-only] [--force] [--process EA EIS CE]
    python 00b_sections.py --force  # rebuild unconditionally
    python 00b_sections.py --check-only  # report staleness without rebuilding
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
PROCESSED_DIR = PHASE2 / "data" / "processed"
OUTPUT_DIR = PHASE2 / "output" / "deliverable04"
SECTIONS_PATH = ANALYSIS_DIR / "document_sections.parquet"
BUILD_SECTIONS_SCRIPT = PHASE2 / "code" / "extract" / "build_document_sections.py"

PROCESS_TYPES = ["CE", "EA", "EIS"]
SOURCE_MAP = {"CE": "ce", "EA": "ea", "EIS": "eis"}

STALE_DAYS = 30  # rebuild if sections are older than this


def pages_mtime(process_type: str) -> float:
    src = SOURCE_MAP[process_type]
    p = PROCESSED_DIR / src / "pages.parquet"
    return p.stat().st_mtime if p.exists() else 0.0


def sections_mtime() -> float:
    return SECTIONS_PATH.stat().st_mtime if SECTIONS_PATH.exists() else 0.0


def check_coverage(sections: pd.DataFrame) -> dict:
    """Report process-type coverage and QA metrics."""
    report: dict = {}
    counts = sections["process_type"].value_counts().to_dict()
    projects = sections.groupby("process_type")["project_id"].nunique().to_dict()
    docs = sections.groupby("process_type")["document_id"].nunique().to_dict()

    for pt in PROCESS_TYPES:
        src = SOURCE_MAP[pt]
        docs_path = PROCESSED_DIR / src / "documents.parquet"
        if docs_path.exists():
            total_docs = len(pd.read_parquet(docs_path, columns=["document_id"]))
        else:
            total_docs = None
        report[pt] = {
            "sections": counts.get(pt, 0),
            "projects": projects.get(pt, 0),
            "docs_with_sections": docs.get(pt, 0),
            "total_docs_in_corpus": total_docs,
            "pct_docs_covered": (
                round(100.0 * docs.get(pt, 0) / total_docs, 1) if total_docs else None
            ),
        }

    # QA flags
    if len(sections) > 0:
        report["_qa"] = {
            "pct_suspicious_heading": round(
                100.0 * sections.get("suspicious_heading", pd.Series(dtype=bool)).sum() / len(sections), 2
            ) if "suspicious_heading" in sections.columns else None,
            "pct_toc_like": round(
                100.0 * sections["is_toc_like"].sum() / len(sections), 2
            ) if "is_toc_like" in sections.columns else None,
            "pct_short_section": round(
                100.0 * sections["short_section"].sum() / len(sections), 2
            ) if "short_section" in sections.columns else None,
        }
    return report


def is_stale(processes: list[str]) -> bool:
    if not SECTIONS_PATH.exists():
        return True
    smtime = sections_mtime()
    for pt in processes:
        if pages_mtime(pt) > smtime:
            return True
    # Also stale if older than threshold
    age_days = (datetime.now().timestamp() - smtime) / 86400
    if age_days > STALE_DAYS:
        return True
    return False


def check_process_coverage(sections: pd.DataFrame, processes: list[str]) -> list[str]:
    """Return process types that have zero rows in current sections file."""
    missing = []
    if sections is None or len(sections) == 0:
        return processes
    covered = set(sections["process_type"].unique())
    for pt in processes:
        if pt == "CE":
            # CE sections are optional per plan §2 policy; do not flag as missing
            continue
        if pt not in covered:
            missing.append(pt)
    return missing


def run_build_sections(processes: list[str], main_only: bool = False) -> None:
    """Invoke build_document_sections.py for the given process types."""
    cmd = [
        sys.executable,
        str(BUILD_SECTIONS_SCRIPT),
        "--process",
        *processes,
    ]
    if main_only:
        cmd.append("--main-only")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"build_document_sections.py exited with code {result.returncode}")


def write_qa_report(report: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for pt in PROCESS_TYPES:
        if pt not in report:
            continue
        r = report[pt]
        rows.append(
            {
                "process_type": pt,
                "sections": r["sections"],
                "projects_with_sections": r["projects"],
                "docs_with_sections": r["docs_with_sections"],
                "total_docs_in_corpus": r["total_docs_in_corpus"],
                "pct_docs_covered": r["pct_docs_covered"],
                "report_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    qa_path = OUTPUT_DIR / "sections_qa_report.csv"
    pd.DataFrame(rows).to_csv(qa_path, index=False)
    print(f"Wrote QA report: {qa_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure document_sections.parquet is current for D4.")
    parser.add_argument("--check-only", action="store_true", help="Report status without rebuilding.")
    parser.add_argument("--force", action="store_true", help="Rebuild unconditionally.")
    parser.add_argument(
        "--process",
        nargs="+",
        choices=PROCESS_TYPES,
        default=["EA", "EIS"],
        help="Process types to check/rebuild (default: EA EIS; CE skipped per §2 policy for short docs).",
    )
    args = parser.parse_args()

    processes = args.process
    print(f"D4 section check: processes={processes}, check_only={args.check_only}, force={args.force}")

    # Load existing sections if present
    sections = None
    if SECTIONS_PATH.exists():
        print(f"Loading existing sections: {SECTIONS_PATH}")
        sections = pd.read_parquet(
            SECTIONS_PATH,
            columns=["project_id", "document_id", "process_type", "suspicious_heading",
                     "is_toc_like", "short_section"],
        )
        print(f"  {len(sections):,} rows, {sections['project_id'].nunique():,} projects")
    else:
        print("document_sections.parquet not found.")

    stale = is_stale(processes) or args.force
    missing_coverage = check_process_coverage(sections, processes) if sections is not None else processes

    print(f"Stale: {stale}  |  Missing coverage for: {missing_coverage}")

    if args.check_only:
        if sections is not None:
            report = check_coverage(sections)
            write_qa_report(report)
        print("Check-only mode; not rebuilding.")
        return

    needs_rebuild = stale or bool(missing_coverage)

    if needs_rebuild:
        rebuild_procs = list(set(processes) | set(missing_coverage))
        # CE is skipped per plan §2 policy unless explicitly requested
        rebuild_procs = [p for p in rebuild_procs if p in processes]
        if not rebuild_procs:
            print("No process types require rebuild.")
        else:
            print(f"Rebuilding sections for: {rebuild_procs}")
            run_build_sections(rebuild_procs)
            # Reload after rebuild
            sections = pd.read_parquet(
                SECTIONS_PATH,
                columns=["project_id", "document_id", "process_type", "suspicious_heading",
                         "is_toc_like", "short_section"],
            )
            print(f"After rebuild: {len(sections):,} rows, {sections['project_id'].nunique():,} projects")
    else:
        print("Sections are current; no rebuild needed.")

    if sections is not None:
        report = check_coverage(sections)
        write_qa_report(report)
        for pt in processes:
            r = report.get(pt, {})
            print(
                f"  {pt}: {r.get('sections', 0):,} sections, "
                f"{r.get('projects', 0):,} projects, "
                f"{r.get('pct_docs_covered')}% docs covered"
            )


if __name__ == "__main__":
    main()
