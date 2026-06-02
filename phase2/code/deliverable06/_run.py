import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import subprocess
import sys
from pathlib import Path

from common import D6_ANALYSIS_DIR, D6_RAW_DIR, PHASE2_DIR


CODE_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = CODE_DIR.parent / "extract"


def run(*args: object) -> None:
    command = [str(arg) for arg in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def latest_ce_snapshot() -> Path | None:
    snapshots = sorted((D6_RAW_DIR / "ce_explorer").glob("exclusions_*.json"))
    return snapshots[-1] if snapshots else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the D6 Stage A FONSI opportunity scan.")
    parser.add_argument("--skip-inventory", action="store_true")
    parser.add_argument("--skip-sections", action="store_true")
    parser.add_argument("--skip-crosswalk", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--topics", action="store_true", help="Run optional NMF diagnostics.")
    parser.add_argument("--skip-input-hashes", action="store_true", help="Use only for smoke tests.")
    return parser.parse_args()


def crosswalk_command(args: argparse.Namespace, snapshot: Path | None = None) -> list[object]:
    command: list[object] = [sys.executable, CODE_DIR / "04_build_ce_crosswalk.py"]
    if snapshot:
        command.extend(["--snapshot", snapshot])
    if args.skip_embeddings:
        command.append("--skip-embeddings")
    return command


def main() -> None:
    args = parse_args()
    if not args.skip_inventory:
        command: list[object] = [sys.executable, CODE_DIR / "01_build_fonsi_inventory.py"]
        if args.skip_input_hashes:
            command.append("--skip-input-hashes")
        run(*command)
    if not args.skip_sections:
        run(
            sys.executable,
            EXTRACT_DIR / "build_document_sections.py",
            "--process",
            "EA",
            "--target-documents",
            D6_ANALYSIS_DIR / "fonsi_section_manifest.parquet",
            "--output",
            D6_ANALYSIS_DIR / "fonsi_document_sections.parquet",
            "--qa-output",
            PHASE2_DIR / "output" / "deliverable06" / "fonsi_document_sections_qa.csv",
        )
    run(sys.executable, CODE_DIR / "03_bootstrap_action_archetypes.py")
    if not args.skip_crosswalk:
        run(*crosswalk_command(args, latest_ce_snapshot()))
    run(sys.executable, CODE_DIR / "05_build_fonsi_packets.py")
    # Packet action text upgrades metadata-only assignments where it supports a seed rule.
    run(sys.executable, CODE_DIR / "03_bootstrap_action_archetypes.py")
    if not args.skip_crosswalk:
        run(*crosswalk_command(args, latest_ce_snapshot()))
    run(sys.executable, CODE_DIR / "06_extract_fonsi_actions.py")
    run(sys.executable, CODE_DIR / "07_analyze_fonsi_patterns.py")
    if args.topics:
        run(sys.executable, CODE_DIR / "08_topic_model_diagnostics.py")
    run(sys.executable, CODE_DIR / "09_render_fonsi_dossiers.py")


if __name__ == "__main__":
    main()

