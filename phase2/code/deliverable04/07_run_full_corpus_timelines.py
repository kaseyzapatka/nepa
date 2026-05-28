"""
Orchestration wrapper for full-corpus D4 timeline extraction.

Runs scripts 02–04 (and optionally 06) in shards by process_type and
project-id hash bucket. Maintains a manifest with shard status, input hashes,
and row counts. Resumes from completed shards by default.

Usage:
    python 07_run_full_corpus_timelines.py [--process CE EA EIS] [--shards 10]
    python 07_run_full_corpus_timelines.py --force              # rerun all shards
    python 07_run_full_corpus_timelines.py --with-api --process EA EIS
    python 07_run_full_corpus_timelines.py --dry-run --shards 2 --process CE
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import hashlib
import importlib.util
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
TIMELINE_DIR = ANALYSIS_DIR / "timeline"
OUTPUT_DIR = PHASE2 / "output" / "deliverable04"
D4_CODE = Path(__file__).parent

PROJECTS_PATH = ANALYSIS_DIR / "projects_combined.parquet"
MANIFEST_PATH = TIMELINE_DIR / "timeline_run_manifest.parquet"

PROCESS_TYPES = ["CE", "EA", "EIS"]
DEFAULT_SHARDS = 5


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    return hashlib.sha1(f"{path}|{stat.st_size}|{stat.st_mtime}".encode()).hexdigest()[:16]


def _run_id(process_type: str, shard_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{process_type}_{shard_id}_{ts}"


def load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_parquet(MANIFEST_PATH)
    return pd.DataFrame(
        columns=[
            "run_id", "script_name", "stage", "process_type", "shard_id",
            "input_paths", "input_hashes", "output_paths", "status",
            "started_at", "completed_at", "n_projects", "n_documents",
            "n_context_packets", "n_candidates", "n_errors", "error_message",
        ]
    )


def save_manifest(manifest: pd.DataFrame) -> None:
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(MANIFEST_PATH, index=False)


def upsert_manifest_row(manifest: pd.DataFrame, row: dict) -> pd.DataFrame:
    run_id = row["run_id"]
    existing_mask = manifest["run_id"] == run_id
    if existing_mask.any():
        for col, val in row.items():
            manifest.loc[existing_mask, col] = val
    else:
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    return manifest


def shard_project_ids(project_ids: list[str], n_shards: int) -> list[list[str]]:
    """Partition project_ids into n_shards buckets deterministically by hash."""
    buckets: list[list[str]] = [[] for _ in range(n_shards)]
    for pid in project_ids:
        bucket = int(hashlib.sha1(pid.encode()).hexdigest(), 16) % n_shards
        buckets[bucket].append(pid)
    return buckets


def write_shard_ids(ids: list[str], path: Path) -> None:
    path.write_text("\n".join(ids))


def is_shard_complete(manifest: pd.DataFrame, process_type: str, shard_id: str, stage: str) -> bool:
    mask = (
        (manifest["process_type"] == process_type) &
        (manifest["shard_id"] == shard_id) &
        (manifest["stage"] == stage) &
        (manifest["status"] == "completed")
    )
    return mask.any()


def run_script(
    script_name: str,
    args: list[str],
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Run a D4 script as a subprocess. Returns (success, error_message)."""
    cmd = [sys.executable, str(D4_CODE / script_name)] + args
    print(f"  Running: {' '.join(cmd)}")
    if dry_run:
        print(f"    [dry-run] would execute: {' '.join(cmd)}")
        return True, ""
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        return False, f"Exit code {result.returncode}"
    return True, ""


def run_shard(
    process_type: str,
    shard_id: str,
    project_ids: list[str],
    manifest: pd.DataFrame,
    force: bool,
    dry_run: bool,
    with_api: bool,
) -> pd.DataFrame:
    """Run all pipeline stages for a single shard."""
    import tempfile

    shard_tmp = Path(tempfile.mktemp(suffix=".txt", prefix=f"shard_{process_type}_{shard_id}_"))
    write_shard_ids(project_ids, shard_tmp)

    stages = [
        ("02_retrieve_timeline_contexts.py", "retrieval"),
        ("03_extract_timeline_candidates.py", "candidates"),
        ("04_select_timeline_dates.py", "selection"),
    ]
    if with_api:
        stages.append(("06_adjudicate_timeline_api.py", "api"))

    for script_name, stage in stages:
        if not force and is_shard_complete(manifest, process_type, shard_id, stage):
            print(f"    [{process_type}/{shard_id}/{stage}] already complete, skipping.")
            continue

        run_id = _run_id(process_type, shard_id)
        manifest_row = {
            "run_id": run_id,
            "script_name": script_name,
            "stage": stage,
            "process_type": process_type,
            "shard_id": shard_id,
            "input_paths": str(shard_tmp),
            "input_hashes": _file_hash(PROJECTS_PATH),
            "output_paths": str(TIMELINE_DIR),
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "n_projects": len(project_ids),
            "n_documents": None,
            "n_context_packets": None,
            "n_candidates": None,
            "n_errors": 0,
            "error_message": None,
        }
        manifest = upsert_manifest_row(manifest, manifest_row)
        save_manifest(manifest)

        extra_args = ["--process", process_type, "--sample-ids", str(shard_tmp), "--append"]
        if stage == "api":
            extra_args = ["--process", process_type, "--mode", "candidate_adjudication"]

        success, err = run_script(script_name, extra_args, dry_run=dry_run)

        manifest_row["status"] = "completed" if success else "failed"
        manifest_row["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_row["n_errors"] = 0 if success else 1
        manifest_row["error_message"] = err if not success else None

        # Try to read output counts
        if success and not dry_run:
            try:
                if stage == "retrieval" and (TIMELINE_DIR / "timeline_context_packets.parquet").exists():
                    pkt = pd.read_parquet(
                        TIMELINE_DIR / "timeline_context_packets.parquet",
                        columns=["project_id"],
                    )
                    manifest_row["n_context_packets"] = pkt[pkt["project_id"].isin(project_ids)].shape[0]
                elif stage == "candidates" and (TIMELINE_DIR / "timeline_candidates.parquet").exists():
                    cands = pd.read_parquet(
                        TIMELINE_DIR / "timeline_candidates.parquet",
                        columns=["project_id"],
                    )
                    manifest_row["n_candidates"] = cands[cands["project_id"].isin(project_ids)].shape[0]
            except Exception:
                pass

        manifest = upsert_manifest_row(manifest, manifest_row)
        save_manifest(manifest)

        if not success:
            print(f"    ERROR in {script_name}: {err}")
            break

    if shard_tmp.exists():
        shard_tmp.unlink()

    return manifest


def check_and_rebuild_sections(process_types: list[str], dry_run: bool) -> None:
    """Ensure document_sections.parquet is current before the full run."""
    build_script = D4_CODE / "00b_build_document_sections.py"
    if not build_script.exists():
        print("WARNING: 00b_build_document_sections.py not found; skipping section check.")
        return
    args = ["--process"] + [p for p in process_types if p != "CE"]  # CE optional per policy
    if dry_run:
        print(f"  [dry-run] would run sections check: {' '.join(args)}")
        return
    cmd = [sys.executable, str(build_script)] + args
    print(f"Checking sections: {' '.join(cmd)}")
    subprocess.run(cmd, capture_output=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate full-corpus D4 timeline extraction.")
    parser.add_argument("--process", nargs="+", choices=PROCESS_TYPES, default=PROCESS_TYPES)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS, help="Number of shards per process type.")
    parser.add_argument("--force", action="store_true", help="Rerun all shards even if completed.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--with-api", action="store_true", help="Run API adjudication after selection.")
    parser.add_argument("--skip-sections-check", action="store_true", help="Skip sections rebuild check.")
    args = parser.parse_args()

    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Full-corpus run: process={args.process} shards={args.shards} force={args.force}")

    # Step 0: Ensure sections are current
    if not args.skip_sections_check:
        check_and_rebuild_sections(args.process, args.dry_run)

    # Step 1: Ensure index is built
    if not (TIMELINE_DIR / "timeline_document_index.parquet").exists() or args.force:
        print("Building timeline index...")
        success, err = run_script(
            "01_build_timeline_index.py",
            ["--process"] + args.process,
            dry_run=args.dry_run,
        )
        if not success:
            raise RuntimeError(f"Index build failed: {err}")
    else:
        print("Timeline index exists; skipping rebuild (use --force to rebuild).")

    # Load project IDs per process type for sharding
    print("Loading project IDs for sharding...")
    projects_df = pd.read_parquet(PROJECTS_PATH, columns=["project_id", "process_type"])
    projects_df = projects_df[projects_df["process_type"].isin(args.process)]

    manifest = load_manifest()
    total_start = time.time()

    for process_type in args.process:
        proc_ids = projects_df[projects_df["process_type"] == process_type]["project_id"].tolist()
        if not proc_ids:
            print(f"No projects for {process_type}; skipping.")
            continue

        shards = shard_project_ids(proc_ids, args.shards)
        print(f"\n=== {process_type}: {len(proc_ids):,} projects → {args.shards} shards ===")

        for shard_idx, shard_ids in enumerate(shards):
            if not shard_ids:
                continue
            shard_id = f"s{shard_idx:02d}"
            print(f"\n  Shard {shard_id}: {len(shard_ids)} projects")

            manifest = run_shard(
                process_type=process_type,
                shard_id=shard_id,
                project_ids=shard_ids,
                manifest=manifest,
                force=args.force,
                dry_run=args.dry_run,
                with_api=args.with_api,
            )

    elapsed = time.time() - total_start
    print(f"\n=== Full-corpus run complete: {elapsed:.0f}s ===")

    # Print manifest summary
    if MANIFEST_PATH.exists():
        final_manifest = pd.read_parquet(MANIFEST_PATH)
        status_counts = final_manifest["status"].value_counts().to_dict()
        print(f"Manifest: {status_counts}")

    # Check output counts
    if not args.dry_run:
        for fname, label in [
            ("timeline_project_dates.parquet", "project dates"),
            ("timeline_context_packets.parquet", "context packets"),
            ("timeline_candidates.parquet", "candidates"),
        ]:
            path = TIMELINE_DIR / fname
            if path.exists():
                df = pd.read_parquet(path, columns=["project_id"])
                print(f"  {label}: {len(df):,} rows, {df['project_id'].nunique():,} projects")


if __name__ == "__main__":
    main()
