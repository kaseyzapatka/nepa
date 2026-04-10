"""
Validation script: compare a refactored regex-prep output against a baseline cache.

Usage:
    python code/extract/validate_regex_refactor.py <baseline.parquet> <refactored.parquet>

Examples:
    # CE
    python code/extract/validate_regex_refactor.py \\
        data/analysis/regex_candidates.parquet \\
        data/analysis/regex_candidates_ce_refactored.parquet

    # EA
    python code/extract/validate_regex_refactor.py \\
        data/analysis/regex_candidates_ea.parquet \\
        data/analysis/regex_candidates_ea_refactored.parquet

    # EIS
    python code/extract/validate_regex_refactor.py \\
        data/analysis/regex_candidates_eis.parquet \\
        data/analysis/regex_candidates_eis_refactored.parquet
"""

import sys
import pandas as pd
from pathlib import Path

# Columns to compare (always exclude run_timestamp — it changes every run)
ALWAYS_EXCLUDE = {"run_timestamp"}
# Columns that carry real information — used for per-row content comparison
CONTENT_COLS = ["project_id", "date", "match", "context", "position", "position_pct",
                "doc_type", "main_document_imputed"]

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
INFO = "\033[36mINFO\033[0m"


def _fmt(label, status, detail=""):
    detail_str = f"  {detail}" if detail else ""
    print(f"  [{status}] {label}{detail_str}")


def compare(baseline_path: str, refactored_path: str, spot_check_n: int = 5):
    baseline_path = Path(baseline_path)
    refactored_path = Path(refactored_path)

    for p in (baseline_path, refactored_path):
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Baseline : {baseline_path.name}")
    print(f"Refactored: {refactored_path.name}")
    print(f"{'='*60}\n")

    base = pd.read_parquet(baseline_path)
    new = pd.read_parquet(refactored_path)

    failures = 0

    # ------------------------------------------------------------------
    # 1. Schema
    # ------------------------------------------------------------------
    print("── Schema ──────────────────────────────────────────────────")
    base_cols = set(base.columns) - ALWAYS_EXCLUDE
    new_cols = set(new.columns) - ALWAYS_EXCLUDE
    shared_cols = base_cols & new_cols
    only_in_base = base_cols - new_cols
    only_in_new = new_cols - base_cols

    if only_in_base:
        _fmt("Columns only in baseline", WARN, str(sorted(only_in_base)))
    if only_in_new:
        _fmt("Columns only in refactored", INFO, str(sorted(only_in_new)))
    if not only_in_base and not only_in_new:
        _fmt("Column sets match", PASS)

    compare_cols = [c for c in CONTENT_COLS if c in shared_cols]
    print(f"  Comparing on {len(compare_cols)} shared content columns: {compare_cols}\n")

    # Compute pid sets early — used in both section 2 and 3
    base_pids = set(base["project_id"].unique())
    new_pids = set(new["project_id"].unique())
    only_base_pids = base_pids - new_pids
    only_new_pids = new_pids - base_pids
    is_sample_run = not only_new_pids and bool(only_base_pids)

    # ------------------------------------------------------------------
    # 2. Overall row count
    # ------------------------------------------------------------------
    print("── Row counts ──────────────────────────────────────────────")
    _fmt(f"Baseline rows  : {len(base):,}", INFO)
    _fmt(f"Refactored rows: {len(new):,}", INFO)
    count_diff = abs(len(base) - len(new))
    count_pct = count_diff / max(len(base), 1) * 100
    if count_diff == 0:
        _fmt("Row counts match exactly", PASS)
    elif is_sample_run:
        _fmt(f"Row count difference: {count_diff:,} ({count_pct:.2f}%) — sample run, expected", INFO)
    elif count_pct < 1:
        _fmt(f"Row count difference: {count_diff:,} ({count_pct:.2f}%)", WARN)
    else:
        _fmt(f"Row count difference: {count_diff:,} ({count_pct:.2f}%)", FAIL)
        failures += 1
    print()

    # ------------------------------------------------------------------
    # 3. Project-level row counts
    # ------------------------------------------------------------------
    print("── Per-project row counts ──────────────────────────────────")
    base_counts = base.groupby("project_id").size().rename("base")
    new_counts = new.groupby("project_id").size().rename("new")

    _fmt(f"Baseline projects  : {len(base_pids):,}", INFO)
    _fmt(f"Refactored projects: {len(new_pids):,}", INFO)

    if only_base_pids:
        status = INFO if is_sample_run else FAIL
        label = "Sample run — baseline has more projects (expected)" if is_sample_run else f"Projects only in baseline: {len(only_base_pids)}"
        _fmt(label, status, f"{len(only_base_pids):,} baseline projects not in refactored output")
        if not is_sample_run:
            failures += 1
    if only_new_pids:
        _fmt(f"Projects only in refactored: {len(only_new_pids)}", FAIL,
             f"e.g. {sorted(only_new_pids)[:3]}")
        failures += 1

    shared_pids = base_pids & new_pids
    comparison = pd.concat([base_counts, new_counts], axis=1).loc[list(shared_pids)]
    mismatch = comparison[comparison["base"] != comparison["new"]]
    if mismatch.empty:
        _fmt(f"Per-project row counts match for all {len(shared_pids):,} shared projects", PASS)
    else:
        _fmt(f"Per-project count mismatches: {len(mismatch)} projects", FAIL)
        failures += 1
        print(mismatch.head(10).to_string())
    print()

    # ------------------------------------------------------------------
    # 4. Content comparison on shared projects
    # ------------------------------------------------------------------
    print("── Content comparison (shared projects, shared columns) ────")

    # Sort both frames the same way for comparison
    sort_keys = [k for k in ["project_id", "date", "match", "position"] if k in compare_cols]

    base_shared = (
        base[base["project_id"].isin(shared_pids)][compare_cols]
        .sort_values(sort_keys)
        .reset_index(drop=True)
    )
    new_shared = (
        new[new["project_id"].isin(shared_pids)][compare_cols]
        .sort_values(sort_keys)
        .reset_index(drop=True)
    )

    if base_shared.shape != new_shared.shape:
        _fmt("Shape mismatch after filtering to shared projects — skipping content diff", WARN)
    else:
        # Compare column by column
        content_ok = True
        for col in compare_cols:
            col_match = base_shared[col].equals(new_shared[col])
            if not col_match:
                n_diff = (base_shared[col] != new_shared[col]).sum()
                _fmt(f"Column '{col}': {n_diff:,} differing values", FAIL)
                failures += 1
                content_ok = False
        if content_ok:
            _fmt("All shared column values match exactly", PASS)
    print()

    # ------------------------------------------------------------------
    # 5. Spot-check individual projects
    # ------------------------------------------------------------------
    print(f"── Spot-check ({spot_check_n} projects) ─────────────────────────────")
    sample_pids = sorted(shared_pids)[:spot_check_n]
    spot_ok = True
    for pid in sample_pids:
        b = base[base["project_id"] == pid][compare_cols].sort_values(sort_keys).reset_index(drop=True)
        n = new[new["project_id"] == pid][compare_cols].sort_values(sort_keys).reset_index(drop=True)
        if b.shape != n.shape or not b.equals(n):
            _fmt(f"MISMATCH for project {pid}", FAIL,
                 f"baseline {len(b)} rows vs refactored {len(n)} rows")
            failures += 1
            spot_ok = False
            # Show first differing row
            if b.shape == n.shape:
                diff_mask = ~b.eq(n).all(axis=1)
                print("    First differing rows:")
                print("    BASELINE:", b[diff_mask].iloc[0].to_dict())
                print("    NEW:     ", n[diff_mask].iloc[0].to_dict())
    if spot_ok:
        _fmt(f"All {spot_check_n} spot-checked projects match", PASS)
    print()

    # ------------------------------------------------------------------
    # 6. doc_type distribution (if available in both)
    # ------------------------------------------------------------------
    if "doc_type" in shared_cols:
        print("── doc_type distribution ───────────────────────────────────")
        base_dt = base["doc_type"].value_counts().rename("baseline")
        new_dt = new["doc_type"].value_counts().rename("refactored")
        dt_cmp = pd.concat([base_dt, new_dt], axis=1).fillna(0).astype(int)
        dt_cmp["diff"] = dt_cmp["refactored"] - dt_cmp["baseline"]
        print(dt_cmp.to_string())
        print()

    # ------------------------------------------------------------------
    # 7. main_document_imputed (if available in both)
    # ------------------------------------------------------------------
    if "main_document_imputed" in shared_cols:
        print("── main_document_imputed distribution ──────────────────────")
        base_imp = base["main_document_imputed"].value_counts().rename("baseline")
        new_imp = new["main_document_imputed"].value_counts().rename("refactored")
        imp_cmp = pd.concat([base_imp, new_imp], axis=1).fillna(0).astype(int)
        imp_cmp["diff"] = imp_cmp["refactored"] - imp_cmp["baseline"]
        print(imp_cmp.to_string())
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("── Summary ─────────────────────────────────────────────────")
    if failures == 0:
        print(f"  [{PASS}] All checks passed — refactored output is equivalent to baseline.")
    else:
        print(f"  [{FAIL}] {failures} check(s) failed — review output above.")
    print()

    return failures == 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    ok = compare(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
