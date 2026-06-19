"""
Verify that a parallel D4 run reproduces a baseline output as a SET of rows.

Used to gate the parallelization of 02_retrieve.py / 03_extract_candidates.py (todo #26):
the parallel path is a pure performance change and must not alter any output value.

"Reproduce" = SET-EQUALITY (same rows, any order). The id columns context_packet_id /
candidate_id are NOT unique in these outputs (legitimate collisions exist), so the check
is a MULTISET comparison over ALL non-timestamp columns, not a key join. Per-run timestamp
columns (created_at, *_run_at) are DROPPED before comparing, since they differ by
construction. Values are compared after type-normalization (a float 2.0 and an int 2 are
equal; None and NaN are equal), so a benign pandas dtype drift between two builds does not
cause a false FAIL.

Usage:
    python _verify_parallel.py A.parquet B.parquet --key context_packet_id   # 02
    python _verify_parallel.py A.parquet B.parquet --key candidate_id        # 03

The --key column is used only to group/triage divergent rows (not to join). Exit code 0 on
PASS, 1 on FAIL. When the only differing rows have retrieval_tier in {tier_b, tier_d,
eis_text_fallback} — page selections that break ties on input row order — the script flags
this as page-selection nondeterminism (an equally-valid realization) rather than a hard
data bug; it still exits 1 so a human decides, but the message says so explicitly.
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Columns dropped before comparison (per-run, differ by construction).
TIMESTAMP_COLS = {"created_at", "index_run_at"}

# Tiers whose membership can legitimately differ by input page row order (tie-breaks):
#   tier_d / eis_text_fallback iterate pages in input order; tier_b sorts by page_num
#   but ties (duplicate/NaN page_number) fall back to input order in the stable sort.
ORDER_SENSITIVE_TIERS = {"tier_b", "tier_d", "eis_text_fallback"}


def _is_timestamp_col(name: str) -> bool:
    return name in TIMESTAMP_COLS or name.endswith("_run_at")


def _normval(v) -> str:
    """Type-normalized string for value comparison (None==NaN; 2==2.0)."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if v is None or v is pd.NA or v is pd.NaT:
        return "∅"
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return "∅" if math.isnan(f) else repr(round(f, 9))
    try:
        if pd.isna(v):
            return "∅"
    except (TypeError, ValueError):
        pass
    return str(v)


def _row_tuples(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    """Type-normalized row tuples over `cols` (order = cols)."""
    norm = {c: df[c].map(_normval).tolist() for c in cols}
    return list(zip(*[norm[c] for c in cols]))


def _triage(rows: list[tuple], cols: list[str], key: str) -> None:
    """Group divergent rows by retrieval_tier and report whether they are confined
    to order-sensitive tiers (page-selection tie-break nondeterminism)."""
    tier_col = "retrieval_tier"
    if tier_col not in cols:
        return
    ti = cols.index(tier_col)
    vc = Counter(r[ti] for r in rows)
    print(f"  divergent rows by {tier_col}:")
    for t, c in sorted(vc.items(), key=lambda kv: -kv[1]):
        print(f"    {t}: {c:,}")
    if set(vc) <= ORDER_SENSITIVE_TIERS:
        print(f"  TRIAGE: all divergent rows are order-sensitive page selections "
              f"({sorted(ORDER_SENSITIVE_TIERS)}). These break ties on input page row "
              "order, so a deterministic re-ordering yields an equally-valid set — not a "
              "logic regression. Human review required.")
    else:
        print("  TRIAGE: divergent rows include NON-order-sensitive tiers "
              f"({sorted(set(vc) - ORDER_SENSITIVE_TIERS)}) — REAL difference, not a "
              "tie-break artifact.")


def verify(path_a: Path, path_b: Path, key: str) -> bool:
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)
    print(f"A: {path_a}  ({len(df_a):,} rows, {df_a.shape[1]} cols)")
    print(f"B: {path_b}  ({len(df_b):,} rows, {df_b.shape[1]} cols)")

    # Compare on the shared, non-timestamp columns.
    cols_a = {c for c in df_a.columns if not _is_timestamp_col(c)}
    cols_b = {c for c in df_b.columns if not _is_timestamp_col(c)}
    dropped = sorted((set(df_a.columns) | set(df_b.columns)) - (cols_a | cols_b))
    if dropped:
        print(f"Dropped timestamp columns: {dropped}")
    only_a_cols, only_b_cols = sorted(cols_a - cols_b), sorted(cols_b - cols_a)
    if only_a_cols or only_b_cols:
        print(f"FAIL: column-set mismatch. A-only={only_a_cols}  B-only={only_b_cols}")
        return False
    cols = sorted(cols_a & cols_b)

    # Multiset comparison over full rows (ids are not unique → no key join).
    ca = Counter(_row_tuples(df_a, cols))
    cb = Counter(_row_tuples(df_b, cols))
    if ca == cb:
        print(f"\nValue check: identical multiset of {len(df_a):,} rows across "
              f"{len(cols)} columns.")
        print("\nPASS: parallel output set-equals baseline.")
        return True

    only_a = list((ca - cb).elements())
    only_b = list((cb - ca).elements())
    print(f"\nROW-SET MISMATCH: {len(only_a):,} rows only in A, {len(only_b):,} only in B "
          f"(multiset diff over {len(cols)} cols).")
    _triage(only_a + only_b, cols, key)
    ki = cols.index(key) if key in cols else None
    for label, rows in (("A", only_a), ("B", only_b)):
        for r in rows[:4]:
            kv = f"{key}={r[ki]}" if ki is not None else ""
            tier = r[cols.index("retrieval_tier")] if "retrieval_tier" in cols else ""
            print(f"  only in {label}: {kv} tier={tier}")
    print("\nFAIL: see divergences above.")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Set-equality check for D4 parallel output.")
    ap.add_argument("path_a")
    ap.add_argument("path_b")
    ap.add_argument("--key", required=True, help="Unique id column (context_packet_id / candidate_id).")
    args = ap.parse_args()
    ok = verify(Path(args.path_a), Path(args.path_b), args.key)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
