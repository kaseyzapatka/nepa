"""D2 Phase 0 — resolve the governing significance framework per project.

Keys on decision_date (the rule in force when the review was decided), with an
applicability_period estimate from initiation_date. Emits regime_assignment_status
rather than silently guessing when the date is missing / proxy / near a cut date.

Cut dates (verified; CEQ removal date verified via the Federal Register API):
  2020-09-14 ceq_2020_rule | 2022-05-20 ceq_2022_phase1 | 2023-06-03 FRA overlay
  2024-07-01 ceq_2024_phase2 | 2025-04-11 ceq_2025_interim_removal
  2026-01-08 ceq_2026_final_removal (91 FR 618, FR Doc 2026-00178)

Run:  CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable02/00_resolve_framework_regime.py
Out:  phase2/data/analysis/deliverable02/project_regime.parquet
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

import common as C

# ordered (effective_date, period_key); a date >= the boundary takes that period
CUTS = [
    (date(1, 1, 1), "pre_2020_ceq"),
    (date(2020, 9, 14), "ceq_2020_rule"),
    (date(2022, 5, 20), "ceq_2022_phase1"),
    (date(2024, 7, 1), "ceq_2024_phase2"),
    (date(2025, 4, 11), "ceq_2025_interim_removal"),
    (date(2026, 1, 8), "ceq_2026_final_removal"),
]
FRA_DATE = date(2023, 6, 3)
BOUNDARY_DAYS = 90  # within this many days of a cut -> boundary_review


def _parse(d) -> date | None:
    if d is None or (isinstance(d, float) and pd.isna(d)) or str(d).strip() in ("", "NaT", "None"):
        return None
    try:
        return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _period(d: date | None) -> str:
    if d is None:
        return "unknown"
    key = CUTS[0][1]
    for boundary, k in CUTS:
        if d >= boundary:
            key = k
    return key


def _near_cut(d: date | None) -> bool:
    if d is None:
        return False
    return any(abs((d - boundary).days) <= BOUNDARY_DAYS for boundary, _ in CUTS[1:])


def _status(d: date | None, is_proxy: bool) -> str:
    if d is None:
        return "missing_date"
    if _near_cut(d):
        return "boundary_review"
    if is_proxy:
        return "assigned_proxy"
    return "assigned_high"


def in_scope_projects() -> pd.DataFrame:
    """Clean EA-source FONSI projects (452) + clean EIS projects + clean EA projects."""
    sql = f"""
    WITH fonsi AS (
        SELECT DISTINCT project_id, 'EA' AS process_type
        FROM read_parquet('{C.FONSI_INVENTORY}')
        WHERE project_energy_type = 'Clean' AND stage_a_ea_source = TRUE
    ),
    eis_ea AS (
        SELECT DISTINCT project_id, process_type
        FROM read_parquet('{C.PROJECTS_COMBINED}')
        WHERE project_energy_type = 'Clean' AND process_type IN ('EA', 'EIS')
    )
    SELECT project_id, process_type FROM fonsi
    UNION
    SELECT project_id, process_type FROM eis_ea
    """
    return C.q(sql)


def main() -> None:
    print("D2 Phase 0: resolving framework regime ...")
    scope = in_scope_projects()
    dates = C.q(f"""
        SELECT project_id, decision_date, decision_source_type, decision_confidence,
               decision_is_proxy, initiation_date
        FROM read_parquet('{C.TIMELINE_DATES}')
    """)
    df = scope.merge(dates, on="project_id", how="left")

    dec = df["decision_date"].map(_parse)
    init = df["initiation_date"].map(_parse)
    proxy = (df["decision_is_proxy"] == True)  # noqa: E712 — object col w/ None -> bool Series

    out = pd.DataFrame({
        "project_id": df["project_id"],
        "process_type": df["process_type"],
        "decision_date": dec.map(lambda d: d.isoformat() if d else ""),
        "decision_source_type": df["decision_source_type"].fillna(""),
        "decision_confidence": df["decision_confidence"],
        "decision_is_proxy": proxy,
        "decision_period": dec.map(_period),
        "applicability_period": init.where(init.notna(), dec).map(_period),
        "fra_overlay": dec.map(lambda d: bool(d and d >= FRA_DATE)),
        "regime_assignment_status": [_status(d, p) for d, p in zip(dec, proxy)],
        "regime_run_at": C.utc_now(),
        "schema_version": C.SCHEMA_VERSION,
    })

    C.write_parquet(out, C.D2_ANALYSIS_DIR / "project_regime.parquet", "regime")
    print("\nregime_assignment_status:")
    print(out["regime_assignment_status"].value_counts().to_string())
    print("\ndecision_period:")
    print(out["decision_period"].value_counts().to_string())


if __name__ == "__main__":
    main()
