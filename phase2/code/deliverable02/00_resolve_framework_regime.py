"""D2 Phase 0 — resolve the governing significance framework per project.

Emits the TWO-PERIOD canonical schema (plan v2.11 §2): `decision_period` (descriptive,
"decided during") and `applicability_period` (legal-method estimate from initiation date).
There is NO single `regime` column. Rather than silently guessing, emits a priority-resolved
`regime_assignment_status` that keeps low/medium-confidence and boundary dates out of the
headline.

Priority (first match wins): missing_date > boundary_review > low_confidence_review >
assigned_proxy > assigned_medium_confidence > assigned_high.  (not_applicable = no process type)

decision_confidence is a VARCHAR with literal sentinels — 'missing'/'None'/'' are treated as
NULL and route a dated non-proxy row to low_confidence_review (never assigned_high).

Cut dates (verified; CEQ removal via the Federal Register API — plan §2 cut-date table):
  2020-09-14 ceq_2020_rule | 2022-05-20 ceq_2022_phase1 | 2023-06-03 FRA overlay
  2024-07-01 ceq_2024_phase2 | 2025-04-11 ceq_2025_interim_removal
  2026-01-08 ceq_2026_final_removal (91 FR 618, FR Doc 2026-00178)

Run:  conda run -n nepa python phase2/code/deliverable02/00_resolve_framework_regime.py
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
# boundary dates for the ±90-day boundary_review check = the real CEQ cuts PLUS the FRA overlay
BOUNDARY_DATES = [b for b, _ in CUTS[1:]] + [FRA_DATE]
BOUNDARY_DAYS = 90
# confidence sentinels that mean "unknown" (route to low_confidence_review)
_UNKNOWN_CONF = {"", "none", "missing", "nan", "null"}


def _parse(d) -> date | None:
    if d is None or (isinstance(d, float) and pd.isna(d)) or str(d).strip() in ("", "NaT", "None", "nan"):
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
    return any(abs((d - b).days) <= BOUNDARY_DAYS for b in BOUNDARY_DATES)


def _norm_conf(c) -> str:
    """Normalize decision_confidence to {high, medium, low, missing}."""
    s = str(c).strip().lower()
    if s in ("high", "medium", "low"):
        return s
    return "missing"


def _status(d: date | None, is_proxy: bool, conf: str, has_process: bool) -> str:
    if not has_process:
        return "not_applicable"
    if d is None:
        return "missing_date"
    if _near_cut(d):
        return "boundary_review"
    if conf in ("missing", "low"):      # unknown/low confidence never silently assigned
        return "low_confidence_review"
    if is_proxy:
        return "assigned_proxy"
    if conf == "medium":
        return "assigned_medium_confidence"
    return "assigned_high"               # conf == high, non-proxy, dated, off-boundary


def in_scope_projects() -> pd.DataFrame:
    """Clean EA-source FONSI projects (452) + clean EIS/EA projects = the 1,326 regime universe."""
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
    proxy = df["decision_is_proxy"].map(lambda x: bool(x) if pd.notna(x) else False)
    conf = df["decision_confidence"].map(_norm_conf)
    has_process = df["process_type"].map(lambda p: str(p).strip() not in ("", "None", "nan"))

    out = pd.DataFrame({
        "project_id": df["project_id"],
        "process_type": df["process_type"].fillna(""),
        "decision_date": dec.map(lambda d: d.isoformat() if d else ""),
        "decision_source_type": df["decision_source_type"].fillna(""),
        "decision_confidence": conf,                       # normalized {high,medium,low,missing}
        "decision_is_proxy": proxy,
        "decision_period": dec.map(_period),
        "applicability_period": init.where(init.notna(), dec).map(_period),
        "fra_overlay": dec.map(lambda d: bool(d and d >= FRA_DATE)),
        "regime_source_date": dec.map(lambda d: d.isoformat() if d else ""),
        "regime_source_date_role": dec.map(lambda d: "decision" if d else "none"),
        "regime_source_date_is_proxy": proxy,
        "regime_source_date_confidence": conf,
        "regime_assignment_status": [
            _status(d, p, c, hp) for d, p, c, hp in zip(dec, proxy, conf, has_process)
        ],
        "regime_notes": [
            f"raw_conf={rc}; applicability_from={'initiation' if _parse(i) is not None else 'decision'}"
            for rc, i in zip(df["decision_confidence"].fillna(""), df["initiation_date"])
        ],
        "regime_run_at": C.utc_now(),
        "schema_version": C.SCHEMA_VERSION,
    })

    C.write_parquet(out, C.PROJECT_REGIME, "regime")
    print("\nregime_assignment_status:")
    print(out["regime_assignment_status"].value_counts().to_string())
    print("\ndecision_period:")
    print(out["decision_period"].value_counts().to_string())


if __name__ == "__main__":
    main()
