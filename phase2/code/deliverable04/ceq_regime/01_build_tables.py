#!/usr/bin/env python
"""Build CEQ-regulatory-regime duration tables for D4 (§sec-ceq-regime).

Segments the D4 review-duration corpus by the CEQ NEPA implementing regulation in
effect at each review's DECISION date (primary) and INITIATION date (sensitivity).
This is a NEW segmentation, independent of `reg_period` (funding eras) and the FRA
statutory split — it does NOT touch `d4_duration_by_period.csv` or any existing output.

CEQ regulatory regimes (rule effective dates):
  1978            — original CEQ regulations (40 C.F.R. 1500-1508), ~1978-2020
  2020_trump      — 2020-09-14, Trump "Final Rule" (presumptive time/page limits)
  2022_phase1     — 2022-05-20, Biden Phase 1 Rule (restored purpose-and-need, effects)
  2024_phase2     — 2024-07-01, Phase 2 Rule (codified the FRA's statutory amendments)
  2025_rescission — 2025-04-11 EFFECTIVE date of the CEQ interim final rule
                    "Removal of NEPA Implementing Regulations" (90 FR 10610, published
                    2025-02-25); agencies revert to their own NEPA procedures. Use the
                    EFFECTIVE date, not the publication date.

Standalone 2024 and 2025 buckets are too thin to analyze on their own (see MIN_N),
so figures/report use a COLLAPSED 4-level regime that merges 2024_phase2 + 2025_rescission
into `2024_phase2_plus`. The CSV keeps BOTH the 5-level fine and 4-level collapsed rows.

# SYNC: replicates two things from 08_create_figures.R verbatim — any change there must
# be mirrored here (the HARD cross-check below turns any drift into a loud failure):
#   (1) the headline duration frame: status in {complete_clear, complete_with_proxy},
#       both dates non-NA, both granularities != "year", month-granularity dates imputed
#       to the mid-month (floor_date(x,"month") + 14 days), duration_days = dec_mid - init_mid,
#       duration_days >= 0.
#   (2) duration_summary_stats(): the percentile/months column set, months = round(days/30.44, 1).

Outputs (phase2/output/deliverable04/diagnostics/):
  d4_duration_by_ceq_regime.csv            — decision-anchored; regime_level in
                                             {collapsed, fine, sensitivity}; anchor="decision"
  d4_duration_by_ceq_regime_initiation.csv — initiation-anchored sensitivity; fine + collapsed
  d4_ceq_regime_composition.csv            — per process x collapsed regime: n, completeness,
                                             pct_decarb/fossil/other, top-agency share

Usage:
  conda run -n nepa python phase2/code/deliverable04/ceq_regime/01_build_tables.py
"""
import os
import sys
from pathlib import Path

import duckdb

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

PHASE2 = Path(__file__).resolve().parents[3]
TL = PHASE2 / "data" / "analysis" / "timeline"
DATES = (TL / "timeline_project_dates.parquet").as_posix()
IDX = (TL / "timeline_document_index.parquet").as_posix()
COMBINED = (PHASE2.parent / "phase1" / "data" / "analysis" / "projects_combined.parquet").as_posix()

DIAG = PHASE2 / "output" / "deliverable04" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)

# --- Regime cut dates + rescission effective date (see module docstring) -----
CEQ_TRUMP = "2020-09-14"
CEQ_PHASE1 = "2022-05-20"
CEQ_PHASE2 = "2024-07-01"
CEQ_RESCIND = "2025-04-11"
FRA_CUT = "2023-06-03"           # for reference / report; not a regime cut here
RECENT_1978_START = "2015-01-01"  # 1978-rule recent-window comparator start
MIN_N = 30
MONTHS = 30.44                    # 08_create_figures.R months conversion — do NOT change

PROCESS_LEVELS = ["CE", "EA", "EIS"]

# SQL expression that tags the 5-level FINE regime from an arbitrary date column.
def fine_expr(col: str) -> str:
    return f"""
        CASE
            WHEN {col} <  DATE '{CEQ_TRUMP}'   THEN '1978'
            WHEN {col} <  DATE '{CEQ_PHASE1}'  THEN '2020_trump'
            WHEN {col} <  DATE '{CEQ_PHASE2}'  THEN '2022_phase1'
            WHEN {col} <  DATE '{CEQ_RESCIND}' THEN '2024_phase2'
            ELSE '2025_rescission'
        END"""


# SQL expression that tags the 4-level COLLAPSED regime (2024 + 2025 merged).
def collapsed_expr(col: str) -> str:
    return f"""
        CASE
            WHEN {col} <  DATE '{CEQ_TRUMP}'  THEN '1978'
            WHEN {col} <  DATE '{CEQ_PHASE1}' THEN '2020_trump'
            WHEN {col} <  DATE '{CEQ_PHASE2}' THEN '2022_phase1'
            ELSE '2024_phase2_plus'
        END"""


def build_headline(con: duckdb.DuckDBPyConnection) -> None:
    """Replicate 08_create_figures.R's headline duration frame (see SYNC note)."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE headline AS
        WITH base AS (
            SELECT project_id, process_type,
                   CAST(initiation_date AS DATE) AS init_d,
                   initiation_date_granularity   AS ig,
                   CAST(decision_date AS DATE)   AS dec_d,
                   decision_date_granularity     AS dg,
                   timeline_status
            FROM read_parquet('{DATES}')
        ), imputed AS (
            SELECT project_id, process_type, init_d, dec_d,
                   -- month-granularity dates imputed to the mid-month (floor + 14 days)
                   CASE WHEN ig = 'month' THEN CAST(date_trunc('month', init_d) AS DATE) + 14
                        ELSE init_d END AS init_mid,
                   CASE WHEN dg = 'month' THEN CAST(date_trunc('month', dec_d) AS DATE) + 14
                        ELSE dec_d END AS dec_mid
            FROM base
            WHERE timeline_status IN ('complete_clear', 'complete_with_proxy')
              AND init_d IS NOT NULL AND dec_d IS NOT NULL
              AND ig <> 'year' AND dg <> 'year'
        )
        SELECT project_id, process_type, init_d, dec_d,
               datediff('day', init_mid, dec_mid) AS duration_days
        FROM imputed
        WHERE datediff('day', init_mid, dec_mid) >= 0
    """)


def summarize(con, regime_sql: str, anchor: str, regime_level: str, where: str = "") -> "pandas.DataFrame":
    """duration_summary_stats() replica grouped by process_type x regime."""
    df = con.execute(f"""
        SELECT process_type,
               {regime_sql} AS ceq_regime,
               COUNT(*)                                                   AS n,
               median(duration_days)                                      AS median_days,
               quantile_cont(duration_days, 0.10)                         AS p10_days,
               quantile_cont(duration_days, 0.25)                         AS p25_days,
               quantile_cont(duration_days, 0.75)                         AS p75_days,
               quantile_cont(duration_days, 0.90)                         AS p90_days,
               avg(duration_days)                                         AS mean_days,
               avg(CASE WHEN duration_days < 365     THEN 1.0 ELSE 0.0 END) AS pct_lt_1y,
               avg(CASE WHEN duration_days > 5 * 365 THEN 1.0 ELSE 0.0 END) AS pct_gt_5y
        FROM headline
        {where}
        GROUP BY 1, 2
    """).df()
    df["median_months"] = (df["median_days"] / MONTHS).round(1)
    df["p10_months"] = (df["p10_days"] / MONTHS).round(1)
    df["p90_months"] = (df["p90_days"] / MONTHS).round(1)
    df["anchor"] = anchor
    df["regime_level"] = regime_level
    df["display"] = df["n"] >= MIN_N
    return df


def order_regime(df):
    """Stable sort: process CE/EA/EIS then chronological regime."""
    reg_order = {"1978": 0, "2020_trump": 1, "2022_phase1": 2,
                 "2024_phase2": 3, "2024_phase2_plus": 3, "2025_rescission": 4,
                 "1978_recent_2015_2020": 5}
    proc_order = {p: i for i, p in enumerate(PROCESS_LEVELS)}
    df = df.copy()
    df["_p"] = df["process_type"].map(proc_order)
    df["_r"] = df["ceq_regime"].map(reg_order)
    return df.sort_values(["_p", "_r"]).drop(columns=["_p", "_r"])


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    build_headline(con)

    # ---- CSV 1: decision-anchored (collapsed + fine + sensitivity) ----------
    dec_collapsed = summarize(con, collapsed_expr("dec_d"), "decision", "collapsed")
    dec_fine = summarize(con, fine_expr("dec_d"), "decision", "fine")
    # 1978-rule recent-window comparator (2015-01-01 .. day before Trump rule)
    sens = summarize(
        con, "'1978_recent_2015_2020'", "decision", "sensitivity",
        where=f"WHERE dec_d >= DATE '{RECENT_1978_START}' AND dec_d < DATE '{CEQ_TRUMP}'",
    )
    decision = order_regime(__import__("pandas").concat([dec_collapsed, dec_fine, sens], ignore_index=True))
    decision.to_csv(DIAG / "d4_duration_by_ceq_regime.csv", index=False)
    print(f"Wrote d4_duration_by_ceq_regime.csv ({len(decision)} rows)")

    # ---- CSV 2: initiation-anchored sensitivity (collapsed + fine) ----------
    ini_collapsed = summarize(con, collapsed_expr("init_d"), "initiation", "collapsed")
    ini_fine = summarize(con, fine_expr("init_d"), "initiation", "fine")
    initiation = order_regime(__import__("pandas").concat([ini_collapsed, ini_fine], ignore_index=True))
    initiation.to_csv(DIAG / "d4_duration_by_ceq_regime_initiation.csv", index=False)
    print(f"Wrote d4_duration_by_ceq_regime_initiation.csv ({len(initiation)} rows)")

    # ---- CSV 3: composition per process x collapsed regime ------------------
    # Energy type (recode Clean->Decarb as 08 does) + one lead agency per project.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE energy AS
        SELECT project_id,
               CASE COALESCE(project_energy_type, 'Other')
                    WHEN 'Clean' THEN 'Decarb' ELSE COALESCE(project_energy_type, 'Other') END AS energy_type
        FROM read_parquet('{COMBINED}')
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE agency AS
        SELECT project_id,
               any_value(lead_agency_harmonized) FILTER (WHERE lead_agency_harmonized IS NOT NULL) AS agency
        FROM read_parquet('{IDX}')
        GROUP BY project_id
    """)
    # Denominator: ALL projects with a decision date in each collapsed regime window
    # (any status/granularity) — lets `completeness` expose the post-2024 coverage ramp.
    decided = con.execute(f"""
        SELECT process_type, {collapsed_expr("CAST(decision_date AS DATE)")} AS ceq_regime,
               COUNT(*) AS n_decided
        FROM read_parquet('{DATES}')
        WHERE decision_date IS NOT NULL
        GROUP BY 1, 2
    """).df()

    comp_base = con.execute(f"""
        SELECT h.process_type,
               {collapsed_expr("h.dec_d")} AS ceq_regime,
               h.project_id,
               COALESCE(e.energy_type, 'Other') AS energy_type,
               a.agency
        FROM headline h
        LEFT JOIN energy e USING (project_id)
        LEFT JOIN agency a USING (project_id)
    """).df()

    pd = __import__("pandas")
    import re

    def clean_agency(a: str) -> str:
        # lead_agency_harmonized is stored as a JSON-list-like string, e.g. '["Department of Energy"]'
        m = re.search(r'"([^"]+)"', str(a))
        return m.group(1) if m else str(a).strip('[]"')

    rows = []
    for (proc, reg), g in comp_base.groupby(["process_type", "ceq_regime"]):
        n = len(g)
        ag = g["agency"].dropna().map(clean_agency)
        if len(ag):
            top_agency = ag.value_counts().index[0]
            top_agency_share = round(ag.value_counts().iloc[0] / n, 3)
        else:
            top_agency, top_agency_share = "", 0.0
        rows.append({
            "process_type": proc,
            "ceq_regime": reg,
            "n": n,
            "pct_decarb": round((g["energy_type"] == "Decarb").mean(), 3),
            "pct_fossil": round((g["energy_type"] == "Fossil").mean(), 3),
            "pct_other": round((g["energy_type"] == "Other").mean(), 3),
            "top_agency": top_agency,
            "top_agency_share": top_agency_share,
        })
    comp = pd.DataFrame(rows).merge(decided, on=["process_type", "ceq_regime"], how="left")
    comp["completeness"] = (comp["n"] / comp["n_decided"]).round(3)
    comp = order_regime(comp)[
        ["process_type", "ceq_regime", "n", "n_decided", "completeness",
         "pct_decarb", "pct_fossil", "pct_other", "top_agency", "top_agency_share"]
    ]
    comp.to_csv(DIAG / "d4_ceq_regime_composition.csv", index=False)
    print(f"Wrote d4_ceq_regime_composition.csv ({len(comp)} rows)")

    # ---- HARD cross-check against 08's headline frame -----------------------
    summ_path = DIAG / "d4_duration_summary.csv"
    if not summ_path.exists():
        sys.exit(f"HARD CHECK FAILED: {summ_path} missing — run 08_create_figures.R first.")
    summ = pd.read_csv(summ_path).set_index("process_type")["n"].to_dict()

    fine = dec_fine.set_index(["process_type", "ceq_regime"])["n"].to_dict()
    coll = dec_collapsed.set_index(["process_type", "ceq_regime"])["n"].to_dict()
    ok = True
    for proc in PROCESS_LEVELS:
        fine_sum = sum(v for (p, _), v in fine.items() if p == proc)
        headline_n = int(summ.get(proc, -1))
        if fine_sum != headline_n:
            print(f"  MISMATCH {proc}: fine-regime sum {fine_sum} != d4_duration_summary n {headline_n}")
            ok = False
        exp = fine.get((proc, "2024_phase2"), 0) + fine.get((proc, "2025_rescission"), 0)
        got = coll.get((proc, "2024_phase2_plus"), 0)
        if exp != got:
            print(f"  MISMATCH {proc}: collapsed 2024+ {got} != fine 2024+2025 {exp}")
            ok = False
    if not ok:
        sys.exit(1)

    # ---- Per-regime n table (verified reference) ----------------------------
    print("\nHARD CHECK PASSED. Decision-anchored fine-regime n's:")
    piv = dec_fine.pivot_table(index="ceq_regime", columns="process_type",
                               values="n", fill_value=0)
    piv = piv.reindex(["1978", "2020_trump", "2022_phase1", "2024_phase2", "2025_rescission"])
    piv = piv[[c for c in PROCESS_LEVELS if c in piv.columns]]
    print(piv.to_string())
    print("\nCollapsed 2024_phase2_plus n's:")
    for proc in PROCESS_LEVELS:
        print(f"  {proc}: {coll.get((proc, '2024_phase2_plus'), 0)}")
    print("\nPer-process totals (== d4_duration_summary.csv):")
    for proc in PROCESS_LEVELS:
        print(f"  {proc}: {int(summ.get(proc, -1))}")
    con.close()


if __name__ == "__main__":
    main()
