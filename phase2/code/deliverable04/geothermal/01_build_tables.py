#!/usr/bin/env python
"""Build geothermal-timeline-by-BLM-field-office tables for D4 (§sec-geothermal).

Answers the deliverable-scope request "geothermal review timelines by BLM field office"
— but the honest answer is that the office-level slice is too thin to compare, so this
builder reframes the geothermal universe into THREE TIERS and shifts the duration work to
the wider tiers where the sample supports it:

  1. office_matched_blm (61) — BLM-led geothermal projects that carry a parseable
     DOI-BLM field-office code (the literal "by field office" inventory).
  2. unmatched_blm (48)      — BLM-led geothermal with NO parseable office code
     (the "no office match" baseline). Office recovery is a DEAD END: a regex sweep of
     every unmatched project's file names + case numbers finds an office-like code in only
     2 of 48 (recovery ceiling +2), so no recovery code is written — the constant is cited.
  3. doe_other (764)         — everything not BLM-led, overwhelmingly DOE grant-era CEs.

The key data finding the section leads with: the `Geothermal` tag spans TWO WORLDS —
BLM western resource geothermal (leases/exploration/plants; NV/CA/UT; median CE ~34d) and
DOE grant-era geothermal (ARRA heat-pump/research CEs, 2010-2012 surge, nationwide incl.
CT/OK/CO/PA; median CE ~8d). Pooling them is meaningless; the split IS the structure. The
wider tiers unlock 375 complete CE timelines (vs 35 office-matched), enabling a state
bubble map and a decision-year timeline chart that office-level data cannot support.

# SYNC: replicates the headline duration frame from 08_create_figures.R verbatim — any
# change there must be mirrored here:
#   status in {complete_clear, complete_with_proxy}; both dates non-NA; both granularities
#   != "year"; month-granularity dates imputed to the mid-month (floor_date(x,"month")+14);
#   duration_days = dec_mid - init_mid; duration_days >= 0; months = round(days/30.44, 1).
# Document-anchored vs register-anchored uses INITIATION source (initiation_source_type):
# "metadata" = register/agency-API administrative start (register-anchored); anything else
# = document-anchored. This mirrors the src3() register/doc split in 08_create_figures.R
# (which keys the register artifact on the initiation date).

Outputs (phase2/output/deliverable04/diagnostics/):
  d4_geothermal_universe.csv        — funnel + tier + lead-agency rows (n, n_ce, n_ea, n_eis)
  d4_geothermal_office_counts.csv   — per office x process (+ ALL rollup) + two baseline rows
  d4_geothermal_state_map.csv       — per state x cohort, CE only (n, medians, centroid, display)
  d4_geothermal_timeline_points.csv — CE annual medians + every EA/EIS project as one row
  d4_geothermal_office_floor.csv    — office-floor summary (0,0,9,2) + recovery_candidates

Usage:
  conda run -n nepa python phase2/code/deliverable04/geothermal/01_build_tables.py
"""
import os
import re
import sys
from pathlib import Path

import duckdb

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

PHASE2 = Path(__file__).resolve().parents[3]
TL = PHASE2 / "data" / "analysis" / "timeline"
DATES = (TL / "timeline_project_dates.parquet").as_posix()
IDX = (TL / "timeline_document_index.parquet").as_posix()
REVIEWS = (PHASE2 / "data" / "analysis" / "deliverable03" / "projects_nepa_reviews.parquet").as_posix()
OFFICES = (PHASE2 / "data" / "analysis" / "deliverable04" / "blm_field_offices.parquet").as_posix()
DOE_REG = (PHASE2 / "data" / "analysis" / "doe_register" / "doe_cx_register.parquet").as_posix()

DIAG = PHASE2 / "output" / "deliverable04" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)

MONTHS = 30.44                 # 08_create_figures.R months conversion — do NOT change
PROCESS_LEVELS = ["CE", "EA", "EIS"]
OFFICE_FLOOR = 10              # minimum complete CE reviews for a stable per-office median

# DOE non-BLM geothermal register-office coverage (verified live 2026-07-20).
DOE_OFFICE_COVERAGE = 456      # of 764 non-BLM geothermal projects carry a canonical register office

# --- DOE CX-register office harmonization (identical rule to field_office/01b_build_doe_offices.py) ---
# 3-part rule: (1) segment after the LAST comma, dash-normalized, trailing-ellipsis stripped
# (with a prefix fallback so a truncated-to-junk suffix recovers the program-office prefix);
# (2) exact vocab match; (3) truncated-variant match once the segment clears a 6-char floor.
CANON_MIN_PREFIX = 6
DOE_VOCAB = [
    "National Energy Technology Laboratory", "Golden Field Office",
    "Savannah River Operations Office", "Bonneville Power Administration",
    "River Protection-Richland Operations Office",
    "Western Area Power Administration-Rocky Mountain Region",
    "Western Area Power Administration-Desert Southwest Region",
    "Western Area Power Administration-Upper Great Plains Region",
    "Western Area Power Administration-Sierra Nevada Region",
    "Advanced Research Projects Agency-Energy", "Idaho Operations Office",
    "Argonne Site Office", "Fermi Site Office", "Sandia Site Office",
    "Nuclear Energy", "Energy Efficiency and Renewable Energy",
]
_DOE_VOCAB_L = [(c, c.lower()) for c in DOE_VOCAB]


def _norm_dash(x: str) -> str:
    return re.sub(r"\s*-\s*", "-", x.strip())


def doe_office_canon(office_raw):
    """Harmonize a raw DOE register office string to a canonical office (or None)."""
    if office_raw is None:
        return None
    s = str(office_raw).strip()
    if not s:
        return None
    seg = ""
    for part in reversed(s.split(",")):
        cand = re.sub(r"[.…\s]+$", "", _norm_dash(part))
        if cand:
            seg = cand
            break
    if not seg:
        seg = _norm_dash(s) or s
    if not seg:
        return None
    segl = seg.lower()
    for c, cl in _DOE_VOCAB_L:
        if segl == cl:
            return c
        if len(segl) >= CANON_MIN_PREFIX and (segl.startswith(cl) or cl.startswith(segl)):
            return c
    return seg

# Office-recovery dead end (documented constant, see module docstring): a regex sweep of all
# 48 unmatched BLM geothermal projects' file names + case numbers finds a genuine office-like
# code in only 2 (EA-1680, CA-170). Recovery ceiling +2 — not worth code (per approved plan).
RECOVERY_CANDIDATES = 2

# Canonical state centroids (base-R `state.center`, lower-48; lat, lon). Alaska/Hawaii carry
# R's map-inset offsets, not true positions, so they are flagged undrawable below.
STATE_CENTROIDS = {
    "Alabama": (32.590, -86.751), "Arizona": (34.219, -111.625), "Arkansas": (34.734, -92.299),
    "California": (36.534, -119.773), "Colorado": (38.678, -105.513), "Connecticut": (41.593, -72.357),
    "Delaware": (38.678, -74.984), "Florida": (27.874, -81.685), "Georgia": (32.333, -83.374),
    "Idaho": (43.565, -113.930), "Illinois": (40.050, -89.378), "Indiana": (40.050, -86.081),
    "Iowa": (41.936, -93.371), "Kansas": (38.420, -98.116), "Kentucky": (37.392, -84.767),
    "Louisiana": (30.618, -92.272), "Maine": (45.623, -68.980), "Maryland": (39.278, -76.646),
    "Massachusetts": (42.364, -71.580), "Michigan": (43.136, -84.687), "Minnesota": (46.394, -94.604),
    "Mississippi": (32.676, -89.806), "Missouri": (38.335, -92.514), "Montana": (46.823, -109.320),
    "Nebraska": (41.336, -99.590), "Nevada": (39.106, -116.851), "New Hampshire": (43.393, -71.392),
    "New Jersey": (39.964, -74.234), "New Mexico": (34.476, -105.942), "New York": (43.136, -75.145),
    "North Carolina": (35.419, -78.469), "North Dakota": (47.252, -100.099), "Ohio": (40.221, -82.596),
    "Oklahoma": (35.505, -97.124), "Oregon": (43.908, -120.068), "Pennsylvania": (40.907, -77.450),
    "Rhode Island": (41.593, -71.124), "South Carolina": (33.619, -80.506), "South Dakota": (44.337, -99.724),
    "Tennessee": (35.677, -86.456), "Texas": (31.390, -98.786), "Utah": (39.106, -111.330),
    "Vermont": (44.251, -72.545), "Virginia": (37.563, -78.201), "Washington": (47.423, -119.746),
    "West Virginia": (38.420, -80.666), "Wisconsin": (44.594, -89.994), "Wyoming": (43.050, -107.256),
}
LOWER_48 = set(STATE_CENTROIDS)  # states that can be drawn on the lower-48 polygon map


def first_json(s: str) -> str:
    """First quoted token of a JSON-list-like string, e.g. '["Nevada"]' -> 'Nevada'."""
    if s is None:
        return ""
    m = re.search(r'"([^"]+)"', str(s))
    return m.group(1) if m else str(s).strip('[]"').strip()


def build_base(con: duckdb.DuckDBPyConnection) -> None:
    """Register the geothermal universe, per-project lead/state, headline frame, cohorts."""
    # Geothermal universe (project-level) with the review process type.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE geo AS
        SELECT project_id, process_type
        FROM read_parquet('{REVIEWS}')
        WHERE tech_group = 'Geothermal'
    """)
    # One lead-agency + state string per project from the document index (JSON-list strings).
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE meta AS
        SELECT project_id,
               any_value(lead_agency_harmonized) AS lead_raw,
               any_value(project_state)          AS state_raw
        FROM read_parquet('{IDX}')
        GROUP BY project_id
    """)
    # Cohort/tier per geothermal project.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cohort AS
        SELECT g.project_id, g.process_type, m.lead_raw, m.state_raw,
               (m.lead_raw LIKE '%Bureau of Land Management%') AS is_blm,
               (g.project_id IN (SELECT project_id FROM read_parquet('{OFFICES}'))) AS office_matched
        FROM geo g
        LEFT JOIN meta m USING (project_id)
    """)
    # Headline duration frame (SYNC with 08_create_figures.R), joined to geothermal only.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE headline AS
        WITH base AS (
            SELECT d.project_id, d.process_type,
                   CAST(d.initiation_date AS DATE) AS init_d, d.initiation_date_granularity AS ig,
                   CAST(d.decision_date AS DATE)   AS dec_d,  d.decision_date_granularity   AS dg,
                   d.initiation_source_type AS init_src, d.timeline_status AS ts
            FROM read_parquet('{DATES}') d
            WHERE d.project_id IN (SELECT project_id FROM geo)
        ), imputed AS (
            SELECT project_id, process_type, dec_d, init_src,
                   CASE WHEN ig = 'month' THEN CAST(date_trunc('month', init_d) AS DATE) + 14 ELSE init_d END AS init_mid,
                   CASE WHEN dg = 'month' THEN CAST(date_trunc('month', dec_d)  AS DATE) + 14 ELSE dec_d  END AS dec_mid
            FROM base
            WHERE ts IN ('complete_clear', 'complete_with_proxy')
              AND init_d IS NOT NULL AND dec_d IS NOT NULL
              AND ig <> 'year' AND dg <> 'year'
        )
        SELECT h.project_id, h.process_type,
               datediff('day', h.init_mid, h.dec_mid) AS dur_days,
               year(h.dec_mid) AS dec_year,
               -- register-anchored iff the INITIATION date came from the agency register/API
               (h.init_src IS NOT DISTINCT FROM 'metadata') AS init_register,
               c.is_blm, c.office_matched, c.state_raw
        FROM imputed h
        LEFT JOIN cohort c USING (project_id)
        WHERE datediff('day', h.init_mid, h.dec_mid) >= 0
    """)


def proc_split(con, table: str, where: str = "") -> dict:
    """Return {'n':, 'n_ce':, 'n_ea':, 'n_eis':} for a table filtered by `where`."""
    w = f"WHERE {where}" if where else ""
    row = con.execute(f"""
        SELECT COUNT(*) n,
               COUNT(*) FILTER (WHERE process_type='CE')  n_ce,
               COUNT(*) FILTER (WHERE process_type='EA')  n_ea,
               COUNT(*) FILTER (WHERE process_type='EIS') n_eis
        FROM {table} {w}
    """).fetchone()
    return {"n": row[0], "n_ce": row[1], "n_ea": row[2], "n_eis": row[3]}


def build_universe(con) -> "pandas.DataFrame":
    import pandas as pd
    rows = []

    def add(stage, d):
        rows.append({"stage": stage, "n": d["n"], "n_ce": d["n_ce"], "n_ea": d["n_ea"], "n_eis": d["n_eis"]})

    add("total_geothermal", proc_split(con, "cohort"))
    add("blm_led",          proc_split(con, "cohort", "is_blm"))
    add("office_matched",   proc_split(con, "cohort", "office_matched"))
    add("unmatched_blm",    proc_split(con, "cohort", "is_blm AND NOT office_matched"))
    add("doe_other",        proc_split(con, "cohort", "NOT is_blm OR is_blm IS NULL"))
    add("complete_timeline_total", proc_split(con, "headline"))

    n_off = con.execute(f"SELECT COUNT(DISTINCT office_code) FROM read_parquet('{OFFICES}') o "
                        f"WHERE o.project_id IN (SELECT project_id FROM geo)").fetchone()[0]
    rows.append({"stage": "distinct_offices", "n": n_off, "n_ce": 0, "n_ea": 0, "n_eis": 0})

    # Lead-agency breakdown (first agency of the JSON list): top 5 by n + Other.
    lead = con.execute("SELECT project_id, process_type, lead_raw FROM cohort").df()
    lead["agency"] = lead["lead_raw"].map(first_json).replace("", "(unknown)")
    counts = lead["agency"].value_counts()
    top5 = list(counts.index[:5])
    for ag in top5:
        sub = lead[lead["agency"] == ag]
        rows.append({"stage": f"lead_agency:{ag}", "n": len(sub),
                     "n_ce": int((sub["process_type"] == "CE").sum()),
                     "n_ea": int((sub["process_type"] == "EA").sum()),
                     "n_eis": int((sub["process_type"] == "EIS").sum())})
    other = lead[~lead["agency"].isin(top5)]
    rows.append({"stage": "lead_agency:Other", "n": len(other),
                 "n_ce": int((other["process_type"] == "CE").sum()),
                 "n_ea": int((other["process_type"] == "EA").sum()),
                 "n_eis": int((other["process_type"] == "EIS").sum())})
    return pd.DataFrame(rows)


def build_office_counts(con) -> "pandas.DataFrame":
    """Per office x process (+ ALL rollup) counts, medians, decision-year span, and the two
    baseline pseudo-office rows. n_parsed = geothermal projects at that office; n_complete /
    n_doc / n_register / median_days_all computed over the headline duration frame."""
    import pandas as pd
    # n_parsed per office x process (from the office map joined to geothermal).
    parsed = con.execute(f"""
        SELECT o.office_code, o.state, g.process_type, COUNT(*) AS n_parsed
        FROM read_parquet('{OFFICES}') o
        JOIN geo g USING (project_id)
        GROUP BY 1, 2, 3
    """).df()
    # Completeness stats per office x process (headline frame, office-matched only).
    stats = con.execute("""
        SELECT o.office_code, h.process_type,
               COUNT(*)                                                       AS n_complete,
               COUNT(*) FILTER (WHERE NOT h.init_register)                    AS n_doc,
               COUNT(*) FILTER (WHERE h.init_register)                        AS n_register,
               median(h.dur_days)                                            AS median_days_all,
               MIN(h.dec_year)                                                AS dec_year_min,
               MAX(h.dec_year)                                                AS dec_year_max
        FROM headline h
        JOIN read_parquet('%s') o USING (project_id)
        GROUP BY 1, 2
    """ % OFFICES).df()

    per = parsed.merge(stats, on=["office_code", "process_type"], how="left")
    # ALL rollup per office.
    parsed_all = parsed.groupby(["office_code", "state"], as_index=False)["n_parsed"].sum()
    stats_all = con.execute("""
        SELECT o.office_code,
               COUNT(*) n_complete,
               COUNT(*) FILTER (WHERE NOT h.init_register) n_doc,
               COUNT(*) FILTER (WHERE h.init_register)     n_register,
               median(h.dur_days) median_days_all,
               MIN(h.dec_year) dec_year_min, MAX(h.dec_year) dec_year_max
        FROM headline h JOIN read_parquet('%s') o USING (project_id)
        GROUP BY 1
    """ % OFFICES).df()
    roll = parsed_all.merge(stats_all, on="office_code", how="left")
    roll["process_type"] = "ALL"

    def baseline(code, where):
        p = proc_split(con, "cohort", where)
        s = con.execute(f"""
            SELECT h.process_type, COUNT(*) n_complete,
                   COUNT(*) FILTER (WHERE NOT h.init_register) n_doc,
                   COUNT(*) FILTER (WHERE h.init_register)     n_register,
                   median(h.dur_days) median_days_all,
                   MIN(h.dec_year) dec_year_min, MAX(h.dec_year) dec_year_max
            FROM headline h WHERE {where.replace('is_blm', 'h.is_blm').replace('office_matched', 'h.office_matched')}
            GROUP BY 1
        """).df()
        parsed_b = con.execute(f"""
            SELECT process_type, COUNT(*) n_parsed FROM cohort WHERE {where} GROUP BY 1
        """).df()
        rows = []
        for _, pr in parsed_b.iterrows():
            sr = s[s["process_type"] == pr["process_type"]]
            rows.append({
                "office_code": code, "state": "", "process_type": pr["process_type"],
                "n_parsed": int(pr["n_parsed"]),
                "n_complete": int(sr["n_complete"].iloc[0]) if len(sr) else 0,
                "n_doc": int(sr["n_doc"].iloc[0]) if len(sr) else 0,
                "n_register": int(sr["n_register"].iloc[0]) if len(sr) else 0,
                "median_days_all": float(sr["median_days_all"].iloc[0]) if len(sr) else None,
                "dec_year_min": sr["dec_year_min"].iloc[0] if len(sr) else None,
                "dec_year_max": sr["dec_year_max"].iloc[0] if len(sr) else None,
            })
        # ALL rollup for the baseline.
        s_all = con.execute(f"""
            SELECT COUNT(*) n_complete, COUNT(*) FILTER (WHERE NOT h.init_register) n_doc,
                   COUNT(*) FILTER (WHERE h.init_register) n_register, median(h.dur_days) median_days_all,
                   MIN(h.dec_year) dec_year_min, MAX(h.dec_year) dec_year_max
            FROM headline h WHERE {where.replace('is_blm', 'h.is_blm').replace('office_matched', 'h.office_matched')}
        """).fetchone()
        rows.append({
            "office_code": code, "state": "", "process_type": "ALL", "n_parsed": p["n"],
            "n_complete": int(s_all[0]), "n_doc": int(s_all[1]), "n_register": int(s_all[2]),
            "median_days_all": float(s_all[3]) if s_all[3] is not None else None,
            "dec_year_min": s_all[4], "dec_year_max": s_all[5],
        })
        return pd.DataFrame(rows)

    base_noff = baseline("(no office match)", "is_blm AND NOT office_matched")
    base_doe  = baseline("(DOE & other)",     "NOT is_blm OR is_blm IS NULL")

    out = pd.concat([per, roll, base_noff, base_doe], ignore_index=True)
    for c in ["n_complete", "n_doc", "n_register"]:
        out[c] = out[c].fillna(0).astype(int)
    return out[["office_code", "state", "process_type", "n_parsed", "n_complete",
                "n_doc", "n_register", "median_days_all", "dec_year_min", "dec_year_max"]]


def build_state_map(con) -> "pandas.DataFrame":
    """Per state x cohort (CE only): n, medians, centroid, display flag (n>=3 & lower-48).

    Restricted to SINGLE-STATE projects — a project whose `project_state` list carries exactly
    one state. Multi-state geothermal projects (10 complete CEs) are genuinely ambiguous to
    place on a state map and are dropped rather than assigned to a first-listed state."""
    import pandas as pd
    # single-state iff the JSON-list string contains no comma (one element).
    df = con.execute("""
        SELECT regexp_extract(state_raw, '"([^"]+)"', 1) AS state,
               CASE WHEN is_blm THEN 'BLM' ELSE 'DOE/Other' END AS cohort,
               COUNT(*) AS n_complete, median(dur_days) AS median_days
        FROM headline
        WHERE process_type = 'CE'
          AND state_raw IS NOT NULL AND strpos(state_raw, ',') = 0
        GROUP BY 1, 2
    """).df()
    df["median_months"] = (df["median_days"] / MONTHS).round(1)
    df["lat"] = df["state"].map(lambda s: STATE_CENTROIDS.get(s, (None, None))[0])
    df["lon"] = df["state"].map(lambda s: STATE_CENTROIDS.get(s, (None, None))[1])
    df["display"] = (df["n_complete"] >= 3) & df["state"].isin(LOWER_48)
    df = df[["state", "cohort", "n_complete", "median_days", "median_months", "lat", "lon", "display"]]
    return df.sort_values(["cohort", "n_complete"], ascending=[True, False]).reset_index(drop=True)


def build_timeline_points(con) -> "pandas.DataFrame":
    """CE annual medians PER COHORT (row_type=ce_year, cohort in {BLM, DOE/Other}) + every
    EA/EIS project as one row (ea_eis_project). Splitting the CE line by cohort lets Fig C draw
    the two geothermal worlds (BLM western-resource vs DOE grant-era) as separate median series."""
    import pandas as pd
    ce = con.execute("""
        SELECT 'ce_year' AS row_type, 'CE' AS process_type, dec_year AS yr,
               COUNT(*) AS n, round(median(dur_days) / %f, 1) AS median_months,
               NULL::DOUBLE AS months,
               CASE WHEN is_blm THEN 'BLM' ELSE 'DOE/Other' END AS cohort
        FROM headline WHERE process_type = 'CE'
        GROUP BY dec_year, CASE WHEN is_blm THEN 'BLM' ELSE 'DOE/Other' END
    """ % MONTHS).df()
    ea_eis = con.execute("""
        SELECT 'ea_eis_project' AS row_type, process_type, dec_year AS yr,
               NULL::BIGINT AS n, NULL::DOUBLE AS median_months,
               round(dur_days / %f, 1) AS months,
               CASE WHEN is_blm THEN 'BLM' ELSE 'DOE/Other' END AS cohort
        FROM headline WHERE process_type IN ('EA', 'EIS')
    """ % MONTHS).df()
    out = pd.concat([ce, ea_eis], ignore_index=True)
    return out.sort_values(["row_type", "process_type", "yr"]).reset_index(drop=True)


def build_office_floor(con) -> "pandas.DataFrame":
    import pandas as pd
    per = con.execute("""
        SELECT o.office_code,
               COUNT(*) FILTER (WHERE h.process_type='CE')                              AS ce_complete,
               COUNT(*) FILTER (WHERE h.process_type='CE' AND NOT h.init_register)       AS ce_doc
        FROM headline h JOIN read_parquet('%s') o USING (project_id)
        GROUP BY 1
    """ % OFFICES).df()
    # Pooled two-worlds CE medians (BLM vs everything-else) — the section's headline contrast.
    tw = con.execute("""
        SELECT CASE WHEN is_blm THEN 'blm' ELSE 'nonblm' END AS coh,
               COUNT(*) AS n, median(dur_days) AS med
        FROM headline WHERE process_type = 'CE' GROUP BY 1
    """).df().set_index("coh")
    return pd.DataFrame([{
        "n_offices_ge_floor":     int((per["ce_complete"] >= OFFICE_FLOOR).sum()),
        "n_offices_ge_floor_doc": int((per["ce_doc"] >= OFFICE_FLOOR).sum()),
        "max_office_complete":    int(per["ce_complete"].max()),
        "max_office_doc":         int(per["ce_doc"].max()),
        "recovery_candidates":    RECOVERY_CANDIDATES,
        "office_floor":           OFFICE_FLOOR,
        "blm_ce_n":               int(tw.loc["blm", "n"]),
        "blm_ce_median_days":     round(float(tw.loc["blm", "med"]), 1),
        "nonblm_ce_n":            int(tw.loc["nonblm", "n"]),
        "nonblm_ce_median_days":  round(float(tw.loc["nonblm", "med"]), 1),
    }])


def build_doe_office_counts(con) -> "pandas.DataFrame":
    """Non-BLM geothermal projects linked to a DOE CX-register office (INTEGER cx join, mode over
    canonical offices). Per office: n_parsed (all timeline states), n_ce, n_ce_complete, median_days
    (complete CE). Plus an ALL rollup = the with-office coverage (expected 456 of 764). The DOE
    grant tier is CE-only, so this is a CE inventory; offices are administering/grant-program
    offices (Golden, NETL, EERE-HQ, RMOTC, …), NOT BLM-style field offices."""
    import pandas as pd
    con.create_function("doe_office_canon", doe_office_canon, ["VARCHAR"], "VARCHAR",
                        null_handling="special")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE geo_doe_office AS
        WITH m AS (
            SELECT DISTINCT c.project_id, CAST(ROUND(i.cx_number) AS BIGINT) AS cxn,
                   doe_office_canon(r.office) AS office
            FROM cohort c
            JOIN read_parquet('{IDX}') i USING (project_id)
            JOIN read_parquet('{DOE_REG}') r ON CAST(ROUND(i.cx_number) AS BIGINT) = r.cx_number
            WHERE (NOT c.is_blm OR c.is_blm IS NULL)
              AND i.cx_number IS NOT NULL AND r.office IS NOT NULL AND r.office <> ''
              AND doe_office_canon(r.office) IS NOT NULL
        ), cnt AS (SELECT project_id, office, COUNT(*) AS n FROM m GROUP BY 1, 2)
        SELECT project_id, office FROM (
            SELECT project_id, office,
                   row_number() OVER (PARTITION BY project_id ORDER BY n DESC, office) AS rn
            FROM cnt
        ) WHERE rn = 1
    """)
    per = con.execute("""
        SELECT o.office,
               COUNT(*)                                                 AS n_parsed,
               COUNT(*) FILTER (WHERE g.process_type = 'CE')            AS n_ce,
               COUNT(h.project_id) FILTER (WHERE h.process_type = 'CE') AS n_ce_complete,
               median(h.dur_days) FILTER (WHERE h.process_type = 'CE')  AS median_days
        FROM geo_doe_office o
        JOIN geo g USING (project_id)
        LEFT JOIN headline h USING (project_id)
        GROUP BY 1
    """).df().sort_values("n_parsed", ascending=False)
    allrow = con.execute("""
        SELECT 'ALL' AS office, COUNT(DISTINCT o.project_id) AS n_parsed,
               COUNT(*) FILTER (WHERE g.process_type = 'CE') AS n_ce,
               COUNT(h.project_id) FILTER (WHERE h.process_type = 'CE') AS n_ce_complete,
               median(h.dur_days) FILTER (WHERE h.process_type = 'CE') AS median_days
        FROM geo_doe_office o JOIN geo g USING (project_id) LEFT JOIN headline h USING (project_id)
    """).df()
    out = pd.concat([per, allrow], ignore_index=True)
    for c in ["n_parsed", "n_ce", "n_ce_complete"]:
        out[c] = out[c].fillna(0).astype(int)
    return out[["office", "n_parsed", "n_ce", "n_ce_complete", "median_days"]]


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    build_base(con)

    universe = build_universe(con)
    universe.to_csv(DIAG / "d4_geothermal_universe.csv", index=False)
    print(f"Wrote d4_geothermal_universe.csv ({len(universe)} rows)")

    offices = build_office_counts(con)
    offices.to_csv(DIAG / "d4_geothermal_office_counts.csv", index=False)
    print(f"Wrote d4_geothermal_office_counts.csv ({len(offices)} rows)")

    smap = build_state_map(con)
    smap.to_csv(DIAG / "d4_geothermal_state_map.csv", index=False)
    print(f"Wrote d4_geothermal_state_map.csv ({len(smap)} rows)")

    tpts = build_timeline_points(con)
    tpts.to_csv(DIAG / "d4_geothermal_timeline_points.csv", index=False)
    print(f"Wrote d4_geothermal_timeline_points.csv ({len(tpts)} rows)")

    floor = build_office_floor(con)
    floor.to_csv(DIAG / "d4_geothermal_office_floor.csv", index=False)
    print(f"Wrote d4_geothermal_office_floor.csv ({len(floor)} rows)")

    doe_off = build_doe_office_counts(con)
    doe_off.to_csv(DIAG / "d4_geothermal_doe_office_counts.csv", index=False)
    print(f"Wrote d4_geothermal_doe_office_counts.csv ({len(doe_off)} rows)")

    # ---- HARD CHECKS (sys.exit(1) on any mismatch) --------------------------
    u = universe.set_index("stage")
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            print(f"  MISMATCH: {msg}")
            ok = False

    # (1) geothermal total == 873
    check(int(u.loc["total_geothermal", "n"]) == 873,
          f"total_geothermal {int(u.loc['total_geothermal','n'])} != 873")
    # (2) tiers sum to 873
    tier_sum = int(u.loc["office_matched", "n"]) + int(u.loc["unmatched_blm", "n"]) + int(u.loc["doe_other", "n"])
    check(tier_sum == 873, f"tier sum {tier_sum} != 873")
    check(int(u.loc["blm_led", "n"]) == int(u.loc["office_matched", "n"]) + int(u.loc["unmatched_blm", "n"]),
          "blm_led != office_matched + unmatched_blm")
    # (3) per-process complete sums agree across CSV 1 / 2 / 4 (CE 375 / EA 8 / EIS 14)
    exp = {"CE": 375, "EA": 8, "EIS": 14}
    for p, col in [("CE", "n_ce"), ("EA", "n_ea"), ("EIS", "n_eis")]:
        check(int(u.loc["complete_timeline_total", col]) == exp[p],
              f"universe complete {p} {int(u.loc['complete_timeline_total', col])} != {exp[p]}")
    # office_counts baseline+office ALL-process complete should sum to the same per-process totals
    oc_all = offices[offices["process_type"].isin(PROCESS_LEVELS)]
    for p in PROCESS_LEVELS:
        s = int(oc_all[oc_all["process_type"] == p]["n_complete"].sum())
        check(s == exp[p], f"office_counts complete {p} {s} != {exp[p]}")
    # timeline points: CE annual n sums to 375; EA/EIS project rows == 22
    ce_sum = int(tpts[tpts["row_type"] == "ce_year"]["n"].sum())
    check(ce_sum == 375, f"timeline CE annual sum {ce_sum} != 375")
    # per-cohort CE annual medians must sum to the cohort totals (BLM 43 / DOE-Other 332 / 375)
    ce_yr = tpts[tpts["row_type"] == "ce_year"]
    ce_blm = int(ce_yr[ce_yr["cohort"] == "BLM"]["n"].sum())
    ce_non = int(ce_yr[ce_yr["cohort"] == "DOE/Other"]["n"].sum())
    check(ce_blm == 43, f"timeline CE BLM cohort sum {ce_blm} != 43")
    check(ce_non == 332, f"timeline CE DOE/Other cohort sum {ce_non} != 332")
    ea_eis_rows = int((tpts["row_type"] == "ea_eis_project").sum())
    check(ea_eis_rows == 22, f"EA+EIS project rows {ea_eis_rows} != 22")
    ea_rows = int(((tpts["row_type"] == "ea_eis_project") & (tpts["process_type"] == "EA")).sum())
    eis_rows = int(((tpts["row_type"] == "ea_eis_project") & (tpts["process_type"] == "EIS")).sum())
    check(ea_rows == 8 and eis_rows == 14, f"EA/EIS rows {ea_rows}/{eis_rows} != 8/14")
    # (4) n_doc + n_register == n_complete everywhere
    bad = offices[(offices["n_doc"] + offices["n_register"]) != offices["n_complete"]]
    check(len(bad) == 0, f"{len(bad)} office rows where n_doc+n_register != n_complete")
    # (5) no office clears the floor (document-anchored)
    check(int(floor["n_offices_ge_floor_doc"].iloc[0]) == 0, "n_offices_ge_floor_doc != 0")
    check(int(floor["max_office_complete"].iloc[0]) == 9 and int(floor["max_office_doc"].iloc[0]) == 2,
          f"office floor max {int(floor['max_office_complete'].iloc[0])}/{int(floor['max_office_doc'].iloc[0])} != 9/2")
    # (6) DOE register-office coverage of the non-BLM geothermal tier == 456
    doe_all = int(doe_off[doe_off["office"] == "ALL"]["n_parsed"].iloc[0])
    doe_per_sum = int(doe_off[doe_off["office"] != "ALL"]["n_parsed"].sum())
    check(doe_all == DOE_OFFICE_COVERAGE, f"DOE office coverage {doe_all} != {DOE_OFFICE_COVERAGE}")
    check(doe_per_sum == DOE_OFFICE_COVERAGE, f"DOE per-office n_parsed sum {doe_per_sum} != {DOE_OFFICE_COVERAGE}")

    if not ok:
        sys.exit(1)

    # ---- Verified-reference printout ---------------------------------------
    print("\nHARD CHECK PASSED. Verified reference block:")
    print("  Funnel:")
    for s in ["total_geothermal", "blm_led", "office_matched", "unmatched_blm", "doe_other",
              "complete_timeline_total", "distinct_offices"]:
        r = u.loc[s]
        print(f"    {s:24s} n={int(r['n']):4d}  (CE {int(r['n_ce'])} / EA {int(r['n_ea'])} / EIS {int(r['n_eis'])})")
    print("  Complete timelines by cohort:")
    for lbl, where in [("BLM", "is_blm"), ("DOE/Other", "NOT is_blm OR is_blm IS NULL"),
                       ("office_matched", "office_matched")]:
        d = con.execute(f"""
            SELECT COUNT(*) n, COUNT(*) FILTER (WHERE process_type='CE') ce,
                   COUNT(*) FILTER (WHERE process_type='EA') ea,
                   COUNT(*) FILTER (WHERE process_type='EIS') eis
            FROM headline WHERE {where}""").fetchone()
        print(f"    {lbl:16s} complete={d[0]:3d}  (CE {d[1]} / EA {d[2]} / EIS {d[3]})")
    fr = floor.iloc[0]
    print(f"  Office floor (>= {int(fr['office_floor'])} complete CE): n_offices={int(fr['n_offices_ge_floor'])}, "
          f"n_offices_doc={int(fr['n_offices_ge_floor_doc'])}, "
          f"max_complete={int(fr['max_office_complete'])}, max_doc={int(fr['max_office_doc'])}, "
          f"recovery_candidates={int(fr['recovery_candidates'])}")
    print("  Two-worlds CE medians (days):")
    tw = con.execute("""
        SELECT CASE WHEN is_blm THEN 'BLM' ELSE 'DOE/Other' END coh,
               COUNT(*) n, median(dur_days) med
        FROM headline WHERE process_type='CE' GROUP BY 1 ORDER BY 1""").df()
    for _, r in tw.iterrows():
        print(f"    {r['coh']:10s} n={int(r['n'])}  median={r['med']:.0f}d")
    con.close()


if __name__ == "__main__":
    main()
