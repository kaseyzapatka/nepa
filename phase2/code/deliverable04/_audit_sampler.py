#!/usr/bin/env python
"""Part-A precision audit sampler (D4 night-2).

Builds the NEWLY-COVERED cohort (dates present in tonight's run but not in the prenight backup),
stratifies by risk flag, samples per stratum, and emits review sheets with the selected date +
the inline evidence_text that justified it. Reviewer marks each correct/wrong/ambiguous.
"""
import os, duckdb, glob
import pandas as pd
from pathlib import Path

con = duckdb.connect()
NEW = 'data/analysis/timeline/timeline_project_dates.parquet'
# Baseline for the "newly covered" cohort. Defaults to the prenight backup (overnight before/after);
# override with AUDIT_BASELINE=<path> to diff against a different snapshot (e.g. a preB_ backup, so
# the cohort is *only* what a selection change touched, not the overnight gains).
PRE = os.environ.get("AUDIT_BASELINE") or sorted(glob.glob('/Users/Dora/git/consulting/nepa/phase2/data/analysis/timeline/timeline_project_dates.prenight_*.parquet'))[-1]
OUT = Path('output/deliverable04/audit'); OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N = 22

con.execute(f"CREATE VIEW a AS SELECT * FROM '{NEW}'")
con.execute(f"""CREATE VIEW b AS SELECT project_id, initiation_date bi, decision_date bd,
  decision_source_type bd_src, initiation_source_type bi_src,
  decision_evidence_text bd_ev, initiation_evidence_text bi_ev FROM '{PRE}'""")


def print_dropped_summary():
    """DROPPED cohort: had a date in the baseline, NULL now. Breakdown by the baseline's
    source_type — the decisive check that a precision change removed false-positives (e.g.
    feis_publication / document_text) and NOT authoritative dates (register / rod)."""
    print("\n########## DROPPED cohort (had in baseline, null now) by baseline source_type ##########")
    for proc, datecol, srccol, lab in [("EIS", "bd", "bd_src", "decision"),
                                       ("CE", "bi", "bi_src", "initiation")]:
        df = con.execute(f"""
          SELECT b.{srccol} AS baseline_src, COUNT(*) c
          FROM b JOIN a USING(project_id)
          WHERE a.process_type='{proc}' AND b.{datecol} IS NOT NULL AND a.{lab}_date IS NULL
          GROUP BY 1 ORDER BY 2 DESC""").df()
        tot = int(df['c'].sum()) if not df.empty else 0
        print(f"\n[{proc}] dropped {lab}: {tot}")
        if not df.empty:
            print("  " + df.to_string(index=False).replace("\n", "\n  "))


def sample_dropped(proc, datecol, srccol, evcol, lab, n=12):
    """Show a sample of dropped rows WITH their baseline evidence, to eyeball whether the
    removed dates were genuinely false-positives."""
    df = con.execute(f"""
      SELECT a.project_id, b.{datecol} AS dropped_date, b.{srccol} AS baseline_src, b.{evcol} AS evidence
      FROM b JOIN a USING(project_id)
      WHERE a.process_type='{proc}' AND b.{datecol} IS NOT NULL AND a.{lab}_date IS NULL
      ORDER BY random() LIMIT {n}""").df()
    df.to_csv(OUT / f"audit_dropped_{proc}_{lab}.csv", index=False)
    print(f"\n########## DROPPED sample: {proc} {lab} (n={len(df)}) ##########")
    for _, r in df.iterrows():
        ev_txt = ' '.join(str(r['evidence'] or '')[:200].split())
        print(f"\n[{proc}] {r['project_id'][:12]}  dropped={r['dropped_date']}  baseline_src={r['baseline_src']}")
        print(f"  BASELINE EVIDENCE: {ev_txt}")


def print_coverage_delta():
    """Aggregate before/after coverage by process type (baseline = PRE)."""
    def cov(path):
        return con.execute(f"""
          SELECT process_type, COUNT(*) n,
            SUM((initiation_date IS NOT NULL)::INT) init_any,
            SUM((decision_date IS NOT NULL)::INT) dec_any,
            SUM((initiation_date IS NOT NULL AND decision_date IS NOT NULL)::INT) both_cov
          FROM '{path}' WHERE process_type IN ('CE','EA','EIS') GROUP BY 1 ORDER BY 1""").df()
    bdf, adf = cov(PRE), cov(NEW)
    m = bdf.merge(adf, on="process_type", suffixes=("_old", "_new"))
    print(f"########## COVERAGE delta ##########\nBASELINE = {Path(PRE).name}")
    for _, r in m.iterrows():
        print(f"\n[{r['process_type']}] n {r['n_old']}->{r['n_new']}")
        for col, lab in [("init_any", "init"), ("dec_any", "dec"), ("both_cov", "both")]:
            o, nw = int(r[f"{col}_old"]), int(r[f"{col}_new"])
            print(f"  {lab:5s}: {o:>6}->{nw:<6} ({100*o/r['n_old']:.1f}%->{100*nw/r['n_new']:.1f}%, "
                  f"{'+' if nw>=o else ''}{nw-o})")


def print_both_estimate():
    """Full-timeline (BOTH = initiation AND decision) coverage, three tiers + share of ALL projects:
      FLOOR  = now, pre-LLM (guaranteed).
      HONEST = if 06 fills gaps using only MODEL-CONFIDENT / AUTHORITATIVE candidates (p_*_cal>=0.5
               OR register metadata) — the realistic "where you'll land".
      LOOSE  = if 06 fills gaps from ANY candidate (upper bound; inflated for EIS decision by the
               document_text false-positives A1 rejects).
    'share-of-all' = each tier as a fraction of the whole CE+EA+EIS universe (the TOTAL row is the
    overall complete-timeline rate)."""
    CANDS = 'data/analysis/timeline/timeline_candidates.parquet'
    con.execute(f"""CREATE OR REPLACE TEMP VIEW cand_avail AS
      SELECT project_id,
        MAX(CASE WHEN candidate_role IN ('clear_initiation','proxy_initiation') THEN 1 ELSE 0 END) hic,
        MAX(CASE WHEN candidate_role IN ('clear_decision','proxy_decision') THEN 1 ELSE 0 END) hdc,
        MAX(CASE WHEN candidate_role IN ('clear_initiation','proxy_initiation')
              AND (TRY_CAST(p_init_cal AS DOUBLE)>=0.5 OR candidate_source_type='metadata') THEN 1 ELSE 0 END) hic_r,
        MAX(CASE WHEN candidate_role IN ('clear_decision','proxy_decision')
              AND (TRY_CAST(p_dec_cal AS DOUBLE)>=0.5 OR candidate_source_type='metadata') THEN 1 ELSE 0 END) hdc_r
      FROM '{CANDS}' GROUP BY 1""")
    df = con.execute(f"""
      WITH j AS (SELECT a.process_type pt,
                   (a.initiation_date IS NOT NULL) inn, (a.decision_date IS NOT NULL) dc,
                   COALESCE(c.hic,0) hic, COALESCE(c.hdc,0) hdc,
                   COALESCE(c.hic_r,0) hic_r, COALESCE(c.hdc_r,0) hdc_r
                 FROM a LEFT JOIN cand_avail c USING(project_id)
                 WHERE a.process_type IN ('CE','EA','EIS'))
      SELECT pt, COUNT(*) n,
        SUM((inn AND dc)::INT) fl,
        SUM(((inn OR hic_r=1) AND (dc OR hdc_r=1))::INT) ho,
        SUM(((inn OR hic=1) AND (dc OR hdc=1))::INT) lo
      FROM j GROUP BY 1 ORDER BY 1""").df()
    G = int(df['n'].sum())
    print(f"\n########## FULL-TIMELINE (BOTH = init AND decision) — n_total={G} ##########")
    print(f"  {'type':4} {'n':>6} | {'FLOOR now':>14} | {'HONEST est':>14} | {'LOOSE ceil':>14} | share-of-all floor->honest")
    s = {'fl': 0, 'ho': 0, 'lo': 0}
    for _, r in df.iterrows():
        n = int(r['n']); fl = int(r['fl']); ho = int(r['ho']); lo = int(r['lo'])
        for k, v in [('fl', fl), ('ho', ho), ('lo', lo)]:
            s[k] += v
        print(f"  {r['pt']:4} {n:>6} | {fl:>6} {100*fl/n:>5.1f}% | {ho:>6} {100*ho/n:>5.1f}% | "
              f"{lo:>6} {100*lo/n:>5.1f}% | {100*fl/G:>5.1f}% -> {100*ho/G:>5.1f}%")
    print(f"  {'ALL':4} {G:>6} | {s['fl']:>6} {100*s['fl']/G:>5.1f}% | {s['ho']:>6} {100*s['ho']/G:>5.1f}% | "
          f"{s['lo']:>6} {100*s['lo']/G:>5.1f}% | {100*s['fl']/G:>5.1f}% -> {100*s['ho']/G:>5.1f}%")


print_coverage_delta()
print_both_estimate()
print_dropped_summary()
sample_dropped("EIS", "bd", "bd_src", "bd_ev", "decision")
sample_dropped("CE", "bi", "bi_src", "bi_ev", "initiation")

def sample(name, where, datecol, evcol, extra=""):
    df = con.execute(f"""
      SELECT a.project_id, a.process_type AS proc, a.{datecol} AS sel_date,
             a.{datecol}_granularity AS gran, a.{evcol} AS evidence,
             a.{datecol[:-5] if datecol.endswith('_date') else datecol}_source_type AS src,
             a.{datecol[:-5] if datecol.endswith('_date') else datecol}_document_id AS doc_id
             {extra}
      FROM a LEFT JOIN b USING(project_id)
      WHERE {where}
      ORDER BY random() LIMIT {N}""").df()
    df['verdict'] = ''  # reviewer fills: correct | wrong | ambiguous
    df.to_csv(OUT / f"audit_{name}.csv", index=False)
    return df

strata = {
  # EIS new decisions that are FEIS-fallback (highest risk: Final-EIS date used as decision proxy)
  "eis_feis_fallback": ("a.process_type='EIS' AND a.decision_date IS NOT NULL AND b.bd IS NULL "
                        "AND a.decision_is_feis_fallback", "decision_date", "decision_evidence_text",
                        ", a.has_rod, a.decision_is_feis_fallback"),
  # EIS new decisions backed by a real ROD (should be clean)
  "eis_rod": ("a.process_type='EIS' AND a.decision_date IS NOT NULL AND b.bd IS NULL "
              "AND a.has_rod AND NOT a.decision_is_feis_fallback", "decision_date",
              "decision_evidence_text", ", a.has_rod"),
  # CE new initiations from the inferred-earliest-date proxy (high risk: stray figure/citation date)
  "ce_proxy_init": ("a.process_type='CE' AND a.initiation_date IS NOT NULL AND b.bi IS NULL "
                    "AND a.initiation_is_proxy", "initiation_date", "initiation_evidence_text", ""),
  # EIS new initiations (scoping/NOI cues)
  "eis_init": ("a.process_type='EIS' AND a.initiation_date IS NOT NULL AND b.bi IS NULL",
               "initiation_date", "initiation_evidence_text", ""),
}

con2 = con
for name, (where, dc, ev, extra) in strata.items():
    # build query inline (source_type/doc columns named explicitly)
    sc = "decision_source_type" if dc == "decision_date" else "initiation_source_type"
    di = "decision_document_id" if dc == "decision_date" else "initiation_document_id"
    pg = "decision_page_number" if dc == "decision_date" else "initiation_page_number"
    df = con.execute(f"""
      SELECT a.project_id, a.process_type AS proc, a.{dc} AS sel_date, a.{dc}_granularity AS gran,
             a.{sc} AS src, a.{di} AS doc_id, a.{pg} AS page,
             a.{ev} AS evidence {extra}
      FROM a LEFT JOIN b USING(project_id)
      WHERE {where} ORDER BY random() LIMIT {N}""").df()
    df['verdict'] = ''
    df.to_csv(OUT / f"audit_{name}.csv", index=False)
    print(f"\n########## {name}  (n={len(df)}) ##########")
    for _, r in df.iterrows():
        ev_txt = ' '.join(str(r['evidence'] or '')[:240].split())
        print(f"\n[{r['proc']}] {r['project_id'][:12]}  date={r['sel_date']} ({r['gran']})  src={r['src']}  p{r['page']}")
        print(f"  EVIDENCE: {ev_txt}")
print(f"\nReview sheets written to {OUT}/audit_*.csv")
