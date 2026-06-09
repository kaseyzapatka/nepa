"""Phase A — read-only EIS recall/funnel audit (D4).

Establishes "where we are" on EIS decision coverage BEFORE any production change.
Writes only to phase2/output/deliverable04/eis_audit/; touches no production parquet.

Answers (see phase2/plans/eis_audit.md Phase A):
  1. FM0 universe drop: 4,130 indexed EIS -> 3,466 output; split no-packets vs no-candidates.
  2. Document-type universe: ROD / FEIS-no-ROD / DEIS-only / none (reproduce 574/1883/801/872).
  3. CONFIRMED-ROD baseline: classify the current EIS selected decisions as
     confirmed_ROD / FEIS / other / unclear  (the key number we do not have yet).
  4. Solution-2 sizing: of FEIS-no-ROD projects, how many already have a FEIS candidate.
  5. ROD selection funnel: apparent-ROD projects -> have candidate -> selected decision.

Read-only. Run:  python phase2/code/deliverable04/_audit_eis_recall.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TL = ROOT / "phase2" / "data" / "analysis" / "timeline"
OUT = ROOT / "phase2" / "training" / "deliverable04" / "eis_validation"
OUT.mkdir(parents=True, exist_ok=True)

IDX_PATH = TL / "timeline_document_index.parquet"
CAND_PATH = TL / "timeline_candidates.parquet"
PROJ_PATH = TL / "timeline_project_dates.parquet"
PKT_PATH = TL / "timeline_context_packets.parquet"

# Document-type classification (document_type_clean is the cleanest signal).
ROD_TYPES = {"ROD"}
FEIS_TYPES = {"FEIS"}
DEIS_TYPES = {"DEIS"}

LINE = "=" * 78


def hdr(s: str) -> None:
    print(f"\n{LINE}\n{s}\n{LINE}")


def main() -> None:
    idx = pd.read_parquet(IDX_PATH)
    proj = pd.read_parquet(PROJ_PATH)
    eis_idx = idx[idx.process_type == "EIS"].copy()
    eis_proj = proj[proj.process_type == "EIS"].copy()

    n_idx = eis_idx.project_id.nunique()
    n_out = len(eis_proj)
    hdr(f"1. FM0 UNIVERSE DROP")
    print(f"  EIS projects in index : {n_idx:,}")
    print(f"  EIS projects in output: {n_out:,}")
    print(f"  Missing from output   : {n_idx - n_out:,}")

    # Split the missing into no-packets vs packets-but-no-candidates.
    cand = pd.read_parquet(CAND_PATH, columns=["project_id", "process_type", "document_id",
                                               "page_number", "candidate_id", "candidate_role",
                                               "role_confidence", "parsed_date", "date_granularity",
                                               "is_proxy", "document_type_clean",
                                               "document_type_category", "ranking_score",
                                               "learned_decision_score", "p_dec_cal",
                                               "selected_for_decision"])
    eis_cand = cand[cand.process_type == "EIS"].copy()
    cand_pids = set(eis_cand.project_id.unique())
    idx_pids = set(eis_idx.project_id.unique())
    out_pids = set(eis_proj.project_id.unique())
    missing_pids = idx_pids - out_pids

    pkt = pd.read_parquet(PKT_PATH, columns=["project_id", "process_type"])
    eis_pkt_pids = set(pkt[pkt.process_type == "EIS"].project_id.unique())
    miss_no_pkt = missing_pids - eis_pkt_pids
    miss_no_cand = missing_pids & eis_pkt_pids  # has packets but dropped (=> no candidates / no row)
    print(f"    of which NO context packets        : {len(miss_no_pkt):,}")
    print(f"    of which packets but no output row : {len(miss_no_cand):,}")
    print(f"  (sanity) EIS projects with >=1 candidate: {len(cand_pids):,}")

    # ---- 2. Document-type universe ----
    def has_type(types):
        m = eis_idx.document_type_clean.isin(types)
        return set(eis_idx.loc[m, "project_id"].unique())

    rod_p = has_type(ROD_TYPES)
    feis_p = has_type(FEIS_TYPES)
    deis_p = has_type(DEIS_TYPES)
    all_p = idx_pids
    g_rod = rod_p
    g_feis_no_rod = feis_p - rod_p
    g_deis_only = deis_p - feis_p - rod_p
    g_none = all_p - rod_p - feis_p - deis_p
    hdr("2. DOCUMENT-TYPE UNIVERSE (project-level, document_type_clean)")
    print(f"  ROD (has a ROD doc)        : {len(g_rod):,}")
    print(f"  FEIS but no ROD            : {len(g_feis_no_rod):,}")
    print(f"  DEIS only (no FEIS/ROD)    : {len(g_deis_only):,}")
    print(f"  None of ROD/FEIS/DEIS      : {len(g_none):,}")
    print(f"  endpoint universe (ROD+FEIS): {len(g_rod | g_feis_no_rod):,}")
    print(f"  no-endpoint (DEIS-only+none): {len(g_deis_only | g_none):,}")

    # ---- 3. CONFIRMED-ROD classification of current selected EIS decisions ----
    sel = eis_proj[eis_proj.decision_date.notna() & (eis_proj.decision_date.astype(str).str.strip() != "")].copy()
    hdr("3. CLASSIFY CURRENT EIS SELECTED DECISIONS (confirmed ROD / FEIS / other / unclear)")
    print(f"  EIS projects with a selected decision_date: {len(sel):,}")

    # Join decision_document_id -> index doc type
    doc_type = (eis_idx[["project_id", "document_id", "document_type_clean", "document_type_category"]]
                .drop_duplicates(["project_id", "document_id"]))
    sel = sel.merge(doc_type, left_on=["project_id", "decision_document_id"],
                    right_on=["project_id", "document_id"], how="left",
                    suffixes=("", "_doc"))

    def classify(row):
        dt = str(row.get("document_type_clean") or "").upper()
        src = str(row.get("decision_source_type") or "").lower()
        if "register" in src or "blm" in src or "doe" in src:
            return "confirmed_ROD"  # authoritative register decision date
        if dt in ROD_TYPES:
            return "confirmed_ROD"
        if dt in FEIS_TYPES:
            return "FEIS_as_decision"
        if dt in DEIS_TYPES or dt in {"OTHER", "APPENDIX", "EA", "FONSI", "DEA", "CE"}:
            return "other"
        return "unclear"

    sel["rod_class"] = sel.apply(classify, axis=1)
    cls = sel.rod_class.value_counts(dropna=False)
    for k, v in cls.items():
        print(f"    {k:18s}: {v:,}")
    print("\n  cross-tab decision_source_type x class:")
    print(pd.crosstab(sel.decision_source_type, sel.rod_class).to_string())
    print("\n  cross-tab decision doc type x class:")
    print(pd.crosstab(sel.document_type_clean.fillna("<none>"), sel.rod_class).to_string())
    print("\n  decision_is_proxy among selected:", sel.decision_is_proxy.value_counts(dropna=False).to_dict())

    # Confirmed ROD coverage over apparent-ROD universe
    confirmed_rod_pids = set(sel.loc[sel.rod_class == "confirmed_ROD", "project_id"])
    hdr("3b. CONFIRMED-ROD COVERAGE (the real baseline Phase C must beat)")
    print(f"  confirmed-ROD selected decisions          : {len(confirmed_rod_pids):,}")
    print(f"  of apparent-ROD universe ({len(g_rod):,})        : "
          f"{len(confirmed_rod_pids & g_rod):,} = "
          f"{100*len(confirmed_rod_pids & g_rod)/max(1,len(g_rod)):.1f}%")
    print(f"  FEIS-as-decision currently in decision_date: {(sel.rod_class=='FEIS_as_decision').sum():,}")

    # ---- 4. Solution-2 sizing: FEIS candidates among FEIS-no-ROD projects ----
    hdr("4. SOLUTION-2 SIZING — FEIS candidates among FEIS-no-ROD projects")
    feis_cand = eis_cand[eis_cand.document_type_clean.isin(FEIS_TYPES)].copy()
    feis_cand_pids = set(feis_cand.project_id.unique())
    have_feis_cand = g_feis_no_rod & feis_cand_pids
    print(f"  FEIS-no-ROD projects                  : {len(g_feis_no_rod):,}")
    print(f"  ...with >=1 candidate from a FEIS doc : {len(have_feis_cand):,} "
          f"({100*len(have_feis_cand)/max(1,len(g_feis_no_rod)):.1f}%)")
    print(f"  ...with NO FEIS-doc candidate (need extraction -> Phase D): "
          f"{len(g_feis_no_rod - feis_cand_pids):,}")
    print(f"  role mix of FEIS-doc candidates:")
    print("   ", feis_cand.candidate_role.value_counts(dropna=False).to_dict())

    # ---- 5. ROD selection funnel ----
    hdr("5. ROD SELECTION FUNNEL (apparent-ROD = has ROD doc)")
    rod_cand = eis_cand[eis_cand.project_id.isin(g_rod)].copy()
    # decision-oriented candidate = role in decision/proxy/body OR sits in a ROD doc
    dec_roles = {"clear_decision", "proxy_decision", "body_text"}
    rod_cand["dec_oriented"] = rod_cand.candidate_role.isin(dec_roles) | rod_cand.document_type_clean.isin(ROD_TYPES)
    rod_with_cand = set(rod_cand.loc[rod_cand.dec_oriented, "project_id"])
    rod_selected = set(sel.project_id) & g_rod
    print(f"  apparent-ROD projects (has ROD doc)        : {len(g_rod):,}")
    print(f"  ...with a decision-oriented candidate      : {len(rod_with_cand):,}")
    print(f"  ...with a SELECTED decision (any class)    : {len(rod_selected):,}")
    print(f"  ...candidate present, NOT selected (the 295): {len(rod_with_cand - rod_selected):,}")
    print(f"  ...NO decision candidate at all (the 46)    : {len(g_rod - rod_with_cand):,}")

    # Best-candidate role mix among ROD projects with candidate but no selection
    blocked = rod_with_cand - rod_selected
    bc = rod_cand[rod_cand.project_id.isin(blocked) & rod_cand.dec_oriented].copy()
    print("\n  role mix of decision-oriented candidates in the blocked ROD projects:")
    print("   ", bc.candidate_role.value_counts(dropna=False).to_dict())
    print("  learned_decision_score on these (describe):")
    lds = pd.to_numeric(bc.learned_decision_score, errors="coerce")
    print("   ", {k: round(float(v), 3) for k, v in lds.describe().to_dict().items()})
    print(f"   share with learned_decision_score > 0: {100*(lds>0).mean():.1f}%")

    # ---- write CSVs ----
    sel[["project_id", "decision_date", "decision_source_type", "decision_is_proxy",
         "decision_document_id", "document_type_clean", "rod_class"]].to_csv(
        OUT / "eis_selected_decision_classification.csv", index=False)
    pd.DataFrame({"project_id": sorted(g_feis_no_rod - feis_cand_pids)}).to_csv(
        OUT / "eis_feis_no_rod_no_candidate.csv", index=False)
    pd.DataFrame({"project_id": sorted(blocked)}).to_csv(
        OUT / "eis_rod_candidate_not_selected.csv", index=False)
    pd.DataFrame({"project_id": sorted(g_rod - rod_with_cand)}).to_csv(
        OUT / "eis_rod_no_candidate.csv", index=False)
    print(f"\nWrote CSVs to {OUT}")


if __name__ == "__main__":
    main()
