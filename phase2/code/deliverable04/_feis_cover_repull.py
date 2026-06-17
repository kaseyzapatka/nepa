#!/usr/bin/env python
"""A2: targeted FEIS cover-page re-pull (parity-preserving).

Builds context packets from FEIS document cover pages (1-N) in 02's packet shape and runs them
through 03's *exact* extract_candidates_from_packet(), so the candidates are identical to what a
full re-pull would surface. Writes new candidates to a SEPARATE file for review; integration
(concat + dedup + 04 --append) happens in Phase B.

Usage: conda run -n nepa python code/deliverable04/_feis_cover_repull.py [--pages N]
"""
import argparse, hashlib, importlib.util
import duckdb, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE2 = HERE.parent.parent
TL = PHASE2 / "data/analysis/timeline"
IDX = TL / "timeline_document_index.parquet"
PAGES = PHASE2 / "data/processed/eis/pages.parquet"
CANDS = TL / "timeline_candidates.parquet"
OUT = TL / "timeline_candidates_feiscover.parquet"

# import 03 (module name starts with a digit -> importlib)
spec = importlib.util.spec_from_file_location("extract03", HERE / "03_extract_candidates.py")
m03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m03)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pages", type=int, default=3); args = ap.parse_args()
    con = duckdb.connect()
    feis = con.execute(f"""SELECT project_id, document_id,
        COALESCE(document_type_clean,'FEIS') dtc, document_type_category dcat
      FROM '{IDX}' WHERE process_type='EIS' AND upper(document_type_clean)='FEIS'""").df()
    docids = "','".join(feis['document_id'].astype(str).unique())
    pg = con.execute(f"""SELECT document_id, page_number, page_text FROM '{PAGES}'
      WHERE document_id IN ('{docids}')
        AND TRY_CAST(regexp_replace(CAST(page_number AS VARCHAR),'[^0-9]','','g') AS INT) BETWEEN 1 AND {args.pages}""").df()
    docmap = feis.drop_duplicates("document_id").set_index("document_id")[["project_id","dtc","dcat"]].to_dict("index")
    print(f"FEIS docs: {feis['document_id'].nunique()} | cover pages (1-{args.pages}) read: {len(pg)}")

    cands = []
    for _, r in pg.iterrows():
        did = r["document_id"]; meta = docmap.get(did)
        if not meta or not (r["page_text"] or "").strip():
            continue
        packet = {
            "context_text": r["page_text"],
            "source_tier": "page_slice",                # document-text branch (NOT metadata)
            "process_type": "EIS",
            "retrieval_reason": "feis_cover_repull",
            "retrieval_tier": "feis_cover_repull",
            "document_type_clean": meta["dtc"],
            "document_type_category": meta["dcat"],
            "heading_title": None,
            "project_id": meta["project_id"],
            "document_id": did,
            "page_start": str(r["page_number"]),
            "section_id": None,
            "context_packet_id": hashlib.sha1(f"feiscover|{did}|{r['page_number']}".encode()).hexdigest()[:20],
        }
        cands.extend(m03.extract_candidates_from_packet(packet))

    if not cands:
        print("no candidates extracted"); return
    new = pd.DataFrame(cands).drop_duplicates("candidate_id")
    # drop ones already present in the main pool (same candidate_id)
    existing = set(con.execute(f"SELECT candidate_id FROM '{CANDS}'").df()["candidate_id"].astype(str))
    new_only = new[~new["candidate_id"].astype(str).isin(existing)]
    new.to_parquet(OUT, index=False)
    print(f"extracted {len(new)} cover candidates ({len(new_only)} NOT already in pool) -> {OUT.name}")
    print(f"distinct FEIS projects with >=1 new cover candidate: {new_only['project_id'].nunique()}")
    # sample for review: month-granularity dates (likely publication months)
    samp = new_only[new_only["date_granularity"].eq("month")].head(12)
    for _, c in samp.iterrows():
        ctx = " ".join(str(c.get("context_text",""))[:90].split())
        print(f"  {c['parsed_date']} [{c['candidate_role']}] {ctx}")

if __name__ == "__main__":
    main()
