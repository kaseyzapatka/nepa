"""D6 v2 — n02: assemble per-project evidence for candidate FONSI projects.

Reuses the existing typed per-project packets (`fonsi_project_packets.parquet`,
which already carries action/finding/resource/condition/boundary text) and
attaches span-level provenance from `fonsi_evidence_spans.parquet` (section_id,
evidence_span_id, source_span_sha256, span_type, page). Falls back to
`fonsi_document_sections.parquet` only for candidate projects missing from the
packets, minting a stable hash for those.

Output: data/analysis/deliverable06/candidate_evidence_packets.parquet
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import json

import duckdb
import pandas as pd

from common import D6_ANALYSIS_DIR, ensure_d6_dirs, sha256_text, utc_now, write_parquet

CORPUS = D6_ANALYSIS_DIR / "candidate_corpus.parquet"
PACKETS = D6_ANALYSIS_DIR / "fonsi_project_packets.parquet"
SPANS = D6_ANALYSIS_DIR / "fonsi_evidence_spans.parquet"
SECTIONS = D6_ANALYSIS_DIR / "fonsi_document_sections.parquet"
OUT = D6_ANALYSIS_DIR / "candidate_evidence_packets.parquet"

TEXT_COLS = ["action_text", "finding_text", "resource_text", "condition_text",
             "boundary_text", "analysis_text"]
# provenance priority: cite action first, then finding/boundary/condition/resource
SPAN_PRIORITY = {"action": 0, "finding": 1, "boundary": 2, "condition": 3, "resource": 4, "fallback": 5}
MAX_SPANS_PER_PROJECT = 8


def _span_provenance(spans: pd.DataFrame) -> dict[str, str]:
    """project_id -> json list of compact provenance dicts (priority-ordered)."""
    spans = spans.copy()
    spans["_pri"] = spans["span_type"].map(SPAN_PRIORITY).fillna(9)
    out: dict[str, str] = {}
    for pid, grp in spans.sort_values(["_pri"]).groupby("project_id"):
        recs = []
        for r in grp.head(MAX_SPANS_PER_PROJECT).itertuples(index=False):
            recs.append({
                "evidence_span_id": str(r.evidence_span_id),
                "section_id": str(r.section_id),
                "source_span_sha256": str(r.source_span_sha256),
                "span_type": str(r.span_type),
                "document_id": str(r.document_id),
                "page_start": None if pd.isna(r.page_start) else int(r.page_start),
                "heading_title": "" if pd.isna(getattr(r, "heading_title", None)) else str(r.heading_title),
            })
        out[str(pid)] = json.dumps(recs)
    return out


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()

    corpus = pd.read_parquet(CORPUS)
    fonsi = corpus.loc[corpus["is_fonsi"]].copy()
    fonsi["project_id"] = fonsi["project_id"].astype(str)
    project_ids = sorted(fonsi["project_id"].unique())
    cats_by_project = (
        fonsi.groupby("project_id")["candidate_category"].agg(lambda s: sorted(set(s))).to_dict()
    )
    id_list = ",".join(f"'{p}'" for p in project_ids)

    con = duckdb.connect()
    packets = con.execute(
        f"select * from read_parquet('{PACKETS}') where cast(project_id as varchar) in ({id_list})"
    ).df()
    packets["project_id"] = packets["project_id"].astype(str)

    spans = con.execute(
        f"""select project_id, document_id, section_id, evidence_span_id, span_type,
                   heading_title, page_start, source_span_sha256
            from read_parquet('{SPANS}') where cast(project_id as varchar) in ({id_list})"""
    ).df()
    spans["project_id"] = spans["project_id"].astype(str)
    prov = _span_provenance(spans)

    records = []
    have = set(packets["project_id"])
    for r in packets.itertuples(index=False):
        pid = str(r.project_id)
        rec = {"project_id": pid, "candidate_categories": json.dumps(cats_by_project.get(pid, []))}
        for col in TEXT_COLS:
            rec[col] = getattr(r, col, "") or ""
        rec["evidence_span_count"] = int(getattr(r, "evidence_span_count", 0) or 0)
        rec["canonical_fonsi_document_id"] = getattr(r, "canonical_fonsi_document_id", None)
        rec["span_provenance"] = prov.get(pid, "[]")
        rec["packet_source"] = "packet"
        rec["evidence_run_at"] = run_at
        records.append(rec)

    # fallback for candidate projects missing from packets (mint hash + ids)
    missing = [p for p in project_ids if p not in have]
    if missing:
        m_list = ",".join(f"'{p}'" for p in missing)
        sec = con.execute(
            f"""select project_id, document_id, section_text
                from read_parquet('{SECTIONS}') where cast(project_id as varchar) in ({m_list})"""
        ).df()
        sec["project_id"] = sec["project_id"].astype(str)
        for pid, grp in sec.groupby("project_id"):
            text = "\n\n".join(str(t) for t in grp["section_text"].tolist())[:20000]
            doc_id = grp["document_id"].iloc[0] if len(grp) else None
            prov_recs = [{
                "evidence_span_id": f"fallback::{pid}", "section_id": f"fallback::{pid}",
                "source_span_sha256": sha256_text(text), "span_type": "fallback",
                "document_id": str(doc_id), "page_start": None, "heading_title": "",
            }]
            rec = {"project_id": pid, "candidate_categories": json.dumps(cats_by_project.get(pid, []))}
            rec["action_text"] = text
            for col in TEXT_COLS[1:]:
                rec[col] = ""
            rec["evidence_span_count"] = 0
            rec["canonical_fonsi_document_id"] = doc_id
            rec["span_provenance"] = json.dumps(prov_recs)
            rec["packet_source"] = "sections_fallback"
            rec["evidence_run_at"] = run_at
            records.append(rec)

    out = pd.DataFrame(records)
    write_parquet(out, OUT)
    print(f"[n02] candidate FONSI projects={len(project_ids)} packets={len(have)} "
          f"fallback={len(missing)} -> {OUT}")
    nonempty = (out[TEXT_COLS].apply(lambda s: s.str.len() > 0)).sum().to_dict()
    print(f"[n02] non-empty text columns: {nonempty}")


if __name__ == "__main__":
    main()
