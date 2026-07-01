"""D2 shared deterministic candidate generator (SQL/regex-first; NO LLM).

Used by BOTH `02_extract_fonsi_significance.py` (extract + adjudicate) and
`03_build_gold_set_queue.py` (labeling worksheet) so the labeled set and the extracted set
come from the SAME candidate universe (no drift). Candidate generation is deterministic; the
LLM only adjudicates candidate windows downstream.

Cue dictionaries generate candidates; the class here is a heuristic *guess* (the LLM/human
assign the final class). Guards the `significant impact` ⊂ `no significant impact` substring trap
with ordered rules.
"""
from __future__ import annotations

import re

import pandas as pd

import common as C
from significance_taxonomy import (
    DETERMINATION_CUES, MITIGATION_OBLIGATIONS, MITIGATION_ROLES, RESOURCE_CROSSWALK,
    THRESHOLD_CUES,
)

ROLES_SQL = ",".join(f"'{r}'" for r in MITIGATION_ROLES)
OBLIG_SQL = ",".join(f"'{o}'" for o in MITIGATION_OBLIGATIONS)
FONSI_ROLES = ("linked_ea", "canonical_fonsi", "supporting_fonsi")

# compiled cue groups
_CUE = {k: [re.compile(p, re.I) for p in pats] for k, pats in DETERMINATION_CUES.items()}
_THRESH = {k: [re.compile(p, re.I) for p in pats] for k, pats in THRESHOLD_CUES.items()}

# flat resource keyword -> (shared_area, subarea)
_RES_KW: list[tuple[str, str, str]] = []
for shared, subs in RESOURCE_CROSSWALK.items():
    for sub, kws in subs.items():
        for kw in kws:
            _RES_KW.append((kw.lower(), shared, sub))
    _RES_KW.append((shared.replace("_", " "), shared, next(iter(subs), shared)))


def _has(text: str, group: str) -> bool:
    return any(p.search(text) for p in _CUE[group])


def classify_determination(text: str) -> tuple[str, str, str]:
    """Ordered heuristic → (candidate_class_guess, polarity_guess, matched_cue_group)."""
    t = text or ""
    mit = _has(t, "explicit_mitigated_lts")
    fonsi = _has(t, "document_outcome")
    lts = _has(t, "explicit_less_than_significant")
    sig = _has(t, "explicit_significant_adverse")
    if mit and (fonsi or lts):
        return "less_than_significant_with_mitigation", "mixed", "explicit_mitigated_lts"
    if fonsi:
        return "no_significant_impact", "no_adverse", "document_outcome"
    if lts:
        return "less_than_significant", "adverse_not_significant", "explicit_less_than_significant"
    if sig and not (fonsi or lts):   # guard the "no significant impact" substring trap
        return "significant_adverse", "adverse_significant", "explicit_significant_adverse"
    if mit:
        return "less_than_significant_with_mitigation", "mixed", "explicit_mitigated_lts"
    return "not_a_determination", "unknown", "none"


def threshold_hits(text: str) -> list[str]:
    t = text or ""
    return [k for k, pats in _THRESH.items() if any(p.search(t) for p in pats)]


def resource_guess(text: str) -> tuple[str, str]:
    t = (text or "").lower()
    for kw, shared, sub in _RES_KW:
        if kw and kw in t:
            return shared, sub
    return "unknown", "unknown"


def generate_fonsi_candidates() -> pd.DataFrame:
    """Finding-span candidate windows for the clean EA-source FONSI corpus."""
    sql = f"""
    WITH corpus_ea AS (
        SELECT project_id FROM read_parquet('{C.SIGNIFICANCE_CORPUS}') WHERE process_type = 'EA'
    ),
    qual AS (
        SELECT DISTINCT project_id, document_id, section_id
        FROM read_parquet('{C.FONSI_CONDITIONS}')
        WHERE condition_role IN ({ROLES_SQL}) AND obligation_level IN ({OBLIG_SQL})
    ),
    qual_pg AS (
        SELECT DISTINCT project_id, document_id, page_number
        FROM read_parquet('{C.FONSI_CONDITIONS}')
        WHERE condition_role IN ({ROLES_SQL}) AND obligation_level IN ({OBLIG_SQL})
    ),
    find AS (
        SELECT s.project_id, s.document_id, s.manifest_role, s.section_id, s.evidence_span_id,
               s.heading_title, s.page_start, s.page_end, s.span_text, s.source_span_sha256
        FROM read_parquet('{C.FONSI_SPANS}') s
        JOIN corpus_ea USING (project_id)
        WHERE s.span_type = 'finding' AND s.manifest_role IN ('linked_ea','canonical_fonsi','supporting_fonsi')
    )
    SELECT find.*, (qual.section_id IS NOT NULL) AS has_qual_cond_same_section,
           EXISTS (SELECT 1 FROM qual_pg q
                   WHERE q.project_id = find.project_id AND q.document_id = find.document_id
                     AND q.page_number BETWEEN find.page_start - 2 AND find.page_end + 2
                  ) AS has_qual_cond_windowed
    FROM find LEFT JOIN qual
      ON find.project_id = qual.project_id AND find.document_id = qual.document_id
     AND find.section_id = qual.section_id
    """
    df = C.q(sql)
    if df.empty:
        return df
    df["source_substrate"] = "d6_evidence_span"
    df["source_unit_id"] = df["evidence_span_id"]
    df["span_char_start"] = None   # D6 spans carry no char offsets (plan §4)
    df["span_char_end"] = None
    cls = df["span_text"].map(classify_determination)
    df["candidate_class_guess"] = cls.map(lambda x: x[0])
    df["determination_polarity_guess"] = cls.map(lambda x: x[1])
    df["matched_cue_group"] = cls.map(lambda x: x[2])
    res = df["span_text"].map(resource_guess)
    df["resource_area_guess"] = res.map(lambda x: x[0])
    df["resource_subarea_guess"] = res.map(lambda x: x[1])
    df["threshold_types_guess"] = df["span_text"].map(lambda t: ",".join(threshold_hits(t)))
    df["evidence_text"] = df["span_text"].str.slice(0, 4000)
    df["evidence_text_sha256"] = df["span_text"].map(C.sha256_text)
    df = df.drop(columns=["span_text"])
    return df


if __name__ == "__main__":
    d = generate_fonsi_candidates()
    print(f"{len(d):,} FONSI finding-span candidates")
    print("\ncandidate_class_guess:")
    print(d["candidate_class_guess"].value_counts().to_string())
    print(f"\nwith same-section qualifying condition: {int(d['has_qual_cond_same_section'].sum()):,}")
    print(f"with >=1 threshold cue: {int((d['threshold_types_guess'] != '').sum()):,}")
