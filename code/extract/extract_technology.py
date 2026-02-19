"""
Technology-specific feature extraction for NEPA projects.

This module centralizes extraction of:
- transmission flags and lengths
- geothermal flags/phases
- pipeline flags and lengths

It can be imported by extract_data.py and also run directly with CLI tags
for one or more technology domains.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------
# CONSTANTS
# --------------------------

MILES_RE = re.compile(r"(?<![a-z0-9])(\d{1,4}(?:,\d{3})*(?:\.\d+)?)\s*(mile|miles|mi)\b", re.IGNORECASE)
FEET_RE = re.compile(r"(?<![a-z0-9])(\d{1,6}(?:,\d{3})*(?:\.\d+)?)\s*(foot|feet|ft)\b", re.IGNORECASE)

TRANSMISSION_HINTS = (
    "transmission",
    "powerline",
    "power line",
    "kV transmission",
    "electric line",
    "line route",
)

PIPELINE_HINTS = (
    "pipeline",
    "pipelines",
    "right-of-way",
    "row",
    "buried line",
    "flowline",
)

TRANSMISSION_BUILD_RE = re.compile(
    r"(?:new\s+transmission\s+line|"
    r"\btransmission\s+line\s+(?:project|route|corridor)\b|"
    r"\b(?:construct(?:ion|ed)?|build(?:ing)?|install(?:ation|ed)?|upgrade(?:d|s)?|rebuild(?:ing)?)\s+"
    r"(?:of\s+)?(?:new\s+)?(?:\d{2,4}\s*-?\s*k\s?v\s+)?transmission\s+line\b|"
    r"double-?circuit\s+(?:\d{2,4}\s*-?\s*k\s?v\s+)?transmission\s+line|"
    r"single-?circuit\s+(?:\d{2,4}\s*-?\s*k\s?v\s+)?transmission\s+line|"
    r"\b\d{2,4}\s*-?\s*k\s?v\s+(?:transmission\s+line|line)\b|"
    r"right-?of-?way.*transmission\s+line|transmission\s+line.*right-?of-?way)",
    re.IGNORECASE,
)

TRANSMISSION_ALTERNATIVE_RE = re.compile(
    r"\b("
    r"alternative(?:s)?|"
    r"route\s+alternative(?:s)?|"
    r"alignment\s+alternative(?:s)?|"
    r"option\s+[a-z0-9]+|"
    r"either\s+route|"
    r"one\s+of\s+(?:two|three|several)\s+routes"
    r")\b",
    re.IGNORECASE,
)

TRANSMISSION_ADDITIVE_RE = re.compile(
    r"\b("
    r"also\s+included|"
    r"in\s+addition|"
    r"along\s+with|"
    r"plus|"
    r"combined|"
    r"new\s+build\s+section|"
    r"upgrade\s+section|"
    r"segment(?:s)?|"
    r"phase(?:s)?|"
    r"lateral(?:s)?|"
    r"spur(?:s)?|"
    r"tap(?:s)?"
    r")\b",
    re.IGNORECASE,
)

GEOTHERMAL_PHASE_PATTERNS = {
    "exploration": [
        r"\bexploration\b",
        r"\bexploratory\b",
        r"\bresource assessment\b",
        r"\bgeophysical survey\b",
        r"\btemperature gradient\b",
    ],
    "drilling": [
        r"\bdrilling\b",
        r"\bdrill pad\b",
        r"\bwell pad\b",
        r"\bproduction well\b",
        r"\binjection well\b",
        r"\bwell stimulation\b",
    ],
    "plant": [
        r"\bpower plant\b",
        r"\bgenerating station\b",
        r"\bsteam plant\b",
        r"\bbinary plant\b",
        r"\bflash plant\b",
        r"\bturbine\b",
        r"\binterconnection\b",
    ],
}


# --------------------------
# TYPES
# --------------------------

@dataclass
class LengthAdjudication:
    selected_length_miles: float
    confidence: str
    source_text: str
    taxonomy: str
    selection_method: str
    selected_candidate_ids: List[str]
    candidate_count: int
    distinct_candidate_count: int
    llm_trigger: bool
    llm_used: bool
    llm_status: str


# --------------------------
# HELPERS
# --------------------------

def _value_to_text(value) -> str:
    """Convert list/JSON/scalar values to a normalized plain-text string."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(v) for v in value if str(v).strip())
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return ""
        if v.startswith("[") and v.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(v)
                    if isinstance(parsed, (list, tuple)):
                        return " ".join(str(x) for x in parsed if str(x).strip())
                except Exception:
                    pass
        return v
    return str(value)


def _series_text(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index)
    return df[col].apply(_value_to_text)


def _match_snippet(text: str, start: int, end: int, max_len: int = 500) -> str:
    """
    Build a bounded snippet centered on the matched token span so the extracted
    numeric mention is always visible in QA fields.
    """
    if not text:
        return ""
    if len(text) <= max_len:
        return text

    center = (start + end) // 2
    half = max_len // 2
    left = max(0, center - half)
    right = min(len(text), left + max_len)
    left = max(0, right - max_len)
    snippet = text[left:right]

    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet = snippet + "..."
    return snippet


def _extract_length_candidates(text: str, hints: Sequence[str], prefix: str) -> List[Dict]:
    """
    Extract all candidate linear lengths from text with source sentences.

    Returns candidate dicts that include sentence-level evidence for QA.
    """
    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = re.split(r"(?<=[\.\?!;])\s+", normalized)
    candidates: List[Dict] = []
    cid = 1

    for s_idx, sent in enumerate(sentences):
        s_lower = sent.lower()
        if "mile post" in s_lower or "mp " in s_lower:
            continue

        matched_hints = [h for h in hints if h.lower() in s_lower]
        hint_score = len(matched_hints)
        if hint_score == 0:
            continue

        for m in MILES_RE.finditer(sent):
            raw = m.group(1)
            val_mi = float(raw.replace(",", ""))
            if 0 < val_mi <= 5000:
                match_start, match_end = m.span()
                candidates.append(
                    {
                        "candidate_id": f"{prefix}_{cid:04d}",
                        "sentence_index": s_idx,
                        "value_miles": round(val_mi, 3),
                        "raw_value": raw,
                        "matched_text": m.group(0),
                        "raw_unit": m.group(2).lower(),
                        "unit_normalized": "miles",
                        "hint_score": hint_score + 2,
                        "hint_terms": matched_hints,
                        "source_text": _match_snippet(sent, match_start, match_end, max_len=500),
                    }
                )
                cid += 1

        for m in FEET_RE.finditer(sent):
            raw = m.group(1)
            val_ft = float(raw.replace(",", ""))
            val_mi = val_ft / 5280.0
            if 0 < val_mi <= 5000:
                match_start, match_end = m.span()
                candidates.append(
                    {
                        "candidate_id": f"{prefix}_{cid:04d}",
                        "sentence_index": s_idx,
                        "value_miles": round(val_mi, 3),
                        "raw_value": raw,
                        "matched_text": m.group(0),
                        "raw_unit": m.group(2).lower(),
                        "unit_normalized": "miles_from_feet",
                        "hint_score": hint_score + 1,
                        "hint_terms": matched_hints,
                        "source_text": _match_snippet(sent, match_start, match_end, max_len=500),
                    }
                )
                cid += 1

    return candidates


def _collapse_candidates_by_value(candidates: List[Dict], tol: float = 0.01) -> List[Dict]:
    """Group near-equal values to avoid treating unit-conversion duplicates as distinct."""
    if not candidates:
        return []

    groups: List[Dict] = []
    for c in sorted(candidates, key=lambda x: x["value_miles"]):
        placed = False
        for g in groups:
            if abs(c["value_miles"] - g["value_miles"]) <= tol:
                g["members"].append(c)
                best = g["best_candidate"]
                if (c["hint_score"], c["value_miles"]) > (best["hint_score"], best["value_miles"]):
                    g["best_candidate"] = c
                placed = True
                break
        if not placed:
            groups.append(
                {
                    "value_miles": c["value_miles"],
                    "members": [c],
                    "best_candidate": c,
                }
            )

    # Snap group value to best candidate value for consistency.
    for g in groups:
        g["value_miles"] = g["best_candidate"]["value_miles"]

    return groups


def _best_single_candidate(candidates: List[Dict]) -> Tuple[float, str, str]:
    """Legacy-compatible best single length choice."""
    if not candidates:
        return np.nan, "none", ""

    best = sorted(candidates, key=lambda x: (x["hint_score"], x["value_miles"]), reverse=True)[0]
    confidence = "high" if best["hint_score"] >= 4 else "medium"
    return best["value_miles"], confidence, best["source_text"]


def _run_llm_transmission_adjudication(
    full_text: str,
    candidates: List[Dict],
    groups: List[Dict],
) -> Dict | None:
    """
    Placeholder for future LLM adjudication.

    Intentionally returns None today. Rule-based adjudication remains primary.
    """
    _ = (full_text, candidates, groups)
    return None


def _adjudicate_transmission_length(
    full_text: str,
    candidates: List[Dict],
    use_llm: bool = False,
) -> LengthAdjudication:
    groups = _collapse_candidates_by_value(candidates)
    candidate_count = len(candidates)
    distinct_count = len(groups)
    llm_trigger = distinct_count >= 2

    if candidate_count == 0:
        return LengthAdjudication(
            selected_length_miles=np.nan,
            confidence="none",
            source_text="",
            taxonomy="none",
            selection_method="none",
            selected_candidate_ids=[],
            candidate_count=0,
            distinct_candidate_count=0,
            llm_trigger=False,
            llm_used=False,
            llm_status="not_triggered",
        )

    text_lower = (full_text or "").lower()
    has_alternative = bool(TRANSMISSION_ALTERNATIVE_RE.search(text_lower))
    has_additive = bool(TRANSMISSION_ADDITIVE_RE.search(text_lower))

    llm_used = False
    llm_status = "not_requested" if llm_trigger else "not_triggered"
    llm_result = None
    if llm_trigger and use_llm:
        llm_status = "not_configured"
        try:
            llm_result = _run_llm_transmission_adjudication(full_text, candidates, groups)
            if llm_result:
                llm_used = True
                llm_status = "success"
        except Exception:
            llm_used = False
            llm_status = "failed_fallback_rule"

    if llm_result:
        # Reserved for future LLM behavior; keep deterministic fallback for now.
        pass

    if distinct_count <= 1:
        best = groups[0]["best_candidate"]
        confidence = "high" if best["hint_score"] >= 4 else "medium"
        return LengthAdjudication(
            selected_length_miles=best["value_miles"],
            confidence=confidence,
            source_text=best["source_text"],
            taxonomy="do_not_sum",
            selection_method="rule",
            selected_candidate_ids=[best["candidate_id"]],
            candidate_count=candidate_count,
            distinct_candidate_count=distinct_count,
            llm_trigger=llm_trigger,
            llm_used=llm_used,
            llm_status=llm_status,
        )

    # Rule order matters: alternatives should not be summed.
    if has_alternative:
        chosen = sorted(
            (g["best_candidate"] for g in groups),
            key=lambda x: (x["value_miles"], x["hint_score"]),
            reverse=True,
        )[0]
        confidence = "high" if chosen["hint_score"] >= 4 else "medium"
        return LengthAdjudication(
            selected_length_miles=chosen["value_miles"],
            confidence=confidence,
            source_text=chosen["source_text"],
            taxonomy="choose_alternative",
            selection_method="rule",
            selected_candidate_ids=[chosen["candidate_id"]],
            candidate_count=candidate_count,
            distinct_candidate_count=distinct_count,
            llm_trigger=llm_trigger,
            llm_used=llm_used,
            llm_status=llm_status,
        )

    if has_additive:
        selected = [g["best_candidate"] for g in groups]
        selected = sorted(selected, key=lambda x: (x["value_miles"], x["hint_score"]), reverse=True)
        value = round(sum(c["value_miles"] for c in selected), 3)
        source = " || ".join(c["source_text"] for c in selected)[:2000]
        confidence = "high" if len(selected) >= 2 else "medium"
        return LengthAdjudication(
            selected_length_miles=value,
            confidence=confidence,
            source_text=source,
            taxonomy="sum",
            selection_method="rule",
            selected_candidate_ids=[c["candidate_id"] for c in selected],
            candidate_count=candidate_count,
            distinct_candidate_count=distinct_count,
            llm_trigger=llm_trigger,
            llm_used=llm_used,
            llm_status=llm_status,
        )

    # Ambiguous multi-candidate case without additive/alternative cues.
    chosen = sorted(
        (g["best_candidate"] for g in groups),
        key=lambda x: (x["value_miles"], x["hint_score"]),
        reverse=True,
    )[0]
    confidence = "high" if chosen["hint_score"] >= 4 else "medium"
    return LengthAdjudication(
        selected_length_miles=chosen["value_miles"],
        confidence=confidence,
        source_text=chosen["source_text"],
        taxonomy="take_max",
        selection_method="rule",
        selected_candidate_ids=[chosen["candidate_id"]],
        candidate_count=candidate_count,
        distinct_candidate_count=distinct_count,
        llm_trigger=llm_trigger,
        llm_used=llm_used,
        llm_status=llm_status,
    )


def _classify_geothermal_phase(text: str) -> str:
    """Return one of: none, exploration, drilling, plant, multi_phase, unknown."""
    txt = (text or "").lower()
    if "geothermal" not in txt:
        return "none"

    matches = []
    for phase, patterns in GEOTHERMAL_PHASE_PATTERNS.items():
        if any(re.search(p, txt, flags=re.IGNORECASE) for p in patterns):
            matches.append(phase)

    if len(matches) == 0:
        return "unknown"
    if len(matches) == 1:
        return matches[0]
    return "multi_phase"


# --------------------------
# DOMAIN EXTRACTORS
# --------------------------

def _add_transmission_columns(
    df: pd.DataFrame,
    full_text: pd.Series,
    context_text: pd.Series,
    type_text: pd.Series,
    use_llm: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    lower_text = full_text.str.lower()
    lower_context = context_text.str.lower()

    out["project_is_transmission_broad"] = (
        lower_text.str.contains(r"\belectricity transmission\b", regex=True)
        | lower_text.str.contains(r"\btransmission line\b", regex=True)
        | lower_text.str.contains(r"\btransmission\b", regex=True)
    )

    out["project_has_transmission_type_tag"] = type_text.str.lower().str.contains(
        r"\belectricity transmission\b", regex=True
    )
    out["project_has_transmission_build_text"] = lower_context.str.contains(TRANSMISSION_BUILD_RE)

    candidates = [
        _extract_length_candidates(txt, TRANSMISSION_HINTS, prefix="tx")
        for txt in full_text.tolist()
    ]

    adjudications = [
        _adjudicate_transmission_length(txt, cands, use_llm=use_llm)
        for txt, cands in zip(full_text.tolist(), candidates)
    ]

    out["project_transmission_length_candidates_json"] = [json.dumps(c, ensure_ascii=False) for c in candidates]
    out["project_transmission_length_candidate_count"] = [a.candidate_count for a in adjudications]
    out["project_transmission_length_distinct_candidate_count"] = [a.distinct_candidate_count for a in adjudications]
    out["project_transmission_length_taxonomy"] = [a.taxonomy for a in adjudications]
    out["project_transmission_length_selection_method"] = [a.selection_method for a in adjudications]
    out["project_transmission_length_selected_candidate_ids"] = [
        json.dumps(a.selected_candidate_ids) for a in adjudications
    ]
    out["project_transmission_length_llm_trigger"] = [a.llm_trigger for a in adjudications]
    out["project_transmission_length_llm_used"] = [a.llm_used for a in adjudications]
    out["project_transmission_length_llm_status"] = [a.llm_status for a in adjudications]

    out["project_transmission_length_miles"] = [a.selected_length_miles for a in adjudications]
    out["project_transmission_length_confidence"] = [a.confidence for a in adjudications]
    out["project_transmission_length_source_text"] = [a.source_text for a in adjudications]

    # Keep only broad transmission projects populated.
    non_broad = ~out["project_is_transmission_broad"]
    out.loc[non_broad, "project_transmission_length_miles"] = np.nan
    out.loc[non_broad, "project_transmission_length_confidence"] = "none"
    out.loc[non_broad, "project_transmission_length_source_text"] = ""
    out.loc[non_broad, "project_transmission_length_candidate_count"] = 0
    out.loc[non_broad, "project_transmission_length_distinct_candidate_count"] = 0
    out.loc[non_broad, "project_transmission_length_taxonomy"] = "none"
    out.loc[non_broad, "project_transmission_length_selection_method"] = "none"
    out.loc[non_broad, "project_transmission_length_selected_candidate_ids"] = "[]"
    out.loc[non_broad, "project_transmission_length_llm_trigger"] = False
    out.loc[non_broad, "project_transmission_length_llm_used"] = False
    out.loc[non_broad, "project_transmission_length_llm_status"] = "not_triggered"
    out.loc[non_broad, "project_transmission_length_candidates_json"] = "[]"

    out["project_is_transmission_strict"] = (
        out["project_has_transmission_type_tag"]
        & out["project_has_transmission_build_text"]
        & (out["project_transmission_length_miles"] >= 1.0)
    )
    out["project_is_transmission"] = out["project_is_transmission_strict"]

    return out


def _add_pipeline_columns(df: pd.DataFrame, full_text: pd.Series) -> pd.DataFrame:
    out = df.copy()
    lower_text = full_text.str.lower()

    out["project_is_pipeline"] = lower_text.str.contains(r"\bpipelines?\b", regex=True)
    out["project_is_carbon_pipeline"] = out["project_is_pipeline"] & lower_text.str.contains(
        r"\b(?:carbon|co2|carbon dioxide)\b", regex=True
    )
    out["project_is_hydrogen_pipeline"] = out["project_is_pipeline"] & lower_text.str.contains(
        r"\bhydrogen\b", regex=True
    )
    out["project_is_natural_gas_pipeline"] = out["project_is_pipeline"] & lower_text.str.contains(
        r"\bnatural gas\b|\bgas pipeline\b", regex=True
    )

    candidates = [
        _extract_length_candidates(txt, PIPELINE_HINTS, prefix="pl")
        for txt in full_text.tolist()
    ]

    best = [_best_single_candidate(cands) for cands in candidates]
    out["project_pipeline_length_candidates_json"] = [json.dumps(c, ensure_ascii=False) for c in candidates]
    out["project_pipeline_length_candidate_count"] = [len(c) for c in candidates]
    out["project_pipeline_length_distinct_candidate_count"] = [len(_collapse_candidates_by_value(c)) for c in candidates]
    out["project_pipeline_length_miles"] = [x[0] for x in best]
    out["project_pipeline_length_confidence"] = [x[1] for x in best]
    out["project_pipeline_length_source_text"] = [x[2] for x in best]

    non_pipeline = ~out["project_is_pipeline"]
    out.loc[non_pipeline, "project_pipeline_length_miles"] = np.nan
    out.loc[non_pipeline, "project_pipeline_length_confidence"] = "none"
    out.loc[non_pipeline, "project_pipeline_length_source_text"] = ""
    out.loc[non_pipeline, "project_pipeline_length_candidate_count"] = 0
    out.loc[non_pipeline, "project_pipeline_length_distinct_candidate_count"] = 0
    out.loc[non_pipeline, "project_pipeline_length_candidates_json"] = "[]"

    out["project_pipeline_group"] = np.select(
        [
            out["project_is_carbon_pipeline"],
            out["project_is_hydrogen_pipeline"],
            out["project_is_natural_gas_pipeline"],
            out["project_is_pipeline"],
        ],
        [
            "carbon_pipeline",
            "hydrogen_pipeline",
            "natural_gas_pipeline",
            "other_pipeline",
        ],
        default="none",
    )

    return out


def _add_geothermal_columns(df: pd.DataFrame, full_text: pd.Series) -> pd.DataFrame:
    out = df.copy()
    lower_text = full_text.str.lower()
    out["project_is_geothermal"] = lower_text.str.contains(r"\bgeothermal\b", regex=True)
    out["project_geothermal_phase"] = full_text.apply(_classify_geothermal_phase)
    return out


# --------------------------
# PUBLIC API
# --------------------------

def normalize_run_targets(run: Sequence[str] | str) -> List[str]:
    if isinstance(run, str):
        targets = [run]
    else:
        targets = list(run)

    targets = [t.strip().lower() for t in targets if t and str(t).strip()]
    if not targets or "all" in targets:
        return ["transmission", "geothermal", "pipeline"]

    allowed = {"transmission", "geothermal", "pipeline"}
    cleaned = [t for t in targets if t in allowed]
    if not cleaned:
        raise ValueError(f"No valid run targets in {targets}. Allowed: {sorted(allowed)}")
    return cleaned


def add_technology_columns(
    df: pd.DataFrame,
    run: Sequence[str] | str = "all",
    use_llm: bool = False,
) -> pd.DataFrame:
    """
    Add technology-specific features to a project dataframe.

    Args:
        df: project dataframe
        run: one or more of transmission/geothermal/pipeline/all
        use_llm: optional LLM adjudication for multi-candidate transmission rows

    Returns:
        DataFrame with requested technology columns added/updated.
    """
    targets = normalize_run_targets(run)
    out = df.copy()

    title_txt = _series_text(out, "project_title")
    desc_txt = _series_text(out, "project_description")
    type_txt = _series_text(out, "project_type")

    full_text = (
        title_txt.fillna("").astype(str)
        + " "
        + desc_txt.fillna("").astype(str)
        + " "
        + type_txt.fillna("").astype(str)
    ).str.strip()

    context_text = (
        title_txt.fillna("").astype(str)
        + " "
        + desc_txt.fillna("").astype(str)
    ).str.strip()

    if "transmission" in targets:
        out = _add_transmission_columns(out, full_text, context_text, type_txt, use_llm=use_llm)

    if "geothermal" in targets:
        out = _add_geothermal_columns(out, full_text)

    if "pipeline" in targets:
        out = _add_pipeline_columns(out, full_text)

    return out


# --------------------------
# CLI
# --------------------------

def _default_projects_combined_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / "data" / "analysis" / "projects_combined.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract technology-specific project features")
    parser.add_argument(
        "--run",
        nargs="+",
        default=["all"],
        choices=["all", "transmission", "geothermal", "pipeline"],
        help="Technology domains to run (default: all)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_projects_combined_path(),
        help="Input parquet file (default: data/analysis/projects_combined.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path (default: overwrite input)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Attempt LLM adjudication for multi-candidate transmission rows",
    )
    return parser


def run_cli(args: argparse.Namespace) -> None:
    in_path = args.input
    out_path = args.output or args.input

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    targets = normalize_run_targets(args.run)
    print(f"Loading: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"Rows loaded: {len(df):,}")
    print(f"Running targets: {', '.join(targets)}")
    print(f"LLM mode: {'on' if args.use_llm else 'off'}")

    updated = add_technology_columns(df, run=targets, use_llm=args.use_llm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    if "transmission" in targets and "project_is_transmission" in updated.columns:
        n_tx = int(updated["project_is_transmission"].fillna(False).sum())
        n_multi = int((updated["project_transmission_length_distinct_candidate_count"].fillna(0) >= 2).sum())
        print(f"Transmission strict projects: {n_tx:,}")
        print(f"Transmission rows with multi distinct candidates: {n_multi:,}")


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    run_cli(cli_args)
