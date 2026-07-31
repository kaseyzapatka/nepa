"""Shared D2/D6 extraction of mitigation, design, and monitoring conditions."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests


RESOURCE_AREAS = {
    "air_quality": [
        "air quality", "air pollution", "criteria pollutant", "naaqs",
        "particulate matter", "pm2.5", "pm10", "ozone", "nox", "voc",
        "greenhouse gas", "ghg",
    ],
    "water": [
        "water quality", "groundwater", "surface water", "wetland", "floodplain",
        "stormwater", "runoff", "aquifer", "stream", "creek", "river",
        "section 404", "section 401",
    ],
    "biological": [
        "biological", "wildlife", "species", "habitat", "vegetation",
        "threatened", "endangered", "esa", "migratory bird", "raptor",
        "bat", "fish", "amphibian",
    ],
    "cultural": [
        "cultural", "historic", "archaeological", "tribal", "section 106",
        "nhpa", "traditional cultural", "sacred", "paleontological",
    ],
    "visual": [
        "visual", "viewshed", "aesthetics", "scenic", "landscape character",
        "visual resources", "visual contrast",
    ],
    "noise": [
        "noise", "vibration", "decibel", "dba", "sound level",
        "construction noise", "operational noise",
    ],
    "soils_geology": [
        "soil", "geology", "geologic", "erosion", "compaction",
        "landslide", "seismic", "subsidence", "prime farmland",
    ],
    "socioeconomic": [
        "socioeconomic", "employment", "jobs", "economy", "income",
        "population", "community", "environmental justice", "ej",
    ],
    "transportation": [
        "transportation", "traffic", "road", "highway", "access",
        "haul route", "vehicle trip", "level of service",
    ],
    "land_use": [
        "land use", "zoning", "compatibility", "adjacent land",
        "general plan", "consistency", "right-of-way", "right of way",
    ],
    "climate_ghg": [
        "climate change", "greenhouse gas", "ghg", "carbon", "co2",
        "methane", "emissions", "global warming",
    ],
    "public_health": [
        "public health", "human health", "hazardous material", "hazmat",
        "toxic", "contamination", "exposure", "risk assessment",
    ],
}

CONDITION_ROLES = [
    "baseline_design_feature",
    "best_management_practice",
    "mitigation_commitment",
    "monitoring_requirement",
    "enforcement_or_permit_condition",
    "legal_or_procedural_boilerplate",
    "uncertain",
]

PROMPT_VERSION = "d6_condition_roles_v1"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(])|\n+")
CONDITION_CANDIDATE_RE = re.compile(
    r"\b(?:shall|must|required?|will|commit(?:s|ted)?|should|may not|"
    r"avoid|limit(?:ed|s)?|restrict(?:ed|s)?|prohibit(?:ed|s)?|"
    r"mitigat(?:e|ed|es|ion)|minimi[sz](?:e|ed|es|ation)|"
    r"best management practices?|bmps?|monitor(?:ing|ed)?|inspect(?:ion|ed)?|"
    r"restore|revegetat(?:e|ed|ion)|setbacks?|no more than|not exceed|"
    r"permit condition|condition of approval|comply|compliance)\b",
    re.IGNORECASE,
)


def _normalize(text: object) -> str:
    return re.sub(r"\s+", " ", "" if text is None else str(text)).strip()


def _sha256(text: object) -> str:
    return hashlib.sha256(_normalize(text).lower().encode("utf-8")).hexdigest()


def classify_resource_area(text: str) -> str:
    lower = text.lower()
    scores = {
        area: sum(lower.count(keyword) for keyword in keywords)
        for area, keywords in RESOURCE_AREAS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] else "unknown"


def classify_resource_area_with_heading(text: str, heading: object = "") -> str:
    """D6 #47 Tier-1: keyword-classify the sentence; if that is 'unknown', inherit the
    resource area from the section HEADING (headings like 'Wildlife', 'Cultural Resources',
    'Water Resources' name a resource explicitly, where the sentence may not). Generic
    structural headings ('Environmental Consequences', 'Mitigation Measures', 'Decision',
    'Finding of No Significant Impact') contain no resource keyword and so stay 'unknown'.

    Pilot (phase2/notes/deliverable06/pilot47_findings.md): resolves ~4.8% of unknowns at
    ~0.72 enrichment-agreement — high precision, opt-in so no existing build changes silently.
    """
    area = classify_resource_area(text)
    if area != "unknown":
        return area
    heading_str = _normalize(heading)
    if heading_str:
        return classify_resource_area(heading_str)
    return "unknown"


def classify_obligation(text: str) -> str:
    if re.search(r"\b(?:shall|must|required|may not|will not|not exceed)\b", text, re.I):
        return "required"
    if re.search(r"\b(?:will|commit(?:s|ted)?|condition of approval)\b", text, re.I):
        return "committed"
    if re.search(r"\b(?:should|recommended?|where feasible)\b", text, re.I):
        return "recommended"
    if re.search(r"\b(?:may|could|can)\b", text, re.I):
        return "uncertain"
    return "descriptive"


def classify_condition_role(text: str) -> tuple[str, str]:
    if re.search(r"\b(?:monitor|inspect|survey|report|audit|verify)\w*\b", text, re.I):
        return "monitoring_requirement", "high"
    if re.search(r"\b(?:best management practices?|bmps?)\b", text, re.I):
        return "best_management_practice", "high"
    if re.search(r"\b(?:mitigat|avoid|minimi[sz]|restore|revegetat|compensat)\w*\b", text, re.I):
        return "mitigation_commitment", "high"
    if re.search(r"\b(?:permit condition|condition of approval|comply|compliance|enforce)\w*\b", text, re.I):
        return "enforcement_or_permit_condition", "medium"
    if re.search(r"\b(?:setback|disturbance|access road|design feature|no more than|not exceed|limited to)\b", text, re.I):
        return "baseline_design_feature", "medium"
    if re.search(r"\b(?:pursuant to|consultation|national environmental policy act|nepa)\b", text, re.I):
        return "legal_or_procedural_boilerplate", "medium"
    return "uncertain", "low"


def _llm_prompt(text: str) -> str:
    return (
        "Classify one NEPA evidence sentence. Return JSON only with keys "
        "condition_role, obligation_level, confidence. condition_role must be "
        f"one of {CONDITION_ROLES}. obligation_level must be required, committed, "
        "recommended, descriptive, or uncertain. confidence must be high, medium, "
        f"or low.\nSentence: {text}"
    )


def _parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(match.group(0))


def _classify_with_llm(text: str, provider: str, model: str) -> dict:
    prompt = _llm_prompt(text)
    if provider == "ollama":
        response = requests.post(
            os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate"),
            json={"model": model or "llama3.2:3b-instruct-q4_K_M", "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return _parse_json_object(response.json()["response"])
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for llm_provider='anthropic'")
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model or "claude-haiku-4-5-20251001",
                "max_tokens": 160,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        response.raise_for_status()
        return _parse_json_object(response.json()["content"][0]["text"])
    raise ValueError("llm_provider must be 'ollama' or 'anthropic' when use_llm=True")


def _value(row: pd.Series, *names: str, default: object = "") -> object:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def extract_condition_rows(
    spans: pd.DataFrame,
    *,
    use_llm: bool = False,
    llm_provider: str = "",
    llm_model: str = "",
    use_heading_inheritance: bool = False,
) -> pd.DataFrame:
    """Return normalized condition rows for bounded evidence spans.

    use_heading_inheritance (D6 #47 Tier-1): when True, a sentence the keyword dict cannot
    place inherits the resource area from its section heading. Default False preserves the
    exact behavior of every existing caller (the current fonsi_conditions build); the D6
    #47 re-tag path enables it. See classify_resource_area_with_heading().
    """
    run_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    for _, span in spans.iterrows():
        source_text = _normalize(_value(span, "span_text", "evidence_text", "section_text"))
        if not source_text:
            continue
        for sentence in SENTENCE_SPLIT_RE.split(source_text):
            sentence = _normalize(sentence)
            if len(sentence) < 20 or not CONDITION_CANDIDATE_RE.search(sentence):
                continue
            role, confidence = classify_condition_role(sentence)
            obligation = classify_obligation(sentence)
            method = "deterministic_parser"
            llm_run_at = ""
            provider = ""
            model = ""
            if use_llm and (role == "uncertain" or confidence == "low"):
                result = _classify_with_llm(sentence, llm_provider, llm_model)
                if result.get("condition_role") in CONDITION_ROLES:
                    role = result["condition_role"]
                if result.get("obligation_level") in {
                    "required", "committed", "recommended", "descriptive", "uncertain"
                }:
                    obligation = result["obligation_level"]
                if result.get("confidence") in {"high", "medium", "low"}:
                    confidence = result["confidence"]
                method = "llm"
                llm_run_at = datetime.now(timezone.utc).isoformat()
                provider = llm_provider
                model = llm_model
            records.append(
                {
                    "project_id": _value(span, "project_id"),
                    "document_id": _value(span, "document_id"),
                    "page_number": _value(span, "page_number", "page_start"),
                    "section_id": _value(span, "section_id", "evidence_span_id"),
                    "resource_area": (
                        classify_resource_area_with_heading(
                            sentence, _value(span, "heading_title", "heading_raw"))
                        if use_heading_inheritance
                        else classify_resource_area(sentence)
                    ),
                    "condition_text": sentence,
                    "condition_role": role,
                    "obligation_level": obligation,
                    "extraction_method": method,
                    "confidence": confidence,
                    "source_span_sha256": _sha256(sentence),
                    "condition_extraction_run_at": run_at,
                    "condition_llm_run_at": llm_run_at,
                    "llm_provider": provider,
                    "llm_model": model,
                    "prompt_version": PROMPT_VERSION if method == "llm" else "",
                }
            )
    out = pd.DataFrame.from_records(records, columns=[
        "project_id", "document_id", "page_number", "section_id", "resource_area",
        "condition_text", "condition_role", "obligation_level", "extraction_method",
        "confidence", "source_span_sha256", "condition_extraction_run_at",
        "condition_llm_run_at", "llm_provider", "llm_model", "prompt_version",
    ])
    return out.drop_duplicates(
        subset=[
            "project_id", "document_id", "source_span_sha256",
            "condition_role", "obligation_level",
        ],
        keep="first",
    ).reset_index(drop=True)


def rollup_conditions_to_significance_rows(
    significance_rows: pd.DataFrame,
    condition_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Attach determination-level mitigation flags and matched-row counts for D2."""
    out = significance_rows.copy()
    relevant = condition_rows.loc[
        condition_rows["condition_role"].isin(
            ["best_management_practice", "mitigation_commitment", "monitoring_requirement"]
        )
    ].copy()
    keys = ["project_id"]
    if "document_id" in out.columns and "document_id" in relevant.columns:
        keys.append("document_id")
    counts = relevant.groupby(keys).size().rename("matched_condition_row_count").reset_index()
    out = out.merge(counts, how="left", on=keys)
    out["matched_condition_row_count"] = out["matched_condition_row_count"].fillna(0).astype(int)
    out["mitigation_flag"] = out["matched_condition_row_count"].gt(0)
    return out


def project_has_mitigation_condition(
    condition_rows: pd.DataFrame,
    *,
    project_id: str,
) -> bool:
    """Return whether one project has a relevant BMP or mitigation condition."""
    return bool(
        condition_rows.loc[
            condition_rows["project_id"].astype(str).eq(str(project_id)),
            "condition_role",
        ].isin(["best_management_practice", "mitigation_commitment"]).any()
    )
