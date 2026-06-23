"""D6 — canonical existing-CE source loader.

Single source of truth for *existing federal categorical exclusions*, used by the
CE crosswalk (n04) and the human-readable catalog renderer (extract_ce_catalog.py).

Source: the CE Explorer export, committed at `phase2/notes/deliverable06/ce.json`
(`{version, exclusions:[...]}`). Chosen over the CEQ government-wide spreadsheet
because it is already structured (one clean record per CE, with canonical eCFR
URLs), needs only stdlib JSON (no openpyxl), and is the same source the v1
pipeline used. CE Explorer is a discovery index — final dossiers should still cite
each CE's canonical `canonical_source_url` (eCFR).

CE Explorer: https://ce.permitting.innovation.gov/data/exclusions.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CE_JSON = HERE.parents[1] / "notes" / "deliverable06" / "ce.json"
CE_EXPLORER_URL = "https://ce.permitting.innovation.gov/data/exclusions.json"


def catalog_version() -> dict:
    """Return the CE Explorer export's `version` block ({version, date})."""
    return json.loads(CE_JSON.read_text()).get("version", {})


def load_ce_catalog() -> pd.DataFrame:
    """Load ce.json into a snapshot-compatible DataFrame (one row per existing CE).

    Column names match the legacy ce_explorer_snapshot schema so the crosswalk can
    consume it unchanged.
    """
    data = json.loads(CE_JSON.read_text())
    version = data.get("version", {})
    df = pd.DataFrame(data.get("exclusions", []))
    out = pd.DataFrame({
        "ce_id": df.get("id"),
        "structured_id": df.get("structuredID"),
        "agency_unit": df.get("unit"),
        "agency_name": df.get("longUnit"),
        "origin": df.get("origin"),
        "canonical_source_url": df.get("originUrl"),
        "context": df.get("context"),
        "additional_context": df.get("additionalContext"),
        "extraordinary_circumstances": df.get("circumstances"),
        "ce_description": df.get("exclusion"),
        "source_url": CE_EXPLORER_URL,
        "source_version": str(version.get("version", "")),
        "source_version_date": str(version.get("date", "")),
    })
    return out
