import os
import sys
from pathlib import Path

import pandas as pd

os.environ["CONDA_DEFAULT_ENV"] = "nepa"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from extract.federal_register import (  # noqa: E402
    _build_search_plans,
    _select_title_phrase,
    pick_best_noi,
)


def _row(**overrides):
    base = {
        "project_title": "",
        "project_department": "",
        "lead_agency": "",
        "project_sponsor": "",
        "project_state": "",
    }
    base.update(overrides)
    return pd.Series(base)


def test_select_title_phrase_strips_generic_prefixes():
    assert _select_title_phrase("Construction and Operation of a Human Genome Laboratory") == (
        "Human Genome Laboratory"
    )

    phrase = _select_title_phrase("License Renewal of Diablo Canyon Nuclear Power Plant, Units 1 and 2")
    assert phrase.lower().startswith("diablo canyon nuclear power plant")
    assert "license renewal" not in phrase.lower()

    phrase = _select_title_phrase(
        "Department of Energy Loan Guarantee for U.S. Geothermal's Neal Hot Springs Geothermal Facility"
    )
    assert "neal hot springs geothermal facility" in phrase.lower()
    assert not phrase.lower().startswith("s ")


def test_build_search_plans_use_cleaned_title_and_keyword_fallback():
    row = _row(
        project_title="Construction and Operation of a Human Genome Laboratory",
        lead_agency="Department of Energy",
        project_state="Tennessee",
    )

    plans = dict(_build_search_plans(row))

    assert "title_only" in plans
    assert '"Human Genome Laboratory"' in plans["title_only"]
    assert "keywords_agency_state" in plans
    assert "human" in plans["keywords_agency_state"]
    assert "genome" in plans["keywords_agency_state"]


def test_pick_best_noi_rejects_low_overlap_false_positive():
    row = _row(
        project_title="Goldendale Energy Storage Project",
        lead_agency="Department of Energy",
        project_state="Washington",
    )

    results = {
        "results": [
            {
                "title": "FFP Project 101, LLC; Notice of Intent To Prepare an Environmental Impact Statement",
                "agency_names": ["Department of Energy"],
                "publication_date": "2023-05-01",
            }
        ]
    }

    assert pick_best_noi(results, row) is None


def test_pick_best_noi_accepts_strong_title_match():
    row = _row(
        project_title="TransWest Express Transmission Project",
        lead_agency="Department of Energy",
        project_state="Wyoming",
    )

    results = {
        "results": [
            {
                "title": (
                    "Notice of Intent To Prepare an Environmental Impact Statement for the "
                    "TransWest Express 600 kV Direct Current Transmission Project in Wyoming, "
                    "Colorado, Utah, and Nevada"
                ),
                "agency_names": ["Department of Energy"],
                "publication_date": "2011-01-14",
            }
        ]
    }

    best = pick_best_noi(results, row)

    assert best is not None
    assert best["title_overlap_count"] >= 2
    assert best["match_score"] >= 9
