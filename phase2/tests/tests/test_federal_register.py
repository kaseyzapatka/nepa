import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

os.environ["CONDA_DEFAULT_ENV"] = "nepa"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from extract import federal_register  # noqa: E402
from extract.federal_register import (  # noqa: E402
    ANALYSIS_DIR,
    BASE_DIR,
    FEDERAL_REGISTER_DIR,
    FederalRegisterConfig,
    _fetch_raw_text,
    _is_valid_noi_title,
    _process_flags,
    _select_title_phrase,
    build_project_noi_matches,
    fetch_federal_register_noi_corpus,
)


def _row(**overrides):
    base = {
        "project_title": "",
        "project_department": "",
        "lead_agency": "",
        "project_sponsor": "",
        "project_state": "",
        "process_type": "EIS",
        "project_energy_type": "Clean",
    }
    base.update(overrides)
    return pd.Series(base)


def test_phase2_default_paths_do_not_point_to_phase1():
    assert BASE_DIR.name == "phase2"
    assert ANALYSIS_DIR == BASE_DIR / "data" / "analysis"
    assert FEDERAL_REGISTER_DIR == ANALYSIS_DIR / "federal_register"
    assert federal_register.DEFAULT_PROJECT_OUTPUT == FEDERAL_REGISTER_DIR / "noi_federal_register.parquet"
    assert "phase1" not in str(ANALYSIS_DIR)


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


def test_corpus_queries_include_notice_of_preparation():
    assert '"Notice of Preparation"' in federal_register.NOI_CORPUS_QUERIES
    assert '"Notice of Preparation" AND "Environmental Impact Statement"' not in federal_register.NOI_CORPUS_QUERIES
    assert '"Environmental Impact Statement" AND "Intent"' not in federal_register.NOI_CORPUS_QUERIES
    assert '"Environmental Assessment" AND "Intent"' not in federal_register.NOI_CORPUS_QUERIES
    assert _is_valid_noi_title("Notice of Preparation of an Environmental Impact Statement")


def test_date_windows_cover_years_and_quarters():
    assert federal_register._iter_date_windows("2023-06-15", "2024-02-10", "year") == [
        ("2023-06-15", "2023-12-31"),
        ("2024-01-01", "2024-02-10"),
    ]

    assert federal_register._iter_date_windows("2024-02-15", "2024-08-05", "quarter") == [
        ("2024-02-15", "2024-03-31"),
        ("2024-04-01", "2024-06-30"),
        ("2024-07-01", "2024-08-05"),
    ]


def test_corpus_matching_uses_energy_terms_as_distinctive_tokens():
    projects = pd.DataFrame(
        [
            {
                "project_id": "p1",
                "project_title": "Cedar Wind Project",
                "lead_agency": "Bureau of Land Management",
                "project_state": "Wyoming",
                "process_type": "EIS",
                "project_energy_type": "Clean",
            }
        ]
    )
    corpus = pd.DataFrame(
        [
            {
                "fr_document_number": "2024-10001",
                "fr_title": "Notice of Intent To Prepare an Environmental Impact Statement for the Cedar Wind Project",
                "fr_publication_date": "2024-01-02",
                "fr_url": "https://example.test/cedar-wind",
                "fr_agency_names": '["Bureau of Land Management"]',
                "fr_type": "Notice",
                "fr_subtype": "",
                "fr_comments_close_on": "",
                "fr_abstract": "",
                "fr_query_terms": '"Notice of Intent"',
            }
        ]
    )

    project_matches, candidates, review = build_project_noi_matches(projects, corpus)

    assert len(candidates) == 1
    assert "wind" in candidates.loc[0, "title_overlap_tokens"].split(", ")
    assert project_matches.loc[0, "noi_match_status"] == "accepted"
    assert review.empty


def test_process_flags_do_not_treat_standalone_ce_as_categorical_exclusion():
    process_match, process_conflict = _process_flags(
        "CE",
        "Notice of Intent To Prepare an Environmental Impact Statement for the Ce River Project",
    )

    assert process_match is False
    assert process_conflict is True


def test_ce_short_title_overlap_requires_review_instead_of_auto_accept():
    projects = pd.DataFrame(
        [
            {
                "project_id": "ce1",
                "project_title": "Special Recreation Permit",
                "lead_agency": "Bureau of Land Management",
                "project_state": "Montana",
                "process_type": "CE",
                "project_energy_type": "Other",
            }
        ]
    )
    corpus = pd.DataFrame(
        [
            {
                "fr_document_number": "2024-30001",
                "fr_title": "Notice of Intent To Collect Fees for Special Recreation Permits in Montana",
                "fr_publication_date": "2024-02-02",
                "fr_url": "https://example.test/recreation",
                "fr_agency_names": '["Bureau of Land Management"]',
                "fr_type": "Notice",
                "fr_subtype": "",
                "fr_comments_close_on": "",
                "fr_abstract": "",
                "fr_query_terms": '"Notice of Intent"',
            }
        ]
    )

    project_matches, candidates, review = build_project_noi_matches(projects, corpus)

    assert len(candidates) == 1
    assert candidates.loc[0, "match_confidence"] == "medium"
    assert candidates.loc[0, "match_reason"] == "ce_match_requires_distinctive_token_review"
    assert project_matches.loc[0, "noi_match_status"] == "review_required"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert len(review) == 1


def test_withdrawal_notice_goes_to_review_even_with_document_evidence():
    projects = pd.DataFrame(
        [
            {
                "project_id": "p1",
                "project_title": "Withdrawal Basin Solar Project 2024-20001",
                "lead_agency": "Bureau of Land Management",
                "project_state": "Nevada",
                "process_type": "EIS",
                "project_energy_type": "Clean",
            }
        ]
    )
    corpus = pd.DataFrame(
        [
            {
                "fr_document_number": "2024-20001",
                "fr_title": "Withdrawal of Notice of Intent To Prepare an Environmental Impact Statement for the Basin Solar Project",
                "fr_publication_date": "2024-02-02",
                "fr_url": "https://example.test/withdrawal",
                "fr_agency_names": '["Bureau of Land Management"]',
                "fr_type": "Notice",
                "fr_subtype": "",
                "fr_comments_close_on": "",
                "fr_abstract": "",
                "fr_query_terms": '"Notice of Intent"',
            }
        ]
    )

    project_matches, candidates, review = build_project_noi_matches(projects, corpus)

    assert len(candidates) == 1
    assert candidates.loc[0, "match_confidence"] == "medium"
    assert candidates.loc[0, "match_reason"] == "termination_or_withdrawal_notice_requires_review"
    assert project_matches.loc[0, "noi_match_status"] == "review_required"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert len(review) == 1


def test_withdrawal_notice_without_document_evidence_is_not_review_candidate():
    projects = pd.DataFrame(
        [
            {
                "project_id": "p1",
                "project_title": "Basin Solar Project",
                "lead_agency": "Bureau of Land Management",
                "project_state": "Nevada",
                "process_type": "EIS",
                "project_energy_type": "Clean",
            }
        ]
    )
    corpus = pd.DataFrame(
        [
            {
                "fr_document_number": "2024-20001",
                "fr_title": "Withdrawal of Notice of Intent To Prepare an Environmental Impact Statement for the Basin Solar Project",
                "fr_publication_date": "2024-02-02",
                "fr_url": "https://example.test/withdrawal",
                "fr_agency_names": '["Bureau of Land Management"]',
                "fr_type": "Notice",
                "fr_subtype": "",
                "fr_comments_close_on": "",
                "fr_abstract": "",
                "fr_query_terms": '"Notice of Intent"',
            }
        ]
    )

    project_matches, candidates, review = build_project_noi_matches(projects, corpus)

    assert len(candidates) == 1
    assert candidates.loc[0, "match_confidence"] == "low"
    assert candidates.loc[0, "match_reason"] == "termination_or_withdrawal_notice_rejected"
    assert project_matches.loc[0, "noi_match_status"] == "unmatched"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert review.empty


def test_fetch_raw_text_retries(monkeypatch):
    calls = {"n": 0}

    class FakeResponse:
        text = "raw notice text"

        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.RequestException("temporary failure")
        return FakeResponse()

    monkeypatch.setattr(federal_register.requests, "get", fake_get)
    monkeypatch.setattr(federal_register.time, "sleep", lambda _: None)

    assert _fetch_raw_text("https://example.test/raw", max_retries=2, retry_backoff_seconds=0) == "raw notice text"
    assert calls["n"] == 2


def test_fetch_raw_text_raises_after_retries(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, timeout):
        calls["n"] += 1
        raise requests.RequestException("persistent failure")

    monkeypatch.setattr(federal_register.requests, "get", fake_get)
    monkeypatch.setattr(federal_register.time, "sleep", lambda _: None)

    with pytest.raises(requests.RequestException):
        _fetch_raw_text("https://example.test/raw", max_retries=3, retry_backoff_seconds=0)
    assert calls["n"] == 3


def test_corpus_fetch_deduplicates_by_document_number(monkeypatch, tmp_path):
    calls = []

    def fake_search_noi(terms, start_date, end_date, per_page, max_retries, retry_backoff_seconds, page=1):
        calls.append((terms, start_date, end_date, page))
        return {
            "total_pages": 1,
            "results": [
                {
                    "title": "Notice of Intent To Prepare an Environmental Impact Statement for Example Project",
                    "publication_date": "2024-01-02",
                    "document_number": "2024-00001",
                    "html_url": "https://example.test/2024-00001",
                    "agency_names": ["Department of Energy"],
                    "type": "Notice",
                }
            ],
        }

    monkeypatch.setattr(federal_register, "search_noi", fake_search_noi)
    config = FederalRegisterConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        throttle_seconds=0,
        show_progress=False,
    )

    corpus = fetch_federal_register_noi_corpus(config, cache_path=tmp_path / "fr_cache.json")

    assert len(corpus) == 1
    assert corpus.loc[0, "fr_document_number"] == "2024-00001"
    assert corpus.loc[0, "fr_query_count"] == len(federal_register.NOI_CORPUS_QUERIES)
    assert len(calls) == len(federal_register.NOI_CORPUS_QUERIES)
    assert {call[1:3] for call in calls} == {("2024-01-01", "2024-12-31")}


def test_capped_year_window_splits_to_quarters_and_writes_report(monkeypatch, tmp_path):
    calls = []

    def fake_search_noi(terms, start_date, end_date, per_page, max_retries, retry_backoff_seconds, page=1):
        calls.append((terms, start_date, end_date, page))
        if start_date == "2024-01-01" and end_date == "2024-12-31":
            return {
                "count": 5000,
                "total_pages": 50,
                "results": [
                    {
                        "title": "Parent capped window should be split",
                        "publication_date": "2024-01-01",
                        "document_number": "parent-window",
                    }
                ],
            }
        return {
            "count": 1,
            "total_pages": 1,
            "results": [
                {
                    "title": f"Notice of Intent To Prepare an Environmental Impact Statement {start_date}",
                    "publication_date": start_date,
                    "document_number": f"doc-{start_date}",
                }
            ],
        }

    monkeypatch.setattr(federal_register, "NOI_CORPUS_QUERIES", ('"Notice of Intent"',))
    monkeypatch.setattr(federal_register, "search_noi", fake_search_noi)
    config = FederalRegisterConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        throttle_seconds=0,
        show_progress=False,
    )
    report_path = tmp_path / "fetch_report.csv"

    corpus = fetch_federal_register_noi_corpus(
        config,
        cache_path=tmp_path / "fr_cache.json",
        fetch_report_output=report_path,
    )

    assert len(corpus) == 4
    assert "parent-window" not in set(corpus["fr_document_number"])
    assert calls == [
        ('"Notice of Intent"', "2024-01-01", "2024-12-31", 1),
        ('"Notice of Intent"', "2024-01-01", "2024-03-31", 1),
        ('"Notice of Intent"', "2024-04-01", "2024-06-30", 1),
        ('"Notice of Intent"', "2024-07-01", "2024-09-30", 1),
        ('"Notice of Intent"', "2024-10-01", "2024-12-31", 1),
    ]

    report = pd.read_csv(report_path)
    assert len(report) == 5
    parent = report[report["window_level"] == "year"].iloc[0]
    assert bool(parent["capped"])
    assert bool(parent["split"])
    assert parent["split_to"] == "quarter"
    assert set(report["window_level"]) == {"year", "quarter"}


def test_cache_key_includes_date_window(monkeypatch, tmp_path):
    calls = []

    def fake_search_noi(terms, start_date, end_date, per_page, max_retries, retry_backoff_seconds, page=1):
        calls.append((terms, start_date, end_date, page))
        return {
            "count": 1,
            "total_pages": 1,
            "results": [
                {
                    "title": f"Notice of Intent To Prepare an Environmental Impact Statement {start_date}",
                    "publication_date": start_date,
                    "document_number": f"doc-{start_date}",
                }
            ],
        }

    monkeypatch.setattr(federal_register, "NOI_CORPUS_QUERIES", ('"Notice of Intent"',))
    monkeypatch.setattr(federal_register, "search_noi", fake_search_noi)
    cache_path = tmp_path / "fr_cache.json"

    config_2024 = FederalRegisterConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        throttle_seconds=0,
        show_progress=False,
    )
    config_2025 = FederalRegisterConfig(
        start_date="2025-01-01",
        end_date="2025-12-31",
        throttle_seconds=0,
        show_progress=False,
    )

    corpus_2024 = fetch_federal_register_noi_corpus(config_2024, cache_path=cache_path)
    corpus_2025 = fetch_federal_register_noi_corpus(config_2025, cache_path=cache_path)

    assert list(corpus_2024["fr_document_number"]) == ["doc-2024-01-01"]
    assert list(corpus_2025["fr_document_number"]) == ["doc-2025-01-01"]
    assert calls == [
        ('"Notice of Intent"', "2024-01-01", "2024-12-31", 1),
        ('"Notice of Intent"', "2025-01-01", "2025-12-31", 1),
    ]


def test_multiple_high_confidence_matches_are_ambiguous():
    projects = pd.DataFrame(
        [
            {
                "project_id": "p1",
                "project_title": "TransWest Express Transmission Project",
                "lead_agency": "Department of Energy",
                "project_state": "Wyoming",
                "process_type": "EIS",
                "project_energy_type": "Clean",
            }
        ]
    )
    corpus = pd.DataFrame(
        [
            {
                "fr_document_number": "2011-00001",
                "fr_title": (
                    "Notice of Intent To Prepare an Environmental Impact Statement for "
                    "the TransWest Express Transmission Project in Wyoming"
                ),
                "fr_publication_date": "2011-01-14",
                "fr_url": "https://example.test/1",
                "fr_agency_names": '["Department of Energy"]',
                "fr_type": "Notice",
                "fr_subtype": "",
                "fr_comments_close_on": "",
                "fr_abstract": "",
                "fr_query_terms": '"Notice of Intent"',
            },
            {
                "fr_document_number": "2011-00002",
                "fr_title": (
                    "Notice of Intent To Prepare an Environmental Impact Statement for "
                    "TransWest Express Transmission Project, Wyoming"
                ),
                "fr_publication_date": "2011-01-15",
                "fr_url": "https://example.test/2",
                "fr_agency_names": '["Department of Energy"]',
                "fr_type": "Notice",
                "fr_subtype": "",
                "fr_comments_close_on": "",
                "fr_abstract": "",
                "fr_query_terms": '"Notice of Intent"',
            },
        ]
    )

    project_matches, candidates, review = build_project_noi_matches(projects, corpus)

    assert len(project_matches) == 1
    assert project_matches.loc[0, "noi_match_status"] == "ambiguous"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert len(candidates) == 2
    assert len(review) == 2


def test_ce_projects_are_included_even_when_unmatched():
    projects = pd.DataFrame(
        [
            {
                "project_id": "ce1",
                "project_title": "Small Categorical Exclusion Project",
                "lead_agency": "Department of Energy",
                "project_state": "Nevada",
                "process_type": "CE",
                "project_energy_type": "Other",
            }
        ]
    )
    corpus = pd.DataFrame(columns=federal_register.CORPUS_OUTPUT_COLUMNS)

    project_matches, candidates, review = build_project_noi_matches(projects, corpus)

    assert list(project_matches["project_id"]) == ["ce1"]
    assert project_matches.loc[0, "noi_match_status"] == "unmatched"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert candidates.empty
    assert review.empty


def test_project_output_has_one_row_per_project_id():
    projects = pd.DataFrame(
        [
            {"project_id": "p1", "project_title": "Example Project", "process_type": "EA"},
            {"project_id": "p1", "project_title": "Example Project Duplicate", "process_type": "EA"},
            {"project_id": "p2", "project_title": "Other Project", "process_type": "EIS"},
        ]
    )
    corpus = pd.DataFrame(columns=federal_register.CORPUS_OUTPUT_COLUMNS)

    project_matches, _, _ = build_project_noi_matches(projects, corpus)

    assert list(project_matches["project_id"]) == ["p1", "p2"]


def test_extract_data_analysis_pipeline_passes_refresh_flag(monkeypatch):
    from extract import extract_data  # noqa: E402

    calls = []
    monkeypatch.setattr(extract_data, "run_project_description_enrichment", lambda: {})
    monkeypatch.setattr(
        extract_data,
        "create_combined_projects",
        lambda refresh_federal_register=False: calls.append(refresh_federal_register),
    )
    monkeypatch.setattr(extract_data, "create_combined_processes", lambda: None)
    monkeypatch.setattr(extract_data, "create_combined_documents", lambda: None)

    extract_data.run_analysis_pipeline()
    extract_data.run_analysis_pipeline(refresh_federal_register=True)

    assert calls == [False, True]
