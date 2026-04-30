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
    _build_noa_title_search_term,
    _fetch_raw_text,
    _is_noi_title,
    _is_noa_title,
    _noa_proximity_check,
    _process_flags,
    _select_title_phrase,
    _supplement_noa_by_title_search,
    build_project_noi_matches,
    build_project_noa_matches,
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
    assert federal_register.DEFAULT_PROJECT_OUTPUT == FEDERAL_REGISTER_DIR / "federal_register.parquet"
    assert federal_register.DEFAULT_NOI_CORPUS_OUTPUT == FEDERAL_REGISTER_DIR / "noi_corups.parquet"
    assert federal_register.DEFAULT_NOA_CORPUS_OUTPUT == FEDERAL_REGISTER_DIR / "noa_corpus.parquet"
    assert federal_register.DEFAULT_NOI_CANDIDATES_OUTPUT == FEDERAL_REGISTER_DIR / "noi_candidates.parquet"
    assert federal_register.DEFAULT_NOA_CANDIDATES_OUTPUT == FEDERAL_REGISTER_DIR / "noa_candidates.parquet"
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
    assert '"Notice of Scoping"' in federal_register.NOI_CORPUS_QUERIES
    assert '"Notice of Public Scoping"' in federal_register.NOI_CORPUS_QUERIES
    assert '"Notice of Preparation" AND "Environmental Impact Statement"' not in federal_register.NOI_CORPUS_QUERIES
    assert '"Environmental Impact Statement" AND "Intent"' not in federal_register.NOI_CORPUS_QUERIES
    assert '"Environmental Assessment" AND "Intent"' not in federal_register.NOI_CORPUS_QUERIES
    assert _is_noi_title("Notice of Preparation of an Environmental Impact Statement")
    assert _is_noi_title("Notice of Scoping for Proposed Solar Project")


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
    assert project_matches.loc[0, "noi_match_status"] == "review_required"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert len(review) == 1


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
    assert project_matches.loc[0, "noi_match_status"] == "review_required"
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


def test_normalize_fr_doc_number_ascii_hyphen():
    from extract.federal_register import _normalize_fr_doc_number
    assert _normalize_fr_doc_number("2024-05618") == "2024-05618"


def test_normalize_fr_doc_number_en_dash():
    from extract.federal_register import _normalize_fr_doc_number
    assert _normalize_fr_doc_number("2024\u201305618") == "2024-05618"


def test_normalize_fr_doc_number_em_dash():
    from extract.federal_register import _normalize_fr_doc_number
    assert _normalize_fr_doc_number("2024\u201405618") == "2024-05618"


def test_extract_fr_doc_numbers_from_text():
    from extract.federal_register import _extract_fr_doc_numbers_from_text
    text = "This notice [FR Doc. 2024\u201305618 Filed 3\u201315\u201324; 8:45 am] is a test."
    results = _extract_fr_doc_numbers_from_text(text)
    assert len(results) == 1
    normalized, raw, pos = results[0]
    assert normalized == "2024-05618"
    assert raw == "2024\u201305618"
    assert pos > 0


def test_required_title_overlap_scales_with_token_count():
    from extract.federal_register import _required_title_overlap
    assert _required_title_overlap(0) == 1   # no tokens → threshold can never be met
    assert _required_title_overlap(1) == 1   # 1 token → require all (1)
    assert _required_title_overlap(2) == 2   # 2 tokens → require all (2)
    assert _required_title_overlap(3) == 2   # 3 tokens → require 2
    assert _required_title_overlap(4) == 3   # 4 tokens → require 3
    assert _required_title_overlap(10) == 3  # 10 tokens → still 3


def test_extract_fr_doc_numbers_parenthetical_and_bare():
    """Regex captures FR Doc numbers in parenthetical and bare (no bracket) forms."""
    from extract.federal_register import _extract_fr_doc_numbers_from_text

    # Parenthetical form: (FR Doc. 2020-11111)
    results_paren = _extract_fr_doc_numbers_from_text("(FR Doc. 2020-11111)")
    assert len(results_paren) == 1
    assert results_paren[0][0] == "2020-11111"

    # Bare prose form: ...published as FR Doc. 2021-22222...
    results_bare = _extract_fr_doc_numbers_from_text("published as FR Doc. 2021-22222 in the Federal Register")
    assert len(results_bare) == 1
    assert results_bare[0][0] == "2021-22222"

    # Original bracket form still works
    results_bracket = _extract_fr_doc_numbers_from_text("[FR Doc. 2024-05618 Filed 3-15-24; 8:45 am]")
    assert len(results_bracket) == 1
    assert results_bracket[0][0] == "2024-05618"


def test_extract_fr_url_doc_number():
    from extract.federal_register import _extract_fr_url_doc_number
    text = "See federalregister.gov/documents/2024/03/18/2024-05618/jackalope-wind for more info."
    results = _extract_fr_url_doc_number(text)
    assert len(results) == 1
    normalized, url_raw, pos = results[0]
    assert normalized == "2024-05618"


def test_noi_proximity_check_accepts_nearby_phrase():
    from extract.federal_register import _noi_proximity_check
    text = "Notice of Intent to Prepare an EIS [FR Doc. 2024-05618 Filed 3-15-24; 8:45 am]"
    # Position of the FR Doc bracket
    pos = text.index("[FR Doc")
    result = _noi_proximity_check(text, pos, "EIS", window=500)
    # The function returns the nearest phrase; "intent to prepare" is closer to the bracket than "notice of intent"
    assert result is not None
    assert result.lower() in ("notice of intent", "intent to prepare", "notice to prepare", "notice of preparation", "notice of scoping", "notice of public scoping")


def test_noi_proximity_check_rejects_distant_phrase():
    from extract.federal_register import _noi_proximity_check
    # FR doc appears far from any NOI phrase
    text = "[FR Doc. 2024-05618 Filed 3-15-24; 8:45 am]" + " " * 600 + "Notice of Intent"
    pos = 0
    result = _noi_proximity_check(text, pos, "EIS", window=500)
    assert result is None


def test_parse_fr_date_text_extracts_date():
    from extract.federal_register import _parse_fr_date_text
    text = "Published in the Federal Register on March 18, 2024"
    raw, parsed = _parse_fr_date_text(text)
    assert parsed == "2024-03-18"
    assert "March 18, 2024" in raw


def test_parse_fr_date_text_returns_empty_when_no_match():
    from extract.federal_register import _parse_fr_date_text
    text = "This document was submitted on March 18, 2024 for review."
    raw, parsed = _parse_fr_date_text(text)
    assert raw == ""
    assert parsed == ""


def test_direct_doc_number_evidence_accepts_with_corroboration():
    """NEPATEC doc number evidence + title corroboration -> accepted (not just review)."""
    projects = pd.DataFrame(
        [
            {
                "project_id": "p1",
                "project_title": "Jackalope Wind Energy Project",
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
                "fr_document_number": "2024-05618",
                "fr_title": "Notice of Intent To Prepare an Environmental Impact Statement for the Jackalope Wind Energy Project",
                "fr_publication_date": "2024-03-18",
                "fr_url": "https://example.test/jackalope",
                "fr_agency_names": '["Bureau of Land Management"]',
                "fr_type": "Notice",
                "fr_subtype": "",
                "fr_comments_close_on": "",
                "fr_abstract": "",
                "fr_query_terms": '"Notice of Intent"',
            }
        ]
    )
    # Simulate NEPATEC evidence table: project p1 has doc number 2024-05618 in its pages
    nepatec_evidence = pd.DataFrame(
        [
            {
                "project_id": "p1",
                "process_type": "EIS",
                "project_title": "Jackalope Wind Energy Project",
                "document_id": "doc-abc",
                "file_name": "Jackalope Federal Register Notice.pdf",
                "document_title": "Federal Register Notice",
                "document_type": "FR Notice",
                "main_document": True,
                "page_number": 1,
                "evidence_type": "fr_doc_noi",
                "fr_document_number": "2024-05618",
                "fr_document_number_raw": "2024\u201305618",
                "fr_url": "",
                "fr_citation": "",
                "fr_date_text": "",
                "fr_date_text_parsed": "",
                "notice_title_snippet": "notice of intent",
                "evidence_context": "[FR Doc. 2024\u201305618 Filed 3\u201315\u201324; 8:45 am]",
                "nearby_noi_phrase": "notice of intent",
                "nearby_project_title_token_count": 2,
                "evidence_rank": 1,
            }
        ]
    )

    project_matches, candidates, review = build_project_noi_matches(
        projects, corpus, nepatec_evidence=nepatec_evidence
    )

    assert project_matches.loc[0, "noi_match_status"] == "accepted"
    assert project_matches.loc[0, "noi_publication_date"] == "2024-03-18"
    assert project_matches.loc[0, "noi_nepatec_evidence_file_name"] == "Jackalope Federal Register Notice.pdf"
    assert candidates.loc[0, "nepatec_fr_document_number_evidence"] == True  # noqa: E712


@pytest.mark.parametrize(
    "wrapped_project_id",
    [
        {"value": "p1"},
        "{'value': 'p1'}",
        '{"value": "p1"}',
    ],
)
def test_direct_doc_number_evidence_accepts_wrapped_project_ids(wrapped_project_id):
    """NEPATEC evidence project IDs can arrive as Arrow struct wrappers."""
    projects = pd.DataFrame(
        [
            {
                "project_id": "p1",
                "project_title": "Jackalope Wind Energy Project",
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
                "fr_document_number": "2024-05618",
                "fr_title": "Notice of Intent To Prepare an Environmental Impact Statement for the Jackalope Wind Energy Project",
                "fr_publication_date": "2024-03-18",
                "fr_url": "https://example.test/jackalope",
                "fr_agency_names": '["Bureau of Land Management"]',
                "fr_type": "Notice",
                "fr_subtype": "",
                "fr_comments_close_on": "",
                "fr_abstract": "",
                "fr_query_terms": '"Notice of Intent"',
            }
        ]
    )
    nepatec_evidence = pd.DataFrame(
        [
            {
                "project_id": wrapped_project_id,
                "process_type": "EIS",
                "project_title": "Jackalope Wind Energy Project",
                "document_id": "doc-abc",
                "file_name": "Jackalope Federal Register Notice.pdf",
                "document_title": "Federal Register Notice",
                "document_type": "FR Notice",
                "main_document": True,
                "page_number": 1,
                "evidence_type": "fr_doc_noi",
                "fr_document_number": "2024-05618",
                "fr_document_number_raw": "2024\u201305618",
                "fr_url": "",
                "fr_citation": "",
                "fr_date_text": "",
                "fr_date_text_parsed": "",
                "notice_title_snippet": "notice of intent",
                "evidence_context": "[FR Doc. 2024\u201305618 Filed 3\u201315\u201324; 8:45 am]",
                "nearby_noi_phrase": "notice of intent",
                "nearby_project_title_token_count": 2,
                "evidence_rank": 1,
            }
        ]
    )

    project_matches, candidates, _ = build_project_noi_matches(
        projects,
        corpus,
        nepatec_evidence=nepatec_evidence,
    )

    assert project_matches.loc[0, "project_id"] == "p1"
    assert project_matches.loc[0, "noi_match_status"] == "accepted"
    assert project_matches.loc[0, "noi_publication_date"] == "2024-03-18"
    assert candidates.loc[0, "project_id"] == "p1"
    assert candidates.loc[0, "nepatec_fr_document_number_evidence"] == True  # noqa: E712


def test_weak_fr_url_evidence_requires_review():
    """URL-only evidence is weaker than a nearby NOI FR Doc bracket."""
    projects = pd.DataFrame(
        [
            {
                "project_id": "p1",
                "project_title": "Cedar Wind Project",
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
                "fr_document_number": "2024-10001",
                "fr_title": "Notice of Intent To Prepare an Environmental Impact Statement for the Sagebrush Solar Project, Nevada",
                "fr_publication_date": "2024-01-02",
                "fr_url": "https://example.test/sagebrush",
                "fr_agency_names": '["Bureau of Land Management"]',
                "fr_type": "Notice",
                "fr_subtype": "",
                "fr_comments_close_on": "",
                "fr_abstract": "",
                "fr_query_terms": '"Notice of Intent"',
            }
        ]
    )
    nepatec_evidence = pd.DataFrame(
        [
            {
                "project_id": "p1",
                "process_type": "EIS",
                "project_title": "Cedar Wind Project",
                "document_id": "doc1",
                "file_name": "references.pdf",
                "document_title": "References",
                "document_type": "Appendix",
                "main_document": True,
                "page_number": 10,
                "evidence_type": "fr_url",
                "fr_document_number": "2024-10001",
                "fr_document_number_raw": "2024-10001",
                "fr_url": "federalregister.gov/documents/2024/01/02/2024-10001/sagebrush",
                "fr_citation": "",
                "fr_date_text": "",
                "fr_date_text_parsed": "",
                "notice_title_snippet": "",
                "evidence_context": "See federalregister.gov/documents/2024/01/02/2024-10001/sagebrush",
                "nearby_noi_phrase": "",
                "nearby_project_title_token_count": 0,
                "evidence_rank": 1,
            }
        ]
    )

    project_matches, candidates, review = build_project_noi_matches(
        projects,
        corpus,
        nepatec_evidence=nepatec_evidence,
    )

    assert project_matches.loc[0, "noi_match_status"] == "review_required"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert candidates.loc[0, "match_reason"] == "nepatec_fr_url_no_title_match_requires_review"
    assert len(review) == 1


def test_process_conflict_with_direct_evidence_goes_to_review():
    """Doc number + title match but process type conflict -> review, not auto-accept."""
    projects = pd.DataFrame([{
        "project_id": "p1",
        "project_title": "Jackalope Wind Energy Project",
        "lead_agency": "Bureau of Land Management",
        "project_state": "Wyoming",
        "process_type": "EIS",       # project is EIS
        "project_energy_type": "Clean",
    }])
    corpus = pd.DataFrame([{
        "fr_document_number": "2024-05618",
        # FR record explicitly says "Environmental Assessment" — process conflict
        "fr_title": "Environmental Assessment for the Jackalope Wind Energy Project",
        "fr_publication_date": "2024-03-18",
        "fr_url": "https://example.test/jackalope",
        "fr_agency_names": '["Bureau of Land Management"]',
        "fr_type": "Notice", "fr_subtype": "", "fr_comments_close_on": "",
        "fr_abstract": "", "fr_query_terms": "direct_fetch",
    }])
    nepatec_evidence = pd.DataFrame([{
        "project_id": "p1", "process_type": "EIS",
        "project_title": "Jackalope Wind Energy Project",
        "document_id": "doc-abc", "file_name": "NOI.pdf",
        "document_title": "Federal Register Notice", "document_type": "FR Notice",
        "main_document": True, "page_number": 1, "evidence_type": "fr_doc_noi",
        "fr_document_number": "2024-05618", "fr_document_number_raw": "2024\u201305618",
        "fr_url": "", "fr_citation": "", "fr_date_text": "", "fr_date_text_parsed": "",
        "notice_title_snippet": "notice of intent",
        "evidence_context": "[FR Doc. 2024\u201305618 Filed 3\u201315\u201324; 8:45 am]",
        "nearby_noi_phrase": "notice of intent",
        "nearby_project_title_token_count": 2, "evidence_rank": 1,
    }])

    project_matches, candidates, review = build_project_noi_matches(
        projects, corpus, nepatec_evidence=nepatec_evidence
    )

    assert project_matches.loc[0, "noi_match_status"] == "review_required"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert candidates.loc[0, "match_reason"] == "nepatec_fr_doc_number_process_conflict_requires_review"


def test_four_token_title_requires_three_overlap():
    """A 4-token project title requires 3 matching tokens to auto-accept."""
    # "TransWest Express Transmission Project" -> tokens: transwest, express, transmission
    # (project is a stopword; only 3 distinctive tokens)
    # "Eagle Mountain Wind Solar Project" -> tokens: eagle, mountain, wind, solar (4)
    # FR title matches only eagle + mountain (2) -> should NOT accept
    projects = pd.DataFrame([{
        "project_id": "p1",
        "project_title": "Eagle Mountain Wind Solar Project",
        "lead_agency": "Bureau of Land Management",
        "project_state": "Nevada",
        "process_type": "EIS",
        "project_energy_type": "Clean",
    }])
    corpus = pd.DataFrame([{
        "fr_document_number": "2024-99999",
        # Only 2 of 4 distinctive tokens match (eagle, mountain) — below threshold of 3
        "fr_title": "Notice of Intent for the Eagle Mountain Pumped Storage Project in Nevada",
        "fr_publication_date": "2024-01-01",
        "fr_url": "https://example.test/eagle",
        "fr_agency_names": '["Bureau of Land Management"]',
        "fr_type": "Notice", "fr_subtype": "", "fr_comments_close_on": "",
        "fr_abstract": "", "fr_query_terms": "direct_fetch",
    }])
    nepatec_evidence = pd.DataFrame([{
        "project_id": "p1", "process_type": "EIS",
        "project_title": "Eagle Mountain Wind Solar Project",
        "document_id": "doc-xyz", "file_name": "NOI.pdf",
        "document_title": "Federal Register Notice", "document_type": "FR Notice",
        "main_document": True, "page_number": 1, "evidence_type": "fr_doc_noi",
        "fr_document_number": "2024-99999", "fr_document_number_raw": "2024-99999",
        "fr_url": "", "fr_citation": "", "fr_date_text": "", "fr_date_text_parsed": "",
        "notice_title_snippet": "notice of intent",
        "evidence_context": "[FR Doc. 2024-99999 Filed 1-01-24; 8:45 am]",
        "nearby_noi_phrase": "notice of intent",
        "nearby_project_title_token_count": 2, "evidence_rank": 1,
    }])

    project_matches, candidates, review = build_project_noi_matches(
        projects, corpus, nepatec_evidence=nepatec_evidence
    )

    # 2 matching tokens < required 3 → review, not accept
    assert project_matches.loc[0, "noi_match_status"] == "review_required"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert candidates.loc[0, "match_reason"] == "nepatec_fr_doc_number_no_title_match_requires_review"


def test_non_noi_fr_title_with_direct_evidence_goes_to_review():
    """FR record with 'Notice of Availability' title must not auto-accept even with strong evidence.

    The NEPATEC doc number may reference a FEIS availability notice rather than an NOI.
    Its publication date would be near end-of-process, not initiation — never populate
    noi_publication_date from such a record.
    """
    projects = pd.DataFrame([{
        "project_id": "p1",
        "project_title": "Jackalope Wind Energy Project",
        "lead_agency": "Bureau of Land Management",
        "project_state": "Wyoming",
        "process_type": "EIS",
        "project_energy_type": "Clean",
    }])
    corpus = pd.DataFrame([{
        "fr_document_number": "2024-99000",
        # Strong title overlap but NOT an NOI — this is a FEIS availability notice
        "fr_title": "Notice of Availability of the Final Environmental Impact Statement for the Jackalope Wind Energy Project",
        "fr_publication_date": "2026-01-15",
        "fr_url": "https://example.test/jackalope-feis",
        "fr_agency_names": '["Bureau of Land Management"]',
        "fr_type": "Notice", "fr_subtype": "", "fr_comments_close_on": "",
        "fr_abstract": "", "fr_query_terms": "direct_fetch",
    }])
    nepatec_evidence = pd.DataFrame([{
        "project_id": "p1", "process_type": "EIS",
        "project_title": "Jackalope Wind Energy Project",
        "document_id": "doc-feis", "file_name": "FEIS_cover.pdf",
        "document_title": "Final EIS", "document_type": "FEIS",
        "main_document": True, "page_number": 1, "evidence_type": "fr_doc_noi",
        "fr_document_number": "2024-99000", "fr_document_number_raw": "2024-99000",
        "fr_url": "", "fr_citation": "", "fr_date_text": "", "fr_date_text_parsed": "",
        "notice_title_snippet": "notice of intent",
        "evidence_context": "[FR Doc. 2024-99000 Filed 1-15-26; 8:45 am]",
        "nearby_noi_phrase": "notice of intent",
        "nearby_project_title_token_count": 2, "evidence_rank": 1,
    }])

    project_matches, candidates, review = build_project_noi_matches(
        projects, corpus, nepatec_evidence=nepatec_evidence
    )

    assert project_matches.loc[0, "noi_match_status"] == "review_required"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert candidates.loc[0, "match_reason"] == "nepatec_fr_doc_number_non_noi_fr_title_requires_review"


def test_title_only_match_never_populates_noi_publication_date():
    """Without NEPATEC evidence, even strong title matches stay in review and never populate noi_publication_date."""
    projects = pd.DataFrame(
        [
            {
                "project_id": "p1",
                "project_title": "Jackalope Wind Energy Project",
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
                "fr_document_number": "2024-05618",
                "fr_title": "Notice of Intent To Prepare an Environmental Impact Statement for the Jackalope Wind Energy Project",
                "fr_publication_date": "2024-03-18",
                "fr_url": "https://example.test/jackalope",
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

    assert project_matches.loc[0, "noi_match_status"] == "review_required"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert len(review) >= 1


def test_ce_nepatec_doc_number_evidence_goes_to_review():
    """CE projects with direct doc number evidence -> review, not auto-accept."""
    projects = pd.DataFrame(
        [
            {
                "project_id": "ce1",
                "project_title": "Blackrock Solar CE Project",
                "lead_agency": "Bureau of Land Management",
                "project_state": "Nevada",
                "process_type": "CE",
                "project_energy_type": "Clean",
            }
        ]
    )
    corpus = pd.DataFrame(
        [
            {
                "fr_document_number": "2024-11111",
                "fr_title": "Notice of Intent To Prepare a Categorical Exclusion for the Blackrock Solar CE Project",
                "fr_publication_date": "2024-05-01",
                "fr_url": "https://example.test/blackrock",
                "fr_agency_names": '["Bureau of Land Management"]',
                "fr_type": "Notice",
                "fr_subtype": "",
                "fr_comments_close_on": "",
                "fr_abstract": "",
                "fr_query_terms": '"Notice of Intent"',
            }
        ]
    )
    nepatec_evidence = pd.DataFrame(
        [
            {
                "project_id": "ce1",
                "process_type": "CE",
                "project_title": "Blackrock Solar CE Project",
                "document_id": "doc-ce1",
                "file_name": "CE Determination.pdf",
                "document_title": "CE Determination",
                "document_type": "CE",
                "main_document": True,
                "page_number": 1,
                "evidence_type": "fr_doc_noi",
                "fr_document_number": "2024-11111",
                "fr_document_number_raw": "2024-11111",
                "fr_url": "",
                "fr_citation": "",
                "fr_date_text": "",
                "fr_date_text_parsed": "",
                "notice_title_snippet": "notice of intent",
                "evidence_context": "[FR Doc. 2024-11111 Filed 4-28-24]",
                "nearby_noi_phrase": "notice of intent",
                "nearby_project_title_token_count": 2,
                "evidence_rank": 1,
            }
        ]
    )

    project_matches, candidates, review = build_project_noi_matches(
        projects, corpus, nepatec_evidence=nepatec_evidence
    )

    assert project_matches.loc[0, "noi_match_status"] == "review_required"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert len(review) >= 1


def test_multiple_nepatec_doc_numbers_go_to_review():
    """When multiple FR records join via NEPATEC evidence, the multi-record case goes to review."""
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
                "fr_title": "Notice of Intent To Prepare an Environmental Impact Statement for the TransWest Express Transmission Project in Wyoming",
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
                "fr_title": "Notice of Intent To Prepare an Environmental Impact Statement for TransWest Express Transmission Project, Wyoming",
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
    # Both doc numbers found in NEPATEC
    nepatec_evidence = pd.DataFrame(
        [
            {
                "project_id": "p1", "process_type": "EIS", "project_title": "TransWest Express Transmission Project",
                "document_id": "doc1", "file_name": "NOI.pdf", "document_title": "NOI", "document_type": "Notice",
                "main_document": True, "page_number": 1, "evidence_type": "fr_doc_noi",
                "fr_document_number": "2011-00001", "fr_document_number_raw": "2011-00001",
                "fr_url": "", "fr_citation": "", "fr_date_text": "", "fr_date_text_parsed": "",
                "notice_title_snippet": "notice of intent", "evidence_context": "[FR Doc. 2011-00001]",
                "nearby_noi_phrase": "notice of intent", "nearby_project_title_token_count": 3, "evidence_rank": 1,
            },
            {
                "project_id": "p1", "process_type": "EIS", "project_title": "TransWest Express Transmission Project",
                "document_id": "doc1", "file_name": "NOI.pdf", "document_title": "NOI", "document_type": "Notice",
                "main_document": True, "page_number": 2, "evidence_type": "fr_doc_noi",
                "fr_document_number": "2011-00002", "fr_document_number_raw": "2011-00002",
                "fr_url": "", "fr_citation": "", "fr_date_text": "", "fr_date_text_parsed": "",
                "notice_title_snippet": "notice of intent", "evidence_context": "[FR Doc. 2011-00002]",
                "nearby_noi_phrase": "notice of intent", "nearby_project_title_token_count": 3, "evidence_rank": 1,
            },
        ]
    )

    project_matches, candidates, review = build_project_noi_matches(
        projects, corpus, nepatec_evidence=nepatec_evidence
    )

    assert len(project_matches) == 1
    assert project_matches.loc[0, "noi_match_status"] == "ambiguous"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])
    assert len(review) == 2


def test_fr_doc_non_noi_not_used_for_accept():
    """FR doc numbers that failed the NOI proximity filter (fr_doc_non_noi) do not produce auto-accepts."""
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
    # Evidence with evidence_type = fr_doc_non_noi (failed proximity filter)
    nepatec_evidence = pd.DataFrame(
        [
            {
                "project_id": "p1", "process_type": "EIS", "project_title": "Cedar Wind Project",
                "document_id": "doc1", "file_name": "appendix.pdf", "document_title": "Appendix",
                "document_type": "Appendix", "main_document": False, "page_number": 5,
                "evidence_type": "fr_doc_non_noi",  # failed proximity filter
                "fr_document_number": "2024-10001", "fr_document_number_raw": "2024-10001",
                "fr_url": "", "fr_citation": "", "fr_date_text": "", "fr_date_text_parsed": "",
                "notice_title_snippet": "", "evidence_context": "See also [FR Doc. 2024-10001]",
                "nearby_noi_phrase": "", "nearby_project_title_token_count": 0, "evidence_rank": 2,
            }
        ]
    )

    project_matches, candidates, review = build_project_noi_matches(
        projects, corpus, nepatec_evidence=nepatec_evidence
    )

    # Should not auto-accept -- fr_doc_non_noi is excluded from the NEPATEC doc number set
    assert project_matches.loc[0, "noi_match_status"] != "accepted"
    assert pd.isna(project_matches.loc[0, "noi_publication_date"])


def test_project_output_includes_nepatec_provenance_columns():
    """PROJECT_OUTPUT_COLUMNS includes the 4 new NEPATEC provenance fields."""
    assert "noi_date_evidence_type" in federal_register.PROJECT_OUTPUT_COLUMNS
    assert "noi_nepatec_evidence_document_id" in federal_register.PROJECT_OUTPUT_COLUMNS
    assert "noi_nepatec_evidence_file_name" in federal_register.PROJECT_OUTPUT_COLUMNS
    assert "noi_nepatec_evidence_page_number" in federal_register.PROJECT_OUTPUT_COLUMNS


def test_candidate_output_includes_nepatec_evidence_column():
    """CANDIDATE_OUTPUT_COLUMNS includes nepatec_fr_document_number_evidence."""
    assert "nepatec_fr_document_number_evidence" in federal_register.CANDIDATE_OUTPUT_COLUMNS


def test_focused_manual_review_exports_are_written(tmp_path):
    projects = pd.DataFrame(
        [
            {
                "project_id": "amb1",
                "process_type": "EIS",
                "project_energy_type": "Clean",
                "project_department": "Department of Energy",
                "lead_agency": "Department of Energy",
                "project_state": "Wyoming",
                "project_sponsor": "Example Sponsor",
            },
            {
                "project_id": "low1",
                "process_type": "EA",
                "project_energy_type": "Other",
                "project_department": "Department of Interior",
                "lead_agency": "Bureau of Land Management",
                "project_state": "Nevada",
                "project_sponsor": "",
            },
            {
                "project_id": "noalow1",
                "process_type": "EIS",
                "project_energy_type": "Clean",
                "project_department": "Department of Interior",
                "lead_agency": "Bureau of Land Management",
                "project_state": "Wyoming",
                "project_sponsor": "",
            },
        ]
    )
    project_matches = pd.DataFrame(
        [
            {
                "project_id": "amb1",
                "project_title": "Ambiguous Project",
                "noi_match_status": "ambiguous",
                "noi_publication_date": None,
                "noi_document_number": "2020-00001",
                "noi_project_title": "Ambiguous NOI",
                "noi_match_score": 65,
                "noi_match_reason": "multiple_high_confidence_candidates",
                "noi_candidate_count": 2,
                "noi_high_confidence_candidate_count": 2,
                "noi_title_overlap_count": 3,
                "noi_title_overlap_tokens": "ambiguous, project, title",
            },
            {
                "project_id": "low1",
                "project_title": "Low Overlap Project",
                "noi_match_status": "accepted",
                "noi_publication_date": "2020-01-02",
                "noi_document_number": "2020-00002",
                "noi_project_title": "Low Title Match",
                "noi_match_score": 25,
                "noi_match_reason": "nepatec_fr_doc_number_with_title_match",
                "noi_candidate_count": 1,
                "noi_high_confidence_candidate_count": 1,
                "noi_title_overlap_count": 1,
                "noi_title_overlap_tokens": "project",
            },
            {
                "project_id": "ok1",
                "project_title": "High Overlap Project",
                "noi_match_status": "accepted",
                "noi_publication_date": "2020-01-03",
                "noi_document_number": "2020-00003",
                "noi_project_title": "High Overlap Project",
                "noi_match_score": 80,
                "noi_match_reason": "nepatec_fr_doc_number_with_title_match",
                "noi_candidate_count": 1,
                "noi_high_confidence_candidate_count": 1,
                "noi_title_overlap_count": 3,
                "noi_title_overlap_tokens": "high, overlap, project",
            },
            {
                "project_id": "noalow1",
                "project_title": "Low Overlap NOA Project",
                "noi_match_status": "unmatched",
                "noi_publication_date": None,
                "noa_match_status": "accepted",
                "noa_availability_date": "2020-01-04",
                "noa_document_number": "2020-00004",
                "noa_fr_title": "Low NOA Title Match",
                "noa_match_score": 25,
                "noa_match_reason": "nepatec_fr_doc_noa_with_title_match",
                "noa_title_overlap_count": 1,
                "noa_title_overlap_tokens": "project",
            },
        ]
    )
    review = pd.DataFrame(
        [
            {
                "project_id": "amb1",
                "project_title": "Ambiguous Project",
                "process_type": "EIS",
                "project_energy_type": "Clean",
                "fr_document_number": "2020-00001",
                "fr_publication_date": "2020-01-01",
                "fr_url": "https://example.test/1",
                "fr_title": "Ambiguous NOI",
                "match_confidence": "high",
                "match_score": 65,
                "match_reason": "nepatec_fr_doc_number_with_title_match",
                "candidate_rank": 1,
                "title_overlap_count": 3,
                "title_overlap_tokens": "ambiguous, project, title",
                "nepatec_fr_document_number_evidence": True,
            },
            {
                "project_id": "other",
                "project_title": "Other Project",
                "process_type": "EIS",
                "project_energy_type": "Clean",
                "fr_document_number": "2020-99999",
                "fr_publication_date": "2020-01-04",
                "fr_url": "https://example.test/other",
                "fr_title": "Other NOI",
                "match_confidence": "medium",
                "match_score": 30,
                "match_reason": "manual_review",
                "candidate_rank": 1,
                "title_overlap_count": 2,
                "title_overlap_tokens": "other, project",
                "nepatec_fr_document_number_evidence": False,
            },
        ]
    )

    counts = federal_register.write_focused_manual_review_exports(
        project_matches,
        review,
        projects,
        output_dir=tmp_path,
    )

    ambiguous_candidates_path = tmp_path / federal_register.DEFAULT_AMBIGUOUS_CANDIDATES_OUTPUT.name

    assert counts[str(ambiguous_candidates_path)] == 1
    assert pd.read_csv(ambiguous_candidates_path).loc[0, "project_id"] == "amb1"
    # ambiguous_projects CSV is no longer written
    assert not (tmp_path / "manual_review_ambiguous_projects.csv").exists()
    assert not (tmp_path / "noi_manual_review_accepted_low_title_overlap.csv").exists()
    assert not (tmp_path / "noa_manual_review_accepted_low_title_overlap.csv").exists()


def test_candidate_review_export_adds_project_context():
    projects = pd.DataFrame([{
        "project_id": "p1",
        "project_title": "Ridgeline Wind Energy Project",
        "process_type": "EIS",
        "project_energy_type": "Clean",
    }])
    review = pd.DataFrame([{
        "project_id": "p1",
        "fr_document_number": "2025-01234",
        "fr_title": "Notice of Availability of the Final Environmental Impact Statement for the Ridgeline Wind Energy Project",
        "fr_publication_date": "2025-06-15",
        "fr_url": "https://example.test/ridgeline-feis",
        "match_score": 42,
        "match_confidence": "medium",
        "match_reason": "noa_doc_number_insufficient_title_overlap",
        "candidate_rank": 1,
        "title_overlap_count": 1,
        "title_overlap_tokens": "ridgeline",
        "process_conflict": False,
        "nepatec_fr_document_number_evidence": True,
    }])

    export = federal_register._candidate_review_export(review, projects)

    assert list(export.columns) == federal_register.FOCUSED_CANDIDATE_REVIEW_COLUMNS
    assert export.loc[0, "project_title"] == "Ridgeline Wind Energy Project"
    assert export.loc[0, "process_type"] == "EIS"
    assert export.loc[0, "project_energy_type"] == "Clean"


# ── NOA (Notice of Availability) tests ──────────────────────────────────────

def test_is_noa_title_recognizes_feis_and_fonsi():
    """_is_noa_title() returns True for FEIS/FONSI/Final EA titles, False for NOI titles."""
    # Standard NOA with "availability" phrasing
    assert _is_noa_title("Notice of Availability of the Final Environmental Impact Statement for the Cedar Wind Project")
    assert _is_noa_title("Finding of No Significant Impact for the Cedar Solar Project Environmental Assessment")
    assert _is_noa_title("Notice of Availability of the Final Environmental Assessment for the Ridgeline Project")
    # FEIS title without "availability" — accepted unconditionally (FR context = NOA notice)
    assert _is_noa_title("Final Environmental Impact Statement for the Cedar Wind Energy Project")
    assert _is_noa_title("Final EIS for the Cedar Wind Energy Project")
    # FSEIS (Final Supplemental EIS) — also accepted unconditionally
    assert _is_noa_title("Final Supplemental Environmental Impact Statement for the Cedar Wind Project")
    assert _is_noa_title("Notice of Availability of the Final Supplemental EIS for the Ridgeline Project")
    # NOI / Draft titles must not match
    assert not _is_noa_title("Notice of Intent To Prepare an Environmental Impact Statement")
    assert not _is_noa_title("Notice of Preparation of an EIS")
    assert not _is_noa_title("Notice of Scoping for the Cedar Wind Project")
    assert not _is_noa_title("Draft Environmental Impact Statement for Cedar Wind Project")


def test_noa_proximity_check_returns_phrase_for_feis_text():
    """_noa_proximity_check() finds a NOA proximity phrase near a doc number reference."""
    text = (
        "The agency published the Final Environmental Impact Statement for the project. "
        "[FR Doc. 2024-12345 Filed 4-1-24] "
        "Comments are due by May 1."
    )
    pos = text.index("[FR Doc.")
    result = _noa_proximity_check(text, pos)
    assert result is not None
    assert "final environmental impact statement" in result


def test_noa_direct_evidence_accepts_eis_project():
    """EIS project + FEIS NOA title + fr_doc_noa evidence → noa_availability_date populated."""
    projects = pd.DataFrame([{
        "project_id": "p1",
        "project_title": "Ridgeline Wind Energy Project",
        "lead_agency": "Bureau of Land Management",
        "project_state": "Wyoming",
        "process_type": "EIS",
        "project_energy_type": "Clean",
    }])
    noa_corpus = pd.DataFrame([{
        "fr_document_number": "2025-01234",
        "fr_title": "Notice of Availability of the Final Environmental Impact Statement for the Ridgeline Wind Energy Project",
        "fr_publication_date": "2025-06-15",
        "fr_url": "https://example.test/ridgeline-feis",
        "fr_agency_names": '["Bureau of Land Management"]',
        "fr_type": "Notice",
        "fr_subtype": "",
        "fr_comments_close_on": "",
        "fr_abstract": "",
        "fr_query_terms": "direct_fetch",
    }])
    nepatec_evidence = pd.DataFrame([{
        "project_id": "p1",
        "process_type": "EIS",
        "project_title": "Ridgeline Wind Energy Project",
        "document_id": "doc-xyz",
        "file_name": "FEIS_Notice.pdf",
        "document_title": "FEIS Notice",
        "document_type": "FR Notice",
        "main_document": True,
        "page_number": 3,
        "evidence_type": "fr_doc_noa",
        "fr_document_number": "2025-01234",
        "fr_document_number_raw": "2025-01234",
        "fr_url": "",
        "fr_citation": "",
        "fr_date_text": "",
        "fr_date_text_parsed": "",
        "notice_title_snippet": "final environmental impact statement",
        "evidence_context": "[FR Doc. 2025-01234]",
        "nearby_noi_phrase": "final environmental impact statement",
        "nearby_project_title_token_count": 2,
        "evidence_rank": 1,
    }])

    noa_matches, candidates, review = build_project_noa_matches(
        projects, noa_corpus, nepatec_evidence=nepatec_evidence
    )

    assert noa_matches.loc[0, "noa_match_status"] == "accepted"
    assert noa_matches.loc[0, "noa_availability_date"] == "2025-06-15"
    assert noa_matches.loc[0, "noa_nepatec_evidence_file_name"] == "FEIS_Notice.pdf"


def test_fonsi_direct_evidence_accepts_ea_project():
    """EA project + FONSI title + fr_doc_noa evidence → noa_availability_date populated."""
    projects = pd.DataFrame([{
        "project_id": "p2",
        "project_title": "Desert Solar Farm Project",
        "lead_agency": "Bureau of Land Management",
        "project_state": "Nevada",
        "process_type": "EA",
        "project_energy_type": "Clean",
    }])
    noa_corpus = pd.DataFrame([{
        "fr_document_number": "2025-05678",
        "fr_title": "Finding of No Significant Impact for the Desert Solar Farm Project",
        "fr_publication_date": "2025-08-20",
        "fr_url": "https://example.test/desert-solar-fonsi",
        "fr_agency_names": '["Bureau of Land Management"]',
        "fr_type": "Notice",
        "fr_subtype": "",
        "fr_comments_close_on": "",
        "fr_abstract": "",
        "fr_query_terms": "direct_fetch",
    }])
    nepatec_evidence = pd.DataFrame([{
        "project_id": "p2",
        "process_type": "EA",
        "project_title": "Desert Solar Farm Project",
        "document_id": "doc-fonsi",
        "file_name": "FONSI_Notice.pdf",
        "document_title": "FONSI",
        "document_type": "FR Notice",
        "main_document": True,
        "page_number": 1,
        "evidence_type": "fr_doc_noa",
        "fr_document_number": "2025-05678",
        "fr_document_number_raw": "2025-05678",
        "fr_url": "",
        "fr_citation": "",
        "fr_date_text": "",
        "fr_date_text_parsed": "",
        "notice_title_snippet": "finding of no significant impact",
        "evidence_context": "[FR Doc. 2025-05678]",
        "nearby_noi_phrase": "finding of no significant impact",
        "nearby_project_title_token_count": 2,
        "evidence_rank": 1,
    }])

    noa_matches, candidates, review = build_project_noa_matches(
        projects, noa_corpus, nepatec_evidence=nepatec_evidence
    )

    assert noa_matches.loc[0, "noa_match_status"] == "accepted"
    assert noa_matches.loc[0, "noa_availability_date"] == "2025-08-20"


def test_noa_process_mismatch_goes_to_review():
    """EA project + FEIS NOA title (not FONSI) → review_required (process mismatch)."""
    projects = pd.DataFrame([{
        "project_id": "p3",
        "project_title": "Highline Solar Project",
        "lead_agency": "Bureau of Land Management",
        "project_state": "Utah",
        "process_type": "EA",  # EA project
        "project_energy_type": "Clean",
    }])
    # FR record is a FEIS NOA (EIS process) — mismatched for an EA project
    noa_corpus = pd.DataFrame([{
        "fr_document_number": "2025-09999",
        "fr_title": "Notice of Availability of the Final Environmental Impact Statement for the Highline Solar Project",
        "fr_publication_date": "2025-09-01",
        "fr_url": "https://example.test/highline-feis",
        "fr_agency_names": '["Bureau of Land Management"]',
        "fr_type": "Notice",
        "fr_subtype": "",
        "fr_comments_close_on": "",
        "fr_abstract": "",
        "fr_query_terms": "direct_fetch",
    }])
    nepatec_evidence = pd.DataFrame([{
        "project_id": "p3",
        "process_type": "EA",
        "project_title": "Highline Solar Project",
        "document_id": "doc-mismatch",
        "file_name": "FEIS_Notice.pdf",
        "document_title": "FEIS Notice",
        "document_type": "FR Notice",
        "main_document": True,
        "page_number": 2,
        "evidence_type": "fr_doc_noa",
        "fr_document_number": "2025-09999",
        "fr_document_number_raw": "2025-09999",
        "fr_url": "",
        "fr_citation": "",
        "fr_date_text": "",
        "fr_date_text_parsed": "",
        "notice_title_snippet": "final environmental impact statement",
        "evidence_context": "[FR Doc. 2025-09999]",
        "nearby_noi_phrase": "final environmental impact statement",
        "nearby_project_title_token_count": 2,
        "evidence_rank": 1,
    }])

    noa_matches, candidates, review = build_project_noa_matches(
        projects, noa_corpus, nepatec_evidence=nepatec_evidence
    )

    assert noa_matches.loc[0, "noa_match_status"] == "review_required"
    assert noa_matches.loc[0, "noa_availability_date"] is None
    assert "mismatch" in noa_matches.loc[0, "noa_match_reason"]


def test_refresh_merges_noa_columns_without_suffixes(tmp_path, monkeypatch):
    """Combined NOI+NOA refresh should persist clean noa_* columns, not _x/_y suffixes."""
    projects = pd.DataFrame([{
        "project_id": "p1",
        "project_title": "Ridgeline Wind Energy Project",
        "lead_agency": "Bureau of Land Management",
        "project_state": "Wyoming",
        "process_type": "EIS",
        "project_energy_type": "Clean",
    }])
    evidence = pd.DataFrame([
        {
            "project_id": "p1",
            "process_type": "EIS",
            "project_title": "Ridgeline Wind Energy Project",
            "document_id": "doc-noi",
            "file_name": "NOI.pdf",
            "document_title": "NOI",
            "document_type": "FR Notice",
            "main_document": True,
            "page_number": 1,
            "evidence_type": "fr_doc_noi",
            "fr_document_number": "2025-00001",
            "fr_document_number_raw": "2025-00001",
            "fr_url": "",
            "fr_citation": "",
            "fr_date_text": "",
            "fr_date_text_parsed": "",
            "notice_title_snippet": "notice of intent",
            "evidence_context": "[FR Doc. 2025-00001]",
            "nearby_noi_phrase": "notice of intent",
            "nearby_project_title_token_count": 3,
            "evidence_rank": 1,
        },
        {
            "project_id": "p1",
            "process_type": "EIS",
            "project_title": "Ridgeline Wind Energy Project",
            "document_id": "doc-noa",
            "file_name": "FEIS.pdf",
            "document_title": "FEIS",
            "document_type": "FR Notice",
            "main_document": True,
            "page_number": 2,
            "evidence_type": "fr_doc_noa",
            "fr_document_number": "2025-00002",
            "fr_document_number_raw": "2025-00002",
            "fr_url": "",
            "fr_citation": "",
            "fr_date_text": "",
            "fr_date_text_parsed": "",
            "notice_title_snippet": "final environmental impact statement",
            "evidence_context": "[FR Doc. 2025-00002]",
            "nearby_noi_phrase": "final environmental impact statement",
            "nearby_project_title_token_count": 3,
            "evidence_rank": 1,
        },
    ])
    fetched = pd.DataFrame([
        {
            "fr_document_number": "2025-00001",
            "fr_title": "Notice of Intent To Prepare an Environmental Impact Statement for the Ridgeline Wind Energy Project",
            "fr_publication_date": "2025-01-15",
            "fr_url": "https://example.test/noi",
            "fr_pdf_url": "",
            "fr_raw_text_url": "",
            "fr_agency_names": '["Bureau of Land Management"]',
            "fr_agencies": "",
            "fr_type": "Notice",
            "fr_subtype": "",
            "fr_comments_close_on": "",
            "fr_abstract": "",
            "fr_excerpts": "",
            "fr_scoping_meeting_dates": "",
            "fr_query_terms": "direct_fetch",
            "fr_query_count": 1,
            "fr_fetch_run_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "fr_document_number": "2025-00002",
            "fr_title": "Notice of Availability of the Final Environmental Impact Statement for the Ridgeline Wind Energy Project",
            "fr_publication_date": "2025-06-15",
            "fr_url": "https://example.test/noa",
            "fr_pdf_url": "",
            "fr_raw_text_url": "",
            "fr_agency_names": '["Bureau of Land Management"]',
            "fr_agencies": "",
            "fr_type": "Notice",
            "fr_subtype": "",
            "fr_comments_close_on": "",
            "fr_abstract": "",
            "fr_excerpts": "",
            "fr_scoping_meeting_dates": "",
            "fr_query_terms": "direct_fetch",
            "fr_query_count": 1,
            "fr_fetch_run_at": "2026-01-01T00:00:00+00:00",
        },
    ])

    def fake_extract(*args, **kwargs):
        return evidence

    def fake_fetch(doc_numbers, **kwargs):
        assert set(doc_numbers) == {"2025-00001", "2025-00002"}
        return fetched

    monkeypatch.setattr(federal_register, "extract_nepatec_federal_register_evidence", fake_extract)
    monkeypatch.setattr(federal_register, "fetch_documents_by_doc_numbers", fake_fetch)

    project_matches = federal_register.refresh_federal_register_noi(
        projects,
        analysis_dir=tmp_path,
        throttle_seconds=0,
        show_progress=False,
        rescan_nepatec_evidence=True,
    )

    suffixed_noa_columns = [
        col for col in project_matches.columns
        if col.startswith("noa_") and (col.endswith("_x") or col.endswith("_y"))
    ]
    assert suffixed_noa_columns == []
    assert project_matches.loc[0, "noi_publication_date"] == "2025-01-15"
    assert project_matches.loc[0, "noa_availability_date"] == "2025-06-15"
    assert project_matches.loc[0, "noa_match_status"] == "accepted"

    persisted = pd.read_parquet(tmp_path / "federal_register" / "federal_register.parquet")
    assert "noa_availability_date" in persisted.columns
    assert "noa_availability_date_y" not in persisted.columns
    assert persisted.loc[0, "noa_availability_date"] == "2025-06-15"


# ── NOA title search (Option 1 supplemental path) ────────────────────────────

def test_build_noa_title_search_term_requires_three_distinctive_tokens():
    """Returns None when title has < 3 distinctive tokens; returns phrase otherwise."""
    # "Solar Project" → only "solar" is distinctive ("project" is a stopword)
    assert _build_noa_title_search_term("Solar Project") is None

    # "Cedar Wind Project" → cedar, wind (project filtered) = 2 tokens
    assert _build_noa_title_search_term("Cedar Wind Project") is None

    # "Cedar Ridge Wind Farm" → cedar, ridge, wind, farm = 4 tokens → phrase returned
    phrase = _build_noa_title_search_term("Cedar Ridge Wind Farm")
    assert phrase is not None
    assert len(phrase) > 0

    # Prefix-stripped title "Construction and Operation of a Cedar Ridge Wind Farm"
    # → after stripping, same as above → phrase returned
    phrase2 = _build_noa_title_search_term("Construction and Operation of a Cedar Ridge Wind Farm")
    assert phrase2 is not None


_UNMATCHED_NOA_ROW = {
    "project_id": "p1",
    "noa_availability_date": None,
    "noa_document_number": None,
    "noa_url": None,
    "noa_fr_title": None,
    "noa_match_status": "unmatched",
    "noa_match_reason": "unmatched",
    "noa_match_score": None,
    "noa_title_overlap_count": None,
    "noa_title_overlap_tokens": None,
    "noa_date_evidence_type": None,
    "noa_nepatec_evidence_document_id": None,
    "noa_nepatec_evidence_file_name": None,
    "noa_nepatec_evidence_page_number": None,
}

_FEIS_API_RESULT = {
    "document_number": "2023-10001",
    "title": "Notice of Availability of the Final Environmental Impact Statement for the Ridgeline Wind Center Project",
    "publication_date": "2023-06-01",
    "html_url": "https://example.test/feis",
    "pdf_url": "",
    "raw_text_url": "",
    "agency_names": [],
    "agencies": [],
    "type": "Notice",
    "subtype": "",
    "comments_close_on": "",
    "abstract": "",
    "excerpts": [],
}


def test_supplement_noa_accepts_feis_for_unmatched_eis(monkeypatch):
    """EIS project with noi_publication_date and ≥3 tokens → FEIS from title search → accepted."""
    # "Ridgeline Wind Center Project" → ridgeline, wind, center (project filtered) = 3 tokens
    project_title = "Ridgeline Wind Center Project"

    noa_matches = pd.DataFrame([dict(_UNMATCHED_NOA_ROW)])
    projects = pd.DataFrame([{
        "project_id": "p1",
        "project_title": project_title,
        "process_type": "EIS",
        "project_energy_type": "Clean",
    }])
    project_matches = pd.DataFrame([{
        "project_id": "p1",
        "project_title": project_title,
        "noi_publication_date": "2022-03-15",
    }])

    monkeypatch.setattr(
        federal_register,
        "_search_noa_by_title_cached",
        lambda *args, **kwargs: [_FEIS_API_RESULT],
    )

    result = _supplement_noa_by_title_search(
        noa_matches, projects, project_matches,
        throttle_seconds=0, cache={}, show_progress=False,
    )

    assert result.loc[0, "noa_availability_date"] == "2023-06-01"
    assert result.loc[0, "noa_match_status"] == "accepted"
    assert result.loc[0, "noa_match_reason"] == "fr_title_search_noi_anchored"
    assert result.loc[0, "noa_date_evidence_type"] == "fr_title_search_noi_anchored"
    assert result.loc[0, "noa_nepatec_evidence_document_id"] is None


def test_supplement_noa_skips_ea_projects_and_missing_noi_date(monkeypatch):
    """EA projects are never attempted. EIS with no noi_publication_date is also skipped."""
    called = []
    monkeypatch.setattr(
        federal_register,
        "_search_noa_by_title_cached",
        lambda *args, **kwargs: called.append(1) or [],
    )

    # EA project — should be skipped regardless of noi_publication_date
    ea_noa = pd.DataFrame([{**_UNMATCHED_NOA_ROW, "project_id": "ea1"}])
    ea_projects = pd.DataFrame([{
        "project_id": "ea1",
        "project_title": "Ridgeline Wind Center Project",
        "process_type": "EA",
        "project_energy_type": "Clean",
    }])
    ea_pm = pd.DataFrame([{"project_id": "ea1", "noi_publication_date": "2022-01-01"}])

    result_ea = _supplement_noa_by_title_search(
        ea_noa, ea_projects, ea_pm,
        throttle_seconds=0, cache={}, show_progress=False,
    )
    assert not called
    assert pd.isna(result_ea.loc[0, "noa_availability_date"])

    # EIS project with no noi_publication_date — should also be skipped
    eis_no_noi = pd.DataFrame([{**_UNMATCHED_NOA_ROW, "project_id": "eis1"}])
    eis_projects = pd.DataFrame([{
        "project_id": "eis1",
        "project_title": "Ridgeline Wind Center Project",
        "process_type": "EIS",
        "project_energy_type": "Clean",
    }])
    eis_pm_no_noi = pd.DataFrame([{"project_id": "eis1", "noi_publication_date": None}])

    result_eis = _supplement_noa_by_title_search(
        eis_no_noi, eis_projects, eis_pm_no_noi,
        throttle_seconds=0, cache={}, show_progress=False,
    )
    assert not called
    assert pd.isna(result_eis.loc[0, "noa_availability_date"])
