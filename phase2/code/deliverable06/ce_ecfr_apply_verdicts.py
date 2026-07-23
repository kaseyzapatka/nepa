"""D6 A1/#37 — apply the MANUAL eCFR coverage adjudication to candidate_ce_coverage.parquet.

Companion to ce_ecfr_verify.py (which builds the empty scaffold + fetches the current eCFR text,
$0). This script carries the reviewer's per-row verdict table and writes it back into
candidate_ce_coverage.parquet + re-renders the worksheet. NO LLM / NO API — the verdicts are a
manual text-adjudication by the reviewer named in ADJUDICATOR.

Verdict vocabulary (from ce_ecfr_verify.COVERAGE_VALUES):
    covers / partially_covers / does_not_cover / unclear

Citation-quality cap (KNOWN CONTEXT): only the 5 eCFR-current URLs that resolved to text
(DOE 10 CFR 1021, USDA 7 CFR 1b.4, FERC 18 CFR 380.4, FRA 23 CFR 771.116, FHWA 23 CFR 771.117)
can earn a clean "covers". agency-doc CEs (DOI/DoD/NIST/FirstNet procedure PDFs, not in the eCFR)
and legacy cgi-bin eCFR URLs (NRC 10 CFR 51.22, TVA 18 CFR 1318) are text-unverifiable and are
capped at partially_covers / unclear / does_not_cover even when the CE description matches. The
one eCFR-current URL that failed to fetch (FTA 23 CFR 771.118, trailing-space bug) is treated as
unclear.

NOTE (updated 2026-07-22): once filled, this file IS wired into the verdicts. 07_classify_and_rank.py
`apply_coverage_gate()` reads it, computes the cell-best coverage per adopt/expand cell, and gates:
does_not_cover flips the cell to develop; covers/partially_covers/unclear set verdict_confidence
(verified/partial) or needs_review. qa_deliverable06.py checks post-gate invariants (baseline
reconciliation adopt+flips==22, verified-has-eCFR-covers, etc.), not a hard adopt==22.

USAGE:  CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable06/ce_ecfr_apply_verdicts.py
"""
from __future__ import annotations

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

from common import D6_ANALYSIS_DIR, utc_now, write_parquet
from ce_ecfr_verify import COVERAGE_OUT, COVERAGE_VALUES, WORKSHEET

ADJUDICATOR = "opus-agent-manual-2026-07-22"

# Per-cell verdict table, in retrieval_rank order (rank 1..5).
# Each row: (coverage_verdict, bound_confirmed, reviewer_notes)
V: dict[str, list[tuple[str, str, str]]] = {
    "Biomass__research_or_demonstration": [
        ("does_not_cover", "na", "BOEM offshore oil/gas APD (agency_doc, not in eCFR); off-scope for biomass R&D."),
        ("covers", "yes", "DOE B5.20 small biomass power plants (<10 MW); text verified in current 10 CFR 1021 App B; covers a biomass demo build within the 10 MW cap."),
        ("does_not_cover", "na", "BOEM production commingling (agency_doc); off-scope."),
        ("partially_covers", "no", "USDA small-scale rural-development CE (text verified 7 CFR 1b.4) but generic financial-assistance scope, not biomass-specific; acres=80 cap not tied to this action."),
        ("does_not_cover", "na", "BOEM production measurement (agency_doc); off-scope."),
    ],
    "Geothermal__new_build": [
        ("partially_covers", "no", "BLM geophysical-exploration NOI (agency_doc/DOI procedures, not eCFR-verifiable); substantively near geothermal exploration, no new road."),
        ("does_not_cover", "na", "BLM mineral lease transfers (agency_doc); administrative, not a build."),
        ("does_not_cover", "na", "BLM unitization/agreements (agency_doc); administrative."),
        ("partially_covers", "no", "BLM geothermal drilling-permit/confirmation CE (agency_doc, acres=20 unverifiable); strong substantive match but not eCFR-confirmable."),
        ("does_not_cover", "na", "BLM suspensions of operations (agency_doc); administrative."),
    ],
    "Geothermal__other": [
        ("partially_covers", "no", "BLM geophysical-exploration NOI (agency_doc); partial geothermal match, not eCFR-verifiable."),
        ("unclear", "na", "NPS wells/comfort stations (agency_doc); tangential."),
        ("does_not_cover", "na", "BLM mineral lease transfers (agency_doc); administrative."),
        ("covers", "na", "DOE B5.19 ground-source heat pumps; text verified 10 CFR 1021 App B; covers the GSHP slice of geothermal-other."),
        ("partially_covers", "no", "BLM geothermal drilling-permit CE (agency_doc, acres=20 unverifiable); partial."),
    ],
    "Geothermal__research_or_demonstration": [
        ("partially_covers", "no", "BLM geophysical-exploration NOI (agency_doc); strong substantive match to geothermal R&D exploration but not eCFR-verifiable."),
        ("partially_covers", "no", "USGS exploratory well drilling, no access road / no significant disturbance (agency_doc); substantive."),
        ("unclear", "na", "NPS underground utilities in disturbed areas (agency_doc); tangential."),
        ("partially_covers", "no", "USGS well logging / aquifer testing (agency_doc); matches assessment-type R&D."),
        ("does_not_cover", "na", "BLM mineral lease transfers (agency_doc); administrative."),
    ],
    "Hydropower__assessment": [
        ("partially_covers", "no", "BOR data-collection / test-excavation studies (agency_doc); strong substantive assessment match, localized impacts."),
        ("partially_covers", "no", "USGS test/exploration drilling & downhole testing (agency_doc); substantive."),
        ("unclear", "na", "BOR minor safety-of-dams construction (agency_doc); construction, not assessment."),
        ("partially_covers", "no", "USGS exploratory groundwater well drilling (agency_doc); partial."),
        ("partially_covers", "no", "USGS hydrologic/water-quality monitoring structures (agency_doc); substantive assessment match."),
    ],
    "Hydropower__new_build": [
        ("does_not_cover", "na", "BOEM research/monitoring devices (agency_doc); off-scope for hydropower new build."),
        ("does_not_cover", "na", "BOEM prelease planning (agency_doc); off-scope."),
        ("does_not_cover", "na", "BOEM Sundry Notices on wells (agency_doc); off-scope."),
        ("does_not_cover", "na", "BOEM production measurement (agency_doc); off-scope."),
        ("does_not_cover", "na", "BOEM production commingling (agency_doc); off-scope."),
    ],
    "Hydropower__research_or_demonstration": [
        ("partially_covers", "no", "BOEM research/monitoring-device install (agency_doc); only a generic R&D-device match, offshore oil/gas source."),
        ("does_not_cover", "na", "BOEM off-lease storage (agency_doc); off-scope."),
        ("does_not_cover", "na", "BOEM production measurement (agency_doc); off-scope."),
        ("does_not_cover", "na", "BOEM lease consolidation (agency_doc); off-scope."),
        ("does_not_cover", "na", "BOEM production commingling (agency_doc); off-scope."),
    ],
    "Hydropower__upgrade": [
        ("covers", "na", "USDA increase-freeboard of an existing NRCS dam; text verified 7 CFR 1b.4; covers dam upgrade for NRCS-standard dams."),
        ("partially_covers", "no", "BOR minor safety-of-dams construction (agency_doc); substantive but not eCFR-verifiable."),
        ("unclear", "na", "NPS underground utilities (agency_doc); tangential."),
        ("covers", "na", "USDA repair/improve existing emergency spillways to safety standards; text verified 7 CFR 1b.4; covers dam upgrade."),
        ("partially_covers", "na", "FERC 380.4(a)(19) water-power project utility lines; text verified 18 CFR 380.4 but scope is line authorization, not generation upgrade."),
    ],
    "Nuclear__assessment": [
        ("does_not_cover", "na", "BOEM offshore lease exploration plan (agency_doc); off-scope."),
        ("covers", "na", "FRA localized geotechnical investigations / test bores; text verified 23 CFR 771.116; covers nuclear site assessment/geotech."),
        ("unclear", "na", "FTA identical geotech CE at 23 CFR 771.118 but eCFR fetch failed (0 chars, trailing-space URL); substance matches verified FRA/FHWA siblings, text unconfirmed."),
        ("covers", "na", "FHWA localized geotechnical investigation / test bores; text verified 23 CFR 771.117; covers site assessment/geotech."),
        ("partially_covers", "no", "BIA geologic mapping/reconnaissance/surface-sampling permits (agency_doc); substantive assessment match, not eCFR-verifiable."),
    ],
    "Nuclear__manufacturing": [
        ("partially_covers", "no", "DA R&D/production/manufacturing at existing enclosed facilities (agency_doc, DoD PDF); substantive match, not eCFR-verifiable."),
        ("does_not_cover", "no", "NRC import-license CE (10 CFR 51.22, legacy cgi-bin URL, unfetched); licensing import, not manufacturing."),
        ("partially_covers", "no", "NIST install/operate manufacturing machinery (agency_doc); substantive."),
        ("does_not_cover", "no", "NRC authorize resume-operation amendment (legacy URL, unfetched); off-scope."),
        ("does_not_cover", "no", "NRC license-amendment safeguards (legacy URL, unfetched); off-scope."),
    ],
    "Nuclear__other": [
        ("partially_covers", "no", "NRC administrative/procedural license amendments (legacy 10 CFR 51.22 URL, unfetched); partial 'other' match, text unconfirmable."),
        ("partially_covers", "no", "NRC reactor-license amendment (legacy URL, unfetched); partial."),
        ("partially_covers", "no", "NRC amendment on safeguards (legacy URL, unfetched); partial."),
        ("covers", "na", "DOE B1.10 onsite storage of activated material at existing facility; text verified 10 CFR 1021 App B; genuine nuclear CE."),
        ("covers", "na", "DOE B2.6 recovery of radioactive sealed sources; text verified 10 CFR 1021 App B; genuine nuclear CE."),
    ],
    "Nuclear__upgrade": [
        ("partially_covers", "no", "NRC admin/procedural amendments (legacy URL, unfetched); only loosely 'upgrade', text unconfirmable."),
        ("does_not_cover", "no", "NRC decommissioning of limited sites (legacy URL); decommissioning, not upgrade."),
        ("does_not_cover", "no", "NRC import licenses (legacy URL); off-scope."),
        ("unclear", "no", "NRC certificate of compliance, gaseous-diffusion facilities (legacy URL, unfetched); ambiguous fit."),
        ("does_not_cover", "no", "NRC scholarship grants (legacy URL); off-scope."),
    ],
    "Other Clean__land_or_row_authorization": [
        ("partially_covers", "no", "BLM short ROW grant for utility service (agency_doc); substantive land/ROW match, not eCFR-verifiable."),
        ("does_not_cover", "no", "BLM wildfire/flood emergency repair (agency_doc, acres=4200); off-scope."),
        ("partially_covers", "no", "BLM short-term ROW / land-use authorizations (agency_doc); substantive."),
        ("partially_covers", "no", "BLM amendments to existing ROW (agency_doc); substantive."),
        ("unclear", "na", "BLM routine signs/culverts on roads (agency_doc); tangential."),
    ],
    "Other Clean__maintenance": [
        ("covers", "na", "USDA revegetation of disturbed sites (herbaceous/woody planting); text verified 7 CFR 1b.4; covers maintenance/restoration."),
        ("partially_covers", "no", "NPS native-species restoration (agency_doc); substantive but not eCFR-verifiable."),
        ("partially_covers", "no", "NPS stabilization by native planting (agency_doc); substantive."),
        ("covers", "na", "USDA minor short-term special uses of NFS lands; text verified 7 CFR 1b.4; covers minor maintenance uses."),
        ("partially_covers", "no", "TVA invasive-plant management <=125 acres (legacy 18 CFR 1318 URL, unfetched); substantive veg-maintenance match, text unconfirmable."),
    ],
    "Other Clean__other": [
        ("covers", "na", "DOE B5.14 CHP/cogeneration modification; text verified 10 CFR 1021 App B."),
        ("covers", "na", "DOE B5.1 actions to conserve energy/water; text verified 10 CFR 1021 App B."),
        ("covers", "na", "DOE B5.10 permanent exemptions for existing powerplants; text verified 10 CFR 1021 App B."),
        ("covers", "na", "DOE B4.3 power-marketing rate changes; text verified 10 CFR 1021 App B."),
        ("covers", "na", "DOE B5.9 temporary exemptions for electric powerplants; text verified 10 CFR 1021 App B."),
    ],
    "Solar__assessment": [
        ("does_not_cover", "na", "BOR classification/certification of irrigable lands (agency_doc); off-scope."),
        ("unclear", "na", "NPS designation of environmental study areas (agency_doc); administrative, only loosely 'assessment'."),
        ("does_not_cover", "na", "NPS grants for land acquisition, no disturbance (agency_doc); off-scope."),
        ("does_not_cover", "na", "USDA soil-erosion control structures on ag lands (text verified 7 CFR 1b.4) but substance is soil conservation, not solar site assessment."),
        ("does_not_cover", "na", "NPS landscaping/maintenance in disturbed areas (agency_doc); off-scope."),
    ],
    "Solar__maintenance": [
        ("partially_covers", "no", "DA invasive-species eradication per IPMP (agency_doc, DoD PDF); substantive veg-maintenance match, not eCFR-verifiable."),
        ("partially_covers", "no", "DA pesticide/herbicide program plan (agency_doc); substantive."),
        ("covers", "na", "USDA planting actions (bareland planting, firebreaks); text verified 7 CFR 1b.4; covers vegetation maintenance."),
        ("does_not_cover", "no", "DOE B5.20 biomass power plants (text verified, mw=10) but off-scope for solar maintenance."),
        ("does_not_cover", "no", "BLM wildfire emergency repair (agency_doc, acres=4200); off-scope."),
    ],
    "Transmission__interconnection": [
        ("covers", "yes", "USDA substation construction/modification for small-scale energy; text verified 7 CFR 1b.4; covers interconnection within acres=10/miles=25/kv=230 bounds."),
        ("covers", "yes", "FERC 380.4(a)(17) electrical interconnections/wheeling with no new substation; text verified 18 CFR 380.4; covers interconnection under kv=115."),
        ("partially_covers", "no", "TVA new transmission line <=10 mi/125 acres (legacy 18 CFR 1318 URL, unfetched); on-topic but text unconfirmable."),
        ("partially_covers", "no", "TVA retire/rebuild lines within existing ROW (legacy URL, unfetched); partial."),
        ("partially_covers", "na", "FERC 380.4(a)(19) water-power project utility lines; text verified but narrow line-authorization scope."),
    ],
    "Transmission__land_or_row_authorization": [
        ("partially_covers", "no", "BLM short ROW grant for utility service (agency_doc); substantive transmission-ROW match, not eCFR-verifiable."),
        ("unclear", "na", "BLM routine signs/culverts (agency_doc); tangential."),
        ("partially_covers", "no", "BLM ROW for overhead line crossing corner of public land (agency_doc); substantive."),
        ("partially_covers", "no", "BLM amendments to existing ROW upgrading (agency_doc); substantive."),
        ("does_not_cover", "na", "BLM incorporation of roads/trails, no construction (agency_doc); off-scope."),
    ],
    "Transmission__maintenance": [
        ("does_not_cover", "no", "BLM wildfire emergency repair (agency_doc, acres=4200); off-scope."),
        ("partially_covers", "no", "TVA invasive-plant management <=125 acres (legacy URL, unfetched); ROW veg maintenance, text unconfirmable."),
        ("does_not_cover", "na", "BLM cultivation in tree nurseries (agency_doc); off-scope."),
        ("does_not_cover", "na", "USDA APHIS routine pest-control measures (text verified 7 CFR 1b.4) but off-topic for transmission maintenance."),
        ("partially_covers", "no", "NPS overhead utility-line ROW, no significant visual intrusion (agency_doc); substantive line-ROW maintenance match."),
    ],
    "Transmission__new_build": [
        ("partially_covers", "no", "USDA substation construction (text verified 7 CFR 1b.4) covers within acres=10/miles=25/kv=230, but new-build FONSIs exceed those bounds -> expand."),
        ("partially_covers", "no", "TVA new transmission line <=10 mi/125 acres (legacy URL, unfetched); on-topic but bounded below FONSI scale and text unconfirmable."),
        ("partially_covers", "no", "FERC interconnection without new substation (text verified 18 CFR 380.4); excludes new-build lines."),
        ("partially_covers", "no", "TVA retire/rebuild lines (legacy URL, unfetched); partial."),
        ("does_not_cover", "na", "FirstNet telecom lines construction (agency_doc); off-scope (telecom)."),
    ],
    "Transmission__research_or_demonstration": [
        ("partially_covers", "no", "BOEM test/exploration drilling in a prior-NEPA project (agency_doc); only a generic R&D match."),
        ("partially_covers", "no", "BOEM research/monitoring-device install (agency_doc); generic R&D match."),
        ("does_not_cover", "na", "BOEM production measurement (agency_doc); off-scope."),
        ("does_not_cover", "na", "BOEM offshore drilling APD (agency_doc); off-scope."),
        ("does_not_cover", "na", "BOEM Sundry Notices on wells (agency_doc); off-scope."),
    ],
    "Transmission__upgrade": [
        ("partially_covers", "no", "USDA substation modification (text verified 7 CFR 1b.4) covers within acres=10/miles=25/kv=230, but upgrade FONSIs exceed -> expand."),
        ("partially_covers", "no", "TVA rebuild lines within existing ROW (legacy URL, unfetched, miles=25); on-topic upgrade but text unconfirmable."),
        ("partially_covers", "no", "FirstNet changes to existing lines <20% pole replacement (agency_doc); on-topic upgrade substance, not eCFR-verifiable."),
        ("partially_covers", "no", "TVA new line <=10 mi (legacy URL, unfetched); partial."),
        ("partially_covers", "no", "FirstNet rebuild power lines for road relocation (agency_doc); partial upgrade match."),
    ],
    "Wind__upgrade": [
        ("covers", "yes", "DOE B5.18 wind turbines (<=2 turbines, <200 ft); text verified 10 CFR 1021 App B; covers a small wind upgrade within the 2-turbine/height cap."),
        ("does_not_cover", "no", "BLM wildfire emergency repair (agency_doc, acres=4200); off-scope."),
        ("does_not_cover", "no", "BLM live-tree harvest <=70 acres (agency_doc); off-scope."),
        ("does_not_cover", "no", "BLM post-fire rehabilitation <=4200 acres (agency_doc); off-scope."),
        ("does_not_cover", "na", "BLM temporary field-work camps (agency_doc); off-scope."),
    ],
}

# best-verdict priority for the cell-level roll-up
_PRIORITY = {"covers": 3, "partially_covers": 2, "unclear": 1, "does_not_cover": 0}


def cell_best(verdicts: list[str]) -> str:
    return max(verdicts, key=lambda v: _PRIORITY.get(v, -1))


def main() -> None:
    cov = pd.read_parquet(COVERAGE_OUT)
    run_at = utc_now()

    # sanity: every (cell, rank) in the table
    for cat, rows in V.items():
        assert len(rows) == 5, f"{cat}: expected 5 rows, got {len(rows)}"
    for v, _, _ in (r for rows in V.values() for r in rows):
        assert v in COVERAGE_VALUES, f"bad verdict {v!r}"

    def apply_row(r):
        table = V.get(r["candidate_category"])
        if not table:
            return r
        verdict, bound, note = table[int(r["retrieval_rank"]) - 1]
        r["coverage_verdict"] = verdict
        r["bound_confirmed"] = bound
        r["reviewer_notes"] = note
        return r

    cov = cov.apply(apply_row, axis=1)
    cov["adjudicated_by"] = ADJUDICATOR
    cov["ce_coverage_extraction_run_at"] = cov["ce_coverage_extraction_run_at"].replace("", run_at)
    # ce_coverage_llm_run_at stays "" (no LLM was used — manual adjudication only)

    assert (cov["coverage_verdict"] != "").all(), "some rows left unadjudicated"
    write_parquet(cov, COVERAGE_OUT)
    print(f"[apply] wrote {len(cov)} adjudicated rows -> {COVERAGE_OUT}")
    print("[apply] row verdict counts:", cov["coverage_verdict"].value_counts().to_dict())

    # cell-level roll-up (best verdict across the 5 ranks)
    best = (cov.groupby(["candidate_category", "verdict"])["coverage_verdict"]
            .apply(lambda s: cell_best(list(s))).reset_index(name="cell_best"))
    print("[apply] cell-best counts:", best["cell_best"].value_counts().to_dict())

    # --- re-render the worksheet with the filled verdicts ---
    src = {"ecfr_current": "eCFR", "ecfr_legacy": "eCFR (legacy URL — unfetched)",
           "agency_doc": "AGENCY DOC — not in eCFR", "none": "no URL"}
    lines = ["---", 'title: "Deliverable 6 — eCFR verification of CE adopt/expand matches"', "---", "",
             "Every adopt/expand verdict rests on a *text-similarity* match to an existing CE, confirmed "
             "here against the **current eCFR text** where the citation resolves. `coverage_verdict` "
             f"({' / '.join(COVERAGE_VALUES)}) is filled per (cell, rank) by **{ADJUDICATOR}**.", "",
             "> Citation-quality cap: only the 5 eCFR-current URLs that resolved to text can earn a clean "
             "`covers`. agency-doc CEs (DOI/DoD/NIST/FirstNet procedure PDFs, not in the eCFR) and legacy "
             "cgi-bin eCFR URLs (NRC 10 CFR 51.22, TVA 18 CFR 1318) are text-unverifiable and capped at "
             "`partially_covers`/`unclear`/`does_not_cover`; FTA 23 CFR 771.118 failed to fetch -> `unclear`.", ""]
    for cat, g in cov.groupby("candidate_category"):
        g = g.sort_values("retrieval_rank")
        cb = cell_best(list(g["coverage_verdict"]))
        lines.append(f"## {cat} — {g['verdict'].iloc[0]}  ·  cell-best: **{cb}**")
        for rr in g.itertuples():
            lines.append(f"- **rank {rr.retrieval_rank}** `{rr.structured_id}` ({rr.agency_name}) "
                         f"score {rr.retrieval_score} — {src.get(rr.source_type, rr.source_type)} "
                         f"→ **{rr.coverage_verdict}** (bound_confirmed: {rr.bound_confirmed})")
            if rr.parsed_bounds:
                lines.append(f"  - bounds: {rr.parsed_bounds}")
            lines.append(f"  - {rr.reviewer_notes}")
            lines.append(f"  - [source]({rr.canonical_source_url})  ·  fetched eCFR text: {rr.ecfr_text_chars} chars")
        lines.append("")
    WORKSHEET.parent.mkdir(parents=True, exist_ok=True)
    WORKSHEET.write_text("\n".join(lines) + "\n")
    print(f"[apply] worksheet -> {WORKSHEET}")


if __name__ == "__main__":
    main()
