---
title: "Deliverable 6 — eCFR verification of CE adopt/expand matches"
---

Every adopt/expand verdict rests on a *text-similarity* match to an existing CE, confirmed here against the **current eCFR text** where the citation resolves. `coverage_verdict` (covers / partially_covers / does_not_cover / unclear) is filled per (cell, rank) by **opus-agent-manual-2026-07-22**.

> Citation-quality cap: only the 5 eCFR-current URLs that resolved to text can earn a clean `covers`. agency-doc CEs (DOI/DoD/NIST/FirstNet procedure PDFs, not in the eCFR) and legacy cgi-bin eCFR URLs (NRC 10 CFR 51.22, TVA 18 CFR 1318) are text-unverifiable and capped at `partially_covers`/`unclear`/`does_not_cover`; FTA 23 CFR 771.118 failed to fetch -> `unclear`.

## Biomass__research_or_demonstration — adopt  ·  cell-best: **covers**
- **rank 1** `BOEM---3-22` (Bureau of Ocean Energy Management) score 0.4005 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM offshore oil/gas APD (agency_doc, not in eCFR); off-scope for biomass R&D.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `DOE-1--5-91` (Department of Energy) score 0.3926 — eCFR → **covers** (bound_confirmed: yes)
  - bounds: mw=10.0
  - DOE B5.20 small biomass power plants (<10 MW); text verified in current 10 CFR 1021 App B; covers a biomass demo build within the 10 MW cap.
  - [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)  ·  fetched eCFR text: 91705 chars
- **rank 3** `BOEM---3-15` (Bureau of Ocean Energy Management) score 0.3891 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM production commingling (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `USDA-1-2-2-56` (Department of Agriculture) score 0.3759 — eCFR → **partially_covers** (bound_confirmed: no)
  - bounds: acres=80.0
  - USDA small-scale rural-development CE (text verified 7 CFR 1b.4) but generic financial-assistance scope, not biomass-specific; acres=80 cap not tied to this action.
  - [source](https://www.ecfr.gov/current/title-7/subtitle-A/part-1b/section-1b.4)  ·  fetched eCFR text: 67708 chars
- **rank 5** `BOEM---3-12` (Bureau of Ocean Energy Management) score 0.364 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM production measurement (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Geothermal__new_build — adopt  ·  cell-best: **partially_covers**
- **rank 1** `BLM---2-13` (Bureau of Land Management) score 0.5204 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BLM geophysical-exploration NOI (agency_doc/DOI procedures, not eCFR-verifiable); substantively near geothermal exploration, no new road.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `BLM---2-9` (Bureau of Land Management) score 0.4851 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BLM mineral lease transfers (agency_doc); administrative, not a build.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `BLM---2-10` (Bureau of Land Management) score 0.4799 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BLM unitization/agreements (agency_doc); administrative.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `BLM---2-14` (Bureau of Land Management) score 0.4684 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - bounds: acres=20.0
  - BLM geothermal drilling-permit/confirmation CE (agency_doc, acres=20 unverifiable); strong substantive match but not eCFR-confirmable.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 5** `BLM---2-11` (Bureau of Land Management) score 0.4444 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BLM suspensions of operations (agency_doc); administrative.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Geothermal__other — adopt  ·  cell-best: **covers**
- **rank 1** `BLM---2-13` (Bureau of Land Management) score 0.4457 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BLM geophysical-exploration NOI (agency_doc); partial geothermal match, not eCFR-verifiable.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `NPS---3-34` (National Park Service) score 0.4265 — AGENCY DOC — not in eCFR → **unclear** (bound_confirmed: na)
  - NPS wells/comfort stations (agency_doc); tangential.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `BLM---2-9` (Bureau of Land Management) score 0.4022 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BLM mineral lease transfers (agency_doc); administrative.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `DOE-1--5-90` (Department of Energy) score 0.3975 — eCFR → **covers** (bound_confirmed: na)
  - DOE B5.19 ground-source heat pumps; text verified 10 CFR 1021 App B; covers the GSHP slice of geothermal-other.
  - [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)  ·  fetched eCFR text: 91705 chars
- **rank 5** `BLM---2-14` (Bureau of Land Management) score 0.3929 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - bounds: acres=20.0
  - BLM geothermal drilling-permit CE (agency_doc, acres=20 unverifiable); partial.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Geothermal__research_or_demonstration — adopt  ·  cell-best: **partially_covers**
- **rank 1** `BLM---2-13` (Bureau of Land Management) score 0.485 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BLM geophysical-exploration NOI (agency_doc); strong substantive match to geothermal R&D exploration but not eCFR-verifiable.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `USGS---1-6` (U.S. Geological Survey) score 0.447 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - USGS exploratory well drilling, no access road / no significant disturbance (agency_doc); substantive.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `NPS---3-39` (National Park Service) score 0.4342 — AGENCY DOC — not in eCFR → **unclear** (bound_confirmed: na)
  - NPS underground utilities in disturbed areas (agency_doc); tangential.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `USGS---1-4` (U.S. Geological Survey) score 0.4302 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - USGS well logging / aquifer testing (agency_doc); matches assessment-type R&D.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 5** `BLM---2-9` (Bureau of Land Management) score 0.4174 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BLM mineral lease transfers (agency_doc); administrative.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Hydropower__assessment — adopt  ·  cell-best: **partially_covers**
- **rank 1** `BOR---2-6` (Bureau of Reclamation) score 0.5653 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BOR data-collection / test-excavation studies (agency_doc); strong substantive assessment match, localized impacts.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `USGS---1-7` (U.S. Geological Survey) score 0.5539 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - USGS test/exploration drilling & downhole testing (agency_doc); substantive.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `BOR---4-25` (Bureau of Reclamation) score 0.5398 — AGENCY DOC — not in eCFR → **unclear** (bound_confirmed: na)
  - BOR minor safety-of-dams construction (agency_doc); construction, not assessment.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `USGS---1-6` (U.S. Geological Survey) score 0.5374 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - USGS exploratory groundwater well drilling (agency_doc); partial.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 5** `USGS---1-5` (U.S. Geological Survey) score 0.518 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - USGS hydrologic/water-quality monitoring structures (agency_doc); substantive assessment match.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Hydropower__new_build — adopt  ·  cell-best: **does_not_cover**
- **rank 1** `BOEM---2-7` (Bureau of Ocean Energy Management) score 0.4514 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM research/monitoring devices (agency_doc); off-scope for hydropower new build.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `BOEM---2-10` (Bureau of Ocean Energy Management) score 0.3994 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM prelease planning (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `BOEM---3-24` (Bureau of Ocean Energy Management) score 0.3949 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM Sundry Notices on wells (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `BOEM---3-12` (Bureau of Ocean Energy Management) score 0.3942 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM production measurement (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 5** `BOEM---3-15` (Bureau of Ocean Energy Management) score 0.382 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM production commingling (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Hydropower__research_or_demonstration — adopt  ·  cell-best: **partially_covers**
- **rank 1** `BOEM---2-7` (Bureau of Ocean Energy Management) score 0.5151 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BOEM research/monitoring-device install (agency_doc); only a generic R&D-device match, offshore oil/gas source.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `BOEM---3-13` (Bureau of Ocean Energy Management) score 0.5016 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM off-lease storage (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `BOEM---3-12` (Bureau of Ocean Energy Management) score 0.501 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM production measurement (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `BOEM---3-17` (Bureau of Ocean Energy Management) score 0.4999 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM lease consolidation (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 5** `BOEM---3-15` (Bureau of Ocean Energy Management) score 0.4915 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM production commingling (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Hydropower__upgrade — adopt  ·  cell-best: **covers**
- **rank 1** `USDA--2-2-51` (Department of Agriculture) score 0.5016 — eCFR → **covers** (bound_confirmed: na)
  - USDA increase-freeboard of an existing NRCS dam; text verified 7 CFR 1b.4; covers dam upgrade for NRCS-standard dams.
  - [source](https://www.ecfr.gov/current/title-7/subtitle-A/part-1b/section-1b.4)  ·  fetched eCFR text: 67708 chars
- **rank 2** `BOR---4-25` (Bureau of Reclamation) score 0.453 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BOR minor safety-of-dams construction (agency_doc); substantive but not eCFR-verifiable.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `NPS---3-39` (National Park Service) score 0.3976 — AGENCY DOC — not in eCFR → **unclear** (bound_confirmed: na)
  - NPS underground utilities (agency_doc); tangential.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `USDA--2-2-49` (Department of Agriculture) score 0.3902 — eCFR → **covers** (bound_confirmed: na)
  - USDA repair/improve existing emergency spillways to safety standards; text verified 7 CFR 1b.4; covers dam upgrade.
  - [source](https://www.ecfr.gov/current/title-7/subtitle-A/part-1b/section-1b.4)  ·  fetched eCFR text: 67708 chars
- **rank 5** `FERC---1-19` (Federal Energy Regulatory Commission) score 0.3815 — eCFR → **partially_covers** (bound_confirmed: na)
  - FERC 380.4(a)(19) water-power project utility lines; text verified 18 CFR 380.4 but scope is line authorization, not generation upgrade.
  - [source](https://www.ecfr.gov/current/title-18/chapter-I/subchapter-W/part-380/section-380.4)  ·  fetched eCFR text: 10623 chars

## Nuclear__assessment — adopt  ·  cell-best: **covers**
- **rank 1** `BOEM---3-20` (Bureau of Ocean Energy Management) score 0.5016 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM offshore lease exploration plan (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `FRA---1-4` (Federal Railroad Administration) score 0.458 — eCFR → **covers** (bound_confirmed: na)
  - FRA localized geotechnical investigations / test bores; text verified 23 CFR 771.116; covers nuclear site assessment/geotech.
  - [source](https://www.ecfr.gov/current/title-23/chapter-I/subchapter-H/part-771#771.116)  ·  fetched eCFR text: 103270 chars
- **rank 3** `FTA---1-16` (Federal Transit Administration) score 0.4405 — eCFR → **unclear** (bound_confirmed: na)
  - FTA identical geotech CE at 23 CFR 771.118 but eCFR fetch failed (0 chars, trailing-space URL); substance matches verified FRA/FHWA siblings, text unconfirmed.
  - [source](https://www.ecfr.gov/current/title-23/section-771.118 )  ·  fetched eCFR text: 0 chars
- **rank 4** `FHWA---1-24` (Federal Highway Administration) score 0.4353 — eCFR → **covers** (bound_confirmed: na)
  - FHWA localized geotechnical investigation / test bores; text verified 23 CFR 771.117; covers site assessment/geotech.
  - [source](https://www.ecfr.gov/current/title-23/section-771.117)  ·  fetched eCFR text: 19274 chars
- **rank 5** `BIA---4-10` (Bureau of Indian Affairs) score 0.4314 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BIA geologic mapping/reconnaissance/surface-sampling permits (agency_doc); substantive assessment match, not eCFR-verifiable.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Nuclear__manufacturing — adopt  ·  cell-best: **partially_covers**
- **rank 1** `DA---8-73` (U.S. Army) score 0.4129 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - DA R&D/production/manufacturing at existing enclosed facilities (agency_doc, DoD PDF); substantive match, not eCFR-verifiable.
  - [source](https://www.denix.osd.mil/nepa/denix-files/sites/55/2025/06/DOD-NEPA-Procedures-APPENDIX-A_FINAL.pdf)  ·  fetched eCFR text: 0 chars
- **rank 2** `NRC---1-15` (Nuclear Regulatory Commission) score 0.41 — eCFR (legacy URL — unfetched) → **does_not_cover** (bound_confirmed: no)
  - NRC import-license CE (10 CFR 51.22, legacy cgi-bin URL, unfetched); licensing import, not manufacturing.
  - [source](https://www.ecfr.gov/cgi-bin/text-idx?SID=5bddd0accdfc420ea5e3e2067fb33a61&mc=true&node=se10.2.51_122&rgn=div8)  ·  fetched eCFR text: 0 chars
- **rank 3** `NIST--1-3-12` (National Institute of Standards and Technology) score 0.404 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - NIST install/operate manufacturing machinery (agency_doc); substantive.
  - [source](https://www.nist.gov/system/files/documents/2025/01/17/Doc%201%20-%20NIST%20NEPA%20Procedures%20Final%201-17.pdf)  ·  fetched eCFR text: 0 chars
- **rank 4** `NRC---1-18` (Nuclear Regulatory Commission) score 0.4011 — eCFR (legacy URL — unfetched) → **does_not_cover** (bound_confirmed: no)
  - NRC authorize resume-operation amendment (legacy URL, unfetched); off-scope.
  - [source](https://www.ecfr.gov/cgi-bin/text-idx?SID=5bddd0accdfc420ea5e3e2067fb33a61&mc=true&node=se10.2.51_122&rgn=div8)  ·  fetched eCFR text: 0 chars
- **rank 5** `NRC---1-12` (Nuclear Regulatory Commission) score 0.4011 — eCFR (legacy URL — unfetched) → **does_not_cover** (bound_confirmed: no)
  - NRC license-amendment safeguards (legacy URL, unfetched); off-scope.
  - [source](https://www.ecfr.gov/cgi-bin/text-idx?SID=5bddd0accdfc420ea5e3e2067fb33a61&mc=true&node=se10.2.51_122&rgn=div8)  ·  fetched eCFR text: 0 chars

## Nuclear__other — adopt  ·  cell-best: **covers**
- **rank 1** `NRC---1-11` (Nuclear Regulatory Commission) score 0.4992 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - NRC administrative/procedural license amendments (legacy 10 CFR 51.22 URL, unfetched); partial 'other' match, text unconfirmable.
  - [source](https://www.ecfr.gov/cgi-bin/text-idx?SID=5bddd0accdfc420ea5e3e2067fb33a61&mc=true&node=se10.2.51_122&rgn=div8)  ·  fetched eCFR text: 0 chars
- **rank 2** `NRC---1-9` (Nuclear Regulatory Commission) score 0.4961 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - NRC reactor-license amendment (legacy URL, unfetched); partial.
  - [source](https://www.ecfr.gov/cgi-bin/text-idx?SID=5bddd0accdfc420ea5e3e2067fb33a61&mc=true&node=se10.2.51_122&rgn=div8)  ·  fetched eCFR text: 0 chars
- **rank 3** `NRC---1-12` (Nuclear Regulatory Commission) score 0.4843 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - NRC amendment on safeguards (legacy URL, unfetched); partial.
  - [source](https://www.ecfr.gov/cgi-bin/text-idx?SID=5bddd0accdfc420ea5e3e2067fb33a61&mc=true&node=se10.2.51_122&rgn=div8)  ·  fetched eCFR text: 0 chars
- **rank 4** `DOE-1--1-10` (Department of Energy) score 0.4818 — eCFR → **covers** (bound_confirmed: na)
  - DOE B1.10 onsite storage of activated material at existing facility; text verified 10 CFR 1021 App B; genuine nuclear CE.
  - [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)  ·  fetched eCFR text: 91705 chars
- **rank 5** `DOE-1--2-42` (Department of Energy) score 0.4812 — eCFR → **covers** (bound_confirmed: na)
  - DOE B2.6 recovery of radioactive sealed sources; text verified 10 CFR 1021 App B; genuine nuclear CE.
  - [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)  ·  fetched eCFR text: 91705 chars

## Nuclear__upgrade — adopt  ·  cell-best: **partially_covers**
- **rank 1** `NRC---1-11` (Nuclear Regulatory Commission) score 0.465 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - NRC admin/procedural amendments (legacy URL, unfetched); only loosely 'upgrade', text unconfirmable.
  - [source](https://www.ecfr.gov/cgi-bin/text-idx?SID=5bddd0accdfc420ea5e3e2067fb33a61&mc=true&node=se10.2.51_122&rgn=div8)  ·  fetched eCFR text: 0 chars
- **rank 2** `NRC---1-20` (Nuclear Regulatory Commission) score 0.4349 — eCFR (legacy URL — unfetched) → **does_not_cover** (bound_confirmed: no)
  - NRC decommissioning of limited sites (legacy URL); decommissioning, not upgrade.
  - [source](https://www.ecfr.gov/cgi-bin/text-idx?SID=5bddd0accdfc420ea5e3e2067fb33a61&mc=true&node=se10.2.51_122&rgn=div8)  ·  fetched eCFR text: 0 chars
- **rank 3** `NRC---1-15` (Nuclear Regulatory Commission) score 0.4216 — eCFR (legacy URL — unfetched) → **does_not_cover** (bound_confirmed: no)
  - NRC import licenses (legacy URL); off-scope.
  - [source](https://www.ecfr.gov/cgi-bin/text-idx?SID=5bddd0accdfc420ea5e3e2067fb33a61&mc=true&node=se10.2.51_122&rgn=div8)  ·  fetched eCFR text: 0 chars
- **rank 4** `NRC---1-19` (Nuclear Regulatory Commission) score 0.4208 — eCFR (legacy URL — unfetched) → **unclear** (bound_confirmed: no)
  - NRC certificate of compliance, gaseous-diffusion facilities (legacy URL, unfetched); ambiguous fit.
  - [source](https://www.ecfr.gov/cgi-bin/text-idx?SID=5bddd0accdfc420ea5e3e2067fb33a61&mc=true&node=se10.2.51_122&rgn=div8)  ·  fetched eCFR text: 0 chars
- **rank 5** `NRC---1-24` (Nuclear Regulatory Commission) score 0.4081 — eCFR (legacy URL — unfetched) → **does_not_cover** (bound_confirmed: no)
  - NRC scholarship grants (legacy URL); off-scope.
  - [source](https://www.ecfr.gov/cgi-bin/text-idx?SID=5bddd0accdfc420ea5e3e2067fb33a61&mc=true&node=se10.2.51_122&rgn=div8)  ·  fetched eCFR text: 0 chars

## Other Clean__land_or_row_authorization — adopt  ·  cell-best: **partially_covers**
- **rank 1** `BLM---5-49` (Bureau of Land Management) score 0.5269 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BLM short ROW grant for utility service (agency_doc); substantive land/ROW match, not eCFR-verifiable.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `BLM---9-68` (Bureau of Land Management) score 0.5189 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: no)
  - bounds: acres=4200.0
  - BLM wildfire/flood emergency repair (agency_doc, acres=4200); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `BLM---5-51` (Bureau of Land Management) score 0.5099 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BLM short-term ROW / land-use authorizations (agency_doc); substantive.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `BLM---5-45` (Bureau of Land Management) score 0.5057 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BLM amendments to existing ROW (agency_doc); substantive.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 5** `BLM---7-64` (Bureau of Land Management) score 0.5013 — AGENCY DOC — not in eCFR → **unclear** (bound_confirmed: na)
  - BLM routine signs/culverts on roads (agency_doc); tangential.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Other Clean__maintenance — adopt  ·  cell-best: **covers**
- **rank 1** `USDA--2-2-35` (Department of Agriculture) score 0.635 — eCFR → **covers** (bound_confirmed: na)
  - USDA revegetation of disturbed sites (herbaceous/woody planting); text verified 7 CFR 1b.4; covers maintenance/restoration.
  - [source](https://www.ecfr.gov/current/title-7/subtitle-A/part-1b/section-1b.4)  ·  fetched eCFR text: 67708 chars
- **rank 2** `NPS---5-55` (National Park Service) score 0.5912 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - NPS native-species restoration (agency_doc); substantive but not eCFR-verifiable.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `NPS---5-53` (National Park Service) score 0.5782 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - NPS stabilization by native planting (agency_doc); substantive.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `USDA--1-1-26` (Department of Agriculture) score 0.5726 — eCFR → **covers** (bound_confirmed: na)
  - USDA minor short-term special uses of NFS lands; text verified 7 CFR 1b.4; covers minor maintenance uses.
  - [source](https://www.ecfr.gov/current/title-7/subtitle-A/part-1b/section-1b.4)  ·  fetched eCFR text: 67708 chars
- **rank 5** `TVA---1-31` (Tennessee Valley Authority) score 0.5725 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - bounds: acres=125.0
  - TVA invasive-plant management <=125 acres (legacy 18 CFR 1318 URL, unfetched); substantive veg-maintenance match, text unconfirmable.
  - [source](https://www.ecfr.gov/cgi-bin/retrieveECFR?gp=&SID=b078548e02c4c905daad07c8245ec2ad&mc=true&n=pt18.2.1318&r=PART&ty=HTML#ap18.2.1318_1202.a)  ·  fetched eCFR text: 0 chars

## Other Clean__other — adopt  ·  cell-best: **covers**
- **rank 1** `DOE-1--5-85` (Department of Energy) score 0.4181 — eCFR → **covers** (bound_confirmed: na)
  - DOE B5.14 CHP/cogeneration modification; text verified 10 CFR 1021 App B.
  - [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)  ·  fetched eCFR text: 91705 chars
- **rank 2** `DOE-1--5-73` (Department of Energy) score 0.407 — eCFR → **covers** (bound_confirmed: na)
  - DOE B5.1 actions to conserve energy/water; text verified 10 CFR 1021 App B.
  - [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)  ·  fetched eCFR text: 91705 chars
- **rank 3** `DOE-1--5-81` (Department of Energy) score 0.3898 — eCFR → **covers** (bound_confirmed: na)
  - DOE B5.10 permanent exemptions for existing powerplants; text verified 10 CFR 1021 App B.
  - [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)  ·  fetched eCFR text: 91705 chars
- **rank 4** `DOE-1--4-61` (Department of Energy) score 0.3821 — eCFR → **covers** (bound_confirmed: na)
  - DOE B4.3 power-marketing rate changes; text verified 10 CFR 1021 App B.
  - [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)  ·  fetched eCFR text: 91705 chars
- **rank 5** `DOE-1--5-80` (Department of Energy) score 0.3817 — eCFR → **covers** (bound_confirmed: na)
  - DOE B5.9 temporary exemptions for electric powerplants; text verified 10 CFR 1021 App B.
  - [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)  ·  fetched eCFR text: 91705 chars

## Solar__assessment — adopt  ·  cell-best: **unclear**
- **rank 1** `BOR---3-7` (Bureau of Reclamation) score 0.5383 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOR classification/certification of irrigable lands (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `NPS---5-52` (National Park Service) score 0.5281 — AGENCY DOC — not in eCFR → **unclear** (bound_confirmed: na)
  - NPS designation of environmental study areas (agency_doc); administrative, only loosely 'assessment'.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `NPS---6-60` (National Park Service) score 0.5266 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - NPS grants for land acquisition, no disturbance (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `USDA--2-2-54` (Department of Agriculture) score 0.5183 — eCFR → **does_not_cover** (bound_confirmed: na)
  - USDA soil-erosion control structures on ag lands (text verified 7 CFR 1b.4) but substance is soil conservation, not solar site assessment.
  - [source](https://www.ecfr.gov/current/title-7/subtitle-A/part-1b/section-1b.4)  ·  fetched eCFR text: 67708 chars
- **rank 5** `NPS---3-42` (National Park Service) score 0.5096 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - NPS landscaping/maintenance in disturbed areas (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Solar__maintenance — adopt  ·  cell-best: **covers**
- **rank 1** `DA---3-35` (U.S. Army) score 0.57 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - DA invasive-species eradication per IPMP (agency_doc, DoD PDF); substantive veg-maintenance match, not eCFR-verifiable.
  - [source](https://www.denix.osd.mil/nepa/denix-files/sites/55/2025/06/DOD-NEPA-Procedures-APPENDIX-A_FINAL.pdf)  ·  fetched eCFR text: 0 chars
- **rank 2** `DA---6-59` (U.S. Army) score 0.5121 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - DA pesticide/herbicide program plan (agency_doc); substantive.
  - [source](https://www.denix.osd.mil/nepa/denix-files/sites/55/2025/06/DOD-NEPA-Procedures-APPENDIX-A_FINAL.pdf)  ·  fetched eCFR text: 0 chars
- **rank 3** `USDA--1-1-14` (Department of Agriculture) score 0.4504 — eCFR → **covers** (bound_confirmed: na)
  - USDA planting actions (bareland planting, firebreaks); text verified 7 CFR 1b.4; covers vegetation maintenance.
  - [source](https://www.ecfr.gov/current/title-7/subtitle-A/part-1b/section-1b.4)  ·  fetched eCFR text: 67708 chars
- **rank 4** `DOE-1--5-91` (Department of Energy) score 0.4461 — eCFR → **does_not_cover** (bound_confirmed: no)
  - bounds: mw=10.0
  - DOE B5.20 biomass power plants (text verified, mw=10) but off-scope for solar maintenance.
  - [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)  ·  fetched eCFR text: 91705 chars
- **rank 5** `BLM---9-68` (Bureau of Land Management) score 0.4429 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: no)
  - bounds: acres=4200.0
  - BLM wildfire emergency repair (agency_doc, acres=4200); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Transmission__interconnection — adopt  ·  cell-best: **covers**
- **rank 1** `USDA-1-2-2-58` (Department of Agriculture) score 0.4912 — eCFR → **covers** (bound_confirmed: yes)
  - bounds: acres=10.0; miles=25.0; kv=230.0
  - USDA substation construction/modification for small-scale energy; text verified 7 CFR 1b.4; covers interconnection within acres=10/miles=25/kv=230 bounds.
  - [source](https://www.ecfr.gov/current/title-7/subtitle-A/part-1b/section-1b.4)  ·  fetched eCFR text: 67708 chars
- **rank 2** `FERC---1-17` (Federal Energy Regulatory Commission) score 0.4345 — eCFR → **covers** (bound_confirmed: yes)
  - bounds: kv=115.0
  - FERC 380.4(a)(17) electrical interconnections/wheeling with no new substation; text verified 18 CFR 380.4; covers interconnection under kv=115.
  - [source](https://www.ecfr.gov/current/title-18/chapter-I/subchapter-W/part-380/section-380.4)  ·  fetched eCFR text: 10623 chars
- **rank 3** `TVA---1-15` (Tennessee Valley Authority) score 0.4246 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - bounds: acres=125.0; miles=10.0
  - TVA new transmission line <=10 mi/125 acres (legacy 18 CFR 1318 URL, unfetched); on-topic but text unconfirmable.
  - [source](https://www.ecfr.gov/cgi-bin/retrieveECFR?gp=&SID=b078548e02c4c905daad07c8245ec2ad&mc=true&n=pt18.2.1318&r=PART&ty=HTML#ap18.2.1318_1202.a)  ·  fetched eCFR text: 0 chars
- **rank 4** `TVA---1-18` (Tennessee Valley Authority) score 0.362 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - bounds: acres=125.0; miles=25.0
  - TVA retire/rebuild lines within existing ROW (legacy URL, unfetched); partial.
  - [source](https://www.ecfr.gov/cgi-bin/retrieveECFR?gp=&SID=b078548e02c4c905daad07c8245ec2ad&mc=true&n=pt18.2.1318&r=PART&ty=HTML#ap18.2.1318_1202.a)  ·  fetched eCFR text: 0 chars
- **rank 5** `FERC---1-19` (Federal Energy Regulatory Commission) score 0.3291 — eCFR → **partially_covers** (bound_confirmed: na)
  - FERC 380.4(a)(19) water-power project utility lines; text verified but narrow line-authorization scope.
  - [source](https://www.ecfr.gov/current/title-18/chapter-I/subchapter-W/part-380/section-380.4)  ·  fetched eCFR text: 10623 chars

## Transmission__land_or_row_authorization — adopt  ·  cell-best: **partially_covers**
- **rank 1** `BLM---5-49` (Bureau of Land Management) score 0.504 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BLM short ROW grant for utility service (agency_doc); substantive transmission-ROW match, not eCFR-verifiable.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `BLM---7-64` (Bureau of Land Management) score 0.4833 — AGENCY DOC — not in eCFR → **unclear** (bound_confirmed: na)
  - BLM routine signs/culverts (agency_doc); tangential.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `BLM---5-46` (Bureau of Land Management) score 0.4803 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BLM ROW for overhead line crossing corner of public land (agency_doc); substantive.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `BLM---5-45` (Bureau of Land Management) score 0.4743 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BLM amendments to existing ROW upgrading (agency_doc); substantive.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 5** `BLM---7-63` (Bureau of Land Management) score 0.469 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BLM incorporation of roads/trails, no construction (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Transmission__maintenance — adopt  ·  cell-best: **partially_covers**
- **rank 1** `BLM---9-68` (Bureau of Land Management) score 0.4207 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: no)
  - bounds: acres=4200.0
  - BLM wildfire emergency repair (agency_doc, acres=4200); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `TVA---1-31` (Tennessee Valley Authority) score 0.4042 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - bounds: acres=125.0
  - TVA invasive-plant management <=125 acres (legacy URL, unfetched); ROW veg maintenance, text unconfirmable.
  - [source](https://www.ecfr.gov/cgi-bin/retrieveECFR?gp=&SID=b078548e02c4c905daad07c8245ec2ad&mc=true&n=pt18.2.1318&r=PART&ty=HTML#ap18.2.1318_1202.a)  ·  fetched eCFR text: 0 chars
- **rank 3** `BLM---3-15` (Bureau of Land Management) score 0.401 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BLM cultivation in tree nurseries (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `USDA--1-1-8` (Department of Agriculture) score 0.3988 — eCFR → **does_not_cover** (bound_confirmed: na)
  - USDA APHIS routine pest-control measures (text verified 7 CFR 1b.4) but off-topic for transmission maintenance.
  - [source](https://www.ecfr.gov/current/title-7/subtitle-A/part-1b/section-1b.4)  ·  fetched eCFR text: 67708 chars
- **rank 5** `NPS---3-37` (National Park Service) score 0.3958 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - NPS overhead utility-line ROW, no significant visual intrusion (agency_doc); substantive line-ROW maintenance match.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Transmission__new_build — expand  ·  cell-best: **partially_covers**
- **rank 1** `USDA-1-2-2-58` (Department of Agriculture) score 0.4056 — eCFR → **partially_covers** (bound_confirmed: no)
  - bounds: acres=10.0; miles=25.0; kv=230.0
  - USDA substation construction (text verified 7 CFR 1b.4) covers within acres=10/miles=25/kv=230, but new-build FONSIs exceed those bounds -> expand.
  - [source](https://www.ecfr.gov/current/title-7/subtitle-A/part-1b/section-1b.4)  ·  fetched eCFR text: 67708 chars
- **rank 2** `TVA---1-15` (Tennessee Valley Authority) score 0.3985 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - bounds: acres=125.0; miles=10.0
  - TVA new transmission line <=10 mi/125 acres (legacy URL, unfetched); on-topic but bounded below FONSI scale and text unconfirmable.
  - [source](https://www.ecfr.gov/cgi-bin/retrieveECFR?gp=&SID=b078548e02c4c905daad07c8245ec2ad&mc=true&n=pt18.2.1318&r=PART&ty=HTML#ap18.2.1318_1202.a)  ·  fetched eCFR text: 0 chars
- **rank 3** `FERC---1-17` (Federal Energy Regulatory Commission) score 0.3855 — eCFR → **partially_covers** (bound_confirmed: no)
  - bounds: kv=115.0
  - FERC interconnection without new substation (text verified 18 CFR 380.4); excludes new-build lines.
  - [source](https://www.ecfr.gov/current/title-18/chapter-I/subchapter-W/part-380/section-380.4)  ·  fetched eCFR text: 10623 chars
- **rank 4** `TVA---1-18` (Tennessee Valley Authority) score 0.3571 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - bounds: acres=125.0; miles=25.0
  - TVA retire/rebuild lines (legacy URL, unfetched); partial.
  - [source](https://www.ecfr.gov/cgi-bin/retrieveECFR?gp=&SID=b078548e02c4c905daad07c8245ec2ad&mc=true&n=pt18.2.1318&r=PART&ty=HTML#ap18.2.1318_1202.a)  ·  fetched eCFR text: 0 chars
- **rank 5** `FirstNet---2-7` (First Responder Network Authority) score 0.3454 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - FirstNet telecom lines construction (agency_doc); off-scope (telecom).
  - [source](https://www.firstnet.gov/sites/default/files/FirstNet_Authority_NEPA_Implementing_Procedures_%28Revised%20June%202025%29.pdf)  ·  fetched eCFR text: 0 chars

## Transmission__research_or_demonstration — adopt  ·  cell-best: **partially_covers**
- **rank 1** `BOEM---2-8` (Bureau of Ocean Energy Management) score 0.4587 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BOEM test/exploration drilling in a prior-NEPA project (agency_doc); only a generic R&D match.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 2** `BOEM---2-7` (Bureau of Ocean Energy Management) score 0.4552 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - BOEM research/monitoring-device install (agency_doc); generic R&D match.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `BOEM---3-12` (Bureau of Ocean Energy Management) score 0.4329 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM production measurement (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `BOEM---3-22` (Bureau of Ocean Energy Management) score 0.4248 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM offshore drilling APD (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 5** `BOEM---3-24` (Bureau of Ocean Energy Management) score 0.4236 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BOEM Sundry Notices on wells (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

## Transmission__upgrade — expand  ·  cell-best: **partially_covers**
- **rank 1** `USDA-1-2-2-58` (Department of Agriculture) score 0.4135 — eCFR → **partially_covers** (bound_confirmed: no)
  - bounds: acres=10.0; miles=25.0; kv=230.0
  - USDA substation modification (text verified 7 CFR 1b.4) covers within acres=10/miles=25/kv=230, but upgrade FONSIs exceed -> expand.
  - [source](https://www.ecfr.gov/current/title-7/subtitle-A/part-1b/section-1b.4)  ·  fetched eCFR text: 67708 chars
- **rank 2** `TVA---1-18` (Tennessee Valley Authority) score 0.3909 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - bounds: acres=125.0; miles=25.0
  - TVA rebuild lines within existing ROW (legacy URL, unfetched, miles=25); on-topic upgrade but text unconfirmable.
  - [source](https://www.ecfr.gov/cgi-bin/retrieveECFR?gp=&SID=b078548e02c4c905daad07c8245ec2ad&mc=true&n=pt18.2.1318&r=PART&ty=HTML#ap18.2.1318_1202.a)  ·  fetched eCFR text: 0 chars
- **rank 3** `FirstNet---2-8` (First Responder Network Authority) score 0.3902 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - FirstNet changes to existing lines <20% pole replacement (agency_doc); on-topic upgrade substance, not eCFR-verifiable.
  - [source](https://www.firstnet.gov/sites/default/files/FirstNet_Authority_NEPA_Implementing_Procedures_%28Revised%20June%202025%29.pdf)  ·  fetched eCFR text: 0 chars
- **rank 4** `TVA---1-15` (Tennessee Valley Authority) score 0.3836 — eCFR (legacy URL — unfetched) → **partially_covers** (bound_confirmed: no)
  - bounds: acres=125.0; miles=10.0
  - TVA new line <=10 mi (legacy URL, unfetched); partial.
  - [source](https://www.ecfr.gov/cgi-bin/retrieveECFR?gp=&SID=b078548e02c4c905daad07c8245ec2ad&mc=true&n=pt18.2.1318&r=PART&ty=HTML#ap18.2.1318_1202.a)  ·  fetched eCFR text: 0 chars
- **rank 5** `FirstNet---2-16` (First Responder Network Authority) score 0.3676 — AGENCY DOC — not in eCFR → **partially_covers** (bound_confirmed: no)
  - FirstNet rebuild power lines for road relocation (agency_doc); partial upgrade match.
  - [source](https://www.firstnet.gov/sites/default/files/FirstNet_Authority_NEPA_Implementing_Procedures_%28Revised%20June%202025%29.pdf)  ·  fetched eCFR text: 0 chars

## Wind__upgrade — adopt  ·  cell-best: **covers**
- **rank 1** `DOE-1--5-89` (Department of Energy) score 0.4682 — eCFR → **covers** (bound_confirmed: yes)
  - DOE B5.18 wind turbines (<=2 turbines, <200 ft); text verified 10 CFR 1021 App B; covers a small wind upgrade within the 2-turbine/height cap.
  - [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)  ·  fetched eCFR text: 91705 chars
- **rank 2** `BLM---9-68` (Bureau of Land Management) score 0.4161 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: no)
  - bounds: acres=4200.0
  - BLM wildfire emergency repair (agency_doc, acres=4200); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 3** `BLM---3-21` (Bureau of Land Management) score 0.3863 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: no)
  - bounds: acres=70.0; miles=0.5
  - BLM live-tree harvest <=70 acres (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 4** `BLM---9-70` (Bureau of Land Management) score 0.3727 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: no)
  - bounds: acres=4200.0
  - BLM post-fire rehabilitation <=4200 acres (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars
- **rank 5** `BLM---10-73` (Bureau of Land Management) score 0.3589 — AGENCY DOC — not in eCFR → **does_not_cover** (bound_confirmed: na)
  - BLM temporary field-work camps (agency_doc); off-scope.
  - [source](https://www.doi.gov/media/document/doi-nepa-appendix-2)  ·  fetched eCFR text: 0 chars

