---
title: "Deliverable 6 — CE coverage verification worksheet"
---

The Analysis-1 **adopt** verdicts rest on a *text-similarity* match between each candidate action type and an existing Categorical Exclusion (CE) — every match is currently `pending` manual verification. This worksheet pairs each candidate with its best-match CE, a first-pass read of the CE text, and the specific bound a reviewer should confirm **against the current eCFR text** before the match is treated as final.

> The *coverage read* below is a first pass over the CE Explorer **snapshot** text — not a legal determination. Confirm each against live eCFR.

## Summary

| Candidate (CE-shaped n) | Best-match CE | Coverage read | Bound to verify |
|---|---|---|---|
| transmission_upgrade (34) | `TVA---1-16` (TVA) | **LIKELY COVERS (qualitative)** | TVA #17 is qualitative ('routine modification ... minor upgrade of existing transmission') with no mileage cap. Confirm the candidate rebuil… |
| geothermal_exploration (9) | `DOE-1--3-43` (DOE) | **LIKELY COVERS** | B3.1 site characterization is qualitative; confirm exploratory drilling depth / well count is within DOE B3.1 scope. |
| solar (9) | `DOE-1--5-87` (DOE) | **PARTIAL — category contamination, re-examine first** | B5.16 covers commercially available solar PV SYSTEMS. But 6 of 9 CE-shaped 'solar' rows are GEN-TIE / interconnection TRANSMISSION lines (th… |
| wind_onshore (7) | `DOE-1--5-89` (DOE) | **CONDITIONAL — turbine-count bound** | B5.18 caps at a SMALL NUMBER (generally <=2) of commercially available wind turbines. Candidates above ~2 turbines EXCEED it -> expand, not … |
| temporary_resource_assessment (4) | `DOE-1--3-43` (DOE) | **LIKELY COVERS** | B3.1 site characterization is qualitative; confirm the met-tower / boring / survey scope is within it. |

## Detail by category

### transmission_upgrade — 34 CE-shaped

- **Best-match CE:** `TVA---1-16` (TVA), match score 0.4469 — [source](https://www.ecfr.gov/cgi-bin/retrieveECFR?gp=&SID=b078548e02c4c905daad07c8245ec2ad&mc=true&n=pt18.2.1318&r=PART&ty=HTML#ap18.2.1318_1202.a)
- **CE text:** 17. Routine modification, repair, and maintenance of, and minor upgrade of and addition to, existing transmission infrastructure, including the addition, retirement, and/or replacement of breakers, transformers, bushings, and relays; transmission line uprate, modification, reconductoring, and clearance resolution; and limited pole replacement. This exclusion also applies to improvements of existin
- **Extraordinary circumstances:** Not Catalogued
- **Candidate scope:** line 1-1534 mi (n=15); agencies: Bureau of Land Management; Bureau of Reclamation; Department of Energy; Power Marketing Administration
- **Coverage read:** **LIKELY COVERS (qualitative)** — TVA #17 explicitly covers modification/repair/maintenance/minor upgrade of EXISTING transmission infrastructure; the CE-shaped candidates are within-ROW modify-existing lines. Cross-agency: BLM/BOR/DOE/PMA would ADOPT a TVA CE.
- **Bound to verify:** TVA #17 is qualitative ('routine modification ... minor upgrade of existing transmission') with no mileage cap. Confirm the candidate rebuilds read as 'minor'; large rebuilds (>~25 mi) already fall to expand, not adopt.

### geothermal_exploration — 9 CE-shaped

- **Best-match CE:** `DOE-1--3-43` (DOE), match score 0.4977 — [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)
- **CE text:** B3.1 SITE CHARACTERIZATION AND ENVIRONMENTAL MONITORING. Site characterization and environmental monitoring (including, but not limited to, siting, construction, modification, operation, and dismantlement and removal or otherwise proper closure (such as of a well) of characterization and monitoring devices, and siting, construction, and associated operation of a small-scale laboratory building or 
- **Extraordinary circumstances:** B. Conditions That Are Integral Elements of the Classes of Actions in Appendix B The classes of actions listed below include the following conditions as integral elements of the classes of actions. To
- **Candidate scope:** 80-80 MW (n=1); agencies: Bureau of Land Management
- **Coverage read:** **LIKELY COVERS** — DOE B3.1 (site characterization & environmental monitoring) covers exploratory drilling / geophysical survey — a close match to geothermal exploration.
- **Bound to verify:** B3.1 site characterization is qualitative; confirm exploratory drilling depth / well count is within DOE B3.1 scope.

### solar — 9 CE-shaped

- **Best-match CE:** `DOE-1--5-87` (DOE), match score 0.6747 — [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)
- **CE text:** B5.16 SOLAR PHOTOVOLTAIC SYSTEMS. (a) The installation, modification, operation, or decommissioning of commercially available solar photovoltaic systems: (1) Located on a building or other structure (such as rooftop, parking lot or facility, or mounted to signage, lighting, gates, or fences); or (2) Located within a previously disturbed or developed area. (b) Covered actions would be in accordance
- **Extraordinary circumstances:** B. Conditions That Are Integral Elements of the Classes of Actions in Appendix B The classes of actions listed below include the following conditions as integral elements of the classes of actions. To
- **Candidate scope:** line 4-6 mi (n=2); 4-10 MW (n=2); agencies: Bureau of Land Management; Department of Energy; National Nuclear Security Administration; Power Marketing Administration
- **Coverage read:** **PARTIAL — category contamination, re-examine first** — B5.16 covers solar PV generation; it does not cover gen-tie transmission lines. The match is valid only for the ~3 actual solar-generation candidates.
- **Bound to verify:** B5.16 covers commercially available solar PV SYSTEMS. But 6 of 9 CE-shaped 'solar' rows are GEN-TIE / interconnection TRANSMISSION lines (the gen-tie precedence rule). B5.16 does NOT cover gen-tie lines. Decide whether gen-ties belong in solar, transmission, or other BEFORE treating this match as adopt.

### wind_onshore — 7 CE-shaped

- **Best-match CE:** `DOE-1--5-89` (DOE), match score 0.4049 — [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)
- **CE text:** B5.18 WIND TURBINES. The installation, modification, operation, and removal of a small number (generally not more than 2) of commercially available wind turbines, with a total height generally less than 200 feet (measured from the ground to the maximum height of blade rotation) that (1) Are located within a previously disturbed or developed area; (2) are located more than 10 nautical miles (about 
- **Extraordinary circumstances:** B. Conditions That Are Integral Elements of the Classes of Actions in Appendix B The classes of actions listed below include the following conditions as integral elements of the classes of actions. To
- **Candidate scope:** 1-30 MW (n=3); agencies: Bureau of Land Management; Department of Energy; Energy Programs
- **Coverage read:** **CONDITIONAL — turbine-count bound** — DOE B5.18 covers <=2 commercially available wind turbines; only small wind adopts, larger wind farms are expand.
- **Bound to verify:** B5.18 caps at a SMALL NUMBER (generally <=2) of commercially available wind turbines. Candidates above ~2 turbines EXCEED it -> expand, not adopt. Confirm turbine counts.

### temporary_resource_assessment — 4 CE-shaped

- **Best-match CE:** `DOE-1--3-43` (DOE), match score 0.5405 — [source](https://www.ecfr.gov/current/title-10/chapter-X/part-1021)
- **CE text:** B3.1 SITE CHARACTERIZATION AND ENVIRONMENTAL MONITORING. Site characterization and environmental monitoring (including, but not limited to, siting, construction, modification, operation, and dismantlement and removal or otherwise proper closure (such as of a well) of characterization and monitoring devices, and siting, construction, and associated operation of a small-scale laboratory building or 
- **Extraordinary circumstances:** B. Conditions That Are Integral Elements of the Classes of Actions in Appendix B The classes of actions listed below include the following conditions as integral elements of the classes of actions. To
- **Candidate scope:** n/a; agencies: Bureau of Land Management; Department of Energy; Forest Service
- **Coverage read:** **LIKELY COVERS** — DOE B3.1 covers temporary site characterization & monitoring — a direct match to temporary resource assessment.
- **Bound to verify:** B3.1 site characterization is qualitative; confirm the met-tower / boring / survey scope is within it.

