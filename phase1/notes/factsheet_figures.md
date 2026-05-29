# Factsheet Figures: Notes and Client Q&A

## Background

A client asked two questions about figures 7 (department collaboration hubs) and 8 (department Sankey) in the phase 1 deliverable 4 report. This document summarizes the answers, the new figures built in response, and the methodological caveats that came out of the discussion.

---

## Q1: Does the analysis capture only NEPA roles, or also Section 7 ESA consultations, USACE Section 404 permits, and other non-NEPA statutory processes?

**Answer: Strictly NEPA roles only.**

The co-agency extraction (`extract_coagency_names.py`) detects agencies by scanning EIS/EA document text for explicit NEPA role labels: Lead Agency, Joint Lead Agency, Co-Lead Agency, Cooperating Agency, and Participating Agency. No statutory consultation language (Section 7, Section 404, etc.) is extracted unless the agency is also formally named in one of those NEPA roles in the document text.

USACE and USFWS appear frequently in the figures not because of their Section 404/Section 7 roles, but because they are routinely designated as NEPA cooperating or lead agencies in energy project EIS documents.

**Verified:** Zero non-NEPA roles exist in the extracted data.

---

## Q2: Can Army Corps of Engineers (USACE) be broken out separately from the rest of DOD?

**Answer: Yes — new figures were built.**

The original figures aggregated all agencies to the department level, placing USACE under "Department of Defense." Because `agency_normalized = "U.S. Army Corps of Engineers"` is preserved in `coagency_name_hits.parquet`, USACE can be split out by remapping its department before building pairs.

**New figures added to `phase1/output/factsheet/figures/`:**
- `fig_department_collaboration_hubs_corps.png` — hub figure with USACE as a separate node
- `fig_department_sankey_filtered_corps.png` — Sankey with USACE as a separate node

**USACE alias coverage confirmed:** All raw variants ("US Army Corps of Engineers", "USACE", "Army Corps of Engineers", "US ARMY CORPS OF ENGINEERS", "US Army CorPs of Engineers") normalize to `"U.S. Army Corps of Engineers"`. No USACE projects are missed.

**Code:** New figures are built inside `phase1/code/factsheet_figures.R`, in a dedicated section after the original Fig 7/8 block. The USACE remap applies to `coagency_name_hits_sankey` before pair-building, then follows the same lead-to-partner, joint-lead, and fallback logic as the original figures.

---

## Methodological Caveat: Tie Counts vs. Project Counts

The hub and Sankey figures use a **tie-count metric** (number of pair appearances), not a unique project count. These differ because a single project can generate multiple pairs — e.g., a project with DOE (lead), DOI (cooperating), and DOD (cooperating) creates two DOD pairs: (DOE, DOD) and (DOI, DOD), giving DOD +2 ties from 1 project.

**Example for DOD:** 104 unique projects with any DOD entity → 124 collaborative ties in the hub table. The gap reflects projects where DOD is paired with more than one other department simultaneously.

### Double-counting in the Corps breakout figures

Splitting USACE out of DOD introduces an additional inflation. Of the 104 EIS projects involving any DOD entity, **69 (66%)** include both USACE and at least one other DOD component (e.g., Air Force, Army). In the original figure, each of those 69 projects contributed 1 DOD tie. In the Corps breakout figure, each contributes 2 ties — one toward USACE and one toward DOD-remaining. This is why USACE (111) + DOD-remaining (118) = 229, far exceeding the original combined DOD count of 124.

**The Corps breakout figures correctly show USACE as a major cooperative actor, but the tie counts are not comparable between the original and the Corps breakout versions.**

### Why bridge scores are higher in the Corps breakout figure

Bridge score = unique partner departments × log(1 + total shared project ties). Splitting USACE out of DOD causes both inputs to increase for most departments: a department that previously had "DOD" as a single partner now has two distinct partners ("USACE" and "DOD-remaining"), and its tie count rises from the same double-counting inflation described above. The higher bridge scores in the Corps breakout figure are therefore an artifact of the split, not evidence of genuinely more collaboration. The scores within the Corps figure are internally consistent and valid for ranking departments relative to each other, but should not be compared against scores from the original combined-DOD figure.

### Project-level breakdown (the cleaner summary)

| Group | Projects |
|---|---|
| USACE only (no other DOD) | 13 |
| USACE + at least one other DOD component | 69 |
| Other DOD only (no USACE) | 22 |
| **Any DOD entity (total)** | **104** |

USACE is present in **82 of 104** DOD-involved EIS reviews, reflecting its frequent role as a NEPA cooperating agency on energy projects.

---

## Suggested language for analysis write-up

> Of the 104 EIS projects involving any DOD entity, 13 involve USACE alone, 69 involve USACE alongside at least one other DOD component (such as the Air Force or Army), and 22 involve a non-USACE DOD component with no Corps involvement. USACE is present in 82 of 104 DOD-involved EIS reviews, reflecting its frequent role as a NEPA cooperating agency.
>
> The hub and Sankey figures count agency pair appearances rather than unique projects. Projects with both USACE and another DOD component generate two pairs in the Corps breakout figures — one for USACE and one for DOD-remaining — so the tie counts for these nodes cannot be summed and compared against the original combined DOD figure.
