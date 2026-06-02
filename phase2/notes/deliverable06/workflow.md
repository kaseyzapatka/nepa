# D6 FONSI Opportunity Review Workflow

## Purpose

The D6 Stage A outputs are review materials for deciding which recurring
patterns in prior EAs and FONSIs deserve deeper Stage B analysis. They are not
proposed categorical exclusions (CEs), legal conclusions, or findings of legal
sufficiency.

The analysis is designed to help reviewers identify:

1. recurring action archetypes;
2. repeated scale limits and bounding conditions;
3. dependence on mitigation, best management practices (BMPs), monitoring, or
   permit conditions;
4. existing same-agency or other-agency CEs that may already address similar
   actions;
5. base rates across CE, EA, FONSI, and EIS records; and
6. representative source evidence for targeted follow-up.

CE Explorer similarity scores are retrieval aids only. A similarity score
helps locate potentially relevant agency materials for manual review. It does
not establish that an existing CE covers an archetype.

---

## Recommended Reading Order

1. Open
   [`fonsi_opportunity_scan.html`](../../output/deliverable06/fonsi_opportunity_scan.html)
   for a compact overview of every archetype.
2. Open
   [`fonsi_candidate_shortlist.html`](../../output/deliverable06/fonsi_candidate_shortlist.html)
   for the current four-category heuristic shortlist.
3. Read the individual dossiers for the categories that appear most promising.
4. Use
   [`fonsi_opportunity_matrix.csv`](../../output/deliverable06/fonsi_opportunity_matrix.csv)
   for the complete analytical fields behind the HTML summaries.
5. Review the QA CSVs before treating any extracted pattern as a Stage B
   candidate.

---

## Opportunity Outputs

### FONSI opportunity scan

[`fonsi_opportunity_scan.html`](../../output/deliverable06/fonsi_opportunity_scan.html)
is the compact summary table for all action archetypes.

Use it to compare:

- the number of classified projects and FONSI projects;
- CE, EA, FONSI, and EIS composition shares;
- mitigation, BMP, and monitoring dependence;
- heuristic assessment totals;
- ranked existing-CE retrieval results; and
- open review gates.

`eis_share` is a composition measure, not an escalation rate. An EIS may
represent a larger or more complex action rather than an escalation of the
same action.

### CATF candidate shortlist

[`fonsi_candidate_shortlist.html`](../../output/deliverable06/fonsi_candidate_shortlist.html)
links the four highest-priority candidate dossiers from the current heuristic
scoring pass:

1. Facility upgrades
2. Electricity transmission
3. Site characterization
4. Hydropower

The shortlist is a starting point for review. It should change after manual
QA, taxonomy refinement, and authoritative CE verification.

### Full opportunity matrix

[`fonsi_opportunity_matrix.csv`](../../output/deliverable06/fonsi_opportunity_matrix.csv)
is the complete analytical table behind the HTML outputs.

It contains:

- archetype labels and descriptions;
- candidate versus comparison-diagnostic scope;
- recommendation tier and gating flags;
- project counts and process-type base rates;
- CE, EA, and EIS classification coverage;
- assignment-method distributions;
- repeated limitations and extracted scale thresholds;
- mitigation, BMP, and monitoring dependence;
- strongest same-agency and other-agency CE retrieval results;
- representative project IDs;
- five reviewer-facing `0-2` assessments; and
- QA notes and version fields.

The heuristic tiers mean:

| Tier | Meaning |
|---|---|
| `advance` | Strong evidence and no unresolved gate. No archetype receives this tier automatically before manual review. |
| `review` | Potentially useful pattern that needs targeted follow-up. |
| `deprioritize` | Weak fit, comparison-only archetype, already-covered category after manual verification, or a pattern blocked by a material gate. |

The assessment total is not a probability or legal score.

---

## What Each Dossier Contains

Every dossier follows the same structure:

1. candidate summary and heuristic tier;
2. recurring action definition;
3. representative FONSI projects;
4. extracted project scales and limitations;
5. mitigation, BMP, monitoring, and permit-condition evidence;
6. ranked existing-CE retrieval results; and
7. unresolved review gates.

The dossiers are deliberately broad Stage A review packets. Reviewers should
split or narrow archetypes when the evidence combines materially different
actions.

---

## Candidate Dossiers

### Facility upgrades

[`facility_upgrade.html`](../../output/deliverable06/dossiers/facility_upgrade.html)

This dossier covers modifications, replacement, maintenance, or upgrades to
existing facilities. It is useful for testing whether bounded work at
existing sites repeats often enough to support a narrow category.

### Electricity transmission

[`electricity_transmission.html`](../../output/deliverable06/dossiers/electricity_transmission.html)

This dossier covers transmission lines, substations, interconnections, and
rights-of-way. Review recurring limits such as length, voltage, disturbed
acreage, and access-road constraints.

### Site characterization

[`site_characterization.html`](../../output/deliverable06/dossiers/site_characterization.html)

This dossier covers surveys, geotechnical work, sampling, and test
activities. It is useful for identifying low-disturbance preliminary actions.

### Hydropower

[`hydropower.html`](../../output/deliverable06/dossiers/hydropower.html)

This dossier covers hydroelectric generation, dams, conduits, and related
actions. Review carefully because the underlying project types and impacts
may vary substantially.

### Solar energy

[`solar_energy.html`](../../output/deliverable06/dossiers/solar_energy.html)

This dossier covers photovoltaic and solar-thermal actions. Use it to inspect
whether recurring patterns exist below clear acreage or facility-modification
thresholds.

### Energy storage

[`energy_storage.html`](../../output/deliverable06/dossiers/energy_storage.html)

This dossier covers battery, pumped-storage, and related projects. It will
likely need subdivision because storage technologies differ materially.

### Wind energy

[`wind_energy.html`](../../output/deliverable06/dossiers/wind_energy.html)

This dossier covers wind facilities and turbines. Review scale, wildlife, and
visual-resource conditions.

### Geothermal exploration

[`geothermal_exploration.html`](../../output/deliverable06/dossiers/geothermal_exploration.html)

This dossier covers exploration drilling, testing, and temporary access. It
is particularly useful for comparing extracted historical patterns against
BLM's newer bounded geothermal-exploration CE.

### Geothermal development

[`geothermal_development.html`](../../output/deliverable06/dossiers/geothermal_development.html)

This dossier covers broader geothermal development and operational actions.
It is separate from geothermal exploration because the disturbance profile
differs.

### Pipeline

[`pipeline.html`](../../output/deliverable06/dossiers/pipeline.html)

This dossier covers construction, replacement, repair, and right-of-way
actions. Review whether maintenance and new construction need separate
archetypes.

### Hydrogen

[`hydrogen.html`](../../output/deliverable06/dossiers/hydrogen.html)

This dossier covers hydrogen production, storage, and transport projects.
Evidence volume is smaller, so the current result is exploratory.

### Carbon management

[`carbon_management.html`](../../output/deliverable06/dossiers/carbon_management.html)

This dossier covers carbon capture, pipelines, sequestration, and related
actions. It is exploratory and likely too heterogeneous without refinement.

---

## Comparison-Only Dossiers

These dossiers remain visible for diagnostics but cannot advance as
clean-energy CE candidates.

### Oil and gas

[`oil_gas.html`](../../output/deliverable06/dossiers/oil_gas.html)

This is a comparison corpus for testing whether an apparent clean-energy
pattern is actually a broader land-management pattern. Incidental oil-and-gas
assignments within the clean-energy comparison universe should also be
reviewed as a taxonomy QA issue.

### Other clean energy

[`other_clean_energy.html`](../../output/deliverable06/dossiers/other_clean_energy.html)

This is a catch-all queue for projects that were not assigned cleanly to a
specific archetype. Use it to find taxonomy gaps and possible new categories.

---

## QA Review Files

### Document-role review

[`fonsi_document_role_review.csv`](../../output/deliverable06/fonsi_document_role_review.csv)

Use this file to validate which records are:

- standalone FONSIs;
- combined EA/FONSI documents;
- FONSI decision notices;
- drafts;
- attachments or appendices; or
- uncertain records.

Also verify whether the selected canonical FONSI is the correct project-level
document. Inventory and canonical-selection QA should be completed before
Stage B.

### Archetype review

[`fonsi_archetype_review.csv`](../../output/deliverable06/fonsi_archetype_review.csv)

Use this file to inspect whether representative projects were assigned to the
correct action categories. Look for false merges, false splits, incidental
keyword matches, and categories that need subdivision.

### CE crosswalk review

[`ce_crosswalk_review.csv`](../../output/deliverable06/ce_crosswalk_review.csv)

Use this file to verify CE Explorer retrieval results against authoritative
agency procedures. The crosswalk stores lexical and embedding similarity to
rank review candidates. A reviewer must confirm whether a CE is actually
relevant and whether it represents:

- same-agency existing coverage;
- an other-agency adoption opportunity;
- a possible expansion opportunity; or
- an uncertain retrieval result.

### Extraction review

[`fonsi_extraction_review.csv`](../../output/deliverable06/fonsi_extraction_review.csv)

Use this file to spot-check extracted action descriptions, condition counts,
scale fields, and access-road constraints. It is especially important to
review projects with unusually high or low condition counts.

### Packet review

[`fonsi_packet_review.csv`](../../output/deliverable06/fonsi_packet_review.csv)

Use this file to inspect the project-level evidence packets before relying on
downstream extraction. Check whether the selected sections describe the
action, finding rationale, resource areas, conditions, and boundaries rather
than generic boilerplate.

### Section-parser QA

[`fonsi_document_sections_qa.csv`](../../output/deliverable06/fonsi_document_sections_qa.csv)

Use this file to spot-check the targeted section parser. Confirm that detected
headings and section spans are real document structures rather than table
rows, headers, footers, or OCR artifacts.

### Topic-model diagnostics

[`fonsi_topic_diagnostics.csv`](../../output/deliverable06/fonsi_topic_diagnostics.csv)

This file contains the optional TF-IDF/NMF diagnostic. Use it to identify
missed archetypes and representative projects. Topic modeling should suggest
taxonomy revisions; it should not drive CE selection directly.

---

## Review Gates Before Stage B

Before selecting candidate categories for deeper substantiation:

1. Review and import at least 100 stratified document-role and canonical
   selection labels.
2. Review seed-taxonomy false merges and splits, including incidental
   `oil_gas` assignments.
3. Benchmark condition-role precision and recall. The deterministic extractor
   intentionally retains an `uncertain` queue for manual review or bounded
   LLM classification.
4. Verify shortlisted CE Explorer matches against canonical agency materials.
5. Select 2-4 candidate categories for Stage B substantiation and matched
   boundary-case review.

The durable pipeline architecture and rerun instructions are documented in
[`phase2/architecture/deliverables/deliverable06.md`](../../architecture/deliverables/deliverable06.md).

