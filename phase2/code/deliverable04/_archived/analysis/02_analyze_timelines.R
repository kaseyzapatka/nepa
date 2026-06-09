# --------------------------
# DELIVERABLE 4: TIMELINES — Segmentation Analysis
# --------------------------
# Duration by CEQ regulation period, agency, project type, pre/post-FRA;
# outlier identification for case study packaging.
#
# [NEEDS TIMELINE]
#
# Initiation date merge hierarchy:
#   noi_publication_date (wins when present — authoritative FR record)
#   → bert_initiation_date
#
# Decision date:
#   bert_decision_date → llm_decision_date (EA/EIS adjudication)
#
# Inputs:
#   - data/analysis/_timeline/timeline_ce.parquet
#   - data/analysis/_timeline/timeline_ea.parquet
#   - data/analysis/_timeline/timeline_eis.parquet
#   - data/analysis/noi_federal_register.parquet
#   - data/analysis/projects_combined.parquet
#
# TODO: Implement — see phase2/plans/deliverable04.md for full specification
