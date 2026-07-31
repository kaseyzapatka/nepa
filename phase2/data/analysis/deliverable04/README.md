# Deliverable 4 (Timelines) — data locations

The D4 timeline pipeline's primary outputs live in `../timeline/`, not here, because they are
shared infrastructure consumed by other deliverables (D3, D5, and D6 all read them — e.g. D5
anchors its CE-spike analysis on `decision_date`). The headline file is
`../timeline/timeline_project_dates.parquet` (one row per project: initiation/decision dates,
granularity, sources, `duration_days`, `timeline_status`); candidates, document index, context
packets, adjudication caches, gold sets, and models are alongside it.

This folder holds only D4-specific side tables:

- `blm_field_offices.parquet` / `doe_offices.parquet` — office lookups for the field-office
  duration breakdowns
- `projects_page_counts.parquet` — regulatory page counts for the FRA page-limit analysis

See `phase2/runbooks/deliverables/deliverable04.md` and
`phase2/architecture/deliverables/deliverable04.md` for the full pipeline.
