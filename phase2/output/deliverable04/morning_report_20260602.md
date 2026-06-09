# D4 Timeline Pipeline — Morning Report
**Run date:** 2026-06-01 night → 2026-06-02  
**Report generated:** 2026-06-02 02:05 PDT (automated monitoring)

---

## Pipeline Outcome: COMPLETED SUCCESSFULLY ✓

| Milestone | Time (PDT) |
|---|---|
| Pipeline launched | 22:49, Jun 1 |
| CE complete (5 shards) | ~23:30 |
| EA complete (5 shards) | ~23:58 |
| EIS complete (5 shards) | ~01:39 |
| Final 05_select_dates.py pass | 01:39–02:02 |
| **[DONE]** | **02:02, Jun 2** |

Total runtime: **3h 13m**. Exit code 0. No errors, no watchdog restarts.

---

## Issues Encountered

**None.** The pipeline ran cleanly from start to finish. The only expected warning that appeared on every shard:

```
WARNING: no trained model found — passing through with neutral scores.
Train with --train once real gold labels exist.
```

This is by design — `04_classify_candidates.py` has no trained model yet and passes through with `classifier_label="unscored"`. See "What's left for today" below.

---

## Coverage by Process Type

| process_type | n_projects | has_init | has_decision | pct_init | pct_decision |
|---|---|---|---|---|---|
| CE | 52,493 | 13,977 | 51,275 | **26.6%** | 97.7% |
| EA | 2,994 | 2,316 | 2,741 | **77.4%** | 91.5% |
| EIS | 3,492 | 2,599 | 1,988 | **74.4%** | 56.9% |
| **Total** | **58,979** | **18,892** | **56,004** | **32.0%** | **94.9%** |

**Notes:**
- **CE initiation at 26.6%** is the known bottleneck (template-specific form layouts, documented in CLAUDE.md as "~30%"). Not a new problem — this is what classifier training tomorrow is meant to address.
- **CE decision at 97.7%** is high because CE decisions are mostly pulled from metadata (agency action dates on the form), not document text.
- **EIS decision at 56.9%** is lower because many EIS projects have a DEIS in the index but no FEIS/ROD yet. The pipeline correctly flagged 448 EIS projects as `deis_only`.
- **EA coverage is strong** — 77.4% initiation, 91.5% decision.

---

## Timeline Status Distribution

| timeline_status | n | pct |
|---|---|---|
| missing_initiation | 39,441 | 66.9% |
| complete_clear | 11,902 | 20.2% |
| complete_with_proxy | 4,542 | 7.7% |
| missing_decision | 2,329 | 3.9% |
| missing_both | 646 | 1.1% |
| manual_review | 119 | 0.2% |

**Status by process type:**

| process_type | timeline_status | n |
|---|---|---|
| CE | missing_initiation | 38,351 |
| CE | complete_clear | 9,473 |
| CE | complete_with_proxy | 3,362 |
| CE | missing_decision | 1,053 |
| CE | missing_both | 165 |
| CE | manual_review | 89 |
| EA | complete_clear | 1,652 |
| EA | missing_initiation | 627 |
| EA | complete_with_proxy | 454 |
| EA | missing_decision | 202 |
| EA | missing_both | 51 |
| EA | manual_review | 8 |
| EIS | missing_decision | 1,074 |
| EIS | complete_clear | 777 |
| EIS | complete_with_proxy | 726 |
| EIS | missing_initiation | 463 |
| EIS | missing_both | 430 |
| EIS | manual_review | 22 |

The `missing_initiation` dominance (66.9% overall) is almost entirely CE — 38,351 of 39,441 missing-initiation rows are CE. This confirms CE initiation is the single largest coverage gap.

---

## Duration Statistics (projects with both dates, complete_clear)

| process_type | n | avg_yrs | median_yrs | min_days | max_days |
|---|---|---|---|---|---|
| CE | 8,769 | 0.4 | 0.1 | 1 | 16,682 |
| EA | 1,455 | 0.7 | 0.2 | 1 | 8,300 |
| EIS | 475 | 3.5 | 2.3 | 2 | 13,173 |

**Flag for review:** CE has a max of 16,682 days (~45 years) and a median of only 0.1 years (~37 days). The very short CE median is consistent with CE being a quick categorical action, but the long tail warrants a spot-check — some CE initiation dates may be matching historical references rather than the actual project start. The classifier training tomorrow should help suppress these. Add to manual review queue if desired.

EIS median of 2.3 years (840 days) is consistent with NEPA literature on full EIS timelines.

---

## Sample Projects (deterministic selection)

### CE (5 samples)
| project_id | timeline_status | initiation_date | init_source | decision_date | decision_source |
|---|---|---|---|---|---|
| 0d094e68-987d-3b74-7da1-9df825da2d2a | missing_initiation | — | — | 2011-12-16 | metadata |
| 1f2a60bc-ddaa-91a9-25ad-377d1425eb72 | missing_initiation | — | — | 2016-07-22 | metadata |
| 593bf81f-c92b-749a-304d-5d7a4902075c | missing_initiation | — | — | 2016-06-23 | metadata |
| 8869b457-3012-4ece-ebe3-b9e23b9c183f | missing_initiation | — | — | 2024-01-12 | metadata |
| a893289d-3654-a230-c97a-addb5d4e6b79 | missing_initiation | — | — | 2017-06-19 | document_text |

All 5 CE samples are `missing_initiation` — consistent with the 73% missing rate for CE.

### EA (5 samples)
| project_id | timeline_status | initiation_date | init_source | decision_date | decision_source | duration_days |
|---|---|---|---|---|---|---|
| 158a2133bb5a59dcbba3e4f5cb29056e | missing_decision | 2000-09-15 | document_text | — | — | — |
| 1c5751a59c1ec8d20af11c6923c44fda | complete_clear | 2023-11-09 | metadata | 2023-11-17 | metadata | 8 |
| 485fb4ea7346f02675be81c09e056a86 | complete_clear | 2014-01-10 | document_text | 2014-02-28 | metadata | 49 |
| b1df3c4857cad40dbbcdbd76aec61194 | complete_clear | 2019-05-06 | metadata | 2019-08-14 | metadata | 100 |
| d8ee6e2675bac9b6fcdb2138b46ab0a5 | complete_clear | 2015-10-30 | metadata | 2016-04-10 | metadata | 163 |

EA sample looks reasonable — durations of 8–163 days for complete EAs are plausible.

### EIS (5 samples)
| project_id | timeline_status | initiation_date | init_source | decision_date | decision_source |
|---|---|---|---|---|---|
| 374d260d936c79612c8539569fabdbb9 | missing_both | — | — | — | — |
| 484cb0cafc575453bed1aeeeb7b50578 | missing_decision | 2022-03-15 | document_text | — | — |
| 5f53b1ef57f79879c3e20fbcde9958f4 | missing_both | — | — | — | — |
| 62c0f37c9da12e8538938464a96216dd | missing_initiation | — | — | 2012-02-24 | document_text |
| 7e5b7e396b4cd54aa895413d7a9e2623 | complete_with_proxy | 2000-02-15 | document_text | 2002-06-28 | document_text |

Two `missing_both` in this sample — consistent with the 430 EIS projects in that bucket, many of which are likely early-stage DEIS-only documents.

---

## Output Files

| File | Rows | Notes |
|---|---|---|
| `phase2/data/analysis/timeline/timeline_project_dates.parquet` | 58,979 | Final output |
| `phase2/data/analysis/timeline/timeline_candidates.parquet` | 445,245 | All candidates, classifier_label="unscored" |
| `phase2/output/deliverable04/timeline_manual_review_queue.csv` | 15,820 | Projects flagged for human review |
| `phase2/output/deliverable04/overnight_run_20260601_2249.log` | — | Full pipeline log |

---

## What's Left for Today

1. **Classifier training (04_classify_candidates.py --train)**
   - Build human gold labels for `timeline_gold_candidate_training.parquet` first (the current file contains only regex echoes — training on those would not help)
   - Start with ~50–100 labeled examples per head (initiation / decision) — SetFit is designed for this regime
   - See `phase2/architecture/deliverables/deliverable04.md` for the labeling strategy

2. **Re-run 04 scoring pass** after training:
   ```
   python phase2/code/deliverable04/04_classify_candidates.py
   ```
   Then re-run `05_select_dates.py` to get classifier-informed date selection.

3. **Spot-check CE duration outliers** — review projects with `duration_days > 5000` in CE. May indicate initiation date extraction picking up historical references.

4. **Review manual_review_queue.csv** (15,820 projects) — prioritize EIS `missing_both` (430 projects) as highest-value targets for the labeling pass.

---

*Monitoring completed. No human intervention was required overnight.*
