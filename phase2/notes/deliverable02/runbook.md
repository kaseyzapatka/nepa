# D2 runbook — significance determinations

> **SUPERSEDED (2026-07-15).** This runbook described the pre-build state of the pipeline
> (plan v2.5, scripts 00–01 only). The full pipeline — FONSI + EIS extraction, gold sets,
> validation, analysis, and report — has since been built and run. Current documentation:
>
> - **Pipeline architecture + results:** `phase2/architecture/deliverables/deliverable02.md`
> - **Design decisions (single source of truth):** `phase2/plans/deliverable02.md`
> - **Dataset schemas:** `phase2/notes/deliverable02/data_dictionary.md`
> - **Reproduction commands:** the *Reproducibility* section of `phase2/reports/deliverable02.qmd`

## Calibration finding (retained for provenance)

BLM/DOE FONSIs **do not** use the CEQA-style phrase "less than significant with mitigation"
(0 hits). They use **"would be significant [absent mitigation]"** and **"with incorporation of
… mitigation"**. The Gate-1 screen (`MITIGATED_SCREEN_CUES`) is recall-oriented; the human
prunes. The precise extractor cue (`DETERMINATION_CUES['explicit_mitigated_lts']`) is separate.
