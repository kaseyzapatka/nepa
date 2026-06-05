# D4 — Project-level gold labeling (ranker training + end-to-end validation)

**Goal.** For ~300 projects, record the **true final initiation candidate and decision candidate**
(by `candidate_id`). This single pass does **double duty**:

1. **Training data for the LightGBM selection ranker (`05b_rank.py`)** — the ranker learns "which
   candidate is THE initiation / THE decision," which needs a project-level target (the true
   candidate per project). Candidate-level init/dec/neither labels can't teach this.
2. **The missing `07_validate` end-to-end gold** (`timeline_gold_projects.parquet`) — it does not
   exist yet, so the pipeline currently has no end-to-end accuracy check. The same picks, converted to
   dates, fill that gap.

This is **much smaller** than the candidate build-out (~300 projects, not thousands of candidates),
but each project requires *judgment*: read the project's candidate options and pick the right ones.
**Aim for 300; 400–500 is better** (LightGBM trains a few hundred groups — more groups = a sturdier
ranker). It is the intensive judgment task, so pace it; it is fully resumable.

**Prerequisites are already DONE** (as of 2026-06-05): the candidate build-out, the SetFit retrain
to `test_v2` (frozen-test F1 init 0.896 / decision 0.892), and `04b --apply` have all run — so the
pool already carries trustworthy **calibrated** classifier probs (`p_init_cal` / `p_dec_cal`). The
ranker script (`05b_rank.py`) is **already written**; this gold pass produces the data it trains on.
Just start at Step 1.

---

## What "the true candidate" means (selection conventions)

Pick, per project, the `candidate_id` that should be the final date — using the **same conventions
`05` aims to follow** (and the definitions in `labeling_rules.md`):

- **Initiation** = the **earliest qualifying start signal**: NOI publication, application/ROW/permit
  received, scoping opened, FERC pre-filing approved, DOE *Initiator* signature. If two valid scoping
  starts exist, pick the **earlier**.
- **Decision** = the **operative determination**, preferring a **precise day-level** date:
  - CE → the NEPA Compliance Officer / Field Manager **signature**, or the **CX cover month** if that's
    all there is (the CX *is* the determination).
  - EA/EIS → the **ROD/FONSI date** (day-level). Do **not** pick a bare EA/EIS cover month — if only a
    month exists, set decision to `none` (the project genuinely lacks a precise decision date).
- **`none`** = no candidate in the list is the true date (extraction missed it, or it doesn't exist).
  Recording `none` is valuable — it measures coverage gaps for `07`.

Pick **only from the candidate list shown** (the ranker ranks existing candidates). If you know the
true date but it isn't among the candidates, put it in `notes` and set the id to `none`.

---

## Hard rules

1. **Append to one CSV** — `phase2/output/deliverable04/project_gold_sample.csv` is the single source
   of truth. Only fill blank rows; never overwrite a filled pick.
2. **Pick by `candidate_id`** from the options view; dates are derived for you in Step 3.
3. **Dedup by `project_id`**; never relabel a project already done.
4. Every project gets a one-line `notes` (which candidate and why, per the conventions above).
5. Resumable — apply in chunks; guard-on-blank makes re-runs idempotent.

Files:
- Picks (sole source): `phase2/output/deliverable04/project_gold_sample.csv`
- Options view (read-only aid): `phase2/output/deliverable04/project_gold_options.txt`
- Candidate pool (read-only): `phase2/data/analysis/timeline/timeline_candidates.parquet`
- Codebook: `phase2/notes/deliverable04/labeling_rules.md`
- End-to-end gold output (Step 5): `phase2/data/analysis/timeline/gold/timeline_gold_projects.parquet`

Every command runs with the `nepa` env (`CONDA_DEFAULT_ENV=nepa`).

---

## Step 1 — Emit the project sample + options view

Samples ~300 projects (stratified ~100 each CE/EA/EIS, requiring ≥2 candidates), writes blank pick
rows to `project_gold_sample.csv`, and writes a per-project options file to choose from.

```python
# phase2/code/deliverable04/_emit_project_gold.py   (create + run once)
import os
if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("activate nepa")
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAND = ROOT / "phase2/data/analysis/timeline/timeline_candidates.parquet"
SAMPLE = ROOT / "phase2/output/deliverable04/project_gold_sample.csv"
OPTIONS = ROOT / "phase2/output/deliverable04/project_gold_options.txt"
PER_PROCESS = 100
SEED = 42

df = pd.read_parquet(CAND)
counts = df.groupby("project_id").size()
ok = counts[counts >= 2].index
pool = df[df["project_id"].isin(ok)]

already = set()
if SAMPLE.exists():
    already = set(pd.read_csv(SAMPLE)["project_id"])

picks = []
for proc, grp in pool.groupby("process_type"):
    pids = pd.Series(grp["project_id"].unique())
    pids = pids[~pids.isin(already)]
    take = pids.sample(min(PER_PROCESS, len(pids)), random_state=SEED)
    picks += [(p, proc) for p in take]

rows = pd.DataFrame(picks, columns=["project_id", "process_type"])
rows["initiation_candidate_id"] = ""   # fill: candidate_id | none
rows["decision_candidate_id"] = ""     # fill: candidate_id | none
rows["notes"] = ""
rows["split"] = ""                      # assigned at ranker train time
header = not SAMPLE.exists()
rows.to_csv(SAMPLE, mode="a", header=header, index=False)

# Options view: per project, list its candidates so the labeler can choose.
def excerpt(ctx):
    ctx = str(ctx or ""); i = ctx.find("[[")
    return (("..." if i > 120 else "") + ctx[max(0, i-120):i+160] + "...") if i >= 0 else ctx[:260]

with open(OPTIONS, "a") as f:
    for pid, proc in picks:
        c = df[df["project_id"] == pid].sort_values("parsed_date")
        f.write(f"\n===== project {pid} [{proc}] — {len(c)} candidates =====\n")
        for _, r in c.iterrows():
            # prefer calibrated probs (written by 04b --apply); fall back to raw
            pi = r.get("p_init_cal", r.get("p_initiation")); pdc = r.get("p_dec_cal", r.get("p_decision"))
            f.write(f"  id={r['candidate_id']} date={r.get('parsed_date')} "
                    f"gran={r.get('date_granularity')} role={r.get('candidate_role')} "
                    f"p_init={float(pi or 0):.2f} p_dec={float(pdc or 0):.2f}\n")
            f.write(f"     {excerpt(r.get('model_context'))}\n")
print(f"Emitted {len(picks)} projects -> {SAMPLE.name} (options -> {OPTIONS.name}).")
```

```bash
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/_emit_project_gold.py
```

---

## Step 2 — Choose the true candidates

For each project in `project_gold_options.txt`, read its candidate list and pick:
- `initiation_candidate_id` — the id of the true initiation date (earliest qualifying start), or `none`.
- `decision_candidate_id` — the id of the true decision date (precise day preferred; CE cover month
  ok; EA/EIS month → `none`), or `none`.

The options view shows each candidate's `p_init` / `p_dec` (classifier signal) as a hint — but the
**conventions above and `labeling_rules.md` decide**, not the probabilities. Work in chunks of ~50.

---

## Step 3 — Apply picks into `project_gold_sample.csv` (dates derived for you)

Write an apply script (same guard-on-blank pattern as the candidate build-out) that sets the two ids
from a `PICKS` list and **derives the dates/granularity** from the candidate pool by `candidate_id`:

```python
# phase2/output/deliverable04/apply_project_gold_<chunk>.py
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
PICKS = [
    {"project_id": "....", "initiation_candidate_id": "....",
     "decision_candidate_id": "none", "notes": "init=earliest scoping NOI; no precise ROD date"},
    # ... one per project in this chunk ...
]
sample = ROOT / "phase2/output/deliverable04/project_gold_sample.csv"
cand = pd.read_parquet(ROOT / "phase2/data/analysis/timeline/timeline_candidates.parquet")
cmap = cand.set_index("candidate_id")[["parsed_date", "date_granularity"]].to_dict("index")

df = pd.read_csv(sample, dtype=str, keep_default_na=False)
pk = pd.DataFrame(PICKS)
m = df.merge(pk, on="project_id", how="left", suffixes=("", "_new"))
blank = m["initiation_candidate_id"].astype(str).str.strip().eq("")
has = m["initiation_candidate_id_new"].notna()
apply = blank & has
for col in ["initiation_candidate_id", "decision_candidate_id", "notes"]:
    m.loc[apply, col] = m.loc[apply, f"{col}_new"]
m[df.columns].to_csv(sample, index=False)
print(f"Applied {int(apply.sum())} project picks to {sample.name}")
```

```bash
CONDA_DEFAULT_ENV=nepa python phase2/output/deliverable04/apply_project_gold_<chunk>.py
```

---

## Step 4 — Verify

```python
import pandas as pd
df = pd.read_csv("phase2/output/deliverable04/project_gold_sample.csv")
print("projects:", len(df), "| filled:",
      int((df["initiation_candidate_id"].fillna('').str.strip() != "").sum()))
print(df["process_type"].value_counts().to_dict())
print("init=none:", int((df["initiation_candidate_id"] == "none").sum()),
      "| dec=none:", int((df["decision_candidate_id"] == "none").sum()))
```

Target: ~300 projects filled, balanced across CE/EA/EIS, with a realistic share of `none`
(coverage gaps are expected and informative).

---

## Step 5 — Build the `07` end-to-end gold parquet

Convert the picks to dates and write the gold parquet `07_validate.py` expects
(`data/analysis/timeline/gold/timeline_gold_projects.parquet`, columns
`project_id, process_type, gold_initiation_date, gold_initiation_granularity, gold_decision_date,
gold_decision_granularity`):

```python
# phase2/code/deliverable04/_build_project_gold_parquet.py   (run once, after labeling)
import os
if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("activate nepa")
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sample = pd.read_csv(ROOT / "phase2/output/deliverable04/project_gold_sample.csv", dtype=str,
                     keep_default_na=False)
cand = pd.read_parquet(ROOT / "phase2/data/analysis/timeline/timeline_candidates.parquet")
cmap = cand.set_index("candidate_id")[["parsed_date", "date_granularity"]].to_dict("index")

def lookup(cid):
    r = cmap.get(str(cid))
    return (r["parsed_date"], r["date_granularity"]) if r else (None, None)

sample = sample[sample["initiation_candidate_id"].str.strip() != ""]  # filled only
gi = sample["initiation_candidate_id"].map(lambda x: lookup(x) if x != "none" else (None, None))
gd = sample["decision_candidate_id"].map(lambda x: lookup(x) if x != "none" else (None, None))
gold = pd.DataFrame({
    "project_id": sample["project_id"],
    "process_type": sample["process_type"],
    "gold_initiation_date": [a for a, _ in gi],
    "gold_initiation_granularity": [b for _, b in gi],
    "gold_decision_date": [a for a, _ in gd],
    "gold_decision_granularity": [b for _, b in gd],
})
out = ROOT / "phase2/data/analysis/timeline/gold/timeline_gold_projects.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
gold.to_parquet(out, index=False)
print(f"Wrote {len(gold)} gold projects -> {out}")
```

```bash
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/_build_project_gold_parquet.py
```

That parquet is the input to `07_validate.py` (end-to-end accuracy). The **`project_gold_sample.csv`**
itself (the per-project candidate_ids) is what the **already-written** `05b_rank.py` trains on.

---

## Step 6 — Train the LightGBM ranker (after gold is labeled)

`05b_rank.py` is already written. Once `project_gold_sample.csv` has ~300+ filled rows:

```bash
pip install lightgbm                      # one-time (not yet in the nepa env)
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/05b_rank.py --train   # fits init + decision rankers, reports held-out top-1/MRR
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/05b_rank.py --eval    # held-out top-1 accuracy per head/process
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/05b_rank.py --apply   # writes learned_init_score / learned_decision_score to the pool
```

`05b` auto-assigns a stratified 80/20 train/test split over the gold projects (seed 42), trains one
lambdarank model per head on the full feature set (calibrated probs + every structural signal), and
measures **top-1 accuracy** (does the ranker's #1 candidate == the gold pick?). After `--apply`, wire
`05` to prefer the learned scores (the exact 4-line hook is in `05b_rank.py`'s docstring).

---

## Notes / guardrails

- Do this **after** the candidate build-out + SetFit retrain + `04b --apply`, so the `p_init`/`p_dec`
  hints in the options view are trustworthy.
- ~300 projects is the floor for a first ranker (LightGBM needs a few hundred groups; it can't
  few-shot). More is better; this is the minimum that makes `05b` worth training.
- `none` is a valid, useful answer — never force a pick when no candidate is the true date.
- Picks are by `candidate_id` only; do not edit pipeline scripts. New scripts here are
  `_emit_project_gold.py`, `apply_project_gold_<chunk>.py`, `_build_project_gold_parquet.py`.
- Keep a held-out share for the ranker's own eval — leave `split` blank; we'll assign it at
  `05b` train time (stratified by process), so the ranker is measured on projects it never trained on.
