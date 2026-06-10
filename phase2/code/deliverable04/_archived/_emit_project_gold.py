import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("activate nepa")

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
CAND = ROOT / "phase2/data/analysis/timeline/timeline_candidates.parquet"
SAMPLE = ROOT / "phase2/training/deliverable04/ranker.csv"
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
rows["initiation_candidate_id"] = ""
rows["decision_candidate_id"] = ""
rows["notes"] = ""
rows["split"] = ""
header = not SAMPLE.exists()
rows.to_csv(SAMPLE, mode="a", header=header, index=False)


def excerpt(ctx):
    ctx = str(ctx or "")
    i = ctx.find("[[")
    if i < 0:
        return ctx[:260]
    return ("..." if i > 120 else "") + ctx[max(0, i - 120) : i + 160] + "..."


with OPTIONS.open("a") as f:
    for pid, proc in picks:
        c = df[df["project_id"] == pid].sort_values("parsed_date")
        f.write(f"\n===== project {pid} [{proc}] - {len(c)} candidates =====\n")
        for _, r in c.iterrows():
            pi = r.get("p_init_cal", r.get("p_initiation"))
            pdc = r.get("p_dec_cal", r.get("p_decision"))
            pi = pd.to_numeric(pi, errors="coerce")
            pdc = pd.to_numeric(pdc, errors="coerce")
            f.write(
                f"  id={r['candidate_id']} date={r.get('parsed_date')} "
                f"gran={r.get('date_granularity')} role={r.get('candidate_role')} "
                f"p_init={0.0 if pd.isna(pi) else float(pi):.2f} "
                f"p_dec={0.0 if pd.isna(pdc) else float(pdc):.2f}\n"
            )
            f.write(f"     {excerpt(r.get('model_context'))}\n")

print(
    f"Emitted {len(picks)} projects -> {SAMPLE.name} "
    f"(options -> {OPTIONS.name})."
)
