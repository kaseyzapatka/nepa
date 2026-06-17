#!/usr/bin/env python
"""D4 NIGHT-2 autonomous driver (decoupled retrain + full run, auto-fallback).

Runs unattended. The full coverage run is GUARANTEED; the retrain is best-effort and gated:
  baseline F1 (current model) -> retrain (04/04b/05b into working models/) -> GATE on frozen-test
  F1 -> PASS keep new model / FAIL or error restore models_current -> full 02..08 run -> validate.

Every stage is wrapped: timeout -> failure, failure -> logged + safe fallback. Worst acceptable
outcome = a clean current-model coverage run. Writes a machine-readable STATUS json after each
stage and a human morning report at the end. NOTHING is merged or pushed (stage-only).

Launch:  conda run -n nepa python code/deliverable04/_night_driver.py   (run in background)
"""
import json, os, re, subprocess, sys, shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # WT/phase2/code/deliverable04
PHASE2 = HERE.parent.parent                            # WT/phase2
TL = PHASE2 / "data/analysis/timeline"
MODELS = TL / "models"
MODELS_CUR = TL / "models_current"
NOTES = PHASE2 / "notes/deliverable04"
TS = datetime.now().strftime("%Y%m%d")
LOG = NOTES / f"nightrun_{TS}.log"
STATUS = NOTES / "nightrun_status.json"
MAIN_TL = Path("/Users/Dora/git/consulting/nepa/phase2/data/analysis/timeline")

# generous per-stage timeouts (seconds); a hang -> failure -> fallback, never an infinite stall
T = {"baseline": 1800, "train": 14400, "calib": 1800, "rank": 2400, "gate": 1800, "run": 36000}
GATE_TOL = 0.01  # new per-head F1 must be >= baseline - tol

status = {"started_at": datetime.now(timezone.utc).isoformat(), "stages": {}, "model_used": None}

# BUGFIX: 04 --eval writes a diagnostics CSV; the dir must exist or eval crashes (rc=1) after
# already printing the F1 we need. Create it up front.
(PHASE2 / "output" / "deliverable04" / "diagnostics").mkdir(parents=True, exist_ok=True)

def save_status():
    STATUS.write_text(json.dumps(status, indent=2))

def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def run(cmd, timeout, capture=False):
    """Run a worktree script under the nepa env. Returns (rc, stdout_or_None)."""
    env = dict(os.environ, CONDA_DEFAULT_ENV="nepa")
    full = ["conda", "run", "--no-capture-output", "-n", "nepa", "python"] + cmd
    log(f"RUN ({timeout}s): {' '.join(cmd)}")
    try:
        if capture:
            r = subprocess.run(full, env=env, timeout=timeout, capture_output=True, text=True)
            with open(LOG, "a") as f:
                f.write(r.stdout or ""); f.write(r.stderr or "")
            return r.returncode, r.stdout
        else:
            with open(LOG, "a") as f:
                r = subprocess.run(full, env=env, timeout=timeout, stdout=f, stderr=subprocess.STDOUT)
            return r.returncode, None
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT after {timeout}s: {' '.join(cmd)}")
        return 124, None
    except Exception as e:
        log(f"EXCEPTION: {e}")
        return 1, None

def parse_f1(out):
    """Parse per-head F1 from 04 --eval stdout lines like 'initiation  P=.. R=.. F1=0.812 ..'."""
    f1 = {}
    for m in re.finditer(r"(initiation|decision|final_eis)\s+P=[\d.]+\s+R=[\d.]+\s+F1=([\d.]+)", out or ""):
        f1[m.group(1)] = float(m.group(2))
    return f1

def restore_current():
    for d in ("candidate_classifier", "candidate_ranker"):
        tgt = MODELS / d
        if tgt.exists():
            shutil.rmtree(tgt)
        shutil.copytree(MODELS_CUR / d, tgt)
    log("RESTORED models/ from models_current/ (current model will be used).")

# ---------------------------------------------------------------- 1. baseline F1
sd = HERE / "04_classify_candidates.py"
rc, out = run([str(sd), "--eval", "--model-dir", str(MODELS_CUR / "candidate_classifier")], T["baseline"], capture=True)
baseline = parse_f1(out)  # parse F1 even if rc!=0: eval prints F1 then may crash writing diagnostics
status["stages"]["baseline_f1"] = {"rc": rc, "f1": baseline}
log(f"baseline F1 = {baseline}")
save_status()

# ---------------------------------------------------------------- 2. retrain (best-effort)
retrain_ok = True
# Gate-relevant retrain = classifier + calibrator only. The ranker (05b --train) needs
# timeline_candidates.parquet, which does not exist until the pipeline's 03 stage runs — so it
# CANNOT run in the pre-run retrain. It is deferred; run_pipeline's 05b --apply uses the existing
# ranker, and the +6 new ranker labels are negligible. (Earlier bug: 05b --train here failed on the
# missing parquet and wrongly discarded a good classifier retrain.)
for name, cmd, to in [
    ("train", [str(sd), "--train"], T["train"]),
    ("calib", [str(HERE / "04b_calibrate.py"), "--fit"], T["calib"]),
]:
    rc, _ = run(cmd, to)
    status["stages"][name] = {"rc": rc}
    save_status()
    if rc != 0:
        log(f"retrain step '{name}' FAILED (rc={rc}); aborting retrain, will fall back.")
        retrain_ok = False
        break

# ---------------------------------------------------------------- 3. gate
gate_pass = False
if retrain_ok:
    rc, out = run([str(sd), "--eval"], T["gate"], capture=True)
    new = parse_f1(out) if rc == 0 else {}
    status["stages"]["gate_f1"] = {"rc": rc, "f1": new}
    if rc == 0 and baseline and new:
        ok_init = new.get("initiation", 0) >= baseline.get("initiation", 0) - GATE_TOL
        ok_dec = new.get("decision", 0) >= baseline.get("decision", 0) - GATE_TOL
        gate_pass = ok_init and ok_dec
        log(f"GATE: init {new.get('initiation')} vs {baseline.get('initiation')} | "
            f"dec {new.get('decision')} vs {baseline.get('decision')} -> {'PASS' if gate_pass else 'FAIL'}")
    else:
        log("GATE: could not evaluate new model; treating as FAIL.")

if gate_pass:
    status["model_used"] = "retrained"
    log("Using RETRAINED model.")
else:
    restore_current()
    status["model_used"] = "current_fallback"
    log("Using CURRENT model (fallback).")
save_status()

# ---------------------------------------------------------------- 4. full run (guaranteed)
rc, _ = run([str(HERE / "run_pipeline.py")], T["run"])
status["stages"]["full_run"] = {"rc": rc}
save_status()
if rc != 0:
    log(f"FULL RUN exited rc={rc} — see log. (May be a late-stage/08 issue; check parquets.)")

# ---------------------------------------------------------------- 5. validate + report
try:
    import duckdb, pandas as pd
    con = duckdb.connect()
    new_p = TL / "timeline_project_dates.parquet"
    # newest prenight backup in MAIN
    backups = sorted(MAIN_TL.glob("timeline_project_dates.prenight_*.parquet"))
    pre = backups[-1] if backups else None

    def cov(path):
        return con.execute(f"""
          SELECT process_type,
            COUNT(*) n,
            SUM(CASE WHEN initiation_date IS NOT NULL AND decision_date IS NOT NULL THEN 1 ELSE 0 END) both,
            SUM(CASE WHEN decision_date IS NOT NULL THEN 1 ELSE 0 END) dec_any,
            SUM(CASE WHEN initiation_date IS NOT NULL THEN 1 ELSE 0 END) init_any
          FROM '{path}' WHERE process_type IN ('CE','EA','EIS') GROUP BY 1 ORDER BY 1""").df()

    new_cov = cov(new_p)
    rep = [f"# D4 Night-2 morning report ({TS})", "",
           f"Model used: **{status['model_used']}**  ·  full-run rc: {status['stages'].get('full_run',{}).get('rc')}",
           f"Baseline F1: {baseline}  ·  Gate F1: {status['stages'].get('gate_f1',{}).get('f1')}", "",
           "## Coverage AFTER (worktree)", "", new_cov.to_markdown(index=False)]
    if pre:
        pre_cov = cov(pre)
        rep += ["", f"## Coverage BEFORE (prenight backup {pre.name})", "", pre_cov.to_markdown(index=False),
                "", "## Delta (both = full timelines)", ""]
        m = new_cov.merge(pre_cov, on="process_type", suffixes=("_new", "_old"))
        m["both_delta"] = m["both_new"] - m["both_old"]
        m["dec_delta"] = m["dec_any_new"] - m["dec_any_old"]
        m["init_delta"] = m["init_any_new"] - m["init_any_old"]
        rep += [m[["process_type", "both_old", "both_new", "both_delta",
                   "dec_delta", "init_delta"]].to_markdown(index=False)]
    rep += ["", "## Hand-back (STAGE ONLY — nothing merged/pushed)",
            "- New label files (gitignored, copy back if accepting): "
            "`nepa-night/phase2/training/deliverable04/{classifier,ranker}.csv`",
            "- New outputs: `nepa-night/phase2/data/analysis/timeline/timeline_{candidates,project_dates}.parquet`",
            "- Prenight backup for diffing: "
            f"`{pre.name if pre else 'NONE'}` in MAIN timeline dir.",
            "- A 15-20 row newly-covered sample per process is the next QC step."]
    (NOTES / f"morning_report_{TS}.md").write_text("\n".join(rep))
    status["report"] = str(NOTES / f"morning_report_{TS}.md")
    log(f"Wrote morning report -> morning_report_{TS}.md")
except Exception as e:
    log(f"VALIDATION/REPORT error: {e}")

status["finished_at"] = datetime.now(timezone.utc).isoformat()
save_status()
log("NIGHT DRIVER COMPLETE.")
