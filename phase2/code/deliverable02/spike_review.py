"""D2 spike review — format LLM adjudications side-by-side with source text.

Run AFTER the ~$1 spike (`02_extract_fonsi_significance.py --sample 30 --model claude-sonnet-5`)
to eyeball what the model did before approving the full batch. Reads whatever is currently in
significance_determinations.parquet and writes a review CSV + a readable markdown file.

Run:  conda run -n nepa python phase2/code/deliverable02/spike_review.py [--limit 30]
Out:  phase2/output/deliverable02/spike_review.csv / spike_review.md
"""
from __future__ import annotations

import argparse

import pandas as pd

import common as C

COLS = ["project_id", "determination_class", "determination_scope", "shared_resource_area",
        "primary_threshold_type", "primary_threshold_status", "mitigation_flag",
        "rationale_text", "extraction_method", "needs_human_review", "page_start", "page_end",
        "evidence_text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    d = pd.read_parquet(C.SIGNIFICANCE_DETERMINATIONS)
    mode = d["extraction_method"].value_counts().to_dict()
    print(f"determinations on disk: {len(d):,}  by method: {mode}")
    if "regex+llm" in mode:
        d = d[d["extraction_method"] == "regex+llm"]   # review the LLM's work, not regex rows
        print(f"reviewing the {len(d):,} LLM-adjudicated rows")
    else:
        print("[note] no LLM rows yet (dry-run only) — showing regex rows as a format preview.")
    d = d.head(args.limit)[COLS]

    C.write_csv(d, C.D2_OUTPUT_DIR / "spike_review.csv", "spike review (tabular)")

    lines = ["# D2 spike review — LLM adjudications vs source text\n"]
    for i, r in enumerate(d.itertuples(index=False), 1):
        lines += [
            f"## {i}. {r.determination_class}  ·  {r.shared_resource_area}  ·  scope={r.determination_scope}",
            f"*project `{r.project_id}` · pp.{r.page_start}-{r.page_end} · "
            f"threshold: {r.primary_threshold_type}/{r.primary_threshold_status} · "
            f"mitigation_flag: {r.mitigation_flag} · needs_review: {r.needs_human_review}*",
            f"\n**LLM rationale:** {r.rationale_text or '(none)'}",
            f"\n> {str(r.evidence_text)[:1200].replace(chr(10), ' ')}"
            + (" …" if len(str(r.evidence_text)) > 1200 else ""),
            "\n---\n"]
    (C.D2_OUTPUT_DIR / "spike_review.md").write_text("\n".join(lines))
    print(f"  wrote {len(d):>7,} rows -> output/deliverable02/spike_review.md")
    print("\nReview questions: (1) class right? (2) resource right? (3) mitigated vs plain-LTS "
          "distinction respected? (4) rationale grounded in the text? "
          "If yes on ~90%+ -> approve the full --batch-run; else tune the prompt in "
          "extract_common._prompt_for and re-spike (~$1).")


if __name__ == "__main__":
    main()
