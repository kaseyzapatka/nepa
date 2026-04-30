#!/usr/bin/env python3
"""
Train a SetFit classifier for DOE CE NEPA trigger classification.

Reads one or more labeled CSVs (manual_trigger column), trains a SetFit model
on all-MiniLM-L6-v2, evaluates per-class accuracy, and saves to disk.

Usage:
    python phase2/code/deliverable01/02_train_setfit_trigger.py

The script auto-discovers labeled CSVs in:
    phase2/data/analysis/nepa_trigger/doe_ce_sample_*.csv

Only rows with a non-empty, non-ambiguous manual_trigger are used.
Re-run this script whenever you add more labeled rows.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).resolve().parents[3]
LABELED_GLOB   = "phase2/data/analysis/nepa_trigger/doe_ce_sample_*.csv"
MODEL_OUT_DIR  = REPO_ROOT / "phase2/models/trigger_setfit"
BASE_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"

VALID_LABELS = {
    "federal_funding",
    "federal_direct_action",
    "federal_land",
    "federal_permit",
    "federal_program",
    "federal_property_transaction",
}

# ---------------------------------------------------------------------------
# Text prep (must match inference-time prep in 01_extract_nepa_trigger.py)
# ---------------------------------------------------------------------------

def prep_text(title: str, description: str) -> str:
    title = str(title or "").strip()
    desc  = str(description or "").strip()
    # Unwrap stringified Python/JSON list: ["..."] or ['...']
    if desc.startswith("[") and desc.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(desc)
            if isinstance(parsed, list):
                desc = " ".join(str(x) for x in parsed)
        except Exception:
            pass
    return f"{title} {desc[:2000]}".strip()


# ---------------------------------------------------------------------------
# Load labeled data
# ---------------------------------------------------------------------------

def load_labeled(glob: str) -> pd.DataFrame:
    paths = sorted(REPO_ROOT.glob(glob))
    if not paths:
        log.error("No labeled CSVs found matching: %s", glob)
        sys.exit(1)

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        log.info("  Loaded %s (%d rows)", p.name, len(df))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined rows before filtering: %d", len(combined))

    # Keep only rows with a valid label
    combined = combined[combined["manual_trigger"].isin(VALID_LABELS)].copy()
    combined["text"] = combined.apply(
        lambda r: prep_text(r.get("project_title", ""), r.get("project_description", "")),
        axis=1,
    )
    combined = combined[combined["text"].str.len() > 20].copy()
    log.info("Rows after filtering to valid labels: %d", len(combined))

    log.info("Class distribution:")
    for label, count in combined["manual_trigger"].value_counts().items():
        log.info("  %-35s %d", label, count)

    return combined


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> None:
    try:
        from setfit import SetFitModel, Trainer, TrainingArguments
        from datasets import Dataset
        NEW_API = True
    except ImportError:
        try:
            from setfit import SetFitModel, SetFitTrainer
            from datasets import Dataset
            NEW_API = False
        except ImportError:
            log.error("setfit and datasets packages are required. Install with: pip install setfit datasets")
            sys.exit(1)

    labels = sorted(df["manual_trigger"].unique().tolist())
    log.info("Training with %d classes: %s", len(labels), labels)

    train_df, eval_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["manual_trigger"]
    )
    log.info("Train: %d  |  Eval: %d", len(train_df), len(eval_df))

    train_ds = Dataset.from_dict({"text": train_df["text"].tolist(), "label": train_df["manual_trigger"].tolist()})
    eval_ds  = Dataset.from_dict({"text": eval_df["text"].tolist(),  "label": eval_df["manual_trigger"].tolist()})

    log.info("Loading base model: %s", BASE_MODEL)
    model = SetFitModel.from_pretrained(BASE_MODEL, labels=labels)

    if NEW_API:
        args = TrainingArguments(
            batch_size=16,
            num_epochs=1,
            num_iterations=20,
            evaluation_strategy="epoch",
            save_strategy="no",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            metric="accuracy",
        )
    else:
        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            num_iterations=20,
            num_epochs=1,
            metric="accuracy",
        )

    log.info("Training…")
    trainer.train()

    # Per-class evaluation
    log.info("Evaluating on held-out set (%d rows)…", len(eval_df))
    preds = model.predict(eval_df["text"].tolist())
    preds = [str(p) for p in preds]
    print("\n" + classification_report(eval_df["manual_trigger"].tolist(), preds, labels=labels))

    # Save
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_OUT_DIR))

    # Save label list alongside model for inference-time loading
    (MODEL_OUT_DIR / "label_list.json").write_text(json.dumps(labels))

    log.info("Model saved to: %s", MODEL_OUT_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Loading labeled data…")
    df = load_labeled(LABELED_GLOB)

    if len(df) < 50:
        log.error("Not enough labeled rows (%d). Label more examples first.", len(df))
        sys.exit(1)

    train(df)
