#!/usr/bin/env python3
"""Run a small post-MVP RAG evaluation set."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = RAG_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nepa_rag.answer import answer_question
from nepa_rag.config import load_config


def load_questions(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML or store eval questions as JSON.") from exc
    data = yaml.safe_load(text)
    return data.get("questions", data if isinstance(data, list) else [])


def main() -> None:
    config = load_config()
    questions_path = RAG_DIR / "eval" / "questions.yaml"
    output_path = RAG_DIR / "eval" / "rag_eval_results.csv"
    questions = load_questions(questions_path)
    if not questions:
        raise SystemExit(f"No eval questions found in {questions_path}")

    rows = []
    for item in questions:
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        result = answer_question(
            config,
            question,
            selected_project_id=item.get("project_id") or None,
            process_types=item.get("process_types") or None,
            force_whole_corpus=bool(item.get("whole_corpus", False)),
        )
        rows.append(
            {
                "id": item.get("id", ""),
                "question": question,
                "scope": result.scope.mode,
                "model": result.model or "",
                "source_count": len(result.evidence),
                "warning_count": len(result.warnings),
                "warnings": " | ".join(result.warnings),
                "answer_preview": result.answer[:500].replace("\n", " "),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[rag-eval] wrote {output_path}")


if __name__ == "__main__":
    main()
