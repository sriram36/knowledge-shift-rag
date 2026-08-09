"""
Shared helpers for experiment scripts.

Handles KnowShiftQA question format → standardised choices dict,
result saving, and answer matching.
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Seed for reproducibility of choice shuffling
SHUFFLE_SEED = 42


def prepare_question(q: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a KnowShiftQA question entry into a standardised format.

    In KnowShiftQA:
      - updated_question.updated  = the correct answer (matching the updated_paragraph)
      - updated_question.random1/2/3 = distractors

    The correct answer is always 'A' in the raw data, so we
    shuffle choices deterministically and track the correct letter.
    """
    uq = q["updated_question"]
    question_text = uq["question"]

    # Build choices: correct answer + 3 distractors
    raw_choices = [
        ("correct", uq["updated"]),
        ("distractor1", uq["random1"]),
        ("distractor2", uq["random2"]),
        ("distractor3", uq["random3"]),
    ]

    # Deterministic shuffle based on question text
    rng = random.Random(SHUFFLE_SEED + hash(question_text) % 10000)
    rng.shuffle(raw_choices)

    letters = ["A", "B", "C", "D"]
    choices = {}
    correct_letter = "A"
    for letter, (tag, text) in zip(letters, raw_choices):
        choices[letter] = text
        if tag == "correct":
            correct_letter = letter

    # Get paragraph reference
    para_info = q.get("paragraph_info", {})

    return {
        "question_text": question_text,
        "choices": choices,
        "correct_letter": correct_letter,
        "correct_text": uq["updated"],
        "question_type": q.get("type", "unknown"),
        "subject": para_info.get("sub", ["unknown"])[0] if para_info.get("sub") else "unknown",
        "paragraph_id": para_info.get("id"),
        "updated_paragraph": q.get("updated_paragraph", ""),
        "verified": q.get("verified"),
    }


def is_correct(predicted_letter: str, correct_letter: str) -> bool:
    """Check if the predicted answer matches the correct one."""
    return predicted_letter.upper().strip() == correct_letter.upper().strip()


def save_results_jsonl(results: list[dict[str, Any]], path: Path) -> None:
    """Save results as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to: {path}")


def create_experiment_metadata(
    system_name: str,
    model: str,
    embedding_model: str,
    top_k: int,
    num_questions: int,
    total_correct: int,
    total_time: float,
) -> dict[str, Any]:
    """Create metadata dict for an experiment run."""
    return {
        "system": system_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "embedding_model": embedding_model,
        "top_k": top_k,
        "num_questions": num_questions,
        "total_correct": total_correct,
        "accuracy": total_correct / num_questions if num_questions > 0 else 0.0,
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_question": round(total_time / num_questions, 3) if num_questions > 0 else 0.0,
    }
