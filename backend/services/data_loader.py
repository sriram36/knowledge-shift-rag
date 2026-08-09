"""
Data loader for KnowShiftQA dataset.

Loads and validates question.json and textbook.json.
Provides structured access to questions, textbook paragraphs,
knowledge-shift information, and the mapping between them.
"""

import json
from pathlib import Path
from typing import Any

from backend.config import config


class KnowShiftDataLoader:
    """Loads and provides access to the KnowShiftQA dataset."""

    def __init__(
        self,
        question_path: str | None = None,
        textbook_path: str | None = None,
    ):
        self.question_path = Path(question_path or config.question_path)
        self.textbook_path = Path(textbook_path or config.textbook_path)

        self.questions: list[dict[str, Any]] = []
        self.textbooks: list[dict[str, Any]] = []

        # Lookup maps built after loading
        self._textbook_by_id: dict[int, dict[str, Any]] = {}

    def load(self) -> "KnowShiftDataLoader":
        """Load both dataset files and build indexes."""
        self._load_questions()
        self._load_textbooks()
        self._build_indexes()
        return self

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_questions(self) -> None:
        with open(self.question_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"Expected question.json root to be a list, got {type(data).__name__}"
            )
        self.questions = data

    def _load_textbooks(self) -> None:
        with open(self.textbook_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"Expected textbook.json root to be a list, got {type(data).__name__}"
            )
        self.textbooks = data

    def _build_indexes(self) -> None:
        self._textbook_by_id = {entry["id"]: entry for entry in self.textbooks}

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_textbook_by_id(self, doc_id: int) -> dict[str, Any] | None:
        """Return a textbook entry by its id field."""
        return self._textbook_by_id.get(doc_id)

    def get_textbook_for_question(self, question: dict[str, Any]) -> dict[str, Any] | None:
        """Return the textbook entry referenced by a question's paragraph_info.id."""
        para_id = question.get("paragraph_info", {}).get("id")
        if para_id is None:
            return None
        return self.get_textbook_by_id(para_id)

    # ------------------------------------------------------------------
    # Summary / inspection helpers
    # ------------------------------------------------------------------

    def get_question_types(self) -> dict[str, int]:
        """Count questions by type."""
        counts: dict[str, int] = {}
        for q in self.questions:
            qtype = q.get("type", "unknown")
            counts[qtype] = counts.get(qtype, 0) + 1
        return dict(sorted(counts.items()))

    def get_subjects(self) -> dict[str, int]:
        """Count questions by subject."""
        counts: dict[str, int] = {}
        for q in self.questions:
            sub = q.get("paragraph_info", {}).get("sub", [])
            subject = sub[0] if sub else "unknown"
            counts[subject] = counts.get(subject, 0) + 1
        return dict(sorted(counts.items()))

    def get_textbook_subjects(self) -> dict[str, int]:
        """Count textbook entries by subject."""
        counts: dict[str, int] = {}
        for t in self.textbooks:
            sub = t.get("sub", [])
            subject = sub[0] if sub else "unknown"
            counts[subject] = counts.get(subject, 0) + 1
        return dict(sorted(counts.items()))

    def get_textbook_modes(self) -> dict[str, int]:
        """Count textbook entries by mode."""
        counts: dict[str, int] = {}
        for t in self.textbooks:
            mode = str(t.get("mode", "unknown"))
            counts[mode] = counts.get(mode, 0) + 1
        return dict(sorted(counts.items()))

    def count_verified(self) -> dict[str, int]:
        """Count questions by verified status."""
        counts: dict[str, int] = {}
        for q in self.questions:
            verified = q.get("verified")
            key = str(verified) if verified is not None else "missing"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def count_with_updated_paragraph(self) -> int:
        """Count questions that have an updated_paragraph field."""
        return sum(
            1 for q in self.questions
            if q.get("updated_paragraph") is not None
        )

    def count_with_updated_question(self) -> int:
        """Count questions that have an updated_question field."""
        return sum(
            1 for q in self.questions
            if q.get("updated_question") is not None
        )

    def generate_summary(self) -> dict[str, Any]:
        """Generate a comprehensive dataset summary."""
        # Inspect question fields from first entry
        question_fields = list(self.questions[0].keys()) if self.questions else []
        textbook_fields = list(self.textbooks[0].keys()) if self.textbooks else []

        # Inspect updated_question subfields
        uq_fields = []
        if self.questions:
            uq = self.questions[0].get("updated_question", {})
            if isinstance(uq, dict):
                uq_fields = list(uq.keys())

        # Check paragraph_info → textbook mapping
        mapped_count = 0
        unmapped_ids = []
        for q in self.questions:
            para_id = q.get("paragraph_info", {}).get("id")
            if para_id is not None and para_id in self._textbook_by_id:
                mapped_count += 1
            elif para_id is not None:
                if para_id not in unmapped_ids:
                    unmapped_ids.append(para_id)

        return {
            "dataset": "KnowShiftQA",
            "questions": {
                "total": len(self.questions),
                "fields": question_fields,
                "updated_question_subfields": uq_fields,
                "types": self.get_question_types(),
                "subjects": self.get_subjects(),
                "verified_status": self.count_verified(),
                "with_updated_paragraph": self.count_with_updated_paragraph(),
                "with_updated_question": self.count_with_updated_question(),
            },
            "textbooks": {
                "total": len(self.textbooks),
                "fields": textbook_fields,
                "subjects": self.get_textbook_subjects(),
                "modes": self.get_textbook_modes(),
            },
            "mapping": {
                "questions_mapped_to_textbook": mapped_count,
                "unmapped_paragraph_ids_count": len(unmapped_ids),
                "unmapped_paragraph_ids_sample": unmapped_ids[:10],
            },
        }
