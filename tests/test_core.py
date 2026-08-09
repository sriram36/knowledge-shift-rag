"""
Unit tests for Knowledge-Shift RAG.
"""

import json
import sys
from pathlib import Path

import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.data_loader import KnowShiftDataLoader
from backend.services.chunker import TextChunker, Chunk
from experiments.helpers import prepare_question, is_correct


# ======================================================================
# Data Loader Tests
# ======================================================================

class TestDataLoader:
    """Tests for KnowShiftDataLoader."""

    def test_load_returns_self(self):
        loader = KnowShiftDataLoader()
        result = loader.load()
        assert result is loader

    def test_questions_is_list(self):
        loader = KnowShiftDataLoader().load()
        assert isinstance(loader.questions, list)
        assert len(loader.questions) > 0

    def test_textbooks_is_list(self):
        loader = KnowShiftDataLoader().load()
        assert isinstance(loader.textbooks, list)
        assert len(loader.textbooks) > 0

    def test_question_count(self):
        loader = KnowShiftDataLoader().load()
        assert len(loader.questions) == 3005

    def test_textbook_count(self):
        loader = KnowShiftDataLoader().load()
        assert len(loader.textbooks) == 2205

    def test_question_has_required_fields(self):
        loader = KnowShiftDataLoader().load()
        q = loader.questions[0]
        assert "paragraph_info" in q
        assert "type" in q
        assert "updated_question" in q
        assert "updated_paragraph" in q

    def test_updated_question_has_choices(self):
        loader = KnowShiftDataLoader().load()
        uq = loader.questions[0]["updated_question"]
        assert "question" in uq
        assert "updated" in uq
        assert "random1" in uq
        assert "random2" in uq
        assert "random3" in uq

    def test_textbook_has_required_fields(self):
        loader = KnowShiftDataLoader().load()
        t = loader.textbooks[0]
        assert "id" in t
        assert "text" in t
        assert "sub" in t
        assert "mode" in t

    def test_all_questions_map_to_textbook(self):
        loader = KnowShiftDataLoader().load()
        for q in loader.questions:
            tb = loader.get_textbook_for_question(q)
            assert tb is not None, f"Question with para_id={q['paragraph_info']['id']} has no textbook"

    def test_summary_generation(self):
        loader = KnowShiftDataLoader().load()
        summary = loader.generate_summary()
        assert summary["questions"]["total"] == 3005
        assert summary["textbooks"]["total"] == 2205
        assert summary["mapping"]["unmapped_paragraph_ids_count"] == 0

    def test_question_types(self):
        loader = KnowShiftDataLoader().load()
        types = loader.get_question_types()
        expected = {"simple_direct", "multihop_direct", "multihop_distant",
                    "multihop_implicit", "distant_implicit"}
        assert set(types.keys()) == expected

    def test_subjects(self):
        loader = KnowShiftDataLoader().load()
        subjects = loader.get_subjects()
        expected = {"Biology", "Chemistry", "Geology", "History", "Physics"}
        assert set(subjects.keys()) == expected


# ======================================================================
# Chunker Tests
# ======================================================================

class TestChunker:
    """Tests for TextChunker."""

    def test_chunk_single_doc(self):
        doc = {"id": 0, "sub": ["Biology", 0], "text": " ".join(["word"] * 500), "mode": "0"}
        chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_preserves_metadata(self):
        doc = {"id": 42, "sub": ["Physics", 3], "text": "Hello world test", "mode": "2"}
        chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].doc_id == 42
        assert chunks[0].subject == "Physics"
        assert chunks[0].mode == "2"

    def test_chunk_id_format(self):
        doc = {"id": 7, "sub": ["History", 1], "text": "Some text here", "mode": "1"}
        chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.chunk_document(doc)
        assert chunks[0].chunk_id == "doc_7_chunk_0"

    def test_empty_text_returns_no_chunks(self):
        doc = {"id": 0, "sub": ["Biology", 0], "text": "", "mode": "0"}
        chunker = TextChunker()
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 0

    def test_chunk_all(self):
        loader = KnowShiftDataLoader().load()
        chunker = TextChunker()
        chunks = chunker.chunk_all(loader.textbooks)
        assert len(chunks) > len(loader.textbooks)  # most docs produce >1 chunk

    def test_chunk_overlap(self):
        words = " ".join([f"word{i}" for i in range(400)])
        doc = {"id": 0, "sub": ["Test", 0], "text": words, "mode": "0"}
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 3  # 0-200, 150-350, 300-400
        # Check overlap: last words of chunk 0 should appear in chunk 1
        c0_words = chunks[0].text.split()
        c1_words = chunks[1].text.split()
        overlap = set(c0_words[-50:]) & set(c1_words[:50])
        assert len(overlap) > 0


# ======================================================================
# Experiment Helpers Tests
# ======================================================================

class TestExperimentHelpers:
    """Tests for experiment helper functions."""

    def test_prepare_question_structure(self):
        q = {
            "paragraph_info": {"id": 1, "sub": ["Biology", 1]},
            "type": "simple_direct",
            "updated_question": {
                "question": "What is X?",
                "updated": "Answer A",
                "random1": "Answer B",
                "random2": "Answer C",
                "random3": "Answer D",
            },
            "updated_paragraph": "Some paragraph",
            "verified": 1,
        }
        prepared = prepare_question(q)
        assert prepared["question_text"] == "What is X?"
        assert len(prepared["choices"]) == 4
        assert prepared["correct_text"] == "Answer A"
        # The correct answer must be one of the choices
        assert prepared["correct_text"] in prepared["choices"].values()
        # correct_letter should point to correct_text
        assert prepared["choices"][prepared["correct_letter"]] == "Answer A"

    def test_is_correct(self):
        assert is_correct("A", "A")
        assert is_correct("a", "A")
        assert is_correct(" B ", "B")
        assert not is_correct("A", "B")

    def test_prepare_question_deterministic(self):
        q = {
            "paragraph_info": {"id": 1, "sub": ["Biology", 1]},
            "type": "simple_direct",
            "updated_question": {
                "question": "What is X?",
                "updated": "Correct",
                "random1": "Wrong1",
                "random2": "Wrong2",
                "random3": "Wrong3",
            },
            "updated_paragraph": "Text",
            "verified": 1,
        }
        # Same input should produce same output
        p1 = prepare_question(q)
        p2 = prepare_question(q)
        assert p1["correct_letter"] == p2["correct_letter"]
        assert p1["choices"] == p2["choices"]


# ======================================================================
# Generation Response Parsing Tests
# ======================================================================

class TestGeneratorParsing:
    """Tests for generator response parsing."""

    def test_parse_valid_json(self):
        from backend.services.generator import Generator
        gen = Generator.__new__(Generator)  # avoid __init__ needing API key
        result = gen._parse_response('{"answer": "A", "answer_text": "test", "reasoning": "because"}')
        assert result["answer"] == "A"
        assert result["error"] is False

    def test_parse_markdown_json(self):
        from backend.services.generator import Generator
        gen = Generator.__new__(Generator)
        raw = '```json\n{"answer": "B", "answer_text": "test"}\n```'
        result = gen._parse_response(raw)
        assert result["answer"] == "B"

    def test_parse_fallback(self):
        from backend.services.generator import Generator
        gen = Generator.__new__(Generator)
        result = gen._parse_response('The answer is definitely "answer": "C" somewhere')
        assert result["answer"] == "C"


# ======================================================================
# Critic Response Parsing Tests
# ======================================================================

class TestCriticParsing:
    """Tests for critic response parsing."""

    def test_parse_valid_json(self):
        from backend.services.critic import Critic
        c = Critic.__new__(Critic)
        result = c._parse_response('{"status": "SUPPORTED", "reason": "clearly stated"}')
        assert result["status"] == "SUPPORTED"
        assert result["error"] is False

    def test_parse_normalizes_status(self):
        from backend.services.critic import Critic
        c = Critic.__new__(Critic)
        result = c._parse_response('{"status": "supported", "reason": "test"}')
        assert result["status"] == "SUPPORTED"

    def test_parse_fallback_unsupported(self):
        from backend.services.critic import Critic
        c = Critic.__new__(Critic)
        result = c._parse_response("The answer is UNSUPPORTED by the evidence.")
        assert result["status"] == "UNSUPPORTED"

    def test_parse_fallback_uncertain(self):
        from backend.services.critic import Critic
        c = Critic.__new__(Critic)
        result = c._parse_response("I'm not sure about this answer.")
        assert result["status"] == "UNCERTAIN"
