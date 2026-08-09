"""
Text chunker for KnowShiftQA textbook data.

Splits textbook paragraphs into overlapping word-level chunks,
preserving metadata for retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.config import config


@dataclass
class Chunk:
    """A single text chunk with its metadata."""
    chunk_id: str          # e.g. "doc_42_chunk_0"
    text: str
    doc_id: int            # original textbook id
    subject: str
    sub_index: int         # subject sub-index from textbook["sub"][1]
    mode: str              # textbook mode
    chunk_index: int       # position within the document
    total_chunks: int      # total chunks for this document
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "subject": self.subject,
            "sub_index": self.sub_index,
            "mode": self.mode,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "word_count": self.word_count,
        }


class TextChunker:
    """Splits textbook entries into overlapping word-level chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or config.chunker.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunker.chunk_overlap

    def chunk_document(self, doc: dict[str, Any]) -> list[Chunk]:
        """Chunk a single textbook entry."""
        text = doc.get("text", "")
        if not text.strip():
            return []

        doc_id = doc["id"]
        sub = doc.get("sub", ["unknown", 0])
        subject = sub[0] if sub else "unknown"
        sub_index = sub[1] if len(sub) > 1 else 0
        mode = str(doc.get("mode", "unknown"))

        words = text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        starts = list(range(0, len(words), step))

        for i, start in enumerate(starts):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(Chunk(
                chunk_id=f"doc_{doc_id}_chunk_{i}",
                text=chunk_text,
                doc_id=doc_id,
                subject=subject,
                sub_index=sub_index,
                mode=mode,
                chunk_index=i,
                total_chunks=0,  # filled below
                word_count=len(chunk_words),
            ))

            # If we've reached the end of the text, stop
            if end >= len(words):
                break

        # Fill total_chunks
        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks

    def chunk_all(self, textbooks: list[dict[str, Any]]) -> list[Chunk]:
        """Chunk all textbook entries."""
        all_chunks: list[Chunk] = []
        for doc in textbooks:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
