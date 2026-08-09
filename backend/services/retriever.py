"""
Retriever service for Knowledge-Shift RAG.

Provides top-k retrieval from the FAISS index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.services.embeddings import EmbeddingService
from backend.config import config


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    chunk_id: str
    text: str
    doc_id: int
    subject: str
    score: float
    chunk_index: int
    total_chunks: int
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "subject": self.subject,
            "score": self.score,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "mode": self.mode,
        }


class Retriever:
    """Top-k retrieval from a FAISS index."""

    def __init__(self, embedding_service: EmbeddingService | None = None):
        self.embedding_service = embedding_service or EmbeddingService()

    def load_index(self) -> None:
        """Load the persisted FAISS index."""
        self.embedding_service.load_index()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: The question or search text.
            top_k: Number of results to return (default from config).

        Returns:
            List of RetrievalResult sorted by descending similarity.
        """
        top_k = top_k or config.retriever.top_k

        query_embedding = self.embedding_service.embed_query(query)
        raw_results = self.embedding_service.search(query_embedding, top_k=top_k)

        results = []
        for r in raw_results:
            results.append(RetrievalResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                doc_id=r["doc_id"],
                subject=r["subject"],
                score=r["score"],
                chunk_index=r["chunk_index"],
                total_chunks=r["total_chunks"],
                mode=r["mode"],
            ))

        return results

    def retrieve_as_context(self, query: str, top_k: int | None = None) -> str:
        """Retrieve and format as a single context string for the LLM."""
        results = self.retrieve(query, top_k)
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source {i} | {r.chunk_id} | {r.subject}]\n{r.text}"
            )
        return "\n\n".join(context_parts)
