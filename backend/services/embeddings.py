"""
Embedding service for Knowledge-Shift RAG.

Generates sentence embeddings using sentence-transformers,
builds and persists a FAISS index with associated metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import config
from backend.services.chunker import Chunk


class EmbeddingService:
    """Manages sentence embeddings and the FAISS vector index."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.embedding.model_name
        self._model = None
        self._index = None
        self._chunks_metadata: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Embed a list of texts, returns (N, dim) float32 array."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # for cosine similarity via inner product
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text, returns (1, dim) float32 array."""
        return self.embed_texts([query], show_progress=False)

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Build a FAISS index from chunks."""
        import faiss

        texts = [c.text for c in chunks]
        self._chunks_metadata = [c.to_dict() for c in chunks]

        print(f"Embedding {len(texts)} chunks with {self.model_name}...")
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        dim = embeddings.shape[1]
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        print(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    def save_index(
        self,
        index_path: str | None = None,
        metadata_path: str | None = None,
    ) -> None:
        """Persist FAISS index and metadata to disk."""
        import faiss

        index_path = index_path or config.retriever.index_path
        metadata_path = metadata_path or config.retriever.metadata_path

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._chunks_metadata, f, ensure_ascii=False)

        print(f"Saved FAISS index to: {index_path}")
        print(f"Saved metadata to: {metadata_path}")

    def load_index(
        self,
        index_path: str | None = None,
        metadata_path: str | None = None,
    ) -> None:
        """Load FAISS index and metadata from disk."""
        import faiss

        index_path = index_path or config.retriever.index_path
        metadata_path = metadata_path or config.retriever.metadata_path

        self._index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self._chunks_metadata = json.load(f)

        print(f"Loaded FAISS index: {self._index.ntotal} vectors")
        print(f"Loaded metadata: {len(self._chunks_metadata)} chunks")

    def search(self, query_embedding: np.ndarray, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Search the FAISS index.

        Returns list of dicts with chunk metadata + similarity score.
        """
        if self._index is None:
            raise RuntimeError("No FAISS index loaded. Call build_index() or load_index() first.")

        top_k = top_k or config.retriever.top_k
        scores, indices = self._index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            meta = self._chunks_metadata[idx].copy()
            meta["score"] = float(score)
            meta["faiss_index"] = int(idx)
            results.append(meta)

        return results

    @property
    def index(self):
        return self._index

    @property
    def chunks_metadata(self) -> list[dict[str, Any]]:
        return self._chunks_metadata
