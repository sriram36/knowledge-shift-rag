"""
Build the FAISS index from KnowShiftQA textbook data.

Usage:
    python build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.services.data_loader import KnowShiftDataLoader
from backend.services.chunker import TextChunker
from backend.services.embeddings import EmbeddingService


def main():
    print("=" * 60)
    print("Building FAISS Index")
    print("=" * 60)

    # Phase 1: Load data
    print("\n[1/4] Loading dataset...")
    loader = KnowShiftDataLoader().load()
    print(f"  Loaded {len(loader.textbooks)} textbook entries")

    # Phase 2: Chunk
    print("\n[2/4] Chunking textbooks...")
    chunker = TextChunker()
    chunks = chunker.chunk_all(loader.textbooks)
    print(f"  Created {len(chunks)} chunks")
    print(f"  Chunk size: {chunker.chunk_size} words, overlap: {chunker.chunk_overlap}")

    # Phase 3: Embed & build index
    print("\n[3/4] Embedding chunks & building FAISS index...")
    embedding_service = EmbeddingService()
    embedding_service.build_index(chunks)

    # Phase 4: Save
    print("\n[4/4] Saving index to disk...")
    embedding_service.save_index()

    print("\n" + "=" * 60)
    print("Index build complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
