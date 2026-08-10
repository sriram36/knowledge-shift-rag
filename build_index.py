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


def apply_knowledge_shifts(loader):
    """
    Append shifted versions of textbook paragraphs from question.json as NEW entries,
    and prepend markers so the LLM can distinguish them.
    Verifies that duplicate questions for the same paragraph do not have conflicting updates.
    """
    shift_map = {}
    for q in loader.questions:
        para_id = q.get("paragraph_info", {}).get("id")
        updated_text = q.get("updated_paragraph")
        
        if para_id is not None and updated_text:
            if para_id in shift_map:
                if shift_map[para_id] != updated_text:
                    raise ValueError(f"Conflict: Paragraph ID {para_id} has conflicting updated_paragraph values.")
            else:
                shift_map[para_id] = updated_text

    shifted_count = 0
    total_entries = len(loader.textbooks)
    paragraphs_with_updates = len(shift_map)
    
    new_entries = []
    
    for entry in loader.textbooks:
        if entry["id"] in shift_map:
            # 1. Mark the original text as stale
            entry["text"] = f"[ORIGINAL TEXT]: {entry['text']}"
            
            # 2. Create the new updated entry
            new_entry = {
                "id": f"{entry['id']}_updated",
                "subject": entry.get("subject", ""),
                "text": f"[NEWEST UPDATE]: {shift_map[entry['id']]}"
            }
            new_entries.append(new_entry)
            shifted_count += 1
            
    # Add all new updated entries to the corpus
    loader.textbooks.extend(new_entries)
            
    without_updates = total_entries - shifted_count
            
    print(f"  Total original textbook entries: {total_entries}")
    print(f"  Number of paragraphs with updates: {paragraphs_with_updates}")
    print(f"  Number of NEW updated entries appended: {shifted_count}")
    print(f"  Number of original entries without updates: {without_updates}")
    print(f"  Final total textbook entries: {len(loader.textbooks)}")
    
    return loader


def main():
    print("=" * 60)
    print("Building FAISS Index")
    print("=" * 60)

    # Phase 1: Load data
    print("\n[1/4] Loading dataset...")
    loader = KnowShiftDataLoader().load()
    print(f"  Loaded {len(loader.textbooks)} textbook entries")

    print("\n[1.5/4] Applying knowledge shifts...")
    loader = apply_knowledge_shifts(loader)

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
