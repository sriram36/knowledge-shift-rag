# Knowledge-Shift-Aware Self-Reflective RAG

A lightweight self-reflective knowledge-repair RAG system evaluated under educational knowledge-shift scenarios using the [KnowShiftQA](https://arxiv.org/pdf/2412.08985) benchmark.

## Project Purpose

This project implements and empirically evaluates a lightweight self-reflective knowledge-repair RAG pipeline under controlled educational knowledge-shift scenarios. It is **not** a reproduction of Self-RAG's training procedure — instead, it implements self-critique and knowledge-repair at inference time using an existing LLM.

### Three Systems

| System | Description |
|--------|-------------|
| **A — Vanilla RAG** | Question → embed → retrieve → LLM → answer |
| **B — Self-Critique RAG** | + second LLM call to classify answer as SUPPORTED/UNCERTAIN/UNSUPPORTED |
| **C — Knowledge-Repair RAG** | + if unsupported: reformulate query → retrieve more evidence → verify → corrected answer |

## Dataset

We use the **KnowShiftQA** benchmark (ACL 2025):
- 3,005 multiple-choice questions across 5 subjects (Biology, Chemistry, Physics, Geology, History)
- 2,205 textbook paragraphs with hypothetical knowledge updates
- 5 question types testing different reasoning requirements

## Architecture

```
question.json / textbook.json
        ↓
   data_loader.py      (load & validate)
        ↓
   chunker.py          (word-level chunks with overlap)
        ↓
   embeddings.py       (sentence-transformers → FAISS)
        ↓
   retriever.py        (top-k similarity search)
        ↓
   generator.py        (LLM answer generation)
        ↓
   critic.py           (self-critique: SUPPORTED/UNCERTAIN/UNSUPPORTED)
        ↓
   repair.py           (knowledge-repair: reformulate → retrieve → verify)
```

## Installation

```bash
# Clone and enter the project
cd knowledge-shift-rag

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required:
- `LLM_API_KEY` — Your OpenAI (or compatible) API key
- `LLM_BASE_URL` — API base URL (default: `https://api.openai.com/v1`)
- `LLM_MODEL` — Model name (default: `gpt-3.5-turbo`)

Optional:
- `EMBEDDING_MODEL` — Sentence transformer model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `RETRIEVER_TOP_K` — Number of chunks to retrieve (default: `5`)
- `CHUNK_SIZE` — Words per chunk (default: `200`)
- `CHUNK_OVERLAP` — Overlap words (default: `30`)

## Build the FAISS Index

This generates embeddings for all textbook chunks and stores them in a FAISS index:

```bash
python build_index.py
```

This takes ~2 minutes on CPU and creates:
- `indexes/faiss_index.bin` — FAISS vector index
- `indexes/chunk_metadata.json` — Chunk metadata mapping

## Run Experiments

### Vanilla RAG (System A)

```bash
# Full dataset (3005 questions)
python -m experiments.run_vanilla_rag

# Quick test (10 questions)
python -m experiments.run_vanilla_rag --num_questions 10
```

### Self-Critique RAG (System B)

```bash
python -m experiments.run_self_critique --num_questions 10
```

### Knowledge-Repair RAG (System C)

```bash
python -m experiments.run_knowledge_repair --num_questions 10
```

### Compare Results

```bash
python -m experiments.compare_results
```

Results are saved to the `results/` directory.

## Run Evaluation

```bash
# Inspect the dataset
python inspect_dataset.py

# Run all experiments with a subset
python -m experiments.run_vanilla_rag --num_questions 50
python -m experiments.run_self_critique --num_questions 50
python -m experiments.run_knowledge_repair --num_questions 50
python -m experiments.compare_results
```

## Start FastAPI

```bash
uvicorn backend.main:app --reload --port 8000
```

Then visit:
- Frontend: http://localhost:8000/
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

## Start Frontend

The frontend is served automatically by FastAPI at http://localhost:8000/.

No separate build step is needed.

## Run Tests

```bash
python -m pytest tests/ -v
```

## Project Limitations

1. **No LLM fine-tuning**: Uses an existing LLM via API; no training or reflection-token training.
2. **CPU-only embeddings**: Sentence-transformers runs on CPU for simplicity.
3. **API cost**: Systems B and C require 2-4x more LLM API calls than System A.
4. **Simple chunking**: Fixed-size word-level chunks; no semantic chunking.
5. **Answer matching**: Relies on LLM returning a letter choice; no fuzzy matching of free-text answers.
6. **Knowledge-repair scope**: "Repair" means retrieving additional evidence and re-answering, not writing back to the knowledge base.
7. **Not a novel algorithm**: The contribution is implementation, integration, and controlled evaluation — not a new method.

## Research Positioning

> "Implement and empirically evaluate a lightweight self-reflective knowledge-repair RAG pipeline under controlled educational knowledge-shift scenarios using the KnowShiftQA benchmark."

Inspired by:
- [Self-RAG](https://arxiv.org/abs/2310.11511) — self-reflection concept
- [KnowShiftQA](https://arxiv.org/abs/2412.08985) — benchmark dataset
- WriteBack-RAG, CounterRefine — related knowledge-repair approaches
