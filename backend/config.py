"""
Configuration for Knowledge-Shift RAG project.
Reads settings from environment variables / .env file.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# --- Path constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
INDEX_DIR = PROJECT_ROOT / "indexes"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = 384  # all-MiniLM-L6-v2 output dimension


@dataclass
class RetrieverConfig:
    """Retrieval configuration."""
    top_k: int = int(os.getenv("RETRIEVER_TOP_K", "5"))
    index_path: str = str(INDEX_DIR / "faiss_index.bin")
    metadata_path: str = str(INDEX_DIR / "chunk_metadata.json")


@dataclass
class ChunkerConfig:
    """Chunking configuration."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "200"))  # words
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "30"))  # words


@dataclass
class LLMConfig:
    """LLM API configuration."""
    api_key: str = os.getenv("LLM_API_KEY", "")
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))


@dataclass
class Config:
    """Master configuration."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Data paths
    question_path: str = str(DATA_DIR / "question.json")
    textbook_path: str = str(DATA_DIR / "textbook.json")


# Global config instance
config = Config()
