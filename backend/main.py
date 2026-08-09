"""
FastAPI application for Knowledge-Shift RAG.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.embeddings import EmbeddingService
from backend.services.retriever import Retriever
from backend.services.generator import Generator
from backend.services.critic import Critic
from backend.services.repair import KnowledgeRepair
from backend.api.chat import router as chat_router
from backend.api.evaluation import router as eval_router


# Shared services (initialized at startup)
services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS index on startup."""
    print("Loading FAISS index...")
    embedding_service = EmbeddingService()
    embedding_service.load_index()

    retriever = Retriever(embedding_service=embedding_service)
    generator = Generator()
    critic = Critic()
    repair = KnowledgeRepair(retriever=retriever)

    services["retriever"] = retriever
    services["generator"] = generator
    services["critic"] = critic
    services["repair"] = repair

    print("Services initialized.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Knowledge-Shift RAG",
    description="A self-reflective knowledge-repair RAG system for educational QA",
    version="0.1.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(chat_router, prefix="/api")
app.include_router(eval_router, prefix="/api")

# Serve frontend static files
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "index_loaded": "retriever" in services,
    }
