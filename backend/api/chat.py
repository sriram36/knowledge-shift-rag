"""
Chat API endpoint.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    mode: str = Field(default="vanilla", description="vanilla | self_critique | knowledge_repair")
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    answer_text: str
    reasoning: str
    sources: list[dict[str, Any]]
    critique: dict[str, Any] | None = None
    repair: dict[str, Any] | None = None
    mode: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a question through the selected RAG system."""
    from backend.main import services

    retriever = services.get("retriever")
    generator = services.get("generator")
    critic = services.get("critic")
    repair_service = services.get("repair")

    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="Services not initialized")

    # Retrieve
    results = retriever.retrieve(request.question, top_k=request.top_k)
    context = "\n\n".join(
        f"[Source {i} | {r.chunk_id} | {r.subject}]\n{r.text}"
        for i, r in enumerate(results, 1)
    )
    sources = [r.to_dict() for r in results]

    # For the API, we don't have pre-defined choices, so generate open-ended
    choices = {"A": "", "B": "", "C": "", "D": ""}

    # Generate
    gen_result = generator.generate(
        question=request.question,
        choices=choices,
        context=context,
    )

    answer = gen_result.get("answer", "")
    answer_text = gen_result.get("answer_text", "")
    reasoning = gen_result.get("reasoning", "")

    critique_result = None
    repair_result = None

    if request.mode in ("self_critique", "knowledge_repair"):
        # Critique
        critique_result = critic.critique(
            question=request.question,
            answer=answer,
            answer_text=answer_text,
            context=context,
        )

        if request.mode == "knowledge_repair":
            status = critique_result.get("status", "UNCERTAIN")
            if status in ("UNCERTAIN", "UNSUPPORTED"):
                repair_result = repair_service.repair(
                    question=request.question,
                    choices=choices,
                    original_context=context,
                    previous_answer=answer,
                    previous_answer_text=answer_text,
                    critique_status=status,
                    critique_reason=critique_result.get("reason", ""),
                )
                answer = repair_result.get("answer", answer)
                answer_text = repair_result.get("answer_text", answer_text)
                reasoning = repair_result.get("reasoning", reasoning)

    return ChatResponse(
        answer=answer,
        answer_text=answer_text,
        reasoning=reasoning,
        sources=sources,
        critique=critique_result,
        repair=repair_result,
        mode=request.mode,
    )
