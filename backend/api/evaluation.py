"""
Evaluation API endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

router = APIRouter()


class EvalRequest(BaseModel):
    system: str = Field(default="vanilla", description="vanilla | self_critique | knowledge_repair")
    num_questions: int = Field(default=10, ge=1, le=3005)
    top_k: int = Field(default=5, ge=1, le=20)


class EvalResponse(BaseModel):
    status: str
    message: str


@router.post("/evaluation/run", response_model=EvalResponse)
async def run_evaluation(request: EvalRequest, background_tasks: BackgroundTasks):
    """
    Trigger an evaluation run in the background.

    Results are saved to results/ directory.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    def _run():
        if request.system == "vanilla":
            from experiments.run_vanilla_rag import run_vanilla_rag
            run_vanilla_rag(num_questions=request.num_questions, top_k=request.top_k)
        elif request.system == "self_critique":
            from experiments.run_self_critique import run_self_critique
            run_self_critique(num_questions=request.num_questions, top_k=request.top_k)
        elif request.system == "knowledge_repair":
            from experiments.run_knowledge_repair import run_knowledge_repair
            run_knowledge_repair(num_questions=request.num_questions, top_k=request.top_k)

    background_tasks.add_task(_run)

    return EvalResponse(
        status="started",
        message=f"Evaluation started: system={request.system}, "
                f"num_questions={request.num_questions}, top_k={request.top_k}. "
                f"Results will be saved to results/ directory.",
    )


@router.get("/evaluation/results")
async def get_results():
    """Read and return the latest experiment summary."""
    import json
    from pathlib import Path
    
    results_path = Path(__file__).resolve().parent.parent.parent / "results" / "experiment_summary.json"
    if not results_path.exists():
        return {"error": "No results found. Run experiments first."}
        
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to load results: {e}"}
