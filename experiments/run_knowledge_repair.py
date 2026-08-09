"""
Experiment: Knowledge-Repair RAG (System C)

Question → retrieve → generate → critique →
  if UNCERTAIN/UNSUPPORTED: reformulate → retrieve more → verify → corrected answer
→ final answer

Usage:
    python -m experiments.run_knowledge_repair [--num_questions N] [--top_k K]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import config, RESULTS_DIR
from backend.services.data_loader import KnowShiftDataLoader
from backend.services.retriever import Retriever
from backend.services.generator import Generator
from backend.services.critic import Critic
from backend.services.repair import KnowledgeRepair
from experiments.helpers import (
    prepare_question,
    is_correct,
    save_results_jsonl,
    create_experiment_metadata,
)


def run_knowledge_repair(num_questions: int | None = None, top_k: int = 5):
    print("=" * 60)
    print("System C — Knowledge-Repair RAG")
    print("=" * 60)

    # Load dataset
    loader = KnowShiftDataLoader().load()
    questions = loader.questions
    if num_questions:
        questions = questions[:num_questions]
    print(f"Evaluating {len(questions)} questions")

    # Init services
    retriever = Retriever()
    retriever.load_index()
    generator = Generator()
    critic = Critic()
    repair = KnowledgeRepair(retriever=retriever)

    results = []
    correct_count = 0
    repair_triggered_count = 0
    repair_success_count = 0
    critique_stats = {"SUPPORTED": 0, "UNCERTAIN": 0, "UNSUPPORTED": 0}
    start_time = time.time()

    for i, q in enumerate(questions):
        q_start = time.time()
        prepared = prepare_question(q)

        # Step 1: Retrieve
        retrieved = retriever.retrieve(prepared["question_text"], top_k=top_k)
        context = "\n\n".join(
            f"[Source {j} | {r.chunk_id} | {r.subject}]\n{r.text}"
            for j, r in enumerate(retrieved, 1)
        )

        # Step 2: Generate initial answer
        gen_result = generator.generate(
            question=prepared["question_text"],
            choices=prepared["choices"],
            context=context,
        )

        initial_answer = gen_result.get("answer", "E")
        initial_answer_text = gen_result.get("answer_text", "")

        # Step 3: Critique
        critique_result = critic.critique(
            question=prepared["question_text"],
            answer=initial_answer,
            answer_text=initial_answer_text,
            context=context,
        )

        critique_status = critique_result.get("status", "UNCERTAIN")
        critique_stats[critique_status] = critique_stats.get(critique_status, 0) + 1

        # Step 4: Repair (if needed)
        repair_result = None
        final_answer = initial_answer
        final_answer_text = initial_answer_text

        if critique_status in ("UNCERTAIN", "UNSUPPORTED"):
            repair_triggered_count += 1

            repair_result = repair.repair(
                question=prepared["question_text"],
                choices=prepared["choices"],
                original_context=context,
                previous_answer=initial_answer,
                previous_answer_text=initial_answer_text,
                critique_status=critique_status,
                critique_reason=critique_result.get("reason", ""),
            )

            final_answer = repair_result.get("answer", initial_answer)
            final_answer_text = repair_result.get("answer_text", initial_answer_text)

            if repair_result.get("repair_successful", False):
                repair_success_count += 1

        correct = is_correct(final_answer, prepared["correct_letter"])
        if correct:
            correct_count += 1

        # Was the initial answer wrong but repair fixed it?
        initial_correct = is_correct(initial_answer, prepared["correct_letter"])

        q_time = time.time() - q_start

        result = {
            "question_index": i,
            "question_text": prepared["question_text"],
            "question_type": prepared["question_type"],
            "subject": prepared["subject"],
            "paragraph_id": prepared["paragraph_id"],
            "choices": prepared["choices"],
            "correct_letter": prepared["correct_letter"],
            "correct_text": prepared["correct_text"],
            "initial_answer": initial_answer,
            "initial_answer_text": initial_answer_text,
            "initial_correct": initial_correct,
            "critique_status": critique_status,
            "critique_reason": critique_result.get("reason", ""),
            "repair_triggered": critique_status in ("UNCERTAIN", "UNSUPPORTED"),
            "repair_result": {
                "answer": repair_result.get("answer", "") if repair_result else "",
                "reasoning": repair_result.get("reasoning", "") if repair_result else "",
                "repair_successful": repair_result.get("repair_successful", False) if repair_result else False,
                "reformulated_query": repair_result.get("reformulated_query", "") if repair_result else "",
            } if repair_result else None,
            "final_answer": final_answer,
            "final_answer_text": final_answer_text,
            "is_correct": correct,
            "initial_was_correct_final_wrong": initial_correct and not correct,
            "initial_was_wrong_final_correct": not initial_correct and correct,
            "retrieved_chunk_ids": [r.chunk_id for r in retrieved],
            "latency_seconds": round(q_time, 3),
            "error": gen_result.get("error", False) or critique_result.get("error", False),
        }
        results.append(result)

        repair_marker = ""
        if critique_status in ("UNCERTAIN", "UNSUPPORTED"):
            if not initial_correct and correct:
                repair_marker = " [REPAIR FIXED]"
            elif initial_correct and not correct:
                repair_marker = " [REPAIR BROKE]"
            else:
                repair_marker = " [REPAIR]"

        print(f"  [{i+1}/{len(questions)}] {'✓' if correct else '✗'} "
              f"pred={final_answer} correct={prepared['correct_letter']} "
              f"critique={critique_status}{repair_marker} "
              f"acc={correct_count/(i+1):.3f} time={q_time:.1f}s")

    total_time = time.time() - start_time

    # Save results
    save_results_jsonl(results, RESULTS_DIR / "knowledge_repair_rag.jsonl")

    metadata = create_experiment_metadata(
        system_name="knowledge_repair_rag",
        model=config.llm.model,
        embedding_model=config.embedding.model_name,
        top_k=top_k,
        num_questions=len(questions),
        total_correct=correct_count,
        total_time=total_time,
    )
    metadata["critique_distribution"] = critique_stats
    metadata["repair_triggered"] = repair_triggered_count
    metadata["repair_successful"] = repair_success_count

    print(f"\n{'=' * 60}")
    print(f"Knowledge-Repair RAG Results:")
    print(f"  Accuracy: {metadata['accuracy']:.4f} ({correct_count}/{len(questions)})")
    print(f"  Critique distribution: {critique_stats}")
    print(f"  Repairs triggered: {repair_triggered_count}")
    print(f"  Repairs successful: {repair_success_count}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'=' * 60}")

    return results, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Knowledge-Repair RAG evaluation")
    parser.add_argument("--num_questions", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    run_knowledge_repair(num_questions=args.num_questions, top_k=args.top_k)
