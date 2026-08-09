"""
Compare results from all three RAG systems.

Reads JSONL result files and produces:
- comparison.csv
- experiment_summary.json

Usage:
    python -m experiments.compare_results
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import RESULTS_DIR


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    results = []
    if not path.exists():
        return results
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_metrics(results: list[dict], system_name: str) -> dict:
    """Compute metrics from a result set."""
    if not results:
        return {"system": system_name, "total": 0, "note": "No results found"}

    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    errors = sum(1 for r in results if r.get("error", False))
    latencies = [r.get("latency_seconds", 0) for r in results]

    # By question type
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        qt = r.get("question_type", "unknown")
        by_type[qt]["total"] += 1
        if r.get("is_correct", False):
            by_type[qt]["correct"] += 1

    type_accuracy = {
        qt: {
            "total": v["total"],
            "correct": v["correct"],
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
        }
        for qt, v in sorted(by_type.items())
    }

    # By subject
    by_subject = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        sub = r.get("subject", "unknown")
        by_subject[sub]["total"] += 1
        if r.get("is_correct", False):
            by_subject[sub]["correct"] += 1

    subject_accuracy = {
        sub: {
            "total": v["total"],
            "correct": v["correct"],
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
        }
        for sub, v in sorted(by_subject.items())
    }

    metrics = {
        "system": system_name,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "errors": errors,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "by_type": type_accuracy,
        "by_subject": subject_accuracy,
    }

    # System B/C specific
    if system_name in ("self_critique_rag", "knowledge_repair_rag"):
        critique_dist = defaultdict(int)
        for r in results:
            status = r.get("critique_status", "UNKNOWN")
            critique_dist[status] += 1
        metrics["critique_distribution"] = dict(critique_dist)

    if system_name == "knowledge_repair_rag":
        repairs = sum(1 for r in results if r.get("repair_triggered", False))
        repair_fixed = sum(1 for r in results if r.get("initial_was_wrong_final_correct", False))
        repair_broke = sum(1 for r in results if r.get("initial_was_correct_final_wrong", False))
        metrics["repairs_triggered"] = repairs
        metrics["repairs_fixed"] = repair_fixed
        metrics["repairs_broke"] = repair_broke

    return metrics


def main():
    print("=" * 60)
    print("Comparing RAG Systems")
    print("=" * 60)

    systems = {
        "vanilla_rag": RESULTS_DIR / "vanilla_rag.jsonl",
        "self_critique_rag": RESULTS_DIR / "self_critique_rag.jsonl",
        "knowledge_repair_rag": RESULTS_DIR / "knowledge_repair_rag.jsonl",
    }

    all_metrics = {}
    for name, path in systems.items():
        results = load_jsonl(path)
        if results:
            metrics = compute_metrics(results, name)
            all_metrics[name] = metrics
            print(f"\n{name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
            print(f"  Errors: {metrics['errors']}")
            print(f"  Avg latency: {metrics['avg_latency']:.3f}s")
        else:
            print(f"\n{name}: No results found at {path}")

    # Save summary
    summary = {
        "systems": all_metrics,
        "comparison_note": "All systems evaluated on the same questions from KnowShiftQA",
    }
    summary_path = RESULTS_DIR / "experiment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved experiment summary to: {summary_path}")

    # Save comparison CSV
    if all_metrics:
        csv_path = RESULTS_DIR / "comparison.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            header = ["metric"]
            for name in systems:
                if name in all_metrics:
                    header.append(name)
            writer.writerow(header)

            # Overall accuracy
            row = ["accuracy"]
            for name in systems:
                if name in all_metrics:
                    row.append(f"{all_metrics[name]['accuracy']:.4f}")
            writer.writerow(row)

            # By type
            all_types = set()
            for m in all_metrics.values():
                all_types.update(m.get("by_type", {}).keys())
            for qt in sorted(all_types):
                row = [f"accuracy_{qt}"]
                for name in systems:
                    if name in all_metrics:
                        bt = all_metrics[name].get("by_type", {}).get(qt, {})
                        row.append(f"{bt.get('accuracy', 0):.4f}")
                writer.writerow(row)

            # Latency
            row = ["avg_latency_s"]
            for name in systems:
                if name in all_metrics:
                    row.append(f"{all_metrics[name]['avg_latency']:.3f}")
            writer.writerow(row)

        print(f"Saved comparison CSV to: {csv_path}")

    print(f"\n{'=' * 60}")
    print("Comparison complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
