"""
Dataset inspection script.
Loads KnowShiftQA, produces results/dataset_summary.json,
and prints a human-readable report.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from backend.services.data_loader import KnowShiftDataLoader
from backend.config import RESULTS_DIR


def main():
    print("=" * 60)
    print("KnowShiftQA Dataset Inspection")
    print("=" * 60)

    loader = KnowShiftDataLoader().load()
    summary = loader.generate_summary()

    # Save JSON summary
    output_path = RESULTS_DIR / "dataset_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary to: {output_path}")

    # Print human-readable report
    print(f"\n--- Questions ---")
    print(f"  Total: {summary['questions']['total']}")
    print(f"  Fields: {summary['questions']['fields']}")
    print(f"  Updated question subfields: {summary['questions']['updated_question_subfields']}")
    print(f"  With updated_paragraph: {summary['questions']['with_updated_paragraph']}")
    print(f"  With updated_question: {summary['questions']['with_updated_question']}")

    print(f"\n  Types:")
    for qtype, count in summary["questions"]["types"].items():
        print(f"    {qtype}: {count}")

    print(f"\n  Subjects:")
    for subject, count in summary["questions"]["subjects"].items():
        print(f"    {subject}: {count}")

    print(f"\n  Verified status:")
    for status, count in summary["questions"]["verified_status"].items():
        print(f"    {status}: {count}")

    print(f"\n--- Textbooks ---")
    print(f"  Total: {summary['textbooks']['total']}")
    print(f"  Fields: {summary['textbooks']['fields']}")

    print(f"\n  Subjects:")
    for subject, count in summary["textbooks"]["subjects"].items():
        print(f"    {subject}: {count}")

    print(f"\n  Modes:")
    for mode, count in summary["textbooks"]["modes"].items():
        print(f"    mode {mode}: {count}")

    print(f"\n--- Mapping ---")
    print(f"  Questions mapped to textbook: {summary['mapping']['questions_mapped_to_textbook']}")
    print(f"  Unmapped paragraph IDs: {summary['mapping']['unmapped_paragraph_ids_count']}")
    if summary["mapping"]["unmapped_paragraph_ids_sample"]:
        print(f"  Sample unmapped IDs: {summary['mapping']['unmapped_paragraph_ids_sample']}")

    # Save original baseline results
    baseline = {
        "source": "KnowShiftQA original retrieval_demo.ipynb (Ada-002 embeddings)",
        "note": "These are the authors' retrieval baseline results, not from our system",
        "metrics": {
            "mean_rank": 1.2728785357737105,
            "hit_at_1": 0.7923460898502496,
            "hit_at_2": 0.8905158069883528,
            "hit_at_3": 0.9224625623960067,
            "hit_at_5": 0.9544093178036606,
            "hit_at_10": 0.9787021630615641,
        },
        "per_type": {
            "simple_direct": {"hit_at_1": 0.8229508196721311},
            "multihop_direct": {"hit_at_1": 0.8076335877862595},
            "multihop_distant": {"hit_at_1": 0.8008474576271186},
            "multihop_implicit": {"hit_at_1": 0.8009592326139089},
            "distant_implicit": {"hit_at_1": 0.7300813008130081},
        },
    }
    baseline_path = RESULTS_DIR / "knowshiftqa_original_baseline.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nSaved original baseline to: {baseline_path}")

    print("\n" + "=" * 60)
    print("Phase 1 — Dataset inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
