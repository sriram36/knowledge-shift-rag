import json
import random
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.services.data_loader import KnowShiftDataLoader
from experiments.helpers import prepare_question

def generate_stratified_sample():
    loader = KnowShiftDataLoader().load()
    questions = loader.questions
    
    grouped = defaultdict(list)
    for i, q in enumerate(questions):
        try:
            prep = prepare_question(q)
            grouped[(prep["subject"], prep["question_type"])].append(i)
        except Exception:
            continue
            
    sampled_indices = []
    random.seed(42)
    
    for key, indices in grouped.items():
        if len(indices) >= 4:
            sampled_indices.extend(random.sample(indices, 4))
        else:
            print(f"Warning: Not enough questions for {key}. Found {len(indices)}")
            sampled_indices.extend(indices)
            
    out_file = Path(__file__).resolve().parent.parent / "data" / "eval_sample_indices.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sampled_indices, f, indent=2)
        
    print(f"Sampled {len(sampled_indices)} questions total.")
    print(f"Saved to {out_file}")

if __name__ == '__main__':
    generate_stratified_sample()
