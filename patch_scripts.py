import sys

def patch_file(filepath, func_name):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Update def run_...(num_questions: int | None = None, top_k: int = 5, sample_file: str | None = None):
    old_def = f"def {func_name}(num_questions: int | None = None, top_k: int = 5):"
    new_def = f"def {func_name}(num_questions: int | None = None, top_k: int = 5, sample_file: str | None = None):"
    content = content.replace(old_def, new_def)
    
    # 2. Update questions assignment
    old_q = '''    questions = loader.questions
    if num_questions:
        questions = questions[:num_questions]'''
    new_q = '''    questions = loader.questions
    if sample_file:
        with open(sample_file, 'r', encoding='utf-8') as f:
            indices = json.load(f)
        questions = [questions[i] for i in indices]
    elif num_questions:
        questions = questions[:num_questions]'''
    content = content.replace(old_q, new_q)
    
    # 3. Update argparse
    if func_name == "run_vanilla_rag":
        old_argparse = '''    parser = argparse.ArgumentParser(description="Run Vanilla RAG evaluation")
    parser.add_argument("--num_questions", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    run_vanilla_rag(num_questions=args.num_questions, top_k=args.top_k)'''
        new_argparse = '''    parser = argparse.ArgumentParser(description="Run Vanilla RAG evaluation")
    parser.add_argument("--num_questions", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--sample_file", type=str, default=None, help="Path to JSON file with question indices")
    args = parser.parse_args()

    run_vanilla_rag(num_questions=args.num_questions, top_k=args.top_k, sample_file=args.sample_file)'''
    elif func_name == "run_self_critique":
        old_argparse = '''    parser = argparse.ArgumentParser(description="Run Self-Critique RAG evaluation")
    parser.add_argument("--num_questions", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    run_self_critique(num_questions=args.num_questions, top_k=args.top_k)'''
        new_argparse = '''    parser = argparse.ArgumentParser(description="Run Self-Critique RAG evaluation")
    parser.add_argument("--num_questions", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sample_file", type=str, default=None)
    args = parser.parse_args()

    run_self_critique(num_questions=args.num_questions, top_k=args.top_k, sample_file=args.sample_file)'''
    elif func_name == "run_knowledge_repair":
        old_argparse = '''    parser = argparse.ArgumentParser(description="Run Knowledge-Repair RAG evaluation")
    parser.add_argument("--num_questions", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    run_knowledge_repair(num_questions=args.num_questions, top_k=args.top_k)'''
        new_argparse = '''    parser = argparse.ArgumentParser(description="Run Knowledge-Repair RAG evaluation")
    parser.add_argument("--num_questions", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sample_file", type=str, default=None)
    args = parser.parse_args()

    run_knowledge_repair(num_questions=args.num_questions, top_k=args.top_k, sample_file=args.sample_file)'''
    content = content.replace(old_argparse, new_argparse)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Patched {filepath}")

patch_file('experiments/run_vanilla_rag.py', 'run_vanilla_rag')
patch_file('experiments/run_self_critique.py', 'run_self_critique')
patch_file('experiments/run_knowledge_repair.py', 'run_knowledge_repair')
