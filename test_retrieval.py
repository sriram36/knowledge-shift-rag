import sys; sys.path.insert(0, '.')
from backend.services.data_loader import KnowShiftDataLoader
from backend.services.retriever import Retriever
from experiments.helpers import prepare_question

loader = KnowShiftDataLoader().load()
retriever = Retriever()
retriever.load_index()

for i in range(2):
    q = loader.questions[i]
    prep = prepare_question(q)
    print('\n' + '='*80)
    print(f'QUESTION {i+1}: {prep["question_text"]}')
    print(f'CORRECT (SHIFTED) ANSWER: {prep["correct_text"]}')
    print('-'*80)
    
    results = retriever.retrieve(prep["question_text"], top_k=3)
    for j, res in enumerate(results):
        print(f'CHUNK {j+1} ({res.chunk_id}): {res.text[:150]}...')
