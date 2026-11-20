
# test_step3.py
from pathlib import Path
from rag.rag_core import DocumentStore
from rag.confidence import assess_confidence

store = DocumentStore()
for pdf in sorted(Path("data/pdfs").glob("*.pdf")):
    store.add_pdf(str(pdf))

queries = [
    "how does the CYK algorithm parse a sentence?",
    "Earley chart",
    "what are the main conclusions?",
    "what is the capital of France?",
    "difference between top-down and bottom-up parsing",
]

for q in queries:
    results = store.hybrid_search(q, k=3)
    conf = assess_confidence(results)
    print(f"\nQuery: {q}")
    print(f"  confidence = {conf.level.upper()} (top dense = {conf.score:.3f})")
    if conf.message:
        print(f"  message    = {conf.message}")