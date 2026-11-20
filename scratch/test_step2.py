
# test_step2.py
from pathlib import Path
from rag.rag_core import DocumentStore

store = DocumentStore()
for pdf in sorted(Path("data/pdfs").glob("*.pdf")):
    store.add_pdf(str(pdf))

queries = [
    "how does the CYK algorithm parse a sentence?",   # concept + exact term
    "Earley chart",                                   # exact keyword
    "what is the capital of France?",                 # off-topic, should score low
]

for query in queries:
    print(f"\nQuery: {query}")
    print("-- dense only --")
    for chunk, score in store.dense_search(query, k=3):
        print(f"  {score:.3f} | {chunk.doc} p.{chunk.page}")
    print("-- hybrid --")
    for chunk, fused, dense in store.hybrid_search(query, k=3):
        print(f"  fused={fused:.4f} dense={dense:.3f} | {chunk.doc} p.{chunk.page}")