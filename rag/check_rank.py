
# check_rank.py
from pathlib import Path
from rag.rag_core import DocumentStore

store = DocumentStore()
for pdf in sorted(Path("data/pdfs").glob("*.pdf")):
    store.add_pdf(str(pdf))

TARGET = 33   # the chunk on p.20 that explains the Predictor

for q in ["explain top-down parsing and also what the Earley predictor does",
          "Earley predictor",
          "what does the Predictor do in Earley parsing?"]:
    results = store.hybrid_search(q, k=len(store.chunks))
    ids = [c.chunk_id for c, _, _ in results]
    rank = ids.index(TARGET) + 1 if TARGET in ids else None
    print(f"hybrid rank of chunk {TARGET}: {rank}  <- {q}")