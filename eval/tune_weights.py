
# eval/tune_weights.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.rag_core import DocumentStore

QUESTIONS = Path(__file__).parent / "rag_questions.json"
K = 4

qs = [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8")) if q["answerable"]]

store = DocumentStore()
for pdf in sorted(Path("data/pdfs").glob("*.pdf")):
    store.add_pdf(str(pdf))


def first_gold_rank(pages, gold):
    for i, p in enumerate(pages, start=1):
        if p in gold:
            return i
    return None


print(f"{len(qs)} answerable questions, top-{K}\n")
print(f"{'bm25_weight':>12} | {'Hit@4':>6} | {'MRR':>5} | missed questions")
print("-" * 70)

for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
    ranks, missed = [], []
    for item in qs:
        res = store.hybrid_search(item["q"], k=K, bm25_weight=w)
        r = first_gold_rank([c.page for c, _, _ in res], set(item["gold_pages"]))
        ranks.append(r)
        if r is None:
            missed.append(item["q"][:40])
    hit = sum(r is not None for r in ranks) / len(ranks)
    mrr = sum(1 / r for r in ranks if r) / len(ranks)
    print(f"{w:>12} | {hit * 100:>5.0f}% | {mrr:>5.2f} | {'; '.join(missed) or '-'}")