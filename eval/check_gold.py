
# eval/check_gold.py
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rag.rag_core import DocumentStore

CHECKS = {
    "what is the difference between stemming and lemmatization?": ["stemming", "lemmatization"],
    "what are the limitations of CYK?": ["limitation", "chomsky", "expensive"],
    "what are the applications of POS tagging?": ["applications", "machine translation", "information extraction"],
    "what are the major challenges in NLP?": ["challenges", "ambiguity", "multilingual"],
    "does Earley parsing require Chomsky Normal Form?": ["chomsky", "cnf"],
}

qs = {q["q"]: q for q in json.loads((HERE / "rag_questions.json").read_text(encoding="utf-8"))}

store = DocumentStore()
for pdf in sorted((HERE.parent / "data" / "pdfs").glob("*.pdf")):
    store.add_pdf(str(pdf))

pages = defaultdict(str)
for c in store.chunks:
    pages[(c.doc, c.page)] += " " + c.text.lower()

for question, terms in CHECKS.items():
    gold = set(qs[question]["gold_pages"])
    print("=" * 72)
    print("Q:", question)
    print("My gold pages:", sorted(gold), "| words searched:", terms)
    rows = []
    for (doc, page), text in pages.items():
        hits = [t for t in terms if t in text]
        if hits:
            rows.append((len(hits), doc, page, hits))
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    for n, doc, page, hits in rows[:8]:
        mark = "GOLD" if page in gold else "    "
        print(f"  {mark}  p.{page:<3} {doc[:30]:<30} {n}/{len(terms)}: {', '.join(hits)}")