
# check_predictor.py
import re
from pathlib import Path
from rag.rag_core import DocumentStore

store = DocumentStore()
for pdf in sorted(Path("data/pdfs").glob("*.pdf")):
    store.add_pdf(str(pdf))

for c in store.chunks:
    m = re.search(r"predict", c.text, re.I)
    if m:
        s, e = max(0, m.start() - 80), min(len(c.text), m.end() + 200)
        print(f"{c.doc} p.{c.page} (chunk {c.chunk_id}):\n  ...{c.text[s:e]}...\n")