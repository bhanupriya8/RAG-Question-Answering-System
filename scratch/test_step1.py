
# test_step1.py
from pathlib import Path
from rag.rag_core import DocumentStore

PDF_DIR = Path("data/pdfs")

store = DocumentStore()

pdf_files = sorted(PDF_DIR.glob("*.pdf"))
if not pdf_files:
    raise SystemExit(f"No PDFs found in {PDF_DIR}. Put at least one .pdf file there.")

for pdf in pdf_files:
    n = store.add_pdf(str(pdf))
    print(f"{pdf.name}: {n} chunks added")

print(f"\nTotal chunks: {len(store.chunks)}\n")

query = "what are the main conclusions?"
print(f"Query: {query}\n")
for chunk, score in store.dense_search(query, k=3):
    print(f"{score:.3f} | {chunk.doc} p.{chunk.page} | {chunk.text[:100]}...")