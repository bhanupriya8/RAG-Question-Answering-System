"""
Builds a FAISS index over the text 'extract' of every knowledge base entry,
so we can do semantic search over facts later.

Run AFTER scrape_wikipedia.py:
    python data/build_text_index.py

Output:
    data/text_index.faiss
    data/text_index_meta.json   -> maps FAISS row -> {title, extract, url}
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(HERE, "knowledge_base.json")
INDEX_PATH = os.path.join(HERE, "text_index.faiss")
META_PATH = os.path.join(HERE, "text_index_meta.json")

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, CPU-friendly


def main():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    print(f"Loaded {len(kb)} knowledge base entries.")
    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [f"{e['title']}. {e['extract']}" for e in kb]
    print("Embedding text...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine similarity
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    print(f"Saved FAISS index ({index.ntotal} vectors, dim={dim}) -> {INDEX_PATH}")
    print(f"Saved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
