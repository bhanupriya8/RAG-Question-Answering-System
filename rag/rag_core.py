# rag/rag_core.py
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "at",
    "is", "are", "was", "were", "be", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "how", "why", "does", "do", "did", "also", "explain",
    "describe", "tell", "me", "about", "from", "as", "can", "you", "please",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    chunk_id: int
    text: str
    doc: str     # file name, e.g. "report_a.pdf"
    page: int    # 1-indexed page number


class DocumentStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None  # shape (n_chunks, dim)
        self.bm25: BM25Okapi | None = None

    def _get_model(self):
        """Load the embedding model only when it is first needed."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model...")
            self._model = SentenceTransformer(
                self.model_name,
                device="cpu"
            )
        return self._model

    def add_pdf(self, path: str, chunk_words: int = 180, overlap: int = 40) -> int:
        """Ingest one PDF. Returns number of chunks added."""
        doc_name = Path(path).name
        if any(c.doc == doc_name for c in self.chunks):
            return 0  # already ingested

        new_chunks = []
        reader = PdfReader(path)
        step = chunk_words - overlap

        for page_num, page in enumerate(reader.pages, start=1):
            words = (page.extract_text() or "").split()
            for start in range(0, max(len(words), 1), step):
                piece = " ".join(words[start:start + chunk_words])
                if len(piece.split()) < 15:   # skip near-empty fragments
                    continue
                new_chunks.append(Chunk(
                    chunk_id=len(self.chunks) + len(new_chunks),
                    text=piece,
                    doc=doc_name,
                    page=page_num,
                ))

        if not new_chunks:
            return 0

        vecs = self._get_model().encode(
            [c.text for c in new_chunks],
            normalize_embeddings=True,   # dot product = cosine similarity
            show_progress_bar=False,
        )
        self.embeddings = vecs if self.embeddings is None else np.vstack([self.embeddings, vecs])
        self.chunks.extend(new_chunks)

        # rebuild keyword index over all chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

        return len(new_chunks)

    def dense_search(self, query: str, k: int = 5, docs: list[str] | None = None):
        """Returns [(Chunk, cosine_score), ...]. Optionally restrict to certain docs."""
        if self.embeddings is None:
            return []
        q = self._get_model().encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores)
        results = []
        for i in order:
            if docs and self.chunks[i].doc not in docs:
                continue
            results.append((self.chunks[i], float(scores[i])))
            if len(results) == k:
                break
        return results


    def hybrid_search(self, query: str, k: int = 5, docs: list[str] | None = None,
                      rrf_k: int = 60, dense_weight: float = 1.0, bm25_weight: float = 4.0):
        """
        Weighted Reciprocal Rank Fusion of dense and BM25 rankings.
        Returns [(Chunk, fused_score, dense_cosine), ...] sorted best first.
        dense_cosine is kept because the confidence check uses it.
        """
        if self.embeddings is None or self.bm25 is None:
            return []

        # dense ranking
        q = self._get_model().encode([query], normalize_embeddings=True)[0]
        dense_scores = self.embeddings @ q
        dense_rank = np.argsort(-dense_scores)

        # keyword ranking
        bm25_scores = np.array(self.bm25.get_scores(tokenize(query)))
        bm25_rank = np.argsort(-bm25_scores)

        # weighted RRF: each ranking contributes weight / (rrf_k + rank)
        fused = {}
        for rank, i in enumerate(dense_rank):
            fused[i] = fused.get(i, 0.0) + dense_weight / (rrf_k + rank + 1)
        for rank, i in enumerate(bm25_rank):
            if bm25_scores[i] > 0:   # ignore chunks with no keyword match
                fused[i] = fused.get(i, 0.0) + bm25_weight / (rrf_k + rank + 1)

        results = []
        for i, score in sorted(fused.items(), key=lambda x: -x[1]):
            if docs and self.chunks[i].doc not in docs:
                continue
            results.append((self.chunks[i], score, float(dense_scores[i])))
            if len(results) == k:
                break
        return results
