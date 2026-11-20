"""
Retrieval-only evaluation for the two course PDFs.
Compares dense, BM25-only, and hybrid retrieval using document+page gold labels.
No LLM/API calls are made by this script.

Run from the RAG_QA project root:
    python eval/rag_eval_v2.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.rag_core import DocumentStore, tokenize
from rag.confidence import assess_confidence

HERE = Path(__file__).resolve().parent
QUESTION_FILE = HERE / "rag_questions_v2.json"
PDF_DIR = ROOT / "data" / "pdfs"
OUTPUT_FILE = HERE / "rag_eval_v2_results.json"
K = 4


def load_store():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_DIR}. Copy the two course PDFs into data/pdfs first."
        )

    expected = {"NLP Unit–I (1).pdf", "Parsing Algorithms in NLP (2).pdf"}
    found = {p.name for p in pdfs}
    missing = expected - found
    if missing:
        print("WARNING: expected course PDF filename(s) not found:", ", ".join(sorted(missing)))

    store = DocumentStore()
    for pdf in pdfs:
        added = store.add_pdf(str(pdf))
        print(f"Loaded {pdf.name}: {added} chunks")
    print(f"Total chunks indexed: {len(store.chunks)}")
    return store


# def ranked_chunks(store, query):
#     """Return complete rankings so gold rank is not truncated at top-k."""
#     n = len(store.chunks)
#     if n == 0:
#         return {"dense": [], "bm25": [], "hybrid": []}

#     qvec = store.model.encode([query], normalize_embeddings=True)[0]
#     dense_scores = store.embeddings @ qvec
#     dense_order = dense_scores.argsort()[::-1].tolist()
#     dense = [store.chunks[i] for i in dense_order]

#     bm25_scores = store.bm25.get_scores(tokenize(query))
#     bm25_order = bm25_scores.argsort()[::-1].tolist()
#     bm25 = [store.chunks[i] for i in bm25_order if bm25_scores[i] > 0]

#     hybrid_rows = store.hybrid_search(query, k=n)
#     hybrid = [row[0] for row in hybrid_rows]
#     confidence = assess_confidence(hybrid_rows)
#     return {"dense": dense, "bm25": bm25, "hybrid": hybrid}, confidence

def ranked_chunks(store, query):
    n = len(store.chunks)

    if n == 0:
        return {
            "dense": [],
            "bm25": [],
            "hybrid": []
        }, assess_confidence([])

    qvec = store.model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    dense_scores = store.embeddings @ qvec
    dense_order = dense_scores.argsort()[::-1].tolist()
    dense = [store.chunks[i] for i in dense_order]

    bm25_scores = store.bm25.get_scores(tokenize(query))
    bm25_order = bm25_scores.argsort()[::-1].tolist()

    bm25 = [
        store.chunks[i]
        for i in bm25_order
        if bm25_scores[i] > 0
    ]

    # Retrieve all chunks for ranking evaluation
    hybrid_rows = store.hybrid_search(query, k=n)
    hybrid = [row[0] for row in hybrid_rows]

    # Use only the production top-4 results for confidence
    confidence_rows = store.hybrid_search(query, k=4)
    confidence = assess_confidence(confidence_rows)

    return {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid
    }, confidence


def gold_set(item):
    return {(x["doc"], int(x["page"])) for x in item.get("gold_sources", [])}


def first_gold_rank(chunks, gold):
    for rank, chunk in enumerate(chunks, start=1):
        if (chunk.doc, chunk.page) in gold:
            return rank
    return None


def evaluate_method(items, ranked, method, k=K):
    answerable = [x for x in items if x["answerable"]]
    ranks = [first_gold_rank(ranked[x["id"]][method], gold_set(x)) for x in answerable]
    hits = sum(r is not None and r <= k for r in ranks)
    rr = sum((1 / r) if r else 0 for r in ranks)
    return {
        "questions": len(answerable),
        f"hit_at_{k}": hits / len(answerable) if answerable else None,
        "mrr": rr / len(answerable) if answerable else None,
        "missed_ids": [
            x["id"] for x, rank in zip(answerable, ranks)
            if rank is None or rank > k
        ],
    }


def evaluate_split(items, ranked, confidences, split_name):
    subset = [x for x in items if x["split"] == split_name]
    answerable = [x for x in subset if x["answerable"]]
    unanswerable = [x for x in subset if not x["answerable"]]
    method_metrics = {
        method: evaluate_method(subset, ranked, method)
        for method in ("dense", "bm25", "hybrid")
    }
    answerable_not_low = sum(
        confidences[x["id"]].level != "low" for x in answerable
    )
    unanswerable_low = sum(
        confidences[x["id"]].level == "low" for x in unanswerable
    )
    return {
        "total_questions": len(subset),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "retrieval": method_metrics,
        "confidence": {
            "answerable_not_low_rate": (
                answerable_not_low / len(answerable) if answerable else None
            ),
            "unanswerable_low_rate": (
                unanswerable_low / len(unanswerable) if unanswerable else None
            ),
        },
    }


def main():
    items = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))
    store = load_store()

    ranked = {}
    confidences = {}
    detail_rows = []
    for i, item in enumerate(items, start=1):
        rankings, conf = ranked_chunks(store, item["q"])
        ranked[item["id"]] = rankings
        confidences[item["id"]] = conf
        gold = gold_set(item)
        detail_rows.append({
            "id": item["id"],
            "q": item["q"],
            "split": item["split"],
            "answerable": item["answerable"],
            "gold_sources": sorted([{"doc": d, "page": p} for d, p in gold],
                                   key=lambda x: (x["doc"], x["page"])),
            "dense_rank": first_gold_rank(rankings["dense"], gold) if gold else None,
            "bm25_rank": first_gold_rank(rankings["bm25"], gold) if gold else None,
            "hybrid_rank": first_gold_rank(rankings["hybrid"], gold) if gold else None,
            "confidence": conf.level,
            "top_dense_cosine": round(conf.score, 4),
        })
        print(f"[{i:02d}/{len(items)}] {item['id']} {conf.level.upper():6} {item['q']}")

    result = {
        "dataset": QUESTION_FILE.name,
        "k": K,
        "total_questions": len(items),
        "answerable": sum(x["answerable"] for x in items),
        "unanswerable": sum(not x["answerable"] for x in items),
        "splits": {
            name: evaluate_split(items, ranked, confidences, name)
            for name in ("dev", "test")
        },
        "all_questions": evaluate_split(
            [{**x, "split": "all"} for x in items],
            ranked, confidences, "all"
        ),
        "details": detail_rows,
        "note": (
            "Tune retrieval weights and confidence thresholds on dev only. "
            "Do not change them after inspecting test results; if changed, create a new test set."
        ),
    }
    OUTPUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== RETRIEVAL RESULTS ===")
    for split, data in result["splits"].items():
        print(f"\n{split.upper()} ({data['total_questions']} questions)")
        for method, metrics in data["retrieval"].items():
            hit_key = f"hit_at_{K}"
            hit = metrics[hit_key]
            print(f"{method:>7}: Hit@{K}={hit:.1%}" if hit is not None else f"{method:>7}: n/a")
            if metrics["mrr"] is not None:
                print(f"         MRR={metrics['mrr']:.3f}")
        print("Confidence:", data["confidence"])
    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
