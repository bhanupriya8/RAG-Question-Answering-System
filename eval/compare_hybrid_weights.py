
"""
Compare hybrid retrieval weights on the existing RAG evaluation dataset.

- Uses the same PDFs, chunking, questions and gold labels.
- Compares dense, BM25 and multiple hybrid configurations.
- Selects configurations using DEV results.
- Reports TEST results separately.
- Does not modify the original evaluation script or project code.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.rag_core import DocumentStore, tokenize

HERE = Path(__file__).resolve().parent
QUESTION_FILE = HERE / "rag_questions_v2.json"
PDF_DIR = ROOT / "data" / "pdfs"
OUTPUT_FILE = HERE / "hybrid_weight_comparison.json"

K = 4

# Keep these settings identical to the original evaluation.
CHUNK_WORDS = 180
OVERLAP = 40

CONFIGS = [
    ("Current_1_0.25", 1.0, 0.25),
    ("Balanced_1_1", 1.0, 1.0),
    ("BM25_1_2", 1.0, 2.0),
    ("BM25_1_4", 1.0, 4.0),
]


def load_store():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")

    store = DocumentStore()

    for pdf in pdfs:
        added = store.add_pdf(
            str(pdf),
            chunk_words=CHUNK_WORDS,
            overlap=OVERLAP
        )
        print(f"Loaded {pdf.name}: {added} chunks")

    print(f"Total chunks indexed: {len(store.chunks)}")
    return store


def gold_set(item):
    return {
        (x["doc"], int(x["page"]))
        for x in item.get("gold_sources", [])
    }


def first_gold_rank(chunks, gold):
    for rank, chunk in enumerate(chunks, start=1):
        if (chunk.doc, chunk.page) in gold:
            return rank
    return None


def get_rankings(store, query):
    n = len(store.chunks)

    if n == 0:
        return {
            "dense": [],
            "bm25": [],
            "hybrid": {}
        }

    # Dense ranking
    qvec = store.model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    dense_scores = store.embeddings @ qvec
    dense_order = dense_scores.argsort()[::-1].tolist()

    dense = [store.chunks[i] for i in dense_order]

    # BM25 ranking, preserving original zero-score filtering
    bm25_scores = store.bm25.get_scores(tokenize(query))
    bm25_order = bm25_scores.argsort()[::-1].tolist()

    bm25 = [
        store.chunks[i]
        for i in bm25_order
        if bm25_scores[i] > 0
    ]

    # Hybrid rankings with different weights
    hybrid = {}

    for name, dense_weight, bm25_weight in CONFIGS:
        rows = store.hybrid_search(
            query,
            k=n,
            rrf_k=60,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight
        )

        hybrid[name] = [row[0] for row in rows]

    return {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid
    }


# def calculate_metrics(items, ranked, method):
#     answerable = [x for x in items if x["answerable"]]

#     ranks = [
#         first_gold_rank(ranked[x["id"]][method], gold_set(x))
#         for x in answerable
#     ]

#     hits = sum(
#         rank is not None and rank <= K
#         for rank in ranks
#     )

#     reciprocal_ranks = [
#         1 / rank if rank is not None else 0
#         for rank in ranks
#     ]

#     return {
#         "questions": len(answerable),
#         "hit_at_4": hits / len(answerable) if answerable else None,
#         "mrr": (
#             sum(reciprocal_ranks) / len(answerable)
#             if answerable else None
#         ),
#         "missed_ids": [
#             item["id"]
#             for item, rank in zip(answerable, ranks)
#             if rank is None or rank > K
#         ]
#     }


def calculate_metrics(items, ranked, method):
    answerable = [x for x in items if x["answerable"]]

    ranks = []

    for item in answerable:
        question_rankings = ranked[item["id"]]

        # Dense and BM25 are stored directly.
        if method in ("dense", "bm25"):
            chunks = question_rankings[method]

        # Hybrid configurations are stored inside "hybrid".
        else:
            chunks = question_rankings["hybrid"][method]

        rank = first_gold_rank(chunks, gold_set(item))
        ranks.append(rank)

    hits = sum(
        rank is not None and rank <= K
        for rank in ranks
    )

    reciprocal_ranks = [
        1 / rank if rank is not None else 0
        for rank in ranks
    ]

    return {
        "questions": len(answerable),
        "hit_at_4": hits / len(answerable) if answerable else None,
        "mrr": (
            sum(reciprocal_ranks) / len(answerable)
            if answerable else None
        ),
        "missed_ids": [
            item["id"]
            for item, rank in zip(answerable, ranks)
            if rank is None or rank > K
        ]
    }


def main():
    items = json.loads(
        QUESTION_FILE.read_text(encoding="utf-8")
    )

    store = load_store()

    ranked = {}

    print("\nEvaluating questions...")

    for i, item in enumerate(items, start=1):
        ranked[item["id"]] = get_rankings(store, item["q"])

        print(
            f"[{i:02d}/{len(items)}] "
            f"{item['id']} - {item['q']}"
        )

    methods = ["dense", "bm25"] + [
        config[0] for config in CONFIGS
    ]

    results = {
        "dataset": QUESTION_FILE.name,
        "total_questions": len(items),
        "k": K,
        "results": {}
    }

    for split in ["dev", "test"]:
        subset = [
            item for item in items
            if item["split"] == split
        ]

        results["results"][split] = {}

        for method in methods:
            metrics = calculate_metrics(
                subset,
                ranked,
                method
            )

            results["results"][split][method] = metrics

    # Save detailed results for every question.
    details = []

    for item in items:
        gold = gold_set(item)
        question_rankings = ranked[item["id"]]

        row = {
            "id": item["id"],
            "question": item["q"],
            "split": item["split"],
            "answerable": item["answerable"]
        }

        row["dense_rank"] = first_gold_rank(
            question_rankings["dense"], gold
        ) if gold else None

        row["bm25_rank"] = first_gold_rank(
            question_rankings["bm25"], gold
        ) if gold else None

        for name, _, _ in CONFIGS:
            row[name + "_rank"] = (
                first_gold_rank(
                    question_rankings["hybrid"][name],
                    gold
                ) if gold else None
            )

        details.append(row)

    results["details"] = details

    OUTPUT_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Print results
    print("\n" + "=" * 65)
    print("HYBRID WEIGHT COMPARISON")
    print("=" * 65)

    for split in ["dev", "test"]:
        print(f"\n{split.upper()} RESULTS")

        print(
            f"{'Method':<20}"
            f"{'Hit@4':>12}"
            f"{'MRR':>12}"
        )

        print("-" * 44)

        for method in methods:
            metrics = results["results"][split][method]

            hit = metrics["hit_at_4"]
            mrr = metrics["mrr"]

            print(
                f"{method:<20}"
                f"{hit:>11.1%}"
                f"{mrr:>12.3f}"
            )

    # Select best hybrid configuration using DEV only.
    dev_results = results["results"]["dev"]

    best = max(
        [config[0] for config in CONFIGS],
        key=lambda name: (
            dev_results[name]["hit_at_4"],
            dev_results[name]["mrr"]
        )
    )

    results["best_hybrid_on_dev"] = best

    OUTPUT_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nBEST HYBRID CONFIGURATION ON DEV:")
    print(best)

    print(f"\nDetailed results saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
