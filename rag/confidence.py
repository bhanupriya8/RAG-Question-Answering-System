
# rag/confidence.py
from dataclasses import dataclass

# Tunable thresholds, calibrated on all-MiniLM-L6-v2 cosine scores.
# Step 7 (evaluation) will help you fine-tune these.
HIGH = 0.45
LOW = 0.35


@dataclass
class Confidence:
    level: str      # "high" | "medium" | "low"
    score: float    # best dense cosine among retrieved chunks
    message: str    # text to show the user / pass to the LLM


def assess_confidence(results) -> Confidence:
    """
    results: output of DocumentStore.hybrid_search(),
             i.e. [(Chunk, fused_score, dense_cosine), ...]
    """
    if not results:
        return Confidence("low", 0.0, "No relevant content was found in the documents.")

    top = max(dense for _, _, dense in results)

    if top >= HIGH:
        return Confidence("high", top, "")
    if top >= LOW:
        return Confidence(
            "medium", top,
            "I found related information, but I'm not fully certain it answers your question.",
        )
    return Confidence(
        "low", top,
        "I couldn't find anything in the uploaded documents that answers this.",
    )