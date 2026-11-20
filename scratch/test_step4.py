
# test_step4.py
import re
from pathlib import Path

from agent.llm_client import DEFAULT_CLIENT
from rag.rag_core import DocumentStore
from rag.answer import answer_question


def llm(prompt: str) -> str:
    """Adapter: wraps your existing client in the llm(prompt) -> str shape."""
    text = DEFAULT_CLIENT.call(
        system_prompt="You are a careful assistant that answers strictly from provided sources.",
        conversation=[{"role": "user", "content": prompt}],
    )
    # some reasoning models wrap their thinking in <think> tags; drop it
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


store = DocumentStore()
for pdf in sorted(Path("data/pdfs").glob("*.pdf")):
    store.add_pdf(str(pdf))

questions = [
    "how does the CYK algorithm parse a sentence?",
    "what is the capital of France?",
    "what are the limitations of CYK?",
]

for q in questions:
    print("=" * 70)
    print("Q:", q)
    ans = answer_question(store, q, llm, k=4)
    print(f"[confidence: {ans.confidence.level} | top dense = {ans.confidence.score:.3f}]\n")
    print(ans.format())