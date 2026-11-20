
# test_step5.py
import re
from pathlib import Path

from agent.llm_client import DEFAULT_CLIENT
from rag.rag_core import DocumentStore
from rag.chat import ChatSession


def llm(prompt: str) -> str:
    text = DEFAULT_CLIENT.call(
        system_prompt="You are a careful assistant that answers strictly from provided sources.",
        conversation=[{"role": "user", "content": prompt}],
    )
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


store = DocumentStore()
for pdf in sorted(Path("data/pdfs").glob("*.pdf")):
    store.add_pdf(str(pdf))

session = ChatSession(store, llm, max_turns=3, k=5)

conversation = [
    "how does the CYK algorithm parse a sentence?",
    "what are its limitations?",                 # "its" -> CYK
    "how is that different from Earley?",        # "that" -> CYK
    "explain that more simply",                  # "that" -> the Earley comparison
    "what is the capital of France?",            # topic switch, should be LOW
]

for q in conversation:
    print("=" * 70)
    print("USER:", q)
    ans, standalone = session.ask(q)
    if standalone != q:
        print(f"[rewritten as: {standalone}]")
    print(f"[confidence: {ans.confidence.level} | top dense = {ans.confidence.score:.3f}]\n")
    print(ans.format())