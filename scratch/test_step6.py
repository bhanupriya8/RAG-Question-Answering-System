
# test_step6.py
import re
from pathlib import Path

from agent.llm_client import DEFAULT_CLIENT
from rag.rag_core import DocumentStore
from rag.answer import answer_question
from rag.agentic import agentic_answer


def llm(prompt: str) -> str:
    text = DEFAULT_CLIENT.call(
        system_prompt="You are a careful assistant that answers strictly from provided sources.",
        conversation=[{"role": "user", "content": prompt}],
    )
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


store = DocumentStore()
for pdf in sorted(Path("data/pdfs").glob("*.pdf")):
    store.add_pdf(str(pdf))


def run_agent(q):
    res = agentic_answer(store, q, llm, max_steps=3, k=4)
    print("\n--- AGENTIC ---")
    for t in res.trace:
        print(f"  step {t['step']}: query='{t['query']}' | +{t['new_chunks']} chunks | {t['decision']}")
    print(f"  LLM calls: {res.llm_calls}\n")
    print(res.answer.format())


# 1) compare single-pass vs agentic on the question that missed a point before
q1 = "what are the advantages and limitations of CYK, and how does Earley overcome them?"
print("=" * 70)
print("Q:", q1)
print("\n--- SINGLE PASS ---")
print(answer_question(store, q1, llm, k=4).format())
run_agent(q1)

# 2) multi-part question that needs more than one search
q2 = "explain top-down parsing and also what the Earley predictor does"
print("=" * 70)
print("Q:", q2)
run_agent(q2)

# 3) off-topic: should stop at step 1 with zero LLM calls
q3 = "what is the capital of France?"
print("=" * 70)
print("Q:", q3)
run_agent(q3)