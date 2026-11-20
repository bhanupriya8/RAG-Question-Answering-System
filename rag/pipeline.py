
# rag/pipeline.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rag.agentic import agentic_answer
from rag.answer import Answer, answer_question
from rag.chat import ChatSession
from rag.rag_core import DocumentStore


@dataclass
class QAResult:
    answer: Answer
    standalone_query: str
    trace: list[dict] = field(default_factory=list)
    llm_calls: int = 0


class DocumentQA:
    """One object that combines: multi-doc store, hybrid search, confidence,
    citations, conversational memory, and the agentic loop."""

    def __init__(self, llm: Callable[[str], str], mode: str = "agentic",
                 max_turns: int = 3, max_steps: int = 3, k: int = 4):
        self.llm = llm
        self.mode = mode                       # "agentic" or "single"
        self.max_steps = max_steps
        self.k = k
        self.store = DocumentStore()
        self.session = ChatSession(self.store, llm, max_turns=max_turns, k=k)

    # ---- documents ----
    def add_pdf(self, path: str) -> int:
        return self.store.add_pdf(path)

    def add_folder(self, folder: str) -> dict[str, int]:
        added = {}
        for pdf in sorted(Path(folder).glob("*.pdf")):
            added[pdf.name] = self.add_pdf(str(pdf))
        return added

    def documents(self) -> list[str]:
        return sorted({c.doc for c in self.store.chunks})

    # ---- question answering ----
    def ask(self, query: str, docs: list[str] | None = None) -> QAResult:
        # standalone = self.session.rewrite(query)       # memory: resolve "it", "that"...
        # calls = 1 if standalone != query else 0         # a rewrite costs one LLM call
        
        had_history = bool(self.session.history)
        standalone = self.session.rewrite(query)
        calls = 1 if had_history else 0

        if self.mode == "agentic":
            res = agentic_answer(self.store, standalone, self.llm,
                                 max_steps=self.max_steps, k=self.k, docs=docs)
            ans, trace, calls = res.answer, res.trace, calls + res.llm_calls
        else:
            # ans = answer_question(self.store, standalone, self.llm, k=self.k, docs=docs)
            # trace = []
            # calls += 0 if ans.confidence.level == "low" else 1
            
        
            ans = answer_question(self.store, standalone, self.llm, k=self.k, docs=docs)
            trace = []
            calls += 1 if ans.sources else 0

        self.session.history.append((query, ans.text))
        return QAResult(ans, standalone, trace, calls)

    def reset(self):
        self.session.reset()