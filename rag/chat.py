
# rag/chat.py
import re
from typing import Callable

from rag.answer import answer_question, Answer

REWRITE_PROMPT = """Rewrite the user's latest question so it is fully self-contained,
using the conversation history to resolve references like "it", "that", "this one",
"the second one", or "what about section 3".
If the question is already self-contained, return it unchanged.
Return ONLY the rewritten question, nothing else.

CONVERSATION HISTORY:
{history}

LATEST QUESTION: {query}
STANDALONE QUESTION:"""


class ChatSession:
    def __init__(self, store, llm: Callable[[str], str], max_turns: int = 3, k: int = 5):
        self.store = store
        self.llm = llm
        self.max_turns = max_turns      # how many past Q/A pairs to remember
        self.k = k
        self.history: list[tuple[str, str]] = []

    def _format_history(self) -> str:
        turns = self.history[-self.max_turns:]
        # truncate long answers so the rewrite prompt stays small
        return "\n".join(f"User: {q}\nAssistant: {a[:300]}" for q, a in turns)

    def rewrite(self, query: str) -> str:
        if not self.history:
            return query                # first question: nothing to resolve
        prompt = REWRITE_PROMPT.format(history=self._format_history(), query=query)
        try:
            out = self.llm(prompt).strip().strip('"')
            out = re.sub(r"^(standalone question:)\s*", "", out, flags=re.I)
        except Exception:
            return query                # if the LLM fails, fall back to the raw query
        # sanity check: reject empty or runaway output
        if not out or len(out) > 300 or "\n" in out:
            return query
        return out

    def ask(self, query: str, docs: list[str] | None = None) -> tuple[Answer, str]:
        """Returns (answer, standalone_query) so you can show/log the rewrite."""
        standalone = self.rewrite(query)
        ans = answer_question(self.store, standalone, self.llm, k=self.k, docs=docs)
        self.history.append((query, ans.text))
        return ans, standalone

    def reset(self):
        self.history.clear()