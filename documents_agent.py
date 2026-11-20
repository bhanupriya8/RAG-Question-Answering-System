
"""
DocumentAgent — the same agentic retrieval loop as ProMSALiteAgent
(agent/agent_loop.py), ported from "search a landmark knowledge base" to
"search the chunks of whatever text document is currently loaded".

Why this exists: summarize.py currently does one-shot retrieval for text
sources — it dumps up to MAX_TEXT_CHARS of the document into the first
turn and truncates anything beyond that, silently losing content. For long
documents, that means the answer to a question can be sitting in the part
that got cut off.

This module fixes that the same way ProMSALiteAgent already proved works
for images: instead of stuffing everything into context up front, the
document is split into overlapping chunks, and the LLM decides — step by
step — whether it needs to search for more relevant chunks before
answering. Same <thinking>/<tool_call>/<answer> format, same trajectory
logging, same "don't repeat an identical search" behavior.

Retrieval here uses simple TF-IDF + cosine similarity over the chunks
(scikit-learn), not a neural embedding model — deliberately lightweight
since this runs per-document at chat time, not pre-built like the
landmark FAISS index.
"""

import json
import re

DOCUMENT_SYSTEM_PROMPT = """You are a careful document question-answering assistant with access to one \
search tool over the chunks of a document that has been loaded.

At every step, first think, then do exactly ONE of:
  1. Call the search tool to look up a specific part of the document, or
  2. Give your final answer.

Available tool:
  - search_document(query): search the document's chunks for text relevant to a query. Use this when \
you need a specific fact, number, quote, or section that you don't already have in front of you.

Respond in EXACTLY this format, nothing else:

<thinking>your reasoning about what you know and what you still need</thinking>
<tool_call>{"name": "search_document", "query": "your search query"}</tool_call>
OR
<answer>your final answer, direct and complete</answer>

Only ever output ONE tag pair (thinking + exactly one of tool_call/answer) per turn.
If a search returns nothing new, try rephrasing the query once, or answer with your best guess from \
what you have — do not repeat the exact same search twice.
"""

THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


# ---------------------------------------------------------------------------
# Chunking + retrieval over the loaded document
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Splits text into overlapping chunks by character count. Simple and
    dependency-free; good enough for TF-IDF retrieval over one document."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


class DocumentSearchTools:
    """TF-IDF + cosine similarity search over a document's chunks, with the
    same de-dup behavior as SearchTools in tools/search_tools.py: identical
    queries within one question don't return the same chunks twice, nudging
    the agent to rephrase or stop instead of looping."""

    def __init__(self, text: str, chunk_size: int = 1500, overlap: int = 200, top_k: int = 3):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks) if self.chunks else None
        self._seen_chunk_indices = set()

    def reset(self):
        """Clear de-dup memory for a new question."""
        self._seen_chunk_indices = set()

    def search_document(self, query: str) -> dict:
        from sklearn.metrics.pairwise import cosine_similarity

        if self.chunk_vectors is None or len(self.chunks) == 0:
            return {"results": [], "note": "Document is empty."}

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.chunk_vectors)[0]

        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked:
            if idx in self._seen_chunk_indices:
                continue
            if scores[idx] <= 0:
                break  # no more relevant matches
            results.append({
                "title": f"Chunk {idx + 1}/{len(self.chunks)}",
                "extract": self.chunks[idx],
                "score": float(scores[idx]),
            })
            self._seen_chunk_indices.add(idx)
            if len(results) >= self.top_k:
                break

        if not results:
            return {"results": [], "note": "No new relevant sections found for this query."}

        return {"results": results}


# ---------------------------------------------------------------------------
# Summarization of a raw search result (mirrors summarize_result in agent_loop.py)
# ---------------------------------------------------------------------------

def summarize_result(llm_client, question: str, raw_result: dict) -> str:
    if not raw_result.get("results"):
        return raw_result.get("note", "No results found.")

    lines = []
    for r in raw_result["results"]:
        lines.append(f"- {r['title']} (relevance {r['score']:.2f}): {r['extract'][:600]}")
    raw_text = "\n".join(lines)

    prompt = (
        f"Question: {question}\n\n"
        f"Raw document search results:\n{raw_text}\n\n"
        "Summarize only the information relevant to answering the question, in 2-4 sentences."
    )
    summary = llm_client.call(
        system_prompt="You summarize document excerpts concisely and factually.",
        conversation=[{"role": "user", "content": prompt}],
    )
    return summary.strip()


# ---------------------------------------------------------------------------
# The agent loop itself — structurally identical to ProMSALiteAgent
# ---------------------------------------------------------------------------

class DocumentAgent:
    def __init__(self, llm_client, search_tools: DocumentSearchTools, max_steps: int = 5, verbose: bool = True):
        self.llm = llm_client
        self.tools = search_tools
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, question: str) -> dict:
        self.tools.reset()  # clear de-dup memory for this new question

        conversation = [{"role": "user", "content": f"Question: {question}"}]
        trajectory = []

        for step in range(1, self.max_steps + 1):
            reply = self.llm.call(DOCUMENT_SYSTEM_PROMPT, conversation)

            thinking_match = THINKING_RE.search(reply)
            thinking = thinking_match.group(1).strip() if thinking_match else ""

            answer_match = ANSWER_RE.search(reply)
            tool_match = TOOL_CALL_RE.search(reply)

            if self.verbose:
                print(f"\n--- Step {step} ---")
                print(f"Thinking: {thinking}")

            if answer_match:
                answer = answer_match.group(1).strip()
                if self.verbose:
                    print(f"Answer: {answer}")
                trajectory.append({"step": step, "thinking": thinking, "action": "stop", "answer": answer})
                return {"answer": answer, "steps_used": step, "trajectory": trajectory, "stopped": "answer"}

            if not tool_match:
                trajectory.append({"step": step, "thinking": thinking, "action": "format_error", "raw": reply})
                return {"answer": reply.strip(), "steps_used": step, "trajectory": trajectory, "stopped": "format_error"}

            try:
                tool_call = json.loads(tool_match.group(1).strip())
            except json.JSONDecodeError:
                trajectory.append({"step": step, "thinking": thinking, "action": "json_error", "raw": tool_match.group(1)})
                conversation.append({"role": "assistant", "content": reply})
                conversation.append({"role": "user", "content": "Your tool_call JSON was invalid. Please retry with valid JSON."})
                continue

            tool_name = tool_call.get("name")
            if self.verbose:
                print(f"Tool call: {tool_call}")

            if tool_name == "search_document":
                raw_result = self.tools.search_document(tool_call.get("query", question))
            else:
                raw_result = {"results": [], "note": f"Unknown tool '{tool_name}'."}

            summary = summarize_result(self.llm, question, raw_result)
            if self.verbose:
                print(f"Result summary: {summary}")

            trajectory.append({
                "step": step, "thinking": thinking, "action": tool_name,
                "args": tool_call, "raw_result": raw_result, "summary": summary,
            })

            conversation.append({"role": "assistant", "content": reply})
            conversation.append({"role": "user", "content": f"Tool result ({tool_name}): {summary}"})

        if self.verbose:
            print("\n--- Budget exhausted, forcing final answer ---")
        conversation.append({"role": "user", "content": "You are out of search steps. Give your best final answer now, in the <answer> format."})
        reply = self.llm.call(DOCUMENT_SYSTEM_PROMPT, conversation)
        answer_match = ANSWER_RE.search(reply)
        answer = answer_match.group(1).strip() if answer_match else reply.strip()
        trajectory.append({"step": self.max_steps + 1, "action": "forced_stop", "answer": answer})
        return {"answer": answer, "steps_used": self.max_steps + 1, "trajectory": trajectory, "stopped": "budget"}