"""
The ProMSA-lite agent loop.

Given an image + question, repeatedly asks the LLM to either:
  - call image_search (look for visually similar landmarks)
  - call text_search  (look up facts by a text query)
  - stop and answer

...until it stops or hits the step budget. Each tool result is summarized
before being added back to the conversation, and repeated searches skip
already-seen results (handled inside SearchTools).

This mirrors ProMSA's <thinking>/<tool_call>/<answer> format, but the
"policy" here is a prompted LLM instead of an RL-trained model.
"""

import json
import re

SYSTEM_PROMPT = """You are a careful visual question-answering assistant with access to two search tools \
over a knowledge base of famous landmarks.

At every step, first think, then do exactly ONE of:
  1. Call a tool to gather more evidence, or
  2. Give your final answer.

Available tools:
  - image_search: find landmarks in the knowledge base that look visually similar to the query image. \
Use this when you are NOT SURE which landmark is in the photo.
  - text_search(query): search landmark facts by a text query. Use this when you KNOW which landmark it \
is but need a specific fact (date, height, location, etc.) to answer the question.

Respond in EXACTLY this format, nothing else:

<thinking>your reasoning about what you know and what you still need</thinking>
<tool_call>{"name": "image_search"}</tool_call>
OR
<tool_call>{"name": "text_search", "query": "your search query"}</tool_call>
OR
<answer>your final answer, short and direct</answer>

Only ever output ONE tag pair (thinking + exactly one of tool_call/answer) per turn.
If a tool returns no new results, try the other tool or answer with your best guess - do not repeat \
the same search.
"""

THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def summarize_result(llm_client, question: str, tool_name: str, raw_result: dict) -> str:
    """Condense a raw search result into a short, question-relevant summary
    before feeding it back into the agent's context (keeps trajectories short,
    mirrors the paper's question-conditioned summarization)."""
    if not raw_result.get("results"):
        return raw_result.get("note", "No results found.")

    lines = []
    for r in raw_result["results"]:
        lines.append(f"- {r['title']}: {r['extract'][:400]}")
    raw_text = "\n".join(lines)

    prompt = (
        f"Question: {question}\n\n"
        f"Raw {tool_name} results:\n{raw_text}\n\n"
        "Summarize only the information relevant to answering the question, in 2-3 sentences. "
        "Mention landmark names explicitly."
    )
    summary = llm_client.call(
        system_prompt="You summarize search results concisely and factually.",
        conversation=[{"role": "user", "content": prompt}],
    )
    return summary.strip()


class ProMSALiteAgent:
    def __init__(self, llm_client, search_tools, max_steps: int = 5, verbose: bool = True):
        self.llm = llm_client
        self.tools = search_tools
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, image_path: str, question: str) -> dict:
        self.tools.reset()  # clear de-dup memory for this new question

        conversation = [{"role": "user", "content": f"Question: {question}"}]
        trajectory = []  # log of every step, for your report/eval

        for step in range(1, self.max_steps + 1):
            reply = self.llm.call(SYSTEM_PROMPT, conversation, image_path=image_path)

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
                # model didn't follow the format - treat its raw reply as a fallback answer
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

            if tool_name == "image_search":
                raw_result = self.tools.image_search(image_path)
            elif tool_name == "text_search":
                raw_result = self.tools.text_search(tool_call.get("query", question))
            else:
                raw_result = {"results": [], "note": f"Unknown tool '{tool_name}'."}

            summary = summarize_result(self.llm, question, tool_name, raw_result)
            if self.verbose:
                print(f"Result summary: {summary}")

            trajectory.append({
                "step": step, "thinking": thinking, "action": tool_name,
                "args": tool_call, "raw_result": raw_result, "summary": summary,
            })

            conversation.append({"role": "assistant", "content": reply})
            conversation.append({"role": "user", "content": f"Tool result ({tool_name}): {summary}"})

        # budget exhausted without a stop - force a final answer
        if self.verbose:
            print("\n--- Budget exhausted, forcing final answer ---")
        conversation.append({"role": "user", "content": "You are out of search steps. Give your best final answer now, in the <answer> format."})
        reply = self.llm.call(SYSTEM_PROMPT, conversation, image_path=image_path)
        answer_match = ANSWER_RE.search(reply)
        answer = answer_match.group(1).strip() if answer_match else reply.strip()
        trajectory.append({"step": self.max_steps + 1, "action": "forced_stop", "answer": answer})
        return {"answer": answer, "steps_used": self.max_steps + 1, "trajectory": trajectory, "stopped": "budget"}
