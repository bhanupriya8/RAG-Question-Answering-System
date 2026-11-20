
# rag/llm_adapter.py
import re


def make_llm():
    """Wrap agent.llm_client.DEFAULT_CLIENT as llm(prompt) -> str."""
    from agent.llm_client import DEFAULT_CLIENT

    def llm(prompt: str) -> str:
        text = DEFAULT_CLIENT.call(
            system_prompt="You are a careful assistant that answers strictly from provided sources.",
            conversation=[{"role": "user", "content": prompt}],
        )
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return llm