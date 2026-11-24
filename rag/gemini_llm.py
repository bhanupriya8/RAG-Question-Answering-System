import os
import re
from google import genai
from google.genai import types
def make_gemini_llm(
    model: str = "gemini-3.1-flash-lite",
    timeout_s: int = 60,
):
    """Create a Gemini LLM function compatible with the RAG pipeline."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. "
            "Set it before connecting to Gemini."
        )
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=timeout_s * 1000),
    )
    def llm(prompt: str) -> str:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        answer = response.text or ""
        answer = re.sub(
            r"<think>.*?</think>",
            "",
            answer,
            flags=re.DOTALL,
        )
        return answer.strip()
    llm.model_name = f"Gemini:{model}"
    return llm
