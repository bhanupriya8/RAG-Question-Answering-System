
# rag/gemini_llm.py
import os
import re
import time


def make_gemini_llm(model: str = "gemini-3.8-flash", max_per_min: int = 4, timeout_s: int = 60):
    """Standalone Gemini client. Does not use agent/llm_client.py."""
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit('GEMINI_API_KEY is not set in this terminal.\n'
                         'Run: $env:GEMINI_API_KEY="YOUR_KEY"')
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=timeout_s * 1000),   # milliseconds
    )
    times: list[float] = []

    def llm(prompt: str) -> str:
        now = time.time()
        while times and now - times[0] > 60:
            times.pop(0)
        if len(times) >= max_per_min:
            wait = 60 - (now - times[0]) + 1
            print(f"  [waiting {wait:.0f}s to stay under the per-minute limit]")
            time.sleep(wait)
        times.append(time.time())

        last = ""
        for attempt in range(1, 6):
            try:
                print(f"  [calling Gemini, attempt {attempt}...]")
                r = client.models.generate_content(model=model, contents=prompt)
                return re.sub(r"<think>.*?</think>", "", r.text or "", flags=re.DOTALL).strip()
            except Exception as e:
                msg = str(e)
                last = msg[:200]
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    raise RuntimeError("Gemini quota exhausted: " + msg[:200])
                if "404" in msg or "NOT_FOUND" in msg:
                    raise RuntimeError(f"Gemini model '{model}' not found: " + msg[:200])
                print(f"  [retry {attempt}/5] {type(e).__name__}: {msg[:100]}")
                time.sleep(10 * attempt)
        raise RuntimeError("Gemini call failed after 5 attempts. Last error: " + last)

    llm.model_name = f"Gemini:{model}"
    return llm