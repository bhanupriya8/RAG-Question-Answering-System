# rag/gemini_llm.py
import os
import re
import time


def make_gemini_llm(model: str = "gemini-3.8-flash", max_per_min: int = 4, timeout_s: int = 60):
    """Standalone Gemini client. Does not use agent/llm_client.py.

    `model` can be one name or several separated by commas, for example
    "gemini-3.8-flash,gemini-3.6-flash,gemini-3.7-flash". The models are tried in that
    order. If one is out of quota, busy or too slow, the next one is used, and the model
    that worked is tried first on the next question. Every model has its own daily quota,
    so a list gives you several allowances instead of one.
    """
    from google import genai
    from google.genai import types

    models = [m.strip() for m in model.split(",") if m.strip()]
    if not models:
        raise SystemExit("No Gemini model name was given.")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit('GEMINI_API_KEY is not set in this terminal.\n'
                         'Run: $env:GEMINI_API_KEY="YOUR_KEY"')
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=timeout_s * 1000),   # milliseconds
    )
    times: list[float] = []
    state = {"first": 0}        # index of the model that worked last

    def throttle():
        now = time.time()
        while times and now - times[0] > 60:
            times.pop(0)
        if len(times) >= max_per_min:
            wait = 60 - (now - times[0]) + 1
            print(f"  [waiting {wait:.0f}s to stay under the per-minute limit]")
            time.sleep(wait)
        times.append(time.time())

    def llm(prompt: str) -> str:
        order = models[state["first"]:] + models[:state["first"]]
        problems = []
        for name in order:
            for attempt in (1, 2):          # a second try only for slow or busy answers
                throttle()
                try:
                    print(f"  [calling {name}, attempt {attempt}...]")
                    r = client.models.generate_content(model=name, contents=prompt)
                except Exception as e:
                    msg = str(e)
                    short = msg[:90].replace("\n", " ")
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        problems.append(f"{name}: quota used up (429)")
                        print(f"  [{name}: quota used up, trying the next model]")
                        break
                    if "404" in msg or "NOT_FOUND" in msg:
                        problems.append(f"{name}: not available to this key (404)")
                        print(f"  [{name}: not available, trying the next model]")
                        break
                    if "400" in msg or "401" in msg or "403" in msg or "INVALID_ARGUMENT" in msg \
                            or "PERMISSION_DENIED" in msg or "API key" in msg:
                        raise RuntimeError(f"Gemini rejected the request: {msg[:300]}")
                    problems.append(f"{name}: {short}")
                    print(f"  [{name}: {short}]")
                    if attempt == 1:
                        time.sleep(5)
                        continue
                    break
                else:
                    if name != models[state["first"]]:
                        print(f"  [now using {name}]")
                    state["first"] = models.index(name)
                    return re.sub(r"<think>.*?</think>", "", r.text or "", flags=re.DOTALL).strip()
        raise RuntimeError("All Gemini models failed. " + " | ".join(problems))

    llm.model_name = "Gemini:" + ",".join(models)
    return llm