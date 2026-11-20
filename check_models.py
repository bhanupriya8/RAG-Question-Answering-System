# check_models.py
"""
Tests which Gemini models work for your key RIGHT NOW. It sends one tiny request to each model
and says, in plain words, why a model fails. It changes nothing in your project.

Run from the RAG_QA folder (your key must be set in this terminal):
    $env:GEMINI_API_KEY="YOUR_KEY"
    python check_models.py
Or test only some models:
    python check_models.py gemini-3.8-flash gemini-3.5-flash-lite
"""

import os
import re
import sys
import time

DEFAULT_MODELS = [
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite",
]


def classify(msg: str) -> str:
    """Turns a Gemini error into one plain sentence."""
    low = msg.lower()
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        if "perday" in low or "per day" in low:
            return "DAILY quota used up. It comes back after the daily reset, not in a few minutes."
        if "perminute" in low:
            m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
            wait = f" Try again in about {float(m.group(1)):.0f}s." if m else ""
            return "Per-minute limit hit. The model itself is fine." + wait
        return "Quota used up (429)."
    if "404" in msg or "NOT_FOUND" in msg:
        return "Not available to this key (404). Google may have retired it for new users."
    if "503" in msg or "UNAVAILABLE" in msg:
        return "Google is overloaded right now (503). Try again later."
    if "504" in msg or "DEADLINE" in msg or "timed out" in low or "timeout" in low:
        return "Too slow, timed out."
    if any(c in msg for c in ("400", "401", "403")) or "API key" in msg or "PERMISSION_DENIED" in msg:
        return "Key problem or request rejected: " + msg[:150].replace("\n", " ")
    return "Other problem: " + msg[:150].replace("\n", " ")


def main():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit('No key in this terminal. Run:  $env:GEMINI_API_KEY="YOUR_KEY"')
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30000))
    models = sys.argv[1:] or DEFAULT_MODELS
    working = []
    print(f"Testing {len(models)} model(s) with one tiny request each ...\n")
    for name in models:
        t0 = time.time()
        try:
            r = client.models.generate_content(model=name, contents="Reply with exactly one word: hello")
            print(f"  OK      {name:<26} answered in {time.time() - t0:.1f}s: {(r.text or '').strip()[:20]!r}")
            working.append(name)
        except Exception as e:
            print(f"  FAILED  {name:<26} {classify(str(e))}")
        time.sleep(1)

    print()
    if working:
        print("Models that work right now. Paste this line into the Model box in the sidebar, then click Connect Gemini:")
        print("  " + ",".join(working))
    else:
        print("No model worked. If most say DAILY quota, wait for the reset. If they say key problem, check the key.")


if __name__ == "__main__":
    main()