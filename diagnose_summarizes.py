
# diagnose_summarizers.py
"""
Finds out why the Part C summarizer scripts do not work on this computer.
It prints a short report. It never prints your API keys.

Run from the RAG_QA folder:
    python diagnose_summarizers.py
Optional checks (each one tests one thing):
    python diagnose_summarizers.py --pdf "data/pdfs/file.pdf"      # can text be read from this PDF?
    python diagnose_summarizers.py --youtube VIDEO_ID_OR_URL        # can the transcript be fetched?
    python diagnose_summarizers.py --llm                            # sends ONE tiny Gemini request
    python diagnose_summarizers.py --llm --model gemini-3.8-flash   # ... with another model name
"""

import argparse
import importlib
import importlib.metadata
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PACKAGES = [
    ("pypdf", "pypdf", "summarize_pdf.py and the document QA system"),
    ("google.genai", "google-genai", "every Gemini call"),
    ("anthropic", "anthropic", "the Claude fallback in summarize_pdf_native.py and summarize_pdf_chat.py"),
    ("playwright.sync_api", "playwright", "web page URLs in summarize_pdf_chat.py"),
    ("pandas", "pandas", "CSV and Excel files in summarize_pdf_chat.py"),
    ("openpyxl", "openpyxl", "Excel files in summarize_pdf_chat.py"),
    ("youtube_transcript_api", "youtube-transcript-api", "summarize_youtube.py"),
    ("dotenv", "python-dotenv", "agent/llm_client.py"),
]


def section(title):
    print(f"\n=== {title} ===")


def check_packages():
    section("1. Python packages")
    for module, pip_name, needed_for in PACKAGES:
        try:
            importlib.import_module(module)
            print(f"  ok       {pip_name}")
        except Exception as e:
            print(f"  MISSING  {pip_name:24s} -> pip install {pip_name}   (needed for {needed_for})"
                  if isinstance(e, ImportError) else f"  ERROR    {pip_name}: {type(e).__name__}: {e}")


def check_keys():
    section("2. API keys in this terminal (values are never shown)")
    gem = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    print(f"  Gemini key set:    {'yes' if gem else 'NO  -> $env:GEMINI_API_KEY=\"YOUR_KEY\"'}")
    print(f"  Anthropic key set: {'yes' if os.environ.get('ANTHROPIC_API_KEY') else 'no  (the Claude fallback cannot work without it)'}")


def check_default_client():
    section("3. Client used by summarize_pdf.py and summarize_youtube.py (agent/llm_client.py)")
    try:
        from agent.llm_client import DEFAULT_CLIENT
        print(f"  {type(DEFAULT_CLIENT).__name__}  model={getattr(DEFAULT_CLIENT, 'model', '?')}")
    except Exception as e:
        print(f"  FAILED to build it: {type(e).__name__}: {str(e)[:250]}")


def check_playwright():
    section("4. Browser for web page URLs (summarize_pdf_chat.py)")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            path = p.chromium.executable_path
        if os.path.exists(path):
            print("  ok       Chromium is installed")
        else:
            print("  MISSING  Chromium -> run: playwright install chromium")
    except Exception as e:
        print(f"  not usable: {type(e).__name__}: {str(e)[:200]}")


def check_audio_support():
    section("5. File types accepted by summarize_pdf_chat.py")
    f = ROOT / "summarize_pdf_chat.py"
    if not f.exists():
        print("  summarize_pdf_chat.py not found in this folder")
        return
    src = f.read_text(encoding="utf-8", errors="ignore")
    audio_exts = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac")
    if any(ext in src for ext in audio_exts):
        print("  audio file types are mentioned in the script")
    else:
        print("  AUDIO IS NOT SUPPORTED: the script has no audio file types, so .mp3/.wav files are rejected")
    print("  supported: pdf, csv, xlsx/xls, txt/md, images, video, URLs")


def check_pdf(path):
    section(f"6. Reading text from {path}")
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        chars = sum(len((pg.extract_text() or "")) for pg in reader.pages)
        print(f"  pages: {len(reader.pages)}   characters of text: {chars}")
        if chars == 0:
            print("  NO TEXT: this is a scanned PDF (images only). summarize_pdf.py cannot read it; "
                  "summarize_pdf_native.py (Gemini) can.")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:250]}")


def check_youtube(value):
    section(f"7. YouTube transcript for {value}")
    m = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", value) or re.match(r"^([0-9A-Za-z_-]{11})$", value)
    if not m:
        print("  could not find an 11-character video id in that text")
        return
    vid = m.group(1)
    try:
        print(f"  youtube-transcript-api version: {importlib.metadata.version('youtube-transcript-api')}")
    except Exception:
        pass
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        snippets = YouTubeTranscriptApi().fetch(vid)
        words = sum(len(s.text.split()) for s in snippets)
        print(f"  ok       {len(snippets)} caption lines, about {words} words")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
        print("  Common reasons: the video has no captions, captions are disabled, YouTube is blocking "
              "requests from this network, or the library version differs from what the script expects.")


def check_llm(model):
    section(f"8. One tiny Gemini request with {model}")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("  skipped: no Gemini key in this terminal")
        return
    t0 = time.time()
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60000))
        r = client.models.generate_content(model=model, contents="Reply with exactly one word: hello")
        print(f"  ok       answered in {time.time() - t0:.1f}s: {(r.text or '').strip()[:40]!r}")
    except Exception as e:
        print(f"  FAILED after {time.time() - t0:.1f}s")
        print(f"  {type(e).__name__}: {str(e)[:400]}")
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            print("  -> quota used up for this model. Wait for the reset, or try another model.")
        elif "503" in msg or "504" in msg:
            print("  -> Google is overloaded or slow. Try again later, or try another model.")
        elif "404" in msg or "NOT_FOUND" in msg:
            print("  -> this model name is not available to your account. Use a name from your model list.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf")
    ap.add_argument("--youtube")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default="gemini-3.7-flash")
    args = ap.parse_args()

    check_packages()
    check_keys()
    check_default_client()
    check_playwright()
    check_audio_support()
    if args.pdf:
        check_pdf(args.pdf)
    if args.youtube:
        check_youtube(args.youtube)
    if args.llm:
        check_llm(args.model)
    print("\nDone. Copy everything above into the chat.")


if __name__ == "__main__":
    main()