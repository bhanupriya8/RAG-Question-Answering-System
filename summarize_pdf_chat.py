
"""
Load a PDF, a URL (news article/webpage), an image, a video, or a data file
(CSV, Excel .xlsx/.xls, plain text/markdown) using NATIVE multimodal
understanding where applicable, then wait for you to say what you want:
summarize it, or ask questions about it. Keeps context across the session.

- URLs to normal webpages are rendered to PDF via a headless browser
  (Playwright), preserving images and layout -- not just scraped text.
- URLs or local files that ARE directly an image/video are detected and
  sent natively as that media type.
- IMPORTANT: Gemini supports video natively; Claude's API currently does
  NOT. So video sources only work via Gemini -- if Gemini fails, there is
  no working fallback for video, and the script will say so clearly rather
  than silently failing in a confusing way.

Tries Gemini first (3.7-flash, then 3.6-flash as backup), falling back to
Claude if both fail (for non-video sources) -- checked on EVERY turn, not
just the first, so a mid-session outage still recovers.

Env vars used:
    GEMINI_API_KEY or GOOGLE_API_KEY   -> for Gemini
    ANTHROPIC_API_KEY                  -> for Claude fallback

Setup (one-time):
    pip install playwright anthropic pandas openpyxl
    playwright install chromium

Usage:
    python summarize_pdf_chat.py "path/to/document.pdf"
    python summarize_pdf_chat.py "path/to/data.csv"
    python summarize_pdf_chat.py "path/to/photo.jpg"
    python summarize_pdf_chat.py "path/to/clip.mp4"
    python summarize_pdf_chat.py "https://example.com/some-article"
    python summarize_pdf_chat.py "https://example.com/photo.png"
    python summarize_pdf_chat.py "https://example.com/video.mp4"

Then just type what you want at the prompt. Type 'exit' or 'quit' to stop.
"""

import sys
import os
import time
import mimetypes
import tempfile
import atexit
from urllib.parse import urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SUMMARY_PROMPT = (
    "Summarize this. Give a concise overview, then list the key points as "
    "bullet points. If there are any charts, diagrams, tables, or images, "
    "describe what they show and how they relate to the surrounding "
    "content."
)

SUMMARY_KEYWORDS = ("summar",)

CONTENT_FOCUS_INSTRUCTION = (
    "This may be a PDF, a full webpage capture, an image, a video, or a "
    "data file (CSV/Excel). If it's a webpage capture, it could contain "
    "advertisements, promotional banners (e.g. sales, subscription "
    "offers), site navigation menus, cookie/consent notices, and other "
    "page chrome that is NOT part of the actual content -- ignore all of "
    "that entirely. Focus only on the real article/document/data/media "
    "content. Do still pay attention to genuine content images tied to the "
    "story itself (photos, figures, charts, diagrams) -- just skip purely "
    "decorative or promotional images like banner ads."
)

MAX_TEXT_CHARS = 400_000

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def get_url_content_type(url: str) -> str | None:
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=15) as resp:
            return resp.headers.get("Content-Type", "").split(";")[0].strip()
    except Exception:
        return None


def download_url_to_temp(url: str, suffix: str) -> tuple[str, str]:
    """Downloads a URL to a temp file. Returns (temp_path, content_type)."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as resp:
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp_path = tmp.name
    return tmp_path, content_type


def guess_url_kind(url: str) -> str:
    """Returns 'image', 'video', or 'webpage'."""
    path_ext = os.path.splitext(urlparse(url).path)[1].lower()
    if path_ext in IMAGE_EXTS:
        return "image"
    if path_ext in VIDEO_EXTS:
        return "video"
    content_type = get_url_content_type(url)
    if content_type:
        if content_type.startswith("image/"):
            return "image"
        if content_type.startswith("video/"):
            return "video"
    return "webpage"


def fetch_url_as_pdf(url: str) -> bytes:
    """Renders a webpage to a PDF using a headless browser, so images, charts,
    and layout are preserved (not just scraped text)."""
    from playwright.sync_api import sync_playwright

    print(f"Fetching URL and rendering to PDF (preserving images): {url}")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until="load", timeout=60000)
            page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(3000)
            page.pdf(path=tmp_path, print_background=True, format="A4")
            browser.close()

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Tabular / text helpers
# ---------------------------------------------------------------------------

def load_csv_as_text(path: str) -> str:
    import pandas as pd

    df = pd.read_csv(path)
    header = (
        f"CSV file: {os.path.basename(path)}\n"
        f"Rows: {len(df)}, Columns: {list(df.columns)}\n\n"
    )
    return header + df.to_csv(index=False)


def load_excel_as_text(path: str) -> str:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None)
    chunks = [f"Excel file: {os.path.basename(path)} ({len(sheets)} sheet(s))\n"]
    for name, df in sheets.items():
        chunks.append(
            f"\n--- Sheet: {name} (Rows: {len(df)}, Columns: {list(df.columns)}) ---\n"
            + df.to_csv(index=False)
        )
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Unified source loader
# ---------------------------------------------------------------------------

def load_source(source: str) -> dict:
    """Returns a dict describing the source:
      {"kind": "pdf",   "bytes": ..., "name": ...}
      {"kind": "text",  "content": ..., "name": ...}
      {"kind": "image", "bytes": ..., "mime": ..., "name": ...}
      {"kind": "video", "path": ...,  "mime": ..., "name": ..., "is_temp": bool}
    """
    if source.startswith("http://") or source.startswith("https://"):
        url_kind = guess_url_kind(source)
        name = os.path.basename(urlparse(source).path) or source

        if url_kind == "image":
            ext = os.path.splitext(urlparse(source).path)[1] or ".jpg"
            tmp_path, content_type = download_url_to_temp(source, ext)
            mime = content_type or mimetypes.guess_type(tmp_path)[0] or "image/jpeg"
            with open(tmp_path, "rb") as f:
                data = f.read()
            os.remove(tmp_path)
            return {"kind": "image", "bytes": data, "mime": mime, "name": name}

        if url_kind == "video":
            ext = os.path.splitext(urlparse(source).path)[1] or ".mp4"
            tmp_path, content_type = download_url_to_temp(source, ext)
            mime = content_type or mimetypes.guess_type(tmp_path)[0] or "video/mp4"
            atexit.register(lambda p=tmp_path: os.path.exists(p) and os.remove(p))
            return {"kind": "video", "path": tmp_path, "mime": mime, "name": name, "is_temp": True}

        return {"kind": "pdf", "bytes": fetch_url_as_pdf(source), "name": source}

    if not os.path.exists(source):
        raise FileNotFoundError(f"File not found at '{source}'")

    ext = os.path.splitext(source)[1].lower()
    name = os.path.basename(source)

    if ext == ".pdf":
        with open(source, "rb") as f:
            return {"kind": "pdf", "bytes": f.read(), "name": name}
    elif ext == ".csv":
        return {"kind": "text", "content": load_csv_as_text(source), "name": name}
    elif ext in (".xlsx", ".xls"):
        return {"kind": "text", "content": load_excel_as_text(source), "name": name}
    elif ext in (".txt", ".md"):
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            return {"kind": "text", "content": f.read(), "name": name}
    elif ext in IMAGE_EXTS:
        mime = mimetypes.guess_type(source)[0] or "image/jpeg"
        with open(source, "rb") as f:
            return {"kind": "image", "bytes": f.read(), "mime": mime, "name": name}
    elif ext in VIDEO_EXTS:
        mime = mimetypes.guess_type(source)[0] or "video/mp4"
        return {"kind": "video", "path": source, "mime": mime, "name": name, "is_temp": False}
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: .pdf, .csv, .xlsx, "
            f".xls, .txt, .md, common image/video formats, or a URL."
        )


def build_first_turn_text(source: dict, user_text: str) -> str:
    if source["kind"] == "text":
        content = source["content"]
        truncated_note = ""
        if len(content) > MAX_TEXT_CHARS:
            content = content[:MAX_TEXT_CHARS]
            truncated_note = "\n\n[... content truncated due to size ...]"
        return (
            f"{CONTENT_FOCUS_INSTRUCTION}\n\n"
            f"[Attached file: {source['name']}]\n\n"
            f"{content}{truncated_note}\n\n---\n\n{user_text}"
        )
    return f"{CONTENT_FOCUS_INSTRUCTION}\n\n{user_text}"


# ---------------------------------------------------------------------------
# Provider sessions
# ---------------------------------------------------------------------------

class GeminiSession:
    """Manages conversation history manually and calls generate_content each
    turn. Retries across candidate models on EVERY turn. Handles pdf/text/
    image inline; video is uploaded via the Files API (Gemini processes it
    in the background, so we poll until it's ready)."""

    def __init__(self, source: dict, models=("gemini-3.7-flash", "gemini-3.6-flash")):
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.\n"
                'In PowerShell run: $env:GEMINI_API_KEY="YOUR_API_KEY"'
            )

        self.client = genai.Client(api_key=api_key)
        self.candidate_models = list(models)
        self.model = None
        self.types = types
        self.source = source
        self._sent = False
        self._uploaded_video = None
        self.history = []

    def _get_uploaded_video(self):
        if self._uploaded_video is not None:
            return self._uploaded_video

        print("Uploading video to Gemini (this may take a moment)...")
        uploaded = self.client.files.upload(file=self.source["path"])
        while getattr(uploaded.state, "name", uploaded.state) == "PROCESSING":
            time.sleep(3)
            uploaded = self.client.files.get(name=uploaded.name)

        state = getattr(uploaded.state, "name", uploaded.state)
        if state == "FAILED":
            raise RuntimeError("Gemini failed to process the uploaded video.")

        self._uploaded_video = uploaded
        return uploaded

    def send(self, text: str) -> str:
        types = self.types
        is_first = not self._sent

        prompt_text = build_first_turn_text(self.source, text) if is_first else text
        parts = [types.Part.from_text(text=prompt_text)]

        if is_first:
            kind = self.source["kind"]
            if kind == "pdf":
                parts.append(
                    types.Part.from_bytes(data=self.source["bytes"], mime_type="application/pdf")
                )
            elif kind == "image":
                parts.append(
                    types.Part.from_bytes(data=self.source["bytes"], mime_type=self.source["mime"])
                )
            elif kind == "video":
                uploaded = self._get_uploaded_video()
                parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type))

        self.history.append(types.Content(role="user", parts=parts))

        ordered = ([self.model] if self.model else []) + [
            m for m in self.candidate_models if m != self.model
        ]
        response = None
        last_error = None
        for candidate in ordered:
            try:
                if candidate != self.model:
                    print(f"Trying Gemini ({candidate})...")
                response = self.client.models.generate_content(
                    model=candidate, contents=self.history
                )
                self.model = candidate
                break
            except Exception as e:
                print(f"  {candidate} failed: {e}")
                last_error = e
                continue
        if response is None:
            self.history.pop()
            raise last_error

        if is_first:
            self._sent = True

        answer = response.text
        self.history.append(response.candidates[0].content)
        return answer


class ClaudeSession:
    """Manages conversation history manually. Handles pdf/text/image.
    Claude's API does not support video input, so video sources raise a
    clear error immediately rather than failing confusingly."""

    def __init__(self, source: dict, model: str = "claude-sonnet-5"):
        import anthropic

        if source["kind"] == "video":
            raise RuntimeError(
                "Claude does not support video input -- only Gemini can "
                "process video sources in this script."
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set.\n"
                'In PowerShell run: $env:ANTHROPIC_API_KEY="YOUR_API_KEY"'
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.source = source
        self._sent = False
        self.messages = []

    def send(self, text: str) -> str:
        import base64

        is_first = not self._sent
        content = []

        if is_first:
            content.append(
                {"type": "text", "text": build_first_turn_text(self.source, text)}
            )
            kind = self.source["kind"]
            if kind == "pdf":
                pdf_b64 = base64.standard_b64encode(self.source["bytes"]).decode("utf-8")
                content.append(
                    {
                        "type": "document",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64},
                    }
                )
            elif kind == "image":
                img_b64 = base64.standard_b64encode(self.source["bytes"]).decode("utf-8")
                content.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": self.source["mime"], "data": img_b64},
                    }
                )
            self._sent = True
        else:
            content.append({"type": "text", "text": text})

        self.messages.append({"role": "user", "content": content})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=self.messages,
        )
        answer = response.content[0].text
        self.messages.append({"role": "assistant", "content": answer})
        return answer


class DualProviderSession:
    """Tries Gemini first; falls back to Claude if it fails (except for
    video sources, where Claude has no support at all). Checked every turn."""

    def __init__(self, source: dict):
        self.source = source
        self.gemini = None
        self.claude = None
        self.provider = None

    def send(self, text: str):
        if self.provider == "Gemini":
            try:
                return self.gemini.send(text), f"Gemini ({self.gemini.model})"
            except Exception as gemini_error:
                print(f"Gemini failed this turn: {gemini_error}")
                if self.source["kind"] == "video":
                    raise RuntimeError(
                        f"Gemini failed and video sources have no Claude fallback.\n"
                        f"Gemini error: {gemini_error}"
                    )
                print("Falling back to Claude for this question...")
                try:
                    if self.claude is None:
                        self.claude = ClaudeSession(self.source)
                        print(
                            "Note: Claude is joining mid-session, so it won't have "
                            "the earlier back-and-forth as context -- just this question."
                        )
                    answer = self.claude.send(text)
                    self.provider = "Claude"
                    return answer, "Claude"
                except Exception as claude_error:
                    raise RuntimeError(
                        f"Both providers failed.\nGemini error: {gemini_error}\n"
                        f"Claude error: {claude_error}"
                    )

        if self.provider == "Claude":
            return self.claude.send(text), "Claude"

        try:
            if self.gemini is None:
                self.gemini = GeminiSession(self.source)
            answer = self.gemini.send(text)
            self.provider = "Gemini"
            return answer, f"Gemini ({self.gemini.model})"
        except Exception as gemini_error:
            print(f"Gemini failed: {gemini_error}")
            if self.source["kind"] == "video":
                raise RuntimeError(
                    f"Gemini failed and video sources have no Claude fallback.\n"
                    f"Gemini error: {gemini_error}"
                )
            print("Falling back to Claude...")
            try:
                if self.claude is None:
                    self.claude = ClaudeSession(self.source)
                answer = self.claude.send(text)
                self.provider = "Claude"
                return answer, "Claude"
            except Exception as claude_error:
                raise RuntimeError(
                    f"Both providers failed.\nGemini error: {gemini_error}\n"
                    f"Claude error: {claude_error}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_pdf_chat.py <path_or_url>")
        print("Supports: PDF, CSV, Excel, txt/md, image, video, or a URL to any of these.")
        return

    source_str = sys.argv[1]

    print(f"\nSource: {source_str}")
    try:
        source = load_source(source_str)
    except Exception as e:
        print(f"Failed to load source: {e}")
        return

    print(f"Detected type: {source['kind']}")
    session = DualProviderSession(source)

    print("\nLoaded. How may I help you?")
    print('What do you want me to do with this? (e.g. "summarize it" or ask a question)')
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit", ""):
            print("Ending session.")
            break

        prompt = SUMMARY_PROMPT if any(k in user_input.lower() for k in SUMMARY_KEYWORDS) else user_input

        try:
            answer, provider = session.send(prompt)
            print(f"\n{provider}: {answer}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()