
"""
Load a PDF, a URL (news article/webpage), an image, a video, an audio file,
a YouTube link, or a data file (CSV, Excel .xlsx/.xls, plain text/markdown),
then wait for you to say what you want: summarize it, or ask questions
about it. Keeps context across the session.

TWO RETRIEVAL MODES, chosen automatically by source type:

  - Native multimodal (PDF / image / video / audio): the whole file is sent
    directly to Gemini (falling back to Claude for pdf/image, since Claude
    has no video/audio support) in one shot per turn. Best when the model
    genuinely needs to "see" the actual file (layout, charts, visuals,
    spoken audio).

  - Agentic multi-step retrieval (CSV / Excel / txt / md / YouTube
    transcript -- i.e. anything that's plain text): instead of dumping the
    whole document into the prompt and truncating anything past
    MAX_TEXT_CHARS, the text is split into overlapping chunks and a
    DocumentAgent (agent/document_agent.py) decides, step by step, whether
    it needs to search for more relevant chunks before answering -- the
    exact same <thinking>/<tool_call>/<answer> loop already proven in
    agent/agent_loop.py (ProMSALiteAgent) for the landmark image project,
    just searching document chunks instead of a landmark knowledge base.
    This means long documents no longer silently lose content past a
    truncation cutoff -- the agent searches for what it actually needs.

This single file replaces / supersedes:
    summarize_pdf.py          (plain-text PDF summary via a generic LLM client)
    summarize_pdf_native.py   (one-shot native multimodal PDF summary)
    summarize_pdf_chat.py     (native multimodal, any source, chat loop)
    summarize_youtube.py      (transcript-based YouTube summary/Q&A)

- URLs to normal webpages are rendered to PDF via a headless browser
  (Playwright), preserving images and layout -- not just scraped text.
- URLs or local files that ARE directly an image/video/audio file are
  detected and sent natively as that media type.
- YouTube URLs (or bare 11-char video IDs) are detected and their
  transcript/captions are pulled via youtube-transcript-api -- no video
  download needed. The transcript then goes through the agentic
  DocumentAgent path, same as any other text source.
- IMPORTANT: Gemini supports video and audio natively; Claude's API
  currently does NOT support either. So local/URL video and audio sources
  only work via Gemini -- if Gemini fails, there is no working fallback
  for video/audio, and the script will say so clearly rather than
  silently failing in a confusing way.

For native multimodal sources (pdf/image/video/audio), tries Gemini first
(3.7-flash, then 3.6-flash as backup), falling back to Claude if both fail
(except for video/audio, which Claude cannot handle at all) -- checked on
EVERY turn, not just the first, so a mid-session outage still recovers.

For text sources, the DocumentAgent uses whichever client is set as
DEFAULT_CLIENT in agent/llm_client.py (Gemini/Groq/OpenAI/OpenRouter --
whichever you've configured there), since that's the same client your
landmark agent already uses.

Env vars used:
    GEMINI_API_KEY or GOOGLE_API_KEY   -> for Gemini (native multimodal sources)
    ANTHROPIC_API_KEY                  -> for Claude fallback (native multimodal, non-video/audio)
    (whatever DEFAULT_CLIENT in agent/llm_client.py needs)  -> for text sources via DocumentAgent

Setup (one-time):
    pip install playwright anthropic google-genai pandas openpyxl youtube-transcript-api scikit-learn
    playwright install chromium

Usage:
    python summarize.py "path/to/document.pdf"
    python summarize.py "path/to/data.csv"
    python summarize.py "path/to/photo.jpg"
    python summarize.py "path/to/clip.mp4"
    python summarize.py "path/to/recording.mp3"
    python summarize.py "https://example.com/some-article"
    python summarize.py "https://example.com/photo.png"
    python summarize.py "https://example.com/video.mp4"
    python summarize.py "https://example.com/audio.mp3"
    python summarize.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    python summarize.py "dQw4w9WgXcQ"
    python summarize.py "dQw4w9WgXcQ" "What are the main points made in this video?"

If you pass a second argument, it's sent immediately as your first question
(useful for one-shot / scripted use, like the old summarize_youtube.py). You
then continue in the same interactive chat. With no second argument you just
land straight in the chat prompt. Type 'exit' or 'quit' to stop.
"""

import sys
import os
import re
import time
import mimetypes
import tempfile
import atexit
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.agent_loop import DocumentAgent, DocumentSearchTools
from agent.llm_client import DEFAULT_CLIENT

SUMMARY_PROMPT = (
    "Summarize this. Give a concise overview, then list the key points as "
    "bullet points. If there are any charts, diagrams, tables, or images, "
    "describe what they show and how they relate to the surrounding "
    "content."
)

SUMMARY_KEYWORDS = ("summar",)

CONTENT_FOCUS_INSTRUCTION = (
    "This may be a PDF, a full webpage capture, an image, a video, an "
    "audio recording, a YouTube transcript, or a data file (CSV/Excel). If "
    "it's a webpage capture, it could contain advertisements, promotional "
    "banners (e.g. sales, subscription offers), site navigation menus, "
    "cookie/consent notices, and other page chrome that is NOT part of the "
    "actual content -- ignore all of that entirely. If it's a YouTube "
    "transcript, note that it may lack punctuation/speaker labels and can "
    "contain minor transcription errors -- read past that to the "
    "substance. If it's an audio recording, focus on the spoken/musical "
    "content itself rather than background noise. Focus only on the real "
    "article/document/data/media content. Do still pay attention to "
    "genuine content images tied to the story itself (photos, figures, "
    "charts, diagrams) -- just skip purely decorative or promotional "
    "images like banner ads."
)

MAX_TEXT_CHARS = 400_000

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma"}


# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------

def try_extract_youtube_id(source: str) -> str | None:
    """Returns a YouTube video ID if `source` looks like a YouTube URL or a
    bare 11-char video ID, else None. Deliberately conservative so it never
    misclassifies an ordinary file path/URL as YouTube."""
    parsed = urlparse(source)
    host = (parsed.netloc or "").lower()

    if "youtube.com" in host:
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            return qs["v"][0]
        match = re.search(r"/(?:embed|shorts)/([0-9A-Za-z_-]{11})", parsed.path)
        if match:
            return match.group(1)
        return None

    if "youtu.be" in host:
        match = re.search(r"^/([0-9A-Za-z_-]{11})", parsed.path)
        if match:
            return match.group(1)
        return None

    # Bare video ID, e.g. "dQw4w9WgXcQ" -- only if it's not an existing
    # local file and not a URL (avoid false positives).
    if (
        not parsed.scheme
        and re.fullmatch(r"[0-9A-Za-z_-]{11}", source)
        and not os.path.exists(source)
    ):
        return source

    return None


def get_youtube_transcript_text(video_id: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    full_text = " ".join(snippet.text for snippet in transcript)
    return full_text.strip()


# ---------------------------------------------------------------------------
# URL helpers (non-YouTube)
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
    """Returns 'image', 'video', 'audio', or 'webpage'."""
    path_ext = os.path.splitext(urlparse(url).path)[1].lower()
    if path_ext in IMAGE_EXTS:
        return "image"
    if path_ext in VIDEO_EXTS:
        return "video"
    if path_ext in AUDIO_EXTS:
        return "audio"
    content_type = get_url_content_type(url)
    if content_type:
        if content_type.startswith("image/"):
            return "image"
        if content_type.startswith("video/"):
            return "video"
        if content_type.startswith("audio/"):
            return "audio"
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


def extract_pdf_text_fallback(pdf_path: str) -> str:
    """Plain-text extraction (no native multimodal). Used only as a last
    resort if both Gemini and Claude are entirely unreachable (e.g. no API
    keys configured at all) -- see load_source's PDF branch note below.
    Kept from the old summarize_pdf.py so PDFs still degrade gracefully."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


# ---------------------------------------------------------------------------
# Unified source loader
# ---------------------------------------------------------------------------

def load_source(source: str) -> dict:
    """Returns a dict describing the source:
      {"kind": "pdf",   "bytes": ..., "name": ...}
      {"kind": "text",  "content": ..., "name": ...}
      {"kind": "image", "bytes": ..., "mime": ..., "name": ...}
      {"kind": "video", "path": ...,  "mime": ..., "name": ..., "is_temp": bool}
      {"kind": "audio", "path": ...,  "mime": ..., "name": ..., "is_temp": bool}
    """
    youtube_id = try_extract_youtube_id(source)
    if youtube_id:
        print(f"Detected YouTube video ID: {youtube_id}")
        print("Fetching transcript...")
        try:
            transcript_text = get_youtube_transcript_text(youtube_id)
        except Exception as e:
            raise RuntimeError(
                f"Error fetching transcript: {e}\n"
                "This usually means the video has no captions/transcript "
                "available, or captions are disabled by the uploader."
            )
        if not transcript_text:
            raise RuntimeError("No transcript text was found for this video.")
        return {
            "kind": "text",
            "content": f"YouTube video transcript (video ID: {youtube_id}):\n\n{transcript_text}",
            "name": f"YouTube video {youtube_id}",
        }

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

        if url_kind == "audio":
            ext = os.path.splitext(urlparse(source).path)[1] or ".mp3"
            tmp_path, content_type = download_url_to_temp(source, ext)
            mime = content_type or mimetypes.guess_type(tmp_path)[0] or "audio/mpeg"
            atexit.register(lambda p=tmp_path: os.path.exists(p) and os.remove(p))
            return {"kind": "audio", "path": tmp_path, "mime": mime, "name": name, "is_temp": True}

        return {"kind": "pdf", "bytes": fetch_url_as_pdf(source), "name": source}

    if not os.path.exists(source):
        raise FileNotFoundError(f"File not found at '{source}'")

    ext = os.path.splitext(source)[1].lower()
    name = os.path.basename(source)

    if ext == ".pdf":
        with open(source, "rb") as f:
            return {"kind": "pdf", "bytes": f.read(), "name": name, "path": source}
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
    elif ext in AUDIO_EXTS:
        mime = mimetypes.guess_type(source)[0] or "audio/mpeg"
        return {"kind": "audio", "path": source, "mime": mime, "name": name, "is_temp": False}
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: .pdf, .csv, .xlsx, "
            f".xls, .txt, .md, common image/video/audio formats, a YouTube "
            f"link, or another URL."
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
            f"[Attached: {source['name']}]\n\n"
            f"{content}{truncated_note}\n\n---\n\n{user_text}"
        )
    return f"{CONTENT_FOCUS_INSTRUCTION}\n\n{user_text}"


# ---------------------------------------------------------------------------
# Provider sessions -- native multimodal (pdf / image / video / audio)
# ---------------------------------------------------------------------------

class GeminiSession:
    """Manages conversation history manually and calls generate_content each
    turn. Retries across candidate models on EVERY turn. Handles pdf/image
    inline; video and audio are uploaded via the Files API (Gemini
    processes them in the background, so we poll until ready)."""

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
        self._uploaded_file = None
        self.history = []

    def _get_uploaded_file(self):
        """Uploads video or audio via the Files API and waits for processing.
        Shared by both kinds since Gemini handles them the same way."""
        if self._uploaded_file is not None:
            return self._uploaded_file

        kind_label = self.source["kind"]
        print(f"Uploading {kind_label} to Gemini (this may take a moment)...")
        uploaded = self.client.files.upload(file=self.source["path"])
        while getattr(uploaded.state, "name", uploaded.state) == "PROCESSING":
            time.sleep(3)
            uploaded = self.client.files.get(name=uploaded.name)

        state = getattr(uploaded.state, "name", uploaded.state)
        if state == "FAILED":
            raise RuntimeError(f"Gemini failed to process the uploaded {kind_label}.")

        self._uploaded_file = uploaded
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
            elif kind in ("video", "audio"):
                uploaded = self._get_uploaded_file()
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
    """Manages conversation history manually. Handles pdf/image. Claude's
    API does not support video or audio input, so those sources raise a
    clear error immediately rather than failing confusingly."""

    def __init__(self, source: dict, model: str = "claude-sonnet-5"):
        import anthropic

        if source["kind"] in ("video", "audio"):
            raise RuntimeError(
                f"Claude does not support {source['kind']} input -- only "
                f"Gemini can process {source['kind']} sources in this script."
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
    video/audio sources, where Claude has no support at all). Checked every
    turn. Used for pdf/image/video/audio sources -- i.e. everything EXCEPT
    plain text, which goes through TextAgentSession instead (see below)."""

    NO_CLAUDE_FALLBACK_KINDS = ("video", "audio")

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
                if self.source["kind"] in self.NO_CLAUDE_FALLBACK_KINDS:
                    raise RuntimeError(
                        f"Gemini failed and {self.source['kind']} sources have no "
                        f"Claude fallback.\nGemini error: {gemini_error}"
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
            if self.source["kind"] in self.NO_CLAUDE_FALLBACK_KINDS:
                raise RuntimeError(
                    f"Gemini failed and {self.source['kind']} sources have no "
                    f"Claude fallback.\nGemini error: {gemini_error}"
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
# Text session -- agentic multi-step retrieval (CSV / Excel / txt / md /
# YouTube transcript), instead of one-shot with truncation.
# ---------------------------------------------------------------------------

class TextAgentSession:
    """Drop-in replacement for DualProviderSession's .send(text) interface,
    but backed by DocumentAgent (agent/document_agent.py) instead of a
    single one-shot call. The document is chunked once up front; each
    question triggers a fresh agent run that searches those chunks as many
    times as it needs (up to max_steps) before answering -- the same
    pattern already proven in agent/agent_loop.py for the landmark project.

    A short rolling memory of prior Q&A in this session is prepended to
    each new question so follow-ups ("what about the second point?") still
    have context, even though each DocumentAgent.run() call is otherwise
    independent.
    """

    def __init__(self, source: dict, max_steps: int = 5, verbose: bool = True):
        self.source = source
        self.search_tools = DocumentSearchTools(source["content"])
        self.agent = DocumentAgent(DEFAULT_CLIENT, self.search_tools, max_steps=max_steps, verbose=verbose)
        self._qa_history: list[str] = []

    def send(self, text: str):
        if self._qa_history:
            recent = "\n\n".join(self._qa_history[-3:])
            question = (
                f"{CONTENT_FOCUS_INSTRUCTION}\n\n"
                f"[Document: {self.source['name']}]\n\n"
                f"Earlier in this conversation:\n{recent}\n\n"
                f"Now answer this new question (search the document if you need to): {text}"
            )
        else:
            question = (
                f"{CONTENT_FOCUS_INSTRUCTION}\n\n"
                f"[Document: {self.source['name']}]\n\n"
                f"{text}"
            )

        result = self.agent.run(question)
        answer = result["answer"]
        self._qa_history.append(f"Q: {text}\nA: {answer}")

        provider_label = f"DocumentAgent ({result['steps_used']} step(s), {result['stopped']})"
        return answer, provider_label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize.py <path_or_url_or_youtube_link> [initial question]")
        print(
            "Supports: PDF, CSV, Excel, txt/md, image, video, audio, "
            "YouTube link, or a URL to any of these."
        )
        return

    source_str = sys.argv[1]
    initial_question = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\nSource: {source_str}")
    try:
        source = load_source(source_str)
    except Exception as e:
        print(f"Failed to load source: {e}")
        return

    print(f"Detected type: {source['kind']}")

    if source["kind"] == "text":
        print("Using agentic multi-step document search (DocumentAgent).")
        session = TextAgentSession(source)
    else:
        print("Using native multimodal understanding (Gemini/Claude).")
        session = DualProviderSession(source)

    print("\nLoaded. How may I help you?")
    print('What do you want me to do with this? (e.g. "summarize it" or ask a question)')
    print("Type 'exit' or 'quit' to stop.\n")

    def ask(user_input: str):
        prompt = SUMMARY_PROMPT if any(k in user_input.lower() for k in SUMMARY_KEYWORDS) else user_input
        try:
            answer, provider = session.send(prompt)
            print(f"\n{provider}: {answer}\n")
        except Exception as e:
            print(f"\nError: {e}\n")

    if initial_question:
        print(f"You: {initial_question}")
        ask(initial_question)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit", ""):
            print("Ending session.")
            break
        ask(user_input)


if __name__ == "__main__":
    main()