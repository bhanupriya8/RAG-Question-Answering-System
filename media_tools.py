

# media_tools.py
"""
Helpers that server.py uses for things the PDF question-answering pipeline cannot do:

  * detecting "summarize ..." requests and summarising whole documents (no retrieval)
  * getting text from a YouTube link or a web page
  * sending an image, audio or video file to Gemini

This module does not import FastAPI or the rag package, so it can be tested on its own.
Install once:  pip install youtube-transcript-api requests pillow google-genai
"""

import html
import os
import re
import time
from pathlib import Path

# One Gemini call handles up to SECTION_CHARS characters. Longer text is summarised
# section by section, then the notes are combined. MAX_SECTIONS caps the number of calls.
SECTION_CHARS = 60_000
MAX_SECTIONS = 6
MAX_TEXT_CHARS = SECTION_CHARS * MAX_SECTIONS

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".mpeg", ".mpg"}
TEXT_EXT = {".txt", ".md", ".csv"}


def classify_file(name: str):
    """Returns 'image', 'audio', 'video', 'text', or None for unsupported types."""
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    if ext in TEXT_EXT:
        return "text"
    return None


# ------------------------------------------------------------ summary requests
_DOC_WORDS = r"(document|documents|pdf|pdfs|file|files|paper|notes|unit|chapter)"


_SUM_NOUN = (r"(document|documents|pdf|pdfs|file|files|paper|notes|unit|chapter|lecture|slides|article|report)")


def is_summary_request(question: str) -> bool:
    """True for 'summarize this', 'summary', 'sum it up', 'give me an overview of the notes', 'tl;dr' ...

    Deliberately strict: 'explain text summarization' is a normal question about the
    documents' content and must still go through retrieval.
    """
    q = question.lower().strip()
    N = _SUM_NOUN
    if re.search(r"\btl;?\s?dr\b", q):
        return True
    if re.search(r"what('s| is| are) (this|the|these) " + N + r" about", q):
        return True
    if re.search(r"what('s| is) (in|inside) (the |this |these |my )?" + N, q):
        return True
    if re.search(r"\b(summary|overview) of (the |this |these |my |all )?" + N, q):
        return True
    if re.search(r"\b(key|main) (points|ideas|takeaways|topics) (of|in|from) (the |this |these |my |all )?" + N, q):
        return True
    if re.search(r"\b(explain|describe|review|go through) (the |this |these |my |all )?" + N + r"( to me)?\W*$", q):
        return True
    starts = re.match(
        r"(please |can you |could you |kindly |just )*"
        r"(summari[sz]e|summary\b|sum (it |this |that )?up|"
        r"give (me )?(a |an |the )?(short |brief |quick )?(summary|overview|gist)|"
        r"write (a |an )?(summary|overview))",
        q,
    )
    return bool(starts)


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def wants_all_documents(question: str) -> bool:
    q = question.lower()
    if re.search(r"\b(all|every|both|each)\s+(of\s+)?(the\s+|my\s+|these\s+|those\s+)?"
                 r"(documents?|pdfs?|files?|papers?|notes)\b", q):
        return True
    return bool(re.search(r"\b(documents|pdfs|files|papers)\b", q) or re.search(r"\b(all|everything)\W*$", q))


def docs_named_in(question: str, all_docs) -> list:
    """Documents whose name (or at least two distinctive words of it) appears in the question."""
    qn = " " + _norm(question) + " "
    qwords = set(qn.split())
    found = []
    for d in all_docs:
        stem = _norm(Path(d).stem)
        toks = [t for t in stem.split() if len(t) > 2 and t not in {"pdf", "the", "and", "for"}]
        if stem and f" {stem} " in qn:
            found.append(d)
        elif len(toks) >= 2 and sum(t in qwords for t in toks) >= 2:
            found.append(d)
    return found


def pick_summary_targets(question: str, selected, all_docs, last_upload):
    """Which documents 'summarize this' means. Returns a list of names, or None for all documents.

    Order: documents ticked in the sidebar, a document named in the question, 'all documents',
    the PDF uploaded most recently ('this'), otherwise everything.
    """
    if selected:
        return list(selected)
    named = docs_named_in(question, all_docs)
    if named:
        return named
    if wants_all_documents(question):
        return None
    if last_upload:
        return list(last_upload)
    return None


# ------------------------------------------------------------ summarising text
def summarize_text(llm, text: str, instruction: str, label: str):
    """llm is a function prompt -> str. Returns (answer, number_of_calls, was_truncated)."""
    text = text.strip()
    truncated = len(text) > MAX_TEXT_CHARS
    text = text[:MAX_TEXT_CHARS]

    if len(text) <= SECTION_CHARS:
        return llm(f"{instruction}\n\n{label}:\n{text}").strip(), 1, truncated

    parts = [text[i:i + SECTION_CHARS] for i in range(0, len(text), SECTION_CHARS)]
    notes = []
    for n, part in enumerate(parts, 1):
        notes.append(
            llm(
                f"This is part {n} of {len(parts)} of a longer {label}. Write detailed notes "
                f"that keep every important fact, name, term and number.\n\n{part}"
            ).strip()
        )
    final = llm(
        f"{instruction}\n\nThese are notes on consecutive parts of the {label}:\n\n"
        + "\n\n".join(notes)
    )
    return final.strip(), len(parts) + 1, truncated


def make_instruction(question: str, label: str) -> str:
    if question:
        return (
            f'Answer this using only the {label} text below: "{question}". '
            "If the text does not contain the answer, say so."
        )
    return (
        f"Summarize the {label} below. Start with what it is about, then list the main "
        "points and any important definitions, numbers or conclusions. "
        "Do not add facts that are not in the text."
    )


# ------------------------------------------------------------ whole-document summaries
def doc_texts(chunks, docs=None):
    """Rebuilds each document's text from the stored chunks, in page order.

    Returns {doc_name: (text, first_page, last_page)}.
    """
    pages = {}
    for c in chunks:
        if docs and c.doc not in docs:
            continue
        pages.setdefault(c.doc, {}).setdefault(c.page, []).append(c.text)
    out = {}
    for doc, by_page in pages.items():
        nums = sorted(by_page)
        text = "\n\n".join(f"[Page {p}]\n" + "\n".join(by_page[p]) for p in nums)
        out[doc] = (text, nums[0], nums[-1])
    return out


PLAIN_MARK = "\u2063\u2063"


def summarize_documents(llm, chunks, question: str, docs=None) -> dict:
    """Summarises whole documents without retrieval. Returns the same JSON shape that
    /api/ask returns for a normal answer, so the existing web page can display it."""
    texts = doc_texts(chunks, docs)
    if not texts:
        return {"kind": "error", "text": "None of the selected documents are loaded."}

    instruction = (
        f'Follow this request using only the text provided: "{question}"\n'
        "If it asks for a summary, write a clear, structured summary: the main topics, key "
        "points, important definitions and conclusions. Do not add facts that are not in the text."
    )

    sections, sources, trace, calls, notes = [], [], [], 0, []
    for i, (doc, (text, first, last)) in enumerate(sorted(texts.items()), 1):
        answer, n, truncated = summarize_text(llm, text, instruction, f"document '{doc}'")
        calls += n
        sections.append(answer if len(texts) == 1 else f"{doc}\n\n{answer}")
        sources.append({"n": i, "doc": doc, "page": first,
                        "snippet": f"Whole document used (pages {first} to {last})."})
        trace.append(f"Summarised {doc} (pages {first} to {last}) in {n} call(s).")
        if truncated:
            notes.append(f"{doc} is very long, so only its first {MAX_TEXT_CHARS:,} characters were used.")

    text = "\n\n".join(sections)
    if notes:
        text += "\n\n(" + " ".join(notes) + ")"
    # PLAIN_MARK is invisible. The web page looks for it and hides the confidence badge and the
    # source list for this answer, because a whole-document summary has neither.
    return {
        "kind": "answer",
        "text": PLAIN_MARK + text,
        "level": "HIGH",
        "score": 1.0,
        "message": "",
        "sources": [],
        "standalone": question,
        "original": question,
        "trace": [],
        "calls": calls,
        "plain": True,
    }


# ------------------------------------------------------------ YouTube and web pages
_YT_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:[^#\s]*&)?v=|shorts/|embed/|live/))([A-Za-z0-9_-]{11})"
)


def youtube_id(url: str):
    m = _YT_RE.search(url)
    return m.group(1) if m else None


def fetch_youtube_transcript(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        raise RuntimeError("YouTube support needs one more package. Run: pip install youtube-transcript-api")
    langs = ["en", "en-US", "en-GB", "hi"]
    try:
        try:  # newer versions of the package
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=langs)
            return " ".join(s.text for s in fetched)
        except AttributeError:  # older versions
            items = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            return " ".join(i["text"] for i in items)
    except Exception as e:
        raise RuntimeError(
            f"Could not get a transcript for this video. It may have no captions. ({str(e)[:200]})"
        ) from e


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


def fetch_page_text(url: str) -> str:
    import requests

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Document QA)"}, timeout=20)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        raise RuntimeError(
            f"This link is not a web page ({ctype or 'unknown type'}). Download the file and upload it instead."
        )
    text = html_to_text(r.text)
    if len(text) < 200:
        raise RuntimeError(
            "Almost no readable text was found. The page may need JavaScript to load its content."
        )
    return text


# ------------------------------------------------------------ Gemini: images, audio, video
def gemini_client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Gemini is not connected. Connect it in the sidebar first.")
    return genai.Client(api_key=key)


def _generate_one_model(client, name: str, contents, retries: int) -> str:
    last = None
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=name, contents=contents).text or ""
        except Exception as e:
            last = e
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
                wait = float(m.group(1)) + 2 if m else 15 * (attempt + 1)
                if wait > 120:  # a long wait means this model's daily quota is gone
                    raise
                time.sleep(wait)
            elif "503" in msg or "UNAVAILABLE" in msg:
                time.sleep(2 * 2 ** attempt)
            else:
                raise
    raise last


def gemini_generate(model: str, contents, client=None, retries: int = 3) -> str:
    """`model` may be one name or several separated by commas (the same format the Model box in
    the sidebar accepts). If one model is out of quota, busy or missing, the next one is tried."""
    client = client or gemini_client()
    names = [m.strip() for m in model.split(",") if m.strip()]
    last = None
    for name in names:
        try:
            return _generate_one_model(client, name, contents, retries)
        except Exception as e:
            last = e
            msg = str(e)
            if any(code in msg for code in ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND", "503", "UNAVAILABLE")):
                continue  # try the next model in the list
            raise
    raise last


def _upload_and_wait(client, path: str):
    f = client.files.upload(file=path)
    for _ in range(60):  # about two minutes
        state = getattr(f.state, "name", None)
        if state in (None, "ACTIVE"):
            return f
        if state == "FAILED":
            raise RuntimeError("Gemini could not process this file.")
        time.sleep(2)
        f = client.files.get(name=f.name)
    raise RuntimeError("Gemini took too long to process this file.")


def media_prompt(kind: str, question: str) -> str:
    if question:
        return question
    return {
        "image": "Describe this image in detail, including any text, charts or labels, then give a short summary.",
        "audio": "Summarise this audio: the main points first, then the important details. Mention who is speaking if it is clear.",
        "video": "Summarise this video: what happens, what is said, and any text or charts shown on screen.",
    }[kind]


def ask_gemini_about_file(model: str, path: str, kind: str, prompt: str):
    """Returns (answer, gemini_file_name). The uploaded audio/video file is kept on Gemini's side
    (it expires by itself after about 2 days) so follow-up questions can look at it again.
    For images the second value is None, because the local copy is reused instead."""
    if kind == "image":
        from PIL import Image

        with Image.open(path) as im:
            img = im.convert("RGB")
        return gemini_generate(model, [prompt, img]), None

    client = gemini_client()
    uploaded = _upload_and_wait(client, path)
    return gemini_generate(model, [uploaded, prompt], client=client), uploaded.name


# ------------------------------------------------------------ follow-up questions about other sources
FOLLOWUP_CHARS = 100_000

_MEDIA_REF = re.compile(
    r"\b(this|the|that|in the|from the)\s+(youtube\s+)?"
    r"(image|photo|picture|video|clip|audio|recording|podcast|transcript|speaker|website|webpage|web page|article|landmark)\b",
    re.I,
)


def mentions_media(question: str) -> bool:
    """'what does the speaker say about X', 'describe this photo again', ..."""
    return bool(_MEDIA_REF.search(question))


def mentions_documents(question: str) -> bool:
    return bool(re.search(r"\b" + _DOC_WORDS + r"\b", question, re.I))


def new_context(kind, label, question, answer, text=None, handle=None) -> dict:
    """What the server remembers about the last image/audio/video/link/file so that
    follow-up questions can be answered about it."""
    return {
        "kind": kind,            # youtube, web, text, image, audio, video
        "label": label,          # e.g. "the image cat.jpg"
        "text": text,            # full text for youtube / web / text files, otherwise None
        "handle": handle or {},  # {"path": ...} for images, {"gemini_file": ...} for audio/video
        "turns": [(question or "Summarize this", answer)],
        "active": True,
    }


def is_media_followup(question: str, ctx, retrieval_level) -> bool:
    """Decides whether a question is about the last other source (True) or about the PDFs (False).

    retrieval_level is a function returning 'HIGH', 'MEDIUM' or 'LOW': how well the question
    matches the PDFs. It is only called when really needed.
    """
    if not ctx:
        return False
    if mentions_media(question):          # "the video", "this photo" ... always means the source
        return True
    if not ctx.get("active"):             # the user already moved on to the PDFs
        return False
    if is_summary_request(question):
        return not mentions_documents(question)
    return retrieval_level() == "LOW"     # nothing in the PDFs matches, so it is probably about the source


def answer_followup(llm, model: str, ctx: dict, question: str) -> str:
    history = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in ctx["turns"][-4:])
    prompt = (
        f"You are answering follow-up questions about {ctx['label']}.\n"
        "Use only that material. If it does not contain the answer, say so.\n\n"
        f"Conversation so far:\n{history}\n\nNew question: {question}"
    )
    handle = ctx.get("handle") or {}

    if ctx.get("text"):
        text = ctx["text"]
        if len(text) > FOLLOWUP_CHARS:
            prompt += f"\n\n(Only the first {FOLLOWUP_CHARS:,} characters of the material are available.)"
        return llm(f"{prompt}\n\nMaterial:\n{text[:FOLLOWUP_CHARS]}").strip()

    if ctx["kind"] == "image":
        from PIL import Image

        with Image.open(handle["path"]) as im:
            img = im.convert("RGB")
        return gemini_generate(model, [prompt, img]).strip()

    client = gemini_client()
    f = client.files.get(name=handle["gemini_file"])
    return gemini_generate(model, [f, prompt], client=client).strip()


# ------------------------------------------------------------ web page additions
# Added to the page by server.py when it serves web/index.html, so nothing has to be edited
# or patched by hand: sidebar that does not scroll away and can be opened/closed, and results
# from "Other sources" shown in the main chat.
UI_EXTRAS = r'''<!-- ui-patch:start -->
<style>
  /* 1. The page itself never scrolls: the sidebar and the chat each scroll on their own. */
  html, body { overflow: hidden; }
  .app { grid-template-rows: minmax(0, 1fr) !important; }
  .sidebar, .main, .messages { min-height: 0; }
  @media (max-width: 860px) { .app { grid-template-rows: auto minmax(0, 1fr) !important; } }

  /* 2. Sidebar that opens and closes */
  .app { transition: grid-template-columns .2s ease; }
  body.sb-closed .app { grid-template-columns: 0 1fr !important; }
  body.sb-closed .sidebar { padding: 0 !important; border-right: 0 !important; overflow: hidden !important; visibility: hidden; }
  #sb-toggle { position: fixed; top: 12px; left: 258px; z-index: 40; padding: 6px 12px; line-height: 1; font-size: 18px; }
  body.sb-closed #sb-toggle { left: 12px; }
  @media (max-width: 860px) {
    body.sb-closed .app { grid-template-columns: 1fr !important; grid-template-rows: minmax(0, 1fr) !important; }
    body.sb-closed .sidebar { display: none; }
    #sb-toggle, body.sb-closed #sb-toggle { left: auto; right: 12px; }
  }

  /* 3. Results from "Other sources" are shown in the chat */
  .messages.xt-has-extra .empty { display: none; }

  /* 4. A summary shows just the text: no confidence badge, notes or source list */
  .assistant.xt-plain .meta, .assistant.xt-plain .block, .assistant.xt-plain .notice { display: none !important; }
</style>
<script>
(function () {
  "use strict";

  /* ---------- sidebar open / close ---------- */
  var KEY = "docqa-sidebar-closed";
  function stored() { try { return localStorage.getItem(KEY) === "1"; } catch (e) { return false; } }
  function save(v) { try { localStorage.setItem(KEY, v ? "1" : "0"); } catch (e) {} }

  var toggle = document.createElement("button");
  toggle.id = "sb-toggle";
  toggle.type = "button";
  toggle.className = "btn btn-secondary";
  toggle.textContent = "\u2630";
  function applyState(closed) {
    document.body.classList.toggle("sb-closed", closed);
    toggle.setAttribute("aria-expanded", closed ? "false" : "true");
    toggle.setAttribute("aria-label", closed ? "Open sidebar" : "Close sidebar");
    toggle.title = closed ? "Open sidebar" : "Close sidebar";
  }
  toggle.addEventListener("click", function () {
    var closed = !document.body.classList.contains("sb-closed");
    applyState(closed);
    save(closed);
  });
  document.body.appendChild(toggle);
  applyState(stored());

  /* ---------- results of "Other sources" go into the chat ---------- */
  var go = document.getElementById("xt-go");
  var panel = document.getElementById("xt-panel");
  if (!go || !panel) { return; }

  // Replace the Run button with a copy, which drops the old click handler
  // (the one that printed the result inside the panel).
  var fresh = go.cloneNode(true);
  go.parentNode.replaceChild(fresh, go);
  go = fresh;

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) { n.className = cls; }
    if (text !== undefined) { n.textContent = text; }   // textContent only: never runs page text as HTML
    return n;
  }
  function thread() { return document.querySelector(".messages"); }
  function addRow(row) {
    var box = thread();
    if (!box) { return; }
    box.classList.add("xt-has-extra");
    box.appendChild(row);
    box.scrollTop = box.scrollHeight;
  }
  function hint(text) {
    var out = $("xt-out");
    if (out) { out.textContent = text; }
  }

  // Small, safe renderer: paragraphs, bullet lists, **bold**, and # headings.
  function inline(parent, text) {
    text.split(/(\*\*[^*]+\*\*)/).forEach(function (part) {
      if (/^\*\*[^*]+\*\*$/.test(part)) {
        parent.appendChild(el("strong", "", part.slice(2, -2)));
      } else if (part) {
        parent.appendChild(document.createTextNode(part));
      }
    });
  }
  function rich(text) {
    var wrap = el("div", "rich");
    var list = null;
    text.split("\n").forEach(function (line) {
      var bullet = line.match(/^\s*[-*\u2022]\s+(.*)$/);
      var heading = line.match(/^#{1,6}\s*(.*)$/);
      if (bullet) {
        if (!list) { list = el("ul"); wrap.appendChild(list); }
        var li = el("li");
        inline(li, bullet[1]);
        list.appendChild(li);
      } else if (line.trim()) {
        list = null;
        var p = el("p");
        if (heading) { p.appendChild(el("strong", "", heading[1])); } else { inline(p, line); }
        wrap.appendChild(p);
      } else {
        list = null;
      }
    });
    return wrap;
  }

  function userRow(text) {
    var row = el("div", "row row-user xt-row");
    row.appendChild(el("div", "bubble bubble-user pre", text));
    return row;
  }
  function aiRow() {
    var row = el("div", "row xt-row");
    row.appendChild(el("div", "avatar", "AI"));
    var bubble = el("div", "bubble bubble-ai");
    var body = el("div", "assistant");
    bubble.appendChild(body);
    row.appendChild(bubble);
    return { row: row, body: body };
  }
  function typing(body) {
    var t = el("span", "typing");
    t.appendChild(el("i"));
    t.appendChild(el("i"));
    t.appendChild(el("i"));
    body.appendChild(t);
  }

  function show(ai, d) {
    ai.body.textContent = "";
    if (d.kind === "error") {
      ai.body.appendChild(el("div", "notice notice-error", d.text));
      return;
    }
    var meta = [];
    if (d.source) { meta.push("Source: " + d.source); }
    if (d.calls) { meta.push(d.calls + " model call" + (d.calls === 1 ? "" : "s")); }
    if (meta.length) { ai.body.appendChild(el("div", "meta", meta.join(" \u00b7 "))); }
    ai.body.appendChild(rich(d.text || "(empty answer)"));
    if (d.trace && d.trace.length) {
      var det = el("details", "block");
      det.appendChild(el("summary", "", "Steps (" + d.trace.length + ")"));
      var ol = el("ol", "trace");
      d.trace.forEach(function (t) { ol.appendChild(el("li", "", t)); });
      det.appendChild(ol);
      ai.body.appendChild(det);
    }
  }

  function resetForm() {
    $("xt-file").value = "";
    $("xt-url").value = "";
    $("xt-q").value = "";
    $("xt-landmark").checked = false;
    hint("");
  }

  go.addEventListener("click", function () {
    var file = $("xt-file").files[0];
    var url = $("xt-url").value.trim();
    var question = $("xt-q").value.trim();
    if (!file && !url) { hint("Choose a file or paste a link first."); return; }

    var fd = new FormData();
    if (file) { fd.append("file", file); }
    fd.append("url", url);
    fd.append("question", question);
    fd.append("landmark", $("xt-landmark").checked ? "true" : "false");

    addRow(userRow((question || "Summarize this") + "\n(" + (file ? file.name : url) + ")"));
    var ai = aiRow();
    typing(ai.body);
    addRow(ai.row);

    panel.hidden = true;     // close the panel so the chat is in view
    go.disabled = true;

    fetch("/api/media", { method: "POST", body: fd })
      .then(function (r) { return r.json(); })
      .then(function (d) { show(ai, d); if (d.kind !== "error") { resetForm(); } })
      .catch(function (e) { show(ai, { kind: "error", text: "Request failed: " + e }); })
      .then(function () {
        go.disabled = false;
        var box = thread();
        if (box) { box.scrollTop = box.scrollHeight; }
      });
  });

  // "Clear conversation" in the sidebar should also clear these messages.
  document.addEventListener("click", function (e) {
    var b = e.target && e.target.closest ? e.target.closest("button") : null;
    if (b && /clear conversation/i.test(b.textContent || "")) {
      var box = thread();
      if (!box) { return; }
      Array.prototype.slice.call(box.querySelectorAll(".xt-row")).forEach(function (n) {
        n.parentNode.removeChild(n);
      });
      box.classList.remove("xt-has-extra");
    }
  });
})();
</script>
<script>
(function () {
  "use strict";
  var MARK = "\u2063\u2063", queued = false;
  function scan() {
    queued = false;
    var items = document.querySelectorAll(".assistant");
    for (var i = 0; i < items.length; i++) {
      if (items[i].textContent.indexOf(MARK) !== -1) { items[i].classList.add("xt-plain"); }
    }
  }
  new MutationObserver(function () {
    if (!queued) { queued = true; window.requestAnimationFrame(scan); }
  }).observe(document.body, { childList: true, subtree: true, characterData: true });
  scan();
})();
</script>
<!-- ui-patch:end -->'''


def add_ui_extras(page: str) -> str:
    """Puts UI_EXTRAS into the page just before </body>. If an older copy is already in the page
    (for example from apply_ui_patch.py), it is replaced by the current one."""
    if "<!-- ui-patch:start -->" in page and "<!-- ui-patch:end -->" in page:
        return re.sub(r"<!-- ui-patch:start -->.*?<!-- ui-patch:end -->", lambda _m: UI_EXTRAS, page, flags=re.S)
    last = None
    for last in re.finditer(r"</body>", page, flags=re.I):
        pass
    if last is None:
        return page + "\n" + UI_EXTRAS + "\n"
    return page[:last.start()] + UI_EXTRAS + "\n" + page[last.start():]