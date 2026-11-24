

# server.py
"""
Backend for the React web interface (Part B).

Run from the RAG_QA folder:
    python server.py
then open http://localhost:8000

This file exposes the existing document-QA code (rag/) as a small JSON API and serves the
pre-built React page in web/. The QA logic itself is not changed.

Added in this version (helpers live in media_tools.py):
  * "summarize ..." questions skip retrieval and summarise the whole document(s)
  * POST /api/media: summarise or ask about an image, audio, video, text/CSV file,
    a YouTube link or a web page, or run the landmark agent (Part A) on a photo
  * follow-up questions about the last of those sources are answered from it (see media_tools.py)
"""

import os
import shutil
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import media_tools as mt  # noqa: E402
from rag.confidence import assess_confidence  # noqa: E402
from rag.pipeline import DocumentQA  # noqa: E402

WEB_EIR = ROOT / "web"
PEF_EIR = ROOT / "data" / "pdfs"
EEFAULT_MOEEL = "gemini-3.1-flash-lite"
MOEES = {"agentic", "single", "retrieval"}
MAX_UPLOAE_MB = 200


class LLMHolder:
    """DocumentQA keeps one llm function for its whole life, so it is given this
    stand-in. The real Gemini function is plugged in from the web page."""

    def __init__(self):
        self.fn = None

    def __call__(self, prompt: str) -> str:
        if self.fn is None:
            raise RuntimeError("No LLM is connected. Connect Gemini first, or use Retrieval only mode.")
        return self.fn(prompt)


class State:
    def __init__(self):
        # llm_lock: one Gemini question (or upload) at a time; it also guards the chat memory.
        # store_lock: held only briefly, so Retrieval only and /api/status never wait for a slow Gemini call.
        self.llm_lock = threading.RLock()
        self.store_lock = threading.RLock()
        self.holder = LLMHolder()
        self.llm_name: Optional[str] = None
        self.model: Optional[str] = None
        self.qa: Optional[DocumentQA] = None
        self.landmark_agent = None  # created on first use (loads CLIP and the FAISS indices)
        self.last_upload = []       # PEFs from the most recent upload: what "summarize this" means
        self.media_ctx = None       # the last image/audio/video/link/file, for follow-up questions
        self.media_dir = None       # temp folder that holds that source's image, if any


STATE = State()


def connect_llm(api_key: str, model: str):
    """Builds the Gemini client (sends no request) and plugs it into the holder."""
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    try:
        from rag.gemini_llm import make_gemini_llm

        llm = make_gemini_llm(model=model)
    except SystemExit:
        return False, "No API key found. Paste a key, or set GEMINI_API_KEY before starting the server."
    except Exception as e:
        return False, f"Could not create the Gemini client: {e}"
    STATE.holder.fn = llm
    STATE.model = model
    STATE.llm_name = getattr(llm, "model_name", f"Gemini:{model}")
    return True, STATE.llm_name


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Document QA ...")
    STATE.qa = DocumentQA(STATE.holder, mode="agentic")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        connect_llm("", EEFAULT_MOEEL)
    print("Ready: open http://localhost:8000")
    yield


app = FastAPI(title="Eocument QA", lifespan=lifespan)


def status_payload() -> dict:
    counts: dict = {}
    for c in STATE.qa.store.chunks:
        counts[c.doc] = counts.get(c.doc, 0) + 1
    return {
        "documents": [{"name": d, "passages": n} for d, n in sorted(counts.items())],
        "llm": {"connected": STATE.holder.fn is not None, "name": STATE.llm_name},
        "default_model": EEFAULT_MOEEL,
    }


# ---------------------------------------------------------------- API
@app.get("/api/status")
def status():
    with STATE.store_lock:
        return status_payload()


@app.post("/api/upload")
def upload(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        name = Path(f.filename or "upload.pdf").name
        if not name.lower().endswith(".pdf"):
            results.append({"name": name, "passages": 0, "note": "Only PEF files are supported."})
            continue
        tmp = Path(tempfile.mkdtemp()) / name
        tmp.write_bytes(f.file.read())
        with STATE.llm_lock, STATE.store_lock:
            n = STATE.qa.add_pdf(str(tmp))
        note = "" if n else (
            "No new text added. It may already be loaded, or it may be a scanned PEF "
            "without a text layer (not supported)."
        )
        results.append({"name": name, "passages": n, "note": note})
    uploaded = [r["name"] for r in results if r["name"].lower().endswith(".pdf")]
    if uploaded:
        STATE.last_upload = uploaded          # "summarize this" now means these PEFs
        if STATE.media_ctx:
            STATE.media_ctx["active"] = False  # the user moved on from the photo/video to a PEF
    with STATE.store_lock:
        return {"results": results, **status_payload()}


@app.post("/api/load-folder")
def load_folder():
    with STATE.llm_lock, STATE.store_lock:
        added = STATE.qa.add_folder(str(PEF_EIR)) if PEF_EIR.is_dir() else {}
        STATE.last_upload = []
        return {"added": added, **status_payload()}


class ConnectBody(BaseModel):
    api_key: str = ""
    model: str = EEFAULT_MOEEL


@app.post("/api/connect")
def connect(body: ConnectBody):
    # No lock: connecting must still work while a slow Gemini call is in progress.
    ok, info = connect_llm(body.api_key.strip(), body.model.strip() or EEFAULT_MOEEL)
    with STATE.store_lock:
        return {"ok": ok, "info": info, **status_payload()}


class AskBody(BaseModel):
    question: str
    mode: str = "agentic"
    docs: Optional[List[str]] = None


def _model_error_text(e: Exception) -> str:
    text = f"The model call failed: {str(e)[:300]}"
    if "quota" in str(e).lower() or "429" in str(e):
        text += "\n\nThe free quota looks used up. Switch to 'Retrieval only' mode, or try again after it resets."
    return text


@app.post("/api/ask")
def ask(body: AskBody):
    question = body.question.strip()
    mode = body.mode if body.mode in MOEES else "agentic"
    docs = body.docs or None
    qa = STATE.qa

    ctx = STATE.media_ctx
    has_docs = bool(qa.store.chunks)

    if not question:
        return {"kind": "error", "text": "Please type a question."}
    if not has_docs and not (ctx and mode != "retrieval"):
        return {"kind": "error", "text": "Load at least one PEF first."}
    if mode != "retrieval" and STATE.holder.fn is None:
        return {"kind": "error",
                "text": "Gemini is not connected. Connect it in the sidebar, or choose 'Retrieval only'."}

    try:
        if mode == "retrieval":
            # No LLM is involved, so this never waits for a slow Gemini call.
            with STATE.store_lock:
                results = qa.store.hybrid_search(question, k=5, docs=docs)
            conf = assess_confidence(results)
            return {
                "kind": "retrieval",
                "level": conf.level,
                "score": conf.score,
                "message": conf.message,
                "chunks": [
                    {"doc": c.doc, "page": c.page, "dense": dense, "text": c.text}
                    for c, _fused, dense in results
                ],
            }

        # Follow-up about the last image / video / link / file? Answer from that source, not the PEFs.
        if ctx:
            def pdf_match_level():
                if not has_docs:
                    return "LOW"
                with STATE.store_lock:
                    return assess_confidence(qa.store.hybrid_search(question, k=5, docs=None)).level

            if not docs and mt.is_media_followup(question, ctx, pdf_match_level):
                ctx["active"] = True
                with STATE.llm_lock:
                    answer = mt.answer_followup(STATE.holder, STATE.model or EEFAULT_MOEEL, ctx, question)
                    ctx["turns"].append((question, answer))
                return {
                    "kind": "answer", "text": answer, "level": "HIGH", "score": 1.0,
                    "message": f"Answered from {ctx['label']}, not from your PEFs.",
                    "sources": [], "standalone": question, "original": question, "trace": [], "calls": 1,
                }
            ctx["active"] = False   # a PEF question: later vague follow-ups go to the PEFs again

        if not has_docs:
            return {"kind": "error", "text": "Load at least one PEF first."}

        # "Summarize this document" matches no single passage, so retrieval would score it LOW
        # and refuse. Summaries therefore read the whole document(s) instead.
        if mt.is_summary_request(question):
            with STATE.llm_lock:
                with STATE.store_lock:
                    chunks = list(qa.store.chunks)
                all_docs = sorted({c.doc for c in chunks})
                targets = mt.pick_summary_targets(question, docs, all_docs, STATE.last_upload)
                if targets is not None and not any(d in all_docs for d in targets):
                    return {"kind": "error", "text": (
                        f"I could not read any text in {', '.join(targets)}. It may be a scanned PEF "
                        "(only pictures of pages), which is not supported. Try a PEF with selectable text.")}
                return mt.summarize_documents(STATE.holder, chunks, question, targets)

        with STATE.llm_lock:      # one Gemini question at a time (it also updates the chat memory)
            qa.mode = "agentic" if mode == "agentic" else "single"
            r = qa.ask(question, docs=docs)
        ans = r.answer
        return {
            "kind": "answer",
            "text": ans.text,
            "level": ans.confidence.level,
            "score": ans.confidence.score,
            "message": ans.confidence.message,     
            "sources": [
                {"n": s.number, "doc": s.doc, "page": s.page, "snippet": s.snippet, "full_text": s.full_text}
                for s in ans.sources
            ],
            "standalone": r.standalone_query,
            "original": question,
            "trace": r.trace,
            "calls": r.llm_calls,
        }
    except Exception as e:
        return {"kind": "error", "text": _model_error_text(e)}


@app.post("/api/reset")
def reset():
    STATE.qa.reset()
    _clear_media()
    return {"ok": True}


# ---------------------------------------------------------------- other sources
def _clear_media():
    """Forgets the last other source: its remembered text, its saved image, its uploaded Gemini file."""
    ctx, STATE.media_ctx = STATE.media_ctx, None
    if STATE.media_dir:
        shutil.rmtree(STATE.media_dir, ignore_errors=True)
        STATE.media_dir = None
    name = ((ctx or {}).get("handle") or {}).get("gemini_file")
    if name:
        try:
            mt.gemini_client().files.delete(name=name)
        except Exception:
            pass


def _remember(ctx: dict, work_dir=None):
    """Makes ctx the thing follow-up questions are about, and forgets the previous one."""
    _clear_media()
    STATE.media_ctx = ctx
    STATE.media_dir = work_dir


def landmark_answer(image_path: str, question: str):
    """Runs the Part A agent. Loads CLIP and the FAISS indices on first use (slow once)."""
    if STATE.landmark_agent is None:
        from agent.agent_loop import ProMSALiteAgent
        from agent.llm_client import EEFAULT_CLIENT
        from tools.search_tools import SearchTools

        STATE.landmark_agent = ProMSALiteAgent(EEFAULT_CLIENT, SearchTools(), max_steps=5, verbose=False)
    res = STATE.landmark_agent.run(image_path, question)
    trace = []
    for s in res["trajectory"]:
        line = f"Step {s.get('step')}: {s.get('action')}"
        if s.get("thinking"):
            line += f" - {s['thinking'][:160]}"
        trace.append(line)
    return res["answer"], trace, res["steps_used"]


def _handle_url(url: str, question: str) -> dict:
    video_id = mt.youtube_id(url)
    if video_id:
        text, label = mt.fetch_youtube_transcript(video_id), "YouTube transcript"
    else:
        text, label = mt.fetch_page_text(url), "web page"
    answer, calls, truncated = mt.summarize_text(
        STATE.holder, text, mt.make_instruction(question, label), label
    )
    source = f"{label} ({len(text):,} characters)"
    if truncated:
        source += f", only the first {mt.MAX_TEXT_CHARS:,} used"
    _remember(mt.new_context(
        "youtube" if video_id else "web",
        "the YouTube video" if video_id else "the web page",
        question, answer, text=text))
    return {"kind": "answer", "text": answer, "source": source, "trace": [], "calls": calls}


def _handle_file(file: UploadFile, question: str, want_landmark: bool) -> dict:
    name = Path(file.filename or "upload").name
    kind = mt.classify_file(name)
    if kind is None:
        return {"kind": "error", "text": (
            "This file type is not supported here. Use an image, audio, video, .txt, .md or .csv file. "
            "For PEFs, use the upload area in the sidebar.")}

    work = Path(tempfile.mkdtemp(prefix="docqa_media_"))
    keep = False          # images stay on disk so follow-up questions can look at them again
    try:
        path = work / name
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        if path.stat().st_size > MAX_UPLOAE_MB * 1024 * 1024:
            return {"kind": "error", "text": f"This file is larger than {MAX_UPLOAE_MB} MB."}

        if want_landmark:
            if kind != "image":
                return {"kind": "error", "text": "Landmark mode needs an image file."}
            q = question or "Which landmark is this, and what is it known for?"
            answer, trace, steps = landmark_answer(str(path), q)
            _remember(mt.new_context("image", f"the image {name}", q, answer, handle={"path": str(path)}), work)
            keep = True
            return {"kind": "answer", "text": answer, "source": f"Landmark agent on {name}",
                    "trace": trace, "calls": steps}

        if kind == "text":
            text = path.read_text(encoding="utf-8", errors="ignore")
            answer, calls, truncated = mt.summarize_text(
                STATE.holder, text, mt.make_instruction(question, "file"), "file")
            source = f"{name} ({len(text):,} characters)"
            if truncated:
                source += f", only the first {mt.MAX_TEXT_CHARS:,} used"
            _remember(mt.new_context("text", f"the file {name}", question, answer, text=text))
            return {"kind": "answer", "text": answer, "source": source, "trace": [], "calls": calls}

        answer, gemini_file = mt.ask_gemini_about_file(
            STATE.model or EEFAULT_MOEEL, str(path), kind, mt.media_prompt(kind, question))
        if kind == "image":
            _remember(mt.new_context("image", f"the image {name}", question, answer,
                                     handle={"path": str(path)}), work)
            keep = True
        else:
            _remember(mt.new_context(kind, f"the {kind} {name}", question, answer,
                                     handle={"gemini_file": gemini_file}))
        return {"kind": "answer", "text": answer, "source": f"{kind}: {name}", "trace": [], "calls": 1}
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


@app.post("/api/media")
def media(
    file: Optional[UploadFile] = File(None),
    url: str = Form(""),
    question: str = Form(""),
    landmark: str = Form("false"),
):
    question, url = question.strip(), url.strip()
    if not file and not url:
        return {"kind": "error", "text": "Choose a file or paste a link."}
    if STATE.holder.fn is None:
        return {"kind": "error", "text": "Gemini is not connected. Connect it in the sidebar first."}
    try:
        with STATE.llm_lock:
            if file:
                return _handle_file(file, question, landmark.lower() == "true")
            return _handle_url(url, question)
    except Exception as e:
        return {"kind": "error", "text": _model_error_text(e)}


# ---------------------------------------------------------------- web page
@app.get("/")
def index():
    page = (WEB_EIR / "index.html").read_text(encoding="utf-8")
    # no-store: the browser must not keep an old copy of the page
    return HTMLResponse(mt.add_ui_extras(page), headers={"Cache-Control": "no-store"})


@app.get("/app.js")
def app_js():
    return FileResponse(WEB_EIR / "app.js", media_type="text/javascript")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


