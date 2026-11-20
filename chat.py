
# chat.py
import sys

from rag.gemini_llm import make_gemini_llm
from rag.pipeline import DocumentQA

HELP = """Commands:
  /docs            list loaded documents
  /add <path.pdf>  load another PDF
  /only <text>     restrict answers to documents whose name contains <text> (/only to clear)
  /agent | /single switch between agentic and single-pass retrieval
  /reset           forget the conversation
  /help, /quit"""


def main():
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs"
    llm = make_gemini_llm()
    print(f"Using model: {llm.model_name}")
    qa = DocumentQA(llm, mode="agentic")
    loaded = qa.add_folder(pdf_dir)
    if not loaded:
        raise SystemExit(f"No PDFs found in {pdf_dir}")
    print("Loaded:", ", ".join(f"{n} ({c} chunks)" for n, c in loaded.items()))
    print(HELP)

    only = None
    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue

        if q.startswith("/"):
            cmd, _, arg = q.partition(" ")
            if cmd == "/quit":
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/docs":
                print("\n".join(qa.documents()))
            elif cmd == "/add":
                print(f"{qa.add_pdf(arg.strip())} chunks added")
            elif cmd == "/only":
                only = [d for d in qa.documents() if arg.strip().lower() in d.lower()] if arg.strip() else None
                print("Restricted to:", only if only else "all documents")
            elif cmd == "/agent":
                qa.mode = "agentic"; print("mode: agentic")
            elif cmd == "/single":
                qa.mode = "single"; print("mode: single-pass")
            elif cmd == "/reset":
                qa.reset(); print("conversation cleared")
            else:
                print("Unknown command. /help")
            continue

        try:
            r = qa.ask(q, docs=only)
        except RuntimeError as e:
            print(f"[LLM error, try again later] {str(e)[:200]}")
            continue

        if r.standalone_query != q:
            print(f"[understood as: {r.standalone_query}]")
        for t in r.trace:
            print(f"[step {t['step']}: '{t['query']}' | +{t['new_chunks']} chunks | {t['decision']}]")
        print(f"[confidence: {r.answer.confidence.level} | top dense = {r.answer.confidence.score:.3f} "
              f"| LLM calls: {r.llm_calls}]\n")
        print(r.answer.format())


if __name__ == "__main__":
    main()