
# eval/rag_eval.py
import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.rag_core import DocumentStore
from rag.confidence import assess_confidence
from rag.answer import answer_question
from rag.agentic import agentic_answer

HERE = Path(__file__).parent
PDF_DIR = HERE.parent / "data" / "pdfs"
QUESTIONS = HERE / "rag_questions.json"
OUT_JSON = HERE / "rag_results.json"
OUT_HTML = HERE / "rag_dashboard.html"
CKPT = HERE / "rag_answer_checkpoint.json"


# ---------------------------------------------------------------- setup
def load_store():
    store = DocumentStore()
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        store.add_pdf(str(pdf))
    return store


def first_gold_rank(pages_ranked, gold):
    for i, p in enumerate(pages_ranked, start=1):
        if p in gold:
            return i
    return None


# ------------------------------------------------- retrieval (no LLM calls)
def retrieval_eval(store, qs, k=4):
    rows = []
    for item in qs:
        gold = set(item["gold_pages"])
        dense = store.dense_search(item["q"], k=k)
        hybrid = store.hybrid_search(item["q"], k=k)
        d_pages = [c.page for c, _ in dense]
        h_pages = [c.page for c, _, _ in hybrid]
        conf = assess_confidence(hybrid)
        rows.append({
            "q": item["q"], "answerable": item["answerable"],
            "dense_rank": first_gold_rank(d_pages, gold) if gold else None,
            "hybrid_rank": first_gold_rank(h_pages, gold) if gold else None,
            "conf": conf.level, "top_dense": round(conf.score, 3),
        })
    return rows


def summarize_retrieval(rows, key, k=4):
    ans = [r for r in rows if r["answerable"]]
    hits = sum(1 for r in ans if r[key] is not None and r[key] <= k)
    mrr = sum(1 / r[key] for r in ans if r[key]) / max(len(ans), 1)
    return {"hit_at_k": hits / max(len(ans), 1), "mrr": mrr}


def calibration(rows):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    return {
        "answerable_not_low": sum(r["conf"] != "low" for r in ans) / max(len(ans), 1),
        "unanswerable_low": sum(r["conf"] == "low" for r in unans) / max(len(unans), 1),
    }


# ------------------------------------------------------ answers (LLM calls)
def keyword_score(text, keywords):
    if not keywords:
        return None
    t = text.lower()
    return sum(kw.lower() in t for kw in keywords) / len(keywords)


def cited_pages(ans):
    return {s.page for s in ans.sources}


def load_checkpoint(model_name, fresh):
    if fresh or not CKPT.exists():
        return []
    data = json.loads(CKPT.read_text(encoding="utf-8"))
    if isinstance(data, list):   # old format, no model recorded
        print("NOTE: old-format checkpoint, model unknown. "
              "If you changed model or provider, re-run with --fresh.")
        return data
    if data.get("model") != model_name:
        print(f"Checkpoint was made with '{data.get('model')}', current model is "
              f"'{model_name}'. Ignoring it and starting fresh.")
        return []
    return data["rows"]


def save_checkpoint(model_name, rows):
    CKPT.write_text(json.dumps({"model": model_name, "rows": rows}, indent=2),
                    encoding="utf-8")


def answer_eval(store, qs, llm, model_name, fresh=False):
    """Returns (rows, stopped_reason or None). Saves after every question."""
    rows = load_checkpoint(model_name, fresh)
    done = {r["q"] for r in rows}
    stopped = None

    for item in qs:
        if item["q"] in done:
            print(f"skip (already done): {item['q'][:55]}")
            continue

        gold, kws = set(item["gold_pages"]), item["keywords"]
        rec = {"q": item["q"], "answerable": item["answerable"]}
        try:
            # single pass (hybrid retrieval, one LLM call)
            t0 = time.time()
            a = answer_question(store, item["q"], llm, k=4)
            rec["single_time"] = round(time.time() - t0, 1)
            rec["single_calls"] = 0 if a.confidence.level == "low" else 1
            rec["single_kw"] = keyword_score(a.text, kws)
            rec["single_cite_prec"] = (len(cited_pages(a) & gold) / len(cited_pages(a))
                                       if gold and a.sources else None)
            rec["single_abstained"] = a.confidence.level == "low"

            # agentic (multi-step)
            t0 = time.time()
            r = agentic_answer(store, item["q"], llm, max_steps=3, k=4)
            rec["agent_time"] = round(time.time() - t0, 1)
            rec["agent_calls"] = r.llm_calls
            rec["agent_kw"] = keyword_score(r.answer.text, kws)
            rec["agent_cite_prec"] = (len(cited_pages(r.answer) & gold) / len(cited_pages(r.answer))
                                      if gold and r.answer.sources else None)
            rec["agent_abstained"] = r.answer.confidence.level == "low"
        except Exception as e:
            stopped = f"{item['q'][:55]} -> {type(e).__name__}: {str(e)[:150]}"
            print(f"\nSTOPPED at: {stopped}")
            print("Progress is saved. Re-run the same command later to resume.")
            break

        rows.append(rec)
        save_checkpoint(model_name, rows)
        print(f"done ({len(rows)}/{len(qs)}): {item['q'][:55]}")
    return rows, stopped


def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v * 100:.0f}%" if pct else f"{v:.2f}"


# ---------------------------------------------------------------- dashboard
def build_html(res):
    r = res["retrieval"]
    cal = res["calibration"]
    e = html.escape

    rows_html = "".join(
        f"<tr><td>{e(x['q'])}</td><td>{x['dense_rank'] or '-'}</td><td>{x['hybrid_rank'] or '-'}</td>"
        f"<td>{x['conf']}</td><td>{x['top_dense']}</td></tr>" for x in res["retrieval_rows"])

    ans_block = ""
    if res.get("answers"):
        a = res["answers"]
        complete = a["n_done"] == a["n_total"]
        banner_style = "" if complete else "background:#fff3cd;padding:8px;border:1px solid #e0c36a;"
        banner = (f"Based on {a['n_done']} of {a['n_total']} questions"
                  + ("" if complete else " (INCOMPLETE: re-run the same command to resume)")
                  + f" | model: {e(str(a['model']))}")
        per_q = "".join(
            f"<tr><td>{e(x['q'])}</td><td>{fmt(x['single_kw'], True)}</td><td>{fmt(x['agent_kw'], True)}</td>"
            f"<td>{fmt(x['single_cite_prec'], True)}</td><td>{fmt(x['agent_cite_prec'], True)}</td>"
            f"<td>{x['single_calls']}</td><td>{x['agent_calls']}</td></tr>"
            for x in res["answer_rows"])
        ans_block = f"""
<h2>Answer quality (LLM-based)</h2>
<p style="{banner_style}"><b>{banner}</b></p>
<table><tr><th>Metric</th><th>Single pass</th><th>Agentic</th></tr>
<tr><td>Key-fact coverage</td><td>{fmt(a['single_kw'], True)}</td><td>{fmt(a['agent_kw'], True)}</td></tr>
<tr><td>Citation precision</td><td>{fmt(a['single_cite'], True)}</td><td>{fmt(a['agent_cite'], True)}</td></tr>
<tr><td>Avg LLM calls</td><td>{fmt(a['single_calls'])}</td><td>{fmt(a['agent_calls'])}</td></tr>
<tr><td>Avg latency (s) *</td><td>{fmt(a['single_time'])}</td><td>{fmt(a['agent_time'])}</td></tr>
</table>
<p><small>* Latency includes API throttling and retry waits, so it is not a reliable speed comparison.</small></p>
<h2>Per-question answers</h2>
<table><tr><th>Question</th><th>Coverage (single)</th><th>Coverage (agentic)</th>
<th>Cite prec. (single)</th><th>Cite prec. (agentic)</th><th>Calls (single)</th><th>Calls (agentic)</th></tr>{per_q}</table>"""

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>RAG evaluation</title>
<style>body{{font-family:system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}td,th{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
th{{background:#f0f0f0}}</style></head><body>
<h1>RAG evaluation results</h1>
<p>{res['n_questions']} questions ({res['n_answerable']} answerable, {res['n_unanswerable']} unanswerable)</p>
<h2>Retrieval (top-4)</h2>
<table><tr><th>Method</th><th>Hit@4</th><th>MRR</th></tr>
<tr><td>Dense only</td><td>{fmt(r['dense']['hit_at_k'], True)}</td><td>{fmt(r['dense']['mrr'])}</td></tr>
<tr><td>Hybrid (BM25 + dense)</td><td>{fmt(r['hybrid']['hit_at_k'], True)}</td><td>{fmt(r['hybrid']['mrr'])}</td></tr></table>
<h2>Confidence calibration</h2>
<table><tr><th>Check</th><th>Result</th></tr>
<tr><td>Answerable questions not flagged LOW</td><td>{fmt(cal['answerable_not_low'], True)}</td></tr>
<tr><td>Unanswerable questions flagged LOW</td><td>{fmt(cal['unanswerable_low'], True)}</td></tr></table>
{ans_block}
<h2>Per-question retrieval</h2>
<table><tr><th>Question</th><th>Dense rank</th><th>Hybrid rank</th><th>Confidence</th><th>Top cosine</th></tr>{rows_html}</table>
</body></html>"""


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true",
                    help="also evaluate answers (single vs agentic); uses LLM calls")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the saved answer checkpoint and recompute everything")
    ap.add_argument("--provider", choices=["client", "gemini"], default="client",
                    help="'gemini' uses rag/gemini_llm.py and ignores agent/llm_client.py")
    args = ap.parse_args()

    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    store = load_store()

    rows = retrieval_eval(store, qs)
    res = {
        "n_questions": len(qs),
        "n_answerable": sum(q["answerable"] for q in qs),
        "n_unanswerable": sum(not q["answerable"] for q in qs),
        "retrieval_rows": rows,
        "retrieval": {"dense": summarize_retrieval(rows, "dense_rank"),
                      "hybrid": summarize_retrieval(rows, "hybrid_rank")},
        "calibration": calibration(rows),
    }

    if args.with_llm:
        if args.provider == "gemini":
            from rag.gemini_llm import make_gemini_llm
            llm = make_gemini_llm()
            model_name = llm.model_name
        else:
            from agent.llm_client import DEFAULT_CLIENT
            model_name = f"{type(DEFAULT_CLIENT).__name__}:{getattr(DEFAULT_CLIENT, 'model', 'unknown')}"

            def llm(prompt):
                t = DEFAULT_CLIENT.call(
                    system_prompt="You are a careful assistant that answers strictly from provided sources.",
                    conversation=[{"role": "user", "content": prompt}])
                return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()

        print(f"\nUsing model: {model_name}\n")

        arows, stopped = answer_eval(store, qs, llm, model_name, fresh=args.fresh)
        res["answer_rows"] = arows
        if arows:
            res["answers"] = {
                "single_kw": avg([x["single_kw"] for x in arows]),
                "agent_kw": avg([x["agent_kw"] for x in arows]),
                "single_cite": avg([x["single_cite_prec"] for x in arows]),
                "agent_cite": avg([x["agent_cite_prec"] for x in arows]),
                "single_calls": avg([x["single_calls"] for x in arows]),
                "agent_calls": avg([x["agent_calls"] for x in arows]),
                "single_time": avg([x["single_time"] for x in arows]),
                "agent_time": avg([x["agent_time"] for x in arows]),
                "n_done": len(arows),
                "n_total": len(qs),
                "model": model_name,
            }

    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    OUT_HTML.write_text(build_html(res), encoding="utf-8")

    r = res["retrieval"]
    print(f"\nQuestions: {res['n_questions']}")
    print(f"Dense  : Hit@4={fmt(r['dense']['hit_at_k'], True)}  MRR={fmt(r['dense']['mrr'])}")
    print(f"Hybrid : Hit@4={fmt(r['hybrid']['hit_at_k'], True)}  MRR={fmt(r['hybrid']['mrr'])}")
    print(f"Calibration: {res['calibration']}")
    if "answers" in res:
        a = res["answers"]
        print(f"Answers cover {a['n_done']} of {a['n_total']} questions "
              f"{'(COMPLETE)' if a['n_done'] == a['n_total'] else '(INCOMPLETE)'}")
    print(f"\nSaved: {OUT_JSON}\nSaved: {OUT_HTML}")


if __name__ == "__main__":
    main()