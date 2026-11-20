# apply_ui_patch.py
"""
Adds three things to web/index.html WITHOUT replacing it, so your own edits stay:

  1. The sidebar no longer scrolls away with the chat. The page stays fixed and the
     sidebar and the chat each scroll on their own.
  2. The sidebar opens and closes with a button (state is remembered).
  3. Results from "Other sources" appear in the chat, like every other answer,
     instead of inside the small panel.

Run from the RAG_QA folder:
    python apply_ui_patch.py
then hard-refresh the browser (Ctrl+F5).

It is safe to run again: it replaces its own block instead of adding a second one.
A copy of your original file is saved once as web/index.html.bak
(to undo: delete web/index.html and rename index.html.bak back).
"""

import re
import shutil
import sys
from pathlib import Path

INDEX = Path(__file__).resolve().parent / "web" / "index.html"
START, END = "<!-- ui-patch:start -->", "<!-- ui-patch:end -->"

PATCH = START + "\n" + r'''<style>
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
''' + "\n" + END


def main():
    if not INDEX.exists():
        sys.exit(f"Could not find {INDEX}. Run this from the RAG_QA folder.")

    page = INDEX.read_text(encoding="utf-8")

    backup = INDEX.with_name("index.html.bak")
    if not backup.exists():
        shutil.copyfile(INDEX, backup)

    if START in page and END in page:
        page = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: PATCH, page, flags=re.S)
    else:
        last = None
        for last in re.finditer(r"</body>", page, flags=re.I):
            pass
        if last:
            page = page[:last.start()] + PATCH + "\n" + page[last.start():]
        else:
            page = page + "\n" + PATCH + "\n"

    INDEX.write_text(page, encoding="utf-8")
    print(f"Patched {INDEX}")
    print("Now hard-refresh the page with Ctrl+F5.")


if __name__ == "__main__":
    main()