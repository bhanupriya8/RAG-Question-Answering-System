# fix_kb.py
import json
import shutil

p = "data/knowledge_base.json"
shutil.copy(p, p + ".bak")                       # backup of the original

text = open(p, encoding="utf-8").read().lstrip()
obj, end = json.JSONDecoder().raw_decode(text)   # reads only the first JSON value
print("Text found after the list:", repr(text[end:end + 200]))

if not isinstance(obj, list) or len(obj) < 50:
    raise SystemExit(f"Unexpected content: {type(obj).__name__} with {len(obj)} items. File not changed.")

with open(p, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, ensure_ascii=False)
print(len(obj), "entries kept. Backup saved as", p + ".bak")