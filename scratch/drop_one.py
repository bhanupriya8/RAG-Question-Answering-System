
# eval/drop_one.py
import json
from pathlib import Path

p = Path("eval/rag_answer_checkpoint.json")
d = json.loads(p.read_text(encoding="utf-8"))
before = len(d["rows"])
d["rows"] = [r for r in d["rows"] if "difference between stemming" not in r["q"]]
p.write_text(json.dumps(d, indent=2), encoding="utf-8")
print(before, "->", len(d["rows"]))