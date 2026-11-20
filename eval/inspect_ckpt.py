
# eval/inspect_ckpt.py
import json
from pathlib import Path

rows = json.loads((Path(__file__).parent / "rag_answer_checkpoint.json").read_text(encoding="utf-8"))
print(f"{len(rows)} rows\n")
print(f"{'single_calls':>12} {'agent_calls':>11} {'s_abst':>6} {'a_abst':>6} {'s_time':>6} {'a_time':>6} | question")
for r in rows:
    print(f"{r.get('single_calls')!s:>12} {r.get('agent_calls')!s:>11} "
          f"{r.get('single_abstained')!s:>6} {r.get('agent_abstained')!s:>6} "
          f"{r.get('single_time')!s:>6} {r.get('agent_time')!s:>6} | {r['q'][:50]}")