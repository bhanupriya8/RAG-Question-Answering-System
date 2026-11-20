
# # eval/check_questions.py
# """
# Checks a Part A question file before you spend LLM calls on it.

# For every question it verifies that:
#   - the landmark exists in data/knowledge_base.json,
#   - the 'support' phrase (the words that contain the answer) appears in the first
#     400 characters of that landmark's text, which is all the agent and the
#     single_step_rag baseline show to the model,
#   - the query image file exists,
#   - the query image is NOT byte-identical to the thumbnail that was indexed.

# Usage (from the RAG_QA folder):
#     python eval/check_questions.py
#     python eval/check_questions.py eval/test_questions_v2.json
# """

# import hashlib
# import json
# import os
# import sys

# from torch.linalg import norm

# HERE = os.path.dirname(os.path.abspath(__file__))
# ROOT = os.path.normpath(os.path.join(HERE, ".."))
# KB_PATH = os.path.join(ROOT, "data", "knowledge_base.json")
# LIMIT = 400  # characters of each fact text that the model actually sees


# def md5(path):
#     with open(path, "rb") as f:
#         return hashlib.md5(f.read()).hexdigest()


# def main():
#     qfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "test_questions_v2.json")

#     with open(KB_PATH, "r", encoding="utf-8") as f:
#         kb = {e["title"]: e for e in json.load(f)}
#     with open(qfile, "r", encoding="utf-8") as f:
#         items = json.load(f)

#     failed = 0
#     for i, item in enumerate(items, 1):
#         notes = []
#         title = item.get("landmark")
#         entry = kb.get(title)

#         if entry is None:
#             notes.append(f"landmark '{title}' is not in knowledge_base.json")
#         else:
#             support = item.get("support", "")
#             if not support:
#                 notes.append("no 'support' phrase to check")
#             else:
#                 # pos = entry["extract"].lower().find(support.lower())
#                 pos = norm(entry["extract"]).find(norm(support))
#                 if pos < 0:
#                     notes.append("support phrase not found in the landmark text")
#                 elif pos + len(support) > LIMIT:
#                     notes.append(
#                         f"support phrase ends at character {pos + len(support)}, "
#                         f"beyond the {LIMIT} the model sees"
#                     )

#         img = os.path.join(ROOT, item["image_path"])
#         if not os.path.exists(img):
#             notes.append("image file is missing")
#         elif entry is not None and entry.get("image_path"):
#             indexed = os.path.join(ROOT, "data", entry["image_path"].replace("\\", "/"))
#             if os.path.exists(indexed) and md5(img) == md5(indexed):
#                 notes.append("query image is identical to the indexed thumbnail")

#         status = "OK  " if not notes else "FAIL"
#         if notes:
#             failed += 1
#         line = f"{status} [{i}/{len(items)}] {item['question'][:60]}"
#         if notes:
#             line += "  <- " + "; ".join(notes)
#         print(line)

#     print(f"\n{len(items) - failed} of {len(items)} questions pass all checks")


# if __name__ == "__main__":
#     main()






# eval/check_questions.py
"""
Checks a Part A question file before you spend LLM calls on it.

For every question it verifies that:
  - the landmark exists in data/knowledge_base.json,
  - the 'support' phrase (the words that contain the answer) appears in the first
    400 characters of that landmark's text, which is all the agent and the
    single_step_rag baseline show to the model,
  - the query image file exists,
  - the query image is NOT byte-identical to the thumbnail that was indexed.

Usage (from the RAG_QA folder):
    python eval/check_questions.py
    python eval/check_questions.py eval/test_questions_v2.json
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
KB_PATH = os.path.join(ROOT, "data", "knowledge_base.json")
LIMIT = 400  # characters of each fact text that the model actually sees


def norm(text):
    """Lower-case and collapse every kind of whitespace (including the
    non-breaking spaces Wikipedia puts between numbers and units, such as
    in '335 km') into single plain spaces."""
    return " ".join(text.split()).lower()


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    qfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "test_questions_v2.json")

    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = {e["title"]: e for e in json.load(f)}
    with open(qfile, "r", encoding="utf-8") as f:
        items = json.load(f)

    failed = 0
    for i, item in enumerate(items, 1):
        notes = []
        title = item.get("landmark")
        entry = kb.get(title)

        if entry is None:
            notes.append(f"landmark '{title}' is not in knowledge_base.json")
        else:
            support = item.get("support", "")
            if not support:
                notes.append("no 'support' phrase to check")
            else:
                pos = norm(entry["extract"]).find(norm(support))
                if pos < 0:
                    notes.append("support phrase not found in the landmark text")
                elif pos + len(support) > LIMIT:
                    notes.append(
                        f"support phrase ends at character {pos + len(support)}, "
                        f"beyond the {LIMIT} the model sees"
                    )

        img = os.path.join(ROOT, item["image_path"])
        if not os.path.exists(img):
            notes.append("image file is missing")
        elif entry is not None and entry.get("image_path"):
            indexed = os.path.join(ROOT, "data", entry["image_path"].replace("\\", "/"))
            if os.path.exists(indexed) and md5(img) == md5(indexed):
                notes.append("query image is identical to the indexed thumbnail")

        status = "OK  " if not notes else "FAIL"
        if notes:
            failed += 1
        line = f"{status} [{i}/{len(items)}] {item['question'][:60]}"
        if notes:
            line += "  <- " + "; ".join(notes)
        print(line)

    print(f"\n{len(items) - failed} of {len(items)} questions pass all checks")


if __name__ == "__main__":
    main()