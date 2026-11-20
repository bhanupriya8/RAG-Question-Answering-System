# """
# Runs direct_answer, single_step_rag, and the full ProMSALiteAgent over a
# test set of (image, question, expected_answer) triples, and reports
# accuracy + average steps used per method.

# Before running this:
#   1. python data/scrape_wikipedia.py
#   2. python data/build_text_index.py
#   3. python data/build_image_index.py
#   4. Fill in test_questions.json (see format below)
#   5. Set up an LLM client in agent/llm_client.py (uncomment one, add your API key)

# Run:
#     python eval/evaluate.py
# """

# import json
# import os
# import sys

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from agent.agent_loop import ProMSALiteAgent
# from agent.llm_client import DEFAULT_CLIENT
# from eval.baselines import direct_answer, single_step_rag
# from tools.search_tools import SearchTools

# HERE = os.path.dirname(os.path.abspath(__file__))
# TEST_SET_PATH = os.path.join(HERE, "test_questions.json")
# RESULTS_PATH = os.path.join(HERE, "results.json")


# def judge_correct(llm_client, question: str, predicted: str, expected: str) -> bool:
#     """Simple LLM-as-judge: is the predicted answer correct given the expected answer?
#     Loose semantic match, not exact string match (predicted phrasing will vary)."""
#     prompt = (
#         f"Question: {question}\n"
#         f"Expected answer: {expected}\n"
#         f"Predicted answer: {predicted}\n\n"
#         "Does the predicted answer correctly match the expected answer in substance "
#         "(ignore minor phrasing differences)? Reply with only YES or NO."
#     )
#     verdict = llm_client.call(
#         system_prompt="You are a strict but fair grader.",
#         conversation=[{"role": "user", "content": prompt}],
#     )
#     return "YES" in verdict.strip().upper()


# def main():
#     if DEFAULT_CLIENT is None:
#         raise RuntimeError(
#             "No LLM client configured. Open agent/llm_client.py, uncomment ONE provider "
#             "class, set your API key as an environment variable, and set DEFAULT_CLIENT."
#         )

#     with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
#         test_set = json.load(f)

#     search_tools = SearchTools(top_k=3)
#     agent = ProMSALiteAgent(DEFAULT_CLIENT, search_tools, max_steps=5, verbose=False)

#     all_results = {"direct": [], "single_step_rag": [], "promsa_lite": []}

#     for i, item in enumerate(test_set, 1):
#         image_path = os.path.join(HERE, "..", item["image_path"])
#         question = item["question"]
#         expected = item["expected_answer"]
#         print(f"\n=== [{i}/{len(test_set)}] {question} ===")

#         pred_direct = direct_answer(DEFAULT_CLIENT, image_path, question)
#         correct_direct = judge_correct(DEFAULT_CLIENT, question, pred_direct, expected)
#         all_results["direct"].append({"question": question, "predicted": pred_direct, "correct": correct_direct})
#         print(f"[direct]          {pred_direct}  -> {'✓' if correct_direct else '✗'}")

#         pred_rag = single_step_rag(DEFAULT_CLIENT, search_tools, image_path, question)
#         correct_rag = judge_correct(DEFAULT_CLIENT, question, pred_rag, expected)
#         all_results["single_step_rag"].append({"question": question, "predicted": pred_rag, "correct": correct_rag})
#         print(f"[single_step_rag] {pred_rag}  -> {'✓' if correct_rag else '✗'}")

#         agent_result = agent.run(image_path, question)
#         pred_agent = agent_result["answer"]
#         correct_agent = judge_correct(DEFAULT_CLIENT, question, pred_agent, expected)
#         all_results["promsa_lite"].append({
#             "question": question, "predicted": pred_agent, "correct": correct_agent,
#             "steps_used": agent_result["steps_used"], "trajectory": agent_result["trajectory"],
#         })
#         print(f"[promsa_lite]     {pred_agent}  -> {'✓' if correct_agent else '✗'}  ({agent_result['steps_used']} steps)")

#     # summary
#     print("\n\n=== SUMMARY ===")
#     for method, results in all_results.items():
#         acc = sum(r["correct"] for r in results) / len(results) * 100
#         print(f"{method}: {acc:.1f}% accuracy ({sum(r['correct'] for r in results)}/{len(results)})")

#     with open(RESULTS_PATH, "w", encoding="utf-8") as f:
#         json.dump(all_results, f, indent=2, ensure_ascii=False)
#     print(f"\nFull results saved to {RESULTS_PATH}")


# if __name__ == "__main__":
#     main()








import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.agent_loop import ProMSALiteAgent
from agent.llm_client import DEFAULT_CLIENT
from eval.baselines import direct_answer, single_step_rag
from tools.search_tools import SearchTools

HERE = os.path.dirname(os.path.abspath(__file__))
# TEST_SET_PATH = os.path.join(HERE, "test_questions.json")
# RESULTS_PATH = os.path.join(HERE, "results.json")

TEST_SET_PATH = os.path.join(HERE, os.environ.get("QUESTIONS_FILE", "test_questions.json"))
RESULTS_PATH = os.path.join(HERE, os.environ.get("RESULTS_FILE", "results.json"))


def judge_correct(llm_client, question: str, predicted: str, expected: str) -> bool:
    """Simple LLM-as-judge: is the predicted answer correct given the expected answer?
    Loose semantic match, not exact string match (predicted phrasing will vary)."""
    prompt = (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Predicted answer: {predicted}\n\n"
        "Does the predicted answer correctly match the expected answer in substance "
        "(ignore minor phrasing differences)? Reply with only YES or NO."
    )
    verdict = llm_client.call(
        system_prompt="You are a strict but fair grader.",
        conversation=[{"role": "user", "content": prompt}],
    )
    return "YES" in verdict.strip().upper()


def load_existing_results():
    """Load previously saved results so a rerun can resume instead of starting over."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"direct": [], "single_step_rag": [], "promsa_lite": []}


def save_results(all_results):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def already_done(all_results, question: str) -> bool:
    """A question counts as done only if it completed all three methods."""
    done_direct = any(r["question"] == question for r in all_results["direct"])
    done_rag = any(r["question"] == question for r in all_results["single_step_rag"])
    done_agent = any(r["question"] == question for r in all_results["promsa_lite"])
    return done_direct and done_rag and done_agent


def main():
    if DEFAULT_CLIENT is None:
        raise RuntimeError(
            "No LLM client configured. Open agent/llm_client.py, uncomment ONE provider "
            "class, set your API key as an environment variable, and set DEFAULT_CLIENT."
        )

    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    search_tools = SearchTools(top_k=3)
    agent = ProMSALiteAgent(DEFAULT_CLIENT, search_tools, max_steps=5, verbose=False)

    all_results = load_existing_results()
    stopped_early = False

    for i, item in enumerate(test_set, 1):
        image_path = os.path.join(HERE, "..", item["image_path"])
        question = item["question"]
        expected = item["expected_answer"]

        if already_done(all_results, question):
            print(f"\n=== [{i}/{len(test_set)}] {question} === (skipped, already completed)")
            continue

        print(f"\n=== [{i}/{len(test_set)}] {question} ===")

        try:
            pred_direct = direct_answer(DEFAULT_CLIENT, image_path, question)
            correct_direct = judge_correct(DEFAULT_CLIENT, question, pred_direct, expected)
            all_results["direct"].append({"question": question, "predicted": pred_direct, "correct": correct_direct})
            print(f"[direct]          {pred_direct}  -> {'✓' if correct_direct else '✗'}")
            save_results(all_results)

            pred_rag = single_step_rag(DEFAULT_CLIENT, search_tools, image_path, question)
            correct_rag = judge_correct(DEFAULT_CLIENT, question, pred_rag, expected)
            all_results["single_step_rag"].append({"question": question, "predicted": pred_rag, "correct": correct_rag})
            print(f"[single_step_rag] {pred_rag}  -> {'✓' if correct_rag else '✗'}")
            save_results(all_results)

            agent_result = agent.run(image_path, question)
            pred_agent = agent_result["answer"]
            correct_agent = judge_correct(DEFAULT_CLIENT, question, pred_agent, expected)
            all_results["promsa_lite"].append({
                "question": question, "predicted": pred_agent, "correct": correct_agent,
                "steps_used": agent_result["steps_used"], "trajectory": agent_result["trajectory"],
            })
            print(f"[promsa_lite]     {pred_agent}  -> {'✓' if correct_agent else '✗'}  ({agent_result['steps_used']} steps)")
            save_results(all_results)

        except RuntimeError as e:
            # Raised by GeminiClient after exhausting retries (quota/rate limit exceeded).
            print(f"\n[STOPPED] Quota/rate limit error on question {i}/{len(test_set)}: {e}")
            print("Progress so far has been saved. Rerun this script later to resume "
                  "from where it left off (already-completed questions will be skipped).")
            stopped_early = True
            break

    # summary
    print("\n\n=== SUMMARY ===")
    for method, results in all_results.items():
        if not results:
            print(f"{method}: no completed results yet")
            continue
        acc = sum(r["correct"] for r in results) / len(results) * 100
        print(f"{method}: {acc:.1f}% accuracy ({sum(r['correct'] for r in results)}/{len(results)})")

    save_results(all_results)
    print(f"\nResults saved to {RESULTS_PATH}")

    if stopped_early:
        print(f"\n{len(test_set) - len(all_results['promsa_lite'])} question(s) remain incomplete. "
              "Rerun once your quota resets (daily quota resets at midnight Pacific Time).")


if __name__ == "__main__":
    main()