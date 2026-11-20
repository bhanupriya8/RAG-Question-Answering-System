"""
Two baselines to compare your agent against - this comparison is what makes
your evaluation results meaningful.

  1. direct_answer: show the LLM the image + question, no search at all.
  2. single_step_rag: do ONE image_search, dump the top result into context,
     then answer. This is the "fixed retrieve-then-generate pipeline" the
     ProMSA paper argues against.
"""


def direct_answer(llm_client, image_path: str, question: str) -> str:
    reply = llm_client.call(
        system_prompt="Answer the visual question as accurately as you can, using only the image.",
        conversation=[{"role": "user", "content": f"Question: {question}\nAnswer concisely."}],
        image_path=image_path,
    )
    return reply.strip()


def single_step_rag(llm_client, search_tools, image_path: str, question: str) -> str:
    search_tools.reset()
    raw_result = search_tools.image_search(image_path)

    if raw_result.get("results"):
        context = "\n".join(f"- {r['title']}: {r['extract'][:400]}" for r in raw_result["results"])
    else:
        context = "(no search results found)"

    prompt = (
        f"Question: {question}\n\n"
        f"Here is some retrieved context that may help:\n{context}\n\n"
        "Answer concisely using the image and the context above."
    )
    reply = llm_client.call(
        system_prompt="Answer the visual question using the image and retrieved context.",
        conversation=[{"role": "user", "content": prompt}],
        image_path=image_path,
    )
    return reply.strip()
