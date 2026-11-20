
# rag/agentic.py
import re
from dataclasses import dataclass, field
from typing import Callable

from rag.answer import Answer, build_prompt, finalize
from rag.confidence import assess_confidence

# AGENT_RULES = """You answer questions using ONLY the numbered sources below.
# Choose exactly one action:
# 1. If the sources cover the question, write the answer. Cite every claim with
#    source numbers in square brackets, e.g. [1] or [2][3]. Be concise.
# 2. If an important part of the question is NOT covered by the sources, reply with
#    exactly one line and nothing else:
#    SEARCH: <2 to 6 keywords>
# Search query rules:
# - Use only technical terms that appear in the question or the sources (algorithm
#   names, operation names). Do not add explanatory words.
# - Do not repeat or rephrase a search that was already done.
# - Never search for something the sources already cover.
# Never use outside knowledge."""

AGENT_RULES = """You are an educational RAG assistant.
Use ONLY the numbered source passages provided below.

Your task is to decide whether the available evidence is sufficient
to answer the user's complete question.

Choose exactly one action:

1. If the sources sufficiently cover the question, write a clear,
   complete answer using the available evidence.
   Cite every important factual claim using source numbers in
   square brackets, such as [1] or [2][3].


2. If important information requested by the user is missing,
   issue up to three focused search requests, each on a separate line,
   using this exact format:
   SEARCH: <2 to 6 keywords>

SEARCH RULES:
- Search only for information that is missing from the current evidence.
- Prefer specific technical terms, concepts and algorithm names
  appearing in the question or sources.
- If the question asks for multiple components, check whether
  the evidence covers all of them.
- Search for useful definitions, steps, examples or explanations
  when the question explicitly requests them and they are missing.
- You may request up to three different searches in one response.
- Put each search request on its own line.
- Do not repeat a search that has already been performed.
- Prioritize different missing aspects rather than similar queries.
- Request only searches that can add useful evidence.
- Do not search unnecessarily when the evidence is already sufficient.
- Never use outside knowledge.
- Treat the user's original question as the fixed scope for all searches.
- For comparison questions, identify the exact categories being compared.
- Search only for information about those requested categories
  and the algorithms explicitly assigned to them by the supplied sources.
- An algorithm merely mentioned in a retrieved passage is NOT
  automatically relevant to the user's question.
- Do not investigate another algorithm just because it is related
  to the same subject or appears elsewhere in the PDF.
- Do not search for algorithms that are outside the requested
  categories, even if they are commonly associated with the topic.
- Do not introduce new comparison categories or topics during retrieval.
- If an algorithm's category is unclear in the supplied sources,
  do not assume its classification or expand the search scope.
- Before issuing a search, verify that its subject belongs to
  the original question and addresses a specific missing requirement.

COVERAGE CHECK FOR BROAD QUESTIONS:
- Break the user's question into all requested parts and comparison dimensions.
- Identify the algorithms or methods that the supplied sources explicitly
  list under each relevant category.
- When advantages or limitations are requested, check each relevant
  algorithm separately.
- Do not assume that evidence about one algorithm covers every algorithm
  belonging to the same broad category.
- When a named algorithm is present in the sources but its requested
  information is missing from the current passages, search specifically
  for that algorithm and the missing information.
- Use focused searches of 2 to 6 keywords.
- Do not conclude that a category has no limitations simply because
  limitations for one of its algorithms were not found.
- Do not assume information is absent from the uploaded PDF merely
  because it is absent from the currently retrieved passages.
- If the search budget is exhausted, clearly identify which specific
  information could not be retrieved.

ANSWER QUALITY:
- Write in clear language suitable for an undergraduate student.
- Explain technical terms when necessary.
- Combine relevant information from multiple passages.
- Do not simply copy retrieved passages.
- Do not invent missing facts or examples.


SOURCE FIDELITY:
- Preserve classifications, terminology, distinctions and
  relationships explicitly stated in the supplied sources.
- Do not merge separate categories or change the classification
  of an algorithm, method or concept.
- Treat examples according to the categories assigned by the sources.
- When sources provide a classification table, preserve its distinctions.
- Do not introduce outside classifications.

LIMITATIONS:
- Attribute each limitation to the specific algorithm, method
  or component described in the sources.
- Do not generalize one algorithm's limitation to an entire category.
- Include only limitations supported by the evidence.
- If requested limitations are not discussed, say so clearly."""

FINAL_RULES = """You are an educational question-answering assistant.
Produce the final answer using ONLY the numbered source passages
provided below.

ANSWER REQUIREMENTS:
1. Answer every part of the user's question.
2. Provide a complete, accurate and understandable explanation.
3. Use simple language suitable for an undergraduate student.
4. Combine relevant information from multiple sources into
   one coherent explanation.
5. Explain important technical terms and processes.
6. Include examples and step-by-step descriptions whenever
   they are supported by the sources and useful for understanding.
7. Include advantages, limitations or comparisons when requested
   or relevant and supported by the sources.
8. Avoid unnecessary repetition and irrelevant details.

EVIDENCE COVERAGE:
- Address every requested part of the question.
- Preserve separate algorithms and methods when comparing broad categories.
- Attribute advantages and limitations to the specific algorithms
  identified in the supplied sources.
- Do not generalize an algorithm-specific limitation to an entire category.
- Do not report that an entire category has no limitations merely
  because one algorithm's limitations are not discussed.
- Clearly distinguish information supported by retrieved passages
  from information that could not be verified.
- If a requested comparison dimension is unsupported, identify it
  rather than filling it with assumptions.

SOURCE CLASSIFICATION AND ACCURACY:
1. Preserve the classifications and terminology used in the sources.
2. Keep Top-Down, Bottom-Up, Dynamic Programming/Chart-Based,
   Network-Based and Probabilistic/Statistical parsing distinct
   when the sources distinguish them.
3. Do not classify RTN as a Top-Down algorithm. The supplied
   document identifies RTN as Network-Based Parsing.
4. Do not confuse an algorithm's limitation with a limitation
   of its entire parsing category.

LIMITATIONS:
1. Explain limitations only when supported by the sources.
2. Attribute each limitation to the specific algorithm.
3. For example, identify the CNF requirement and computational
   cost for long sentences as CYK limitations, not general
   limitations of every Bottom-Up parser.
4. If the sources do not state limitations for an algorithm,
   explicitly say so rather than inventing them.

COMPARISON:
When the user explicitly requests a comparison, provide a
Markdown comparison table whenever the available evidence
supports one.
Use clear comparison dimensions such as starting point,
direction, working process, representative algorithms,
advantages and limitations.
Do not fill unsupported table cells with guesses.

STRUCTURE:
Organize educational answers where appropriate using:
- Introduction or definition.
- Detailed explanation.
- Important components, types, operations or steps.
- Working process and examples.
- Advantages and limitations, when supported.
- Short conclusion.

Do not force unnecessary headings into simple factual answers.

CITATIONS:
- Cite every important factual claim using individual source numbers.
- Use adjacent references for multiple sources, such as [1][2][3].
- Never use comma-separated citation groups such as [1, 2, 3].
- Never format citation numbers as bold text.
- Place citations close to the claims they support.
- Use only valid source numbers provided in the prompt.
- Do not cite a source unless it supports the associated claim.

MISSING INFORMATION:
- Answer the parts that are supported by the sources.
- Clearly identify any important information that is missing.
- Never invent algorithm names, values, examples or explanations.
- Never use outside knowledge to fill gaps.
- If the sources cannot answer the question, clearly explain this.

COMPLETENESS:
Do not sacrifice important explanations merely to make
the answer short.
The final response should help a student understand
the requested concept, not merely repeat retrieved passages."""

# SEARCH_RE = re.compile(r"^\s*SEARCH:\s*(.+)", re.IGNORECASE)


# @dataclass
# class AgenticResult:
#     answer: Answer
#     trace: list[dict] = field(default_factory=list)   # one entry per step
#     llm_calls: int = 0


# def agentic_answer(store, query: str, llm: Callable[[str], str], max_steps: int = 3,
#                    k: int = 4, max_evidence: int = 8,
#                    docs: list[str] | None = None) -> AgenticResult:
#     evidence: dict[int, tuple] = {}      # chunk_id -> (chunk, fused, dense)
#     trace: list[dict] = []
#     searches_done: list[str] = []        # refined queries already tried
#     seen_queries = {query.lower()}
#     current = query
#     force_final = False
#     conf = None
#     calls = 0

#     for step in range(1, max_steps + 1):
#         # # refined searches (step 2+) retrieve a few more chunks than the first pass
#         # results = store.hybrid_search(current, k=k if step == 1 else k + 2, docs=docs)

#         # Skip another retrieval if the model repeated a search query.
#         # Use the evidence already collected to generate the final answer.
#         if force_final:
#             results = []
#         else:
#             results = store.hybrid_search(current, k=k if step == 1 else k + 2, docs=docs)

#         # Step 1: confidence gate on the user's original question
#         # if step == 1:
#         #     conf = assess_confidence(results)
#         #     if conf.level == "low":
#         #         trace.append({"step": 1, "query": current, "new_chunks": 0,
#         #                       "decision": "stop (low confidence)"})
#         #         return AgenticResult(Answer(conf.message, conf), trace, calls)

#         if step == 1:
#             conf = assess_confidence(results)
#             if not results:
#                 trace.append({"step": 1, "query": current, "new_chunks": 0,
#                               "decision": "stop (no retrieved passages)"})
#                 return AgenticResult(Answer(conf.message, conf), trace, calls)

#         added = 0
#         for item in results:
#             chunk = item[0]
#             if chunk.chunk_id not in evidence and len(evidence) < max_evidence:
#                 evidence[chunk.chunk_id] = item
#                 added += 1

#         if step > 1 and added == 0:       # search found nothing new -> stop searching
#             force_final = True
#         is_last = force_final or step == max_steps

#         rules = FINAL_RULES if is_last else AGENT_RULES
#         if searches_done and not is_last:
#             rules += "\n\nSearches already done (do not repeat): " + "; ".join(searches_done)

#         prompt, sources = build_prompt(query, list(evidence.values()), rules)
#         raw = llm(prompt).strip()
#         calls += 1

#         m = SEARCH_RE.match(raw)
#         # if m and not is_last:
#         #     nxt = m.group(1).strip().splitlines()[0].strip('"')
#         #     trace.append({"step": step, "query": current, "new_chunks": added,
#         #                   "decision": f"search -> {nxt}"})
#         #     if nxt.lower() in seen_queries:   # model repeated itself
#         #         force_final = True
#         #     else:
#         #         seen_queries.add(nxt.lower())
#         #         searches_done.append(nxt)
#         #         current = nxt
#         #     continue
        
        
#         if m and not is_last:
#             nxt = m.group(1).strip().splitlines()[0].strip('"')

#             if nxt.lower() in seen_queries:   # model repeated itself
#                 trace.append({
#                     "step": step,
#                     "query": current,
#                     "new_chunks": added,
#                     "decision": "stop (repeated search query; generating final answer)"
#                 })
#                 force_final = True
#             else:
#                 trace.append({
#                     "step": step,
#                     "query": current,
#                     "new_chunks": added,
#                     "decision": f"search -> {nxt}"
#                 })
#                 seen_queries.add(nxt.lower())
#                 searches_done.append(nxt)
#                 current = nxt

#             continue


#         trace.append({"step": step, "query": current, "new_chunks": added, "decision": "answer"})
#         if m:                                 # model ignored the "final answer" rule
#             raw = "I couldn't find enough information in the documents to answer this fully."
#         return AgenticResult(finalize(raw, sources, conf), trace, calls)

#     # safety net (loop always returns above)
#     return AgenticResult(Answer("No answer produced.", conf), trace, calls)



SEARCH_RE = re.compile(
    r"^\s*SEARCH:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE
)

MAX_SEARCH_QUERIES_PER_BATCH = 3


def extract_search_queries(text: str) -> list[str]:
    """Extract up to three SEARCH requests from an LLM response."""

    matches = SEARCH_RE.findall(text)

    queries = []
    seen = set()

    for item in matches:
        query = item.strip().strip('"').strip("'")
        normalized = " ".join(query.lower().split())

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        queries.append(query)

        if len(queries) >= MAX_SEARCH_QUERIES_PER_BATCH:
            break

    return queries


@dataclass
class AgenticResult:
    answer: Answer
    trace: list[dict] = field(default_factory=list)
    llm_calls: int = 0


def agentic_answer(
    store,
    query: str,
    llm: Callable[[str], str],
    max_steps: int = 3,
    k: int = 4,
    max_evidence: int = 8,
    docs: list[str] | None = None
) -> AgenticResult:

    evidence: dict[int, tuple] = {}
    trace: list[dict] = []

    searches_done: list[str] = []

    seen_queries = {
        " ".join(query.lower().split())
    }

    pending_queries: list[str] = []
    current = query

    force_final = False
    conf = None
    calls = 0

    for step in range(1, max_steps + 1):

        # Step 1 retrieves the original question.
        # Later steps retrieve all pending focused queries.
        if step == 1:
            queries_to_run = [query]

        elif force_final:
            queries_to_run = []

        # else:
        #     queries_to_run = pending_queries
        #     pending_queries = []

        # added = 0
        # initial_results = []
        
        else:
            queries_to_run = pending_queries
            pending_queries = []

        # Preserve the queries actually executed in this step.
        batch_label = (
            "; ".join(queries_to_run)
            if queries_to_run
            else current
        )

        # added = 0
        
        # # Retrieve evidence for every query in this batch.
        # for search_query in queries_to_run:

        #     if len(evidence) >= max_evidence:
        #         break

        #     results = store.hybrid_search(
        #         search_query,
        #         k=k if step == 1 else k + 2,
        #         docs=docs
        #     )

        #     if step == 1:
        #         initial_results = results

        #     for item in results:

        #         chunk = item[0]

        #         if chunk.chunk_id in evidence:
        #             continue

        #         if len(evidence) >= max_evidence:
        #             break

        #         evidence[chunk.chunk_id] = item
        #         added += 1
        
        
        added = 0
        batch_results = []

        # First retrieve results for every query in this batch.
        for search_query in queries_to_run:

            results = store.hybrid_search(
                search_query,
                k=k if step == 1 else k + 2,
                docs=docs
            )

            if step == 1:
                initial_results = results

            batch_results.append(results)

        # Collect evidence in round-robin order.
        # This prevents the first query from consuming the entire evidence budget.
        max_results = max(
            (len(results) for results in batch_results),
            default=0
        )

        for rank_index in range(max_results):

            if len(evidence) >= max_evidence:
                break

            for results in batch_results:

                if len(evidence) >= max_evidence:
                    break

                if rank_index >= len(results):
                    continue

                item = results[rank_index]
                chunk = item[0]

                # Do not store duplicate chunks.
                if chunk.chunk_id in evidence:
                    continue

                evidence[chunk.chunk_id] = item
                added += 1

        # Assess confidence using the original question retrieval.
        if step == 1:

            conf = assess_confidence(initial_results)

            if not initial_results:

                trace.append({
                    "step": 1,
                    # "query": query,
                    "query": batch_label,
                    "new_chunks": 0,
                    "decision": "stop (no retrieved passages)"
                })

                return AgenticResult(
                    Answer(conf.message, conf),
                    trace,
                    calls
                )

        # No new evidence means that another search round
        # is unlikely to improve the answer.
        if step > 1 and added == 0:
            force_final = True

        is_last = force_final or step == max_steps

        rules = FINAL_RULES if is_last else AGENT_RULES

        if searches_done and not is_last:

            rules += (
                "\n\nSearches already done (do not repeat): "
                + "; ".join(searches_done)
            )

        prompt, sources = build_prompt(
            query,
            list(evidence.values()),
            rules
        )

        raw = llm(prompt).strip()
        calls += 1

        requested_queries = extract_search_queries(raw)

        # Process additional searches when the search budget allows.
        if requested_queries and not is_last:

            new_queries = []

            for next_query in requested_queries:

                normalized = " ".join(
                    next_query.lower().split()
                )

                if normalized in seen_queries:
                    continue

                if normalized in {
                    " ".join(q.lower().split())
                    for q in new_queries
                }:
                    continue

                new_queries.append(next_query)

            # Every requested search was already performed.
            if not new_queries:

                trace.append({
                    "step": step,
                    # "query": current,
                    "query": batch_label,
                    "new_chunks": added,
                    "decision": (
                        "stop (repeated search query; "
                        "generating final answer)"
                    )
                })

                force_final = True

            else:

                for next_query in new_queries:

                    seen_queries.add(
                        " ".join(next_query.lower().split())
                    )

                    searches_done.append(next_query)

                pending_queries = new_queries
                current = "; ".join(new_queries)

                trace.append({
                    "step": step,
                    "query": batch_label,
                    "new_chunks": added,
                    "decision": "search -> " + "; ".join(new_queries)
                })

            continue

        # The model requested another search after the budget ended.
        if requested_queries:

            raw = (
                "I couldn't find enough information in the "
                "documents to answer this fully."
            )

        trace.append({
            "step": step,
            "query": batch_label,
            "new_chunks": added,
            "decision": "answer"
        })

        return AgenticResult(
            finalize(raw, sources, conf),
            trace,
            calls
        )

    return AgenticResult(
        Answer("No answer produced.", conf),
        trace,
        calls
    )
