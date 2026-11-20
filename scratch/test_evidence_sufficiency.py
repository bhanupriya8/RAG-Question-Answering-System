
from types import SimpleNamespace

from rag.answer import answer_question, Source, finalize
from rag.confidence import Confidence
from rag.agentic import agentic_answer


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.search_calls = []

    def hybrid_search(self, query, k=5, docs=None):
        self.search_calls.append(query)

        if isinstance(self.results, dict):
            return self.results.get(query, [])

        return self.results


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = 0
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def make_chunk(chunk_id, text):
    return SimpleNamespace(
        chunk_id=chunk_id,
        doc="Test_Document.pdf",
        page=5,
        text=text
    )


def make_low_similarity_results():
    chunk = make_chunk(
        "test_chunk_1",
        "The document discusses Earley parsing and its chart operations."
    )

    return [(chunk, 0.10, 0.20)]


def test_single_pass_low_similarity():
    print("\nTEST 1: Single-pass with low similarity")

    store = FakeStore(make_low_similarity_results())
    llm = FakeLLM(
        "The retrieved passage discusses Earley parsing, "
        "but does not explain nullable non-terminals."
    )

    result = answer_question(
        store,
        "How are nullable non-terminals handled?",
        llm
    )

    assert llm.calls == 1
    assert "nullable non-terminals" in result.text.lower()

    print("PASS: LLM was called despite low similarity.")
    print("Similarity level:", result.confidence.level)
    print("LLM calls:", llm.calls)


def test_agentic_low_similarity():
    print("\nTEST 2: Agentic with low similarity")

    store = FakeStore(make_low_similarity_results())
    llm = FakeLLM(
        "The retrieved passage does not explain how nullable "
        "non-terminals are handled."
    )

    result = agentic_answer(
        store,
        "How are nullable non-terminals handled?",
        llm,
        max_steps=1
    )

    assert llm.calls == 1
    assert result.llm_calls == 1

    print("PASS: Agentic mode continued despite low similarity.")
    print("Similarity level:", result.answer.confidence.level)
    print("LLM calls:", result.llm_calls)
    print("Trace:", result.trace)


def test_empty_retrieval():
    print("\nTEST 3: Empty retrieval")

    store = FakeStore([])
    llm = FakeLLM("This response should never be generated.")

    result = answer_question(store, "Any question?", llm)

    assert llm.calls == 0
    assert not result.sources

    print("PASS: System stopped safely when retrieval was empty.")
    print("LLM calls:", llm.calls)


def test_new_search_adds_evidence():
    print("\nTEST 4: New search retrieves additional evidence")

    chunk1 = make_chunk("chunk_1", "Earley parsing uses a chart.")
    chunk2 = make_chunk("chunk_2", "The scanner advances chart states.")

    store = FakeStore({
        "Explain Earley parsing": [(chunk1, 0.8, 0.6)],
        "Earley scanner": [(chunk2, 0.7, 0.5)]
    })

    llm = FakeLLM([
        "SEARCH: Earley scanner",
        "Earley parsing uses a chart. The scanner advances chart states. [1][2]"
    ])

    result = agentic_answer(
        store,
        "Explain Earley parsing",
        llm,
        max_steps=3
    )

    assert store.search_calls == ["Explain Earley parsing", "Earley scanner"]
    assert llm.calls == 2
    assert result.llm_calls == 2
    assert len(result.answer.sources) == 2
    assert "scanner" in result.answer.text.lower()

    print("PASS: Additional evidence was retrieved and used.")
    print("Search calls:", store.search_calls)
    print("LLM calls:", result.llm_calls)
    print("Trace:", result.trace)


def test_repeated_search():
    print("\nTEST 5: Repeated search does not cause unnecessary retrieval")

    chunk = make_chunk("chunk_1", "Earley parsing uses a chart.")

    store = FakeStore({
        "Explain Earley parsing": [(chunk, 0.8, 0.6)]
    })

    llm = FakeLLM([
        "SEARCH: Explain Earley parsing",
        "The available evidence explains that Earley parsing uses a chart. [1]"
    ])

    result = agentic_answer(
        store,
        "Explain Earley parsing",
        llm,
        max_steps=3
    )

    assert store.search_calls == ["Explain Earley parsing"]
    assert llm.calls == 2
    assert result.llm_calls == 2
    # assert result.trace[-1]["decision"] == "answer"
    assert result.trace[0]["decision"] == (
        "stop (repeated search query; generating final answer)"
    )

    print("PASS: Repeated search skipped another retrieval.")
    print("Search calls:", store.search_calls)
    print("LLM calls:", result.llm_calls)
    print("Trace:", result.trace)


def test_no_new_evidence():
    print("\nTEST 6: Search finds no new evidence")

    chunk = make_chunk("chunk_1", "Earley parsing uses a chart.")

    store = FakeStore({
        "Explain Earley parsing": [(chunk, 0.8, 0.6)],
        "Earley chart": [(chunk, 0.8, 0.6)]
    })

    llm = FakeLLM([
        "SEARCH: Earley chart",
        "The evidence explains that Earley parsing uses a chart. [1]"
    ])

    result = agentic_answer(
        store,
        "Explain Earley parsing",
        llm,
        max_steps=3
    )

    assert store.search_calls == ["Explain Earley parsing", "Earley chart"]
    assert llm.calls == 2
    assert result.trace[-1]["decision"] == "answer"

    print("PASS: Agent stopped searching when no new evidence appeared.")
    print("Search calls:", store.search_calls)
    print("Trace:", result.trace)


def test_maximum_steps():
    print("\nTEST 7: Maximum search steps are respected")

    chunk1 = make_chunk("chunk_1", "Earley parsing uses a chart.")
    chunk2 = make_chunk("chunk_2", "Earley parsing uses prediction.")

    store = FakeStore({
        "Explain Earley parsing": [(chunk1, 0.8, 0.6)],
        "Earley prediction": [(chunk2, 0.7, 0.5)]
    })

    llm = FakeLLM([
        "SEARCH: Earley prediction",
        "The answer is based on the available evidence. [1][2]"
    ])

    result = agentic_answer(
        store,
        "Explain Earley parsing",
        llm,
        max_steps=2
    )

    assert len(store.search_calls) == 2
    assert llm.calls == 2
    assert result.trace[-1]["decision"] == "answer"

    print("PASS: Agent respected the maximum number of steps.")
    print("Search calls:", store.search_calls)
    print("LLM calls:", result.llm_calls)


def test_grouped_citations_finalize():
    print("\nTEST 8: Grouped citations map correctly to sources")

    sources = [
        Source(i, f"Doc{i}.pdf", i, f"Snippet {i}")
        for i in range(1, 6)
    ]

    confidence = Confidence("high", 0.75, "")

    raw_text = (
        "First claim [1]. "
        "Grouped claim [3, 5]. "
        "Adjacent claim [2][4]. "
        "Invalid citation [99]. "
        "Mixed citation [1, 99]."
    )

    result = finalize(raw_text, sources, confidence, 1)

    expected_text = (
        "First claim [1]. "
        "Grouped claim [2][3]. "
        "Adjacent claim [4][5]. "
        "Invalid citation. "
        "Mixed citation [1]."
    )

    assert result.text == expected_text

    expected_sources = [
        ("Doc1.pdf", 1),
        ("Doc3.pdf", 3),
        ("Doc5.pdf", 5),
        ("Doc2.pdf", 2),
        ("Doc4.pdf", 4),
    ]

    actual_sources = [
        (s.doc, s.page) for s in result.sources
    ]

    assert actual_sources == expected_sources
    assert [s.number for s in result.sources] == [1, 2, 3, 4, 5]
    assert result.llm_calls == 1

    print("PASS: Grouped citations were expanded and renumbered.")
    print("Final answer:", result.text)
    print("Source mapping:", actual_sources)



def test_multiple_search_queries():
    print("\nTEST 9: Multiple focused searches in one batch")

    chunk1 = make_chunk("chunk_1", "Earley parsing uses a chart.")
    chunk2 = make_chunk("chunk_2", "Recursive Descent uses recursive procedures.")
    chunk3 = make_chunk("chunk_3", "Shift-Reduce parsing uses shift and reduce operations.")

    store = FakeStore({
        "Explain parsing": [(chunk1, 0.8, 0.6)],
        "Recursive Descent advantages": [(chunk2, 0.7, 0.5)],
        "Shift-Reduce advantages": [(chunk3, 0.7, 0.5)]
    })

    llm = FakeLLM([
        "SEARCH: Recursive Descent advantages\n"
        "SEARCH: Shift-Reduce advantages",

        "The evidence explains all three concepts. [1][2][3]"
    ])

    result = agentic_answer(
        store,
        "Explain parsing",
        llm,
        max_steps=3
    )

    assert store.search_calls == [
        "Explain parsing",
        "Recursive Descent advantages",
        "Shift-Reduce advantages"
    ]
    assert llm.calls == 2
    assert len(result.answer.sources) == 3
    assert len(result.trace) == 2

    print("PASS: Multiple searches retrieved additional evidence.")
    print("Search calls:", store.search_calls)
    print("Trace:", result.trace)


def test_duplicate_queries_in_batch():
    print("\nTEST 10: Duplicate queries in a batch are skipped")

    chunk1 = make_chunk("chunk_1", "Earley parsing uses a chart.")
    chunk2 = make_chunk("chunk_2", "Earley parsing uses prediction.")

    store = FakeStore({
        "Explain Earley": [(chunk1, 0.8, 0.6)],
        "Earley prediction": [(chunk2, 0.7, 0.5)]
    })

    llm = FakeLLM([
        "SEARCH: Earley prediction\n"
        "SEARCH: Earley prediction\n"
        "SEARCH: Explain Earley",

        "The evidence explains Earley parsing. [1][2]"
    ])

    result = agentic_answer(
        store,
        "Explain Earley",
        llm,
        max_steps=3
    )

    assert store.search_calls == [
        "Explain Earley",
        "Earley prediction"
    ]
    assert llm.calls == 2
    assert len(result.answer.sources) == 2

    print("PASS: Duplicate and previously searched queries were skipped.")
    print("Search calls:", store.search_calls)


def test_duplicate_chunks_across_batch():
    print("\nTEST 11: Duplicate chunks across searches are stored once")

    chunk1 = make_chunk("chunk_1", "Earley parsing uses a chart.")
    chunk2 = make_chunk("chunk_2", "Earley parsing uses prediction.")

    store = FakeStore({
        "Explain Earley": [(chunk1, 0.8, 0.6)],
        "Earley chart": [(chunk1, 0.8, 0.6)],
        "Earley prediction": [(chunk2, 0.7, 0.5)]
    })

    llm = FakeLLM([
        "SEARCH: Earley chart\n"
        "SEARCH: Earley prediction",

        "The evidence explains chart and prediction. [1][2]"
    ])

    result = agentic_answer(
        store,
        "Explain Earley",
        llm,
        max_steps=3
    )

    assert len(store.search_calls) == 3
    assert llm.calls == 2
    assert len(result.answer.sources) == 2

    print("PASS: Duplicate chunks were not added twice.")
    print("Search calls:", store.search_calls)
    print("Unique sources:", len(result.answer.sources))


def test_balanced_evidence_collection():
    print("\nTEST 12: Balanced evidence collection across search queries")

    initial_chunks = [
        make_chunk(f"initial_{i}", f"Initial evidence passage {i}.")
        for i in range(4)
    ]

    query_a_chunks = [
        SimpleNamespace(
            chunk_id=f"a_{i}",
            doc="Query_A.pdf",
            page=i + 1,
            text=f"Evidence from query A, passage {i}."
        )
        for i in range(6)
    ]

    query_b_chunks = [
        SimpleNamespace(
            chunk_id=f"b_{i}",
            doc="Query_B.pdf",
            page=i + 1,
            text=f"Evidence from query B, passage {i}."
        )
        for i in range(6)
    ]

    query_c_chunks = [
        SimpleNamespace(
            chunk_id=f"c_{i}",
            doc="Query_C.pdf",
            page=i + 1,
            text=f"Evidence from query C, passage {i}."
        )
        for i in range(6)
    ]

    store = FakeStore({
        "Explain parsing": [
            (chunk, 0.9, 0.8) for chunk in initial_chunks
        ],
        "Query A": [
            (chunk, 0.8, 0.7) for chunk in query_a_chunks
        ],
        "Query B": [
            (chunk, 0.8, 0.7) for chunk in query_b_chunks
        ],
        "Query C": [
            (chunk, 0.8, 0.7) for chunk in query_c_chunks
        ]
    })

    llm = FakeLLM([
        "SEARCH: Query A\n"
        "SEARCH: Query B\n"
        "SEARCH: Query C",

        "The answer uses all available evidence. "
        "[1][2][3][4][5][6][7][8]"
    ])

    result = agentic_answer(
        store,
        "Explain parsing",
        llm,
        max_steps=3,
        k=4,
        max_evidence=8
    )

    source_docs = {
        source.doc for source in result.answer.sources
    }

    assert store.search_calls == [
        "Explain parsing",
        "Query A",
        "Query B",
        "Query C"
    ]

    assert len(result.answer.sources) == 8

    assert "Query_A.pdf" in source_docs
    assert "Query_B.pdf" in source_docs
    assert "Query_C.pdf" in source_docs

    print("PASS: All three queries contributed evidence.")
    print("Search calls:", store.search_calls)
    print("Evidence documents:", source_docs)
    print("Total evidence:", len(result.answer.sources))


if __name__ == "__main__":
    tests = [
        test_single_pass_low_similarity,
        test_agentic_low_similarity,
        test_empty_retrieval,
        test_new_search_adds_evidence,
        test_repeated_search,
        test_no_new_evidence,
        test_maximum_steps,
        test_grouped_citations_finalize,
        test_multiple_search_queries,
        test_duplicate_queries_in_batch,
        test_duplicate_chunks_across_batch,
        test_balanced_evidence_collection,
    ]

    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as error:
            print("FAILED:", test.__name__)
            print(type(error).__name__ + ":", error)

    print("\n" + "=" * 50)
    print(f"RESULT: {passed}/{len(tests)} tests passed")
    print("=" * 50)
