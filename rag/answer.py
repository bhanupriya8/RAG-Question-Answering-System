
# rag/answer.py
# import re
# from dataclasses import dataclass, field
# from typing import Callable

# from rag.confidence import assess_confidence, Confidence

# SYSTEM_RULES = """You answer questions using ONLY the numbered sources below.
# Rules:
# - Cite every claim with the source number in square brackets, e.g. [1] or [2][3].
# - If the sources do not contain the answer, say so. Do not use outside knowledge.
# - Be concise."""


# @dataclass
# class Source:
#     number: int
#     doc: str
#     page: int
#     snippet: str


# @dataclass
# class Answer:
#     text: str
#     confidence: Confidence
#     sources: list[Source] = field(default_factory=list)   # only the sources actually cited

#     def format(self) -> str:
#         out = [self.text]
#         if self.confidence.level == "medium":
#             out.insert(0, f"⚠ {self.confidence.message}\n")
#         if self.sources:
#             out.append("\nSources:")
#             for s in self.sources:
#                 out.append(f"  [{s.number}] {s.doc}, p.{s.page}: \"{s.snippet}...\"")
#         return "\n".join(out)


# def build_prompt(query: str, results) -> tuple[str, list[Source]]:
#     sources = []
#     blocks = []
#     for n, (chunk, _fused, _dense) in enumerate(results, start=1):
#         sources.append(Source(n, chunk.doc, chunk.page, chunk.text[:120].replace("\n", " ")))
#         blocks.append(f"[{n}] ({chunk.doc}, page {chunk.page})\n{chunk.text}")
#     prompt = f"{SYSTEM_RULES}\n\nSOURCES:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {query}\nANSWER:"
#     return prompt, sources


# def answer_question(store, query: str, llm: Callable[[str], str] | None, k: int = 5,
#                     docs: list[str] | None = None) -> Answer:
#     results = store.hybrid_search(query, k=k, docs=docs)
#     conf = assess_confidence(results)

#     # LOW confidence: do not call the LLM at all
#     if conf.level == "low":
#         return Answer(conf.message, conf)

#     prompt, sources = build_prompt(query, results)

#     if llm is None:                       # dry run: just show the prompt
#         return Answer(prompt, conf, sources)

#     text = llm(prompt).strip()
#     cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
#     used = [s for s in sources if s.number in cited]
#     return Answer(text, conf, used)




# rag/answer.py
import re
from dataclasses import dataclass, field
from typing import Callable

from rag.confidence import assess_confidence, Confidence

# SYSTEM_RULES = """You answer questions using ONLY the numbered sources below.
# Rules:
# - Cite every claim with the source number in square brackets, e.g. [1] or [2][3].
# - If the sources do not contain the answer, say so. Do not use outside knowledge.
# - Be concise."""

SYSTEM_RULES = """You are an educational question-answering assistant.
Your task is to produce complete, clear, accurate answers using ONLY
the numbered source passages provided below.

CORE RULES:
1. Use only information supported by the supplied sources.
   Never invent facts or introduce unsupported outside knowledge.
2. Answer every part of the user's question.
3. Combine information from multiple relevant passages into one coherent
   explanation. Do not simply copy or list retrieved passages.
4. Explain concepts in simple, clear language suitable for an
   undergraduate student.
5. Explain important technical terms when they are needed to
   understand the answer.
# 6. Cite every important factual claim using source numbers in square
#    brackets, such as [1] or [2][3]. Place citations close to the
#    claims they support.
6. Cite every important factual claim using individual source
   numbers in square brackets.
   For multiple sources, use adjacent references such as [1][2][3].
   Never use comma-separated groups such as [1, 2, 3].
   Never format citation numbers as bold text.
   Place citations close to the claims they support.
7. Use examples from the supplied sources whenever they help explain
   the concept. Do not invent examples and present them as source facts.
8. Do not repeat the same information unnecessarily.

ANSWER STRUCTURE:
For educational or technical questions, organize the answer as appropriate:
- Introduction or definition.
- Detailed explanation of the concept.
- Important components, types, operations or steps.
- Working process or examples, when supported by the sources.
- Advantages, limitations or comparisons when requested or relevant
  and supported by the sources.
- A short conclusion when the question requires a detailed explanation.

Do not force unnecessary headings into simple factual answers.
Use headings, paragraphs, numbered steps or bullet points wherever
they make the explanation easier to understand.

EVIDENCE AND MISSING INFORMATION:
- First determine which parts of the question are supported by the sources.
- If only part of the question is supported, answer the supported parts
  and explicitly identify what information is missing.
- If the sources do not contain enough information to answer,
  clearly say that the answer cannot be determined from the
  uploaded documents.
- Never guess missing algorithm names, values, results or explanations.
- Do not claim that a source supports information that it does not contain.

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
- If requested limitations are not discussed, say so clearly.

COMPARISON:
- When a comparison is explicitly requested, provide a Markdown
  comparison table whenever the evidence supports one.
- Use relevant comparison dimensions such as working process,
  starting point, advantages and limitations.
- Do not fill unsupported comparison cells with guesses.

COMPLETENESS:
Produce an answer that is sufficiently detailed for the question.
Do not sacrifice important explanations merely to make the answer short.
At the same time, avoid unnecessary repetition and irrelevant details."""


@dataclass
class Source:
    number: int
    doc: str
    page: int
    snippet: str
    full_text: str = ""

# @dataclass
# class Answer:
#     text: str
#     confidence: Confidence
#     sources: list[Source] = field(default_factory=list)   # only the sources actually cited

@dataclass
class Answer:
    text: str
    confidence: Confidence
    sources: list[Source] = field(default_factory=list)
    llm_calls: int = 0


    def format(self) -> str:
        out = [self.text]
        if self.confidence.level == "medium":
            out.insert(0, f"⚠ {self.confidence.message}\n")
        if self.sources:
            out.append("\nSources:")
            for s in self.sources:
                out.append(f"  [{s.number}] {s.doc}, p.{s.page}: \"{s.snippet}...\"")
        return "\n".join(out)


def build_prompt(query: str, results, rules: str = SYSTEM_RULES) -> tuple[str, list[Source]]:
    """results: [(Chunk, fused_score, dense_cosine), ...]"""
    sources = []
    blocks = []
    for n, (chunk, _fused, _dense) in enumerate(results, start=1):
        sources.append(Source(n, chunk.doc, chunk.page, chunk.text[:120].replace("\n", " "), chunk.text))
        blocks.append(f"[{n}] ({chunk.doc}, page {chunk.page})\n{chunk.text}")
    prompt = f"{rules}\n\nSOURCES:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {query}\nANSWER:"
    return prompt, sources


# def finalize(text: str, sources: list[Source], conf: Confidence) -> Answer:
#     """Keep only cited sources and renumber them 1, 2, 3... in order of first use."""
#     valid = {s.number: s for s in sources}
#     order: list[int] = []
#     for m in re.findall(r"\[(\d+)\]", text):
#         n = int(m)
#         if n in valid and n not in order:
#             order.append(n)

#     remap = {old: new for new, old in enumerate(order, start=1)}

#     def swap(m):
#         n = int(m.group(1))
#         return f"[{remap[n]}]" if n in remap else m.group(0)

#     new_text = re.sub(r"\[(\d+)\]", swap, text)
#     used = [Source(remap[o], valid[o].doc, valid[o].page, valid[o].snippet) for o in order]
#     return Answer(new_text, conf, used)


# def finalize(
#     text: str,
#     sources: list[Source],
#     conf: Confidence,
#     llm_calls: int = 0
# ) -> Answer:
#     """Keep only cited sources and renumber them 1, 2, 3... in order of first use."""
#     valid = {s.number: s for s in sources}
#     order: list[int] = []

#     for m in re.findall(r"\[(\d+)\]", text):
#         n = int(m)
#         if n in valid and n not in order:
#             order.append(n)

#     remap = {old: new for new, old in enumerate(order, start=1)}

#     def swap(m):
#         n = int(m.group(1))
#         return f"[{remap[n]}]" if n in remap else m.group(0)

#     new_text = re.sub(r"\[(\d+)\]", swap, text)

#     used = [
#         Source(remap[o], valid[o].doc, valid[o].page, valid[o].snippet)
#         for o in order
#     ]

#     return Answer(new_text, conf, used, llm_calls)



def finalize(
    text: str,
    sources: list[Source],
    conf: Confidence,
    llm_calls: int = 0
) -> Answer:
    """
    Keep only valid cited sources and renumber them in order of first use.

    Supports:
    [1]
    [1][2]
    [1, 2, 3]

    Every citation in the final answer maps to the Sources panel.
    """
    valid = {s.number: s for s in sources}

    # Expand grouped citations and remove invalid source numbers.
    citation_pattern = re.compile(
        r"\[((?:\s*\d+\s*)(?:,\s*\d+\s*)*)\]"
    )

    def expand_citation(match):
        numbers = [int(n) for n in re.findall(r"\d+", match.group(1))]

        # Preserve order while removing duplicates and invalid numbers.
        valid_numbers = list(dict.fromkeys(
            n for n in numbers if n in valid
        ))

        return "".join(f"[{n}]" for n in valid_numbers)

    expanded_text = citation_pattern.sub(expand_citation, text)

    # Determine source order from the citations that remain.
    order = []
    for match in re.finditer(r"\[(\d+)\]", expanded_text):
        number = int(match.group(1))
        if number in valid and number not in order:
            order.append(number)

    # Renumber sources from 1 in order of first citation.
    remap = {
        old: new
        for new, old in enumerate(order, start=1)
    }

    def replace_citation(match):
        number = int(match.group(1))
        return f"[{remap[number]}]" if number in remap else ""

    new_text = re.sub(r"\[(\d+)\]", replace_citation, expanded_text)

    # Avoid unwanted spaces before punctuation after invalid citations
    # have been removed.
    new_text = re.sub(r"[ \t]+([,.;:!?])", r"\1", new_text)

    used = [
        Source(
            remap[number],
            valid[number].doc,
            valid[number].page,
            valid[number].snippet,
            valid[number].full_text
        )
        
        for number in order
    ]

    return Answer(new_text, conf, used, llm_calls)


# def answer_question(store, query: str, llm: Callable[[str], str] | None, k: int = 5,
#                     docs: list[str] | None = None) -> Answer:
#     results = store.hybrid_search(query, k=k, docs=docs)
#     conf = assess_confidence(results)

#     # LOW confidence: do not call the LLM at all
#     if conf.level == "low":
#         return Answer(conf.message, conf)

#     prompt, sources = build_prompt(query, results)

#     if llm is None:                       # dry run: just show the prompt
#         return Answer(prompt, conf, sources)

#     text = llm(prompt).strip()
#     return finalize(text, sources, conf)

def answer_question(store, query: str, llm: Callable[[str], str] | None, k: int = 5,
                    docs: list[str] | None = None) -> Answer:
    results = store.hybrid_search(query, k=k, docs=docs)
    conf = assess_confidence(results)

    # Stop only when no passages were retrieved.
    # Low similarity alone should not prevent evidence examination.
    if not results:
        return Answer(conf.message, conf)

    prompt, sources = build_prompt(query, results)

    if llm is None:
        return Answer(prompt, conf, sources)

    # text = llm(prompt).strip()
    # return finalize(text, sources, conf)
 
    text = llm(prompt).strip()
    return finalize(text, sources, conf, llm_calls=1)