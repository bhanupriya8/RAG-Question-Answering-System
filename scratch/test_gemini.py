
# test_gemini.py
from rag.gemini_llm import make_gemini_llm

llm = make_gemini_llm()
print(llm.model_name)
print(llm("Reply with exactly one word: hello"))