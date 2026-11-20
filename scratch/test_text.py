
from agent.llm_client import DEFAULT_CLIENT

print(type(DEFAULT_CLIENT).__name__, DEFAULT_CLIENT.model)
print(DEFAULT_CLIENT.call(
    "Answer briefly.",
    [{"role": "user", "content": "What is the capital of France? One word."}],
))