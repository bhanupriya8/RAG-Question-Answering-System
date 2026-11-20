
from agent.llm_client import DEFAULT_CLIENT

print(type(DEFAULT_CLIENT).__name__, DEFAULT_CLIENT.model)
print(DEFAULT_CLIENT.call(
    "Answer briefly.",
    [{"role": "user", "content": "Which landmark is in this photo? One short answer."}],
    image_path="data/query_images/Petra.jpg",
))