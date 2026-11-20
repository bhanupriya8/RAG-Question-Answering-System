
from agent.agent_loop import ProMSALiteAgent
from agent.llm_client import DEFAULT_CLIENT
from tools.search_tools import SearchTools

# Load search tools
tools = SearchTools()

# Create ProMSA-Lite agent
agent = ProMSALiteAgent(DEFAULT_CLIENT, tools, max_steps=5)

# Image and question
image_path = "data/images/Eiffel_Tower.jpg"
question = "Who designed this structure?"

# Run agent
agent_result = agent.run(image_path, question)

# Print result
print("\n===== RESULT =====")
print("Answer:", agent_result["answer"])
print("Steps used:", agent_result["steps_used"])