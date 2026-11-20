
"""
Test the ProMSA-Lite agent on ANY image and question — not limited to test_questions.json.

Usage:
    python test_unseen.py "path/to/your_image.jpg" "Your question here?"

Example:
    python test_unseen.py "data/images/Big_Ben.jpg" "What city is this located in?"
    python test_unseen.py "C:/Users/priya/Pictures/my_photo.jpg" "What is shown in this image?"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.agent_loop import ProMSALiteAgent
from agent.llm_client import DEFAULT_CLIENT
from tools.search_tools import SearchTools


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_unseen.py <image_path> <question>")
        print('Example: python test_unseen.py "data/images/Big_Ben.jpg" "What city is this in?"')
        return

    image_path = sys.argv[1]
    question = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"Error: image not found at '{image_path}'")
        return

    print(f"\nImage: {image_path}")
    print(f"Question: {question}\n")
    print("Loading search tools and agent...\n")

    search_tools = SearchTools(top_k=3)
    agent = ProMSALiteAgent(DEFAULT_CLIENT, search_tools, max_steps=5, verbose=True)

    result = agent.run(image_path, question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print(f"\n(Used {result['steps_used']} step(s))")


if __name__ == "__main__":
    main()