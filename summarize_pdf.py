
"""
Summarize a PDF using the existing LLM client (Gemini/Groq/OpenAI/OpenRouter —
whichever is set as DEFAULT_CLIENT in agent/llm_client.py).

Usage:
    python summarize_pdf.py "path/to/document.pdf"

Example:
    python summarize_pdf.py "C:/Users/priya/Downloads/report.pdf"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.llm_client import DEFAULT_CLIENT


def extract_pdf_text(pdf_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_pdf.py <path_to_pdf>")
        print('Example: python summarize_pdf.py "C:/Users/priya/Downloads/report.pdf"')
        return

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: file not found at '{pdf_path}'")
        return

    print(f"\nReading: {pdf_path}")
    pdf_text = extract_pdf_text(pdf_path)

    if not pdf_text:
        print(
            "\nWarning: no text could be extracted from this PDF.\n"
            "This usually means it's a scanned/image-based PDF rather than a "
            "text-based one. Extracting text from scanned PDFs requires OCR, "
            "which this script doesn't currently support."
        )
        return

    print(f"Extracted {len(pdf_text)} characters of text.\n")
    print("Summarizing...\n")

    # Guard against extremely long documents overwhelming a single prompt.
    # Most providers have a context limit; trim if needed.
    max_chars = 30000
    if len(pdf_text) > max_chars:
        print(f"(Document is long — using the first {max_chars} characters for the summary.)\n")
        pdf_text = pdf_text[:max_chars]

    summary = DEFAULT_CLIENT.call(
        system_prompt="You are a helpful assistant that summarizes documents clearly and concisely.",
        conversation=[
            {
                "role": "user",
                "content": (
                    "Summarize the following document. Give a concise overview, "
                    "then list the key points as bullet points.\n\n"
                    f"{pdf_text}"
                ),
            }
        ],
    )

    print("=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()