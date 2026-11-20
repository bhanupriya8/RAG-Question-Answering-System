
"""
Summarize a PDF using NATIVE multimodal understanding (text + embedded images,
charts, diagrams) — via Gemini first, falling back to Claude if Gemini fails
(quota exhausted, server overloaded, etc).

This gives better summaries than plain text extraction because the model can
actually "see" figures, charts, and photos in the PDF alongside the text,
understanding how they relate to each other.

Env vars used:
    GEMINI_API_KEY or GOOGLE_API_KEY   -> for Gemini
    ANTHROPIC_API_KEY                  -> for Claude fallback

Usage:
    python summarize_pdf_native.py "path/to/document.pdf"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPT = (
    "Summarize this document. Give a concise overview, then list the key "
    "points as bullet points. If there are any charts, diagrams, tables, "
    "or images in the document, describe what they show and how they "
    "relate to the surrounding text."
)


def summarize_with_gemini(pdf_bytes: bytes, model: str = "gemini-3.7-flash") -> str:
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.\n"
            'In PowerShell run: $env:GEMINI_API_KEY="YOUR_API_KEY"'
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=[
            {
                "parts": [
                    {"text": PROMPT},
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": pdf_bytes,
                        }
                    },
                ]
            }
        ],
    )
    return response.text


def summarize_with_claude(pdf_bytes: bytes, model: str = "claude-sonnet-5") -> str:
    import base64
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set.\n"
            'In PowerShell run: $env:ANTHROPIC_API_KEY="YOUR_API_KEY"'
        )

    client = anthropic.Anthropic(api_key=api_key)
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    response = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    return response.content[0].text


def summarize_pdf_natively(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    try:
        print("Trying Gemini (gemini-3.7-flash)...")
        return summarize_with_gemini(pdf_bytes)
    except Exception as gemini_error:
        print(f"Gemini failed: {gemini_error}\n")
        print("Falling back to Claude (claude-sonnet-5)...")
        try:
            return summarize_with_claude(pdf_bytes)
        except Exception as claude_error:
            raise RuntimeError(
                f"Both providers failed.\nGemini error: {gemini_error}\n"
                f"Claude error: {claude_error}"
            )


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_pdf_native.py <path_to_pdf>")
        print('Example: python summarize_pdf_native.py "C:/Users/priya/Downloads/report.pdf"')
        return

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: file not found at '{pdf_path}'")
        return

    print(f"\nReading: {pdf_path}")
    print("Sending full PDF (text + images) for native understanding...\n")

    summary = summarize_pdf_natively(pdf_path)

    print("=" * 60)
    print("SUMMARY (native multimodal — text + images understood together):")
    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()