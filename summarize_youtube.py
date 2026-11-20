
"""
Summarize a YouTube video, or answer a question about it, using its transcript
(captions) + your existing LLM client. No video download needed.

Install:
    pip install youtube-transcript-api

Usage:
    python summarize_youtube.py "<youtube_url_or_video_id>"
    python summarize_youtube.py "<youtube_url_or_video_id>" "your question here"

Examples:
    python summarize_youtube.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    python summarize_youtube.py "dQw4w9WgXcQ" "What are the main points made in this video?"

If no question is given, it just produces a general summary.
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.llm_client import DEFAULT_CLIENT


def extract_video_id(url_or_id: str) -> str:
    """Accepts a full YouTube URL or a bare video ID and returns the video ID."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",   # watch?v=ID or youtu.be/ID
        r"^([0-9A-Za-z_-]{11})$",            # bare ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract a video ID from: {url_or_id}")


def get_transcript_text(video_id: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)

    full_text = " ".join(snippet.text for snippet in transcript)
    return full_text.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_youtube.py <youtube_url_or_id> [question]")
        return

    url_or_id = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        video_id = extract_video_id(url_or_id)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"\nVideo ID: {video_id}")
    print("Fetching transcript...\n")

    try:
        transcript_text = get_transcript_text(video_id)
    except Exception as e:
        print(
            f"Error fetching transcript: {e}\n\n"
            "This usually means the video has no captions/transcript available, "
            "or captions are disabled by the uploader."
        )
        return

    if not transcript_text:
        print("No transcript text was found for this video.")
        return

    print(f"Transcript length: {len(transcript_text)} characters\n")

    # Guard against very long transcripts overwhelming a single prompt.
    max_chars = 30000
    if len(transcript_text) > max_chars:
        print(f"(Transcript is long — using the first {max_chars} characters.)\n")
        transcript_text = transcript_text[:max_chars]

    if question:
        print(f"Question: {question}\n")
        prompt = (
            f"Here is the transcript of a YouTube video:\n\n{transcript_text}\n\n"
            f"Based on this transcript, answer the following question as accurately "
            f"as you can: {question}"
        )
        label = "ANSWER"
    else:
        print("Summarizing...\n")
        prompt = (
            f"Here is the transcript of a YouTube video:\n\n{transcript_text}\n\n"
            f"Summarize this video. Give a concise overview, then list the key "
            f"points as bullet points."
        )
        label = "SUMMARY"

    result = DEFAULT_CLIENT.call(
        system_prompt="You are a helpful assistant that understands video transcripts and answers clearly.",
        conversation=[{"role": "user", "content": prompt}],
    )

    print("=" * 60)
    print(f"{label}:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()