"""
A small wrapper so the agent loop doesn't care which LLM provider you use.
Once you get an API key, uncomment the matching class below and set it as
DEFAULT_CLIENT at the bottom. All three take an image path (or None) + the
running conversation text, and return the model's raw text reply.

Every provider below supports vision (image input) - required since our
agent needs to look at the query photo.
"""

"""
LLM client wrapper for ProMSA-Lite.

This version uses Google's current google-genai SDK for Gemini.
The API key is read from the GEMINI_API_KEY environment variable.
"""

from dotenv import load_dotenv
load_dotenv()

import base64
import os
import time
from collections import deque

from PIL import Image


def _load_and_resize_image(image_path, max_dim=768):
    """Load an image, convert it to a proper RGB JPEG regardless of the source
    format (AVIF, WEBP, PNG, etc.), and resize it so its longest side is at
    most max_dim, keeping aspect ratio. Ensures downstream APIs always
    receive genuine, universally-supported JPEG data."""
    image = Image.open(image_path)

    # Convert to RGB — handles AVIF/PNG-with-alpha/CMYK/etc. that can otherwise
    # produce invalid or unsupported bytes when saved as JPEG.
    if image.mode != "RGB":
        image = image.convert("RGB")

    image.thumbnail((max_dim, max_dim))
    return image


def encode_image_b64(image_path: str) -> str:
    """Load an image, ensure it's genuine JPEG (converting from AVIF/WEBP/PNG/etc.
    if needed), and return it as base64."""
    import io

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---------------------------------------------------------------------
# Option 1: Anthropic (Claude)
# ---------------------------------------------------------------------

class ClaudeClient:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def call(
        self,
        system_prompt: str,
        conversation: list,
        image_path: str = None
    ) -> str:

        messages = list(conversation)

        if image_path and messages:

            messages[0] = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encode_image_b64(image_path),
                        },
                    },
                    {
                        "type": "text",
                        "text": messages[0]["content"],
                    },
                ],
            }

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )

        return resp.content[0].text


# ---------------------------------------------------------------------
# Option 2: OpenAI
# ---------------------------------------------------------------------

class OpenAIClient:
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def call(
        self,
        system_prompt: str,
        conversation: list,
        image_path: str = None
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ] + list(conversation)

        if image_path and len(messages) > 1:

            messages[1] = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": messages[1]["content"],
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                + encode_image_b64(image_path)
                            )
                        },
                    },
                ],
            }

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
        )

        return resp.choices[0].message.content


# ---------------------------------------------------------------------
# Option 3: Google Gemini
#
# Install:
#     python -m pip install -U google-genai
#
# API key:
#     $env:GEMINI_API_KEY="YOUR_KEY"
# ---------------------------------------------------------------------

class GeminiClient:

    # Class-level so throttling applies across ALL instances/calls
    _call_times = deque()
    _max_calls_per_window = 4       # stay under the 5 RPM free-tier cap
    _window_seconds = 60.0

    def __init__(
        self,
        model: str = "gemini-3.6-flash"
    ):

        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.\n\n"
                "In PowerShell run:\n"
                '$env:GEMINI_API_KEY="YOUR_API_KEY"'
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = model

    # -------------------------------------------------------------
    # Sliding-window throttle
    # -------------------------------------------------------------

    def _throttle(self):
        now = time.time()

        # Drop timestamps older than the window
        while GeminiClient._call_times and now - GeminiClient._call_times[0] > GeminiClient._window_seconds:
            GeminiClient._call_times.popleft()

        if len(GeminiClient._call_times) >= GeminiClient._max_calls_per_window:
            wait = GeminiClient._window_seconds - (now - GeminiClient._call_times[0]) + 1.0
            print(f"[throttle] {len(GeminiClient._call_times)} calls in last "
                  f"{GeminiClient._window_seconds:.0f}s, sleeping {wait:.1f}s...")
            time.sleep(wait)
            now = time.time()
            while GeminiClient._call_times and now - GeminiClient._call_times[0] > GeminiClient._window_seconds:
                GeminiClient._call_times.popleft()

        GeminiClient._call_times.append(time.time())

    # -------------------------------------------------------------
    # Retry-aware call wrapper (handles both 429 rate limits and
    # 503 "model overloaded" server errors)
    # -------------------------------------------------------------

    def _generate_with_retry(self, contents, max_retries: int = 3):
        from google.genai import errors as genai_errors

        for attempt in range(1, max_retries + 1):
            self._throttle()
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    # config={"http_options": {"timeout": 30000}},  # 30 second timeout in ms
                    config={"http_options": {"timeout": 120000}},  # 120 second timeout in ms
                )       
                
            except genai_errors.ClientError as e:
                if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
                    retry_delay = self._extract_retry_delay(e)
                    if retry_delay > 300:
                        raise RuntimeError(
                            f"Gemini quota exhausted (server asks to wait {retry_delay / 3600:.1f} hours).")
                    print(f"[429] Rate limited. Retrying in {retry_delay:.1f}s "
                          f"(attempt {attempt}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue
                raise
            except genai_errors.ServerError as e:
                print(f"      actual error: {str(e)[:200]}")
                # 503 UNAVAILABLE / high demand on Google's side — transient, worth retrying
                wait = min(10 * attempt, 60)  # simple backoff: 10s, 20s, 30s... capped at 60s
                print(f"[503] Model overloaded. Retrying in {wait}s "
                      f"(attempt {attempt}/{max_retries})...")
                time.sleep(wait)
                continue
                
            

        raise RuntimeError(
            f"Gemini API call failed after {max_retries} retries (rate limit or server unavailable)."
        )

    @staticmethod
    def _extract_retry_delay(error, default: float = 20.0) -> float:
        """Pulls the server-suggested retryDelay (e.g. '34s') out of the error, if present."""
        try:
            details = error.details.get("error", {}).get("details", [])
            for d in details:
                if d.get("@type", "").endswith("RetryInfo"):
                    delay_str = d.get("retryDelay", "")
                    if delay_str.endswith("s"):
                        return float(delay_str[:-1]) + 3.0  # buffer
        except Exception:
            pass
        return default

    # -------------------------------------------------------------
    # Public call method
    # -------------------------------------------------------------

    def call(
        self,
        system_prompt: str,
        conversation: list,
        image_path: str = None
    ) -> str:

        history_text = "\n\n".join(
            f"[{m['role'].upper()}]: {m['content']}"
            for m in conversation
        )

        prompt = (
            system_prompt
            + "\n\n"
            + history_text
        )

        if not image_path:
            response = self._generate_with_retry(contents=prompt)
            return response.text

        image = _load_and_resize_image(image_path)

        response = self._generate_with_retry(contents=[prompt, image])
        return response.text



# ---------------------------------------------------------------------
# Option 4: Groq
#
# Install:
#     pip install groq
#
# API key:
#     $env:GROQ_API_KEY="YOUR_KEY"
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Option 4: Groq
#
# Install:
#     pip install groq
#
# API key:
#     $env:GROQ_API_KEY="YOUR_KEY"
# ---------------------------------------------------------------------

class GroqClient:

    # Class-level so throttling applies across ALL instances/calls
    _call_times = deque()
    _max_calls_per_window = 6       # stay safely under Groq's 30 RPM cap
    _window_seconds = 60.0

    def __init__(self, model: str = "qwen/qwen3.6-27b"):
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set.\n\n"
                "In PowerShell run:\n"
                '$env:GROQ_API_KEY="YOUR_API_KEY"'
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def _throttle(self):
        now = time.time()

        while GroqClient._call_times and now - GroqClient._call_times[0] > GroqClient._window_seconds:
            GroqClient._call_times.popleft()

        if len(GroqClient._call_times) >= GroqClient._max_calls_per_window:
            wait = GroqClient._window_seconds - (now - GroqClient._call_times[0]) + 2.0
            print(f"[throttle] {len(GroqClient._call_times)} calls in last "
                  f"{GroqClient._window_seconds:.0f}s, sleeping {wait:.1f}s...")
            time.sleep(wait)
            now = time.time()
            while GroqClient._call_times and now - GroqClient._call_times[0] > GroqClient._window_seconds:
                GroqClient._call_times.popleft()

        GroqClient._call_times.append(time.time())

    def _generate_with_retry(self, messages, max_retries: int = 6):
        import groq

        for attempt in range(1, max_retries + 1):
            self._throttle()
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=1024,  # reduced from 2048 to use fewer tokens/call
                )
            except groq.InternalServerError as e:
                wait = min(10 * attempt, 60)
                print(f"[503] Groq overloaded. Retrying in {wait}s "
                      f"(attempt {attempt}/{max_retries})...")
                time.sleep(wait)
            except groq.RateLimitError as e:
                wait = min(20 * attempt, 60)
                print(f"[429] Groq rate limited. Retrying in {wait}s "
                      f"(attempt {attempt}/{max_retries})...")
                time.sleep(wait)

        raise RuntimeError(f"Groq API call failed after {max_retries} retries.")

    def call(
        self,
        system_prompt: str,
        conversation: list,
        image_path: str = None
    ) -> str:

        messages = [{"role": "system", "content": system_prompt}]

        if image_path and conversation:
            user_text = conversation[0]["content"]
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_b64(image_path)}"
                        },
                    },
                ],
            })
            messages.extend(conversation[1:])
        else:
            messages.extend(conversation)

        resp = self._generate_with_retry(messages)
        return resp.choices[0].message.content
    
    
# ---------------------------------------------------------------------
# Option 5: OpenRouter (free-tier vision models, OpenAI-compatible API)
#
# Install:
#     pip install openai   (OpenRouter uses the OpenAI SDK, just a different base_url)
#
# Sign up (no card required for free models):
#     https://openrouter.ai/keys
#
# API key:
#     $env:OPENROUTER_API_KEY="YOUR_KEY"
#
# Free vision-capable model IDs (verify current list at
# https://openrouter.ai/models?fmt=cards&supported_parameters=tools&max_price=0
# since free models rotate without notice):
#     "nvidia/nemotron-nano-12b-v2-vl:free"   <- vision-capable, good default
#     "openrouter/free"                        <- auto-router, picks any free model that fits
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Option 5: OpenRouter (free-tier vision models, OpenAI-compatible API)
#
# Install:
#     pip install openai   (OpenRouter uses the OpenAI SDK, just a different base_url)
#
# Sign up (no card required for free models):
#     https://openrouter.ai/keys
#
# API key:
#     $env:OPENROUTER_API_KEY="YOUR_KEY"
#
# Free vision-capable model IDs change over time — verify the current list at
# https://openrouter.ai/models?max_price=0 before relying on a hardcoded name.
# As of writing, known-working free vision models include:
#     "minimax/minimax-m3:free"
#     "google/gemma-4-31b-it:free"
#     "openrouter/free"   <- auto-router, picks any currently-working free model
# ---------------------------------------------------------------------

class OpenRouterClient:
    def __init__(self, model: str = "minimax/minimax-m3:free"):
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set.\n\n"
                "In PowerShell run:\n"
                '$env:OPENROUTER_API_KEY="YOUR_API_KEY"'
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def _generate_with_retry(self, messages, max_retries: int = 6):
        import openai

        for attempt in range(1, max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4096,
                )
                
            except openai.NotFoundError as e:
                # Model ID no longer exists / was pulled from OpenRouter's free lineup.
                # Not worth retrying — surface immediately so the model name can be swapped.
                raise RuntimeError(
                    f"OpenRouter model '{self.model}' was not found (it may have been "
                    f"deprecated or renamed). Check current free models at "
                    f"https://openrouter.ai/models?max_price=0 and update the model name.\n"
                    f"Original error: {e}"
                )
            except openai.RateLimitError as e:
                wait = min(15 * attempt, 60)
                print(f"[429] OpenRouter rate limited. Retrying in {wait}s "
                      f"(attempt {attempt}/{max_retries})...")
                time.sleep(wait)
            except (openai.InternalServerError, openai.APIConnectionError) as e:
                wait = min(10 * attempt, 60)
                print(f"[error] OpenRouter issue: {type(e).__name__}. Retrying in {wait}s "
                      f"(attempt {attempt}/{max_retries})...")
                time.sleep(wait)

        raise RuntimeError(f"OpenRouter API call failed after {max_retries} retries.")

    def call(
        self,
        system_prompt: str,
        conversation: list,
        image_path: str = None
    ) -> str:

        messages = [{"role": "system", "content": system_prompt}]

        if image_path and conversation:
            user_text = conversation[0]["content"]
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_b64(image_path)}"
                        },
                    },
                ],
            })
            messages.extend(conversation[1:])
        else:
            messages.extend(conversation)

        resp = self._generate_with_retry(messages)

        if not resp.choices or resp.choices[0].message.content is None:
            raise RuntimeError(
                "OpenRouter returned an empty/unusable response. This can happen if the "
                "selected free model doesn't actually support vision, or hit a content "
                f"filter. Raw response: {resp}\n"
                "Try a different model like 'google/gemma-4-31b-it:free' or check "
                "https://openrouter.ai/models?max_price=0 for current free vision models."
            )

        return resp.choices[0].message.content
  
# ---------------------------------------------------------------------
# Default provider
# ---------------------------------------------------------------------



# DEFAULT_CLIENT = GeminiClient(model="gemini-3.6-flash")
# DEFAULT_CLIENT = GroqClient(model="qwen/qwen3.6-27b")


# DEFAULT_CLIENT = OpenRouterClient(model="openrouter/free")
# DEFAULT_CLIENT = OpenRouterClient(model="minimax/minimax-m3:free")

# DEFAULT_CLIENT = OpenRouterClient(model="nvidia/nemotron-3.5-super-vl:free")
# DEFAULT_CLIENT = OpenAIClient(model="gpt-4o-mini")

DEFAULT_CLIENT = GeminiClient(model="gemini-3.8-flash")

# DEFAULT_CLIENT = GeminiClient(model="gemini-3.7-flash")
