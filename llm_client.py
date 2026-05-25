"""
llm_client.py
=============
LLM provider abstraction layer.

This module lets the system run with any of:
  - Anthropic Claude API  (--llm api)    — paid, highest accuracy
  - Groq API              (--llm groq)   — FREE cloud API, no download, near-GPT-4 quality
  - Local Ollama          (--llm ollama) — free, runs on your machine

DESIGN PATTERN — Duck Typing:
  All clients expose the same interface:
    client.messages.parse(system=..., messages=..., output_format=PydanticModel)
    client.messages.stream(system=..., messages=...)
    client.get_model_name()

  The agents and orchestrator don't need to know which backend is active —
  they just call the same methods and get the same data structures back.

GROQ API (best free option):
  - Sign up at https://console.groq.com/ (completely free)
  - Add GROQ_API_KEY to your .env file
  - Uses Llama 3.3-70B: near GPT-4 quality, blazing fast
  - Free tier: 14,400 requests/day — plenty for stock analysis
  - No installation, no local GPU, no downloads

HOW NON-ANTHROPIC STRUCTURED OUTPUT WORKS:
  Anthropic has native structured output (Pydantic via output_format=).
  Groq/Ollama don't. So we:
    1. Add the JSON schema to the system prompt as instructions
    2. Use JSON mode (Groq: response_format=json_object, Ollama: format=json)
    3. Parse the JSON string → Pydantic model ourselves

INSTALL OLLAMA (if using --llm ollama):
  1. Download: https://ollama.ai
  2. Pull a model: ollama pull llama3.2
  3. Run: ollama serve  (or it runs as a background service after install)
"""

import json
import os
import requests
import threading
from typing import Type

# ── Live progress callback (thread-local) ──────────────────────────────────
# The Streamlit app sets a callback so retry/error events surface in the UI
# immediately instead of being buried in stdout. Worker threads inherit it
# via ThreadPoolExecutor(initializer=...) — see orchestrator.run_analysis.
_progress_local = threading.local()


def set_progress_callback(cb) -> None:
    """Register a callback(level: str, message: str) for this thread."""
    _progress_local.cb = cb


def _emit_progress(level: str, message: str) -> None:
    """Fire the registered callback if any; always print to stdout too."""
    cb = getattr(_progress_local, "cb", None)
    if cb is not None:
        try:
            cb(level, message)
        except Exception:
            pass
    print(f"  [{level}] {message}")


# ── Shared helpers ─────────────────────────────────────────────────────────


class _ParseResult:
    """
    Wraps the output of parse() so both clients return
    result.parsed_output — same interface as Anthropic SDK.
    """
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


# ── Ollama helper utilities ────────────────────────────────────────────────


def _python_type_example(annotation) -> object:
    """Return a plausible JSON-serialisable example value for a Python type annotation."""
    import typing
    origin = getattr(annotation, "__origin__", None)
    args   = getattr(annotation, "__args__", ())

    # Optional[X] → X
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _python_type_example(non_none[0])
        return None

    if origin is list:
        inner = args[0] if args else float
        return [_python_type_example(inner)]

    if annotation is float or annotation is int:
        return 0.0
    if annotation is str:
        return "string"
    if annotation is bool:
        return False

    # Nested Pydantic model
    try:
        if hasattr(annotation, "model_fields"):
            return _build_example_json(annotation)
    except Exception:
        pass

    return None


def _build_example_json(model_class) -> dict:
    """Build a flat example dict from a Pydantic model's field annotations.

    This is much simpler than model_json_schema() and avoids the $defs
    section that confuses smaller local models.  Where a field has metadata
    (ge/le/description) we embed a comment-style hint so the model knows
    the valid range.
    """
    example = {}
    for field_name, field_info in model_class.model_fields.items():
        annotation = field_info.annotation
        value      = _python_type_example(annotation)

        # For numeric fields with range constraints, replace 0 with midpoint
        # and add a description hint so the model stays in range.
        meta = field_info.metadata  # list of annotated constraints
        ge = le = None
        for m in meta:
            if hasattr(m, "ge"):
                ge = m.ge
            if hasattr(m, "le"):
                le = m.le
        if ge is not None and le is not None and isinstance(value, (int, float)):
            value = (ge + le) / 2

        example[field_name] = value
    return example


def _clean_ollama_response(data: dict, model_class) -> dict:
    """Remove schema-metadata keys that smaller models echo back verbatim.

    Keeps only keys that are actual fields of the Pydantic model, then
    filters None values out of list[float] / list[int] fields.
    """
    import typing
    valid_fields = set(model_class.model_fields.keys())

    # Drop top-level keys that aren't model fields (e.g. "$defs", "$schema")
    cleaned = {k: v for k, v in data.items() if k in valid_fields}

    # For list fields, filter out None elements so Pydantic doesn't choke
    for field_name, field_info in model_class.model_fields.items():
        if field_name not in cleaned:
            continue
        annotation = field_info.annotation
        origin = getattr(annotation, "__origin__", None)
        args   = getattr(annotation, "__args__", ())

        # Unwrap Optional[list[...]]
        if origin is typing.Union:
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                annotation = non_none[0]
                origin = getattr(annotation, "__origin__", None)
                args   = getattr(annotation, "__args__", ())

        if origin is list and isinstance(cleaned[field_name], list):
            cleaned[field_name] = [x for x in cleaned[field_name] if x is not None]
            continue

        # Coerce float → int when the field expects int (model often returns 234.1)
        eff_annotation = annotation
        eff_origin = origin
        eff_args   = args
        # Unwrap Optional[int]
        if eff_origin is typing.Union:
            non_none = [a for a in eff_args if a is not type(None)]
            if non_none:
                eff_annotation = non_none[0]
        if eff_annotation is int:
            val = cleaned.get(field_name)
            if isinstance(val, float):
                cleaned[field_name] = int(round(val))

        # Clamp numeric scalars to their ge/le bounds so validation passes
        # even when the model returns e.g. 60 for a 0-10 score field.
        meta = field_info.metadata or []
        ge = le = None
        for m in meta:
            if hasattr(m, "ge"):
                ge = m.ge
            if hasattr(m, "le"):
                le = m.le
        val = cleaned.get(field_name)
        if isinstance(val, (int, float)):
            if ge is not None:
                val = max(val, ge)
            if le is not None:
                val = min(val, le)
            cleaned[field_name] = val

    return cleaned


# ── Ollama implementation ───────────────────────────────────────────────────


class _OllamaTextDelta:
    def __init__(self, text: str):
        self.type = "text_delta"
        self.text = text


class _OllamaStreamEvent:
    """
    Mimics Anthropic's content_block_delta event so the orchestrator's
    existing streaming loop works without modification.

    The orchestrator checks:
        event.type == "content_block_delta"
        event.delta.type == "text_delta"
        event.delta.text  → the text chunk
    """
    def __init__(self, text: str):
        self.type = "content_block_delta"
        self.delta = _OllamaTextDelta(text)


class _OllamaStream:
    """
    Context manager that streams from Ollama's /api/chat endpoint
    and yields events compatible with the Anthropic streaming format.
    """
    def __init__(self, model: str, base_url: str, system: str, messages: list, max_tokens: int):
        self.model = model
        self.base_url = base_url
        self.system = system
        self.messages = messages
        self.max_tokens = max_tokens
        self._full_text = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        if self.messages:
            msgs.extend(self.messages)

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": msgs,
                    "stream": True,
                    "options": {"num_predict": self.max_tokens},
                },
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Lost connection to Ollama during streaming. "
                "Make sure Ollama is running: ollama serve"
            )

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    self._full_text += chunk
                    yield _OllamaStreamEvent(chunk)
            except json.JSONDecodeError:
                continue


class _OllamaMessages:
    """
    Mimics the anthropic.Anthropic().messages interface for Ollama.
    Provides .parse() and .stream() that the agents and orchestrator call.
    """

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def parse(self, model=None, max_tokens=4096, system=None,
              messages=None, output_format=None, **kwargs):
        """
        Call Ollama and parse the JSON response into a Pydantic model.

        The model= kwarg from agents contains "claude-opus-4-6" — we ignore it
        and always use the Ollama model configured at startup.

        Extra kwargs (thinking, output_config) are silently ignored —
        they're Anthropic-specific and don't apply to Ollama.
        """
        msgs = []

        # Build system prompt, injecting a flat example instead of the raw schema.
        # Smaller models (e.g. llama3.1:8b) echo the $defs structure back when
        # given the full model_json_schema(), producing unusable output.
        system_content = system or ""
        if output_format:
            example = _build_example_json(output_format)
            example_str = json.dumps(example, indent=2)
            system_content += (
                f"\n\n===CRITICAL OUTPUT REQUIREMENT===\n"
                f"You MUST respond with ONLY valid JSON. No markdown code blocks, "
                f"no explanations, no text before or after. "
                f"Your entire response must be a single JSON object with EXACTLY these fields "
                f"(fill in real values — do NOT copy this example verbatim):\n"
                f"{example_str}"
            )

        if system_content:
            msgs.append({"role": "system", "content": system_content})
        if messages:
            msgs.extend(messages)

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": msgs,
                    "stream": False,
                    "format": "json",          # forces JSON output mode
                    "options": {"num_predict": max_tokens},
                },
                timeout=180,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama. Make sure it is running: ollama serve"
            )

        content = resp.json()["message"]["content"]

        if output_format:
            try:
                # Strip markdown code fences if model added them anyway
                text = content.strip()
                if text.startswith("```"):
                    text = text.split("```", 2)[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                if text.endswith("```"):
                    text = text[:-3].strip()

                data = json.loads(text)
                # Smaller Ollama models sometimes echo schema $defs as top-level
                # keys. Strip any non-field metadata before validating.
                data = _clean_ollama_response(data, output_format)
                parsed = output_format.model_validate(data)
                return _ParseResult(parsed)
            except Exception as e:
                raise ValueError(
                    f"Ollama response could not be parsed as {output_format.__name__}: {e}\n"
                    f"Raw response (first 300 chars): {content[:300]}"
                )

        return _ParseResult(content)

    def stream(self, model=None, max_tokens=8192, system=None,
               messages=None, **kwargs):
        """
        Return an _OllamaStream context manager.
        Extra kwargs (thinking, output_config) are silently ignored.
        """
        return _OllamaStream(
            model=self.model,
            base_url=self.base_url,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )


# ── Groq implementation ────────────────────────────────────────────────────


class _GroqMessages:
    """
    Mimics the anthropic.Anthropic().messages interface for Groq API.
    Provides .parse() and .stream() that the agents and orchestrator call.

    Groq is OpenAI-API-compatible. We use json_object response_format
    (same pattern as Ollama) — inject schema in system prompt, parse manually.

    ERROR HANDLING:
      Groq raises typed exceptions (groq.RateLimitError, groq.AuthenticationError,
      groq.APIError, etc.). We catch them explicitly and either retry (rate limit)
      or re-raise with a clear user-facing message (auth, server errors).

    TOKEN CAP:
      Groq free tier: 6,000 output tokens/minute across all calls.
      3 parallel agents × large max_tokens would blow past that instantly.
      We cap at GROQ_MAX_OUTPUT_TOKENS (2048) so all three agents fit
      within the per-minute budget. The responses are already short by design.
    """

    GROQ_MAX_OUTPUT_TOKENS = 2048   # hard cap for Groq free tier
    GROQ_MAX_RETRIES = 3
    GROQ_RETRY_WAIT = 20            # seconds between rate-limit retries
    # Stock analysis needs structured, consistent numeric output (scores, verdicts).
    # Groq defaults to temperature=1.0 which produces wildly different values per run.
    GROQ_TEMPERATURE = 0.1
    GROQ_SEED = 42                  # reproducibility — same input gives same output

    def __init__(self, client, model: str):
        self._client = client
        self.model = model

    @staticmethod
    def _wrap_groq_error(exc: Exception) -> Exception:
        """Convert a Groq SDK exception into a clear, user-facing message."""
        name = type(exc).__name__
        msg  = str(exc)
        if "AuthenticationError" in name or "401" in msg:
            return ValueError(
                "Groq rejected the API key (401 Unauthorized).\n"
                "The key was found in .env but Groq says it's invalid or revoked.\n"
                "→ Double-check the key at https://console.groq.com/keys\n"
                "→ Make sure you copied the full key (starts with 'gsk_')\n"
                "→ Restart Streamlit after editing .env so the new key is loaded"
            )
        if "RateLimitError" in name or "429" in msg:
            return ConnectionError(
                "Groq free-tier rate limit reached (6,000 output tokens/min).\n"
                "Wait 60 seconds, then try again — or switch to Anthropic API for no rate limits."
            )
        if "NotFoundError" in name or "404" in msg:
            return ValueError(
                f"Groq model not found. Check the model name in the dropdown.\n"
                f"Error: {msg}"
            )
        if "APIConnectionError" in name or "APITimeoutError" in name:
            return ConnectionError(f"Cannot reach Groq API: {msg}")
        return exc  # any other error, pass through as-is

    def _call_with_retry(self, msgs: list, max_tokens: int, use_json_mode: bool) -> object:
        """Call Groq API with automatic retry on rate-limit errors."""
        import time
        effective_tokens = min(max_tokens, self.GROQ_MAX_OUTPUT_TOKENS)
        kwargs = dict(
            model=self.model,
            messages=msgs,
            max_tokens=effective_tokens,
            temperature=self.GROQ_TEMPERATURE,
            seed=self.GROQ_SEED,
        )
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exc = None
        for attempt in range(self.GROQ_MAX_RETRIES):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                name = type(exc).__name__
                is_rate_limit = "RateLimitError" in name or "429" in str(exc)
                if is_rate_limit and attempt < self.GROQ_MAX_RETRIES - 1:
                    wait = self.GROQ_RETRY_WAIT * (attempt + 1)
                    _emit_progress(
                        "rate_limit",
                        f"Groq rate limit — waiting {wait}s before retry "
                        f"({attempt + 1}/{self.GROQ_MAX_RETRIES})",
                    )
                    time.sleep(wait)
                    continue
                # Out of retries or non-rate-limit error
                wrapped = self._wrap_groq_error(exc)
                _emit_progress(
                    "rate_limit" if is_rate_limit else "error",
                    f"Groq call failed: {wrapped}",
                )
                raise wrapped from exc
        raise self._wrap_groq_error(last_exc) from last_exc

    def parse(self, model=None, max_tokens=4096, system=None,
              messages=None, output_format=None, **kwargs):
        """
        Call Groq and parse the JSON response into a Pydantic model.

        model= from agents is a Claude model name — ignored, we use self.model.
        Anthropic-specific kwargs (thinking, output_config) are silently ignored.
        """
        msgs = []

        system_content = system or ""
        if isinstance(system_content, list):
            # Unwrap cache_control blocks (passed by TrackedMessages for Anthropic)
            system_content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in system_content
            )

        if output_format:
            example = _build_example_json(output_format)
            example_str = json.dumps(example, indent=2)
            system_content += (
                f"\n\n===CRITICAL OUTPUT REQUIREMENT===\n"
                f"You MUST respond with ONLY valid JSON. No markdown code blocks, "
                f"no explanations, no text before or after. "
                f"Your entire response must be a single JSON object with EXACTLY these fields "
                f"(fill in real values — do NOT copy this example verbatim):\n"
                f"{example_str}"
            )

        if system_content:
            msgs.append({"role": "system", "content": system_content})
        if messages:
            msgs.extend(messages)

        response = self._call_with_retry(
            msgs=msgs,
            max_tokens=max_tokens,
            use_json_mode=(output_format is not None),
        )
        content = response.choices[0].message.content or ""

        if output_format:
            try:
                text = content.strip()
                if text.startswith("```"):
                    text = text.split("```", 2)[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                if text.endswith("```"):
                    text = text[:-3].strip()
                data = json.loads(text)
                data = _clean_ollama_response(data, output_format)
                parsed = output_format.model_validate(data)
                return _ParseResult(parsed)
            except (json.JSONDecodeError, Exception) as e:
                raise ValueError(
                    f"Groq response could not be parsed as {output_format.__name__}: {e}\n"
                    f"Raw response (first 400 chars): {content[:400]}"
                ) from e

        return _ParseResult(content)

    def stream(self, model=None, max_tokens=8192, system=None,
               messages=None, **kwargs):
        """
        Return a _GroqStream context manager.
        Anthropic-specific kwargs (thinking, output_config) are silently ignored.
        """
        return _GroqStream(
            client=self._client,
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=min(max_tokens, self.GROQ_MAX_OUTPUT_TOKENS),
        )


class _GroqStream:
    """
    Context manager that streams from Groq's chat completions endpoint
    and yields events compatible with the Anthropic streaming format.
    Includes proper error wrapping so Groq SDK exceptions surface cleanly.
    """

    def __init__(self, client, model: str, system, messages: list, max_tokens: int):
        self._client = client
        self.model = model
        self.system = system
        self.messages = messages
        self.max_tokens = max_tokens
        self._full_text = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        msgs = []
        # Handle system as string or list (cache_control blocks from Anthropic path)
        if self.system:
            if isinstance(self.system, list):
                system_text = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in self.system
                )
            else:
                system_text = self.system
            if system_text:
                msgs.append({"role": "system", "content": system_text})
        if self.messages:
            msgs.extend(self.messages)

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=msgs,
                max_tokens=self.max_tokens,
                temperature=_GroqMessages.GROQ_TEMPERATURE,
                seed=_GroqMessages.GROQ_SEED,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = (delta.content or "") if delta else ""
                if content:
                    self._full_text += content
                    yield _OllamaStreamEvent(content)
        except Exception as exc:
            raise _GroqMessages._wrap_groq_error(exc) from exc


# ── Client classes ─────────────────────────────────────────────────────────


class GroqLLMClient:
    """
    FREE cloud LLM client backed by Groq API.

    WHY GROQ?
      - Completely free tier (14,400 requests/day)
      - No installation, no downloads, no local GPU required
      - Uses Llama 3.3-70B — near GPT-4 quality
      - Blazing fast inference (custom LPU hardware)
      - OpenAI-compatible API

    SETUP (one-time, 2 minutes):
      1. Sign up at https://console.groq.com/ (free, no credit card)
      2. Create an API key in the console
      3. Add to .env:  GROQ_API_KEY=your-key-here

    Good Groq models for financial analysis:
      llama-3.3-70b-versatile   — best quality, recommended (default)
      llama-3.1-8b-instant      — faster, lower quality
      mixtral-8x7b-32768        — good structured output, 32K context
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in .env.\n"
                "Get a free key at: https://console.groq.com/keys\n"
                "Add this line to your .env file:  GROQ_API_KEY=gsk_your-key-here\n"
                "Then restart Streamlit."
            )
        self.model = model
        self._client = Groq(api_key=api_key)
        self.messages = _GroqMessages(self._client, model)

    def get_model_name(self) -> str:
        return f"Groq / {self.model}"


class OllamaLLMClient:
    """
    Free, local LLM client backed by Ollama.

    Requirements:
      - Ollama installed (https://ollama.ai)
      - A model pulled: ollama pull llama3.2
      - Ollama running: ollama serve  (or auto-started)

    Good free models to try:
      llama3.2        — recommended (3.8B, fast, good quality)
      llama3.1:8b     — larger, better reasoning
      mistral         — good at structured JSON output
      qwen2.5:7b      — strong on financial text
    """

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self._base_url = base_url
        self.messages = _OllamaMessages(model, base_url)

    def get_model_name(self) -> str:
        return f"Ollama / {self.model}"


class AnthropicLLMClient:
    """
    Paid Claude API client backed by Anthropic.
    Wraps .messages with TrackedMessages so every parse() / stream() call
    is automatically recorded in cost_tracker.tracker.
    """

    def __init__(self):
        import anthropic
        from cost_tracker import TrackedMessages, tracker as _tracker
        self._client = anthropic.Anthropic()
        # Wrap the real messages interface with cost tracking
        self.messages = TrackedMessages(self._client.messages, _tracker)

    def get_model_name(self) -> str:
        return "claude-opus-4-6"


# ── Factory ────────────────────────────────────────────────────────────────


def get_available_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """
    Return a list of model names currently pulled in Ollama.
    Returns an empty list if Ollama is not reachable.
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [m["name"] for m in models]
    except Exception:
        return []


def create_llm_client(provider: str, model: str = None):
    """
    Create the appropriate LLM client.

    Args:
        provider:  "api"    → Anthropic Claude (requires ANTHROPIC_API_KEY, paid)
                   "groq"   → Groq cloud API   (requires GROQ_API_KEY, FREE)
                   "ollama" → Local Ollama     (requires Ollama running, FREE)
        model:     Override model name for Ollama or Groq
                   Ollama default: "llama3.2"
                   Groq default:   "llama-3.3-70b-versatile"

    Returns:
        A client object with .messages.parse(), .messages.stream(), .get_model_name()
    """
    if provider == "api":
        return AnthropicLLMClient()

    if provider == "groq":
        groq_model = model or "llama-3.3-70b-versatile"
        return GroqLLMClient(model=groq_model)

    if provider == "ollama":
        ollama_model = model or "llama3.2"

        # ── Connectivity check ──────────────────────────────────────
        try:
            requests.get("http://localhost:11434/api/version", timeout=5).raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Ollama is not running. Start it with: ollama serve"
            )
        except Exception as e:
            raise ConnectionError(f"Ollama connectivity check failed: {e}")

        # ── Model availability check — show actual list on failure ──
        available = get_available_ollama_models()
        if available and ollama_model not in available:
            available_str = ", ".join(available)
            raise ValueError(
                f"Ollama model '{ollama_model}' is not pulled.\n"
                f"Available models: {available_str}\n"
                f"To pull it run: ollama pull {ollama_model}"
            )

        return OllamaLLMClient(model=ollama_model)

    raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'api', 'groq', or 'ollama'.")
