"""
llm_client.py
=============
Unified LLM client wrapper that normalizes the interface across
Anthropic API, Groq, and local Ollama into a single consistent API.

All clients expose:
  client.messages.stream(**kwargs)   → context manager yielding Anthropic-style events
  client.messages.parse(output_format=PydanticModel, **kwargs) → ParsedResponse
  client.get_model_name()            → str

This lets orchestrator.py and all agents work identically regardless
of which provider is configured.
"""

import json
import re
import os
from contextlib import contextmanager
from typing import Any, Type
from pydantic import BaseModel


# ── Shared helpers ─────────────────────────────────────────────────────

class ParsedResponse:
    """Wraps structured output so callers can access .parsed_output."""
    def __init__(self, parsed_output: Any):
        self.parsed_output = parsed_output


def _extract_json(text: str) -> str:
    """Extract the first JSON object from a string, handling markdown fences."""
    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    # Find outermost JSON object
    start = text.find("{")
    if start == -1:
        return text.strip()

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:].strip()


def _schema_injection(system: str, output_format: Type[BaseModel]) -> str:
    """Append the Pydantic JSON schema to the system prompt."""
    schema = json.dumps(output_format.model_json_schema(), indent=2)
    return (
        system
        + f"\n\nReturn ONLY valid JSON that exactly matches this schema "
        f"(no extra keys, no markdown):\n{schema}"
    )


# ── Fake streaming event classes (mimic Anthropic SDK event shape) ────

class _TextDelta:
    def __init__(self, text: str):
        self.type = "text_delta"
        self.text = text


class _ContentBlockDelta:
    def __init__(self, text: str):
        self.type = "content_block_delta"
        self.delta = _TextDelta(text)


class _FakeStreamContext:
    """
    Context manager that wraps a list of pre-collected chunks
    into Anthropic-compatible streaming events.
    Used for Groq and Ollama where we collect first, then emit.
    """
    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __iter__(self):
        for chunk in self._chunks:
            if chunk:
                yield _ContentBlockDelta(chunk)


# ── Anthropic ─────────────────────────────────────────────────────────

class _AnthropicMessages:
    def __init__(self, client):
        self._client = client

    def stream(self, **kwargs):
        """Delegate to Anthropic SDK, stripping unsupported custom params."""
        kwargs.pop("output_format", None)
        kwargs.pop("thinking", None)      # Extended thinking — future enhancement
        kwargs.pop("output_config", None) # Not a standard Anthropic API param
        return self._client.messages.stream(**kwargs)

    def parse(self, output_format: Type[BaseModel], **kwargs):
        """Call Anthropic and parse the response into a Pydantic model."""
        kwargs.pop("thinking", None)
        kwargs.pop("output_config", None)

        system = kwargs.pop("system", "")
        kwargs["system"] = _schema_injection(system, output_format)

        response = self._client.messages.create(**kwargs)
        text = response.content[0].text

        json_str = _extract_json(text)
        data = json.loads(json_str)
        parsed = output_format.model_validate(data)
        return ParsedResponse(parsed)


class AnthropicLLMClient:
    def __init__(self):
        import anthropic
        self._client = anthropic.Anthropic()
        self.messages = _AnthropicMessages(self._client)

    def get_model_name(self) -> str:
        return "claude-opus-4-6"


# ── Groq ──────────────────────────────────────────────────────────────

class _GroqMessages:
    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def _build_groq_messages(self, system: str, messages: list) -> list:
        result = []
        if system:
            result.append({"role": "system", "content": system})
        result.extend(messages)
        return result

    def stream(self, **kwargs):
        model = kwargs.pop("model", self._model)
        system = kwargs.pop("system", "")
        messages = kwargs.pop("messages", [])
        max_tokens = kwargs.pop("max_tokens", 4096)
        kwargs.pop("output_format", None)
        kwargs.pop("thinking", None)
        kwargs.pop("output_config", None)

        groq_messages = self._build_groq_messages(system, messages)
        chunks = []

        response = self._client.chat.completions.create(
            model=model,
            messages=groq_messages,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                chunks.append(delta.content)

        return _FakeStreamContext(chunks)

    def parse(self, output_format: Type[BaseModel], **kwargs):
        model = kwargs.pop("model", self._model)
        system = kwargs.pop("system", "")
        messages = kwargs.pop("messages", [])
        max_tokens = kwargs.pop("max_tokens", 4096)
        kwargs.pop("thinking", None)
        kwargs.pop("output_config", None)

        system = _schema_injection(system, output_format)
        groq_messages = self._build_groq_messages(system, messages)

        response = self._client.chat.completions.create(
            model=model,
            messages=groq_messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        json_str = _extract_json(text)
        data = json.loads(json_str)
        parsed = output_format.model_validate(data)
        return ParsedResponse(parsed)


class GroqLLMClient:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = model
        self.messages = _GroqMessages(self._client, model)

    def get_model_name(self) -> str:
        return self._model


# ── Ollama ────────────────────────────────────────────────────────────

class _OllamaMessages:
    def __init__(self, model: str, base_url: str):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def _call(self, system: str, messages: list, max_tokens: int, stream: bool = False):
        import requests
        payload = {
            "model": self._model,
            "messages": [],
            "stream": stream,
            "options": {"num_predict": max_tokens},
        }
        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].extend(messages)

        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120,
            stream=stream,
        )
        resp.raise_for_status()
        return resp

    def stream(self, **kwargs):
        system = kwargs.pop("system", "")
        messages = kwargs.pop("messages", [])
        max_tokens = kwargs.pop("max_tokens", 4096)
        kwargs.pop("model", None)
        kwargs.pop("output_format", None)
        kwargs.pop("thinking", None)
        kwargs.pop("output_config", None)

        import requests as req_lib
        chunks = []
        resp = self._call(system, messages, max_tokens, stream=True)
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        chunks.append(content)
                except json.JSONDecodeError:
                    pass

        return _FakeStreamContext(chunks)

    def parse(self, output_format: Type[BaseModel], **kwargs):
        system = kwargs.pop("system", "")
        messages = kwargs.pop("messages", [])
        max_tokens = kwargs.pop("max_tokens", 4096)
        kwargs.pop("model", None)
        kwargs.pop("thinking", None)
        kwargs.pop("output_config", None)

        system = _schema_injection(system, output_format)
        resp = self._call(system, messages, max_tokens, stream=False)
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        json_str = _extract_json(text)
        parsed_data = json.loads(json_str)
        parsed = output_format.model_validate(parsed_data)
        return ParsedResponse(parsed)


class OllamaLLMClient:
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self._model = model
        self.messages = _OllamaMessages(model, base_url)

    def get_model_name(self) -> str:
        return f"ollama/{self._model}"


# ── Factory ───────────────────────────────────────────────────────────

def create_llm_client(provider: str, model: str = None):
    """
    Create the appropriate LLM client based on the provider.

    Args:
        provider: "api" (Anthropic), "groq", or "ollama"
        model:    Model name override (used for Groq/Ollama)

    Returns:
        One of AnthropicLLMClient, GroqLLMClient, OllamaLLMClient
    """
    if provider == "api":
        return AnthropicLLMClient()
    elif provider == "groq":
        return GroqLLMClient(model or "llama-3.3-70b-versatile")
    elif provider == "ollama":
        return OllamaLLMClient(model or "llama3.2")
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'api', 'groq', or 'ollama'.")
