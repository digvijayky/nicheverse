"""Pluggable LLM providers for code annotation: Claude, GPT, or a local model.

Each backend is imported lazily so the core package never depends on an LLM SDK.
Install the optional extra with ``pip install nicheverse[llm]`` (anthropic, openai,
requests) to enable ``anthropic`` / ``openai`` / ``ollama`` providers.
"""

from __future__ import annotations

import json
import os
import re

__all__ = ["call_llm", "parse_json", "PROVIDERS"]

PROVIDERS = ("anthropic", "openai", "ollama")


def call_llm(
    prompt: str,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    system: str | None = None,
    max_tokens: int = 1200,
    temperature: float = 0.0,
) -> str:
    """Send ``prompt`` to an LLM and return the raw text response.

    Parameters
    ----------
    provider
        ``"anthropic"`` (Claude), ``"openai"`` (GPT), or ``"ollama"`` (a local
        Ollama server; also covers any OpenAI-compatible local endpoint).
    model
        Model id; defaults per provider (Claude Opus / GPT-4o / llama3).
    api_key
        Overrides the ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` environment variable.
    system
        Optional system prompt.
    """
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        kw: dict = dict(
            model=model or "claude-opus-4-8",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kw["system"] = system
        resp = client.messages.create(**kw)
        return "".join(getattr(b, "text", "") for b in resp.content)
    if provider == "openai":
        import openai

        client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return (
            client.chat.completions.create(
                model=model or "gpt-4o", messages=msgs, max_tokens=max_tokens, temperature=temperature
            )
            .choices[0]
            .message.content
        )
    if provider == "ollama":
        import requests

        base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        full = (system + "\n\n" if system else "") + prompt
        r = requests.post(
            f"{base}/api/generate",
            json={"model": model or "llama3", "prompt": full, "stream": False, "format": "json"},
            timeout=600,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    raise ValueError(f"unknown provider {provider!r}; choose one of {PROVIDERS}")


def parse_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response (tolerant of prose wrappers)."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}
