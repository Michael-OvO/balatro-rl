"""Policy backends for the LLM agent.

A Policy maps a chat-style message list to the assistant's text reply. The same
interface serves the frozen baseline (this file's FrozenEndpointPolicy, talking to
a vLLM OpenAI-compatible endpoint) and, later, the verl training rollout worker.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Policy(Protocol):
    def generate(self, messages: list[dict]) -> str:
        """messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        Returns the assistant's text reply."""
        ...


class FrozenEndpointPolicy:
    """Talks to an OpenAI-compatible chat endpoint (e.g. a vLLM server). Inject a
    pre-built `client` (tests pass a fake); otherwise an openai.OpenAI client is built
    from base_url/api_key. Used for the M1 frozen baseline and for eval."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str = "EMPTY",
                 temperature: float = 0.7, max_tokens: int = 512, client=None):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI            # optional dep: pip install -e '.[llm]'
            self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages,
            temperature=self._temperature, max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content or ""
