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
