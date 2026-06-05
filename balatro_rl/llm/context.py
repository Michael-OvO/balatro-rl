"""Bounded multi-turn conversation context for the LLM agent.

Keeps a static system prompt, a sliding window of the most recent (observation,
assistant-reply) turns, and a one-line deterministic note for the turns dropped out
of the window so context length stays bounded across a ~300-turn game. A model-written
plan slot and a richer summary are deferred (YAGNI) behind this same interface.
"""
from __future__ import annotations


class ConversationContext:
    def __init__(self, system_prompt: str, window_turns: int = 12):
        self._system = system_prompt
        self._window = window_turns
        self._turns: list[tuple[str, str]] = []   # (observation, assistant_reply)
        self._dropped = 0

    def render(self, observation: str) -> list[dict]:
        system_content = self._system
        if self._dropped:
            system_content += f"\n\n(Summary: {self._dropped} earlier turns elided.)"
        messages = [{"role": "system", "content": system_content}]
        for obs, reply in self._turns[-self._window:]:
            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": observation})
        self._pending_obs = observation
        return messages

    def update(self, assistant_reply: str, observation: str = "") -> None:
        obs = getattr(self, "_pending_obs", observation)
        self._turns.append((obs, assistant_reply))
        if len(self._turns) > self._window:
            self._dropped = len(self._turns) - self._window
