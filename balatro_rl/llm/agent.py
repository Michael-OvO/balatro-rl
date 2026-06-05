"""LLMAgent: an LLM policy with the same act(state, mask) -> action_id interface as
the baseline agents, so it drops into runner.run_episode + the eval harness unchanged.

Per act(): serialize the state, build + render the legal menu, ask the Policy, parse
the chosen action (validated legal), and feed the turn back into the multi-turn context.
On an unparseable / illegal reply it retries with the error, then falls back to a safe
legal action so a game never deadlocks during a baseline run.
"""
from __future__ import annotations

import numpy as np

from .actions_text import build_menu, parse_action, render_menu
from .context import ConversationContext
from .policy_client import Policy
from .serialize import serialize_state

SYSTEM_PROMPT = (
    "You are an expert Balatro player. Each turn you see the game state and a list of "
    "legal actions. Think briefly, then reply with exactly one JSON object choosing your "
    'action: {"choice": <index>} for a listed action, or {"action": "play"|"discard"|'
    '"target", "cards": [hand indices]} to play/discard/target cards. Your goal is to '
    "clear blinds and win Ante 8."
)


class LLMAgent:
    def __init__(self, policy: Policy, window_turns: int = 12, max_retries: int = 2,
                 system_prompt: str = SYSTEM_PROMPT):
        self._policy = policy
        self._ctx = ConversationContext(system_prompt=system_prompt, window_turns=window_turns)
        self._max_retries = max_retries

    def reset(self) -> None:
        self._ctx = ConversationContext(system_prompt=self._ctx._system,
                                        window_turns=self._ctx._window)

    def act(self, state, mask) -> int:
        # Build the menu once and reuse the caller's mask (run_episode passes legal_mask(state));
        # parse_action then validates against these instead of recomputing legal_actions each retry.
        menu = build_menu(state)
        observation = serialize_state(state) + "\n\n" + render_menu(menu)
        messages = self._ctx.render(observation)
        reply, chosen = "", None
        for _ in range(self._max_retries + 1):
            reply = self._policy.generate(messages)
            res = parse_action(reply, state, menu=menu, mask=mask)
            if res.error is None:
                chosen = res.action_id
                break
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That action was invalid ({res.error}). "
                                             "Reply with one valid JSON action."},
            ]
        if chosen is None:
            chosen = int(np.flatnonzero(mask)[0])    # safe legal fallback
            reply = "(no valid action parsed; defaulted to the first legal action)"
        self._ctx.update(assistant_reply=reply, observation="")
        return chosen
