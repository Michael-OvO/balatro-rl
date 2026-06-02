"""Single-game Balatro environment. Thin pure adapter over the engine:
reset() -> (obs, mask); step(action_id) -> (obs, reward, done, info, mask).
The action is validated against the legal mask (the policy must mask).
"""
from __future__ import annotations

import numpy as np

from ..engine import engine
from .actions import decode, legal_mask
from .obs import encode
from .rewards import make_reward


class BalatroEnv:
    def __init__(self, reward_name: str = "shaped"):
        self._reward = make_reward(reward_name)
        self.state = None

    def reset(self, seed: int = 0):
        self.state = engine.reset(seed)
        self._reward.reset()
        return encode(self.state), legal_mask(self.state)

    def step(self, action_id: int):
        mask = legal_mask(self.state)
        assert mask[action_id], f"illegal action {action_id} (decoded {decode(action_id)})"
        prev = self.state
        verb, arg = decode(action_id)
        nxt, info = engine.step(prev, (verb, arg))
        self.state = nxt
        reward = float(self._reward(prev, action_id, nxt, info))
        done = bool(nxt.done)
        new_mask = legal_mask(nxt) if not done else np.zeros_like(mask)
        return encode(nxt), reward, done, info, new_mask
