"""Pluggable reward objectives. A reward is a callable
(prev_state, action_id, next_state, info) -> float, with a reset() for any
internal state. Objectives are the experiment variable (win vs score vs depth).
"""
from __future__ import annotations

import math


def _symlog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


class WinAnte8:
    """Sparse: +1 on win, -1 on loss, small bonus per blind cleared."""
    def reset(self):
        pass

    def __call__(self, prev, action_id, nxt, info):
        if nxt.done and nxt.won:
            return 1.0
        if nxt.done and not nxt.won:
            return -1.0
        return 0.1 if info.get("cleared") else 0.0


class MaxDepth:
    """Reward progress in antes/blinds; big terminal bonus scaling with depth."""
    def reset(self):
        pass

    def __call__(self, prev, action_id, nxt, info):
        r = 0.0
        if info.get("cleared"):
            r += 1.0
        if nxt.done and nxt.won:
            r += 10.0
        return r


class Shaped:
    """Potential-based shaping (Ng 1999): F = gamma*Phi(s') - Phi(s).

    Phi favors within-blind progress (log chips/required ratio), money, and ante;
    bounded by construction. Plus milestone bonuses on clear/win.
    """
    def __init__(self, gamma: float = 0.999):
        self.gamma = gamma

    def reset(self):
        pass

    def _phi(self, s) -> float:
        ratio = min(s.round_score / max(s.required, 1), 1.0)
        return 1.0 * ratio + 0.05 * _symlog(s.money) + 0.5 * s.ante

    def __call__(self, prev, action_id, nxt, info):
        shaped = self.gamma * self._phi(nxt) - self._phi(prev)
        if info.get("cleared"):
            shaped += 1.0
        if nxt.done and nxt.won:
            shaped += 10.0
        return shaped


_FACTORY = {"win_ante8": WinAnte8, "max_depth": MaxDepth, "shaped": Shaped}
REWARD_NAMES = tuple(_FACTORY)


def make_reward(name: str):
    return _FACTORY[name]()
