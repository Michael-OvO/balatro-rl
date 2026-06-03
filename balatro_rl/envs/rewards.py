"""Pluggable reward objectives. A reward is a callable
(prev_state, action_id, next_state, info) -> float, with a reset() for any
internal state. Objectives are the experiment variable (win vs score vs depth).
"""
from __future__ import annotations

import math

from ..engine.hands import HAND_BASE, HandType


def _symlog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def _shaped_potential(s) -> float:
    """Φ(s) for potential-based shaping (Ng 1999): within-blind progress (capped at
    1), money, and ante. Shared by `shaped`, `shaped_scaled`, and `hand_quality` so
    the three differ ONLY where the sweep intends — never in the shaping backbone."""
    ratio = min(s.round_score / max(s.required, 1), 1.0)
    return ratio + 0.05 * _symlog(s.money) + 0.5 * s.ante


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

    def __call__(self, prev, action_id, nxt, info):
        shaped = self.gamma * _shaped_potential(nxt) - _shaped_potential(prev)
        if info.get("cleared"):
            shaped += 1.0
        if nxt.done and nxt.won:
            shaped += 10.0
        return shaped


class HandQuality:
    """Explicitly reward forming higher-TIER poker hands, not just raw chips.

    Probe for the blind-1 wall: does telling the agent "good hands matter"
    (a Three-of-a-Kind beats a High Card) — rather than "more chips" — get it
    off High Card / Pair? Each PLAY pays a bounded bonus for the hand's tier
    (HandType / HAND_BASE), so a rarer/stronger hand is always worth strictly
    more than a weaker one regardless of which exact cards filled it.

    Reward per step:
      PLAY:   q_weight * tier(info["hand_type"])          [explicit tier signal]
            + Shaped-style potential shaping toward clearing the blind
      DISCARD/SHOP/REORDER/LEAVE: only the potential-shaping term (no tier; no
            hand was formed).
      milestones: +1 on a cleared blind, +10 on a win (same as Shaped).

    tier(ht) = log1p(base_chips*base_mult) / log1p(max base) in (0, 1], a
    monotone ladder over the 12 HandTypes (HIGH_CARD ~0.23 ... FLUSH_FIVE 1.0),
    so the signal is dense, bounded, and scale-compatible with "shaped".
    """
    # Normalizer: log1p of the strongest hand's base potential (FLUSH_FIVE).
    _MAX_BASE = max(c * m for c, m in HAND_BASE.values())  # 2560
    _NORM = math.log1p(_MAX_BASE)

    def __init__(self, gamma: float = 0.999, q_weight: float = 1.0):
        self.gamma = gamma
        self.q_weight = q_weight

    def reset(self):
        pass

    @classmethod
    def _tier(cls, hand_type: int) -> float:
        c, m = HAND_BASE[HandType(hand_type)]
        return math.log1p(c * m) / cls._NORM

    def __call__(self, prev, action_id, nxt, info):
        r = self.gamma * _shaped_potential(nxt) - _shaped_potential(prev)
        # Only PLAY actions carry a hand_type. DISCARD/SHOP/LEAVE_SHOP do not.
        if info.get("verb") == "play" and "hand_type" in info:
            r += self.q_weight * self._tier(info["hand_type"])
        if info.get("cleared"):
            r += 1.0
        if nxt.done and nxt.won:
            r += 10.0
        return r


class HandQualityQ05(HandQuality):
    """`hand_quality` with HALF the tier weight (q_weight=0.5) — separates 'tier
    signal too weak' from 'signal present, representation can't act on it'."""
    def __init__(self, gamma: float = 0.999):
        super().__init__(gamma=gamma, q_weight=0.5)


class ShapedScaled:
    """The `shaped` reward with its potential term multiplied by `scale`, leaving
    the +1/+10 milestones unscaled. Reward-SCALE control: at scale=1.5 the mean
    per-step magnitude ~matches `hand_quality`'s, so a hand_quality win can be
    attributed to the tier-ladder CONTENT rather than to a merely-bigger reward."""
    def __init__(self, gamma: float = 0.999, scale: float = 1.5):
        self.gamma = gamma
        self.scale = scale

    def reset(self):
        pass

    def __call__(self, prev, action_id, nxt, info):
        r = self.scale * (self.gamma * _shaped_potential(nxt) - _shaped_potential(prev))
        if info.get("cleared"):
            r += 1.0
        if nxt.done and nxt.won:
            r += 10.0
        return r


_FACTORY = {"win_ante8": WinAnte8, "max_depth": MaxDepth, "shaped": Shaped,
            "hand_quality": HandQuality, "hand_quality_q05": HandQualityQ05,
            "shaped_scaled": ShapedScaled}
REWARD_NAMES = tuple(_FACTORY)


def make_reward(name: str):
    return _FACTORY[name]()
