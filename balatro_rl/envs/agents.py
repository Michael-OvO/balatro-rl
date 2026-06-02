"""Baseline (non-learning) agents: a masked-uniform RandomAgent and a simple
GreedyAgent (play the highest-scoring hand; in the shop, buy then leave). These
are baselines and future behavioral-cloning teachers.
"""
from __future__ import annotations

import numpy as np

from ..engine.engine import Verb, legal_actions
from ..engine.scoring import score_play
from ..engine.state import Phase
from .actions import encode_action


class RandomAgent:
    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def act(self, state, mask) -> int:
        legal = np.flatnonzero(mask)
        return int(self._rng.choice(legal))


class GreedyAgent:
    """Heuristic: in PLAYING, play the highest-scoring legal hand (no discards
    unless forced); in SHOP, buy the first affordable offer, else leave."""

    def act(self, state, mask) -> int:
        if state.phase == Phase.SHOP:
            for verb, arg in legal_actions(state):
                if verb == Verb.BUY:
                    return encode_action(verb, arg)
            return encode_action(Verb.LEAVE_SHOP, 0)
        # PLAYING: choose the best-scoring PLAY; fall back to a discard if no play.
        best_id, best_score = None, -1
        for verb, arg in legal_actions(state):
            if verb == Verb.PLAY:
                sc = score_play([state.hand[i] for i in arg]).score
                if sc > best_score:
                    best_score, best_id = sc, encode_action(verb, arg)
        if best_id is not None:
            return best_id
        # no play available (e.g. 0 hands left but discards remain): discard lowest few
        for verb, arg in legal_actions(state):
            if verb == Verb.DISCARD:
                return encode_action(verb, arg)
        return int(np.flatnonzero(mask)[0])
