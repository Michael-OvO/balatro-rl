"""Fixed flat action space + legal mask over the engine's variable actions.

Layout (NUM_ACTIONS = 465):
  [0, 218)        PLAY    subset = _SUBSETS[id]
  [218, 436)      DISCARD subset = _SUBSETS[id - 218]
  SHOP_BASE=436:
    +0..+1        BUY  offer slot 0..1   (CARD_SLOTS)
    +2..+6        SELL joker slot 0..4   (JOKER_SLOTS)
    +7            REROLL
    +8..+27       REORDER pair _PAIRS[k]  (20 ordered (i,j), i!=j, over JOKER_SLOTS)
    +28           LEAVE_SHOP
Subsets are enumerated over MAX_HAND=8 slots; subsets referencing absent hand
slots are simply never legal (the engine never offers them), so they mask out.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..engine.engine import Verb, legal_actions
from ..engine.state import Phase

MAX_HAND = 8
MAX_SELECT = 5
MAX_JOKERS = 5     # JOKER_SLOTS
MAX_SHOP = 2       # CARD_SLOTS

_SUBSETS: list[tuple[int, ...]] = [
    c for size in range(1, MAX_SELECT + 1) for c in itertools.combinations(range(MAX_HAND), size)
]
_SUBSET_INDEX: dict[tuple[int, ...], int] = {c: i for i, c in enumerate(_SUBSETS)}
PLAY_N = len(_SUBSETS)                 # 218

_PAIRS: list[tuple[int, int]] = [(i, j) for i in range(MAX_JOKERS) for j in range(MAX_JOKERS) if i != j]
_PAIR_INDEX: dict[tuple[int, int], int] = {p: k for k, p in enumerate(_PAIRS)}

SHOP_BASE = 2 * PLAY_N                  # 436
_BUY0 = SHOP_BASE                       # +0..+1
_SELL0 = SHOP_BASE + MAX_SHOP           # +2..+6
_REROLL = _SELL0 + MAX_JOKERS           # +7
_REORDER0 = _REROLL + 1                 # +8..+27
_LEAVE = _REORDER0 + len(_PAIRS)        # +28
NUM_ACTIONS = _LEAVE + 1                # 465


def decode(action_id: int):
    """Flat id -> engine (Verb, arg)."""
    if action_id < PLAY_N:
        return Verb.PLAY, _SUBSETS[action_id]
    if action_id < 2 * PLAY_N:
        return Verb.DISCARD, _SUBSETS[action_id - PLAY_N]
    if action_id < _SELL0:
        return Verb.BUY, action_id - _BUY0
    if action_id < _REROLL:
        return Verb.SELL, action_id - _SELL0
    if action_id == _REROLL:
        return Verb.REROLL, 0
    if action_id < _LEAVE:
        return Verb.REORDER, _PAIRS[action_id - _REORDER0]
    if action_id == _LEAVE:
        return Verb.LEAVE_SHOP, 0
    raise ValueError(f"action_id out of range: {action_id}")


def _encode(verb, arg) -> int:
    """Engine (Verb, arg) -> flat id (inverse of decode)."""
    if verb == Verb.PLAY:
        return _SUBSET_INDEX[tuple(arg)]
    if verb == Verb.DISCARD:
        return PLAY_N + _SUBSET_INDEX[tuple(arg)]
    if verb == Verb.BUY:
        return _BUY0 + arg
    if verb == Verb.SELL:
        return _SELL0 + arg
    if verb == Verb.REROLL:
        return _REROLL
    if verb == Verb.REORDER:
        return _REORDER0 + _PAIR_INDEX[tuple(arg)]
    if verb == Verb.LEAVE_SHOP:
        return _LEAVE
    raise ValueError(f"unknown verb: {verb}")


def legal_mask(state) -> np.ndarray:
    """Boolean array of length NUM_ACTIONS: True where the flat id is legal now."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for verb, arg in legal_actions(state):
        mask[_encode(verb, arg)] = True
    return mask
