"""Fixed flat action space + legal mask over the engine's variable actions.

Layout (NUM_ACTIONS = 708):
  [0, 218)        PLAY    subset = _SUBSETS[id]
  [218, 436)      DISCARD subset = _SUBSETS[id - 218]
  SHOP_BASE=436:
    +0..+3        BUY  offer slot 0..3   (MAX_SHOP=4 = CARD_SLOTS + Overstock(+Plus); joker OR consumable)
    +4..+9        SELL joker slot 0..5   (MAX_JOKERS=6, incl. the Antimatter slot)
    +10           REROLL
    +11..+40      REORDER pair _PAIRS[k]  (30 ordered (i,j), i!=j, over MAX_JOKERS)
    +41           LEAVE_SHOP
  USE0=478:
    +0..+2        USE consumable slot 0..2 (MAX_CONSUM=3; no-target apply / arm a targeting Tarot)
  --- E5 widening blocks ---
  USE_TARGET0=481:
    +0..+217      USE_TARGET subset = _SUBSETS[id - 481] (apply the ARMED targeting Tarot)
  OPEN0=699:
    +0..+1        OPEN pack offer slot 0..1 (MAX_PACK)
  PICK0=701:
    +0..+4        PICK pack item slot 0..4 (MAX_PACK_ITEMS)
  SKIP_PACK=706
  BUY_VOUCHER=707
Subsets are enumerated over MAX_HAND=8 slots; subsets referencing absent hand
slots are simply never legal (the engine never offers them), so they mask out.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..engine.engine import Verb, legal_actions

MAX_HAND = 8
MAX_SELECT = 5
MAX_JOKERS = 6     # JOKER_SLOTS (5) + the Antimatter voucher's +1; the real-game cap
MAX_SHOP = 4       # CARD_SLOTS (2) + Overstock + Overstock Plus (the real-game max shop card slots)
MAX_CONSUM = 3     # consumable slots: base 2 + the Crystal Ball voucher's +1 (USE action ids)
MAX_PACK = 2       # PACK_SLOTS (booster-pack offer slots)
MAX_PACK_ITEMS = 5  # most an OPEN_PACK ever reveals (Jumbo/Mega = 5 shown)

_SUBSETS: list[tuple[int, ...]] = [
    c for size in range(1, MAX_SELECT + 1) for c in itertools.combinations(range(MAX_HAND), size)
]
_SUBSET_INDEX: dict[tuple[int, ...], int] = {c: i for i, c in enumerate(_SUBSETS)}
PLAY_N = len(_SUBSETS)                 # 218

_PAIRS: list[tuple[int, int]] = [(i, j) for i in range(MAX_JOKERS) for j in range(MAX_JOKERS) if i != j]
_PAIR_INDEX: dict[tuple[int, int], int] = {p: k for k, p in enumerate(_PAIRS)}

SHOP_BASE = 2 * PLAY_N                  # 436
_BUY0 = SHOP_BASE                       # +0..+3   (436..439; MAX_SHOP=4, kind-agnostic offer slot)
_SELL0 = SHOP_BASE + MAX_SHOP           # +4..+9   (440..445; MAX_JOKERS=6)
_REROLL = _SELL0 + MAX_JOKERS           # +10      (446)
_REORDER0 = _REROLL + 1                 # +11..+40 (447..476; 30 pairs)
_LEAVE = _REORDER0 + len(_PAIRS)        # +41      (477)
_USE0 = _LEAVE + 1                       # USE consumable 0..MAX_CONSUM-1 (478..480)
_USE_TARGET0 = _USE0 + MAX_CONSUM        # USE_TARGET subset 0..217 (481..698)
_OPEN0 = _USE_TARGET0 + PLAY_N           # OPEN pack 0..1 (699..700)
_PICK0 = _OPEN0 + MAX_PACK               # PICK item 0..4 (701..705)
_SKIP_PACK = _PICK0 + MAX_PACK_ITEMS     # 706
_BUY_VOUCHER = _SKIP_PACK + 1            # 707
NUM_ACTIONS = _BUY_VOUCHER + 1           # 708


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
    if action_id < _USE_TARGET0:
        return Verb.USE, action_id - _USE0
    if action_id < _OPEN0:
        return Verb.USE_TARGET, _SUBSETS[action_id - _USE_TARGET0]
    if action_id < _PICK0:
        return Verb.OPEN, action_id - _OPEN0
    if action_id < _SKIP_PACK:
        return Verb.PICK, action_id - _PICK0
    if action_id == _SKIP_PACK:
        return Verb.SKIP_PACK, 0
    if action_id == _BUY_VOUCHER:
        return Verb.BUY_VOUCHER, 0
    raise ValueError(f"action_id out of range: {action_id}")


def encode_action(verb, arg) -> int:
    """Engine (Verb, arg) -> flat id (inverse of decode)."""
    if verb == Verb.PLAY:
        return _SUBSET_INDEX[tuple(arg)]
    if verb == Verb.DISCARD:
        return PLAY_N + _SUBSET_INDEX[tuple(arg)]
    if verb == Verb.BUY:
        if arg >= MAX_SHOP:
            raise ValueError(f"BUY offer slot {arg} >= MAX_SHOP ({MAX_SHOP})")
        return _BUY0 + arg
    if verb == Verb.SELL:
        return _SELL0 + arg
    if verb == Verb.REROLL:
        return _REROLL
    if verb == Verb.REORDER:
        return _REORDER0 + _PAIR_INDEX[tuple(arg)]
    if verb == Verb.LEAVE_SHOP:
        return _LEAVE
    if verb == Verb.USE:
        return _USE0 + arg
    if verb == Verb.USE_TARGET:
        return _USE_TARGET0 + _SUBSET_INDEX[tuple(arg)]
    if verb == Verb.OPEN:
        return _OPEN0 + arg
    if verb == Verb.PICK:
        return _PICK0 + arg
    if verb == Verb.SKIP_PACK:
        return _SKIP_PACK
    if verb == Verb.BUY_VOUCHER:
        return _BUY_VOUCHER
    raise ValueError(f"unknown verb: {verb}")


def legal_mask(state) -> np.ndarray:
    """Boolean array of length NUM_ACTIONS: True where the flat id is legal now.

    The flat action space only enumerates card subsets over MAX_HAND=8 slots.
    A hand of a different size is handled gracefully (no assert): PLAY/DISCARD/
    USE_TARGET actions are clamped/masked to the present min(len(hand), MAX_HAND)
    slots, so a larger hand only offers subsets over the first 8 slots (indices >=
    MAX_HAND stay illegal because they have no flat id) and a smaller hand offers
    fewer subsets. Shop/pack slots beyond their fixed caps likewise mask out.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for verb, arg in legal_actions(state):
        # Subset actions referencing a slot >= MAX_HAND have no flat encoding; skip them
        # (they mask out) instead of raising. Fixed-size shop/pack/consumable slots beyond
        # their caps likewise have no flat id.
        if verb in (Verb.PLAY, Verb.DISCARD, Verb.USE_TARGET) and any(i >= MAX_HAND for i in arg):
            continue
        # A shop offer slot beyond MAX_SHOP has no flat id; skip it (it masks out) rather than
        # let encode_action alias it onto the SELL block. Can't happen at the real-game cap
        # (CARD_SLOTS + Overstock + Overstock Plus = 4 = MAX_SHOP) but degrades gracefully.
        if verb == Verb.BUY and arg >= MAX_SHOP:
            continue
        if verb == Verb.USE and arg >= MAX_CONSUM:
            continue
        if verb == Verb.OPEN and arg >= MAX_PACK:
            continue
        if verb == Verb.PICK and arg >= MAX_PACK_ITEMS:
            continue
        # Defensive: a joker slot beyond MAX_JOKERS has no flat id (can't happen at the current
        # caps, but degrades gracefully if a future voucher raises the engine cap past it).
        if verb == Verb.SELL and arg >= MAX_JOKERS:
            continue
        if verb == Verb.REORDER and any(i >= MAX_JOKERS for i in arg):
            continue
        mask[encode_action(verb, arg)] = True
    return mask
