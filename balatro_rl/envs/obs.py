"""Encode a GameState into a dict of numpy arrays for the (future) JAX agent.

Faithful, fixed-shape entity arrays + masks + a global scalar vector. Exponential
scalars (score/required/money/counter) are symlog-compressed. Card ranks 2..14 map
to one-hot index rank-2 (13 ranks); suits 0..3 one-hot.
"""
from __future__ import annotations

import math

import numpy as np

from ..engine.state import GameState, Phase
from .actions import MAX_HAND, MAX_JOKERS, MAX_SHOP

CARD_FEAT = 17        # rank one-hot(13) + suit one-hot(4)
GLOBAL_FEAT = 16
N_PHASES = 4
N_LEVELS = 12

OBS_SHAPES: dict[str, tuple] = {
    "global": (GLOBAL_FEAT,),
    "hand": (MAX_HAND, CARD_FEAT),
    "hand_mask": (MAX_HAND,),
    "joker_types": (MAX_JOKERS,),
    "joker_counter": (MAX_JOKERS,),
    "joker_mask": (MAX_JOKERS,),
    "shop_types": (MAX_SHOP,),
    "shop_cost": (MAX_SHOP,),
    "shop_mask": (MAX_SHOP,),
    "levels": (N_LEVELS,),
    "deck_rank_hist": (13,),
    "deck_suit_hist": (4,),
}


def symlog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def _card_vec(card) -> np.ndarray:
    v = np.zeros(CARD_FEAT, dtype=np.float32)
    v[card.rank - 2] = 1.0          # ranks 2..14 -> 0..12
    v[13 + card.suit] = 1.0         # suits 0..3
    return v


def encode(state: GameState) -> dict[str, np.ndarray]:
    g = np.zeros(GLOBAL_FEAT, dtype=np.float32)
    g[0] = symlog(state.round_score)
    g[1] = symlog(state.required)
    g[2] = min(state.round_score / max(state.required, 1), 2.0)
    g[3] = state.hands_left
    g[4] = state.discards_left
    g[5] = symlog(state.money)
    g[6] = state.ante
    g[7] = state.blind_index
    g[8] = state.rerolls_done
    g[9] = state.hand_size
    g[10] = len(state.jokers)
    g[11] = len(state.shop_offers)
    phase = int(state.phase)
    if 0 <= phase < N_PHASES:
        g[12 + phase] = 1.0          # one-hot phase over g[12..15]

    hand = np.zeros((MAX_HAND, CARD_FEAT), dtype=np.float32)
    hand_mask = np.zeros(MAX_HAND, dtype=np.float32)
    for i, card in enumerate(state.hand[:MAX_HAND]):
        hand[i] = _card_vec(card)
        hand_mask[i] = 1.0

    joker_types = np.zeros(MAX_JOKERS, dtype=np.int32)
    joker_counter = np.zeros(MAX_JOKERS, dtype=np.float32)
    joker_mask = np.zeros(MAX_JOKERS, dtype=np.float32)
    for i, js in enumerate(state.jokers[:MAX_JOKERS]):
        joker_types[i] = int(js.type)        # 0 = empty; agent embeds over joker vocab
        joker_counter[i] = symlog(js.counter)
        joker_mask[i] = 1.0

    shop_types = np.zeros(MAX_SHOP, dtype=np.int32)
    shop_cost = np.zeros(MAX_SHOP, dtype=np.float32)
    shop_mask = np.zeros(MAX_SHOP, dtype=np.float32)
    from ..engine.shop import joker_cost
    for i, js in enumerate(state.shop_offers[:MAX_SHOP]):
        shop_types[i] = int(js.type)
        shop_cost[i] = float(joker_cost(js.type))
        shop_mask[i] = 1.0

    levels = np.asarray(state.levels, dtype=np.float32)

    deck_rank = np.zeros(13, dtype=np.float32)
    deck_suit = np.zeros(4, dtype=np.float32)
    for card in state.deck:
        deck_rank[card.rank - 2] += 1.0
        deck_suit[card.suit] += 1.0

    return {
        "global": g, "hand": hand, "hand_mask": hand_mask,
        "joker_types": joker_types, "joker_counter": joker_counter, "joker_mask": joker_mask,
        "shop_types": shop_types, "shop_cost": shop_cost, "shop_mask": shop_mask,
        "levels": levels, "deck_rank_hist": deck_rank, "deck_suit_hist": deck_suit,
    }
