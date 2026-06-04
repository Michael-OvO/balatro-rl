"""Encode a GameState into a dict of numpy arrays for the (future) JAX agent.

Faithful, fixed-shape entity arrays + masks + a global scalar vector. Exponential
scalars (score/required/money/counter) are symlog-compressed. Card ranks 2..14 map
to one-hot index rank-2 (13 ranks); suits 0..3 one-hot.
"""
from __future__ import annotations

import math

import numpy as np

from ..engine.bosses import BossEffect, boss_debuffed_idx
from ..engine.cards import Edition, Enhancement, Seal
from ..engine.jokers.base import aggregate_rules
from ..engine.state import GameState
from .actions import MAX_CONSUM, MAX_HAND, MAX_JOKERS, MAX_SHOP

# Per-card features (Phase D widening): rank(13) + suit(4) + enhancement(9) + edition(4)
# + seal(5) + is_debuffed(1) + is_face_down(1). is_face_down is always 0 for now (the
# face-down bosses are deferred) but reserved so the obs schema is stable across the retrain.
_N_ENH, _N_EDITION, _N_SEAL = len(Enhancement), len(Edition), len(Seal)   # 9, 4, 5
_ENH0, _EDITION0, _SEAL0 = 17, 17 + _N_ENH, 17 + _N_ENH + _N_EDITION       # 17, 26, 30
_DEBUFF_I, _FACEDOWN_I = _SEAL0 + _N_SEAL, _SEAL0 + _N_SEAL + 1             # 35, 36
CARD_FEAT = _FACEDOWN_I + 1                                                 # 37
GLOBAL_FEAT = 19      # ...g[16]=#consumables, g[17]=consumable_slots, g[18]=boss-active
N_PHASES = 4
N_LEVELS = 12
N_BOSS = len(BossEffect)          # 29 (NONE + 28); boss one-hot
CONSUM_VOCAB = 128                # flat consumable id space (kind*32 + type_id), for the agent Embed


def consum_vocab_id(con) -> int:
    """Flat embedding id for a consumable: kind*32 + type_id (0 = empty slot)."""
    return con.kind * 32 + con.type_id


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
    "boss_onehot": (N_BOSS,),
    "consum_types": (MAX_CONSUM,),
    "consum_mask": (MAX_CONSUM,),
}


def symlog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def _card_vec(card, debuffed: bool = False, face_down: bool = False) -> np.ndarray:
    v = np.zeros(CARD_FEAT, dtype=np.float32)
    v[card.rank - 2] = 1.0                  # ranks 2..14 -> 0..12
    v[13 + card.suit] = 1.0                 # suits 0..3
    v[_ENH0 + card.enhancement] = 1.0       # enhancement one-hot (NONE..STONE)
    v[_EDITION0 + card.edition] = 1.0       # edition one-hot (NONE..POLY)
    v[_SEAL0 + card.seal] = 1.0             # seal one-hot (NONE..PURPLE)
    v[_DEBUFF_I] = 1.0 if debuffed else 0.0
    v[_FACEDOWN_I] = 1.0 if face_down else 0.0
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
    g[16] = len(state.consumables)
    g[17] = state.consumable_slots
    g[18] = 1.0 if state.boss else 0.0

    # Which hand cards the active boss would debuff (suit/face bosses). Empty off a boss
    # blind, so every is_debuffed feature is 0 and the widened obs matches the plain game.
    boss = BossEffect(state.boss)
    rules = aggregate_rules(state.jokers)
    debuffed = set(boss_debuffed_idx(boss, state.hand[:MAX_HAND], rules)) if state.boss else set()
    hand = np.zeros((MAX_HAND, CARD_FEAT), dtype=np.float32)
    hand_mask = np.zeros(MAX_HAND, dtype=np.float32)
    for i, card in enumerate(state.hand[:MAX_HAND]):
        hand[i] = _card_vec(card, debuffed=i in debuffed)
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
    # E1: the agent is BLIND to consumable offers. A JOKER offer encodes its joker-vocab
    # type id; a non-joker offer encodes type 0 (no info beyond "an offer exists"). Cost
    # and mask are still set for every offer (same fields/shapes; just kind-aware).
    from ..engine.shop import ShopKind
    for i, offer in enumerate(state.shop_offers[:MAX_SHOP]):
        shop_types[i] = int(offer.type_id) if offer.kind == ShopKind.JOKER else 0
        shop_cost[i] = float(offer.cost)
        shop_mask[i] = 1.0

    levels = np.asarray(state.levels, dtype=np.float32)

    deck_rank = np.zeros(13, dtype=np.float32)
    deck_suit = np.zeros(4, dtype=np.float32)
    for card in state.deck:
        deck_rank[card.rank - 2] += 1.0
        deck_suit[card.suit] += 1.0

    boss_onehot = np.zeros(N_BOSS, dtype=np.float32)
    if 0 <= state.boss < N_BOSS:
        boss_onehot[state.boss] = 1.0

    consum_types = np.zeros(MAX_CONSUM, dtype=np.int32)
    consum_mask = np.zeros(MAX_CONSUM, dtype=np.float32)
    for i, con in enumerate(state.consumables[:MAX_CONSUM]):
        consum_types[i] = consum_vocab_id(con)
        consum_mask[i] = 1.0

    return {
        "global": g, "hand": hand, "hand_mask": hand_mask,
        "joker_types": joker_types, "joker_counter": joker_counter, "joker_mask": joker_mask,
        "shop_types": shop_types, "shop_cost": shop_cost, "shop_mask": shop_mask,
        "levels": levels, "deck_rank_hist": deck_rank, "deck_suit_hist": deck_suit,
        "boss_onehot": boss_onehot, "consum_types": consum_types, "consum_mask": consum_mask,
    }
