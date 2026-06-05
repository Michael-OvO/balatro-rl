"""Encode a GameState into a dict of numpy arrays for the JAX agent.

Faithful, fixed-shape entity arrays + masks + a global scalar vector. Exponential
scalars (score/required/money/counter) are symlog-compressed. Card ranks 2..14 map
to one-hot index rank-2 (13 ranks); suits 0..3 one-hot.

E5 widening: the agent now SEES every acquisition system — consumable shop offers
(`shop_consum`), booster-pack offers + the revealed-item stream, the voucher slot +
owned set, and the pending targeting-Tarot. The `global` block gained the OPEN_PACK
phase bit and the pack/voucher/pending scalars (so the schema differs from the pre-E5
net — the E5 retrain is from scratch, not a warm-start).
"""
from __future__ import annotations

import math

import numpy as np

from ..engine.bosses import BossEffect, boss_debuffed_idx
from ..engine.cards import Edition, Enhancement, Seal
from ..engine.consumables import max_targets
from ..engine.jokers.base import aggregate_rules
from ..engine.packs import PackItemKind
from ..engine.shop import SHOP_TO_CONSUMABLE_KIND, ShopKind
from ..engine.state import GameState
from ..engine.vouchers import VoucherType
from .actions import MAX_CONSUM, MAX_HAND, MAX_JOKERS, MAX_PACK, MAX_PACK_ITEMS, MAX_SHOP

# Per-card features: rank(13) + suit(4) + enhancement(9) + edition(4)
# + seal(5) + is_debuffed(1) + is_face_down(1). is_face_down is always 0 for now (the
# face-down bosses are deferred) but reserved so the obs schema is stable across the retrain.
_N_ENH, _N_EDITION, _N_SEAL = len(Enhancement), len(Edition), len(Seal)   # 9, 4, 5
_ENH0, _EDITION0, _SEAL0 = 17, 17 + _N_ENH, 17 + _N_ENH + _N_EDITION       # 17, 26, 30
_DEBUFF_I, _FACEDOWN_I = _SEAL0 + _N_SEAL, _SEAL0 + _N_SEAL + 1             # 35, 36
CARD_FEAT = _FACEDOWN_I + 1                                                 # 37
N_PHASES = 5          # PLAYING / WON / LOST / SHOP / OPEN_PACK (one-hot at g[12..16])
# global layout: g[0..11] scalars, g[12..16] phase one-hot, g[17]=#consum, g[18]=consum_slots,
# g[19]=boss-active, g[20]=pending-active, g[21]=pending-max-targets, g[22]=#pack_offers, g[23]=pack_picks
GLOBAL_FEAT = 24
N_LEVELS = 12
N_BOSS = len(BossEffect)          # 29 (NONE + 28); boss one-hot
N_VOUCHER = len(VoucherType)      # 24; vouchers_owned multi-hot (index = VoucherType-1)
CONSUM_VOCAB = 128                # flat consumable id space (kind*32 + type_id), for the agent Embed
PACK_KIND_VOCAB = 8               # PackKind 1..5 (+ 0 pad); small Embed
PACK_SIZE_VOCAB = 4               # PackSize 1..3 (+ 0 pad)
VOUCHER_VOCAB = N_VOUCHER + 1     # VoucherType 1..24 (+ 0 = no offer)


def consum_vocab_id(con) -> int:
    """Flat embedding id for a consumable: kind*32 + type_id (0 = empty slot)."""
    return con.kind * 32 + con.type_id


def _shop_consum_id(offer) -> int:
    """Consum-vocab id for a NON-joker shop offer (0 for a joker offer / empty slot). Uses the
    ConsumableKind the offer becomes when owned (ShopKind and ConsumableKind number differently)."""
    if offer.kind == ShopKind.JOKER:
        return 0
    return SHOP_TO_CONSUMABLE_KIND[offer.kind] * 32 + offer.type_id


OBS_SHAPES: dict[str, tuple] = {
    "global": (GLOBAL_FEAT,),
    "hand": (MAX_HAND, CARD_FEAT),
    "hand_mask": (MAX_HAND,),
    "joker_types": (MAX_JOKERS,),
    "joker_counter": (MAX_JOKERS,),
    "joker_mask": (MAX_JOKERS,),
    "shop_types": (MAX_SHOP,),
    "shop_consum": (MAX_SHOP,),
    "shop_cost": (MAX_SHOP,),
    "shop_mask": (MAX_SHOP,),
    "levels": (N_LEVELS,),
    "deck_rank_hist": (13,),
    "deck_suit_hist": (4,),
    "boss_onehot": (N_BOSS,),
    "consum_types": (MAX_CONSUM,),
    "consum_mask": (MAX_CONSUM,),
    # E5 pack offers (shop) + revealed-item stream (OPEN_PACK)
    "pack_kind": (MAX_PACK,),
    "pack_size": (MAX_PACK,),
    "pack_cost": (MAX_PACK,),
    "pack_offer_mask": (MAX_PACK,),
    "pack_item_joker": (MAX_PACK_ITEMS,),
    "pack_item_consum": (MAX_PACK_ITEMS,),
    "pack_open_mask": (MAX_PACK_ITEMS,),
    # E5 voucher slot + owned set
    "voucher_offer": (1,),
    "voucher_offer_mask": (1,),
    "vouchers_owned": (N_VOUCHER,),
    # E5 pending targeting-Tarot (the consumable awaiting target cards)
    "pending_consum": (1,),
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
        g[12 + phase] = 1.0          # one-hot phase over g[12..16] (incl. OPEN_PACK)
    g[17] = len(state.consumables)
    g[18] = state.consumable_slots
    g[19] = 1.0 if state.boss else 0.0
    # E5 pending targeting two-step: which Tarot is armed + how many cards it wants.
    pending_consum = np.zeros(1, dtype=np.int32)
    if state.pending_consumable >= 0:
        con = state.consumables[state.pending_consumable]
        g[20] = 1.0
        g[21] = max_targets(con)
        pending_consum[0] = consum_vocab_id(con)
    g[22] = len(state.pack_offers)
    g[23] = state.pack_picks

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

    # Shop offers: a JOKER offer encodes its joker-vocab id in shop_types (consum 0); a
    # consumable offer encodes its consum-vocab id in shop_consum (types 0). Cost + mask
    # are set for every offer. The agent now sees both (E5).
    shop_types = np.zeros(MAX_SHOP, dtype=np.int32)
    shop_consum = np.zeros(MAX_SHOP, dtype=np.int32)
    shop_cost = np.zeros(MAX_SHOP, dtype=np.float32)
    shop_mask = np.zeros(MAX_SHOP, dtype=np.float32)
    for i, offer in enumerate(state.shop_offers[:MAX_SHOP]):
        shop_types[i] = int(offer.type_id) if offer.kind == ShopKind.JOKER else 0
        shop_consum[i] = _shop_consum_id(offer)
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

    # E5 booster-pack OFFERS (shop): kind/size embed ids + cost + mask.
    pack_kind = np.zeros(MAX_PACK, dtype=np.int32)
    pack_size = np.zeros(MAX_PACK, dtype=np.int32)
    pack_cost = np.zeros(MAX_PACK, dtype=np.float32)
    pack_offer_mask = np.zeros(MAX_PACK, dtype=np.float32)
    for i, pack in enumerate(state.pack_offers[:MAX_PACK]):
        pack_kind[i] = int(pack.kind)
        pack_size[i] = int(pack.size)
        pack_cost[i] = float(pack.cost)
        pack_offer_mask[i] = 1.0

    # E5 revealed pack ITEMS (OPEN_PACK): a JOKER item -> its joker id; a CONSUMABLE item ->
    # its consum-vocab id. One of the two is the 0 pad per slot, so the net's two embeds compose.
    pack_item_joker = np.zeros(MAX_PACK_ITEMS, dtype=np.int32)
    pack_item_consum = np.zeros(MAX_PACK_ITEMS, dtype=np.int32)
    pack_open_mask = np.zeros(MAX_PACK_ITEMS, dtype=np.float32)
    for i, item in enumerate(state.pack_open[:MAX_PACK_ITEMS]):
        if item.kind == PackItemKind.JOKER:
            pack_item_joker[i] = int(item.payload.type)
        else:
            pack_item_consum[i] = consum_vocab_id(item.payload)
        pack_open_mask[i] = 1.0

    # E5 voucher: the shop's single offer (embed id, 0 = none) + the owned multi-hot.
    voucher_offer = np.asarray([state.voucher_offer], dtype=np.int32)
    voucher_offer_mask = np.asarray([1.0 if state.voucher_offer else 0.0], dtype=np.float32)
    vouchers_owned = np.zeros(N_VOUCHER, dtype=np.float32)
    for v in state.vouchers:
        if 1 <= int(v) <= N_VOUCHER:
            vouchers_owned[int(v) - 1] = 1.0

    return {
        "global": g, "hand": hand, "hand_mask": hand_mask,
        "joker_types": joker_types, "joker_counter": joker_counter, "joker_mask": joker_mask,
        "shop_types": shop_types, "shop_consum": shop_consum,
        "shop_cost": shop_cost, "shop_mask": shop_mask,
        "levels": levels, "deck_rank_hist": deck_rank, "deck_suit_hist": deck_suit,
        "boss_onehot": boss_onehot, "consum_types": consum_types, "consum_mask": consum_mask,
        "pack_kind": pack_kind, "pack_size": pack_size, "pack_cost": pack_cost,
        "pack_offer_mask": pack_offer_mask,
        "pack_item_joker": pack_item_joker, "pack_item_consum": pack_item_consum,
        "pack_open_mask": pack_open_mask,
        "voucher_offer": voucher_offer, "voucher_offer_mask": voucher_offer_mask,
        "vouchers_owned": vouchers_owned, "pending_consum": pending_consum,
    }
