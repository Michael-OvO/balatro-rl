"""Branchless JAX joker scoring (Phase 2).

`score_with_jokers` reproduces engine.scoring.score_play's ordered fold for the
~45 pure-scoring jokers on plain cards. Dispatch is four lax.switch branch tables
(on_score / on_held / independent / retrigger), one tiny pure branch per in-scope
joker, indexed by a dense id (0 = empty/out-of-scope = no-op). With an all-zero
loadout the fold reduces exactly to scoring.score_core.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as _np
from jax import lax
from typing import NamedTuple

from balatro_rl.engine_jax.config import MAX_HAND, MAX_SELECT
from balatro_rl.engine_jax.scoring import (
    detect_hand_type, _card_chip, _scoring_mask, _N_RANK_BUCKETS,
    HAND_BASE_CHIPS, HAND_BASE_MULT, HAND_INC_CHIPS, HAND_INC_MULT,
)
from balatro_rl.envs.actions import MAX_JOKERS

F0 = jnp.float32(0.0)
F1 = jnp.float32(1.0)
I0 = jnp.int32(0)
JOKER_SLOTS = MAX_JOKERS - 1  # real cap (Antimatter +1 slot excluded); empty_joker_slots = JOKER_SLOTS - n_jokers

# --- in-scope JokerType ids (== engine.jokers.base.JokerType values) ----------
# Dense index 0 is the no-op; in-scope ids get dense indices 1..N in THIS order.
INSCOPE_IDS = (
    1,                              # JOKER
    2, 3, 4, 5, 119,                # suit +mult: Greedy Lusty Wrathful Gluttonous Onyx
    118,                            # suit +chips: Arrowhead
    33, 104, 78,                    # face: Scary Smiley Photograph
    31, 39, 40, 41, 101,            # rank: Fibonacci EvenSteven OddTodd Scholar WalkieTalkie
    36, 109,                        # retrigger: Hack SockAndBuskin
    6, 7, 8, 9, 10,                 # contains +mult: Jolly Zany Mad Crazy Droll
    11, 12, 13, 14, 15,             # contains +chips: Sly Wily Clever Devious Crafty
    131, 132, 133, 134, 135,        # contains xmult: Duo Trio Family Order Tribe
    16, 22, 23, 34, 17, 93, 53, 43, 62,  # context: Half Banner Mystic Abstract Stencil Bull Blue Supernova CardSharp
    128, 122,                       # scoring-suit-set xmult: SeeingDouble FlowerPot
    72, 48,                         # held: Baron(on_held) Blackboard(independent)
    52, 37,                         # rule flags: Splash Pareidolia (no effect branch)
)
N_INSCOPE = len(INSCOPE_IDS)        # 47
SPLASH_ID = 52
PAREIDOLIA_ID = 37

_MAX_ID = max(INSCOPE_IDS) + 1
_dense_np = _np.zeros(_MAX_ID, dtype=_np.int32)
for _d, _jid in enumerate(INSCOPE_IDS, start=1):
    _dense_np[_jid] = _d
DENSE_MAP = jnp.asarray(_dense_np)   # id -> dense index (0 for empty/unknown)


def _dense(jid):
    """Map a JokerType id (clamped to the table) to its dense index, 0 if absent."""
    jid = jnp.clip(jnp.asarray(jid, jnp.int32), 0, _MAX_ID - 1)
    return DENSE_MAP[jid]


# --- independent-branch context (read-only aggregates) ------------------------
class IndepCtx(NamedTuple):
    contains_pair: jnp.ndarray
    contains_two_pair: jnp.ndarray
    contains_trip: jnp.ndarray
    contains_quad: jnp.ndarray
    contains_straight: jnp.ndarray
    contains_flush: jnp.ndarray
    has_club_and_other: jnp.ndarray
    all_four_suits: jnp.ndarray
    all_dark: jnp.ndarray
    n_jokers: jnp.ndarray
    empty_slots: jnp.ndarray
    money: jnp.ndarray
    discards_left: jnp.ndarray
    deck_count: jnp.ndarray
    plays_run_ht: jnp.ndarray
    plays_round_ht: jnp.ndarray
    played_count: jnp.ndarray


# --- branch tables (populated in Tasks 2.3-2.5; all no-op for now) -------------
def _noop_score(r, s, f, ff):   return (I0, F0, F1)
def _noop_held(r, s, f):        return (I0, F0, F1)
def _noop_indep(c):             return (I0, F0, F1)
def _noop_retrig(r, s, f):      return I0

# Each table has N_INSCOPE + 1 entries (index 0 = no-op). Tasks 2.3-2.5 replace
# the no-ops at the dense indices they implement.
ON_SCORE_BRANCHES = [_noop_score] * (N_INSCOPE + 1)
ON_HELD_BRANCHES  = [_noop_held]  * (N_INSCOPE + 1)
INDEP_BRANCHES    = [_noop_indep] * (N_INSCOPE + 1)
RETRIG_BRANCHES   = [_noop_retrig] * (N_INSCOPE + 1)


def _set(table, jid, fn):
    table[_dense_np[jid]] = fn


def _contains_predicates(p_rank, p_suit, p_mask):
    """contains_* over the PLAYED cards (mirror engine.hands.contains)."""
    m = p_mask.astype(jnp.int32)
    n = jnp.sum(m)
    rb = p_rank.astype(jnp.int32) - 2
    oh = (rb[:, None] == jnp.arange(_N_RANK_BUCKETS)[None, :]).astype(jnp.int32) * m[:, None]
    rc = jnp.sum(oh, axis=0)                       # rank counts[13]
    n_ge2 = jnp.sum(rc >= 2)
    has_trip = jnp.any(rc >= 3)
    has_quad = jnp.any(rc >= 4)
    contains_pair = n_ge2 >= 1
    contains_two_pair = n_ge2 >= 2                 # full house -> trip rank + pair rank = 2
    # straight / flush reuse detect_hand_type's logic via a cheap recompute.
    so = (p_suit.astype(jnp.int32)[:, None] == jnp.arange(4)[None, :]).astype(jnp.int32) * m[:, None]
    suit_counts = jnp.sum(so, axis=0)
    is_flush = (n == 5) & (jnp.max(suit_counts) == 5)
    present = rc > 0
    ace = present[12]
    low_pad = jnp.concatenate([ace[None], present]).astype(jnp.int32)
    windows = jnp.stack([low_pad[i:i + 5] for i in range(low_pad.shape[0] - 4)], axis=0)
    has_run5 = jnp.any(jnp.sum(windows, axis=1) == 5)
    is_straight = (n == 5) & (jnp.sum(present) == 5) & has_run5
    return (contains_pair, contains_two_pair, has_trip, has_quad, is_straight, is_flush)


def score_with_jokers(p_rank, p_suit, p_mask, h_rank, h_suit, h_mask, levels, jokers,
                      *, money, discards_left, deck_count, hand_plays_run, hand_plays_round):
    """Ordered fold matching score_play (plain cards). Returns (hand_type, chips, mult, score)."""
    p_rank = jnp.asarray(p_rank, jnp.int32); p_suit = jnp.asarray(p_suit, jnp.int32)
    p_mask = jnp.asarray(p_mask, jnp.bool_)
    h_rank = jnp.asarray(h_rank, jnp.int32); h_suit = jnp.asarray(h_suit, jnp.int32)
    h_mask = jnp.asarray(h_mask, jnp.bool_)
    levels = jnp.asarray(levels, jnp.int32)
    jokers = jnp.asarray(jokers, jnp.int32)

    ht = detect_hand_type(p_rank, p_suit, p_mask)
    splash = jnp.any(jokers == SPLASH_ID)
    all_face = jnp.any(jokers == PAREIDOLIA_ID)

    base_sm = _scoring_mask(ht, p_rank, p_mask)
    scoring_mask = jnp.where(splash, p_mask, base_sm)

    lvl = levels[ht]
    chips = (HAND_BASE_CHIPS[ht] + HAND_INC_CHIPS[ht] * (lvl - 1)).astype(jnp.int32)
    mult = (HAND_BASE_MULT[ht] + HAND_INC_MULT[ht] * (lvl - 1)).astype(jnp.float32)

    face_played = ((p_rank == 11) | (p_rank == 12) | (p_rank == 13) | all_face)
    face_held = ((h_rank == 11) | (h_rank == 12) | (h_rank == 13) | all_face)

    # first scoring face slot (played-slot index), else -1
    sf = scoring_mask & face_played
    first_face_idx = jnp.where(jnp.any(sf), jnp.argmax(sf.astype(jnp.int32)), jnp.int32(-1))

    # independent-branch aggregates
    (c_pair, c_two, c_trip, c_quad, c_str, c_flush) = _contains_predicates(p_rank, p_suit, p_mask)
    sc_suits = scoring_mask[:, None] & (p_suit[:, None] == jnp.arange(4)[None, :])
    suit_present = jnp.any(sc_suits, axis=0)         # bool[4] among scoring cards
    has_club_and_other = suit_present[2] & jnp.any(suit_present & (jnp.arange(4) != 2))
    all_four = jnp.all(suit_present)
    held_dark = (~h_mask) | (h_suit == 0) | (h_suit == 2)
    all_dark = jnp.all(held_dark)
    n_jokers = jnp.sum(jokers != 0)
    empty_slots = jnp.maximum(0, JOKER_SLOTS - n_jokers)
    idx = IndepCtx(c_pair, c_two, c_trip, c_quad, c_str, c_flush,
                   has_club_and_other, all_four, all_dark, n_jokers, empty_slots,
                   jnp.asarray(money, jnp.int32), jnp.asarray(discards_left, jnp.int32),
                   jnp.asarray(deck_count, jnp.int32),
                   hand_plays_run[ht], hand_plays_round[ht], jnp.sum(p_mask))

    dense_slots = _dense(jokers)                      # int32[MAX_JOKERS]

    # ---- Phase A: scored cards L->R, with retriggers ----
    for i in range(MAX_SELECT):
        r = p_rank[i]; s = p_suit[i]; f = face_played[i]
        in_scoring = scoring_mask[i]
        ff = (jnp.int32(i) == first_face_idx)
        retrig = I0
        for slot in range(MAX_JOKERS):
            retrig = retrig + lax.switch(dense_slots[slot], RETRIG_BRANCHES, r, s, f)
        for pk in range(1 + MAX_JOKERS):             # static unroll bound
            active = in_scoring & (jnp.int32(pk) < (1 + retrig))
            chips = chips + jnp.where(active, _card_chip(r), 0)
            for slot in range(MAX_JOKERS):
                dc, dm, xm = lax.switch(dense_slots[slot], ON_SCORE_BRANCHES, r, s, f, ff)
                chips = chips + jnp.where(active, dc, 0)
                mult = mult + jnp.where(active, dm, F0)
                mult = mult * jnp.where(active, xm, F1)

    # ---- Phase B: held cards ----
    for j in range(MAX_HAND):
        r = h_rank[j]; s = h_suit[j]; f = face_held[j]
        occ = h_mask[j]
        for slot in range(MAX_JOKERS):
            dc, dm, xm = lax.switch(dense_slots[slot], ON_HELD_BRANCHES, r, s, f)
            chips = chips + jnp.where(occ, dc, 0)
            mult = mult + jnp.where(occ, dm, F0)
            mult = mult * jnp.where(occ, xm, F1)

    # ---- Phase C: independent jokers, slot order ----
    for slot in range(MAX_JOKERS):
        dc, dm, xm = lax.switch(dense_slots[slot], INDEP_BRANCHES, idx)
        chips = chips + dc
        mult = mult + dm
        mult = mult * xm

    score = jnp.floor(chips.astype(jnp.float32) * mult).astype(jnp.int32)
    return ht.astype(jnp.int32), chips.astype(jnp.int32), mult.astype(jnp.float32), score
