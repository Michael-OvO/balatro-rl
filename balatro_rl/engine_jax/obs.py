"""JAX-native observation encoder and legal mask for the core engine.

``encode_core(state)`` mirrors Python ``envs.obs.encode`` field-for-field. The
joker keys (``joker_types`` / ``joker_counter`` / ``joker_mask``) and the
``global[10]`` joker count are filled from ``state.jokers`` (Phase 2; counters
stay 0 — the loadout is stateless). All shop / consumable / pack / voucher /
boss entries are zeroed out (those systems are out of scope for the JAX core
engine).

``legal_mask_core(state)`` produces a bool[NUM_ACTIONS=708] mask where only the
PLAY ids [0, 218) and DISCARD ids [218, 436) that are legal given the current
hand occupancy are True. Ids >= 436 (shop / pack / voucher) are always False.

Both functions are ``jit``- and ``vmap``-compatible: no Python branching on
traced values; the ``_SUBSETS`` card-index table is a static constant.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from balatro_rl.engine_jax.state import CoreState
from balatro_rl.envs.obs import (
    CARD_FEAT,
    GLOBAL_FEAT,
    N_LEVELS,
    N_PHASES,
    OBS_SHAPES,
    _DEBUFF_I,
    _EDITION0,
    _ENH0,
    _FACEDOWN_I,
    _N_EDITION,
    _N_ENH,
    _N_SEAL,
    _SEAL0,
)
from balatro_rl.envs.actions import MAX_HAND, _SUBSETS, PLAY_N, NUM_ACTIONS

# ---------------------------------------------------------------------------
# Static subset table (built once at import time, never retraced)
# ---------------------------------------------------------------------------

# _SUBSETS is a list of tuples of varying length (1..MAX_SELECT).
# We pad each to MAX_SELECT=5 with a sentinel value (-1 → invalid slot).
# Shape: [218, 5] int32.
MAX_SELECT = 5
_N_SUBSETS = len(_SUBSETS)  # 218

_SUBSET_TABLE: jnp.ndarray = jnp.array(
    [list(s) + [-1] * (MAX_SELECT - len(s)) for s in _SUBSETS],
    dtype=jnp.int32,
)  # [218, 5]

# Length of each subset (number of real card indices). Shape: [218] int32.
_SUBSET_LEN: jnp.ndarray = jnp.array(
    [len(s) for s in _SUBSETS],
    dtype=jnp.int32,
)  # [218]

# ---------------------------------------------------------------------------
# symlog (must match math.copysign(math.log1p(abs(x)), x) exactly)
# ---------------------------------------------------------------------------

def _symlog(x):
    """sign(x) * log1p(|x|) — mirrors Python envs.obs.symlog."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x.astype(jnp.float32)))


# ---------------------------------------------------------------------------
# Card feature vector (37-dim, core plain cards only)
#
# Layout (from envs/obs.py):
#   [0:13]   rank one-hot  (rank 2..14 → index rank-2)
#   [13:17]  suit one-hot  (suit 0..3)
#   [17:26]  enhancement one-hot  (9 values; NONE = index 0)
#   [26:30]  edition one-hot      (4 values; NONE = index 0)
#   [30:35]  seal one-hot         (5 values; NONE = index 0)
#   [35]     is_debuffed          (always 0 for core)
#   [36]     is_face_down         (always 0 for core)
# ---------------------------------------------------------------------------

def _hand_features(hand_rank: jnp.ndarray, hand_suit: jnp.ndarray) -> jnp.ndarray:
    """Build the 8×37 hand feature matrix for core plain cards.

    Args:
        hand_rank: int8[8] rank values (0..12, stored as rank-2).
        hand_suit: int8[8] suit values (0..3).

    Returns:
        float32[8, 37] feature matrix. Slots are filled regardless of mask;
        the caller zeros unoccupied slots via ``hand_mask``.
    """
    # rank one-hot: hand_rank already stores rank-2 (0..12 per state.py comment).
    # Actually CoreState stores rank 2..14 in int8 per the spec; we subtract 2.
    rank_idx = hand_rank.astype(jnp.int32)  # values 2..14
    rank_oh = jnp.zeros((MAX_HAND, 13), dtype=jnp.float32)
    rank_oh = rank_oh.at[jnp.arange(MAX_HAND), rank_idx - 2].set(1.0)

    # suit one-hot
    suit_idx = hand_suit.astype(jnp.int32)  # values 0..3
    suit_oh = jnp.zeros((MAX_HAND, 4), dtype=jnp.float32)
    suit_oh = suit_oh.at[jnp.arange(MAX_HAND), suit_idx].set(1.0)

    # enhancement one-hot: NONE = index 0 (the only possible value for plain cards)
    enh_oh = jnp.zeros((MAX_HAND, _N_ENH), dtype=jnp.float32)
    enh_oh = enh_oh.at[:, 0].set(1.0)  # NONE = 0

    # edition one-hot: NONE = index 0
    edition_oh = jnp.zeros((MAX_HAND, _N_EDITION), dtype=jnp.float32)
    edition_oh = edition_oh.at[:, 0].set(1.0)  # NONE = 0

    # seal one-hot: NONE = index 0
    seal_oh = jnp.zeros((MAX_HAND, _N_SEAL), dtype=jnp.float32)
    seal_oh = seal_oh.at[:, 0].set(1.0)  # NONE = 0

    # debuff + facedown: always 0 for core (no boss, no face-down mechanic)
    extras = jnp.zeros((MAX_HAND, 2), dtype=jnp.float32)

    # Concatenate: [13 + 4 + 9 + 4 + 5 + 2] = 37
    return jnp.concatenate([rank_oh, suit_oh, enh_oh, edition_oh, seal_oh, extras], axis=1)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

def encode_core(state: CoreState) -> dict:
    """Encode a CoreState into the OBS_SHAPES dict.

    Core fields (global, hand, hand_mask, levels, deck_rank_hist, deck_suit_hist)
    are computed from ``state``.  Joker fields come from ``state.jokers``
    (types + occupancy mask + count in global[10]; counters stay 0 — stateless).
    All shop / consumable / pack / voucher / boss fields are zeroed to the
    correct shape and dtype.

    The output dict has EXACTLY the same keys and shapes as ``envs.obs.OBS_SHAPES``.
    """
    # -- Global vector [24] ---------------------------------------------------
    rs = state.round_score.astype(jnp.float32)
    req = state.required.astype(jnp.float32)

    g = jnp.zeros(GLOBAL_FEAT, dtype=jnp.float32)
    g = g.at[0].set(_symlog(rs))
    g = g.at[1].set(_symlog(req))
    g = g.at[2].set(jnp.minimum(rs / jnp.maximum(req, 1.0), 2.0))
    g = g.at[3].set(state.hands_left.astype(jnp.float32))
    g = g.at[4].set(state.discards_left.astype(jnp.float32))
    g = g.at[5].set(_symlog(state.money.astype(jnp.float32)))
    g = g.at[6].set(state.ante.astype(jnp.float32))
    g = g.at[7].set(state.blind_index.astype(jnp.float32))
    # g[8] = rerolls_done = 0 (not tracked in core)
    g = g.at[9].set(state.hand_size.astype(jnp.float32))
    # -- Jokers (Phase 2): types from the loadout, counter=0 (stateless), mask from occupancy.
    jt = state.jokers.astype(jnp.int32)                       # [MAX_JOKERS]
    jmask = (jt != 0).astype(jnp.float32)
    g = g.at[10].set(jnp.sum(jt != 0).astype(jnp.float32))    # g[10] = #jokers
    # g[11] = len(shop_offers) = 0
    # Phase one-hot over g[12..16]: phase values 0..4 map to g[12+phase].
    # In core the engine only reaches PLAYING(0), WON(1), LOST(2); SHOP/OPEN_PACK
    # never occur. Guard: clamp phase to [0, N_PHASES) before indexing.
    phase = state.phase.astype(jnp.int32)
    phase_clamped = jnp.clip(phase, 0, N_PHASES - 1)
    phase_oh = jnp.zeros(N_PHASES, dtype=jnp.float32).at[phase_clamped].set(1.0)
    g = g.at[12:12 + N_PHASES].set(phase_oh)
    # g[17] = len(consumables) = 0
    # g[18] = consumable_slots = 2 (Python default; Crystal Ball voucher is out of scope)
    g = g.at[18].set(2.0)
    # g[19..23] = boss / pending / pack fields = 0

    # -- Hand [8, 37] + hand_mask [8] ----------------------------------------
    hand = _hand_features(state.hand_rank, state.hand_suit)
    # Zero out unoccupied slots (hand_mask == False → 0 features).
    mask_f = state.hand_mask.astype(jnp.float32)[:, None]  # [8,1] broadcast
    hand = hand * mask_f

    hand_mask = state.hand_mask.astype(jnp.float32)  # [8] float32

    # -- Levels [12] ----------------------------------------------------------
    levels = state.levels.astype(jnp.float32)  # [12]

    # -- Deck histograms [13] / [4] --------------------------------------------
    # Python encode() counts only state.deck (undrawn cards). CoreState stores the
    # full 52-card draw order; cards from deck_ptr onward are undrawn. We replicate
    # this by masking out already-drawn slots (slot index < deck_ptr) before summing.
    slot_idx = jnp.arange(52, dtype=jnp.int32)
    undrawn_mask = slot_idx >= state.deck_ptr  # [52] bool — True = undrawn

    rank_oh = jax.nn.one_hot(
        state.deck_rank.astype(jnp.int32) - 2, 13, dtype=jnp.float32,
    )  # [52, 13]
    deck_rank_hist = jnp.sum(rank_oh * undrawn_mask[:, None], axis=0)  # [13]

    suit_oh = jax.nn.one_hot(
        state.deck_suit.astype(jnp.int32), 4, dtype=jnp.float32,
    )  # [52, 4]
    deck_suit_hist = jnp.sum(suit_oh * undrawn_mask[:, None], axis=0)  # [4]

    # -- All out-of-scope arrays zeroed to correct shape / dtype --------------
    from balatro_rl.envs.obs import N_BOSS, N_VOUCHER
    from balatro_rl.envs.actions import MAX_JOKERS, MAX_SHOP, MAX_CONSUM, MAX_PACK, MAX_PACK_ITEMS

    return {
        "global":           g,
        "hand":             hand,
        "hand_mask":        hand_mask,
        "joker_types":      jt,
        "joker_counter":    jnp.zeros(MAX_JOKERS, dtype=jnp.float32),
        "joker_mask":       jmask,
        "shop_types":       jnp.zeros(MAX_SHOP, dtype=jnp.int32),
        "shop_consum":      jnp.zeros(MAX_SHOP, dtype=jnp.int32),
        "shop_cost":        jnp.zeros(MAX_SHOP, dtype=jnp.float32),
        "shop_mask":        jnp.zeros(MAX_SHOP, dtype=jnp.float32),
        "levels":           levels,
        "deck_rank_hist":   deck_rank_hist,
        "deck_suit_hist":   deck_suit_hist,
        # boss_onehot: Python does if 0 <= state.boss < N_BOSS: boss_onehot[state.boss]=1.0.
        # In the core engine boss is always BossEffect.NONE=0, so index 0 is always set.
        "boss_onehot":      jnp.zeros(N_BOSS, dtype=jnp.float32).at[0].set(1.0),
        "consum_types":     jnp.zeros(MAX_CONSUM, dtype=jnp.int32),
        "consum_mask":      jnp.zeros(MAX_CONSUM, dtype=jnp.float32),
        "pack_kind":        jnp.zeros(MAX_PACK, dtype=jnp.int32),
        "pack_size":        jnp.zeros(MAX_PACK, dtype=jnp.int32),
        "pack_cost":        jnp.zeros(MAX_PACK, dtype=jnp.float32),
        "pack_offer_mask":  jnp.zeros(MAX_PACK, dtype=jnp.float32),
        "pack_item_joker":  jnp.zeros(MAX_PACK_ITEMS, dtype=jnp.int32),
        "pack_item_consum": jnp.zeros(MAX_PACK_ITEMS, dtype=jnp.int32),
        "pack_open_mask":   jnp.zeros(MAX_PACK_ITEMS, dtype=jnp.float32),
        "voucher_offer":    jnp.zeros(1, dtype=jnp.int32),
        "voucher_offer_mask": jnp.zeros(1, dtype=jnp.float32),
        "vouchers_owned":   jnp.zeros(N_VOUCHER, dtype=jnp.float32),
        "pending_consum":   jnp.zeros(1, dtype=jnp.int32),
    }


# ---------------------------------------------------------------------------
# Legal mask
# ---------------------------------------------------------------------------

def legal_mask_core(state: CoreState) -> jnp.ndarray:
    """Return a bool[NUM_ACTIONS=708] legal-action mask for a CoreState.

    PLAY ids [0, 218): subset k is legal iff all card indices in _SUBSETS[k]
        are < hand_count (i.e. the slot is occupied).
    DISCARD ids [218, 436): same card-index check, plus discards_left > 0.
    Ids [436, 708): always False (shop / pack / voucher — out of JAX core scope).

    Uses the static _SUBSET_TABLE [218, 5] padded with -1 for unused slots.
    A slot index of -1 is treated as valid (it is a pad, not a real index), so
    legality is determined only for the first ``_SUBSET_LEN[k]`` entries.
    """
    hand_count = jnp.sum(state.hand_mask.astype(jnp.int32))  # scalar

    # For each subset k (218 total) and each position p (0..4):
    # real_slot[k, p] = True iff p < len(_SUBSETS[k]) (i.e. not padding).
    # card_ok[k, p]   = True iff _SUBSET_TABLE[k, p] < hand_count (or is pad).
    # subset_legal[k] = all real slots have card_ok.

    # _SUBSET_TABLE: [218, 5] int32 (padding = -1)
    # _SUBSET_LEN:   [218]    int32

    # Build a [218, 5] boolean mask: True where this is a real (non-pad) slot.
    pos_range = jnp.arange(MAX_SELECT, dtype=jnp.int32)  # [5]
    real_slot = pos_range[None, :] < _SUBSET_LEN[:, None]  # [218, 5] bool

    # For each real slot, check index < hand_count.
    idx_ok = _SUBSET_TABLE < hand_count  # [218, 5] bool (pad slots: -1 < hand_count → False)

    # A subset is legal iff ALL real slots have idx_ok.
    # Equivalently: no real slot has idx_ok==False, i.e. all(idx_ok | ~real_slot).
    all_ok = jnp.all(idx_ok | ~real_slot, axis=1)  # [218] bool

    # PLAY ids [0, 218): legal = all_ok AND phase == PLAYING (0).
    # In a done state, hands_left would be 0, but hands_left isn't directly
    # used here — in practice the Python oracle also allows play even at 0 hands
    # (the engine raises; legal_mask doesn't filter). We match Python: phase check only.
    # Python legal_mask delegates to legal_actions(state) which checks phase internally.
    # For core: PLAYING=0. The parity test only checks non-terminal PLAYING states.
    is_playing = (state.phase == 0)  # scalar bool

    play_legal = all_ok & is_playing  # [218] bool

    # DISCARD ids [218, 436): same subset check + discards_left > 0.
    has_discards = state.discards_left > 0  # scalar bool
    discard_legal = all_ok & is_playing & has_discards  # [218] bool

    # Concatenate: [218 play | 218 discard | 272 zeros]
    shop_zeros = jnp.zeros(NUM_ACTIONS - 2 * PLAY_N, dtype=jnp.bool_)
    return jnp.concatenate([play_legal, discard_legal, shop_zeros])
