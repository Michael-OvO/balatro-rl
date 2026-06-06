"""Scoring kernels for the JAX engine.

Task 1.1: ``detect_hand_type`` — a branchless, jit/vmap-able poker-hand
classifier that matches the Python oracle ``balatro_rl.engine.hands.evaluate``
bit-for-bit on the played-card subset (1..5 cards).

Card encoding (same everywhere in the engine):
    rank: int8 in 2..14 (Ace = 14), suit: int8 in 0..3, empty slot rank 0.
A ``mask`` of bool[5] marks which of the 5 slots hold a real played card.

HandType ordering (mirrors HandType IntEnum in engine/hands.py):
    0 HIGH_CARD  1 PAIR  2 TWO_PAIR  3 THREE_OF_A_KIND  4 STRAIGHT
    5 FLUSH  6 FULL_HOUSE  7 FOUR_OF_A_KIND  8 STRAIGHT_FLUSH
    9 FIVE_OF_A_KIND  10 FLUSH_HOUSE  11 FLUSH_FIVE

Oracle rules mirrored (see engine/hands.py::evaluate / _is_straight / _is_flush):
  * counts = sorted(rank_counts.values(), reverse=True); the if/elif ladder is
    reproduced here as a highest-wins jnp.where chain.
  * Straight: requires n == 5 AND exactly 5 distinct ranks present, with either
    a span of 4 (max-min == 4) OR the Ace-low wheel {2,3,4,5,14}. The wheel is
    handled by duplicating the Ace as a virtual low card and scanning for any
    run of 5 consecutive present ranks.
  * Flush: requires n == 5 AND all 5 cards share one suit (max suit-count == 5).
  * FIVE_OF_A_KIND / FLUSH_HOUSE / FLUSH_FIVE are 5-card-only by construction
    (they need a 5-count or a 3+2 split, both of which imply n == 5).

This module deliberately uses no enhancement/stone/wild semantics: those are
absent from the standard 52-card deck the JAX engine seeds from, and the oracle
reduces to plain rank/suit comparisons in that regime.
"""
from __future__ import annotations

import jax.numpy as jnp

# HandType integer codes (kept local so this kernel has no Python-enum import
# on the hot path; values match engine/hands.py::HandType).
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8
FIVE_OF_A_KIND = 9
FLUSH_HOUSE = 10
FLUSH_FIVE = 11

# Number of rank buckets: ranks 2..14 -> 13 buckets (rank r -> bucket r-2).
_N_RANK_BUCKETS = 13


def detect_hand_type(ranks, suits, mask) -> jnp.ndarray:
    """Classify up to 5 played cards into a HandType code (int32 scalar).

    Args:
        ranks: int8[5], rank per slot (2..14; 0 for empty).
        suits: int8[5], suit per slot (0..3; ignored where mask is False).
        mask:  bool[5], True for slots holding a real played card.

    Returns:
        int32 scalar HandType code in 0..11, matching the oracle's ``evaluate``.

    Branchless: no Python control flow over traced values; jit- and vmap-able.
    """
    ranks = jnp.asarray(ranks).astype(jnp.int32)
    suits = jnp.asarray(suits).astype(jnp.int32)
    m = jnp.asarray(mask).astype(jnp.int32)

    n = jnp.sum(m)  # number of valid played cards

    # --- rank histogram over 13 buckets (rank 2 -> 0 ... Ace 14 -> 12) -------
    # One-hot each slot's rank bucket, zero out masked-off slots, sum over slots.
    rank_bucket = ranks - 2  # may be -2 for empty (rank 0); masked out below.
    rank_oh = (rank_bucket[:, None] == jnp.arange(_N_RANK_BUCKETS)[None, :])
    rank_oh = rank_oh.astype(jnp.int32) * m[:, None]
    rank_counts = jnp.sum(rank_oh, axis=0)  # int32[13]

    # --- suit histogram over 4 buckets --------------------------------------
    suit_oh = (suits[:, None] == jnp.arange(4)[None, :])
    suit_oh = suit_oh.astype(jnp.int32) * m[:, None]
    suit_counts = jnp.sum(suit_oh, axis=0)  # int32[4]

    # --- count-derived predicates (mirror sorted(counts, reverse=True)) ------
    max_count = jnp.max(rank_counts)            # counts[0]
    distinct = jnp.sum(rank_counts > 0)         # number of distinct ranks present
    num_count2 = jnp.sum(rank_counts == 2)      # ranks appearing exactly twice
    num_count3 = jnp.sum(rank_counts == 3)      # ranks appearing exactly thrice

    is_five_count = max_count == 5                          # counts == [5]
    is_quad = max_count == 4                                # counts[0] == 4
    # counts == [3, 2]: exactly two distinct ranks, one a trip, the other a pair.
    is_full = (max_count == 3) & (num_count3 == 1) & (num_count2 == 1) & (distinct == 2)
    is_trip = max_count == 3                                # counts[0] == 3
    is_two_pair = (max_count == 2) & (num_count2 >= 2)      # counts[:2] == [2, 2]
    is_pair = max_count == 2                                # counts[0] == 2

    # --- flush: n == 5 and all five cards share one suit --------------------
    is_flush = (n == 5) & (jnp.max(suit_counts) == 5)

    # --- straight: n == 5, exactly 5 distinct ranks, run of 5 consecutive ---
    # Presence over ranks 2..14 (index r-2). Build a 14-slot vector where the
    # Ace (bucket 12) is ALSO mirrored to a virtual low position so the wheel
    # A-2-3-4-5 forms a run. Layout of `low_pad` (length 14):
    #   index 0      -> virtual Ace-low (present iff Ace present)
    #   index 1..13  -> ranks 2..14 (bucket 0..12)
    present = rank_counts > 0                               # bool[13]
    ace_present = present[12]                               # rank 14 -> bucket 12
    low_pad = jnp.concatenate([ace_present[None], present])  # bool[14]
    low_pad = low_pad.astype(jnp.int32)
    # Any window of 5 consecutive present slots? Sliding sum of width 5.
    windows = jnp.stack([low_pad[i:i + 5] for i in range(low_pad.shape[0] - 4)], axis=0)
    has_run5 = jnp.any(jnp.sum(windows, axis=1) == 5)
    is_straight = (n == 5) & (distinct == 5) & has_run5

    # --- first-match-wins selection (mirror evaluate()'s if/elif ladder) ----
    # The oracle's ladder, in priority order (first match wins), is:
    #   FLUSH_FIVE > FLUSH_HOUSE > FIVE_OF_A_KIND > STRAIGHT_FLUSH >
    #   FOUR_OF_A_KIND > FULL_HOUSE > FLUSH > STRAIGHT > THREE_OF_A_KIND >
    #   TWO_PAIR > PAIR > HIGH_CARD.
    # We reproduce it with a where-chain built LOWEST priority first: each later
    # jnp.where overwrites earlier ones, so the highest-priority true predicate
    # is the one that survives. (Order here is the EXACT reverse of the ladder.)
    ht = jnp.int32(HIGH_CARD)
    ht = jnp.where(is_pair, PAIR, ht)
    ht = jnp.where(is_two_pair, TWO_PAIR, ht)
    ht = jnp.where(is_trip, THREE_OF_A_KIND, ht)
    ht = jnp.where(is_straight, STRAIGHT, ht)
    ht = jnp.where(is_flush, FLUSH, ht)
    ht = jnp.where(is_full, FULL_HOUSE, ht)
    ht = jnp.where(is_quad, FOUR_OF_A_KIND, ht)
    ht = jnp.where(is_flush & is_straight, STRAIGHT_FLUSH, ht)
    ht = jnp.where(is_five_count, FIVE_OF_A_KIND, ht)
    ht = jnp.where(is_flush & is_full, FLUSH_HOUSE, ht)
    ht = jnp.where(is_flush & is_five_count, FLUSH_FIVE, ht)

    return ht.astype(jnp.int32)
