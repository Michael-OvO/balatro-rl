"""JAX engine step functions: reset (and future step/play/discard).

Only ``reset`` is implemented here for Task 0.4.  Future tasks will add
``step``, ``play``, and ``discard``.

Note: ``reset`` accepts host-provided deck ordering (ranks + suits as length-52
sequences) and a host-computed ``required`` score so that the JAX engine can be
seeded byte-identically to the Python oracle.  Real JAX-native seeding/shuffling
will be added in Task 1.7; for now the PRNG key is fixed at [0, 0].
"""
from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from balatro_rl.engine_jax.config import (
    DISCARDS_PER_BLIND,
    HANDS_PER_BLIND,
    MAX_HAND,
    N_HAND_TYPES,
    Phase,
    STARTING_MONEY,
)
from balatro_rl.engine_jax.state import DECK_SIZE, CoreState


def reset(
    deck_rank: Sequence[int],
    deck_suit: Sequence[int],
    required: int,
    scale_unused: float = 1.0,
) -> CoreState:
    """Build an initial CoreState from a host-provided 52-card draw order.

    Parameters
    ----------
    deck_rank:
        Length-52 sequence of rank values (int, 2..14) in full draw order:
        positions 0..7 are the cards dealt to the opening hand; 8..51 are the
        remaining draw pile (front = next draw).  Must match the Python oracle's
        ``deck_from_python(gs)`` output to achieve parity.
    deck_suit:
        Length-52 sequence of suit values (int, 0..3), same ordering.
    required:
        Required score to beat the first blind, host-computed via
        ``engine.required_score(1, 0, scale)``.  Stored directly.
    scale_unused:
        Curriculum scale parameter (not used here; kept for API symmetry with
        Python's ``reset(seed, scale, ...)``.  The host computes ``required``
        externally, so this value does not affect state.

    Returns
    -------
    CoreState
        Fully initialised game state for ante=1, small blind, opening hand
        already dealt.
    """
    # Convert host sequences to fixed-dtype JAX arrays.
    dr = jnp.asarray(deck_rank, dtype=jnp.int8)   # shape (52,)
    ds = jnp.asarray(deck_suit, dtype=jnp.int8)   # shape (52,)

    # The first MAX_HAND (8) cards are the opening hand.
    hand_rank = dr[:MAX_HAND]
    hand_suit = ds[:MAX_HAND]
    hand_mask = jnp.ones((MAX_HAND,), dtype=bool)

    # deck_ptr points to the next undrawn card (slot 8 after dealing the hand).
    deck_ptr = jnp.array(MAX_HAND, dtype=jnp.int32)

    # Python oracle initialises levels to all-ones (1-based, level 1 = base).
    # Confirmed from engine.py line: levels=tuple([1] * 12)
    levels = jnp.ones((N_HAND_TYPES,), dtype=jnp.int32)

    # Play-count trackers start at zero.
    hand_plays_run   = jnp.zeros((N_HAND_TYPES,), dtype=jnp.int32)
    hand_plays_round = jnp.zeros((N_HAND_TYPES,), dtype=jnp.int32)

    # PRNG key: fixed placeholder; real seeding added in Task 1.7.
    rng = jnp.array([0, 0], dtype=jnp.uint32)

    return CoreState(
        deck_rank=dr,
        deck_suit=ds,
        deck_ptr=deck_ptr,

        hand_rank=hand_rank,
        hand_suit=hand_suit,
        hand_mask=hand_mask,

        ante=jnp.array(1, dtype=jnp.int32),
        blind_index=jnp.array(0, dtype=jnp.int32),
        round_score=jnp.array(0, dtype=jnp.int32),
        required=jnp.array(required, dtype=jnp.int32),
        hands_left=jnp.array(HANDS_PER_BLIND, dtype=jnp.int32),
        discards_left=jnp.array(DISCARDS_PER_BLIND, dtype=jnp.int32),
        hand_size=jnp.array(MAX_HAND, dtype=jnp.int32),

        money=jnp.array(STARTING_MONEY, dtype=jnp.int32),

        levels=levels,
        hand_plays_run=hand_plays_run,
        hand_plays_round=hand_plays_round,

        phase=jnp.array(Phase.PLAYING, dtype=jnp.int32),
        done=jnp.array(False, dtype=bool),
        won=jnp.array(False, dtype=bool),

        rng=rng,
    )
