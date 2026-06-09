"""CoreState pytree for the JAX-native Balatro engine.

``CoreState`` is a ``typing.NamedTuple``, which JAX automatically treats as a
pytree — no manual ``register_pytree_node`` needed.  Every field is a
fixed-shape JAX array; scalar fields use shape ``()`` so that ``jax.vmap``
adds a leading batch dimension uniformly.
"""
from typing import NamedTuple

import jax.numpy as jnp

from balatro_rl.engine_jax.config import MAX_HAND, N_HAND_TYPES

# Full standard deck size (never changes).
DECK_SIZE: int = 52


class CoreState(NamedTuple):
    """Immutable game state container.

    Field order mirrors the spec exactly so that downstream code using
    positional destructuring is unambiguous.
    """

    # -- Deck ------------------------------------------------------------------
    deck_rank: jnp.ndarray          # int8[52]  rank of each deck slot (0-12)
    deck_suit: jnp.ndarray          # int8[52]  suit of each deck slot (0-3)
    deck_ptr:  jnp.ndarray          # int32[]   next draw position

    # -- Current hand ----------------------------------------------------------
    hand_rank: jnp.ndarray          # int8[8]   rank of each held card
    hand_suit: jnp.ndarray          # int8[8]   suit of each held card
    hand_mask: jnp.ndarray          # bool[8]   True = slot occupied

    # -- Blind / run progress --------------------------------------------------
    ante:         jnp.ndarray       # int32[]   current ante (1-based)
    blind_index:  jnp.ndarray       # int32[]   0=Small, 1=Big, 2=Boss
    round_score:  jnp.ndarray       # int32[]   chips scored this round
    required:     jnp.ndarray       # int32[]   chips required to beat blind
    hands_left:   jnp.ndarray       # int32[]   hands remaining this blind
    discards_left: jnp.ndarray      # int32[]   discards remaining this blind
    hand_size:    jnp.ndarray       # int32[]   current draw-up-to size
    required_table: jnp.ndarray     # int32[9,3] required[ante, blind_index]; ante 1..8, row 0 unused

    # -- Economy ---------------------------------------------------------------
    money: jnp.ndarray              # int32[]   current dollars

    # -- Hand-type levelling ---------------------------------------------------
    levels:          jnp.ndarray    # int32[12] level of each hand type (1-based)
    hand_plays_run:  jnp.ndarray    # int32[12] times played this run
    hand_plays_round: jnp.ndarray   # int32[12] times played this round

    # -- Episode bookkeeping ---------------------------------------------------
    phase: jnp.ndarray              # int32[]   Phase enum value
    done:  jnp.ndarray              # bool[]
    won:   jnp.ndarray              # bool[]

    # -- RNG state -------------------------------------------------------------
    rng: jnp.ndarray                # uint32[2] JAX PRNG key


def zeros_state() -> CoreState:
    """Return a ``CoreState`` with all fields zeroed to their correct dtype/shape.

    Intended as a canonical blank slate for ``jax.vmap`` stacking, shape
    assertions in tests, and engine reset functions that fill fields in-place.
    """
    return CoreState(
        deck_rank=jnp.zeros((DECK_SIZE,), dtype=jnp.int8),
        deck_suit=jnp.zeros((DECK_SIZE,), dtype=jnp.int8),
        deck_ptr=jnp.zeros((), dtype=jnp.int32),

        hand_rank=jnp.zeros((MAX_HAND,), dtype=jnp.int8),
        hand_suit=jnp.zeros((MAX_HAND,), dtype=jnp.int8),
        hand_mask=jnp.zeros((MAX_HAND,), dtype=bool),

        ante=jnp.zeros((), dtype=jnp.int32),
        blind_index=jnp.zeros((), dtype=jnp.int32),
        round_score=jnp.zeros((), dtype=jnp.int32),
        required=jnp.zeros((), dtype=jnp.int32),
        hands_left=jnp.zeros((), dtype=jnp.int32),
        discards_left=jnp.zeros((), dtype=jnp.int32),
        hand_size=jnp.zeros((), dtype=jnp.int32),
        required_table=jnp.zeros((9, 3), dtype=jnp.int32),

        money=jnp.zeros((), dtype=jnp.int32),

        levels=jnp.zeros((N_HAND_TYPES,), dtype=jnp.int32),
        hand_plays_run=jnp.zeros((N_HAND_TYPES,), dtype=jnp.int32),
        hand_plays_round=jnp.zeros((N_HAND_TYPES,), dtype=jnp.int32),

        phase=jnp.zeros((), dtype=jnp.int32),
        done=jnp.zeros((), dtype=bool),
        won=jnp.zeros((), dtype=bool),

        rng=jnp.zeros((2,), dtype=jnp.uint32),
    )
