"""Parity harness utilities: extract comparable core fields from Python GameState
and JAX CoreState into plain dicts, and helpers to seed / compare them.

Draw-order convention (traced from engine.py):
  _draw(hand, deck, hand_size) does:
      drawn = deck[:need]          # <- front-slice, NOT pop() from the end
      return hand + drawn, deck[need:]

  reset() calls:
      deck, rng = rng.shuffle(list(master_deck))  # full 52-card shuffle
      hand, deck = _draw([], deck, HAND_SIZE)      # hand = shuffled[0:8]
      # gs.hand  = cards at positions [0..7]  of the shuffled order
      # gs.deck  = cards at positions [8..51] of the shuffled order

  Step refills (PLAY / DISCARD) call:
      hand, deck = _draw(remaining, list(state.deck), target)
      # i.e. next draws come from the FRONT of gs.deck, in gs.deck order.

Full 52-card draw order (index 0 = first card ever drawn):
  draw_order[0:8]  = gs.hand  (in the order they appear in the hand tuple)
  draw_order[8:52] = gs.deck  (front-to-back, same order Python draws from)

A JAX engine that keeps deck_rank/deck_suit as a flat 52-entry ring buffer
(deck_ptr pointing to the next undrawn slot) must be seeded with this array
so that its position-0 matches Python's first-drawn card.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from balatro_rl.engine.state import GameState
    from balatro_rl.engine_jax.state import CoreState


# ---------------------------------------------------------------------------
# Python GameState extractor
# ---------------------------------------------------------------------------

def python_core_fields(gs: "GameState") -> dict:
    """Extract comparable scalar + hand fields from a Python GameState.

    Returns a dict with:
      ante, blind_index, round_score, required, hands_left, discards_left,
      hand_size, money, phase (int), done (bool), won (bool),
      hand (sorted list of (rank, suit) tuples),
      levels (tuple of ints).

    The hand is sorted so callers can do set-equality without worrying about
    slot ordering.
    """
    hand = sorted(
        (int(c.rank), int(c.suit)) for c in gs.hand
    )
    return {
        "ante":          int(gs.ante),
        "blind_index":   int(gs.blind_index),
        "round_score":   int(gs.round_score),
        "required":      int(gs.required),
        "hands_left":    int(gs.hands_left),
        "discards_left": int(gs.discards_left),
        "hand_size":     int(gs.hand_size),
        "money":         int(gs.money),
        "phase":         int(gs.phase),
        "done":          bool(gs.done),
        "won":           bool(gs.won),
        "hand":          hand,
        "levels":        tuple(int(x) for x in gs.levels),
    }


# ---------------------------------------------------------------------------
# JAX CoreState extractor
# ---------------------------------------------------------------------------

def jax_core_fields(cs: "CoreState") -> dict:
    """Extract comparable scalar + hand fields from a JAX CoreState.

    Mirrors python_core_fields exactly — same keys, same sort order for hand.
    Only cards where hand_mask is True are included in the hand list.
    JAX scalars (shape-() arrays) are converted to Python ints/bools.
    """
    import numpy as np

    # Extract held cards (mask selects occupied slots).
    mask = np.asarray(cs.hand_mask, dtype=bool)
    ranks = np.asarray(cs.hand_rank, dtype=int)
    suits = np.asarray(cs.hand_suit, dtype=int)
    hand = sorted(
        (int(ranks[i]), int(suits[i]))
        for i in range(len(mask))
        if mask[i]
    )

    levels = tuple(int(x) for x in np.asarray(cs.levels))

    return {
        "ante":          int(cs.ante),
        "blind_index":   int(cs.blind_index),
        "round_score":   int(cs.round_score),
        "required":      int(cs.required),
        "hands_left":    int(cs.hands_left),
        "discards_left": int(cs.discards_left),
        "hand_size":     int(cs.hand_size),
        "money":         int(cs.money),
        "phase":         int(cs.phase),
        "done":          bool(cs.done),
        "won":           bool(cs.won),
        "hand":          hand,
        "levels":        levels,
    }


# ---------------------------------------------------------------------------
# Deck reconstruction
# ---------------------------------------------------------------------------

def deck_from_python(gs: "GameState"):
    """Return the full 52-card draw order from a post-reset Python GameState.

    Draw direction (from engine.py _draw and reset):
      _draw does `drawn = deck[:need]` (front-slice).  reset() calls
      `hand, deck = _draw([], shuffled_deck, HAND_SIZE)`, so:
        gs.hand  = shuffled[0:8]   (drawn first, front of original order)
        gs.deck  = shuffled[8:52]  (remaining, front = next to be drawn)

    Reconstruction:
      draw_order[0:8]  = gs.hand   (hand tuple order = draw order within hand)
      draw_order[8:52] = gs.deck   (front-to-back = Python's next-draw order)

    Returns:
      ranks: list[int]  length 52, rank values 2..14
      suits: list[int]  length 52, suit values 0..3
    """
    cards = list(gs.hand) + list(gs.deck)
    assert len(cards) == 52, (
        f"Expected 52 cards total, got {len(cards)} "
        f"(hand={len(gs.hand)}, deck={len(gs.deck)})"
    )
    ranks = [int(c.rank) for c in cards]
    suits = [int(c.suit) for c in cards]
    return ranks, suits


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def assert_states_equal(py: dict, jx: dict) -> None:
    """Assert that two core-field dicts are equal.

    Checks every scalar key first (for a specific error message), then the
    hand multiset and levels tuple. Raises AssertionError naming the first
    mismatching field.
    """
    scalar_keys = [
        "ante", "blind_index", "round_score", "required",
        "hands_left", "discards_left", "hand_size", "money",
        "phase", "done", "won",
    ]
    for key in scalar_keys:
        py_val = py[key]
        jx_val = jx[key]
        assert py_val == jx_val, (
            f"Field '{key}' mismatch: Python={py_val!r}, JAX={jx_val!r}"
        )

    if py["levels"] != jx["levels"]:
        raise AssertionError(
            f"Field 'levels' mismatch:\n  Python={py['levels']}\n  JAX={jx['levels']}"
        )

    if py["hand"] != jx["hand"]:
        raise AssertionError(
            f"Field 'hand' (sorted) mismatch:\n  Python={py['hand']}\n  JAX={jx['hand']}"
        )
