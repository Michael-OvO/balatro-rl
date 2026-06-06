"""Tests for branchless detect_hand_type (Task 1.1).

The Python oracle is balatro_rl.engine.hands.evaluate(): its returned HandType
is the spec. Step 1 pins explicit hands; Step 2 runs a seeded 500-hand parity
sweep drawn from a standard 52-card deck (no duplicates) against the oracle.
"""
import random

import jax.numpy as jnp

from balatro_rl.engine.cards import Card, standard_deck
from balatro_rl.engine.hands import evaluate
from balatro_rl.engine_jax.scoring import detect_hand_type


def H(cards):
    """Classify a list of (rank, suit) via detect_hand_type, padding to 5 slots."""
    r = [c[0] for c in cards]
    s = [c[1] for c in cards]
    m = [True] * len(cards)
    while len(r) < 5:
        r.append(0)
        s.append(0)
        m.append(False)
    return int(
        detect_hand_type(
            jnp.array(r, jnp.int8),
            jnp.array(s, jnp.int8),
            jnp.array(m, bool),
        )
    )


# --- Step 1: explicit hands -------------------------------------------------

def test_pair():
    assert H([(5, 0), (5, 1)]) == 1  # PAIR


def test_flush():
    assert H([(2, 0), (5, 0), (9, 0), (11, 0), (13, 0)]) == 5  # FLUSH


def test_straight():
    assert H([(5, 0), (6, 1), (7, 2), (8, 3), (9, 0)]) == 4  # STRAIGHT


def test_straight_flush():
    assert H([(5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]) == 8


def test_wheel_straight():
    assert H([(14, 0), (2, 1), (3, 2), (4, 3), (5, 0)]) == 4  # A-2-3-4-5


def test_high_card():
    assert H([(2, 0), (7, 1), (9, 2), (11, 3), (13, 0)]) == 0


def test_ace_high_straight():
    assert H([(10, 0), (11, 1), (12, 2), (13, 3), (14, 0)]) == 4  # 10-J-Q-K-A


def test_two_pair():
    assert H([(5, 0), (5, 1), (9, 2), (9, 3), (2, 0)]) == 2  # TWO_PAIR


def test_three_of_a_kind():
    assert H([(5, 0), (5, 1), (5, 2), (9, 3), (2, 0)]) == 3


def test_full_house():
    assert H([(5, 0), (5, 1), (5, 2), (9, 3), (9, 0)]) == 6  # FULL_HOUSE


def test_four_of_a_kind():
    assert H([(5, 0), (5, 1), (5, 2), (5, 3), (9, 0)]) == 7


def test_single_card():
    assert H([(14, 0)]) == 0  # HIGH_CARD


def test_pair_three_cards():
    assert H([(5, 0), (5, 1), (9, 2)]) == 1  # PAIR (n<5 cannot be straight)


def test_four_cards_no_straight():
    # 4 distinct cards: span 4 but only 4 cards -> oracle requires n==5 for straight
    assert H([(5, 0), (6, 1), (7, 2), (8, 3)]) == 0  # HIGH_CARD


def test_wheel_flush():
    assert H([(14, 0), (2, 0), (3, 0), (4, 0), (5, 0)]) == 8  # STRAIGHT_FLUSH wheel


def test_flush_not_straight():
    # span > 4 distinct, same suit -> FLUSH, not straight flush
    assert H([(2, 0), (4, 0), (6, 0), (8, 0), (14, 0)]) == 5


def test_almost_straight_pair_blocks():
    # 4 distinct ranks (one paired) cannot be a straight (needs 5 distinct)
    assert H([(5, 0), (6, 1), (7, 2), (8, 3), (8, 0)]) == 1  # PAIR


# --- Step 2: seeded randomized parity vs the oracle -------------------------

def test_parity_vs_oracle_500():
    rng = random.Random(0)
    deck = standard_deck()
    mismatches = []
    for _ in range(500):
        cards = rng.sample(deck, 5)  # 5 distinct cards from a real 52-card deck
        want = int(evaluate(cards)[0])
        got = H([(c.rank, c.suit) for c in cards])
        if want != got:
            mismatches.append(([(c.rank, c.suit) for c in cards], want, got))
    assert not mismatches, f"{len(mismatches)} mismatches, first few: {mismatches[:5]}"


def test_parity_vs_oracle_variable_size():
    """Also sweep 1..5 card hands (oracle accepts 1..5)."""
    rng = random.Random(1)
    deck = standard_deck()
    mismatches = []
    for _ in range(500):
        n = rng.randint(1, 5)
        cards = rng.sample(deck, n)
        want = int(evaluate(cards)[0])
        got = H([(c.rank, c.suit) for c in cards])
        if want != got:
            mismatches.append(([(c.rank, c.suit) for c in cards], want, got))
    assert not mismatches, f"{len(mismatches)} mismatches, first few: {mismatches[:5]}"
