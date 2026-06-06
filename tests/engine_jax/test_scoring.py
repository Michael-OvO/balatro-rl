"""Tests for branchless base scoring (Task 1.2).

``score_core(ranks, suits, mask, levels) -> (hand_type, chips, mult, score)``
reproduces the no-joker / plain-card path of the Python oracle
``balatro_rl.engine.scoring.score_play(played, jokers=(), levels=...)``.

Step 1 pins exact-value hands. Step 2 runs a seeded 500-play parity sweep over a
standard 52-card deck (no duplicates), VARYING the per-HandType level so the
level math (HAND_BASE + HAND_INC * (level - 1)) is exercised. We compare the
full ``(hand_type, chips, mult, score)`` integer tuple against the oracle.
"""
import random

import jax
import jax.numpy as jnp

from balatro_rl.engine.cards import Card, standard_deck
from balatro_rl.engine.hands import HandType
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine_jax.scoring import score_core

N_HAND_TYPES = 12


def S(cards, levels=None):
    """Run score_core on a list of (rank, suit), padding to 5 slots.

    Returns a 4-tuple of Python ints (hand_type, chips, mult, score).
    """
    r = [c[0] for c in cards]
    s = [c[1] for c in cards]
    m = [True] * len(cards)
    while len(r) < 5:
        r.append(0)
        s.append(0)
        m.append(False)
    if levels is None:
        levels = [1] * N_HAND_TYPES
    ht, chips, mult, score = score_core(
        jnp.array(r, jnp.int8),
        jnp.array(s, jnp.int8),
        jnp.array(m, bool),
        jnp.array(levels, jnp.int32),
    )
    return int(ht), int(chips), int(mult), int(score)


# --- Step 1: explicit exact-value hands -------------------------------------

def test_pair_of_fives_lvl1():
    # PAIR (ht=1) lvl1: base_chips 10 + (5+5)=20, mult 2 -> score 40
    ht, chips, mult, score = S([(5, 0), (5, 1)])
    assert (ht, chips, mult, score) == (1, 20, 2, 40)


def test_two_pair_lvl1():
    # TWO_PAIR (ht=2) lvl1: base 20 + (5+5+9+9=28)=48, mult 2 -> 96
    ht, chips, mult, score = S([(5, 0), (5, 1), (9, 2), (9, 3), (2, 0)])
    assert (ht, chips, mult, score) == (2, 48, 2, 96)


def test_high_card_lvl1():
    # HIGH_CARD (ht=0) lvl1: base 5 + only the highest valid card scores.
    # Highest here is K(13) -> chip 10. chips 5+10=15, mult 1 -> 15.
    ht, chips, mult, score = S([(2, 0), (7, 1), (9, 2), (11, 3), (13, 0)])
    assert (ht, chips, mult, score) == (0, 15, 1, 15)


def test_high_card_ace_chip_11():
    # Ace high-card: chip value 11. base 5 + 11 = 16, mult 1.
    ht, chips, mult, score = S([(14, 0)])
    assert (ht, chips, mult, score) == (0, 16, 1, 16)


def test_three_of_a_kind_lvl1():
    # THREE (ht=3) lvl1: base 30 + (5+5+5=15)=45, mult 3 -> 135. Kicker excluded.
    ht, chips, mult, score = S([(5, 0), (5, 1), (5, 2), (9, 3), (2, 0)])
    assert (ht, chips, mult, score) == (3, 45, 3, 135)


def test_four_of_a_kind_kicker_excluded():
    # FOUR (ht=7) lvl1: base 60 + (5*4=20)=80, mult 7 -> 560. The 9 kicker excluded.
    ht, chips, mult, score = S([(5, 0), (5, 1), (5, 2), (5, 3), (9, 0)])
    assert (ht, chips, mult, score) == (7, 80, 7, 560)


def test_full_house_all_score():
    # FULL_HOUSE (ht=6) lvl1: base 40 + (5+5+5+9+9=33)=73, mult 4 -> 292.
    ht, chips, mult, score = S([(5, 0), (5, 1), (5, 2), (9, 3), (9, 0)])
    assert (ht, chips, mult, score) == (6, 73, 4, 292)


def test_flush_all_score():
    # FLUSH (ht=5) lvl1: base 35 + chips(2,5,9,J->10,K->10)=2+5+9+10+10=36 -> 71, mult 4 -> 284.
    ht, chips, mult, score = S([(2, 0), (5, 0), (9, 0), (11, 0), (13, 0)])
    assert (ht, chips, mult, score) == (5, 71, 4, 284)


def test_straight_all_score():
    # STRAIGHT (ht=4) lvl1: base 30 + (5+6+7+8+9=35)=65, mult 4 -> 260.
    ht, chips, mult, score = S([(5, 0), (6, 1), (7, 2), (8, 3), (9, 0)])
    assert (ht, chips, mult, score) == (4, 65, 4, 260)


def test_straight_flush_all_score():
    # STRAIGHT_FLUSH (ht=8) lvl1: base 100 + (5+6+7+8+9=35)=135, mult 8 -> 1080.
    ht, chips, mult, score = S([(5, 0), (6, 0), (7, 0), (8, 0), (9, 0)])
    assert (ht, chips, mult, score) == (8, 135, 8, 1080)


def test_pair_level_3():
    # PAIR lvl3: base_chips 10 + 15*(3-1) = 40; + (5+5)=10 cards = 50.
    # mult 2 + 1*(3-1) = 4. score 50*4 = 200.
    levels = [1] * N_HAND_TYPES
    levels[HandType.PAIR] = 3
    ht, chips, mult, score = S([(5, 0), (5, 1)], levels)
    assert (ht, chips, mult, score) == (1, 50, 4, 200)


# --- Step 2: seeded randomized parity vs the oracle (varying levels) --------

def _oracle(cards, levels):
    res = score_play([Card(c[0], c[1]) for c in cards], jokers=(),
                     levels=tuple(levels))
    return int(res.hand_type), int(res.chips), int(res.mult), int(res.score)


def test_parity_vs_score_play_500_varying_levels():
    rng = random.Random(0)
    deck = standard_deck()
    mismatches = []
    for _ in range(500):
        cards = [(c.rank, c.suit) for c in rng.sample(deck, 5)]
        # Random per-HandType levels in 1..5 so level math is exercised.
        levels = [rng.randint(1, 5) for _ in range(N_HAND_TYPES)]
        want = _oracle(cards, levels)
        got = S(cards, levels)
        if want != got:
            mismatches.append((cards, levels, want, got))
    assert not mismatches, f"{len(mismatches)} mismatches, first few: {mismatches[:5]}"


def test_parity_vs_score_play_variable_size_varying_levels():
    rng = random.Random(1)
    deck = standard_deck()
    mismatches = []
    for _ in range(500):
        n = rng.randint(1, 5)
        cards = [(c.rank, c.suit) for c in rng.sample(deck, n)]
        levels = [rng.randint(1, 5) for _ in range(N_HAND_TYPES)]
        want = _oracle(cards, levels)
        got = S(cards, levels)
        if want != got:
            mismatches.append((cards, levels, want, got))
    assert not mismatches, f"{len(mismatches)} mismatches, first few: {mismatches[:5]}"


# --- jit / vmap-ability -----------------------------------------------------

def test_jit_vmap():
    ranks = jnp.array([[5, 5, 9, 9, 2], [2, 5, 9, 11, 13]], jnp.int8)
    suits = jnp.array([[0, 1, 2, 3, 0], [0, 0, 0, 0, 0]], jnp.int8)
    mask = jnp.ones((2, 5), bool)
    levels = jnp.ones((2, N_HAND_TYPES), jnp.int32)
    f = jax.jit(jax.vmap(score_core))
    ht, chips, mult, score = f(ranks, suits, mask, levels)
    # row 0: TWO_PAIR 96; row 1: FLUSH 284
    assert int(score[0]) == 96
    assert int(score[1]) == 284
