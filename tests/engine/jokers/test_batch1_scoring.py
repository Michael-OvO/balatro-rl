"""Batch 1 jokers: suit on-scored, hand-type indep (+Mult/+Chips), Half, and
per-card on-scored effects (Fibonacci, Even Steven, Odd Todd, Scholar,
Walkie Talkie, Smiley Face). Values per docs/reference/jokers.md."""
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401  (registers jokers)


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


# --- suit on-scored: +3 Mult per scored card of the suit ---

def test_lusty_per_heart():
    # Pair of Kings, both Hearts (suit 1). base (10,2)+20chips=30, mult 2.
    # Lusty +3 per scored ♥ -> +6 -> mult 8 -> 240.  # wiki: /w/Lusty_Joker
    res = score_play([C(13, 1), C(13, 1), C(3, 0), C(7, 2), C(9, 3)],
                     jokers=(J(JokerType.LUSTY),))
    assert res.mult == 8.0 and res.score == 30 * 8


def test_wrathful_per_spade():
    # Pair of Kings, both Spades (suit 0). +6 mult -> 8 -> 240.  # wiki: /w/Wrathful_Joker
    res = score_play([C(13, 0), C(13, 0), C(3, 1), C(7, 2), C(9, 3)],
                     jokers=(J(JokerType.WRATHFUL),))
    assert res.mult == 8.0 and res.score == 30 * 8


def test_gluttonous_per_club():
    # Pair of Kings, both Clubs (suit 2). +6 mult -> 8 -> 240.  # wiki: /w/Gluttonous_Joker
    res = score_play([C(13, 2), C(13, 2), C(3, 0), C(7, 1), C(9, 3)],
                     jokers=(J(JokerType.GLUTTONOUS),))
    assert res.mult == 8.0 and res.score == 30 * 8


def test_suit_jokers_ignore_other_suits():
    # No hearts scored -> Lusty adds nothing.
    res = score_play([C(13, 0), C(13, 0), C(3, 2), C(7, 2), C(9, 3)],
                     jokers=(J(JokerType.LUSTY),))
    assert res.mult == 2.0


# --- hand-type +Mult (independent) ---

def test_jolly_pair():
    # Pair of 3s contains a Pair. base (10,2) 16 chips, +8 mult -> 10 -> 160.  # wiki: /w/Jolly_Joker
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.JOLLY),))
    assert res.mult == 10.0 and res.score == 160


def test_jolly_no_pair_no_bonus():
    res = score_play([C(14, 0), C(7, 1), C(2, 2)], jokers=(J(JokerType.JOLLY),))
    assert res.mult == 1.0


def test_zany_three_of_a_kind():
    # Trip 3s: base (30,3) chips=30+9=39, mult 3. +12 -> 15 -> 585.  # wiki: /w/Zany_Joker
    res = score_play([C(3, 0), C(3, 1), C(3, 2), C(7, 3), C(9, 0)],
                     jokers=(J(JokerType.ZANY),))
    assert res.mult == 15.0


def test_mad_two_pair():
    # Two Pair 3s/7s: base (20,2) chips=20+3+3+7+7=40, mult 2. +10 -> 12.  # wiki: /w/Mad_Joker
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(7, 3), C(9, 0)],
                     jokers=(J(JokerType.MAD),))
    assert res.mult == 12.0


def test_crazy_straight():
    # Straight 3-7: base (30,4) mult 4. +12 -> 16.  # wiki: /w/Crazy_Joker
    res = score_play([C(3, 0), C(4, 1), C(5, 2), C(6, 3), C(7, 0)],
                     jokers=(J(JokerType.CRAZY),))
    assert res.mult == 16.0


def test_droll_flush():
    # Flush (all ♥): base (35,4) mult 4. +10 -> 14.  # wiki: /w/Droll_Joker
    res = score_play([C(3, 1), C(7, 1), C(9, 1), C(11, 1), C(13, 1)],
                     jokers=(J(JokerType.DROLL),))
    assert res.mult == 14.0


# --- hand-type +Chips (independent) ---

def test_sly_pair_chips():
    # Pair of 3s: 16 chips +50 -> 66, mult 2 -> 132.  # wiki: /w/Sly_Joker
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.SLY),))
    assert res.chips == 66 and res.score == 132


def test_wily_three_of_a_kind_chips():
    # Trip 3s: base chips 39 + 100 -> 139.  # wiki: /w/Wily_Joker
    res = score_play([C(3, 0), C(3, 1), C(3, 2), C(7, 3), C(9, 0)],
                     jokers=(J(JokerType.WILY),))
    assert res.chips == 139


def test_clever_two_pair_chips():
    # Two Pair 3s/7s: base chips 40 + 80 -> 120.  # wiki: /w/Clever_Joker
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(7, 3), C(9, 0)],
                     jokers=(J(JokerType.CLEVER),))
    assert res.chips == 120


def test_devious_straight_chips():
    # Straight 3-7: base chips 30 + 3+4+5+6+7 = 55, +100 -> 155.  # wiki: /w/Devious_Joker
    res = score_play([C(3, 0), C(4, 1), C(5, 2), C(6, 3), C(7, 0)],
                     jokers=(J(JokerType.DEVIOUS),))
    assert res.chips == 155


def test_crafty_flush_chips():
    # Flush ♥ 3,7,9,J,K: base chips 35 + 3+7+9+10+10 = 74, +80 -> 154.  # wiki: /w/Crafty_Joker
    res = score_play([C(3, 1), C(7, 1), C(9, 1), C(11, 1), C(13, 1)],
                     jokers=(J(JokerType.CRAFTY),))
    assert res.chips == 154


def test_half_joker_three_or_fewer():
    # High card Ace (1 card path uses 3 cards here): <=3 cards -> +20 mult.
    # High card Ace: base (5,1) chips 16, mult 1 +20 -> 21.  # wiki: /w/Half_Joker
    res = score_play([C(14, 0), C(7, 1), C(2, 2)], jokers=(J(JokerType.HALF),))
    assert res.mult == 21.0


def test_half_joker_no_bonus_with_four_cards():
    res = score_play([C(14, 0), C(7, 1), C(2, 2), C(5, 3)], jokers=(J(JokerType.HALF),))
    assert res.mult == 1.0


# --- per-card on-scored ---

def test_fibonacci_counts_specific_ranks():
    # Pair of 3s (3 in {A,2,3,5,8}); kickers 7,9,2. 2 is also Fibonacci.
    # Scored cards (pair only) = the two 3s -> +8 each = +16 mult.  # wiki: /w/Fibonacci
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(11, 0)],
                     jokers=(J(JokerType.FIBONACCI),))
    assert res.mult == 2.0 + 16.0


def test_even_steven_even_ranks():
    # Pair of 4s: both even -> +4 each = +8 mult.  # wiki: /w/Even_Steven
    res = score_play([C(4, 0), C(4, 1), C(7, 2), C(9, 3), C(11, 0)],
                     jokers=(J(JokerType.EVEN_STEVEN),))
    assert res.mult == 2.0 + 8.0


def test_odd_todd_ace_counts_odd():
    # Pair of Aces (rank 14 counts ODD for Odd Todd): +31 chips each = +62.
    # base pair (10,2) chips = 10 + 11 + 11 = 32, +62 -> 94.  # wiki: /w/Odd_Todd
    res = score_play([C(14, 0), C(14, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.ODD_TODD),))
    assert res.chips == 32 + 62


def test_odd_todd_skips_even_and_faces():
    # Pair of Kings (face, neither even nor odd-triggering) -> no chips.
    res = score_play([C(13, 0), C(13, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.ODD_TODD),))
    # Only base chips: 10 + 10 + 10 = 30.
    assert res.chips == 30


def test_scholar_aces():
    # Pair of Aces: +20 chips and +4 mult each = +40 chips, +8 mult.
    # base chips 32 + 40 -> 72, mult 2 + 8 -> 10.  # wiki: /w/Scholar
    res = score_play([C(14, 0), C(14, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.SCHOLAR),))
    assert res.chips == 72 and res.mult == 10.0


def test_walkie_talkie_tens_and_fours():
    # Pair of 10s: +10 chips +4 mult each = +20 chips, +8 mult.
    # base pair chips = 10 + 10 + 10 = 30, +20 -> 50, mult 2 + 8 -> 10.  # wiki: /w/Walkie_Talkie
    res = score_play([C(10, 0), C(10, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.WALKIE_TALKIE),))
    assert res.chips == 50 and res.mult == 10.0


def test_smiley_face_per_face():
    # Pair of Kings (both face): +5 mult each = +10 -> mult 12.  # wiki: /w/Smiley_Face
    res = score_play([C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)],
                     jokers=(J(JokerType.SMILEY_FACE),))
    assert res.mult == 12.0
