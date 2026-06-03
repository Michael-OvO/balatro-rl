"""Phase D1a: level-aware scoring. Planet cards raise a hand type's level; a hand at level
L scores HAND_BASE + HAND_LEVEL_INC*(L-1). Byte-identical at level 1 (all hands start there),
so this is a no-op foundation until Planets (raise) and The Arm (lower) move levels.

Per-level increments verified against balatrowiki.org/w/Planet_Cards.
"""
import dataclasses

from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import reset, step, Verb
from balatro_rl.engine.hands import HandType, HAND_BASE, HAND_LEVEL_INC, leveled_base
from balatro_rl.engine.scoring import score_play


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def _levels(**by_name):
    lv = [1] * 12
    for name, val in by_name.items():
        lv[int(HandType[name])] = val
    return tuple(lv)


# ============================================================================
# leveled_base helper
# ============================================================================

def test_leveled_base_level_1_is_hand_base():
    for ht in HandType:
        assert leveled_base(ht, _levels()) == HAND_BASE[ht]
        assert leveled_base(ht, ()) == HAND_BASE[ht]          # empty -> level 1


def test_leveled_base_applies_increment_per_level():
    # Pair base (10,2); +15 chips +1 mult per level. Level 3 -> (10+30, 2+2) = (40, 4).
    assert leveled_base(HandType.PAIR, _levels(PAIR=3)) == (40, 4)
    # Flush base (35,4); +15/+2 per level. Level 2 -> (50, 6).
    assert leveled_base(HandType.FLUSH, _levels(FLUSH=2)) == (50, 6)


def test_every_hand_type_has_a_level_increment():
    assert set(HAND_LEVEL_INC) == set(HandType)


# ============================================================================
# score_play uses levels
# ============================================================================

def test_score_play_level_1_byte_identical():
    hand = [C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)]   # pair of Kings
    assert score_play(hand).score == score_play(hand, levels=_levels()).score


def test_score_play_scales_with_level():
    hand = [C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)]   # pair of Kings, both score
    # Level 1: base (10,2) + K + K = chips 10+10+10=30, mult 2 -> 60.
    assert score_play(hand).score == 60
    # Level 3: base (40,4); chips 40+10+10=60, mult 4 -> 240.
    res = score_play(hand, levels=_levels(PAIR=3))
    assert res.chips == 40 + 10 + 10 and res.mult == 4.0 and res.score == 240


def test_level_scaling_through_engine():
    st = dataclasses.replace(reset(seed=0), levels=_levels(PAIR=2),
                             hand=(C(13, 0), C(13, 1), C(5, 2), C(7, 3), C(9, 0)),
                             required=10_000_000)
    _nxt, info = step(st, (Verb.PLAY, (0, 1)))
    # Pair level 2: base (25,3); chips 25+10+10=45, mult 3 -> 135.
    assert info["score"] == 135


def test_flint_halves_the_leveled_base():
    # Level 3 Pair base (40,4) under Flint -> (20, 2). chips 20+10+10=40, mult 2 -> 80.
    hand = [C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)]
    res = score_play(hand, levels=_levels(PAIR=3), flint=True)
    assert res.chips == 20 + 10 + 10 and res.mult == 2.0
