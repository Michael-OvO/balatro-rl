# tests/engine/jokers/test_integration.py
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


def test_additive_left_of_xmult_beats_reverse_order():
    # Joker(+4) then Cavendish(x3) vs Cavendish(x3) then Joker(+4), High-card Ace (16 chips, base mult 1).
    add_then_x = score_play([C(14), C(7), C(2)], jokers=(J(JokerType.JOKER), J(JokerType.CAVENDISH)))
    x_then_add = score_play([C(14), C(7), C(2)], jokers=(J(JokerType.CAVENDISH), J(JokerType.JOKER)))
    assert add_then_x.mult == (1 + 4) * 3          # 15
    assert x_then_add.mult == (1 * 3) + 4          # 7
    assert add_then_x.score > x_then_add.score


def test_stacked_jokers_end_to_end():
    # Greedy + Scary Face + Joker on a Pair of K♦.
    # base (10,2)+20 chips=30; Scary +30*2=+60 -> chips 90; Greedy +3*2=+6, Joker +4 -> mult 12.
    res = score_play([C(13, 3), C(13, 3), C(3, 0), C(7, 0), C(9, 0)],
                     jokers=(J(JokerType.GREEDY), J(JokerType.SCARY_FACE), J(JokerType.JOKER)))
    assert res.chips == 90 and res.mult == 12.0 and res.score == 90 * 12
