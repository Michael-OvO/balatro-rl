import importlib
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401  (registers jokers)


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


def test_joker_plus_4_mult():
    # High-card Ace: 16 chips, base mult 1, +4 -> mult 5, score 80.  # wiki: /w/Joker
    res = score_play([C(14), C(7), C(2)], jokers=(J(JokerType.JOKER),))
    assert res.mult == 5.0 and res.score == 80


def test_cavendish_x3():
    # High-card Ace: 16 chips, mult 1 -> x3 = 3, score 48.  # wiki: /w/Cavendish
    res = score_play([C(14), C(7), C(2)], jokers=(J(JokerType.CAVENDISH),))
    assert res.mult == 3.0 and res.score == 48


def test_greedy_per_diamond():
    # Pair of Kings (♦=suit3): two kings are ♦. base (10,2)+20 chips=30.
    # Greedy +3 per scored ♦; both kings score -> +6 mult -> mult 8 -> 240.  # wiki: /w/Greedy_Joker
    res = score_play([C(13, 3), C(13, 3), C(3, 0), C(7, 0), C(9, 1)],
                     jokers=(J(JokerType.GREEDY),))
    assert res.mult == 8.0 and res.score == 30 * 8


def test_scary_face_per_face():
    # Pair of Kings: both kings are face -> +30 chips each = +60. chips 30+60=90.  # wiki: /w/Scary_Face
    # Use mixed suits on kickers to avoid unintended flush.
    res = score_play([C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)],
                     jokers=(J(JokerType.SCARY_FACE),))
    assert res.chips == 90 and res.score == 90 * 2


def test_photograph_first_face_only():
    # Two kings score; Photograph x2 applies to the FIRST scoring face card only.
    # base (10,2)+20 chips=30 chips, mult 2 -> x2 once = 4 -> 120.  # wiki: /w/Photograph
    # Use mixed suits on kickers to avoid unintended flush.
    res = score_play([C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)],
                     jokers=(J(JokerType.PHOTOGRAPH),))
    assert res.mult == 4.0 and res.score == 30 * 4
