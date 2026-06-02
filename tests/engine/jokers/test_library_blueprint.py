# tests/engine/jokers/test_library_blueprint.py
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


def test_blueprint_copies_joker_to_right():
    # Blueprint (slot0) left of Joker(+4). Both give +4 -> +8 mult.
    # High-card Ace: 16 chips, mult 1+8=9 -> 144.  # wiki: /w/Blueprint
    jokers = (J(JokerType.BLUEPRINT), J(JokerType.JOKER))
    res = score_play([C(14), C(7), C(2)], jokers=jokers)
    assert res.mult == 9.0 and res.score == 16 * 9


def test_blueprint_copies_greedy():
    # Blueprint + Greedy: each scored ♦ gives +3 twice. Pair of K♦: two ♦ -> +12 mult.
    jokers = (J(JokerType.BLUEPRINT), J(JokerType.GREEDY))
    res = score_play([C(13, 3), C(13, 3), C(3, 0), C(7, 0), C(9, 0)], jokers=jokers)
    # base (10,2)+20 chips=30; mult 2 + (2 cards x 2 jokers x 3) = 2+12 = 14 -> 420.
    assert res.mult == 14.0 and res.score == 420


def test_blueprint_cannot_copy_pareidolia():
    # Blueprint right-neighbor is Pareidolia (copyable=False) -> Blueprint does nothing,
    # but real Pareidolia still applies its rule. Scary Face hits all via real Pareidolia only.
    # Use mixed suits on kickers to avoid unintended flush.
    jokers = (J(JokerType.BLUEPRINT), J(JokerType.PAREIDOLIA), J(JokerType.SCARY_FACE))
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=jokers)
    # Pareidolia active once -> both 3s "face" -> Scary +30*2. chips 10+6+60=76, mult 2 -> 152.
    assert res.chips == 76 and res.score == 152


def test_blueprint_rightmost_does_nothing():
    # Blueprint with no right neighbor contributes nothing. High-card Ace -> 16.
    res = score_play([C(14), C(7), C(2)], jokers=(J(JokerType.JOKER), J(JokerType.BLUEPRINT)))
    assert res.mult == 5.0 and res.score == 80   # only the real Joker's +4
