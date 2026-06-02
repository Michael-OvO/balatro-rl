from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState, REGISTRY
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


def test_splash_makes_nonscoring_diamonds_trigger_greedy():
    # Pair of 3s + three ♦ kickers. Without Splash only the 3s score.
    # With Splash all 5 score; Greedy +3 per ♦. Three ♦ kickers + (3s are ♠) -> +9 mult.
    cards = [C(3, 0), C(3, 0), C(7, 3), C(9, 3), C(2, 3)]  # three diamonds among kickers
    res = score_play(cards, jokers=(J(JokerType.SPLASH), J(JokerType.GREEDY)))
    # chips: base 10 + (3+3+7+9+2)=34 -> 34.  mult: 2 + 3*3 = 11.  score 34*11=374.
    assert res.chips == 34 and res.mult == 11.0 and res.score == 374


def test_pareidolia_makes_scary_face_hit_all_cards():
    # Pair of 3s; Pareidolia -> all scored cards are "face" -> Scary Face +30 each.
    # Only the two 3s score (no Splash). chips = 10 + (3+3) + 30*2 = 76. mult 2 -> 152.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.PAREIDOLIA), J(JokerType.SCARY_FACE)))
    assert res.chips == 76 and res.score == 152


def test_pareidolia_photograph_hits_first_card_any_rank():
    # Pareidolia: first scoring card (a 3) is "face" -> Photograph x2 once.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.PAREIDOLIA), J(JokerType.PHOTOGRAPH)))
    # chips = 10 + 6 = 16, mult 2 -> x2 = 4 -> 64.
    assert res.mult == 4.0 and res.score == 64


def test_rule_jokers_not_copyable():
    assert REGISTRY[JokerType.SPLASH].copyable is False
    assert REGISTRY[JokerType.PAREIDOLIA].copyable is False
