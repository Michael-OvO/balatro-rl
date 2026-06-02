from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


def test_baron_one_king_held():
    # Play a low pair; hold one King. base pair (5,4)? -> use a pair of 3s:
    # Pair of 3s: base (10,2) + (3+3)=6 -> 16 chips, mult 2. Baron x1.5 -> mult 3 -> 48.  # wiki: /w/Baron
    # Use mixed suits on kickers to avoid unintended flush.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=(J(JokerType.BARON),),
                     held=(C(13),))
    assert res.mult == 3.0 and res.score == 16 * 3


def test_baron_two_kings_held_exponential():
    # Two Kings held -> x1.5^2 = x2.25.  mult 2 -> 4.5 -> int(16*4.5)=72.  # wiki: /w/Baron
    # Use mixed suits on kickers to avoid unintended flush.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=(J(JokerType.BARON),),
                     held=(C(13), C(13)))
    assert res.mult == 4.5 and res.score == int(16 * 4.5)


def test_hack_retriggers_low_cards_chips():
    # Pair of 3s scored twice each via Hack. Base pair (10,2). Without Hack chips=10+3+3=16.
    # Hack: each scoring 3 triggers twice -> chips 10 + (3+3)*2 = 22, mult 2 -> 44.  # wiki: /w/Hack
    # Use mixed suits on kickers to avoid unintended flush.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=(J(JokerType.HACK),))
    assert res.chips == 22 and res.score == 44


def test_hack_retrigger_refires_greedy():
    # 3 of Diamonds (suit 3) scored; Hack retriggers it; Greedy +3 mult each trigger.
    # Pair of 3♦: base (10,2). chips = 10 + (3+3)*2 = 22 (Hack). Greedy fires on each
    # scored ♦ each trigger: 2 cards x 2 triggers x +3 = +12 mult -> mult 14 -> 22*14=308.
    res = score_play([C(3, 3), C(3, 3), C(7, 0), C(9, 0), C(2, 0)],
                     jokers=(J(JokerType.HACK), J(JokerType.GREEDY)))
    assert res.chips == 22 and res.mult == 14.0 and res.score == 22 * 14


def test_photograph_reapplies_on_retrigger():
    # Pareidolia (all face) + Hack (retrigger 2-5) + Photograph, pair of 4s.
    # First-face card (idx 0) is a 4 -> Hack retriggers it -> fires twice -> Photograph x2 each -> x4.
    # chips: base 10 + 4*2 (card0 twice) + 4*2 (card1 twice) = 26.
    # mult:  base 2, x2 on each of card0's two triggers -> 2*2*2 = 8.
    res = score_play([C(4, 0), C(4, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.PAREIDOLIA), J(JokerType.HACK), J(JokerType.PHOTOGRAPH)))
    assert res.chips == 26 and res.mult == 8.0


def test_held_xmult_applies_before_independent_addmult():
    # Baron (held King x1.5) resolves in the held phase, BEFORE Joker's independent +4.
    # Pair of 3s: 16 chips, base mult 2.  x1.5 -> 3, then +4 -> 7   (NOT (2+4)*1.5 = 9).
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.BARON), J(JokerType.JOKER)), held=(C(13),))
    assert res.mult == 7.0 and res.score == 112
