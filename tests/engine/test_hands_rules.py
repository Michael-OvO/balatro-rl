from balatro_rl.engine.cards import Card
from balatro_rl.engine.hands import evaluate, is_face, HandType
from balatro_rl.engine.jokers.base import RuleFlags


def C(rank, suit):
    return Card(rank=rank, suit=suit)


def test_evaluate_default_rules_unchanged():
    # Backward compatible: no rules arg behaves like Plan 1.
    ht, idx = evaluate([C(13, 0), C(13, 1), C(3, 3), C(7, 3), C(9, 1)])
    assert ht == HandType.PAIR
    assert sorted(idx) == [0, 1]


def test_splash_makes_all_cards_score():
    rules = RuleFlags(splash=True)
    ht, idx = evaluate([C(13, 0), C(13, 1), C(3, 3), C(7, 3), C(9, 1)], rules)
    assert ht == HandType.PAIR                 # hand type unchanged
    assert sorted(idx) == [0, 1, 2, 3, 4]      # but every card scores


def test_is_face_normal():
    assert is_face(C(13, 0), RuleFlags()) is True    # King
    assert is_face(C(12, 0), RuleFlags()) is True    # Queen
    assert is_face(C(11, 0), RuleFlags()) is True     # Jack
    assert is_face(C(10, 0), RuleFlags()) is False
    assert is_face(C(2, 0), RuleFlags()) is False


def test_is_face_with_pareidolia():
    assert is_face(C(2, 0), RuleFlags(all_face=True)) is True
    assert is_face(C(10, 0), RuleFlags(all_face=True)) is True
