from balatro_rl.engine.cards import Card
from balatro_rl.engine.hands import HandType
from balatro_rl.engine.scoring import score_play


def C(rank, suit=None):
    # Default suit varies by rank to avoid unintended flushes in 5-card hands.
    return Card(rank=rank, suit=rank % 4 if suit is None else suit)


def test_pair_of_kings():
    # base (10,2); scoring cards = two Kings -> chips 10 + 10 + 10 = 30; 30*2 = 60
    res = score_play([C(13), C(13), C(3), C(7), C(9)])
    assert res.hand_type == HandType.PAIR
    assert res.chips == 30
    assert res.mult == 2
    assert res.score == 60


def test_high_card_ace():
    # base (5,1); Ace chips 11 -> chips 16; 16*1 = 16
    res = score_play([C(14), C(7), C(2)])
    assert res.hand_type == HandType.HIGH_CARD
    assert res.chips == 16
    assert res.mult == 1
    assert res.score == 16


def test_flush_all_cards_score():
    # base (35,4); all five score: 2+5+7+9+10(K? use 10) chips
    res = score_play([C(2, 1), C(5, 1), C(7, 1), C(9, 1), C(10, 1)])
    assert res.hand_type == HandType.FLUSH
    assert res.chips == 35 + (2 + 5 + 7 + 9 + 10)  # 68
    assert res.mult == 4
    assert res.score == 68 * 4


def test_four_of_a_kind_excludes_kicker_chips():
    # base (60,7); four 9s score (9*4=36), kicker King does NOT add chips
    res = score_play([C(9), C(9), C(9), C(9), C(13)])
    assert res.hand_type == HandType.FOUR_OF_A_KIND
    assert res.chips == 60 + 36
    assert res.score == (60 + 36) * 7


def test_scoring_idx_recorded():
    res = score_play([C(13), C(13), C(3)])
    assert sorted(res.scoring_idx) == [0, 1]
