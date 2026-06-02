from balatro_rl.engine.cards import Card
from balatro_rl.engine.hands import HandType, HAND_BASE, evaluate


def C(rank, suit=None):
    # Default suit varies by rank to avoid unintended flushes in 5-card hands.
    return Card(rank=rank, suit=rank % 4 if suit is None else suit)


def test_high_card_scores_only_highest():
    ht, idx = evaluate([C(14), C(7), C(3)])
    assert ht == HandType.HIGH_CARD
    assert idx == [0]  # the Ace


def test_pair_scores_only_the_pair():
    # K K 3 7 9 -> Pair, scoring cards are the two Kings (indices 0,1)
    ht, idx = evaluate([C(13), C(13), C(3), C(7), C(9)])
    assert ht == HandType.PAIR
    assert sorted(idx) == [0, 1]


def test_two_pair():
    ht, idx = evaluate([C(13), C(13), C(7), C(7), C(2)])
    assert ht == HandType.TWO_PAIR
    assert sorted(idx) == [0, 1, 2, 3]


def test_three_of_a_kind_scores_only_the_three():
    ht, idx = evaluate([C(9), C(9), C(9), C(2), C(5)])
    assert ht == HandType.THREE_OF_A_KIND
    assert sorted(idx) == [0, 1, 2]


def test_four_of_a_kind_excludes_kicker():
    ht, idx = evaluate([C(9), C(9), C(9), C(9), C(5)])
    assert ht == HandType.FOUR_OF_A_KIND
    assert sorted(idx) == [0, 1, 2, 3]  # kicker (index 4) does NOT score


def test_full_house_scores_all_five():
    ht, idx = evaluate([C(9), C(9), C(9), C(2), C(2)])
    assert ht == HandType.FULL_HOUSE
    assert sorted(idx) == [0, 1, 2, 3, 4]


def test_flush_scores_all_five():
    ht, idx = evaluate([C(2, 1), C(5, 1), C(7, 1), C(9, 1), C(13, 1)])
    assert ht == HandType.FLUSH
    assert sorted(idx) == [0, 1, 2, 3, 4]


def test_straight_ace_high():
    ht, _ = evaluate([C(10, 0), C(11, 1), C(12, 2), C(13, 3), C(14, 0)])
    assert ht == HandType.STRAIGHT


def test_straight_ace_low():
    ht, _ = evaluate([C(14, 0), C(2, 1), C(3, 2), C(4, 3), C(5, 0)])
    assert ht == HandType.STRAIGHT


def test_straight_flush_beats_straight_and_flush():
    ht, _ = evaluate([C(6, 1), C(7, 1), C(8, 1), C(9, 1), C(10, 1)])
    assert ht == HandType.STRAIGHT_FLUSH


def test_hand_base_table_values():
    assert HAND_BASE[HandType.HIGH_CARD] == (5, 1)
    assert HAND_BASE[HandType.PAIR] == (10, 2)
    assert HAND_BASE[HandType.STRAIGHT] == (30, 4)
    assert HAND_BASE[HandType.FLUSH] == (35, 4)
    assert HAND_BASE[HandType.FOUR_OF_A_KIND] == (60, 7)
    assert HAND_BASE[HandType.STRAIGHT_FLUSH] == (100, 8)
