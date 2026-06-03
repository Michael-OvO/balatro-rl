from balatro_rl.engine.cards import Card
from balatro_rl.engine.hands import HandType, contains


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def test_pair_contains_only_pair():
    got = contains([C(3, 0), C(3, 1), C(7, 2)])
    assert got == frozenset({HandType.PAIR})


def test_two_pair_contains_pair_and_two_pair():
    got = contains([C(3, 0), C(3, 1), C(7, 2), C(7, 3)])
    assert got == frozenset({HandType.PAIR, HandType.TWO_PAIR})


def test_three_of_a_kind_contains_pair_and_trips():
    got = contains([C(3, 0), C(3, 1), C(3, 2), C(7, 3)])
    assert got == frozenset({HandType.PAIR, HandType.THREE_OF_A_KIND})


def test_full_house_contains_pair_two_pair_trips_full():
    got = contains([C(3, 0), C(3, 1), C(3, 2), C(7, 3), C(7, 0)])
    assert got == frozenset({
        HandType.PAIR, HandType.TWO_PAIR,
        HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE,
    })


def test_four_of_a_kind_contains_pair_trips_quad_not_two_pair():
    got = contains([C(3, 0), C(3, 1), C(3, 2), C(3, 3), C(7, 0)])
    assert got == frozenset({
        HandType.PAIR, HandType.THREE_OF_A_KIND, HandType.FOUR_OF_A_KIND,
    })
    assert HandType.TWO_PAIR not in got
    assert HandType.FULL_HOUSE not in got


def test_flush_detected():
    got = contains([C(3, 1), C(7, 1), C(9, 1), C(11, 1), C(13, 1)])
    assert got == frozenset({HandType.FLUSH})


def test_straight_detected():
    got = contains([C(3, 0), C(4, 1), C(5, 2), C(6, 3), C(7, 0)])
    assert got == frozenset({HandType.STRAIGHT})


def test_ace_low_straight_detected():
    got = contains([C(14, 0), C(2, 1), C(3, 2), C(4, 3), C(5, 0)])
    assert got == frozenset({HandType.STRAIGHT})


def test_straight_flush_contains_straight_and_flush():
    got = contains([C(3, 1), C(4, 1), C(5, 1), C(6, 1), C(7, 1)])
    assert got == frozenset({
        HandType.STRAIGHT, HandType.FLUSH, HandType.STRAIGHT_FLUSH,
    })


def test_high_card_contains_nothing():
    got = contains([C(14, 0), C(7, 1), C(2, 2)])
    assert got == frozenset()


def test_straight_needs_five_cards():
    # 4 sequential cards do NOT contain a Straight (base rules, no Four Fingers).
    got = contains([C(3, 0), C(4, 1), C(5, 2), C(6, 3)])
    assert got == frozenset()


def test_flush_needs_five_cards():
    # 4 same-suit cards do NOT contain a Flush.
    got = contains([C(3, 1), C(7, 1), C(9, 1), C(11, 1)])
    assert got == frozenset()
