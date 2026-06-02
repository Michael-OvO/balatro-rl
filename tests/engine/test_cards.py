from balatro_rl.engine.cards import Card, rank_chip_value, standard_deck, card_str


def test_rank_chip_values():
    assert rank_chip_value(14) == 11   # Ace
    assert rank_chip_value(13) == 10   # King
    assert rank_chip_value(11) == 10   # Jack
    assert rank_chip_value(10) == 10   # Ten
    assert rank_chip_value(7) == 7
    assert rank_chip_value(2) == 2


def test_standard_deck_is_52_unique_cards():
    deck = standard_deck()
    assert len(deck) == 52
    assert len(set(deck)) == 52
    assert all(2 <= c.rank <= 14 for c in deck)
    assert all(0 <= c.suit <= 3 for c in deck)


def test_card_defaults_have_no_modifiers():
    c = Card(rank=14, suit=0)
    assert (c.enhancement, c.edition, c.seal) == (0, 0, 0)


def test_card_str():
    assert card_str(Card(rank=13, suit=0)) == "K♠"   # K♠
    assert card_str(Card(rank=7, suit=1)) == "7♥"    # 7♥
