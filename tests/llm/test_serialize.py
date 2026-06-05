import dataclasses

from balatro_rl.engine import engine
from balatro_rl.engine.cards import Card, Enhancement
from balatro_rl.llm.serialize import serialize_state


def test_header_has_ante_score_and_resources():
    state = engine.reset(0)
    text = serialize_state(state)
    assert "Ante 1" in text
    assert "Small blind" in text                 # blind_index 0
    assert f"{state.round_score}/{state.required}" in text
    assert "Hands left:" in text and "Discards left:" in text
    assert f"${state.money}" in text


def test_hand_block_lists_every_card_with_an_index():
    state = engine.reset(0)
    text = serialize_state(state)
    assert "Hand:" in text
    for i in range(len(state.hand)):
        assert f"[{i}]" in text


def test_enhanced_card_shows_its_modifier():
    state = engine.reset(0)
    hand = list(state.hand)
    hand[0] = Card(rank=hand[0].rank, suit=hand[0].suit, enhancement=int(Enhancement.STEEL))
    state = dataclasses.replace(state, hand=tuple(hand))
    text = serialize_state(state)
    assert "Steel" in text
