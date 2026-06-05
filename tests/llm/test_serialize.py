import dataclasses

from balatro_rl.engine import engine
from balatro_rl.engine.cards import Card, Enhancement
from balatro_rl.llm.serialize import serialize_state
from balatro_rl.engine.consumables import Consumable, ConsumableKind, PlanetType
from balatro_rl.engine.jokers.base import JokerState, JokerType
from balatro_rl.engine.shop import ShopItem, ShopKind
from balatro_rl.engine.state import Phase


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


def test_jokers_block_names_and_describes_each_joker():
    state = engine.reset(0)
    state = dataclasses.replace(state, jokers=(JokerState(type=JokerType.BARON),))
    text = serialize_state(state)
    assert "Baron" in text
    assert "King" in text                     # from descriptions.joker_desc(BARON)


def test_consumables_block_lists_owned_consumables():
    state = engine.reset(0)
    con = Consumable(kind=int(ConsumableKind.PLANET), type_id=int(PlanetType.PLUTO))
    state = dataclasses.replace(state, consumables=(con,))
    text = serialize_state(state)
    assert "Pluto" in text


def test_shop_block_lists_offers_with_cost_when_in_shop():
    state = engine.reset(0)
    offer = ShopItem(kind=int(ShopKind.JOKER), type_id=int(JokerType.JOKER), cost=2)
    state = dataclasses.replace(state, phase=Phase.SHOP, shop_offers=(offer,))
    text = serialize_state(state)
    assert "Shop" in text
    assert "Joker" in text
    assert "$2" in text


def test_armed_targeting_tarot_is_visible_in_serialize():
    # When a card-targeting Tarot is ARMED (pending_consumable set), the only legal move is
    # USE_TARGET; serialize must surface the armed consumable, its target-count, and the format.
    from balatro_rl.engine.consumables import TarotType, max_targets
    state = engine.reset(0)                         # PLAYING with a hand
    magician = Consumable(kind=int(ConsumableKind.TAROT), type_id=int(TarotType.THE_MAGICIAN))
    state = dataclasses.replace(state, consumables=(magician,), pending_consumable=0)
    text = serialize_state(state)
    assert "ARMED" in text
    assert "Magician" in text
    assert "target" in text.lower()
    assert str(max_targets(magician)) in text       # the target-count bound is shown
