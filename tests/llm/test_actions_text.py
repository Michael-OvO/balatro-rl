import dataclasses

from balatro_rl.engine import engine
from balatro_rl.engine.state import Phase
from balatro_rl.engine.shop import ShopItem, ShopKind
from balatro_rl.engine.jokers.base import JokerType
from balatro_rl.llm.actions_text import build_menu, render_menu


def test_playing_state_offers_play_and_discard_not_discrete_subsets():
    state = engine.reset(0)                       # PLAYING, full hand
    menu = build_menu(state)
    assert menu.can_play is True
    assert menu.can_discard is True
    # PLAY/DISCARD are emitted as card-index calls, never enumerated as discrete options.
    assert all("play" not in o.label.lower() for o in menu.options)


def test_shop_state_lists_buy_and_leave_as_numbered_options():
    state = engine.reset(0)
    offer = ShopItem(kind=int(ShopKind.JOKER), type_id=int(JokerType.JOKER), cost=2)
    state = dataclasses.replace(state, phase=Phase.SHOP, money=10, shop_offers=(offer,))
    menu = build_menu(state)
    labels = [o.label for o in menu.options]
    assert any("Buy" in l and "Joker" in l for l in labels)
    assert any("Leave" in l for l in labels)
    # options are contiguously indexed from 0
    assert [o.index for o in menu.options] == list(range(len(menu.options)))


def test_render_menu_includes_indices_and_card_instructions():
    state = engine.reset(0)
    text = render_menu(build_menu(state))
    assert "play" in text.lower() and "cards" in text.lower()
