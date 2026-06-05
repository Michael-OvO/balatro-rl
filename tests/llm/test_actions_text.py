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


from balatro_rl.envs.actions import decode, legal_mask
from balatro_rl.llm.actions_text import parse_action


def test_build_menu_four_offer_shop_buys_decode_to_buy_not_sell():
    # Regression for the BUY/SELL id collision: a voucher-raised 4-offer shop must produce
    # four distinct "Buy" options whose action_ids decode to BUY (not SELL joker slots).
    state = engine.reset(0)
    offers = tuple(ShopItem(int(ShopKind.JOKER), int(JokerType.JOKER), 1) for _ in range(4))
    state = dataclasses.replace(state, phase=Phase.SHOP, money=100, shop_offers=offers)
    buy_opts = [o for o in build_menu(state).options if o.label.startswith("Buy")]
    assert len(buy_opts) == 4
    for o in buy_opts:
        assert decode(o.action_id)[0].name == "BUY"      # not SELL
    assert len({o.action_id for o in buy_opts}) == 4      # all distinct


def test_parse_menu_choice_returns_legal_action_id():
    state = engine.reset(0)
    offer = ShopItem(kind=int(ShopKind.JOKER), type_id=int(JokerType.JOKER), cost=2)
    state = dataclasses.replace(state, phase=Phase.SHOP, money=10, shop_offers=(offer,))
    menu = build_menu(state)
    leave = next(o for o in menu.options if "Leave" in o.label)
    res = parse_action(f'{{"choice": {leave.index}}}', state)
    assert res.error is None
    assert res.action_id == leave.action_id
    assert legal_mask(state)[res.action_id]


def test_parse_play_cards_returns_legal_play_id():
    state = engine.reset(0)                              # PLAYING, hands_left > 0
    res = parse_action('{"action": "play", "cards": [0]}', state)
    assert res.error is None
    verb, arg = decode(res.action_id)
    assert verb.name == "PLAY" and arg == (0,)


def test_parse_rejects_illegal_choice_index():
    state = engine.reset(0)
    res = parse_action('{"choice": 999}', state)
    assert res.error is not None and res.action_id is None


def test_parse_rejects_unparseable_reply():
    state = engine.reset(0)
    res = parse_action("I think I will play the kings.", state)
    assert res.error is not None


def test_parse_rejects_legal_shape_but_illegal_action():
    # A play-cards call is well-formed and encodable, but playing is illegal in SHOP.
    # The mask gate must reject it (this is the engine-safety boundary).
    state = engine.reset(0)
    state = dataclasses.replace(state, phase=Phase.SHOP, money=10, shop_offers=())
    res = parse_action('{"action": "play", "cards": [0]}', state)
    assert res.action_id is None
    assert res.error is not None
