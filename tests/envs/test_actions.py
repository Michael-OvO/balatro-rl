import numpy as np
from balatro_rl.engine.engine import Verb, reset, legal_actions, step
from balatro_rl.engine.state import Phase
from balatro_rl.envs.actions import (
    NUM_ACTIONS, decode, encode_action, legal_mask, PLAY_N, SHOP_BASE, _LEAVE,
    _USE0, _USE_TARGET0, _OPEN0, _PICK0, _SKIP_PACK, _BUY_VOUCHER,
)


def test_action_space_size():
    assert PLAY_N == 218            # C(8,1..5)
    # 218 play + 218 discard + 42 shop(buy4 sell6 reroll1 reorder30 leave1) + 3 USE
    #   + 218 USE_TARGET + 2 OPEN + 5 PICK + 1 SKIP + 1 VOUCHER
    assert NUM_ACTIONS == 708


def test_decode_e5_blocks_roundtrip():
    assert decode(_USE0 + 1) == (Verb.USE, 1)
    assert decode(_USE_TARGET0) == (Verb.USE_TARGET, (0,))
    assert decode(_OPEN0 + 1) == (Verb.OPEN, 1)
    assert decode(_PICK0 + 4) == (Verb.PICK, 4)
    assert decode(_SKIP_PACK) == (Verb.SKIP_PACK, 0)
    assert decode(_BUY_VOUCHER) == (Verb.BUY_VOUCHER, 0)
    assert _BUY_VOUCHER == NUM_ACTIONS - 1
    for i in range(NUM_ACTIONS):           # every id roundtrips
        v, a = decode(i)
        assert encode_action(v, a) == i


def test_buy_ids_never_collide_with_sell_ids():
    # Regression: with Overstock / Overstock Plus vouchers the engine offers up to 4 shop
    # CARD slots, so legal_actions emits (BUY, 2)/(BUY, 3). Those must encode to ids DISJOINT
    # from the SELL block. Before MAX_SHOP was raised to 4, encode_action(Verb.BUY, 2) aliased
    # onto SELL joker slot 0 (id 438), so a "buy the 3rd offer" choice silently SOLD a joker.
    buy_ids = {encode_action(Verb.BUY, i) for i in range(4)}
    sell_ids = {encode_action(Verb.SELL, j) for j in range(6)}
    assert buy_ids.isdisjoint(sell_ids)
    for i in range(4):
        assert decode(encode_action(Verb.BUY, i)) == (Verb.BUY, i)


def test_four_offer_shop_buy_actions_are_distinct_and_legal():
    # A voucher-raised 4-offer shop: every BUY decodes back to a BUY (not a SELL), and the
    # legal mask marks exactly the engine-legal actions (no collision undercount).
    import dataclasses
    from balatro_rl.engine.shop import ShopItem, ShopKind
    from balatro_rl.engine.jokers.base import JokerType
    s = reset(seed=1)
    offers = tuple(ShopItem(int(ShopKind.JOKER), int(JokerType.JOKER), 1) for _ in range(4))
    s = dataclasses.replace(s, phase=Phase.SHOP, money=100, shop_offers=offers)
    eng = legal_actions(s)
    buys = [a for a in eng if a[0] == Verb.BUY]
    assert {arg for _v, arg in buys} == {0, 1, 2, 3}
    for verb, arg in buys:
        assert decode(encode_action(verb, arg)) == (Verb.BUY, arg)
    mask = legal_mask(s)
    assert mask.sum() == len(eng)            # no collision -> exact 1:1 count


def test_decode_play_and_discard():
    assert decode(0) == (Verb.PLAY, (0,))
    assert decode(PLAY_N) == (Verb.DISCARD, (0,))         # first discard id
    v, arg = decode(PLAY_N - 1)                            # last play subset
    assert v == Verb.PLAY and len(arg) == 5


def test_decode_shop_actions():
    assert decode(SHOP_BASE) == (Verb.BUY, 0)
    assert decode(SHOP_BASE + 3) == (Verb.BUY, 3)          # BUY 0..3 (MAX_SHOP=4)
    assert decode(SHOP_BASE + 4) == (Verb.SELL, 0)         # SELL 0..5 (MAX_JOKERS=6) starts after BUY
    assert decode(SHOP_BASE + 10) == (Verb.REROLL, 0)      # after buy4 + sell6
    assert decode(_LEAVE) == (Verb.LEAVE_SHOP, 0)
    assert decode(_LEAVE + 1) == (Verb.USE, 0)             # USE ids appended after LEAVE_SHOP
    v, arg = decode(SHOP_BASE + 11)                         # first reorder (after buy4 + sell6 + reroll1)
    assert v == Verb.REORDER and isinstance(arg, tuple) and arg[0] != arg[1]


def test_legal_mask_matches_engine_in_playing():
    s = reset(seed=1)
    mask = legal_mask(s)
    assert mask.dtype == np.bool_ and mask.shape == (NUM_ACTIONS,)
    # Every engine-legal action maps to a True mask bit, and counts match.
    eng = legal_actions(s)
    assert mask.sum() == len(eng)
    # Each legal action decodes back to an engine action in the legal set.
    legal_ids = np.flatnonzero(mask)
    decoded = {decode(int(i)) for i in legal_ids}
    assert decoded == set(eng)


def test_legal_mask_in_shop_only_shop_actions():
    import dataclasses
    from balatro_rl.engine.cards import Card
    s = reset(seed=1)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3),
            Card(2, 0), Card(3, 0), Card(4, 0), Card(5, 0))
    s = dataclasses.replace(s, hand=hand, required=10, money=100)
    s, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))   # -> SHOP
    assert s.phase == Phase.SHOP
    mask = legal_mask(s)
    # No play/discard bits set in the shop.
    assert mask[:2 * PLAY_N].sum() == 0
    assert mask[_LEAVE]                          # LEAVE_SHOP always legal
