"""Regression tests for the E5 review findings — the voucher-enabled action paths the original
E5 spot-checks missed (they ran without the relevant vouchers owned):

  BUG-1  Overstock raises the shop to 4 offers; a BUY id for slot 2/3 must NOT alias into the
         SELL block (silent buy->sell corruption) — the legal_mask BUY skip-guard + MAX_SHOP=4.
  BUG-2  Judgement's joker-slot cap must be voucher-aware (Antimatter -> can fill the 6th slot).
  RISK-2 The Death Tarot needs EXACTLY 2 targets; the pending two-step must offer only size-2
         (a size-1 pick would silently waste it) and must not arm with <2 hand cards.
"""
import dataclasses

import numpy as np

from balatro_rl.engine.consumables import TarotType, tarot
from balatro_rl.engine.engine import Verb, legal_actions, reset, step
from balatro_rl.engine.jokers.base import JokerState, JokerType
from balatro_rl.engine.shop import CONSUMABLE_COST, ShopItem, ShopKind
from balatro_rl.engine.state import Phase
from balatro_rl.engine.vouchers import VoucherType
from balatro_rl.envs.actions import MAX_SHOP, decode, encode_action, legal_mask


def C(rank, suit=0):
    from balatro_rl.engine.cards import Card
    return Card(rank=rank, suit=suit)


# --------------------------------------------------------------- BUG-1: BUY/SELL no aliasing
def test_overstock_four_offers_buy_never_aliases_sell():
    # SHOP with Overstock owned and 4 joker offers (the real Overstock+Plus max).
    offers = tuple(ShopItem(int(ShopKind.JOKER), int(JokerType.BLUEPRINT), 4) for _ in range(4))
    s = dataclasses.replace(reset(0), phase=Phase.SHOP, money=50,
                            vouchers=(int(VoucherType.OVERSTOCK), int(VoucherType.OVERSTOCK_PLUS)),
                            shop_offers=offers)
    # Every BUY the engine offers must round-trip to a BUY id that decodes back to BUY (not SELL).
    for verb, arg in legal_actions(s):
        if verb == Verb.BUY:
            aid = encode_action(verb, arg)
            assert decode(aid) == (Verb.BUY, arg), f"BUY {arg} aliased to {decode(aid)}"
    # The mask agrees with the engine exactly (no extra/aliased bits).
    mask = legal_mask(s)
    eng = {encode_action(v, a) for v, a in legal_actions(s)
           if not (v == Verb.BUY and a >= MAX_SHOP)}
    assert set(np.flatnonzero(mask)) == eng


def test_buy_slot_three_actually_buys_not_sells():
    # Buying offer slot 3 (only reachable with Overstock) must BUY a joker, not SELL one.
    offers = tuple(ShopItem(int(ShopKind.JOKER), int(JokerType.BLUEPRINT), 4) for _ in range(4))
    s = dataclasses.replace(reset(0), phase=Phase.SHOP, money=50,
                            vouchers=(int(VoucherType.OVERSTOCK), int(VoucherType.OVERSTOCK_PLUS)),
                            shop_offers=offers, jokers=())
    aid = encode_action(Verb.BUY, 3)
    assert legal_mask(s)[aid]                      # slot 3 is legal (MAX_SHOP=4)
    nxt, info = step(s, decode(aid))
    assert info["verb"] == "buy" and len(nxt.jokers) == 1   # bought, did NOT sell


# --------------------------------------------------------------- BUG-2: Judgement voucher-aware
def test_judgement_fills_sixth_slot_with_antimatter():
    # 5 jokers + Antimatter (cap 6): Judgement must create a 6th joker, not silently no-op.
    five = tuple(JokerState(type=int(JokerType.JOKER)) for _ in range(5))
    s = dataclasses.replace(reset(0), consumables=(tarot(TarotType.JUDGEMENT),),
                            jokers=five, vouchers=(int(VoucherType.ANTIMATTER),))
    nxt, _ = step(s, (Verb.USE, 0))
    assert len(nxt.jokers) == 6 and nxt.consumables == ()   # 6th joker created, tarot consumed


def test_judgement_noops_at_cap_without_voucher():
    five = tuple(JokerState(type=int(JokerType.JOKER)) for _ in range(5))
    s = dataclasses.replace(reset(0), consumables=(tarot(TarotType.JUDGEMENT),), jokers=five)
    nxt, _ = step(s, (Verb.USE, 0))
    assert len(nxt.jokers) == 5 and nxt.consumables == ()   # no slot (cap 5), tarot consumed


# --------------------------------------------------------------- RISK-2: Death exactly-2
def test_death_arms_and_offers_only_size_two():
    s = dataclasses.replace(reset(0), consumables=(tarot(TarotType.DEATH),),
                            hand=(C(2, 0), C(3, 1), C(4, 2), C(5, 3)))
    armed, info = step(s, (Verb.USE, 0))
    assert info["verb"] == "use_arm" and armed.pending_consumable == 0
    opts = legal_actions(armed)
    assert opts and all(v == Verb.USE_TARGET and len(a) == 2 for v, a in opts)   # ONLY size-2


def test_death_not_armable_with_one_card():
    # Hand of 1 -> Death can't be satisfied (needs 2), so it's not even offered (no stuck arm).
    s = dataclasses.replace(reset(0), consumables=(tarot(TarotType.DEATH),), hand=(C(2, 0),))
    assert not any(v == Verb.USE for v, _a in legal_actions(s))


def test_death_apply_copies_right_onto_left():
    s = dataclasses.replace(reset(0), consumables=(tarot(TarotType.DEATH),),
                            hand=(C(2, 0), C(10, 1)), master_deck=(C(2, 0), C(10, 1)))
    s = dataclasses.replace(s, master_deck=s.hand)        # share identity for the by-id replace
    armed, _ = step(s, (Verb.USE, 0))
    nxt, _ = step(armed, (Verb.USE_TARGET, (0, 1)))       # left(0) becomes a copy of right(1)
    assert nxt.hand[0].rank == 10 and nxt.hand[1].rank == 10
    assert nxt.pending_consumable == -1 and nxt.consumables == ()
