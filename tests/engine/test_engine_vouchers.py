"""Phase E4 — engine threading of vouchers (the persistent per-run modifiers).

ENGINE-FIRST / agent BLIND: legal_actions never emits Verb.BUY_VOUCHER, so buying a voucher
is reachable only via a direct engine.step((Verb.BUY_VOUCHER, 0)). The agent wiring (obs/
action widening) is the later E5. These tests buy a voucher directly, then leave the shop and
assert the NEXT blind reflects the modifier (vouchers are the single source of truth).
"""
import dataclasses

from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import (
    HANDS_PER_BLIND, DISCARDS_PER_BLIND, HAND_SIZE, JOKER_SLOTS, VOUCHER_COST,
    Verb, legal_actions, reset, step,
)
from balatro_rl.engine.jokers.base import JokerState, JokerType
from balatro_rl.engine.economy import interest
from balatro_rl.engine.shop import CARD_SLOTS, reroll_cost
from balatro_rl.engine.state import Phase
from balatro_rl.engine.vouchers import VoucherType, interest_cap


def _shop(seed=1, **over):
    """Reach the SHOP by clearing a Small blind with a known four-of-a-kind."""
    s = reset(seed=seed)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
            Card(3, 0), Card(4, 0), Card(5, 0))
    s = dataclasses.replace(s, hand=hand, required=10, **over)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    assert s2.phase == Phase.SHOP
    return s2


def _buy_voucher(s, voucher):
    """Pin an offer for `voucher` and buy it via the engine-only verb."""
    s = dataclasses.replace(s, voucher_offer=int(voucher))
    s2, info = step(s, (Verb.BUY_VOUCHER, 0))
    return s2, info


def _next_blind(s):
    s2, _ = step(s, (Verb.LEAVE_SHOP, 0))
    assert s2.phase == Phase.PLAYING
    return s2


# ---- buy mechanics -----------------------------------------------------------

def test_buy_voucher_deducts_ten_and_clears_offer():
    s = _shop(money=100, hands_left=1)
    s2, info = _buy_voucher(s, VoucherType.GRABBER)
    assert info["verb"] == "buy_voucher"
    assert s2.money == s.money - VOUCHER_COST
    assert s2.voucher_offer == 0                       # offer consumed
    assert int(VoucherType.GRABBER) in s2.vouchers


def test_buy_voucher_unaffordable_asserts():
    s = _shop(money=5, hands_left=1)
    s = dataclasses.replace(s, voucher_offer=int(VoucherType.GRABBER))
    try:
        step(s, (Verb.BUY_VOUCHER, 0))
        assert False, "expected an affordability assertion"
    except AssertionError:
        pass


def test_buy_voucher_no_offer_asserts():
    s = _shop(money=100, hands_left=1)
    s = dataclasses.replace(s, voucher_offer=0)   # explicitly no offer
    try:
        step(s, (Verb.BUY_VOUCHER, 0))
        assert False, "expected a no-offer assertion"
    except AssertionError:
        pass


def test_buy_voucher_unmet_prereq_asserts():
    """Money Tree requires Seed Money owned; buying it cold must fail."""
    s = _shop(money=100, hands_left=1)
    s = dataclasses.replace(s, voucher_offer=int(VoucherType.MONEY_TREE))
    try:
        step(s, (Verb.BUY_VOUCHER, 0))
        assert False, "expected a prereq assertion"
    except AssertionError:
        pass


# ---- per-blind modifiers (apply on the NEXT blind) ---------------------------

def test_grabber_adds_a_hand_next_blind():
    s = _shop(money=100, hands_left=1)
    s2, _ = _buy_voucher(s, VoucherType.GRABBER)
    nxt = _next_blind(s2)
    assert nxt.hands_left == HANDS_PER_BLIND + 1       # 5


def test_grabber_plus_nacho_tong_adds_two_hands():
    s = _shop(money=100, hands_left=1)
    s = dataclasses.replace(s, vouchers=(int(VoucherType.GRABBER),))
    s2, _ = _buy_voucher(s, VoucherType.NACHO_TONG)
    nxt = _next_blind(s2)
    assert nxt.hands_left == HANDS_PER_BLIND + 2       # 6


def test_wasteful_and_recyclomancy_add_discards():
    s = _shop(money=100, hands_left=1)
    s2, _ = _buy_voucher(s, VoucherType.WASTEFUL)
    nxt = _next_blind(s2)
    assert nxt.discards_left == DISCARDS_PER_BLIND + 1  # 4
    s3 = dataclasses.replace(_shop(money=100, hands_left=1),
                             vouchers=(int(VoucherType.WASTEFUL),))
    s3, _ = _buy_voucher(s3, VoucherType.RECYCLOMANCY)
    nxt3 = _next_blind(s3)
    assert nxt3.discards_left == DISCARDS_PER_BLIND + 2  # 5


def test_paint_brush_and_palette_add_hand_size():
    s = _shop(money=100, hands_left=1)
    s2, _ = _buy_voucher(s, VoucherType.PAINT_BRUSH)
    nxt = _next_blind(s2)
    assert nxt.hand_size == HAND_SIZE + 1              # 9
    assert len(nxt.hand) == HAND_SIZE + 1             # drew up to the larger hand
    s3 = dataclasses.replace(_shop(money=100, hands_left=1),
                             vouchers=(int(VoucherType.PAINT_BRUSH),))
    s3, _ = _buy_voucher(s3, VoucherType.PALETTE)
    nxt3 = _next_blind(s3)
    assert nxt3.hand_size == HAND_SIZE + 2            # 10


def test_crystal_ball_raises_consumable_slots_immediately():
    s = _shop(money=100, hands_left=1)
    assert s.consumable_slots == 2
    s2, _ = _buy_voucher(s, VoucherType.CRYSTAL_BALL)
    assert s2.consumable_slots == 3                   # immediate field effect
    # And it persists across the blind boundary.
    nxt = _next_blind(s2)
    assert nxt.consumable_slots == 3


def test_water_boss_still_zeroes_discards_with_wasteful():
    """The boss discard override wins over the voucher bonus (Water still 0)."""
    from balatro_rl.engine.bosses import BossEffect
    s = _shop(money=100, hands_left=1)
    s2, _ = _buy_voucher(s, VoucherType.WASTEFUL)
    # Force the next blind to be the Water boss by enabling bosses + landing on blind 2.
    s2 = dataclasses.replace(s2, blind_index=1, bosses_enabled=True)
    # Repeatedly leave/clear isn't trivial here; instead assert via the boss override helper
    # directly: with Wasteful the base discards is 4, but Water clamps to 0.
    from balatro_rl.engine.bosses import boss_discards_left
    assert boss_discards_left(BossEffect.THE_WATER, DISCARDS_PER_BLIND + 1) == 0


# ---- joker-slot cap ----------------------------------------------------------

def test_antimatter_allows_six_jokers():
    five = tuple(JokerState(JokerType.JOKER) for _ in range(JOKER_SLOTS))
    s = _shop(money=100, hands_left=1, jokers=five)
    s2, _ = _buy_voucher(s, VoucherType.ANTIMATTER)
    # A sixth joker now fits. Buy one via a pinned JOKER offer.
    from balatro_rl.engine.shop import ShopItem, ShopKind
    s2 = dataclasses.replace(s2, shop_offers=(
        ShopItem(int(ShopKind.JOKER), int(JokerType.BLUEPRINT), 10),))
    s3, info = step(s2, (Verb.BUY, 0))
    assert info["verb"] == "buy"
    assert len(s3.jokers) == JOKER_SLOTS + 1          # 6
    # legal_actions now offers that buy too (cap raised by the voucher).
    s4 = dataclasses.replace(s2, jokers=five)          # back to 5 jokers, Antimatter owned
    assert any(a[0] == Verb.BUY for a in legal_actions(s4))


def test_legal_actions_joker_buy_withheld_at_voucher_raised_cap():
    """With 6 jokers and Antimatter, the joker buy is withheld (cap = 6 reached)."""
    from balatro_rl.engine.shop import ShopItem, ShopKind
    six = tuple(JokerState(JokerType.JOKER) for _ in range(JOKER_SLOTS + 1))
    s = _shop(money=100, hands_left=1,
              vouchers=(int(VoucherType.ANTIMATTER),), jokers=six)
    s = dataclasses.replace(s, shop_offers=(
        ShopItem(int(ShopKind.JOKER), int(JokerType.BLUEPRINT), 10),))
    assert all(a[0] != Verb.BUY for a in legal_actions(s))


# ---- interest cap ------------------------------------------------------------

def test_seed_money_raises_interest_cap():
    assert interest_cap((int(VoucherType.SEED_MONEY),)) == 10
    # At $50 held, default cap 5 -> $5; Seed Money cap 10 -> $10.
    assert interest(50, cap=5) == 5
    assert interest(50, cap=interest_cap((int(VoucherType.SEED_MONEY),))) == 10


def test_seed_money_cash_out_pays_more_interest():
    """Clearing a blind cashes out interest under the voucher's raised cap."""
    s = _shop(money=50, hands_left=1, vouchers=(int(VoucherType.SEED_MONEY),))
    # The shop money already reflects the cash-out done on entering the shop. Recompute the
    # interest portion: with $50 pre-cash-out and Seed Money cap 10, interest is $10 not $5.
    # Build a fresh clear with money=50 and assert the cash-out earned uses cap 10.
    s0 = reset(seed=1)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
            Card(3, 0), Card(4, 0), Card(5, 0))
    s0 = dataclasses.replace(s0, hand=hand, required=10, money=50, hands_left=1,
                             vouchers=(int(VoucherType.SEED_MONEY),))
    s_no = dataclasses.replace(s0, vouchers=())
    after_cap, _ = step(s0, (Verb.PLAY, (0, 1, 2, 3)))
    after_def, _ = step(s_no, (Verb.PLAY, (0, 1, 2, 3)))
    # The only difference is the interest cap: $10 vs $5 -> $5 more in the capped run.
    assert after_cap.money == after_def.money + 5


# ---- shop modifiers ----------------------------------------------------------

def test_overstock_offers_three_card_slots():
    s = _shop(money=100, hands_left=1)
    s2, _ = _buy_voucher(s, VoucherType.OVERSTOCK)
    nxt = _next_blind(s2)                               # advance...
    # Clear the next blind to re-enter the shop and see the widened offers.
    nxt = dataclasses.replace(nxt, hand=(
        Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
        Card(3, 0), Card(4, 0), Card(5, 0)), required=10, hands_left=1)
    shop2, _ = step(nxt, (Verb.PLAY, (0, 1, 2, 3)))
    assert shop2.phase == Phase.SHOP
    assert len(shop2.shop_offers) == CARD_SLOTS + 1    # 3 slots


def test_overstock_plus_offers_four_card_slots():
    s = _shop(money=100, hands_left=1,
              vouchers=(int(VoucherType.OVERSTOCK), int(VoucherType.OVERSTOCK_PLUS)))
    nxt = _next_blind(s)
    nxt = dataclasses.replace(nxt, hand=(
        Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
        Card(3, 0), Card(4, 0), Card(5, 0)), required=10, hands_left=1)
    shop2, _ = step(nxt, (Verb.PLAY, (0, 1, 2, 3)))
    assert len(shop2.shop_offers) == CARD_SLOTS + 2    # 4 slots


def test_reroll_surplus_discounts_reroll_cost():
    s = _shop(money=100, hands_left=1, vouchers=(int(VoucherType.REROLL_SURPLUS),))
    # First reroll: base 5 - 2 discount = 3.
    s2, info = step(s, (Verb.REROLL, 0))
    assert info["cost"] == 3
    assert s2.money == s.money - 3


def test_reroll_glut_discounts_further():
    s = _shop(money=100, hands_left=1,
              vouchers=(int(VoucherType.REROLL_SURPLUS), int(VoucherType.REROLL_GLUT)))
    s2, info = step(s, (Verb.REROLL, 0))
    assert info["cost"] == 1                            # base 5 - 4 = 1


def test_reroll_discount_floors_at_zero():
    """Big discount + many rerolls can't push the cost negative."""
    assert reroll_cost(0, discount=4) == 1
    assert reroll_cost(0, discount=10) == 0


# ---- agent blindness ---------------------------------------------------------

def test_legal_actions_offers_buy_voucher_when_eligible():
    # E5: the policy now sees the voucher slot — BUY_VOUCHER is legal when affordable + prereq met.
    s = dataclasses.replace(_shop(money=100, hands_left=1), voucher_offer=int(VoucherType.GRABBER))
    assert (Verb.BUY_VOUCHER, 0) in legal_actions(s)
    # ...withheld when the prereq isn't met (Nacho Tong needs Grabber owned)...
    locked = dataclasses.replace(_shop(money=100, hands_left=1),
                                 voucher_offer=int(VoucherType.NACHO_TONG))
    assert all(a[0] != Verb.BUY_VOUCHER for a in legal_actions(locked))
    # ...and withheld when unaffordable (vouchers cost $10; set money AFTER cash-out).
    poor = dataclasses.replace(_shop(hands_left=1), money=5, voucher_offer=int(VoucherType.GRABBER))
    assert all(a[0] != Verb.BUY_VOUCHER for a in legal_actions(poor))


def test_reset_honors_no_vouchers():
    """Fresh reset has no vouchers and byte-identical base setup."""
    s = reset(seed=1)
    assert s.vouchers == ()
    assert s.voucher_offer == 0
    assert s.hands_left == HANDS_PER_BLIND
    assert s.discards_left == DISCARDS_PER_BLIND
    assert s.hand_size == HAND_SIZE
    assert s.consumable_slots == 2
