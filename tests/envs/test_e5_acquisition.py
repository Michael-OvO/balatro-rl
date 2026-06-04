"""E5 end-to-end: the agent's flat action codec drives every acquisition system through the
engine (legal_mask -> decode -> engine.step), proving the obs/action widening, the engine
unblock, and the codec all agree index-for-index. Values wiki-verified in the E1-E4 suites;
here we only assert the agent can *reach* and *resolve* each system via the masked flat space.
"""
import dataclasses

import numpy as np

from balatro_rl.engine.consumables import PlanetType, TarotType, planet, tarot
from balatro_rl.engine.engine import Verb, legal_actions, reset, step
from balatro_rl.engine.packs import Pack, PackKind, PackSize
from balatro_rl.engine.shop import CONSUMABLE_COST, ShopItem, ShopKind
from balatro_rl.engine.state import Phase
from balatro_rl.engine.vouchers import VoucherType
from balatro_rl.envs.actions import decode, encode_action, legal_mask


def _legal_ids(state):
    """The agent's view: the set of flat ids the mask says are legal."""
    return set(np.flatnonzero(legal_mask(state)).tolist())


def _mask_matches_engine(state):
    """Every flat-encodable engine action is masked True, and nothing else is."""
    ids = _legal_ids(state)
    eng = {encode_action(v, a) for v, a in legal_actions(state)}
    return ids == eng


def _shop(money=20, **over):
    """A SHOP-phase state (money/offers overridable)."""
    base = dict(shop_offers=(), pack_offers=(), voucher_offer=0)
    base.update(over)
    return dataclasses.replace(reset(0), phase=Phase.SHOP, money=money, **base)


# --------------------------------------------------------------------------- consumable buy
def test_agent_buys_planet_from_shop():
    s = _shop(shop_offers=(ShopItem(int(ShopKind.PLANET), int(PlanetType.MERCURY), CONSUMABLE_COST),))
    buy_id = encode_action(Verb.BUY, 0)
    assert buy_id in _legal_ids(s) and _mask_matches_engine(s)
    nxt, info = step(s, decode(buy_id))
    assert info["verb"] == "buy" and len(nxt.consumables) == 1
    assert nxt.consumables[0].type_id == int(PlanetType.MERCURY)


# --------------------------------------------------------------------------- booster pack flow
def test_agent_opens_pack_and_picks():
    s = _shop(money=20, pack_offers=(Pack(kind=int(PackKind.CELESTIAL),
                                          size=int(PackSize.NORMAL), cost=4),))
    open_id = encode_action(Verb.OPEN, 0)
    assert open_id in _legal_ids(s)
    opened, info = step(s, decode(open_id))
    assert opened.phase == Phase.OPEN_PACK and info["verb"] == "open"
    # In OPEN_PACK the mask offers PICK ids (the pickable items) + SKIP_PACK, nothing else.
    ids = _legal_ids(opened)
    assert encode_action(Verb.SKIP_PACK, 0) in ids and _mask_matches_engine(opened)
    pick_ids = [i for i in ids if decode(i)[0] == Verb.PICK]
    assert pick_ids, "a Celestial pack should reveal pickable Planets"
    picked, pinfo = step(opened, decode(pick_ids[0]))
    assert pinfo["verb"] == "pick"
    # Normal pack picks 1 -> back to SHOP with the item in hand.
    assert picked.phase == Phase.SHOP and len(picked.consumables) == 1


# --------------------------------------------------------------------------- voucher buy
def test_agent_buys_voucher():
    s = _shop(money=20, voucher_offer=int(VoucherType.GRABBER))
    vid = encode_action(Verb.BUY_VOUCHER, 0)
    assert vid in _legal_ids(s) and _mask_matches_engine(s)
    nxt, info = step(s, decode(vid))
    assert info["verb"] == "buy_voucher" and int(VoucherType.GRABBER) in nxt.vouchers


# --------------------------------------------------------------------------- targeting two-step
def test_agent_arms_and_targets_tarot():
    # PLAYING with a known hand + a card-targeting Tarot (The Sun -> all selected to Hearts).
    s = dataclasses.replace(reset(0), consumables=(tarot(TarotType.THE_SUN),))
    arm_id = encode_action(Verb.USE, 0)
    assert arm_id in _legal_ids(s)
    armed, info = step(s, decode(arm_id))
    assert info["verb"] == "use_arm" and armed.pending_consumable == 0
    # Now the ONLY legal ids are USE_TARGET subsets (bounded to the Tarot's reach of 3).
    ids = _legal_ids(armed)
    assert ids and all(decode(i)[0] == Verb.USE_TARGET for i in ids) and _mask_matches_engine(armed)
    assert all(len(decode(i)[1]) <= 3 for i in ids)         # The Sun targets up to 3
    # Apply to the first three cards -> they become Hearts; pending clears; Tarot consumed.
    target_id = encode_action(Verb.USE_TARGET, (0, 1, 2))
    assert target_id in ids
    done, _ = step(armed, decode(target_id))
    assert done.pending_consumable == -1 and done.consumables == ()
    assert all(c.suit == 1 for c in done.hand[:3])


def test_shop_mask_round_trips_with_everything_on_offer():
    # All four systems offered at once -> the mask still matches the engine exactly.
    s = _shop(money=50,
              shop_offers=(ShopItem(int(ShopKind.PLANET), int(PlanetType.MARS), CONSUMABLE_COST),),
              pack_offers=(Pack(kind=int(PackKind.BUFFOON), size=int(PackSize.NORMAL), cost=4),),
              voucher_offer=int(VoucherType.GRABBER))
    assert _mask_matches_engine(s)
    ids = _legal_ids(s)
    verbs = {decode(i)[0] for i in ids}
    assert {Verb.BUY, Verb.OPEN, Verb.BUY_VOUCHER, Verb.LEAVE_SHOP} <= verbs
