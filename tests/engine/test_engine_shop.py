import dataclasses
from balatro_rl.engine.cards import Card
from balatro_rl.engine.consumables import ConsumableKind, PlanetType, planet
from balatro_rl.engine.engine import Verb, reset, legal_actions, step
from balatro_rl.engine.state import Phase
from balatro_rl.engine.jokers.base import JokerType, JokerState
from balatro_rl.engine.shop import CONSUMABLE_COST, ShopItem, ShopKind
import balatro_rl.engine.jokers.library  # noqa: F401


def _clearable(seed=1, **over):
    """A state one play away from clearing, with a known big hand."""
    s = reset(seed=seed)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
            Card(3, 0), Card(4, 0), Card(5, 0))
    return dataclasses.replace(s, hand=hand, required=10, **over)


def test_clearing_a_blind_enters_shop_and_pays_out():
    s = _clearable(money=10, hands_left=3)   # Small blind, $10 held, 3 hands left
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))  # four-of-a-kind kings, clears
    assert s2.phase == Phase.SHOP
    # cash-out = reward(small $3) + interest(10 -> $2) + leftover hands (2 left after play) $2 = $7
    assert s2.money == 10 + 3 + 2 + 2
    assert len(s2.shop_offers) == 2
    assert s2.blind_index == 0   # NOT advanced yet (advance happens on leave)


def _force_offers(s, offers):
    """Replace the shop offers with a known typed set (engine offers are now ShopItems)."""
    return dataclasses.replace(s, shop_offers=tuple(offers))


def test_shop_buy_adds_joker_and_spends():
    s = _clearable(money=100, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))  # enter shop
    assert s2.phase == Phase.SHOP
    # Pin a known JOKER offer so the buy assertions don't depend on the random roll.
    s2 = _force_offers(s2, [ShopItem(int(ShopKind.JOKER), int(JokerType.BLUEPRINT), 10)])
    s3, info = step(s2, (Verb.BUY, 0))
    assert info["verb"] == "buy" and info["kind"] == int(ShopKind.JOKER)
    assert info["type_id"] == int(JokerType.BLUEPRINT) and info["cost"] == 10
    assert s3.money == s2.money - 10
    assert s3.jokers[-1].type == JokerType.BLUEPRINT
    assert len(s3.shop_offers) == 0


def test_shop_buy_planet_adds_consumable():
    """A Planet offer is bought into a consumable slot (kind-aware BUY), $3, jokers untouched.
    Reachable only via a direct engine.step — the policy stays blind to consumable offers."""
    s = _clearable(money=100, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))   # enter shop
    s2 = _force_offers(s2, [ShopItem(int(ShopKind.PLANET), int(PlanetType.MERCURY),
                                     CONSUMABLE_COST)])
    assert s2.consumables == () and s2.consumable_slots == 2
    s3, info = step(s2, (Verb.BUY, 0))
    assert info["verb"] == "buy" and info["kind"] == int(ShopKind.PLANET)
    assert info["type_id"] == int(PlanetType.MERCURY) and info["cost"] == CONSUMABLE_COST
    assert s3.money == s2.money - CONSUMABLE_COST
    assert len(s3.consumables) == 1
    con = s3.consumables[-1]
    # Stored under ConsumableKind (so USE/obs read it right), not the raw ShopKind.
    assert con.kind == int(ConsumableKind.PLANET) and con.type_id == int(PlanetType.MERCURY)
    assert s3.jokers == ()                        # joker slots untouched
    assert len(s3.shop_offers) == 0
    # The bought Planet is a valid consumable: USE it and Mercury levels up Pair.
    from balatro_rl.engine.hands import HandType
    before = s3.levels[int(HandType.PAIR)]
    s4, uinfo = step(s3, (Verb.USE, 0))
    assert uinfo["verb"] == "use"
    assert s4.levels[int(HandType.PAIR)] == before + 1 and s4.consumables == ()


def test_legal_actions_offers_consumable_buy():
    """E5: the policy now sees consumable offers — BUY is offered for both kinds (slot permitting)."""
    s = _clearable(money=100, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    s2 = _force_offers(s2, [
        ShopItem(int(ShopKind.PLANET), int(PlanetType.MERCURY), CONSUMABLE_COST),
        ShopItem(int(ShopKind.JOKER), int(JokerType.BLUEPRINT), 10),
    ])
    buys = {a[1] for a in legal_actions(s2) if a[0] == Verb.BUY}
    assert buys == {0, 1}     # both the Planet (slot 0) and the JOKER (slot 1) are buyable
    # ...but a full consumable inventory withholds the Planet buy (the joker stays buyable).
    full = dataclasses.replace(s2, consumables=(planet(PlanetType.PLUTO), planet(PlanetType.MARS)))
    assert {a[1] for a in legal_actions(full) if a[0] == Verb.BUY} == {1}


def test_shop_sell_returns_money_and_frees_slot():
    s = _clearable(money=100, hands_left=1, jokers=(JokerState(JokerType.BARON),))
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    s3, info = step(s2, (Verb.SELL, 0))   # sell Baron (cost 8 -> sell 4)
    assert info["verb"] == "sell" and info["value"] == 4
    assert s3.money == s2.money + 4
    assert s3.jokers == ()


def test_shop_reroll_costs_and_replaces_offers():
    s = _clearable(money=100, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    before = s2.money
    s3, info = step(s2, (Verb.REROLL, 0))
    assert info["verb"] == "reroll" and info["cost"] == 5
    assert s3.money == before - 5
    assert s3.rerolls_done == 1
    s4, info2 = step(s3, (Verb.REROLL, 0))
    assert info2["cost"] == 6   # +1 each reroll


def test_shop_reorder_jokers():
    s = _clearable(money=10, hands_left=1,
                   jokers=(JokerState(JokerType.JOKER), JokerState(JokerType.BARON)))
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    s3, info = step(s2, (Verb.REORDER, (0, 1)))   # move slot0 -> slot1
    assert info["verb"] == "reorder"
    assert [j.type for j in s3.jokers] == [JokerType.BARON, JokerType.JOKER]


def test_leave_shop_advances_to_next_blind():
    s = _clearable(money=10, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))   # cleared Small -> shop
    s3, info = step(s2, (Verb.LEAVE_SHOP, 0))
    assert s3.phase == Phase.PLAYING
    assert s3.blind_index == 1       # advanced to Big
    assert s3.round_score == 0 and s3.hands_left == 4 and len(s3.hand) == 8
    assert s3.shop_offers == () and s3.rerolls_done == 0


def test_clearing_ante8_boss_wins_no_shop():
    # _clearable already sets required=10; passing it again would duplicate the kwarg.
    s = _clearable(ante=8, blind_index=2, hands_left=1)
    s2, info = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    assert s2.done and s2.won and s2.phase == Phase.WON


def test_legal_actions_in_shop():
    s = _clearable(money=100, hands_left=1)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    # Pin an affordable JOKER offer so BUY is guaranteed regardless of the random roll.
    s2 = _force_offers(s2, [ShopItem(int(ShopKind.JOKER), int(JokerType.BLUEPRINT), 10)])
    verbs = {a[0] for a in legal_actions(s2)}
    assert Verb.LEAVE_SHOP in verbs
    assert Verb.BUY in verbs     # affordable joker offer exists
    assert Verb.REROLL in verbs


def test_shop_action_cap_forces_leave():
    """REORDER is a free no-op; without a bound the greedy agent loops it forever
    (env never returns done). The per-visit cap forces LEAVE_SHOP -> progress."""
    from balatro_rl.engine.engine import SHOP_ACTION_CAP
    s = _clearable(money=10, hands_left=1,
                   jokers=(JokerState(JokerType.JOKER), JokerState(JokerType.BARON)))
    cur, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))      # clear -> shop
    assert cur.phase == Phase.SHOP and cur.shop_steps == 0
    for _ in range(SHOP_ACTION_CAP):
        assert Verb.REORDER in {a[0] for a in legal_actions(cur)}   # still allowed below cap
        cur, _ = step(cur, (Verb.REORDER, (0, 1)))
    assert cur.shop_steps == SHOP_ACTION_CAP
    assert legal_actions(cur) == [(Verb.LEAVE_SHOP, 0)]             # loop impossible
    nxt, _ = step(cur, (Verb.LEAVE_SHOP, 0))
    assert nxt.phase == Phase.PLAYING and nxt.shop_steps == 0       # fresh blind resets it
