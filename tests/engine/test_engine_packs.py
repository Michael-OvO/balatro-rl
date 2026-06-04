"""Phase E3 — engine OPEN_PACK sub-phase (buy pack -> pick K-of-M -> back to SHOP).

ENGINE-FIRST / agent BLIND: legal_actions in SHOP never emits Verb.OPEN, so the agent
never enters OPEN_PACK. Packs are reachable only via a direct engine.step((Verb.OPEN, i)).
The agent wiring (obs/action widening) is the later E5.
"""
import dataclasses

from balatro_rl.engine.cards import Card
from balatro_rl.engine.consumables import ConsumableKind
from balatro_rl.engine.engine import JOKER_SLOTS, Verb, legal_actions, reset, step
from balatro_rl.engine.jokers.base import JokerState, JokerType
from balatro_rl.engine.packs import (
    Pack, PackItem, PackItemKind, PackKind, PackSize, open_pack,
)
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.state import Phase


def _shop(seed=1, **over):
    """Reach the SHOP by clearing a Small blind with a known four-of-a-kind."""
    s = reset(seed=seed)
    hand = (Card(13, 0), Card(13, 1), Card(13, 2), Card(13, 3), Card(2, 0),
            Card(3, 0), Card(4, 0), Card(5, 0))
    s = dataclasses.replace(s, hand=hand, required=10, **over)
    s2, _ = step(s, (Verb.PLAY, (0, 1, 2, 3)))
    assert s2.phase == Phase.SHOP
    return s2


def _arcana_mega():
    return Pack(kind=int(PackKind.ARCANA), size=int(PackSize.MEGA), cost=8)


def test_shop_generates_pack_offers():
    s = _shop(money=100, hands_left=1)
    assert len(s.pack_offers) == 2
    for p in s.pack_offers:
        assert isinstance(p, Pack)
        assert p.kind in (PackKind.ARCANA, PackKind.CELESTIAL, PackKind.BUFFOON)


def test_buy_pack_deducts_cost_and_enters_open_pack():
    s = _shop(money=100, hands_left=1)
    s = dataclasses.replace(s, pack_offers=(_arcana_mega(),))
    s2, info = step(s, (Verb.OPEN, 0))
    assert info["verb"] == "open"
    assert s2.phase == Phase.OPEN_PACK
    assert s2.money == s.money - 8           # Mega costs $8
    assert s2.pack_offers == ()              # the bought pack left the shop
    assert s2.pack_picks == 2                # Mega picks 2
    assert len(s2.pack_open) == 5            # Mega shows 5


def test_buy_pack_unaffordable_asserts():
    s = _shop(money=2, hands_left=1)
    s = dataclasses.replace(s, pack_offers=(_arcana_mega(),))
    try:
        step(s, (Verb.OPEN, 0))
        assert False, "expected an affordability assertion"
    except AssertionError:
        pass


def test_arcana_mega_pick_two_adds_two_consumables_then_shop():
    s = _shop(money=100, hands_left=1)
    s = dataclasses.replace(s, pack_offers=(_arcana_mega(),),
                            consumables=(), consumable_slots=5)
    s2, _ = step(s, (Verb.OPEN, 0))
    assert s2.phase == Phase.OPEN_PACK and s2.pack_picks == 2
    s3, info = step(s2, (Verb.PICK, 0))
    assert info["verb"] == "pick"
    assert s3.phase == Phase.OPEN_PACK       # one pick remaining
    assert s3.pack_picks == 1
    assert len(s3.consumables) == 1
    assert s3.consumables[0].kind == int(ConsumableKind.TAROT)
    assert len(s3.pack_open) == 4            # picked item removed from the reveal
    s4, _ = step(s3, (Verb.PICK, 0))
    assert s4.phase == Phase.SHOP            # picks exhausted -> back to shop
    assert s4.pack_picks == 0
    assert s4.pack_open == ()
    assert len(s4.consumables) == 2


def test_buffoon_pick_one_adds_joker():
    s = _shop(money=100, hands_left=1, jokers=())
    pack = Pack(kind=int(PackKind.BUFFOON), size=int(PackSize.NORMAL), cost=4)
    s = dataclasses.replace(s, pack_offers=(pack,))
    s2, _ = step(s, (Verb.OPEN, 0))
    assert s2.phase == Phase.OPEN_PACK and s2.pack_picks == 1
    assert len(s2.pack_open) == 2
    want = s2.pack_open[0].payload          # the JokerState we'll pick
    s3, info = step(s2, (Verb.PICK, 0))
    assert info["verb"] == "pick"
    assert s3.phase == Phase.SHOP           # single pick exhausted
    assert len(s3.jokers) == 1
    assert s3.jokers[-1].type == want.type


def test_skip_pack_ends_picking_with_no_picks():
    s = _shop(money=100, hands_left=1)
    s = dataclasses.replace(s, pack_offers=(_arcana_mega(),), consumables=())
    s2, _ = step(s, (Verb.OPEN, 0))
    assert s2.phase == Phase.OPEN_PACK
    s3, info = step(s2, (Verb.SKIP_PACK, 0))
    assert info["verb"] == "skip_pack"
    assert s3.phase == Phase.SHOP
    assert s3.consumables == ()             # skipped -> nothing taken
    assert s3.pack_open == () and s3.pack_picks == 0


def test_pick_respects_joker_slot_cap():
    """A joker pick is illegal/withheld when 5 jokers are already held."""
    full = tuple(JokerState(JokerType.JOKER) for _ in range(JOKER_SLOTS))
    s = _shop(money=100, hands_left=1, jokers=full)
    pack = Pack(kind=int(PackKind.BUFFOON), size=int(PackSize.NORMAL), cost=4)
    s = dataclasses.replace(s, pack_offers=(pack,))
    s2, _ = step(s, (Verb.OPEN, 0))
    assert s2.phase == Phase.OPEN_PACK
    # No PICK is legal (all items are jokers; slots full); only SKIP_PACK remains.
    verbs = {a[0] for a in legal_actions(s2)}
    assert Verb.PICK not in verbs
    assert Verb.SKIP_PACK in verbs


def test_pick_respects_consumable_slot_cap():
    s = _shop(money=100, hands_left=1)
    # No free consumable slots: 2 held, cap 2.
    from balatro_rl.engine.consumables import planet, PlanetType
    held = (planet(PlanetType.PLUTO), planet(PlanetType.MERCURY))
    s = dataclasses.replace(s, pack_offers=(_arcana_mega(),),
                            consumables=held, consumable_slots=2)
    s2, _ = step(s, (Verb.OPEN, 0))
    verbs = {a[0] for a in legal_actions(s2)}
    assert Verb.PICK not in verbs           # no free consumable slot -> no pick
    assert Verb.SKIP_PACK in verbs


def test_legal_actions_open_pack_emits_pick_and_skip():
    s = _shop(money=100, hands_left=1)
    s = dataclasses.replace(s, pack_offers=(_arcana_mega(),), consumable_slots=5)
    s2, _ = step(s, (Verb.OPEN, 0))
    actions = legal_actions(s2)
    picks = sorted(a[1] for a in actions if a[0] == Verb.PICK)
    assert picks == [0, 1, 2, 3, 4]         # all 5 revealed items pickable (slots free)
    assert (Verb.SKIP_PACK, 0) in actions
    # The agent never sees OPEN in the shop (blind), and OPEN_PACK doesn't leak PLAY/BUY etc.
    assert all(a[0] in (Verb.PICK, Verb.SKIP_PACK) or a[0] == Verb.USE for a in actions)


def test_legal_actions_shop_offers_open_when_affordable():
    # E5: the policy now sees pack offers — OPEN is legal when the pack is affordable...
    s = dataclasses.replace(_shop(money=100, hands_left=1), pack_offers=(_arcana_mega(),))
    assert (Verb.OPEN, 0) in legal_actions(s)
    # ...and withheld when it isn't (Mega costs $8; set money AFTER cash-out so it sticks).
    poor = dataclasses.replace(_shop(hands_left=1), money=5, pack_offers=(_arcana_mega(),))
    assert all(a[0] != Verb.OPEN for a in legal_actions(poor))


def test_open_pack_full_consumes_remaining_after_picks():
    """After exhausting picks the engine returns to SHOP so shopping can continue."""
    s = _shop(money=100, hands_left=1, consumable_slots=5)
    s = dataclasses.replace(s, pack_offers=(_arcana_mega(),), consumables=())
    s2, _ = step(s, (Verb.OPEN, 0))
    s3, _ = step(s2, (Verb.PICK, 0))
    s4, _ = step(s3, (Verb.PICK, 0))
    assert s4.phase == Phase.SHOP
    # Still in the shop with leftover pack offers gone, ready for more shop actions.
    assert (Verb.LEAVE_SHOP, 0) in legal_actions(s4)
