"""Phase E3 — booster pack model (packs.py): roll_pack / open_pack / layout / cost.

Engine-first: the agent stays BLIND to packs (legal_actions never offers Verb.OPEN),
so packs are reachable only via direct engine.step. These tests pin the pack VALUE
TYPES + content generation; the engine OPEN_PACK flow is tested in test_engine_packs.py.
"""
from balatro_rl.engine.consumables import ConsumableKind, IMPLEMENTED_TAROTS, PlanetType
from balatro_rl.engine.jokers.base import JokerState, JokerType
from balatro_rl.engine.packs import (
    PACK_LAYOUT, Pack, PackItemKind, PackKind, PackSize, open_pack, pack_cost, roll_pack,
)
from balatro_rl.engine.rng import RNG


def test_pack_layout_matches_wiki():
    # Arcana / Celestial: Normal (3,1) / Jumbo (5,1) / Mega (5,2)
    for k in (PackKind.ARCANA, PackKind.CELESTIAL):
        assert PACK_LAYOUT[(k, PackSize.NORMAL)] == (3, 1)
        assert PACK_LAYOUT[(k, PackSize.JUMBO)] == (5, 1)
        assert PACK_LAYOUT[(k, PackSize.MEGA)] == (5, 2)
    # Buffoon: Normal (2,1) / Jumbo (4,1) / Mega (4,2)
    assert PACK_LAYOUT[(PackKind.BUFFOON, PackSize.NORMAL)] == (2, 1)
    assert PACK_LAYOUT[(PackKind.BUFFOON, PackSize.JUMBO)] == (4, 1)
    assert PACK_LAYOUT[(PackKind.BUFFOON, PackSize.MEGA)] == (4, 2)


def test_pack_cost_by_size():
    assert pack_cost(PackSize.NORMAL) == 4
    assert pack_cost(PackSize.JUMBO) == 6
    assert pack_cost(PackSize.MEGA) == 8


def test_roll_pack_returns_valid_kinds_and_sizes():
    rng = RNG.from_seed(7)
    kinds = set()
    sizes = set()
    for _ in range(400):
        pack, rng = roll_pack(rng)
        assert isinstance(pack, Pack)
        assert pack.kind in (PackKind.ARCANA, PackKind.CELESTIAL, PackKind.BUFFOON)
        assert pack.size in (PackSize.NORMAL, PackSize.JUMBO, PackSize.MEGA)
        assert pack.cost == pack_cost(pack.size)
        assert (pack.kind, pack.size) in PACK_LAYOUT
        kinds.add(pack.kind)
        sizes.add(pack.size)
    # Over 400 rolls every kind and size should appear.
    assert kinds == {PackKind.ARCANA, PackKind.CELESTIAL, PackKind.BUFFOON}
    assert sizes == {PackSize.NORMAL, PackSize.JUMBO, PackSize.MEGA}


def test_roll_pack_is_deterministic():
    a, _ = roll_pack(RNG.from_seed(42))
    b, _ = roll_pack(RNG.from_seed(42))
    assert a == b


def test_open_arcana_pack_yields_tarot_items():
    pack = Pack(kind=int(PackKind.ARCANA), size=int(PackSize.MEGA), cost=8)
    items, picks, rng = open_pack(pack, RNG.from_seed(3))
    assert picks == 2                  # Mega picks 2
    assert len(items) == 5             # Mega shows 5
    for it in items:
        assert it.kind == PackItemKind.CONSUMABLE
        con = it.payload
        assert con.kind == int(ConsumableKind.TAROT)
        assert con.type_id in {int(t) for t in IMPLEMENTED_TAROTS}


def test_open_celestial_pack_yields_planet_items():
    pack = Pack(kind=int(PackKind.CELESTIAL), size=int(PackSize.NORMAL), cost=4)
    items, picks, rng = open_pack(pack, RNG.from_seed(5))
    assert picks == 1
    assert len(items) == 3
    for it in items:
        assert it.kind == PackItemKind.CONSUMABLE
        con = it.payload
        assert con.kind == int(ConsumableKind.PLANET)
        assert con.type_id in {int(p) for p in PlanetType}


def test_open_buffoon_pack_yields_joker_items():
    pack = Pack(kind=int(PackKind.BUFFOON), size=int(PackSize.NORMAL), cost=4)
    items, picks, rng = open_pack(pack, RNG.from_seed(9))
    assert picks == 1
    assert len(items) == 2             # Buffoon Normal shows 2
    for it in items:
        assert it.kind == PackItemKind.JOKER
        js = it.payload
        assert isinstance(js, JokerState)
        assert isinstance(js.type, (JokerType, int))


def test_open_pack_is_deterministic():
    pack = Pack(kind=int(PackKind.ARCANA), size=int(PackSize.JUMBO), cost=6)
    a_items, a_picks, _ = open_pack(pack, RNG.from_seed(11))
    b_items, b_picks, _ = open_pack(pack, RNG.from_seed(11))
    assert a_picks == b_picks == 1
    assert [i.payload for i in a_items] == [i.payload for i in b_items]
