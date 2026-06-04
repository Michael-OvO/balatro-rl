from balatro_rl.engine.consumables import PlanetType
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.jokers.base import JokerType, Rarity, REGISTRY
import balatro_rl.engine.jokers.library  # noqa: F401
from balatro_rl.engine.shop import (
    CONSUMABLE_COST, ShopItem, ShopKind, generate_offers, item_cost, joker_cost,
    reroll_cost, sell_value,
)


def test_reroll_cost_scaling():  # base 5, +1 each, per economy-shop.md §6
    assert reroll_cost(0) == 5
    assert reroll_cost(1) == 6
    assert reroll_cost(3) == 8


def test_sell_value_floor_min_1():
    assert sell_value(JokerType.JOKER) == 1      # cost 2 -> floor(2/2)=1
    assert sell_value(JokerType.BARON) == 4      # cost 8 -> 4
    assert sell_value(JokerType.HACK) == 3       # cost 6 -> 3
    assert sell_value(JokerType.JOKER, sell_bonus=3) == 4   # +Egg bonus


def test_joker_cost_reads_registry():
    assert joker_cost(JokerType.BLUEPRINT) == 10
    assert joker_cost(JokerType.JOKER) == 2


def test_generate_offers_deterministic_and_valid():
    offers_a, _ = generate_offers(RNG.from_seed(42), 2)
    offers_b, _ = generate_offers(RNG.from_seed(42), 2)
    assert offers_a == offers_b          # deterministic per seed
    assert len(offers_a) == 2
    for item in offers_a:
        assert isinstance(item, ShopItem)
        assert item.kind in (ShopKind.JOKER, ShopKind.PLANET)   # E1 scope
        if item.kind == ShopKind.JOKER:
            jt = JokerType(item.type_id)
            assert REGISTRY[jt].rarity != Rarity.LEGENDARY      # never legendary in shop
            assert item.cost == joker_cost(jt)
        else:
            assert PlanetType(item.type_id) in PlanetType       # valid planet
            assert item.cost == CONSUMABLE_COST                 # wiki: Planet costs $3


def test_generate_offers_advances_rng():
    rng = RNG.from_seed(7)
    _, rng2 = generate_offers(rng, 2)
    assert rng2 != rng


def test_item_cost_joker_and_planet():
    joker = ShopItem(kind=int(ShopKind.JOKER), type_id=int(JokerType.BLUEPRINT), cost=10)
    planet = ShopItem(kind=int(ShopKind.PLANET), type_id=int(PlanetType.PLUTO), cost=3)
    assert item_cost(joker) == joker_cost(JokerType.BLUEPRINT) == 10
    assert item_cost(planet) == CONSUMABLE_COST == 3


def test_kind_weighting_is_roughly_20_to_4():
    """Wiki composition (balatrowiki.org/w/Shop): Joker 20 / Planet 4 in E1 scope ->
    ~20/24 jokers, ~4/24 planets over many slots. Allow generous slack on the sample."""
    rng = RNG.from_seed(123)
    n = 6000
    offers, _ = generate_offers(rng, n)
    jokers = sum(1 for o in offers if o.kind == ShopKind.JOKER)
    planets = sum(1 for o in offers if o.kind == ShopKind.PLANET)
    assert jokers + planets == n                       # only the two E1 kinds appear
    assert all(o.kind in (ShopKind.JOKER, ShopKind.PLANET) for o in offers)
    # Expected fractions 20/24 ~= 0.833 and 4/24 ~= 0.167; ±0.05 absolute tolerance.
    assert abs(jokers / n - 20 / 24) < 0.05
    assert abs(planets / n - 4 / 24) < 0.05
    assert all(o.cost == CONSUMABLE_COST for o in offers if o.kind == ShopKind.PLANET)
