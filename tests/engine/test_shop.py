from balatro_rl.engine.consumables import (
    DEFERRED_TAROTS, IMPLEMENTED_TAROTS, PlanetType, TarotType,
)
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
        assert item.kind in (ShopKind.JOKER, ShopKind.TAROT, ShopKind.PLANET)   # E2 scope
        if item.kind == ShopKind.JOKER:
            jt = JokerType(item.type_id)
            assert REGISTRY[jt].rarity != Rarity.LEGENDARY      # never legendary in shop
            assert item.cost == joker_cost(jt)
        elif item.kind == ShopKind.PLANET:
            assert PlanetType(item.type_id) in PlanetType       # valid planet
            assert item.cost == CONSUMABLE_COST                 # wiki: Planet costs $3
        else:
            tt = TarotType(item.type_id)                        # valid, implemented tarot
            assert tt not in DEFERRED_TAROTS
            assert item.cost == CONSUMABLE_COST                 # wiki: Tarot costs $3


def test_generate_offers_advances_rng():
    rng = RNG.from_seed(7)
    _, rng2 = generate_offers(rng, 2)
    assert rng2 != rng


def test_item_cost_joker_and_planet():
    joker = ShopItem(kind=int(ShopKind.JOKER), type_id=int(JokerType.BLUEPRINT), cost=10)
    planet = ShopItem(kind=int(ShopKind.PLANET), type_id=int(PlanetType.PLUTO), cost=3)
    assert item_cost(joker) == joker_cost(JokerType.BLUEPRINT) == 10
    assert item_cost(planet) == CONSUMABLE_COST == 3


def test_kind_weighting_is_roughly_20_4_4():
    """Wiki composition (balatrowiki.org/w/Shop): Joker 20 / Tarot 4 / Planet 4 (E2 scope) ->
    ~20/28 jokers, ~4/28 tarots, ~4/28 planets over many slots. Generous slack on the sample."""
    rng = RNG.from_seed(123)
    n = 6000
    offers, _ = generate_offers(rng, n)
    jokers = sum(1 for o in offers if o.kind == ShopKind.JOKER)
    tarots = sum(1 for o in offers if o.kind == ShopKind.TAROT)
    planets = sum(1 for o in offers if o.kind == ShopKind.PLANET)
    assert jokers + tarots + planets == n              # only the three E2 kinds appear
    assert all(o.kind in (ShopKind.JOKER, ShopKind.TAROT, ShopKind.PLANET) for o in offers)
    # Expected fractions 20/28 ~= 0.714, 4/28 ~= 0.143 each; ±0.05 absolute tolerance.
    assert abs(jokers / n - 20 / 28) < 0.05
    assert abs(tarots / n - 4 / 28) < 0.05
    assert abs(planets / n - 4 / 28) < 0.05
    assert all(o.cost == CONSUMABLE_COST for o in offers
               if o.kind in (ShopKind.TAROT, ShopKind.PLANET))
    # Tarot offers only ever roll IMPLEMENTED tarots (the two deferred ones never appear).
    assert all(TarotType(o.type_id) in IMPLEMENTED_TAROTS
               for o in offers if o.kind == ShopKind.TAROT)
