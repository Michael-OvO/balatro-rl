from balatro_rl.engine.rng import RNG
from balatro_rl.engine.jokers.base import JokerType, JokerState, Rarity, REGISTRY
import balatro_rl.engine.jokers.library  # noqa: F401
from balatro_rl.engine.shop import generate_offers, joker_cost, reroll_cost, sell_value


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
    for js in offers_a:
        assert isinstance(js, JokerState)
        assert REGISTRY[js.type].rarity != Rarity.LEGENDARY   # never legendary in shop


def test_generate_offers_advances_rng():
    rng = RNG.from_seed(7)
    _, rng2 = generate_offers(rng, 2)
    assert rng2 != rng
