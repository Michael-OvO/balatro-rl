import dataclasses
from balatro_rl.engine.jokers.base import JokerType, JokerState, REGISTRY
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.shop import sell_value
import balatro_rl.engine.jokers.library  # noqa: F401


def test_golden_joker_pays_4_at_round_end():  # wiki: /w/Golden_Joker
    eff = REGISTRY[JokerType.GOLDEN_JOKER]
    js = JokerState(type=JokerType.GOLDEN_JOKER)
    js2, money_delta, destroy, _ = eff.on_round_end(None, js, RNG.from_seed(1))
    assert money_delta == 4 and destroy is False and js2 is js


def test_egg_gains_3_sell_value_each_round():  # wiki: /w/Egg
    eff = REGISTRY[JokerType.EGG]
    js = JokerState(type=JokerType.EGG)
    js2, money_delta, destroy, _ = eff.on_round_end(None, js, RNG.from_seed(1))
    assert money_delta == 0 and destroy is False
    assert js2.sell_bonus == 3
    # sell value reflects the bonus (Egg cost 4 -> floor=2, +3 = 5)
    assert sell_value(JokerType.EGG, js2.sell_bonus) == 5


def test_cavendish_usually_survives_round_end():  # wiki: /w/Cavendish  (1 in 1000)
    eff = REGISTRY[JokerType.CAVENDISH]
    js = JokerState(type=JokerType.CAVENDISH)
    # seed 1's first random() is >= 0.001, so it survives and rng advances.
    js2, money_delta, destroy, rng2 = eff.on_round_end(None, js, RNG.from_seed(1))
    assert destroy is False and money_delta == 0
    assert rng2 != RNG.from_seed(1)   # rng consumed by the roll


def test_cavendish_destroys_on_low_roll():
    eff = REGISTRY[JokerType.CAVENDISH]
    js = JokerState(type=JokerType.CAVENDISH)
    # Find a seed whose first random() < 0.001 by scanning; assert destroy fires.
    from balatro_rl.engine.rng import RNG as _RNG
    seed = next(s for s in range(100000) if _RNG.from_seed(s).random()[0] < 0.001)
    _, _, destroy, _ = eff.on_round_end(None, js, _RNG.from_seed(seed))
    assert destroy is True
