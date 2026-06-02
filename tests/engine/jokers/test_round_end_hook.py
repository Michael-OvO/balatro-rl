from balatro_rl.engine.jokers.base import JokerEffect, JokerType, JokerState
from balatro_rl.engine.rng import RNG


def test_default_on_round_end_is_noop():
    eff = JokerEffect()
    js = JokerState(type=JokerType.JOKER)
    rng = RNG.from_seed(1)
    js2, money_delta, destroy, rng2 = eff.on_round_end(None, js, rng)
    assert js2 is js
    assert money_delta == 0
    assert destroy is False
    assert rng2 is rng
