import dataclasses
from balatro_rl.engine.state import GameState, Phase
from balatro_rl.engine.engine import reset


def test_phase_has_shop():
    assert Phase.SHOP.name == "SHOP"


def test_reset_shop_fields_default_empty():
    s = reset(seed=1)
    assert s.shop_offers == ()
    assert s.rerolls_done == 0
    assert s.phase == Phase.PLAYING
