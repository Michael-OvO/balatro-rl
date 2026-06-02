import dataclasses

import pytest

from balatro_rl.engine.cards import Card
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.state import GameState, Phase


def make_state(**overrides) -> GameState:
    base = dict(
        deck=(Card(2, 0),), hand=(Card(3, 0),),
        ante=1, blind_index=0, round_score=0, required=300,
        hands_left=4, discards_left=3, hand_size=8,
        levels=tuple([1] * 12), money=4,
        rng=RNG.from_seed(0), phase=Phase.PLAYING, done=False, won=False,
    )
    base.update(overrides)
    return GameState(**base)


def test_state_is_frozen():
    s = make_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.money = 99


def test_levels_has_twelve_entries():
    s = make_state()
    assert len(s.levels) == 12


def test_replace_produces_new_state():
    s = make_state(money=4)
    s2 = dataclasses.replace(s, money=10)
    assert s.money == 4 and s2.money == 10
