# tests/engine/test_engine_jokers.py
import dataclasses
from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import Verb, reset, step
from balatro_rl.engine.jokers.base import JokerType, JokerState
import balatro_rl.engine.jokers.library  # noqa: F401


def test_reset_starts_with_no_jokers():
    s = reset(seed=1)
    assert s.jokers == ()


def test_play_uses_jokers_in_score():
    s = reset(seed=1)
    # Force a known hand and a Joker(+4); play the pair.
    hand = (Card(13, 0), Card(13, 1), Card(3, 2), Card(7, 2), Card(9, 2),
            Card(2, 0), Card(4, 0), Card(5, 0))
    s = dataclasses.replace(s, hand=hand, jokers=(JokerState(JokerType.JOKER),))
    s2, info = step(s, (Verb.PLAY, (0, 1)))
    # Pair of Kings: chips 30, mult 2+4=6 -> 180.
    assert info["score"] == 180
    assert s2.round_score == 180


def test_ride_the_bus_counter_advances_through_step():
    s = reset(seed=1)
    hand = (Card(2, 0), Card(2, 1), Card(7, 2), Card(9, 2), Card(3, 2),
            Card(4, 0), Card(5, 0), Card(6, 0))
    s = dataclasses.replace(s, hand=hand,
                            jokers=(JokerState(JokerType.RIDE_THE_BUS, counter=0.0),),
                            required=10_000_000)  # don't clear; keep playing
    s2, _ = step(s, (Verb.PLAY, (0, 1)))           # faceless pair -> counter 1
    assert s2.jokers[0].counter == 1.0
