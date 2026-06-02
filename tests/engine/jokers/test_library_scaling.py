# tests/engine/jokers/test_library_scaling.py
import dataclasses
from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.hands import evaluate, is_face
from balatro_rl.engine.jokers.base import (
    JokerType, JokerState, REGISTRY, aggregate_rules,
)
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def _play_update(js, played):
    """Mimic engine.step's lifecycle call: compute scoring + rules, then on_play."""
    rules = aggregate_rules((js,))
    _, scoring_idx = evaluate(list(played), rules)
    eff = REGISTRY[js.type]
    return eff.on_play(None, list(played), list(scoring_idx), rules, js)


def test_ride_the_bus_increments_on_faceless_hand():
    js = JokerState(JokerType.RIDE_THE_BUS, counter=0.0)
    js = _play_update(js, [C(2), C(2), C(7), C(9), C(3)])   # no face cards
    assert js.counter == 1.0
    js = _play_update(js, [C(5), C(5), C(7), C(9), C(3)])   # still faceless
    assert js.counter == 2.0


def test_ride_the_bus_resets_on_scoring_face():
    js = JokerState(JokerType.RIDE_THE_BUS, counter=5.0)
    js = _play_update(js, [C(13), C(13), C(7), C(9), C(3)])  # kings score -> reset
    assert js.counter == 0.0


def test_ride_the_bus_applies_counter_as_mult():
    # counter 3 -> +3 mult. Pair of 3s: 16 chips, mult 2+3=5 -> 80.
    # Use mixed suits on kickers to avoid unintended flush.
    js = JokerState(JokerType.RIDE_THE_BUS, counter=3.0)
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=(js,))
    assert res.mult == 5.0 and res.score == 80


def test_pareidolia_suppresses_ride_the_bus():
    # With Pareidolia all cards are face -> a faceless-looking hand still "has face" -> reset.
    rtb = JokerState(JokerType.RIDE_THE_BUS, counter=4.0)
    pare = JokerState(JokerType.PAREIDOLIA)
    rules = aggregate_rules((rtb, pare))
    played = [C(2), C(2), C(7), C(9), C(3)]
    _, scoring_idx = evaluate(played, rules)
    rtb2 = REGISTRY[JokerType.RIDE_THE_BUS].on_play(None, played, list(scoring_idx), rules, rtb)
    assert rtb2.counter == 0.0
