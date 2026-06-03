"""Batch 4 jokers that use the on_discard hook: Faceless Joker, Green Joker, Ramen.
Values verified against balatrowiki.org.
"""
import dataclasses

from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import reset, step, Verb
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.hands import evaluate
from balatro_rl.engine.jokers.base import (
    JokerType, JokerState, REGISTRY, Rarity, aggregate_rules,
)
from balatro_rl.engine.rng import RNG
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


class _FakeState:
    """Minimal stand-in carrying the owned-joker tuple (Faceless reads state.jokers
    for Pareidolia via aggregate_rules)."""
    def __init__(self, jokers):
        self.jokers = tuple(jokers)


def _discard(js, discarded, others=()):
    """Mimic the engine's on_discard fold for a single joker (rules from owned set)."""
    state = _FakeState((js,) + tuple(others))
    eff = REGISTRY[js.type]
    return eff.on_discard(state, list(discarded), js, RNG.from_seed(1))


# --- Faceless Joker (57): earn $5 if >=3 face cards discarded at once ---  wiki: /w/Faceless_Joker

def test_faceless_joker_rarity_cost():
    eff = REGISTRY[JokerType.FACELESS_JOKER]
    assert eff.rarity == Rarity.COMMON and eff.cost == 4


def test_faceless_joker_pays_5_on_three_face_cards():
    js = JokerState(JokerType.FACELESS_JOKER)
    js2, money, _ = _discard(js, [C(11), C(12), C(13)])  # J Q K
    assert money == 5 and js2 is js


def test_faceless_joker_pays_5_on_more_than_three_faces():
    js = JokerState(JokerType.FACELESS_JOKER)
    js2, money, _ = _discard(js, [C(11), C(12), C(13), C(13), C(2)])  # 4 faces
    assert money == 5


def test_faceless_joker_no_pay_under_three_faces():
    js = JokerState(JokerType.FACELESS_JOKER)
    _, money, _ = _discard(js, [C(11), C(12), C(2), C(3)])  # only 2 faces
    assert money == 0


def test_faceless_joker_respects_pareidolia():
    # With Pareidolia every card is a face card -> any 3 discards trigger it.
    js = JokerState(JokerType.FACELESS_JOKER)
    pare = JokerState(JokerType.PAREIDOLIA)
    _, money, _ = _discard(js, [C(2), C(3), C(4)], others=(pare,))
    assert money == 5


# --- Green Joker (58): +1 Mult per hand played, -1 per discard, floor 0 ---  wiki: /w/Green_Joker

def test_green_joker_rarity_cost():
    eff = REGISTRY[JokerType.GREEN_JOKER]
    assert eff.rarity == Rarity.COMMON and eff.cost == 4


def test_green_joker_starts_at_zero_mult():
    js = JokerState(JokerType.GREEN_JOKER, counter=0.0)
    res = score_play([C(14), C(7), C(2)], jokers=(js,))  # high card, base mult 1
    assert res.mult == 1.0


def test_green_joker_increments_per_hand_played():
    js = JokerState(JokerType.GREEN_JOKER, counter=0.0)
    eff = REGISTRY[JokerType.GREEN_JOKER]
    played = [C(2), C(3), C(4)]
    _, scoring_idx = evaluate(played, aggregate_rules((js,)))
    js = eff.on_play(None, played, list(scoring_idx), aggregate_rules((js,)), js)
    assert js.counter == 1.0
    js = eff.on_play(None, played, list(scoring_idx), aggregate_rules((js,)), js)
    assert js.counter == 2.0


def test_green_joker_decrements_per_discard():
    js = JokerState(JokerType.GREEN_JOKER, counter=3.0)
    js2, money, _ = _discard(js, [C(2), C(3), C(4)])  # 3 cards, but -1 per DISCARD action
    assert money == 0
    assert js2.counter == 2.0


def test_green_joker_floored_at_zero():
    js = JokerState(JokerType.GREEN_JOKER, counter=0.0)
    js2, _, _ = _discard(js, [C(2)])
    assert js2.counter == 0.0


def test_green_joker_applies_counter_as_mult():
    # counter 4 -> +4 mult. High-card Ace: 16 chips, mult 1+4=5 -> 80.
    js = JokerState(JokerType.GREEN_JOKER, counter=4.0)
    res = score_play([C(14), C(7), C(2)], jokers=(js,))
    assert res.mult == 5.0 and res.score == 80


# --- Ramen (100): X2 Mult, -X0.01 per card discarded, destroyed at 100 discards ---  wiki: /w/Ramen

def test_ramen_rarity_cost():
    eff = REGISTRY[JokerType.RAMEN]
    assert eff.rarity == Rarity.UNCOMMON and eff.cost == 6


def test_ramen_starts_at_x2():
    js = JokerState(JokerType.RAMEN, counter=0.0)
    res = score_play([C(14), C(7), C(2)], jokers=(js,))  # 16 chips, mult 1 -> *2 = 2 -> 32
    assert res.mult == 2.0 and res.score == 32


def test_ramen_loses_per_card_discarded():
    js = JokerState(JokerType.RAMEN, counter=0.0)
    js2, money, _ = _discard(js, [C(2), C(3), C(4)])  # 3 cards
    assert money == 0
    assert js2.counter == 3.0
    # xmult now 2.0 - 0.01*3 = 1.97
    res = score_play([C(14), C(7), C(2)], jokers=(js2,))
    assert abs(res.mult - 1.97) < 1e-9


def test_ramen_counter_reaches_100():
    eff = REGISTRY[JokerType.RAMEN]
    js = JokerState(JokerType.RAMEN, counter=99.0)
    js2, _, _ = eff.on_discard(None, [C(2)], js, RNG.from_seed(1))
    assert js2.counter == 100.0


def test_ramen_destroy_when_counter_at_or_above_100():
    eff = REGISTRY[JokerType.RAMEN]
    assert eff.destroy_when(JokerState(JokerType.RAMEN, counter=100.0)) is True
    assert eff.destroy_when(JokerState(JokerType.RAMEN, counter=101.0)) is True
    assert eff.destroy_when(JokerState(JokerType.RAMEN, counter=99.0)) is False


def test_ramen_destroyed_in_engine_after_100th_card():
    # Engine fold must drop Ramen when its counter crosses 100 (X1 reached -> eaten).
    state = reset(seed=11)
    js = JokerState(JokerType.RAMEN, counter=98.0)
    state = dataclasses.replace(state, jokers=(js,))
    nxt, _ = step(state, (Verb.DISCARD, (0, 1)))  # 2 cards -> counter 100 -> destroyed
    assert nxt.jokers == ()


def test_default_destroy_when_is_false():
    from balatro_rl.engine.jokers.base import JokerEffect
    eff = JokerEffect()
    assert eff.destroy_when(JokerState(JokerType.JOKER, counter=999.0)) is False
