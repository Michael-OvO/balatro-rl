"""Batch 1 retrigger (Sock and Buskin), economy/destroy (Gros Michel), and
scaling (Runner, Ice Cream) jokers."""
import dataclasses

from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.hands import evaluate
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.jokers.base import (
    JokerType, JokerState, REGISTRY, aggregate_rules,
)
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


def _play_update(js, played):
    rules = aggregate_rules((js,))
    _, scoring_idx = evaluate(list(played), rules)
    eff = REGISTRY[js.type]
    return eff.on_play(None, list(played), list(scoring_idx), rules, js)


# --- Sock and Buskin: retrigger each scored face card ---

def test_sock_and_buskin_retriggers_faces():
    # Pair of Kings (face) scored twice each via retrigger. Kicker faces? use non-faces.
    # base pair (10,2). Without joker chips = 10 + 10 + 10 = 30.
    # Each King re-scores once more -> +10 chips each = +20 -> chips 50.  # wiki: /w/Sock_and_Buskin
    res = score_play([C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)],
                     jokers=(J(JokerType.SOCK_AND_BUSKIN),))
    assert res.chips == 50


def test_sock_and_buskin_ignores_non_faces():
    # Pair of 3s: no faces -> no retrigger. chips = 10 + 3 + 3 = 16.
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)],
                     jokers=(J(JokerType.SOCK_AND_BUSKIN),))
    assert res.chips == 16


def test_sock_and_buskin_refires_smiley_face():
    # Smiley Face +5 mult per face trigger; Sock and Buskin doubles each King's trigger.
    # Two kings, each fires twice -> 4 triggers x +5 = +20 mult -> mult 22.  # wiki: /w/Sock_and_Buskin
    res = score_play([C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)],
                     jokers=(J(JokerType.SOCK_AND_BUSKIN), J(JokerType.SMILEY_FACE)))
    assert res.mult == 2.0 + 20.0


# --- Gros Michel: +15 Mult; 1 in 6 self-destroy at end of round ---

def test_gros_michel_plus_15_mult():
    # High card Ace: base (5,1) chips 16, mult 1 + 15 -> 16 -> 256.  # wiki: /w/Gros_Michel
    res = score_play([C(14, 0), C(7, 1), C(2, 2)], jokers=(J(JokerType.GROS_MICHEL),))
    assert res.mult == 16.0 and res.score == 16 * 16


def test_gros_michel_can_destroy_at_round_end():
    eff = REGISTRY[JokerType.GROS_MICHEL]
    js = J(JokerType.GROS_MICHEL)
    # Search seeds for a low roll (<1/6) and a high roll (>=1/6) to prove both paths.
    destroyed_seen = False
    survived_seen = False
    for seed in range(200):
        rng = RNG.from_seed(seed)
        roll, _ = rng.random()
        js2, money, destroy, _ = eff.on_round_end(None, js, RNG.from_seed(seed))
        assert money == 0
        assert destroy == (roll < 1 / 6)
        destroyed_seen |= destroy
        survived_seen |= not destroy
    assert destroyed_seen and survived_seen


# --- Runner: +15 Chips per played hand containing a Straight (start +0) ---

def test_runner_scales_on_straights():
    js = J(JokerType.RUNNER)
    assert js.counter == 0.0
    js = _play_update(js, [C(3, 0), C(4, 1), C(5, 2), C(6, 3), C(7, 0)])  # straight
    assert js.counter == 15.0
    js = _play_update(js, [C(13, 0), C(13, 1), C(7, 2), C(9, 3), C(2, 0)])  # no straight
    assert js.counter == 15.0
    js = _play_update(js, [C(2, 0), C(3, 1), C(4, 2), C(5, 3), C(6, 0)])  # straight again
    assert js.counter == 30.0


def test_runner_applies_counter_as_chips():
    # counter 30 -> +30 chips. High card Ace base chips 16 -> 46.  # wiki: /w/Runner
    js = JokerState(JokerType.RUNNER, counter=30.0)
    res = score_play([C(14, 0), C(7, 1), C(2, 2)], jokers=(js,))
    assert res.chips == 46


# --- Ice Cream: +100 Chips, -5 per hand played (counter starts 0) ---

def test_ice_cream_starts_at_100():
    js = J(JokerType.ICE_CREAM)
    res = score_play([C(14, 0), C(7, 1), C(2, 2)], jokers=(js,))
    # High card Ace base chips 16 + 100 -> 116.  # wiki: /w/Ice_Cream
    assert res.chips == 116


def test_ice_cream_decays_5_per_hand():
    js = J(JokerType.ICE_CREAM)
    js = _play_update(js, [C(14, 0), C(7, 1), C(2, 2)])  # counter 1 -> +95
    res = score_play([C(14, 0), C(7, 1), C(2, 2)], jokers=(js,))
    assert res.chips == 16 + 95
    js = _play_update(js, [C(14, 0), C(7, 1), C(2, 2)])  # counter 2 -> +90
    res = score_play([C(14, 0), C(7, 1), C(2, 2)], jokers=(js,))
    assert res.chips == 16 + 90


def test_ice_cream_floors_at_zero():
    # counter 25 -> 100 - 125 = -25 -> clamped to 0.
    js = JokerState(JokerType.ICE_CREAM, counter=25.0)
    res = score_play([C(14, 0), C(7, 1), C(2, 2)], jokers=(js,))
    assert res.chips == 16  # no bonus
