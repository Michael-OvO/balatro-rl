"""Batch B2b-i: event-scaling enhancement jokers (Glass Joker, Lucky Cat) + the
per-hand HandEvents plumbing that feeds their scaling counters.

Values verified against balatrowiki.org:
  - Glass Joker (120, Uncommon $6): gains X0.75 Mult per Glass card DESTROYED, starts
    X1 -> xmult = 1 + 0.75 * (#glass shattered so far). Shatter happens AFTER scoring,
    so this hand's shatters scale the NEXT hand (independent reads the persistent
    counter only).
  - Lucky Cat (91, Uncommon $6): gains X0.25 Mult every time a Lucky card SUCCESSFULLY
    triggers (a card hitting BOTH money + mult counts as ONE). Lucky triggers fire in the
    scored-card phase (before the joker phase), so THIS hand's triggers scale THIS hand
    (independent reads persistent counter + this-hand ctx.lucky_triggers).

PLUMBING: scoring surfaces per-hand events on ScoreResult (lucky_triggered; glass count =
len(destroyed_idx)); engine.step folds them into each joker's on_hand_events(js, events)
AFTER on_play. A hand with no enhancement events produces an all-zero HandEvents and the
fold is skipped -> byte-identical to the pre-Batch-8 game (no counter touched, no rng).
"""
import dataclasses

from balatro_rl.engine.cards import Card, Enhancement
from balatro_rl.engine.engine import reset, step, Verb
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import (
    HandEvents, JokerState, JokerType, REGISTRY, Rarity, ScoreContext,
)
from balatro_rl.engine.rng import RNG
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0, enh=0):
    return Card(rank=rank, suit=suit, enhancement=enh)


def J(t, counter=0.0):
    return JokerState(type=t, counter=counter)


# ============================================================================
# HandEvents value type + on_hand_events default
# ============================================================================

def test_hand_events_defaults_all_zero():
    e = HandEvents()
    assert e.glass_destroyed == 0 and e.lucky_triggered == 0


def test_on_hand_events_default_is_noop():
    # A joker that doesn't override on_hand_events returns its JokerState unchanged.
    js = J(JokerType.JOKER, counter=5.0)
    out = REGISTRY[JokerType.JOKER].on_hand_events(js, HandEvents(glass_destroyed=9))
    assert out == js


# ============================================================================
# Glass Joker (120): X(1 + 0.75 * destroyed), NEXT-hand scaling
# ============================================================================

def test_glass_joker_independent_uses_persistent_counter_only():
    eff = REGISTRY[JokerType.GLASS_JOKER].independent(ScoreContext(), J(JokerType.GLASS_JOKER, 3))
    assert abs(eff.xmult - 3.25) < 1e-9          # 1 + 0.75*3, matches wiki "3 destroyed -> X3.25"


def test_glass_joker_x1_at_counter_zero():
    eff = REGISTRY[JokerType.GLASS_JOKER].independent(ScoreContext(), J(JokerType.GLASS_JOKER, 0))
    assert eff.xmult == 1.0


def test_glass_joker_on_hand_events_adds_destroyed_to_counter():
    js2 = REGISTRY[JokerType.GLASS_JOKER].on_hand_events(
        J(JokerType.GLASS_JOKER, 2), HandEvents(glass_destroyed=2))
    assert js2.counter == 4.0


def test_glass_joker_ignores_lucky_events():
    # Glass Joker scales on glass only; a lucky-trigger event must not move its counter.
    js2 = REGISTRY[JokerType.GLASS_JOKER].on_hand_events(
        J(JokerType.GLASS_JOKER, 1), HandEvents(lucky_triggered=4))
    assert js2.counter == 1.0


def test_glass_joker_counter_increments_via_engine_on_shatter():
    # End-to-end: a Glass card shatters during engine.step -> HandEvents.glass_destroyed=1
    # reaches Glass Joker's on_hand_events -> counter +1, and the card leaves master_deck.
    seed = _seed_with_first_roll(lambda r: r < 1 / 4)   # the shatter roll (1-in-4) succeeds
    st = reset(seed=0, card_mods={0: {"enhancement": Enhancement.GLASS}})
    glass = st.master_deck[0]
    st = dataclasses.replace(
        st, jokers=(J(JokerType.GLASS_JOKER),),
        hand=(glass, C(7, 1), C(9, 2), C(2, 3), C(4, 0)),
        required=10_000_000,            # don't clear -> stay in the blind (PLAY branch)
        rng=RNG.from_seed(seed))        # shatter roll is the first rng draw this hand
    nxt, _info = step(st, (Verb.PLAY, (0,)))
    assert nxt.jokers[0].counter == 1.0
    assert len(nxt.master_deck) == len(st.master_deck) - 1


# ============================================================================
# Lucky Cat (91): X(1 + 0.25 * triggers), SAME-hand scaling
# ============================================================================

def test_lucky_cat_independent_reads_persistent_plus_this_hand():
    js = J(JokerType.LUCKY_CAT, 3)
    eff = REGISTRY[JokerType.LUCKY_CAT].independent(ScoreContext(lucky_triggers=2), js)
    assert abs(eff.xmult - (1 + 0.25 * (3 + 2))) < 1e-9   # 2.25


def test_lucky_cat_on_hand_events_persists_triggers():
    js2 = REGISTRY[JokerType.LUCKY_CAT].on_hand_events(
        J(JokerType.LUCKY_CAT, 1), HandEvents(lucky_triggered=3))
    assert js2.counter == 4.0


def test_lucky_cat_ignores_glass_events():
    js2 = REGISTRY[JokerType.LUCKY_CAT].on_hand_events(
        J(JokerType.LUCKY_CAT, 2), HandEvents(glass_destroyed=5))
    assert js2.counter == 2.0


def test_lucky_card_trigger_counted_on_score_result():
    # A Lucky card whose mult roll (1-in-5) succeeds -> ScoreResult.lucky_triggered == 1.
    seed = _seed_with_first_roll(lambda r: r < 1 / 5)
    res = score_play([C(7, 0, Enhancement.LUCKY)], rng=RNG.from_seed(seed))
    assert res.lucky_triggered == 1


def test_lucky_card_no_trigger_not_counted():
    # Neither roll succeeds (first roll >= 1/5 covers the mult roll; pick a seed whose
    # first TWO draws both miss their thresholds) -> lucky_triggered == 0.
    seed = _seed_with_first_two_rolls(lambda a, b: a >= 1 / 5 and b >= 1 / 15)
    res = score_play([C(7, 0, Enhancement.LUCKY)], rng=RNG.from_seed(seed))
    assert res.lucky_triggered == 0


def test_lucky_card_both_rolls_counts_as_one_trigger():
    # Wiki: a card triggering BOTH money and mult counts as ONE activation for Lucky Cat.
    seed = _seed_with_first_two_rolls(lambda a, b: a < 1 / 5 and b < 1 / 15)
    res = score_play([C(7, 0, Enhancement.LUCKY)], rng=RNG.from_seed(seed))
    assert res.lucky_triggered == 1


def test_lucky_cat_scales_same_hand_vs_baseline():
    # Same seed: with Lucky Cat the hand's mult is exactly X1.25 the no-Lucky-Cat mult
    # (one lucky trigger this hand -> 1 + 0.25*1). The lucky +20 mult is identical in both.
    seed = _seed_with_first_roll(lambda r: r < 1 / 5)
    lucky = C(7, 0, Enhancement.LUCKY)
    base = score_play([lucky], rng=RNG.from_seed(seed))
    with_cat = score_play([lucky], jokers=(J(JokerType.LUCKY_CAT),), rng=RNG.from_seed(seed))
    assert base.lucky_triggered == 1
    assert abs(with_cat.mult - base.mult * 1.25) < 1e-9


# ============================================================================
# byte-compat: an unmodified hand produces zero events and touches no counter/rng
# ============================================================================

def test_unmodified_hand_has_zero_events_and_rng_unchanged():
    res = score_play([C(5, 0), C(5, 1), C(7, 2)], rng=RNG.from_seed(42))
    assert res.lucky_triggered == 0 and res.destroyed_idx == ()
    assert res.rng == RNG.from_seed(42)        # no extra rng drawn


def test_scaling_jokers_dont_move_without_events_through_engine():
    # Glass Joker + Lucky Cat present but the played hand has no glass/lucky -> counters stay 0.
    st = reset(seed=1)
    st = dataclasses.replace(
        st, jokers=(J(JokerType.GLASS_JOKER), J(JokerType.LUCKY_CAT)),
        hand=(C(5, 0), C(5, 1), C(7, 2), C(9, 3), C(2, 0)), required=10_000_000)
    nxt, _info = step(st, (Verb.PLAY, (0, 1)))
    assert nxt.jokers[0].counter == 0.0 and nxt.jokers[1].counter == 0.0


# ============================================================================
# rarity / cost
# ============================================================================

def test_batch8_jokers_declare_rarity_and_cost():
    for jt, rar, cost in ((JokerType.GLASS_JOKER, Rarity.UNCOMMON, 6),
                          (JokerType.LUCKY_CAT, Rarity.UNCOMMON, 6)):
        eff = REGISTRY[jt]
        assert eff.rarity == rar and eff.cost == cost


# ============================================================================
# helpers
# ============================================================================

def _seed_with_first_roll(pred):
    for seed in range(20000):
        r, _ = RNG.from_seed(seed).random()
        if pred(r):
            return seed
    raise AssertionError("no seed found")


def _seed_with_first_two_rolls(pred):
    for seed in range(20000):
        rng = RNG.from_seed(seed)
        a, rng = rng.random()
        b, rng = rng.random()
        if pred(a, b):
            return seed
    raise AssertionError("no seed found")
