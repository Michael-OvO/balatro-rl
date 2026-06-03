"""Batch 6 STEP 2: hand-play-count jokers. Values verified against balatrowiki.org:
  - Supernova (43, Common $5): independent, +Mult = times this poker hand has been
    played this RUN. The current play IS counted (retroactive + this hand): the
    first-ever play of a hand type gives +1 Mult.
  - Card Sharp (62, Uncommon $6): independent, X3 Mult if this poker hand has ALREADY
    been played this ROUND (a prior play this round; the current play is excluded).
  - Obelisk (75, Rare $8): scaling xMult, +X0.2 per consecutive hand played that is
    NOT (one of) your most-played run hand(s); resets to X1 (before scoring) when you
    play one that becomes the strict most-played hand. js.counter = #(+0.2 steps).

ScoreContext exposes the PRE-increment counts for the CURRENT hand_type
(hand_plays_run / hand_plays_round); engine.step increments AFTER the on_play fold.
"""
import dataclasses

import pytest

from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import reset, step, Verb
from balatro_rl.engine.hands import HandType, evaluate
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import (
    JokerType, JokerState, REGISTRY, Rarity, aggregate_rules,
)
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t, **kw):
    return JokerState(type=t, **kw)


def _pair_run_counts(times):
    """A length-12 run-count tuple with PAIR set to `times`."""
    counts = [0] * 12
    counts[int(HandType.PAIR)] = times
    return tuple(counts)


# ============================================================================
# rarity / cost
# ============================================================================

def test_batch6_jokers_declare_rarity_and_cost():
    expected = {
        JokerType.SUPERNOVA: (Rarity.COMMON, 5),
        JokerType.CARD_SHARP: (Rarity.UNCOMMON, 6),
        JokerType.OBELISK: (Rarity.RARE, 8),
    }
    for jt, (rar, cost) in expected.items():
        eff = REGISTRY[jt]
        assert eff.rarity == rar, jt
        assert eff.cost == cost, jt


# ============================================================================
# Supernova (43)  — +Mult = run plays of this hand type, INCLUDING current play
# ============================================================================

def test_supernova_first_ever_play_gives_plus_1_mult():
    # PRE-increment run count 0 -> first play counts -> +1 Mult.
    # Pair of 3s: base mult 2 -> 2 + 1 = 3.
    js = (J(JokerType.SUPERNOVA),)
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=js,
                     hand_plays_run=_pair_run_counts(0))
    assert res.mult == pytest.approx(3.0)


def test_supernova_includes_current_play_on_top_of_history():
    # Pair already played 4 times this run -> this 5th play gives +5 Mult.
    js = (J(JokerType.SUPERNOVA),)
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=js,
                     hand_plays_run=_pair_run_counts(4))
    assert res.mult == pytest.approx(2.0 + 5.0)  # base 2 + (4 prior + 1 current)


def test_supernova_only_counts_the_played_hand_type():
    # Run history is on FLUSH, but we play a PAIR -> PAIR count is 0 -> +1 only.
    counts = [0] * 12
    counts[int(HandType.FLUSH)] = 9
    js = (J(JokerType.SUPERNOVA),)
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=js,
                     hand_plays_run=tuple(counts))
    assert res.mult == pytest.approx(3.0)  # base 2 + 1 (first pair), flush history ignored


def test_supernova_end_to_end_through_engine_increments_each_play():
    # First play of a hand type -> +1; second play of the SAME type -> +2 (history grows).
    s = reset(seed=1)
    hand = (C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0))
    s = dataclasses.replace(s, hand=hand, required=10_000_000,
                            jokers=(J(JokerType.SUPERNOVA),))
    _, info1 = step(s, (Verb.PLAY, (0, 1, 2, 3, 4)))
    s2 = dataclasses.replace(s, hand=hand, hand_plays_run=_pair_run_counts(1),
                             hand_plays_round=_pair_run_counts(1))
    _, info2 = step(s2, (Verb.PLAY, (0, 1, 2, 3, 4)))
    assert info2["mult"] == info1["mult"] + 1


# ============================================================================
# Card Sharp (62)  — X3 if this hand type ALREADY played this ROUND (excl. current)
# ============================================================================

def test_card_sharp_no_xmult_on_first_play_this_round():
    # PRE-increment round count 0 -> not "already played" -> no X3.
    js = (J(JokerType.CARD_SHARP),)
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=js,
                     hand_plays_round=_pair_run_counts(0))
    assert res.mult == pytest.approx(2.0)  # base pair mult only


def test_card_sharp_x3_when_already_played_this_round():
    # Round count 1 (played once before) -> X3.  base mult 2 -> 6.
    js = (J(JokerType.CARD_SHARP),)
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=js,
                     hand_plays_round=_pair_run_counts(1))
    assert res.mult == pytest.approx(6.0)


def test_card_sharp_uses_round_count_not_run_count():
    # Played a lot this RUN but 0 this round -> no X3 (it's round-scoped).
    js = (J(JokerType.CARD_SHARP),)
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=js,
                     hand_plays_run=_pair_run_counts(9),
                     hand_plays_round=_pair_run_counts(0))
    assert res.mult == pytest.approx(2.0)


def test_card_sharp_only_the_scoring_hand_type_qualifies():
    # Round history is on TWO_PAIR; we play a PAIR -> PAIR round count 0 -> no X3.
    counts = [0] * 12
    counts[int(HandType.TWO_PAIR)] = 3
    js = (J(JokerType.CARD_SHARP),)
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=js,
                     hand_plays_round=tuple(counts))
    assert res.mult == pytest.approx(2.0)


def test_card_sharp_end_to_end_fires_on_second_play_this_round():
    s = reset(seed=1)
    hand = (C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0))
    s = dataclasses.replace(s, hand=hand, required=10_000_000,
                            jokers=(J(JokerType.CARD_SHARP),))
    _, info1 = step(s, (Verb.PLAY, (0, 1, 2, 3, 4)))   # first PAIR this round -> no X3
    s2 = dataclasses.replace(s, hand=hand, hand_plays_round=_pair_run_counts(1),
                             hand_plays_run=_pair_run_counts(1))
    _, info2 = step(s2, (Verb.PLAY, (0, 1, 2, 3, 4)))  # second PAIR -> X3
    assert info2["mult"] == pytest.approx(info1["mult"] * 3)


# ============================================================================
# Obelisk (75)  — scaling +X0.2 per non-most-played consecutive hand; reset to X1
# ============================================================================

def _obelisk_play(js, played, run_counts):
    """Mimic engine.step lifecycle: build a state stub with run counts, call on_play."""
    state = dataclasses.replace(reset(seed=0), hand_plays_run=tuple(run_counts))
    rules = aggregate_rules((js,))
    _, scoring_idx = evaluate(list(played), rules)
    return REGISTRY[JokerType.OBELISK].on_play(state, list(played), list(scoring_idx),
                                               rules, js)


def test_obelisk_independent_xmult_from_counter():
    # Obelisk updates BEFORE scoring; to read counter 5 as-is, this PAIR must be a
    # NON-most-played hand (another hand type leads), so it gains a step: 5 -> 6.
    # X(1 + 0.2*6) = X2.2. Pair base mult 2 -> 4.4.
    leader = [0] * 12
    leader[int(HandType.FLUSH)] = 9  # FLUSH leads -> PAIR isn't most-played
    js = J(JokerType.OBELISK, counter=5.0)
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=(js,),
                     hand_plays_run=tuple(leader))
    assert res.mult == pytest.approx(4.4)


def test_obelisk_starts_at_x1_when_playing_most_played_hand():
    # No run history -> the hand being played IS the strict most-played (count 1>0) ->
    # reset before scoring -> X1.
    js = J(JokerType.OBELISK, counter=0.0)
    res = score_play([C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)], jokers=(js,))
    assert res.mult == pytest.approx(2.0)  # base pair mult, X1


def test_obelisk_gains_when_playing_a_non_most_played_hand():
    # Run history: PAIR played 5 times (strict leader). We play a FLUSH (count 0->1),
    # which is NOT the most-played -> gain +0.2 step (counter 0 -> 1).
    counts = [0] * 12
    counts[int(HandType.PAIR)] = 5
    js = J(JokerType.OBELISK, counter=0.0)
    flush = [C(2, 1), C(5, 1), C(8, 1), C(10, 1), C(13, 1)]
    js2 = _obelisk_play(js, flush, counts)
    assert js2.counter == 1.0


def test_obelisk_resets_when_playing_the_strict_most_played_hand():
    # PAIR is the leader (5) and ahead of everything; playing PAIR again (->6) keeps it
    # the strict unique max -> reset to X1 (counter 0).
    counts = [0] * 12
    counts[int(HandType.PAIR)] = 5
    js = J(JokerType.OBELISK, counter=7.0)
    pair = [C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)]
    js2 = _obelisk_play(js, pair, counts)
    assert js2.counter == 0.0


def test_obelisk_tie_does_not_reset():
    # PAIR=5 (leader), TWO_PAIR=4. Playing TWO_PAIR brings it to 5 -> TIES the leader,
    # does not become the strict max -> wiki "safe until you break the tie" -> gains.
    counts = [0] * 12
    counts[int(HandType.PAIR)] = 5
    counts[int(HandType.TWO_PAIR)] = 4
    js = J(JokerType.OBELISK, counter=3.0)
    two_pair = [C(3, 0), C(3, 1), C(7, 2), C(7, 3), C(2, 0)]
    js2 = _obelisk_play(js, two_pair, counts)
    assert js2.counter == 4.0  # gained, no reset


def test_obelisk_breaking_the_tie_resets():
    # PAIR=5, TWO_PAIR=5 (tied leaders). Playing TWO_PAIR (->6) breaks the tie and
    # becomes strict max -> reset.
    counts = [0] * 12
    counts[int(HandType.PAIR)] = 5
    counts[int(HandType.TWO_PAIR)] = 5
    js = J(JokerType.OBELISK, counter=9.0)
    two_pair = [C(3, 0), C(3, 1), C(7, 2), C(7, 3), C(2, 0)]
    js2 = _obelisk_play(js, two_pair, counts)
    assert js2.counter == 0.0


def test_obelisk_reset_happens_before_scoring_end_to_end():
    # Wiki: "Obelisk resets before the hand is scored." Play the strict most-played hand
    # while counter is high -> the hand scores at X1, and the next state's counter is 0.
    s = reset(seed=1)
    pair = (C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0))
    run_counts = list([0] * 12)
    run_counts[int(HandType.PAIR)] = 5  # PAIR is strict leader; playing it -> reset
    s = dataclasses.replace(s, hand=pair, required=10_000_000,
                            hand_plays_run=tuple(run_counts),
                            jokers=(J(JokerType.OBELISK, counter=10.0),))
    nxt, info = step(s, (Verb.PLAY, (0, 1, 2, 3, 4)))
    # Scored at X1 (reset before scoring): pair base mult 2, no xMult boost.
    assert info["mult"] == pytest.approx(2.0)
    # Counter reset for future hands.
    assert nxt.jokers[0].counter == 0.0


def test_obelisk_gain_applies_to_the_current_hand():
    # Counter starts 0; play a non-most-played hand. The update happens BEFORE scoring
    # (symmetric with the reset), so the current hand benefits from the new +X0.2 step.
    s = reset(seed=1)
    flush = (C(2, 1), C(5, 1), C(8, 1), C(10, 1), C(13, 1))
    run_counts = list([0] * 12)
    run_counts[int(HandType.PAIR)] = 5  # PAIR leads; FLUSH is not most-played
    s = dataclasses.replace(s, hand=flush, required=10_000_000,
                            hand_plays_run=tuple(run_counts),
                            jokers=(J(JokerType.OBELISK, counter=0.0),))
    nxt, info = step(s, (Verb.PLAY, (0, 1, 2, 3, 4)))
    # counter steps 0 -> 1 before scoring -> X1.2. FLUSH base mult 4 -> 4.8.
    assert info["mult"] == pytest.approx(4.8)
    assert nxt.jokers[0].counter == 1.0  # persisted
