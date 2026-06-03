"""Score-breakdown trace for the replay viewer. score_play(trace=[...]) records an ordered
running-total event per contribution (base -> card chips -> jokers -> enhancements). The
trace is OFF on the engine hot path (trace=None) -> byte-identical scoring. engine.explain_play
re-runs a PLAY deterministically with the trace so the viewer's breakdown matches the real play.
"""
import dataclasses

from balatro_rl.engine.cards import Card, Edition, Enhancement
from balatro_rl.engine.engine import reset, step, explain_play, Verb
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerState, JokerType
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0, **mods):
    return Card(rank=rank, suit=suit, **mods)


def J(t):
    return JokerState(type=t)


# ============================================================================
# byte-compat: tracing does not change the score
# ============================================================================

def test_trace_does_not_change_score():
    hand = [C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)]   # pair of Kings
    plain = score_play(hand)
    traced = score_play(hand, trace=[])
    assert plain.score == traced.score and plain.mult == traced.mult


def test_default_score_play_has_no_trace_overhead():
    # No trace arg -> ctx.trace stays None; score is the normal pair value.
    res = score_play([C(13, 0), C(13, 1)])           # pair of Kings: base 10 + 10 + 10 = 30 x2
    assert res.score == 60


# ============================================================================
# trace content
# ============================================================================

def test_trace_starts_with_base_and_records_card_chips():
    trace = []
    score_play([C(13, 0), C(13, 1), C(3, 2), C(7, 3), C(9, 0)], trace=trace)
    assert trace[0]["label"].startswith("PAIR base")
    assert trace[0]["chips"] == 10 and trace[0]["mult"] == 2.0
    # the two Kings each add +10 chips
    king_events = [e for e in trace if "+10 chips" in e["label"]]
    assert len(king_events) == 2
    assert trace[-1]["chips"] == 30 and trace[-1]["mult"] == 2.0   # final running total -> 60


def test_trace_labels_joker_contribution():
    trace = []
    score_play([C(13, 0), C(13, 1)], jokers=(J(JokerType.JOKER),), trace=trace)  # +4 Mult
    assert any(e["label"] == "JOKER" for e in trace)
    assert trace[-1]["mult"] == 6.0                               # base 2 + Joker 4


def test_trace_labels_enhancement_contribution():
    trace = []
    score_play([C(13, 0, enhancement=Enhancement.BONUS), C(13, 1)], trace=trace)
    assert any("Bonus +30 chips" in e["label"] for e in trace)


def test_trace_records_edition():
    trace = []
    score_play([C(13, 0, edition=Edition.HOLO), C(13, 1)], trace=trace)
    assert any("Holo +10 mult" in e["label"] for e in trace)


# ============================================================================
# explain_play: faithful, matches the real engine.step score
# ============================================================================

def test_explain_play_matches_real_play_score():
    st = dataclasses.replace(reset(seed=0),
                             hand=(C(13, 0), C(13, 1), C(5, 2), C(7, 3), C(9, 0)),
                             jokers=(J(JokerType.JOKER),), required=10_000_000)
    exp = explain_play(st, (0, 1))
    _nxt, info = step(st, (Verb.PLAY, (0, 1)))
    assert exp["score"] == info["score"]
    assert exp["trace"] and exp["trace"][0]["label"].startswith("PAIR base")


def test_explain_play_reflects_levels_and_boss_flint():
    from balatro_rl.engine.bosses import BossEffect
    from balatro_rl.engine.hands import HandType
    lv = [1] * 12
    lv[int(HandType.PAIR)] = 3
    st = dataclasses.replace(reset(seed=0), levels=tuple(lv), boss=int(BossEffect.THE_FLINT),
                             hand=(C(13, 0), C(13, 1), C(5, 2), C(7, 3), C(9, 0)),
                             required=10_000_000)
    exp = explain_play(st, (0, 1))
    _nxt, info = step(st, (Verb.PLAY, (0, 1)))
    assert exp["score"] == info["score"]
    assert "Flint" in exp["trace"][0]["label"] and "lvl 3" in exp["trace"][0]["label"]
