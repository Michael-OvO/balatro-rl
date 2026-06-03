"""Batch 2 engine extension: ScoreContext exposes read-only game-state info
(n_jokers, empty_joker_slots, money, hands_left, discards_left, deck_count).

Verifies score_play populates the fields and engine.step threads them from
GameState. The scalar values are wiki-independent (engine plumbing)."""
import dataclasses

from balatro_rl.engine.cards import Card
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import JokerType, JokerState, ScoreContext, Effect
import balatro_rl.engine.jokers.library  # noqa: F401  (registers jokers)
from balatro_rl.engine import engine
from balatro_rl.engine.engine import reset, step, Verb, JOKER_SLOTS


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


def J(t):
    return JokerState(type=t)


# --- ScoreContext defaults so old contexts still construct ---

def test_score_context_new_fields_default():
    ctx = ScoreContext()
    assert ctx.n_jokers == 0
    assert ctx.empty_joker_slots == 0
    assert ctx.money == 0
    assert ctx.hands_left == 0
    assert ctx.discards_left == 0
    assert ctx.deck_count == 0


# --- score_play populates the fields from its args ---

def test_score_play_populates_state_fields_via_probe_joker():
    captured = {}

    class _Probe:
        copyable = True
        rarity = None
        cost = 4
        def independent(self, ctx, js):
            captured.update(
                n_jokers=ctx.n_jokers,
                empty=ctx.empty_joker_slots,
                money=ctx.money,
                hands=ctx.hands_left,
                discards=ctx.discards_left,
                deck=ctx.deck_count,
            )
            return Effect()
        def on_score(self, ctx, card, index, js): return Effect()
        def on_held(self, ctx, card, js): return Effect()
        def retrigger(self, ctx, card, js): return 0
        def rules(self):
            from balatro_rl.engine.jokers.base import NO_RULES
            return NO_RULES
        def on_play(self, *a): return a[-1]
        def on_round_end(self, state, js, rng): return js, 0, False, rng

    from balatro_rl.engine.jokers.base import REGISTRY
    saved = REGISTRY.get(JokerType.JOKER)
    REGISTRY[JokerType.JOKER] = _Probe()
    try:
        score_play([C(14), C(7), C(2)], jokers=(J(JokerType.JOKER),),
                   joker_slots=5, money=12, hands_left=3, discards_left=2,
                   deck_count=33)
    finally:
        REGISTRY[JokerType.JOKER] = saved

    assert captured["n_jokers"] == 1
    assert captured["empty"] == 4          # 5 slots - 1 owned
    assert captured["money"] == 12
    assert captured["hands"] == 3
    assert captured["discards"] == 2
    assert captured["deck"] == 33


def test_empty_joker_slots_never_negative():
    # More jokers than slots (defensive): empty slots clamps at 0.
    res = score_play([C(14), C(7), C(2)],
                     jokers=tuple(J(JokerType.JOKER) for _ in range(7)),
                     joker_slots=5)
    # Just ensure it runs and produces a result (7 base Jokers, +4 mult each).
    assert res.score > 0


# --- engine threads GameState info into score_play (Bull reads money) ---

def test_engine_threads_money_into_scoring_via_bull():
    st = reset(seed=0)
    st = dataclasses.replace(st, money=10, jokers=(J(JokerType.BULL),))
    # Force a deterministic hand: pair of 3s + kickers (no flush).
    st = dataclasses.replace(st, hand=(C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)))
    nxt, info = step(st, (Verb.PLAY, (0, 1, 2, 3, 4)))
    # Bull: +2 chips per $1 of $10 = +20 chips on top of pair base (10 + 3 + 3 = 16) -> 36.
    assert info["chips"] == 36


def test_engine_threads_deck_count_into_scoring_via_blue_joker():
    st = reset(seed=0)
    deck_n = len(st.deck)
    st = dataclasses.replace(st, jokers=(J(JokerType.BLUE_JOKER),),
                             hand=(C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)))
    nxt, info = step(st, (Verb.PLAY, (0, 1, 2, 3, 4)))
    # Blue Joker: +2 chips per remaining deck card. Base pair chips 16.
    assert info["chips"] == 16 + 2 * deck_n
