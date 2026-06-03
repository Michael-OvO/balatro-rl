"""Batch B2a: full-deck enhancement counts on ScoreContext + 6 economy/deck-reading
jokers. Values verified against balatrowiki.org:
  - Steel Joker (32, Uncommon $7): independent X(1 + 0.2 * #Steel in full deck) Mult.
  - Stone Joker (89, Uncommon $6): independent +25 Chips per Stone card in full deck.
  - Golden Ticket (106, Common $5): scored Gold-enhancement card earns $4.
  - Rough Gem (116, Uncommon $7): scored Diamond (suit==3) earns $1.
  - Business Card (42, Common $4): scored face card, 1 in 2 to earn $2 (rolls ctx.rng).
  - Reserved Parking (82, Common $6): each held face card, 1 in 2 to earn $1 (rolls ctx.rng).

RNG ORDER: Business Card consumes ctx.rng in the on_score phase (per scored FACE card,
L->R, BEFORE that card's mod fold). Reserved Parking consumes ctx.rng in the held phase
(per held FACE card, in held order), which runs AFTER all scored cards. Only face cards
draw rng, so a joker-absent / non-face game draws zero extra rng (byte-compatible).
"""
import dataclasses

import pytest

from balatro_rl.engine.cards import Card, Enhancement
from balatro_rl.engine.engine import reset, step, Verb, JOKER_SLOTS, make_master_deck
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import (
    Effect, JokerType, JokerState, REGISTRY, Rarity, ScoreContext,
)
from balatro_rl.engine.rng import RNG
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0, enh=0):
    return Card(rank=rank, suit=suit, enhancement=enh)


def J(t):
    return JokerState(type=t)


# ============================================================================
# STEP 1: ScoreContext.deck_enh_counts (full-deck enhancement histogram)
# ============================================================================

def test_score_context_deck_enh_counts_default_zeros():
    ctx = ScoreContext()
    assert ctx.deck_enh_counts == tuple([0] * len(Enhancement))


def test_score_play_populates_deck_enh_counts():
    captured = {}

    class _Probe:
        copyable = True
        rarity = None
        cost = 4
        def independent(self, ctx, js):
            captured["counts"] = ctx.deck_enh_counts
            return Effect()
        def on_score(self, ctx, card, index, js): return Effect()
        def on_held(self, ctx, card, js): return Effect()
        def retrigger(self, ctx, card, js): return 0
        def rules(self):
            from balatro_rl.engine.jokers.base import NO_RULES
            return NO_RULES
        def on_play(self, *a): return a[-1]
        def on_round_end(self, state, js, rng): return js, 0, False, rng

    saved = REGISTRY.get(JokerType.JOKER)
    REGISTRY[JokerType.JOKER] = _Probe()
    try:
        score_play([C(14), C(7), C(2)], jokers=(J(JokerType.JOKER),),
                   deck_enh_counts=(40, 0, 0, 0, 0, 5, 0, 0, 7))
    finally:
        REGISTRY[JokerType.JOKER] = saved
    assert captured["counts"] == (40, 0, 0, 0, 0, 5, 0, 0, 7)


def test_score_play_deck_enh_counts_defaults_when_omitted():
    captured = {}

    class _Probe:
        copyable = True
        rarity = None
        cost = 4
        def independent(self, ctx, js):
            captured["counts"] = ctx.deck_enh_counts
            return Effect()
        def on_score(self, ctx, card, index, js): return Effect()
        def on_held(self, ctx, card, js): return Effect()
        def retrigger(self, ctx, card, js): return 0
        def rules(self):
            from balatro_rl.engine.jokers.base import NO_RULES
            return NO_RULES
        def on_play(self, *a): return a[-1]
        def on_round_end(self, state, js, rng): return js, 0, False, rng

    saved = REGISTRY.get(JokerType.JOKER)
    REGISTRY[JokerType.JOKER] = _Probe()
    try:
        score_play([C(14), C(7), C(2)], jokers=(J(JokerType.JOKER),))
    finally:
        REGISTRY[JokerType.JOKER] = saved
    assert captured["counts"] == tuple([0] * len(Enhancement))


def test_engine_threads_master_deck_enh_counts():
    # Two Steel + three Stone cards in the master deck -> ctx.deck_enh_counts reflects them.
    mods = {0: {"enhancement": Enhancement.STEEL},
            1: {"enhancement": Enhancement.STEEL},
            2: {"enhancement": Enhancement.STONE},
            3: {"enhancement": Enhancement.STONE},
            4: {"enhancement": Enhancement.STONE}}
    captured = {}

    class _Probe:
        copyable = True
        rarity = None
        cost = 4
        def independent(self, ctx, js):
            captured["counts"] = ctx.deck_enh_counts
            return Effect()
        def on_score(self, ctx, card, index, js): return Effect()
        def on_held(self, ctx, card, js): return Effect()
        def retrigger(self, ctx, card, js): return 0
        def rules(self):
            from balatro_rl.engine.jokers.base import NO_RULES
            return NO_RULES
        def on_play(self, *a): return a[-1]
        def on_round_end(self, state, js, rng): return js, 0, False, rng

    saved = REGISTRY.get(JokerType.JOKER)
    REGISTRY[JokerType.JOKER] = _Probe()
    try:
        st = reset(seed=0, card_mods=mods)
        st = dataclasses.replace(st, jokers=(J(JokerType.JOKER),),
                                 hand=(C(3, 0), C(3, 1), C(7, 2), C(9, 3), C(2, 0)))
        step(st, (Verb.PLAY, (0, 1, 2, 3, 4)))
    finally:
        REGISTRY[JokerType.JOKER] = saved
    counts = captured["counts"]
    assert counts[Enhancement.STEEL] == 2
    assert counts[Enhancement.STONE] == 3


# ============================================================================
# STEP 2: the 6 jokers
# ============================================================================

# --- Steel Joker (32): X(1 + 0.2 * #Steel in deck) Mult, independent ---

def test_steel_joker_xmult_scales_with_full_deck_steel():
    # 3 Steel cards in the master deck -> X(1 + 0.6) = X1.6 Mult.
    res = score_play([C(5), C(7)], jokers=(J(JokerType.STEEL_JOKER),),
                     deck_enh_counts=_counts(steel=3))
    base = score_play([C(5), C(7)]).score
    assert res.score == int(base * 1.6)


def test_steel_joker_x1_with_no_steel():
    res = score_play([C(5), C(7)], jokers=(J(JokerType.STEEL_JOKER),),
                     deck_enh_counts=_counts())
    base = score_play([C(5), C(7)]).score
    assert res.score == base


# --- Stone Joker (89): +25 Chips per Stone card in deck, independent ---

def test_stone_joker_chips_scale_with_full_deck_stone():
    # 4 Stone cards -> +100 chips.
    res = score_play([C(5), C(7)], jokers=(J(JokerType.STONE_JOKER),),
                     deck_enh_counts=_counts(stone=4))
    base = score_play([C(5), C(7)])
    assert res.chips == base.chips + 100


def test_stone_joker_no_chips_with_no_stone():
    res = score_play([C(5), C(7)], jokers=(J(JokerType.STONE_JOKER),),
                     deck_enh_counts=_counts())
    base = score_play([C(5), C(7)])
    assert res.chips == base.chips


# --- Golden Ticket (106): scored GOLD-enhancement card earns $4 ---

def test_golden_ticket_pays_4_per_scored_gold_card():
    # Wiki: "Played Gold cards earn $4 when SCORED" -> the gold card must be part of
    # the scoring hand. A pair scores both cards, so the gold one triggers ($4).
    res = score_play([C(7, 0, Enhancement.GOLD), C(7, 1)],
                     jokers=(J(JokerType.GOLDEN_TICKET),))
    assert res.money_delta == 4


def test_golden_ticket_two_gold_cards():
    # Pair of Gold cards -> both score -> $8.
    res = score_play([C(7, 0, Enhancement.GOLD), C(7, 1, Enhancement.GOLD)],
                     jokers=(J(JokerType.GOLDEN_TICKET),))
    assert res.money_delta == 8


def test_golden_ticket_no_money_without_gold():
    res = score_play([C(5), C(7)], jokers=(J(JokerType.GOLDEN_TICKET),))
    assert res.money_delta == 0


# --- Rough Gem (116): scored Diamond (suit==3) earns $1 ---

def test_rough_gem_pays_1_per_scored_diamond():
    # Pair of Diamonds (both score) -> $2.
    res = score_play([C(5, 3), C(5, 3), C(7, 0)],
                     jokers=(J(JokerType.ROUGH_GEM),))
    assert res.money_delta == 2


def test_rough_gem_no_money_without_diamonds():
    res = score_play([C(5, 0), C(7, 1)], jokers=(J(JokerType.ROUGH_GEM),))
    assert res.money_delta == 0


# --- Business Card (42): scored face card 1 in 2 to earn $2 (rolls ctx.rng) ---

def test_business_card_pays_on_winning_roll():
    # Find a seed whose first roll is < 0.5 (win).
    seed = _seed_with_first_roll(lambda r: r < 0.5)
    res = score_play([C(13, 0), C(7, 1)], jokers=(J(JokerType.BUSINESS_CARD),),
                     rng=RNG.from_seed(seed))
    assert res.money_delta == 2


def test_business_card_no_pay_on_losing_roll():
    seed = _seed_with_first_roll(lambda r: r >= 0.5)
    res = score_play([C(13, 0), C(7, 1)], jokers=(J(JokerType.BUSINESS_CARD),),
                     rng=RNG.from_seed(seed))
    assert res.money_delta == 0


def test_business_card_draws_no_rng_without_face_cards():
    # No face card -> no roll consumed -> rng unchanged.
    res = score_play([C(5, 0), C(7, 1)], jokers=(J(JokerType.BUSINESS_CARD),),
                     rng=RNG.from_seed(42))
    assert res.rng == RNG.from_seed(42)


def test_business_card_rolls_once_per_face_card():
    # Two SCORING face cards both winning -> $4. A pair of Kings scores both cards
    # (a high-card hand would only score the single highest). Pick a seed whose first
    # TWO rolls are < 0.5 (one roll per scored face, L->R).
    seed = _seed_with_first_two_rolls(lambda a, b: a < 0.5 and b < 0.5)
    res = score_play([C(13, 0), C(13, 1)], jokers=(J(JokerType.BUSINESS_CARD),),
                     rng=RNG.from_seed(seed))
    assert res.money_delta == 4


# --- Reserved Parking (82): each HELD face card 1 in 2 to earn $1 (rolls ctx.rng) ---

def test_reserved_parking_pays_on_winning_roll():
    seed = _seed_with_first_roll(lambda r: r < 0.5)
    res = score_play([C(5, 0), C(7, 1)],
                     held=(C(13, 0),),
                     jokers=(J(JokerType.RESERVED_PARKING),),
                     rng=RNG.from_seed(seed))
    assert res.money_delta == 1


def test_reserved_parking_no_pay_on_losing_roll():
    seed = _seed_with_first_roll(lambda r: r >= 0.5)
    res = score_play([C(5, 0), C(7, 1)],
                     held=(C(13, 0),),
                     jokers=(J(JokerType.RESERVED_PARKING),),
                     rng=RNG.from_seed(seed))
    assert res.money_delta == 0


def test_reserved_parking_draws_no_rng_without_held_face():
    res = score_play([C(5, 0), C(7, 1)],
                     held=(C(5, 0), C(8, 1)),
                     jokers=(J(JokerType.RESERVED_PARKING),),
                     rng=RNG.from_seed(42))
    assert res.rng == RNG.from_seed(42)


def test_reserved_parking_held_money_flows_through_engine_step():
    # The IMPORTANT verification: a held face card under Reserved Parking actually
    # changes money via engine.step (held-phase money_delta -> ScoreResult -> engine).
    seed = _engine_seed_winning_held_roll()
    st = reset(seed=seed)
    st = dataclasses.replace(
        st, jokers=(J(JokerType.RESERVED_PARKING),),
        # Play two non-face cards; hold a King (face). required high so we don't clear.
        hand=(C(5, 0), C(7, 1), C(13, 2), C(8, 3)), required=10_000_000,
        # reset() advances rng by shuffling, so set the scoring rng explicitly: the
        # held King's roll is the FIRST (only) draw this hand (played 5,7 are non-face,
        # so no scored-card roll precedes it), and seed's first roll < 0.5 wins -> +$1.
        rng=RNG.from_seed(seed))
    money_before = st.money
    nxt, info = step(st, (Verb.PLAY, (0, 1)))
    assert nxt.money == money_before + 1


# ============================================================================
# rarity / cost
# ============================================================================

def test_batch7_jokers_declare_rarity_and_cost():
    expected = {
        JokerType.STEEL_JOKER: (Rarity.UNCOMMON, 7),
        JokerType.STONE_JOKER: (Rarity.UNCOMMON, 6),
        JokerType.GOLDEN_TICKET: (Rarity.COMMON, 5),
        JokerType.ROUGH_GEM: (Rarity.UNCOMMON, 7),
        JokerType.BUSINESS_CARD: (Rarity.COMMON, 4),
        JokerType.RESERVED_PARKING: (Rarity.COMMON, 6),
    }
    for jt, (rar, cost) in expected.items():
        eff = REGISTRY[jt]
        assert eff.rarity == rar, jt
        assert eff.cost == cost, jt


# ============================================================================
# helpers
# ============================================================================

def _counts(*, steel=0, stone=0):
    counts = [0] * len(Enhancement)
    counts[Enhancement.STEEL] = steel
    counts[Enhancement.STONE] = stone
    return tuple(counts)


def _seed_with_first_roll(pred):
    for seed in range(10000):
        r, _ = RNG.from_seed(seed).random()
        if pred(r):
            return seed
    raise AssertionError("no seed found")


def _seed_with_first_two_rolls(pred):
    for seed in range(10000):
        rng = RNG.from_seed(seed)
        a, rng = rng.random()
        b, rng = rng.random()
        if pred(a, b):
            return seed
    raise AssertionError("no seed found")


def _engine_seed_winning_held_roll():
    # The engine threads state.rng into score_play. Reserved Parking rolls in the
    # held phase (no scored-card rng for non-face plays of 5,7). Find a seed whose
    # first rng draw (= the held roll) is < 0.5.
    return _seed_with_first_roll(lambda r: r < 0.5)
