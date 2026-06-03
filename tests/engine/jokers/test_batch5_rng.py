"""Batch 5: probabilistic scoring (rng-in-ScoreContext) + per-round randomized
state (on_round_start). Values verified against balatrowiki.org:
  - Misprint (27, Common $4): +0..+23 Mult, changes every hand.
  - Bloodstone (117, Uncommon $7): 1 in 2 chance per scored Heart -> X1.5 Mult.
  - Ancient Joker (99, Rare $8): X1.5 Mult per scored card of [suit]; re-rolled each round.
  - The Idol (127, Uncommon $6): X2 Mult per scored [rank] of [suit]; re-rolled each round.
  - Mail-In Rebate (83, Common $4): $5 per discarded card of [rank]; rank re-rolled each round.
"""
import dataclasses

import pytest

from balatro_rl.engine.cards import Card
from balatro_rl.engine.engine import reset, step, Verb, JOKER_SLOTS
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.hands import evaluate
from balatro_rl.engine.jokers.base import (
    Effect, JokerEffect, JokerType, JokerState, REGISTRY, Rarity,
    ScoreContext, aggregate_rules,
)
from balatro_rl.engine.rng import RNG
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


# ============================================================================
# STEP 1: rng-in-scoring  (ScoreContext.rng, threaded back into GameState)
# ============================================================================

def test_score_context_has_rng_field():
    ctx = ScoreContext()
    assert hasattr(ctx, "rng")


def test_score_play_populates_ctx_rng_from_kwarg():
    # A probe joker records the float it draws from ctx.rng during scoring.
    seen = {}

    class _Probe(JokerEffect):
        def independent(self, ctx, js):
            roll, ctx.rng = ctx.rng.random()
            seen["roll"] = roll
            return Effect()

    REGISTRY[JokerType.JOKER] = _Probe()
    expected, _ = RNG.from_seed(123).random()
    js = (JokerState(JokerType.JOKER),)
    score_play([C(14), C(7), C(2)], jokers=js, rng=RNG.from_seed(123))
    assert seen["roll"] == expected


def test_score_play_threads_advanced_rng_out():
    # score_play must return the advanced rng (after probabilistic hooks consumed it).
    class _Probe(JokerEffect):
        def independent(self, ctx, js):
            _, ctx.rng = ctx.rng.random()
            return Effect()

    REGISTRY[JokerType.JOKER] = _Probe()
    seed_rng = RNG.from_seed(99)
    _, advanced = seed_rng.random()
    res = score_play([C(14), C(7), C(2)], jokers=(JokerState(JokerType.JOKER),),
                     rng=RNG.from_seed(99))
    assert res.rng == advanced


def test_score_play_returns_same_rng_when_unconsumed():
    # No probabilistic joker -> rng comes back unchanged.
    res = score_play([C(14), C(7), C(2)], rng=RNG.from_seed(5))
    assert res.rng == RNG.from_seed(5)


def test_engine_play_advances_state_rng_after_scoring():
    # A consuming joker in a real PLAY must advance state.rng (persisted to next state).
    class _Probe(JokerEffect):
        def independent(self, ctx, js):
            _, ctx.rng = ctx.rng.random()
            return Effect()

    REGISTRY[JokerType.MISPRINT] = _Probe()
    state = reset(seed=7)
    state = dataclasses.replace(state, jokers=(JokerState(JokerType.MISPRINT),),
                                required=10_000_000)  # don't clear the blind
    before = state.rng
    nxt, _ = step(state, (Verb.PLAY, (0,)))
    assert nxt.rng != before


def test_engine_play_rng_unchanged_without_probabilistic_joker():
    state = reset(seed=7)
    state = dataclasses.replace(state, required=10_000_000)
    before = state.rng
    nxt, _ = step(state, (Verb.PLAY, (0,)))
    assert nxt.rng == before


def test_engine_play_seed_deterministic():
    # Same seed + same actions -> identical rng-driven roll persisted.
    class _Probe(JokerEffect):
        def independent(self, ctx, js):
            roll, ctx.rng = ctx.rng.random()
            return Effect(mult=int(roll * 24))

    REGISTRY[JokerType.MISPRINT] = _Probe()

    def run():
        state = reset(seed=2024)
        state = dataclasses.replace(state, jokers=(JokerState(JokerType.MISPRINT),),
                                    required=10_000_000)
        nxt, info = step(state, (Verb.PLAY, (0, 1)))
        return info["mult"], nxt.rng

    assert run() == run()


# ============================================================================
# Misprint (27)
# ============================================================================

def test_misprint_rarity_cost():
    eff = REGISTRY[JokerType.MISPRINT]
    assert eff.rarity == Rarity.COMMON and eff.cost == 4


def test_misprint_adds_mult_in_range_0_to_23():
    js = (JokerState(JokerType.MISPRINT),)
    seen = set()
    for seed in range(200):
        res = score_play([C(14), C(7), C(2)], jokers=js, rng=RNG.from_seed(seed))
        added = res.mult - 1.0  # base high-card mult is 1
        assert added == int(added)            # integer Mult
        assert 0 <= added <= 23               # wiki range
        seen.add(int(added))
    assert min(seen) <= 2 and max(seen) >= 21  # spans most of the range


def test_misprint_matches_explicit_roll():
    rng = RNG.from_seed(55)
    roll, _ = rng.random()
    res = score_play([C(14), C(7), C(2)], jokers=(JokerState(JokerType.MISPRINT),),
                     rng=RNG.from_seed(55))
    assert res.mult - 1.0 == int(roll * 24)


def test_misprint_advances_rng():
    res = score_play([C(14), C(7), C(2)], jokers=(JokerState(JokerType.MISPRINT),),
                     rng=RNG.from_seed(55))
    assert res.rng != RNG.from_seed(55)


# ============================================================================
# Bloodstone (117)
# ============================================================================

def test_bloodstone_rarity_cost():
    eff = REGISTRY[JokerType.BLOODSTONE]
    assert eff.rarity == Rarity.UNCOMMON and eff.cost == 7


def test_bloodstone_only_consumes_rng_for_hearts():
    # No Hearts scored -> rng untouched.
    res = score_play([C(14, 0), C(7, 2), C(2, 3)],  # spade, club, diamond
                     jokers=(JokerState(JokerType.BLOODSTONE),), rng=RNG.from_seed(8))
    assert res.rng == RNG.from_seed(8)


def test_bloodstone_x1_5_on_heart_when_roll_succeeds():
    # Find a seed whose first roll < 0.5 (success) for a single Heart.
    seed = next(s for s in range(1000) if RNG.from_seed(s).random()[0] < 0.5)
    res = score_play([C(5, 1)],  # one Heart, high card: chips 5+5=... base mult 1
                     jokers=(JokerState(JokerType.BLOODSTONE),), rng=RNG.from_seed(seed))
    assert res.mult == pytest.approx(1.5)


def test_bloodstone_no_xmult_when_roll_fails():
    seed = next(s for s in range(1000) if RNG.from_seed(s).random()[0] >= 0.5)
    res = score_play([C(5, 1)], jokers=(JokerState(JokerType.BLOODSTONE),),
                     rng=RNG.from_seed(seed))
    assert res.mult == pytest.approx(1.0)


def test_bloodstone_consumes_one_roll_per_heart():
    # Two SCORING Hearts (a pair, so both cards score) -> two rolls consumed.
    rng = RNG.from_seed(20)
    _, r1 = rng.random()
    _, r2 = r1.random()
    res = score_play([C(5, 1), C(5, 1)], jokers=(JokerState(JokerType.BLOODSTONE),),
                     rng=RNG.from_seed(20))
    assert res.rng == r2


# ============================================================================
# STEP 2: on_round_start hook  (per-round randomized state, threaded rng)
# ============================================================================

def test_default_on_round_start_is_noop():
    eff = JokerEffect()
    js = JokerState(type=JokerType.JOKER)
    rng = RNG.from_seed(1)
    js2, rng2 = eff.on_round_start(None, js, rng)
    assert js2 is js
    assert rng2 is rng


def test_reset_runs_round_start_fold_no_op_without_jokers():
    # reset() has no jokers yet (acquired via shop), so the start-of-blind fold is a
    # no-op here; real coverage of the fold runs through _advance_blind below. We just
    # confirm reset still produces a seeded, joker-empty state.
    state = reset(seed=3)
    assert state.jokers == ()
    assert reset(seed=3).rng == state.rng


def test_advance_blind_folds_on_round_start_and_threads_rng():
    # On leaving the shop -> next blind, every joker's on_round_start runs and rng advances.
    class _Stamp(JokerEffect):
        def on_round_start(self, state, js, rng):
            val, rng = rng.randint(0, 3)
            return dataclasses.replace(js, counter=float(val)), rng

    REGISTRY[JokerType.ANCIENT_JOKER] = _Stamp()
    from balatro_rl.engine.state import Phase
    state = reset(seed=42)
    js = JokerState(JokerType.ANCIENT_JOKER, counter=-1.0)
    shop = dataclasses.replace(state, phase=Phase.SHOP, jokers=(js,), shop_offers=())
    nxt, _ = step(shop, (Verb.LEAVE_SHOP, 0))
    # counter set deterministically (0..3), rng consumed.
    assert 0.0 <= nxt.jokers[0].counter <= 3.0
    assert nxt.rng != shop.rng


def test_on_round_start_deterministic_and_varies_across_rounds():
    class _Stamp(JokerEffect):
        def on_round_start(self, state, js, rng):
            val, rng = rng.randint(0, 3)
            return dataclasses.replace(js, counter=float(val)), rng

    REGISTRY[JokerType.ANCIENT_JOKER] = _Stamp()
    from balatro_rl.engine.state import Phase

    def round1_counter(seed):
        state = reset(seed=seed)
        js = JokerState(JokerType.ANCIENT_JOKER)
        shop = dataclasses.replace(state, phase=Phase.SHOP, jokers=(js,))
        nxt, _ = step(shop, (Verb.LEAVE_SHOP, 0))
        return nxt.jokers[0].counter

    # deterministic from seed
    assert round1_counter(7) == round1_counter(7)
    # across a sweep of seeds the per-round value is not constant
    vals = {round1_counter(s) for s in range(20)}
    assert len(vals) > 1


# ============================================================================
# Ancient Joker (99)
# ============================================================================

def test_ancient_joker_rarity_cost():
    eff = REGISTRY[JokerType.ANCIENT_JOKER]
    assert eff.rarity == Rarity.RARE and eff.cost == 8


def test_ancient_joker_on_round_start_picks_a_suit():
    eff = REGISTRY[JokerType.ANCIENT_JOKER]
    js = JokerState(JokerType.ANCIENT_JOKER, counter=-1.0)
    js2, _ = eff.on_round_start(None, js, RNG.from_seed(4))
    assert int(js2.counter) in (0, 1, 2, 3)


def test_ancient_joker_x1_5_per_scored_card_of_chosen_suit():
    # Force the chosen suit to Hearts (1). A pair of Hearts scores both cards
    # (base mult 2) -> X1.5 * X1.5 -> 2 * 2.25 = 4.5.
    js = JokerState(JokerType.ANCIENT_JOKER, counter=1.0)
    res = score_play([C(5, 1), C(5, 1)], jokers=(js,))
    assert res.mult == pytest.approx(4.5)


def test_ancient_joker_single_scored_heart_is_x1_5():
    # High card: only the highest card scores. One scoring Heart -> base 1 * 1.5.
    js = JokerState(JokerType.ANCIENT_JOKER, counter=1.0)  # Hearts
    res = score_play([C(9, 1), C(2, 0)], jokers=(js,))
    assert res.mult == pytest.approx(1.5)


def test_ancient_joker_no_xmult_for_other_suits():
    js = JokerState(JokerType.ANCIENT_JOKER, counter=1.0)  # Hearts
    res = score_play([C(9, 2), C(2, 3)], jokers=(js,))  # no Hearts, high card
    assert res.mult == pytest.approx(1.0)


def test_ancient_joker_suit_changes_across_rounds():
    eff = REGISTRY[JokerType.ANCIENT_JOKER]
    js = JokerState(JokerType.ANCIENT_JOKER)
    suits = set()
    rng = RNG.from_seed(0)
    for _ in range(40):
        js, rng = eff.on_round_start(None, js, rng)
        suits.add(int(js.counter))
    assert len(suits) > 1  # not stuck on one suit


# ============================================================================
# The Idol (127)
# ============================================================================

def test_idol_rarity_cost():
    eff = REGISTRY[JokerType.THE_IDOL]
    assert eff.rarity == Rarity.UNCOMMON and eff.cost == 6


def test_idol_on_round_start_encodes_rank_and_suit():
    eff = REGISTRY[JokerType.THE_IDOL]
    js = JokerState(JokerType.THE_IDOL)
    js2, _ = eff.on_round_start(None, js, RNG.from_seed(11))
    code = int(js2.counter)
    rank = code // 4
    suit = code % 4
    assert 2 <= rank <= 14
    assert 0 <= suit <= 3


def test_idol_x2_on_matching_rank_and_suit():
    # Encode 7 of Diamonds (rank 7, suit 3) -> counter = 7*4 + 3 = 31.
    # High card (the 7 scores, base mult 1): X2 on the matching card -> 2.0.
    js = JokerState(JokerType.THE_IDOL, counter=float(7 * 4 + 3))
    res = score_play([C(7, 3), C(2, 0)], jokers=(js,))  # one matching card scores
    assert res.mult == pytest.approx(2.0)


def test_idol_no_xmult_on_rank_match_wrong_suit():
    # Pair of 7s (base mult 2) but suits are spade/club -> Idol does not fire.
    js = JokerState(JokerType.THE_IDOL, counter=float(7 * 4 + 3))  # 7 of Diamonds
    res = score_play([C(7, 0), C(7, 2)], jokers=(js,))
    assert res.mult == pytest.approx(2.0)  # base pair mult only, no xMult


def test_idol_no_xmult_on_suit_match_wrong_rank():
    # High card (only the 9♦ scores): diamond but not a 7 -> no xMult.
    js = JokerState(JokerType.THE_IDOL, counter=float(7 * 4 + 3))  # 7 of Diamonds
    res = score_play([C(9, 3), C(2, 3)], jokers=(js,))
    assert res.mult == pytest.approx(1.0)


def test_idol_stacks_for_multiple_matches():
    # A pair of 7 of Diamonds (base mult 2): both score -> X2 * X2 -> 2 * 4 = 8.
    js = JokerState(JokerType.THE_IDOL, counter=float(7 * 4 + 3))  # 7 of Diamonds
    res = score_play([C(7, 3), C(7, 3)], jokers=(js,))
    assert res.mult == pytest.approx(8.0)


# ============================================================================
# Mail-In Rebate (83)
# ============================================================================

def test_mail_in_rebate_rarity_cost():
    eff = REGISTRY[JokerType.MAIL_IN_REBATE]
    assert eff.rarity == Rarity.COMMON and eff.cost == 4


def test_mail_in_rebate_on_round_start_picks_rank():
    eff = REGISTRY[JokerType.MAIL_IN_REBATE]
    js = JokerState(JokerType.MAIL_IN_REBATE)
    js2, _ = eff.on_round_start(None, js, RNG.from_seed(13))
    assert 2 <= int(js2.counter) <= 14


def test_mail_in_rebate_pays_5_per_discarded_matching_rank():
    eff = REGISTRY[JokerType.MAIL_IN_REBATE]
    js = JokerState(JokerType.MAIL_IN_REBATE, counter=7.0)  # target rank 7
    # Three 7s among the discards -> $15.
    js2, money, _ = eff.on_discard(None, [C(7, 0), C(7, 1), C(2, 2), C(7, 3)], js,
                                   RNG.from_seed(1))
    assert money == 15


def test_mail_in_rebate_no_pay_for_non_matching_discards():
    eff = REGISTRY[JokerType.MAIL_IN_REBATE]
    js = JokerState(JokerType.MAIL_IN_REBATE, counter=7.0)
    _, money, _ = eff.on_discard(None, [C(2, 0), C(9, 1), C(13, 2)], js, RNG.from_seed(1))
    assert money == 0


def test_mail_in_rebate_rank_varies_across_rounds():
    eff = REGISTRY[JokerType.MAIL_IN_REBATE]
    js = JokerState(JokerType.MAIL_IN_REBATE)
    ranks = set()
    rng = RNG.from_seed(0)
    for _ in range(60):
        js, rng = eff.on_round_start(None, js, rng)
        ranks.add(int(js.counter))
    assert len(ranks) > 1
