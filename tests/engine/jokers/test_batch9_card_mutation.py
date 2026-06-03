"""Batch B2b-ii: card-mutation enhancement jokers (Vampire, Midas Mask) + the
master-deck mutation channel they ride.

Values verified against balatrowiki.org:
  - Vampire (68, Uncommon $7): gains X0.1 Mult per scored ENHANCED card, and REMOVES the
    enhancement BEFORE it takes effect (so the enhancement does NOT score that hand). The
    X Mult gain applies the SAME hand (the lucky-cat pattern: independent reads persistent
    counter + this-hand ctx.vampire_consumed). Edition/seal are kept; only the enhancement
    is stripped (in the persistent master_deck).
  - Midas Mask (76, Uncommon $7): all SCORED face cards become Gold cards (respects
    Pareidolia -> all scored cards become Gold). Pure master_deck mutation; no scoring
    change the hand it fires (Gold's effect is held-at-round-end money).

PLUMBING: scoring detects these via RuleFlags (vampire/midas), records per-card enhancement
overrides on ScoreResult.mutations [(played_idx, Enhancement)], and surfaces
ScoreResult.vampire_consumed. engine.step applies the mutations to master_deck by identity
(after Glass destruction) and folds vampire_consumed into Vampire's on_hand_events. A hand
with neither joker records no mutations and draws no extra rng -> byte-identical.
"""
import dataclasses

from balatro_rl.engine.cards import Card, Edition, Enhancement
from balatro_rl.engine.engine import reset, step, Verb
from balatro_rl.engine.scoring import score_play
from balatro_rl.engine.jokers.base import (
    HandEvents, JokerState, JokerType, REGISTRY, Rarity, RuleFlags, ScoreContext,
)
import balatro_rl.engine.jokers.library  # noqa: F401


def C(rank, suit=0, enh=0):
    return Card(rank=rank, suit=suit, enhancement=enh)


def J(t, counter=0.0):
    return JokerState(type=t, counter=counter)


# ============================================================================
# RuleFlags: vampire / midas merge
# ============================================================================

def test_rule_flags_vampire_midas_default_false_and_merge():
    assert RuleFlags().vampire is False and RuleFlags().midas is False
    merged = RuleFlags(vampire=True).merge(RuleFlags(midas=True))
    assert merged.vampire and merged.midas


def test_vampire_and_midas_jokers_publish_rules():
    assert REGISTRY[JokerType.VAMPIRE].rules().vampire is True
    assert REGISTRY[JokerType.MIDAS_MASK].rules().midas is True


# ============================================================================
# Vampire (68): strip enhancement, X0.1 per enhanced scored card, same-hand
# ============================================================================

def test_vampire_independent_reads_persistent_plus_this_hand():
    eff = REGISTRY[JokerType.VAMPIRE].independent(
        ScoreContext(vampire_consumed=2), J(JokerType.VAMPIRE, 1))
    assert abs(eff.xmult - (1 + 0.1 * (1 + 2))) < 1e-9    # 1.3


def test_vampire_on_hand_events_persists_consumed():
    js2 = REGISTRY[JokerType.VAMPIRE].on_hand_events(
        J(JokerType.VAMPIRE, 1), HandEvents(vampire_consumed=2))
    assert js2.counter == 3.0


def test_vampire_suppresses_enhancement_chips():
    # A Bonus card (+30 chips) under Vampire scores its rank chips but NOT the +30.
    bonus = C(7, 0, Enhancement.BONUS)
    base = score_play([bonus])
    vamp = score_play([bonus], jokers=(J(JokerType.VAMPIRE),))
    assert vamp.chips == base.chips - 30
    assert vamp.vampire_consumed == 1


def test_vampire_keeps_edition_while_stripping_enhancement():
    # Bonus enhancement (+30, stripped) + Foil edition (+50, kept). Net vs a plain card:
    # only the Foil +50 survives.
    card = Card(rank=7, suit=0, enhancement=Enhancement.BONUS, edition=Edition.FOIL)
    plain = score_play([Card(rank=7, suit=0)])
    vamp = score_play([card], jokers=(J(JokerType.VAMPIRE),))
    assert vamp.chips == plain.chips + 50


def test_vampire_counts_only_enhanced_scored_cards():
    # Pair (both 7s score): one Bonus + one plain -> exactly 1 enhanced scored card.
    res = score_play([C(7, 0, Enhancement.BONUS), C(7, 1)], jokers=(J(JokerType.VAMPIRE),))
    assert res.vampire_consumed == 1
    assert res.mutations == ((0, Enhancement.NONE),)


def test_vampire_no_consume_without_enhanced_cards():
    res = score_play([C(7, 0), C(7, 1)], jokers=(J(JokerType.VAMPIRE),))
    assert res.vampire_consumed == 0 and res.mutations == ()


def test_vampire_lucky_card_draws_no_rng_when_stripped():
    # Vampire removes a Lucky enhancement BEFORE it rolls -> no rng drawn for it.
    from balatro_rl.engine.rng import RNG
    res = score_play([C(7, 0, Enhancement.LUCKY)], jokers=(J(JokerType.VAMPIRE),),
                     rng=RNG.from_seed(7))
    assert res.rng == RNG.from_seed(7)        # the stripped Lucky card never rolled
    assert res.lucky_triggered == 0 and res.vampire_consumed == 1


def test_vampire_strips_enhancement_in_master_deck_via_engine():
    st = reset(seed=0, card_mods={0: {"enhancement": Enhancement.BONUS}})
    bonus = st.master_deck[0]
    st = dataclasses.replace(
        st, jokers=(J(JokerType.VAMPIRE),),
        hand=(bonus, C(7, 1), C(9, 2), C(2, 3), C(4, 0)), required=10_000_000)
    nxt, _info = step(st, (Verb.PLAY, (0,)))
    same = [c for c in nxt.master_deck if c.rank == bonus.rank and c.suit == bonus.suit]
    assert same and all(c.enhancement == Enhancement.NONE for c in same)
    assert nxt.jokers[0].counter == 1.0       # gained X0.1 from the one enhanced card


# ============================================================================
# Midas Mask (76): scored face cards become Gold (pure master_deck mutation)
# ============================================================================

def test_midas_marks_scored_face_for_gold():
    res = score_play([C(13, 0), C(13, 1)], jokers=(J(JokerType.MIDAS_MASK),))   # pair of Kings
    assert set(res.mutations) == {(0, Enhancement.GOLD), (1, Enhancement.GOLD)}


def test_midas_ignores_non_face():
    res = score_play([C(5, 0), C(5, 1)], jokers=(J(JokerType.MIDAS_MASK),))
    assert res.mutations == ()


def test_midas_no_immediate_score_change():
    # Converting to Gold has no on-score effect (Gold pays at round end), so the hand's
    # score is identical with or without Midas Mask.
    hand = [C(13, 0), C(13, 1)]
    assert score_play(hand).score == score_play(hand, jokers=(J(JokerType.MIDAS_MASK),)).score


def test_midas_converts_face_to_gold_in_master_deck_via_engine():
    st = reset(seed=0)
    king = next(c for c in st.master_deck if c.rank == 13)
    st = dataclasses.replace(
        st, jokers=(J(JokerType.MIDAS_MASK),),
        hand=(king, C(7, 1), C(9, 2), C(2, 3), C(4, 0)), required=10_000_000)
    nxt, _info = step(st, (Verb.PLAY, (0,)))
    same = [c for c in nxt.master_deck if c.rank == 13 and c.suit == king.suit]
    assert same and all(c.enhancement == Enhancement.GOLD for c in same)


def test_midas_respects_pareidolia_in_master_deck_via_engine():
    st = reset(seed=0)
    five = next(c for c in st.master_deck if c.rank == 5)
    st = dataclasses.replace(
        st, jokers=(J(JokerType.MIDAS_MASK), J(JokerType.PAREIDOLIA)),
        hand=(five, C(7, 1), C(9, 2), C(2, 3), C(4, 0)), required=10_000_000)
    nxt, _info = step(st, (Verb.PLAY, (0,)))
    same = [c for c in nxt.master_deck if c.rank == 5 and c.suit == five.suit]
    assert same and all(c.enhancement == Enhancement.GOLD for c in same)   # non-face -> gold


def test_midas_leaves_non_face_in_master_deck_unchanged():
    st = reset(seed=0)
    five = next(c for c in st.master_deck if c.rank == 5)
    st = dataclasses.replace(
        st, jokers=(J(JokerType.MIDAS_MASK),),
        hand=(five, C(7, 1), C(9, 2), C(2, 3), C(4, 0)), required=10_000_000)
    nxt, _info = step(st, (Verb.PLAY, (0,)))
    same = [c for c in nxt.master_deck if c.rank == 5 and c.suit == five.suit]
    assert same and all(c.enhancement == Enhancement.NONE for c in same)


# ============================================================================
# byte-compat
# ============================================================================

def test_no_mutations_or_consume_without_these_jokers():
    res = score_play([C(13, 0, Enhancement.BONUS), C(13, 1)])
    assert res.mutations == () and res.vampire_consumed == 0


# ============================================================================
# rarity / cost
# ============================================================================

def test_batch9_rarity_cost():
    for jt in (JokerType.VAMPIRE, JokerType.MIDAS_MASK):
        assert REGISTRY[jt].rarity == Rarity.UNCOMMON and REGISTRY[jt].cost == 7
