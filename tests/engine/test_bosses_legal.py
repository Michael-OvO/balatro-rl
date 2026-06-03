"""Phase C2: boss legal-mask + blind-setup effects.

Blind setup (applied when entering the boss blind): The Water (0 discards), The Manacle
(-1 hand size), The Needle (1 hand). Legal-action filters: The Psychic (must play exactly
5 cards), The Eye (no repeated hand type this round), The Mouth (only one hand type all
round). Eye/Mouth read the existing per-round play history (hand_plays_round).

All effects are no-ops off a boss blind (state.boss == 0) -> the default game and the
ante-7 agent's action space are unchanged. Verified against balatrowiki.org/w/Blinds.
"""
import dataclasses

from balatro_rl.engine.cards import Card
from balatro_rl.engine.bosses import (
    BossEffect, boss_hand_size_delta, boss_hands_left, boss_discards_left, select_boss,
)
from balatro_rl.engine.engine import (
    reset, legal_actions, _advance_blind, Verb,
    HAND_SIZE, HANDS_PER_BLIND, DISCARDS_PER_BLIND,
)
from balatro_rl.engine.hands import HandType
from balatro_rl.engine.rng import RNG


def C(rank, suit=0):
    return Card(rank=rank, suit=suit)


_HAND8 = (C(13, 0), C(13, 1), C(2, 2), C(3, 3), C(4, 0), C(6, 1), C(8, 2), C(9, 3))


def _on_boss(boss, **over):
    """A boss-blind PLAYING state with `boss` active (8-card hand by default)."""
    base = dict(boss=int(boss), blind_index=2, hand=_HAND8)
    base.update(over)
    return dataclasses.replace(reset(seed=0), **base)


def _advance_to_boss(target, start_ante=1):
    """Drive _advance_blind (bosses enabled) until it selects `target`; return that state."""
    for s in range(4000):
        st = dataclasses.replace(reset(seed=s, enable_bosses=True),
                                 ante=start_ante, blind_index=1)
        nxt, _info = _advance_blind(st)
        if BossEffect(nxt.boss) == target:
            return nxt
    raise AssertionError(f"no seed produced {target}")


# ============================================================================
# blind-setup helpers
# ============================================================================

def test_blind_setup_helpers():
    assert boss_hand_size_delta(BossEffect.THE_MANACLE) == -1
    assert boss_hand_size_delta(BossEffect.THE_HOOK) == 0
    assert boss_hands_left(BossEffect.THE_NEEDLE, HANDS_PER_BLIND) == 1
    assert boss_hands_left(BossEffect.THE_HOOK, HANDS_PER_BLIND) == HANDS_PER_BLIND
    assert boss_discards_left(BossEffect.THE_WATER, DISCARDS_PER_BLIND) == 0
    assert boss_discards_left(BossEffect.THE_HOOK, DISCARDS_PER_BLIND) == DISCARDS_PER_BLIND


def test_advance_blind_applies_boss_setup_invariant():
    # Whatever boss is selected, the blind's hand_size/hands/discards match its spec.
    st = dataclasses.replace(reset(seed=7, enable_bosses=True), ante=2, blind_index=1)
    nxt, _info = _advance_blind(st)
    boss = BossEffect(nxt.boss)
    assert nxt.hand_size == HAND_SIZE + boss_hand_size_delta(boss)
    assert len(nxt.hand) == nxt.hand_size
    assert nxt.hands_left == boss_hands_left(boss, HANDS_PER_BLIND)
    assert nxt.discards_left == boss_discards_left(boss, DISCARDS_PER_BLIND)


def test_manacle_reduces_hand_size_to_7():
    nxt = _advance_to_boss(BossEffect.THE_MANACLE, start_ante=1)
    assert nxt.hand_size == 7 and len(nxt.hand) == 7


def test_water_starts_with_zero_discards():
    nxt = _advance_to_boss(BossEffect.THE_WATER, start_ante=2)
    assert nxt.discards_left == 0


def test_needle_gives_one_hand():
    nxt = _advance_to_boss(BossEffect.THE_NEEDLE, start_ante=2)
    assert nxt.hands_left == 1


def test_boss_setup_resets_on_next_blind():
    # After a Manacle boss blind, advancing to the next ante's small blind restores hand 8.
    manacle = _advance_to_boss(BossEffect.THE_MANACLE, start_ante=1)
    nxt, _info = _advance_blind(manacle)     # boss(2) -> next ante small(0)
    assert nxt.blind_index == 0 and nxt.boss == 0 and nxt.hand_size == HAND_SIZE


# ============================================================================
# legal-action filters
# ============================================================================

def test_psychic_only_allows_5_card_plays():
    acts = legal_actions(_on_boss(BossEffect.THE_PSYCHIC))
    plays = [(v, c) for v, c in acts if v == Verb.PLAY]
    assert plays and all(len(c) == 5 for _v, c in plays)
    assert any(v == Verb.DISCARD for v, _c in acts)         # discards unrestricted


def test_eye_forbids_already_played_hand_type():
    hpr = [0] * 12
    hpr[int(HandType.HIGH_CARD)] = 1                        # a high card was already played
    acts = legal_actions(_on_boss(BossEffect.THE_EYE, hand_plays_round=tuple(hpr)))
    plays = [c for v, c in acts if v == Verb.PLAY]
    assert not any(len(c) == 1 for c in plays)             # every 1-card play is HIGH_CARD -> blocked
    assert (0, 1) in plays                                  # the pair of Kings (new type) is allowed


def test_mouth_locks_to_the_played_hand_type():
    hpr = [0] * 12
    hpr[int(HandType.PAIR)] = 1                             # round locked to PAIR
    acts = legal_actions(_on_boss(BossEffect.THE_MOUTH, hand_plays_round=tuple(hpr)))
    plays = [c for v, c in acts if v == Verb.PLAY]
    assert (0, 1) in plays                                  # PAIR still allowed
    assert not any(len(c) == 1 for c in plays)             # HIGH_CARD now forbidden


def test_mouth_allows_any_type_before_first_play():
    acts = legal_actions(_on_boss(BossEffect.THE_MOUTH))    # nothing played this round
    plays = [c for v, c in acts if v == Verb.PLAY]
    assert any(len(c) == 1 for c in plays)                 # high card allowed (locks on first play)


def test_no_boss_leaves_legal_actions_unfiltered():
    acts = legal_actions(reset(seed=0))                    # boss 0
    plays = [c for v, c in acts if v == Verb.PLAY]
    assert any(len(c) == 1 for c in plays) and any(len(c) == 5 for c in plays)
