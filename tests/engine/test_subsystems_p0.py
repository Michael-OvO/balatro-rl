"""P0 subsystems backbone tests (docs/specs/2026-06-03-subsystems-plan.md, Phase A).

These cover the three dormant-plumbing pieces and, crucially, assert that with no
card mods present (the current game) every deck / shuffle / score / money value is
byte-identical to the pre-P0 behavior.
"""
import dataclasses

import numpy as np
import pytest

from balatro_rl.engine.cards import Card, standard_deck
from balatro_rl.engine.engine import (
    Verb, reset, step, _advance_blind, HAND_SIZE,
)
from balatro_rl.engine.rng import RNG
from balatro_rl.engine.scoring import ScoreResult, score_play
from balatro_rl.engine.state import GameState
from balatro_rl.envs.actions import MAX_HAND, NUM_ACTIONS, legal_mask


# --------------------------------------------------------------------------
# Piece 1: persistent master deck
# --------------------------------------------------------------------------

def test_reset_seeds_master_deck_from_standard_deck():
    s = reset(seed=3)
    assert len(s.master_deck) == 52
    # master_deck (before any mod) is exactly standard_deck() in canonical order.
    assert s.master_deck == tuple(standard_deck())


def test_reset_shuffle_is_byte_identical_to_pre_p0():
    """The working deck must be the SAME as the old `rng.shuffle(standard_deck())`."""
    for seed in range(8):
        s = reset(seed=seed)
        rng = RNG.from_seed(seed)
        old_deck, old_rng = rng.shuffle(standard_deck())
        # Reconstruct the pre-P0 hand/deck split exactly.
        old_hand, old_deck = old_deck[:HAND_SIZE], old_deck[HAND_SIZE:]
        assert s.hand == tuple(old_hand), f"hand differs for seed {seed}"
        assert s.deck == tuple(old_deck), f"deck differs for seed {seed}"
        assert s.rng == old_rng, f"rng differs for seed {seed}"


def test_advance_blind_reshuffles_from_master_deck_byte_identical():
    """With an unmodified master deck, _advance_blind matches the old shuffle(standard_deck())."""
    s = reset(seed=11)
    # Put the state at a blind boundary (simulate leaving the shop -> advance).
    nxt, _ = _advance_blind(s)
    # Old behavior: shuffle a fresh standard_deck() with the same rng.
    old_deck, old_rng = s.rng.shuffle(standard_deck())
    old_hand, old_deck = old_deck[:s.hand_size], old_deck[s.hand_size:]
    assert nxt.hand == tuple(old_hand)
    assert nxt.deck == tuple(old_deck)
    assert nxt.rng == old_rng
    # master_deck carried forward unchanged.
    assert nxt.master_deck == s.master_deck


def test_card_mod_survives_blind_transition_and_reappears():
    """A mod written onto a master_deck Card rides through a blind transition."""
    s = reset(seed=2)
    # Tag one specific card (a fixed identity) with a fake enhancement.
    target = s.master_deck[0]
    tagged = dataclasses.replace(target, enhancement=5)
    new_master = (tagged,) + s.master_deck[1:]
    s = dataclasses.replace(s, master_deck=new_master)

    # The mod survives an arbitrary number of blind transitions...
    cur = s
    for _ in range(3):
        cur, _ = _advance_blind(cur)
        assert any(c.enhancement == 5 for c in cur.master_deck), "mod erased on blind transition"
        # And the tagged card eventually appears in a drawn hand or the deck.
        assert any(c.enhancement == 5 for c in (cur.deck + cur.hand)), "mod not dealt"

    # Exactly one card carries the mod (we only tagged one).
    assert sum(c.enhancement == 5 for c in cur.master_deck) == 1


# --------------------------------------------------------------------------
# Piece 2: score-result money + destruction threading
# --------------------------------------------------------------------------

def test_real_game_score_result_has_no_side_effects():
    """With the unmodified game, money_delta is always 0 and destroyed_idx always ()."""
    s = reset(seed=4)
    for combo in [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4)]:
        selected = [s.hand[i] for i in combo]
        res = score_play(selected, jokers=s.jokers)
        assert res.money_delta == 0
        assert res.destroyed_idx == ()


def test_real_play_never_changes_money_or_shrinks_master_deck():
    s = reset(seed=9)
    money_before = s.money
    s2, info = step(s, (Verb.PLAY, (0,)))
    # No scoring side effects on the real game.
    assert s2.money == money_before
    assert len(s2.master_deck) == 52
    assert s2.master_deck == s.master_deck


def test_engine_applies_money_delta_probe(monkeypatch):
    """PROBE: a forced ScoreResult.money_delta > 0 raises the engine's money."""
    import balatro_rl.engine.engine as eng
    real_score_play = eng.score_play

    def fake_score_play(selected, **kw):
        res = real_score_play(selected, **kw)
        return dataclasses.replace(res, money_delta=7)

    monkeypatch.setattr(eng, "score_play", fake_score_play)
    s = reset(seed=1)
    money_before = s.money
    s2, _ = step(s, (Verb.PLAY, (0,)))
    assert s2.money == money_before + 7


def test_engine_applies_destruction_probe(monkeypatch):
    """PROBE: a forced ScoreResult.destroyed_idx drops the played card from master_deck."""
    import balatro_rl.engine.engine as eng
    real_score_play = eng.score_play

    def fake_score_play(selected, **kw):
        res = real_score_play(selected, **kw)
        return dataclasses.replace(res, destroyed_idx=(0,))

    monkeypatch.setattr(eng, "score_play", fake_score_play)
    s = reset(seed=1)
    # Identity of the card being played (index 0) -> it should leave master_deck.
    played_card_id = id(s.hand[0])
    assert any(id(c) == played_card_id for c in s.master_deck)
    s2, _ = step(s, (Verb.PLAY, (0,)))
    assert len(s2.master_deck) == 51
    assert not any(id(c) == played_card_id for c in s2.master_deck), "destroyed card still owned"


def test_score_result_side_effect_fields_default_to_noop():
    res = ScoreResult(score=0, hand_type=0, chips=0, mult=1.0, scoring_idx=())
    assert res.money_delta == 0
    assert res.destroyed_idx == ()


# --------------------------------------------------------------------------
# Piece 3: relaxed hand-size assert / clamped legal mask
# --------------------------------------------------------------------------

def _normal_state(seed=1):
    return reset(seed=seed)


def test_legal_mask_unchanged_for_normal_8_card_hand():
    """The mask for a standard 8-card hand must be byte-identical to enumerating
    every engine action directly (no clamping kicks in)."""
    s = _normal_state(seed=1)
    assert len(s.hand) == MAX_HAND
    mask = legal_mask(s)
    # Reproduce the old (assert-based) behavior: every engine action sets its bit.
    from balatro_rl.engine.engine import legal_actions
    from balatro_rl.envs.actions import encode_action
    expected = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for verb, arg in legal_actions(s):
        expected[encode_action(verb, arg)] = True
    assert np.array_equal(mask, expected)


def test_legal_mask_smaller_hand_does_not_raise_and_clamps():
    """A 6-card hand: no assert, only in-range subsets offered, absent slots illegal."""
    s = reset(seed=1)
    six = s.hand[:6]
    s = dataclasses.replace(s, hand=six)
    mask = legal_mask(s)  # must not raise
    # Any legal PLAY/DISCARD subset only references slots 0..5.
    from balatro_rl.envs.actions import decode
    for i in np.flatnonzero(mask):
        verb, arg = decode(int(i))
        if verb in (Verb.PLAY, Verb.DISCARD):
            assert all(j < 6 for j in arg), f"subset {arg} references an absent slot"
    # A subset touching slot 6 or 7 must be illegal.
    from balatro_rl.envs.actions import encode_action
    assert not mask[encode_action(Verb.PLAY, (0, 6))]
    assert not mask[encode_action(Verb.PLAY, (7,))]


def test_legal_mask_larger_hand_does_not_raise_and_clamps_to_max_hand():
    """A 9-card hand: no assert; only subsets over the first MAX_HAND=8 slots."""
    s = reset(seed=1)
    nine = s.hand + (Card(2, 0),)  # 9 cards
    s = dataclasses.replace(s, hand=nine)
    assert len(s.hand) == 9
    mask = legal_mask(s)  # must not raise
    from balatro_rl.envs.actions import decode
    for i in np.flatnonzero(mask):
        verb, arg = decode(int(i))
        if verb in (Verb.PLAY, Verb.DISCARD):
            # No flat id references slot >= MAX_HAND, so the 9th card is never offered.
            assert all(j < MAX_HAND for j in arg)
