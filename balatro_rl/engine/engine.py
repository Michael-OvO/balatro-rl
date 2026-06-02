"""The Tier-0 engine seam: reset / legal_actions / step.

step(state, action) -> (state', info) is a pure function (RNG rides inside the
state). Action = (Verb, tuple-of-hand-indices). The flat-id encoding + legal mask
used by the RL agent live in the env layer (Plan 3); here we use plain tuples.

Tier-0 has no shop: clearing a blind advances directly to the next blind with a
freshly shuffled deck and full hand. Shop/economy arrive in Plan 2.
"""
from __future__ import annotations

import dataclasses
import itertools
from enum import IntEnum

from .blinds import required_score
from .cards import standard_deck
from .rng import RNG
from .scoring import score_play
from .state import GameState, Phase

STARTING_MONEY = 4
HANDS_PER_BLIND = 4
DISCARDS_PER_BLIND = 3
HAND_SIZE = 8
MAX_SELECT = 5


class Verb(IntEnum):
    PLAY = 0
    DISCARD = 1


def _draw(hand: list, deck: list, hand_size: int) -> tuple[list, list]:
    """Draw from the front of the (pre-shuffled) deck up to hand_size."""
    need = max(0, hand_size - len(hand))
    drawn = deck[:need]
    return hand + drawn, deck[need:]


def reset(seed: int) -> GameState:
    rng = RNG.from_seed(seed)
    deck, rng = rng.shuffle(standard_deck())
    hand, deck = _draw([], deck, HAND_SIZE)
    return GameState(
        deck=tuple(deck), hand=tuple(hand), ante=1, blind_index=0,
        round_score=0, required=required_score(1, 0),
        hands_left=HANDS_PER_BLIND, discards_left=DISCARDS_PER_BLIND,
        hand_size=HAND_SIZE, levels=tuple([1] * 12), money=STARTING_MONEY,
        rng=rng, phase=Phase.PLAYING, done=False, won=False,
    )


def legal_actions(state: GameState) -> list[tuple[Verb, tuple[int, ...]]]:
    if state.done:
        return []
    actions: list[tuple[Verb, tuple[int, ...]]] = []
    n = len(state.hand)
    for size in range(1, min(MAX_SELECT, n) + 1):
        for combo in itertools.combinations(range(n), size):
            if state.hands_left > 0:
                actions.append((Verb.PLAY, combo))
            if state.discards_left > 0:
                actions.append((Verb.DISCARD, combo))
    return actions


def _advance_blind(state: GameState, round_score: int, info: dict) -> tuple[GameState, dict]:
    if state.blind_index < 2:
        new_ante, new_blind = state.ante, state.blind_index + 1
    else:
        new_ante, new_blind = state.ante + 1, 0
    if new_ante > 8:
        won = dataclasses.replace(state, round_score=round_score,
                                  done=True, won=True, phase=Phase.WON)
        return won, {**info, "cleared": True, "result": "won"}
    deck, rng = state.rng.shuffle(standard_deck())
    hand, deck = _draw([], deck, state.hand_size)
    nxt = dataclasses.replace(
        state, ante=new_ante, blind_index=new_blind,
        deck=tuple(deck), hand=tuple(hand), round_score=0,
        required=required_score(new_ante, new_blind),
        hands_left=HANDS_PER_BLIND, discards_left=DISCARDS_PER_BLIND, rng=rng,
    )
    return nxt, {**info, "cleared": True, "result": "blind_cleared"}


def step(state: GameState, action: tuple[Verb, tuple[int, ...]]) -> tuple[GameState, dict]:
    assert not state.done, "step() called on a terminal state"
    verb, idx = action
    assert 1 <= len(idx) <= MAX_SELECT, "must select 1..5 cards"
    assert len(set(idx)) == len(idx), "duplicate card indices"
    assert all(0 <= i < len(state.hand) for i in idx), "index out of range"

    selected = [state.hand[i] for i in idx]
    chosen = set(idx)
    remaining = [c for i, c in enumerate(state.hand) if i not in chosen]

    if verb == Verb.DISCARD:
        assert state.discards_left > 0, "no discards left"
        hand, deck = _draw(remaining, list(state.deck), state.hand_size)
        nxt = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                                  discards_left=state.discards_left - 1)
        return nxt, {"verb": "discard", "discarded": len(idx)}

    # PLAY
    assert state.hands_left > 0, "no hands left"
    res = score_play(selected)
    round_score = state.round_score + res.score
    hands_left = state.hands_left - 1
    info = {"verb": "play", "score": res.score, "hand_type": int(res.hand_type),
            "chips": res.chips, "mult": res.mult}

    if round_score >= state.required:
        # Blind cleared: _advance_blind reshuffles a fresh deck and redraws,
        # so we deliberately skip the redraw here.
        return _advance_blind(state, round_score, info)

    hand, deck = _draw(remaining, list(state.deck), state.hand_size)
    if hands_left <= 0:
        lost = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                                   round_score=round_score, hands_left=0,
                                   done=True, won=False, phase=Phase.LOST)
        return lost, {**info, "result": "lost"}
    nxt = dataclasses.replace(state, hand=tuple(hand), deck=tuple(deck),
                              round_score=round_score, hands_left=hands_left)
    return nxt, info
