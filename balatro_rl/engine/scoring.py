"""Scoring pipeline. Folds joker hooks over a played hand.

Order (per wiki https://balatrowiki.org/w/Scoring):
  1. each scoring card L->R, repeated (1 + retriggers): card chips + on_score hooks
  2. each held card: on_held hooks
  3. independent jokers in slot order
Additive (+chips/+mult) applies before ×mult within the fold (slot order matters).
With no jokers the result is identical to the Plan-1 base scoring.
"""
from __future__ import annotations

import dataclasses

from .cards import Card, rank_chip_value
from .hands import HAND_BASE, HandType, contains, evaluate, is_face
from .jokers.base import (
    NO_RULES, ScoreContext, aggregate_rules, resolve_providers,
)
from .rng import RNG


@dataclasses.dataclass(frozen=True, slots=True)
class ScoreResult:
    score: int
    hand_type: HandType
    chips: int
    mult: float
    scoring_idx: tuple[int, ...]
    rng: object = None    # rng after probabilistic hooks consumed it (threaded back to GameState)
    # Side effects threaded out of the pure scoring fold; the engine applies them
    # (mirrors how res.rng is written back). Both default to no-op: no joker /
    # enhancement produces them yet, so they are always 0 / () for the current game.
    money_delta: int = 0              # money gained during scoring (Lucky/Gold seal/Gold enh)
    destroyed_idx: tuple[int, ...] = ()  # indices into the PLAYED hand to destroy (Glass)


def _apply(ctx: ScoreContext, eff) -> None:
    # Within one Effect, additive applies before multiplicative (+chips/+mult, then xmult).
    # Dual-stat ("++") jokers needing a different interleaving should emit separate Effects
    # across hooks rather than combining +mult and xmult in a single Effect.
    ctx.chips += eff.chips
    ctx.mult += eff.mult
    ctx.mult *= eff.xmult


def score_play(played, jokers: tuple = (), held: tuple = (), *,
               joker_slots: int = 5, money: int = 0, hands_left: int = 0,
               discards_left: int = 0, deck_count: int = 0,
               hand_plays_run: tuple = (), hand_plays_round: tuple = (),
               rng=None) -> ScoreResult:
    """Score one played hand.

    The keyword-only scalars carry read-only game-state info to state-reading
    jokers (Bull, Banner, Mystic Summit, Blue Joker, ...). The engine threads
    them from GameState; callers that only need pure hand scoring may omit them.

    `hand_plays_run` / `hand_plays_round` are the length-12 per-HandType play-count
    tuples (PRE-increment of THIS hand). score_play indexes them by the hand type it
    evaluates and exposes the current-hand count on ScoreContext for Supernova /
    Card Sharp. Empty tuples (default) read as 0 so pure-scoring callers still work.

    `rng` feeds probabilistic scoring jokers (Misprint, Bloodstone). Hooks consume
    it by reassigning ctx.rng in place; the ADVANCED rng is returned on ScoreResult
    so the engine can write it back to GameState (keeps a fixed seed deterministic).
    Defaults to a fixed-seed RNG so pure-scoring callers still construct and roll.
    """
    if rng is None:
        rng = RNG.from_seed(0)
    played = list(played)
    rules = aggregate_rules(jokers) if jokers else NO_RULES
    hand_type, scoring_idx = evaluate(played, rules)
    base_chips, base_mult = HAND_BASE[hand_type]

    ht = int(hand_type)
    plays_run = hand_plays_run[ht] if ht < len(hand_plays_run) else 0
    plays_round = hand_plays_round[ht] if ht < len(hand_plays_round) else 0
    plays_run_max_other = max(
        (c for i, c in enumerate(hand_plays_run) if i != ht), default=0)
    n_jokers = len(jokers)
    ctx = ScoreContext(chips=base_chips, mult=float(base_mult), played=played,
                       scoring_idx=list(scoring_idx), held=list(held),
                       hand_type=hand_type, rules=rules,
                       contains=contains(played),
                       n_jokers=n_jokers,
                       empty_joker_slots=max(0, joker_slots - n_jokers),
                       money=money, hands_left=hands_left,
                       discards_left=discards_left, deck_count=deck_count,
                       hand_plays_run=plays_run, hand_plays_round=plays_round,
                       hand_plays_run_max_other=plays_run_max_other, rng=rng)
    ctx.first_face_idx = next((i for i in scoring_idx if is_face(played[i], rules)), None)
    providers = resolve_providers(jokers)

    # 1) played scoring cards, left to right, with retriggers
    for i in scoring_idx:
        card = played[i]
        retriggers = sum(eff.retrigger(ctx, card, js) for eff, js in providers)
        for _ in range(1 + retriggers):
            ctx.chips += rank_chip_value(card.rank)
            for eff, js in providers:
                _apply(ctx, eff.on_score(ctx, card, i, js))

    # 2) held-in-hand cards
    for card in held:
        for eff, js in providers:
            _apply(ctx, eff.on_held(ctx, card, js))

    # 3) independent jokers, slot order
    for eff, js in providers:
        _apply(ctx, eff.independent(ctx, js))

    # int(floor) matches the game. NOTE: in deep Endless, mult is a float and products
    # above 2**53 lose integer precision (scores may differ from the game by >=1 at extreme
    # antes). Revisit with exact/bignum scoring when Endless is implemented.
    return ScoreResult(score=int(ctx.chips * ctx.mult), hand_type=hand_type,
                       chips=ctx.chips, mult=ctx.mult, scoring_idx=tuple(scoring_idx),
                       rng=ctx.rng, money_delta=ctx.money_delta,
                       destroyed_idx=tuple(ctx.destroyed_idx))
