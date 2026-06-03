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

from .cards import Card, Edition, Enhancement, Seal, is_stone, rank_chip_value
from .hands import HAND_BASE, HandType, contains, evaluate, is_face
from .jokers.base import (
    DECK_ENH_ZEROS, NO_RULES, ScoreContext, aggregate_rules, resolve_providers,
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
    lucky_triggered: int = 0          # # Lucky cards that triggered this hand (Lucky Cat scaling)


def _apply(ctx: ScoreContext, eff) -> None:
    # Within one Effect, additive applies before multiplicative (+chips/+mult, then xmult).
    # Dual-stat ("++") jokers needing a different interleaving should emit separate Effects
    # across hooks rather than combining +mult and xmult in a single Effect.
    ctx.chips += eff.chips
    ctx.mult += eff.mult
    ctx.mult *= eff.xmult


def _score_card_mods(ctx: ScoreContext, card: Card) -> None:
    """Apply a SCORED card's enhancement/edition/seal (wiki-verified values).

    Additive (+chips/+mult) is applied before multiplicative (xmult), matching the
    base fold order. RNG is consumed ONLY by Lucky (its two independent rolls) and
    nothing else here, so an unmodified card draws zero rng -> byte-identical to
    the pre-Phase-B game. Glass shatter is rolled separately, after all scoring.
    Seal money (Gold) and Lucky money flow into ctx.money_delta, never the product.
    Callers MUST skip debuffed cards (their mods are nullified by the boss blind).
    """
    enh = card.enhancement
    # --- enhancement: additive first ---
    if enh == Enhancement.BONUS:
        ctx.chips += 30
    elif enh == Enhancement.MULT:
        ctx.mult += 4
    elif enh == Enhancement.STONE:
        ctx.chips += 50                 # Stone's flat value (it has no rank chips)
    elif enh == Enhancement.LUCKY:
        # Two INDEPENDENT rolls: 1-in-5 -> +20 Mult, 1-in-15 -> +$20. Both consume
        # ctx.rng (mult roll first, then money roll), mirroring how Misprint/
        # Bloodstone reassign ctx.rng in place; the advanced rng threads back out.
        triggered = False
        roll, ctx.rng = ctx.rng.random()
        if roll < 1 / 5:
            ctx.mult += 20
            triggered = True
        roll, ctx.rng = ctx.rng.random()
        if roll < 1 / 15:
            ctx.money_delta += 20
            triggered = True
        # Lucky Cat scales per Lucky card that triggers; both rolls hitting still counts
        # ONCE (wiki). Counted here (inside any retrigger loop) so each re-score that hits
        # is its own event; surfaced on ScoreResult for the engine to persist.
        if triggered:
            ctx.lucky_triggers += 1
    # --- edition: additive (Foil/Holo) then multiplicative (Poly) ---
    ed = card.edition
    if ed == Edition.FOIL:
        ctx.chips += 50
    elif ed == Edition.HOLO:
        ctx.mult += 10
    # --- enhancement xmult (Glass) then edition xmult (Poly) ---
    if enh == Enhancement.GLASS:
        ctx.mult *= 2
    if ed == Edition.POLY:
        ctx.mult *= 1.5
    # --- seal money (Gold pays $3 when the card scores) ---
    if card.seal == Seal.GOLD:
        ctx.money_delta += 3
    # BLUE/PURPLE seals create consumables (planet on round-end-if-held / tarot on
    # discard) -> DEFERRED to the consumables phase (Phase D); no-op on score here.


def score_play(played, jokers: tuple = (), held: tuple = (), *,
               joker_slots: int = 5, money: int = 0, hands_left: int = 0,
               discards_left: int = 0, deck_count: int = 0,
               hand_plays_run: tuple = (), hand_plays_round: tuple = (),
               deck_enh_counts: tuple = (), debuffed_idx: tuple = (),
               rng=None) -> ScoreResult:
    """Score one played hand.

    The keyword-only scalars carry read-only game-state info to state-reading
    jokers (Bull, Banner, Mystic Summit, Blue Joker, ...). The engine threads
    them from GameState; callers that only need pure hand scoring may omit them.

    `hand_plays_run` / `hand_plays_round` are the length-12 per-HandType play-count
    tuples (PRE-increment of THIS hand). score_play indexes them by the hand type it
    evaluates and exposes the current-hand count on ScoreContext for Supernova /
    Card Sharp. Empty tuples (default) read as 0 so pure-scoring callers still work.

    `deck_enh_counts` is the full-deck enhancement histogram (counts per Enhancement
    over GameState.master_deck) for jokers that scale off owned enhanced cards (Steel
    Joker X Mult per Steel, Stone Joker +Chips per Stone). Empty (default) normalizes to
    all-zeros on the context, so pure-scoring callers still construct.

    `rng` feeds probabilistic scoring jokers (Misprint, Bloodstone) AND probabilistic
    card mods (Lucky's two rolls, Glass's shatter roll). Hooks consume it by reassigning
    ctx.rng in place; the ADVANCED rng is returned on ScoreResult so the engine can write
    it back to GameState (keeps a fixed seed deterministic). Defaults to a fixed-seed RNG
    so pure-scoring callers still construct and roll. RNG ORDER (per scored card, L->R):
    joker on_score hooks (incl Bloodstone's per-Heart roll) THEN the card's mod fold
    (Lucky mult roll, then Lucky money roll); Misprint's roll is independent (phase 3).
    Glass shatter is rolled once per surviving Glass scoring card AFTER all scoring. Only
    cards that actually carry Lucky/Glass consume rng, so an all-unmodified hand draws
    exactly the same rng as before Phase B (byte-identical).

    `debuffed_idx` lists PLAYED-hand indices whose enhancement/edition/seal are NULLIFIED
    (boss-blind debuff; Phase C supplies it). A debuffed card still scores its normal rank
    chips, but no mod fires and no mod rng is drawn for it. Always () for the current game.
    """
    if rng is None:
        rng = RNG.from_seed(0)
    played = list(played)
    rules = aggregate_rules(jokers) if jokers else NO_RULES
    hand_type, scoring_idx = evaluate(played, rules)
    # Stone cards always score even when they otherwise would not (wiki: Stone Card),
    # so force their indices into the scoring set (like Splash), preserving order and
    # without duplicating any already scored. Unmodified hands carry no Stone -> no-op.
    if any(is_stone(c) for c in played):
        present = set(scoring_idx)
        scoring_idx = tuple(
            i for i in range(len(played))
            if i in present or is_stone(played[i]))
    base_chips, base_mult = HAND_BASE[hand_type]
    debuffed = frozenset(debuffed_idx)

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
                       hand_plays_run_max_other=plays_run_max_other,
                       deck_enh_counts=tuple(deck_enh_counts) or DECK_ENH_ZEROS,
                       rng=rng)
    ctx.first_face_idx = next((i for i in scoring_idx if is_face(played[i], rules)), None)
    providers = resolve_providers(jokers)

    # 1) played scoring cards, left to right, with retriggers
    for i in scoring_idx:
        card = played[i]
        skip_mods = i in debuffed   # boss-blind debuff nullifies this card's mods
        retriggers = sum(eff.retrigger(ctx, card, js) for eff, js in providers)
        # RED seal retriggers this card once more (re-scoring re-applies everything).
        if not skip_mods and card.seal == Seal.RED:
            retriggers += 1
        for _ in range(1 + retriggers):
            # A Stone card has no rank, so it adds no rank chips here; its flat +50
            # comes from the mod fold below (and is skipped when debuffed).
            if not is_stone(card):
                ctx.chips += rank_chip_value(card.rank)
            for eff, js in providers:
                _apply(ctx, eff.on_score(ctx, card, i, js))
            if not skip_mods:
                _score_card_mods(ctx, card)

    # 2) held-in-hand cards
    for card in held:
        for eff, js in providers:
            _apply(ctx, eff.on_held(ctx, card, js))
    # 2b) held-card mods: a held STEEL card gives X1.5 Mult (wiki: Steel Card).
    # Reuses the held phase; unmodified held cards are a no-op (no Steel present).
    # NOTE: held cards are not (yet) addressable by boss debuffs, so no debuff skip
    # here -- Phase C debuffs only ever target PLAYED cards.
    for card in held:
        if card.enhancement == Enhancement.STEEL:
            ctx.mult *= 1.5

    # 3) independent jokers, slot order
    for eff, js in providers:
        _apply(ctx, eff.independent(ctx, js))

    # 4) Glass shatter: AFTER all scoring is finished, each scored GLASS card has a
    # 1-in-4 chance to be destroyed (wiki: Glass Card). One roll per Glass scoring
    # card, in scoring (L->R) order, consuming ctx.rng only for Glass cards. Debuffed
    # Glass cannot shatter (its mod is nullified). destroyed_idx holds PLAYED indices.
    for i in scoring_idx:
        if i in debuffed:
            continue
        if played[i].enhancement == Enhancement.GLASS:
            roll, ctx.rng = ctx.rng.random()
            if roll < 1 / 4:
                ctx.destroyed_idx.append(i)

    # int(floor) matches the game. NOTE: in deep Endless, mult is a float and products
    # above 2**53 lose integer precision (scores may differ from the game by >=1 at extreme
    # antes). Revisit with exact/bignum scoring when Endless is implemented.
    return ScoreResult(score=int(ctx.chips * ctx.mult), hand_type=hand_type,
                       chips=ctx.chips, mult=ctx.mult, scoring_idx=tuple(scoring_idx),
                       rng=ctx.rng, money_delta=ctx.money_delta,
                       destroyed_idx=tuple(ctx.destroyed_idx),
                       lucky_triggered=ctx.lucky_triggers)
