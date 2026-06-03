"""Joker engine core: value types, the hook protocol, the registry, and the
copy/rule resolution helpers. Each joker is a small JokerEffect registered by
JokerType; the scoring pipeline (scoring.py) folds their hooks.
"""
from __future__ import annotations

import dataclasses
from enum import IntEnum


class JokerType(IntEnum):
    JOKER = 1
    CAVENDISH = 61
    GREEDY = 2
    SCARY_FACE = 33
    PHOTOGRAPH = 78
    BARON = 72
    HACK = 36
    RIDE_THE_BUS = 44
    SPLASH = 52
    PAREIDOLIA = 37
    BLUEPRINT = 123
    GOLDEN_JOKER = 90
    EGG = 46
    # --- Batch 1 ---
    LUSTY = 3
    WRATHFUL = 4
    GLUTTONOUS = 5
    JOLLY = 6
    ZANY = 7
    MAD = 8
    CRAZY = 9
    DROLL = 10
    SLY = 11
    WILY = 12
    CLEVER = 13
    DEVIOUS = 14
    CRAFTY = 15
    HALF = 16
    FIBONACCI = 31
    GROS_MICHEL = 38
    EVEN_STEVEN = 39
    ODD_TODD = 40
    SCHOLAR = 41
    RUNNER = 49
    ICE_CREAM = 50
    WALKIE_TALKIE = 101
    SMILEY_FACE = 104
    SOCK_AND_BUSKIN = 109
    # --- Batch 2: state-reading jokers ---
    JOKER_STENCIL = 17
    BANNER = 22
    MYSTIC_SUMMIT = 23
    ABSTRACT_JOKER = 34
    BLUE_JOKER = 53
    SQUARE_JOKER = 65
    BULL = 93
    POPCORN = 97
    SPARE_TROUSERS = 98
    WEE_JOKER = 124
    # --- Batch 3: hand-contains xMult, suit on-scored, suit-reading, economy ---
    THE_DUO = 131
    THE_TRIO = 132
    THE_FAMILY = 133
    THE_ORDER = 134
    THE_TRIBE = 135
    ONYX_AGATE = 119
    ARROWHEAD = 118
    SEEING_DOUBLE = 128
    FLOWER_POT = 122
    BLACKBOARD = 48
    TO_THE_MOON = 84
    DELAYED_GRATIFICATION = 35
    # --- Batch 4: on_discard lifecycle jokers ---
    FACELESS_JOKER = 57
    GREEN_JOKER = 58
    RAMEN = 100
    # --- Batch 5: probabilistic (rng-in-scoring) + per-round randomized state ---
    MISPRINT = 27
    BLOODSTONE = 117
    ANCIENT_JOKER = 99
    THE_IDOL = 127
    MAIL_IN_REBATE = 83
    # --- Batch 6: hand-play counts (run/round) ---
    SUPERNOVA = 43
    CARD_SHARP = 62
    OBELISK = 75


class Rarity(IntEnum):
    COMMON = 0
    UNCOMMON = 1
    RARE = 2
    LEGENDARY = 3


@dataclasses.dataclass(frozen=True, slots=True)
class Effect:
    """A scoring contribution. Applied as: chips += chips; mult += mult; mult *= xmult."""
    chips: int = 0
    mult: float = 0.0
    xmult: float = 1.0


NO_EFFECT = Effect()


@dataclasses.dataclass(frozen=True, slots=True)
class RuleFlags:
    splash: bool = False     # every played card scores (Splash)
    all_face: bool = False   # all cards count as face cards (Pareidolia)

    def merge(self, other: "RuleFlags") -> "RuleFlags":
        return RuleFlags(splash=self.splash or other.splash,
                         all_face=self.all_face or other.all_face)


NO_RULES = RuleFlags()


@dataclasses.dataclass(frozen=True, slots=True)
class JokerState:
    """Per-instance joker state. `counter` holds scaling value (e.g. Ride the Bus mult)."""
    type: JokerType
    edition: int = 0      # 0 = base (editions are a later plan)
    counter: float = 0.0
    sell_bonus: int = 0   # extra sell value beyond floor(cost/2), e.g. from Egg


@dataclasses.dataclass(slots=True)
class ScoreContext:
    """Mutable scratch used only during one hand's scoring (never stored in state).

    The trailing fields expose read-only game-state info to scoring jokers
    (Abstract Joker, Joker Stencil, Bull, Banner, Mystic Summit, Blue Joker...).
    They are populated in scoring.score_play from the owning GameState; they
    default sensibly so contexts built without state still construct.
    """
    chips: int = 0
    mult: float = 0.0
    played: list = dataclasses.field(default_factory=list)
    scoring_idx: list = dataclasses.field(default_factory=list)
    held: list = dataclasses.field(default_factory=list)
    hand_type: object = None
    rules: RuleFlags = NO_RULES
    first_face_idx: int | None = None
    contains: frozenset = frozenset()
    # --- read-only game-state info (for state-reading jokers) ---
    n_jokers: int = 0           # number of owned jokers
    empty_joker_slots: int = 0  # JOKER_SLOTS - n_jokers
    money: int = 0             # current money ($)
    hands_left: int = 0         # hands remaining this blind
    discards_left: int = 0      # discards remaining this blind
    deck_count: int = 0         # cards left in the draw pile
    # Times the CURRENT hand_type has been played, PRE-increment of this play
    # (i.e. NOT counting the hand being scored). Supernova adds +1 to include the
    # current play; Card Sharp fires when hand_plays_round >= 1 (already this round).
    hand_plays_run: int = 0     # this run
    hand_plays_round: int = 0   # this round
    # Max run play-count among all OTHER hand types (PRE-increment). Lets Obelisk
    # decide AT SCORING TIME whether the hand being played becomes the strict
    # most-played hand (reset) -- wiki: "Obelisk resets before the hand is scored".
    hand_plays_run_max_other: int = 0
    # Probabilistic-scoring RNG. Hooks that consume randomness reassign it in place
    # (`roll, ctx.rng = ctx.rng.random()`); score_play threads the advanced rng back
    # out to GameState so a fixed seed reproduces every roll. Defaults to a fixed seed
    # so contexts built without state still construct (and roll deterministically).
    rng: object = None
    # Side-effect accumulators threaded out on ScoreResult and applied by the engine.
    # No current hook touches them (always 0 / empty now); Phase B's Lucky/Gold/Glass
    # enhancements will accumulate here. destroyed_idx holds indices into `played`.
    money_delta: int = 0
    destroyed_idx: list = dataclasses.field(default_factory=list)


class JokerEffect:
    """Base joker behaviour. Subclasses override only the hooks they need.

    `copyable` declares whether Blueprint/Brainstorm may copy this joker's
    scoring hooks (passive/rule/economy jokers set it False — see wiki).
    """
    copyable: bool = True
    rarity: "Rarity" = None      # set by each joker; Rarity enum
    cost: int = 4                # base shop buy price ($); set by each joker

    def independent(self, ctx, js: "JokerState") -> Effect:
        return NO_EFFECT

    def on_score(self, ctx, card, index: int, js: "JokerState") -> Effect:
        return NO_EFFECT

    def on_held(self, ctx, card, js: "JokerState") -> Effect:
        return NO_EFFECT

    def retrigger(self, ctx, card, js: "JokerState") -> int:
        return 0

    def rules(self) -> RuleFlags:
        return NO_RULES

    def on_round_start(self, state, js: "JokerState", rng):
        """Start-of-blind lifecycle (folded in engine.reset and _advance_blind).
        Returns (updated JokerState, rng); rng is threaded for jokers that pick a
        random per-round target (suit/rank), stashed in js.counter. Default no-op."""
        return js, rng

    def on_play(self, state, played, scoring_idx, rules, js: "JokerState") -> "JokerState":
        """Lifecycle after a hand is played; return updated JokerState (scaling)."""
        return js

    def on_round_end(self, state, js: "JokerState", rng):
        """End-of-round (cash-out) lifecycle. Returns
        (updated JokerState, money_delta:int, destroy:bool, rng).
        rng is threaded for probabilistic effects (e.g. self-destroy)."""
        return js, 0, False, rng

    def on_discard(self, state, discarded, js: "JokerState", rng):
        """After cards are discarded (DISCARD action). `discarded` is the list of
        Card just discarded. Returns (updated JokerState, money_delta:int, rng);
        rng is threaded for probabilistic effects. Scaling counters persist via js."""
        return js, 0, rng

    def destroy_when(self, js: "JokerState") -> bool:
        """Whether this joker should be removed given its current state (consulted
        by the engine after lifecycle folds). Used by self-consuming scalers like
        Ramen (eaten once its X Mult would fall to X1)."""
        return False


REGISTRY: dict[JokerType, JokerEffect] = {}


def register(joker_type: JokerType):
    def deco(cls):
        REGISTRY[joker_type] = cls()
        return cls
    return deco


def aggregate_rules(jokers: tuple) -> RuleFlags:
    flags = NO_RULES
    for js in jokers:
        flags = flags.merge(REGISTRY[js.type].rules())
    return flags


def _blueprint_target(jokers: tuple, i: int) -> int | None:
    """Index of the joker Blueprint at slot i ultimately copies (walks right past
    chained Blueprints; None if it runs off the end).

    The `seen` set is defensive scaffolding for future non-linear copy jokers
    (e.g. Brainstorm copies the leftmost joker); for Blueprint alone, indices
    strictly increase so a cycle is impossible."""
    j = i + 1
    seen = set()
    while j < len(jokers) and jokers[j].type == JokerType.BLUEPRINT:
        if j in seen:
            return None
        seen.add(j)
        j += 1
    return j if j < len(jokers) else None


def resolve_providers(jokers: tuple) -> list:
    """Return [(JokerEffect, JokerState)] in slot order, with Blueprint resolved to
    its target's *copyable* effect (using the target's state)."""
    out = []
    for i, js in enumerate(jokers):
        if js.type == JokerType.BLUEPRINT:
            tgt = _blueprint_target(jokers, i)
            if tgt is None:
                continue
            teff = REGISTRY[jokers[tgt].type]
            if teff.copyable:
                out.append((teff, jokers[tgt]))
        else:
            out.append((REGISTRY[js.type], js))
    return out
