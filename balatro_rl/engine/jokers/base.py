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
    """Mutable scratch used only during one hand's scoring (never stored in state)."""
    chips: int = 0
    mult: float = 0.0
    played: list = dataclasses.field(default_factory=list)
    scoring_idx: list = dataclasses.field(default_factory=list)
    held: list = dataclasses.field(default_factory=list)
    hand_type: object = None
    rules: RuleFlags = NO_RULES
    first_face_idx: int | None = None
    contains: frozenset = frozenset()


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

    def on_play(self, state, played, scoring_idx, rules, js: "JokerState") -> "JokerState":
        """Lifecycle after a hand is played; return updated JokerState (scaling)."""
        return js

    def on_round_end(self, state, js: "JokerState", rng):
        """End-of-round (cash-out) lifecycle. Returns
        (updated JokerState, money_delta:int, destroy:bool, rng).
        rng is threaded for probabilistic effects (e.g. self-destroy)."""
        return js, 0, False, rng


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
