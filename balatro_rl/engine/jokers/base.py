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


class JokerEffect:
    """Base joker behaviour. Subclasses override only the hooks they need.

    `copyable` declares whether Blueprint/Brainstorm may copy this joker's
    scoring hooks (passive/rule/economy jokers set it False — see wiki).
    """
    copyable: bool = True

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
    chained Blueprints; None if it runs off the end)."""
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
