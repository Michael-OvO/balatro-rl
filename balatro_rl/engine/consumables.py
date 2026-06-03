"""Consumables (Phase D): Tarot / Planet / Spectral cards held in consumable slots and
applied via the USE action. This module is the consumable VALUE TYPES + effect resolution;
the engine owns the slots (GameState.consumables) and the USE step.

D1b implements Planet cards (level up a poker hand). Tarots (enhance/transform selected
cards) and Spectrals arrive in later sub-phases. Acquisition is a stopgap (states are
constructed with consumables directly, like make_master_deck for enhancements) until shop
offers are wired. Values verified against balatrowiki.org.
"""
from __future__ import annotations

import dataclasses
from enum import IntEnum

from .hands import HandType


class ConsumableKind(IntEnum):
    NONE = 0
    PLANET = 1
    TAROT = 2
    SPECTRAL = 3


class PlanetType(IntEnum):
    PLUTO = 1
    MERCURY = 2
    URANUS = 3
    VENUS = 4
    SATURN = 5
    JUPITER = 6
    EARTH = 7
    MARS = 8
    NEPTUNE = 9
    PLANET_X = 10
    CERES = 11
    ERIS = 12


# Which poker hand each Planet levels up (balatrowiki.org/w/Planet_Cards).
PLANET_HAND: dict[PlanetType, HandType] = {
    PlanetType.PLUTO: HandType.HIGH_CARD,
    PlanetType.MERCURY: HandType.PAIR,
    PlanetType.URANUS: HandType.TWO_PAIR,
    PlanetType.VENUS: HandType.THREE_OF_A_KIND,
    PlanetType.SATURN: HandType.STRAIGHT,
    PlanetType.JUPITER: HandType.FLUSH,
    PlanetType.EARTH: HandType.FULL_HOUSE,
    PlanetType.MARS: HandType.FOUR_OF_A_KIND,
    PlanetType.NEPTUNE: HandType.STRAIGHT_FLUSH,
    PlanetType.PLANET_X: HandType.FIVE_OF_A_KIND,
    PlanetType.CERES: HandType.FLUSH_HOUSE,
    PlanetType.ERIS: HandType.FLUSH_FIVE,
}


@dataclasses.dataclass(frozen=True, slots=True)
class Consumable:
    """An owned consumable: `kind` (ConsumableKind) + `type_id` (the specific card within
    that kind, e.g. a PlanetType value). POD so it rides GameState cleanly."""
    kind: int
    type_id: int


def planet(ptype: PlanetType) -> Consumable:
    """Convenience constructor for a Planet consumable."""
    return Consumable(kind=int(ConsumableKind.PLANET), type_id=int(ptype))


def apply_consumable(state, con: Consumable) -> dict:
    """Resolve a consumable's effect into GameState field overrides (applied by the engine's
    USE step, which also removes the card). A Planet raises its hand type's level by 1.
    Raises for not-yet-implemented kinds (Tarot/Spectral) so an unknown USE fails loudly."""
    if con.kind == ConsumableKind.PLANET:
        ht = int(PLANET_HAND[PlanetType(con.type_id)])
        levels = list(state.levels)
        levels[ht] += 1
        return {"levels": tuple(levels)}
    raise NotImplementedError(f"consumable kind {con.kind} not implemented yet")
