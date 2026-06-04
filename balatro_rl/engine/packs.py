"""Booster packs (Phase E3): the pack VALUE TYPES + roll/open generation.

A booster pack is bought in the shop; buying it enters an OPEN_PACK sub-phase where you
pick K-of-M revealed items, add the picks to your run (consumables / jokers), then resume
the shop. This module owns the pack model + content generation; the engine owns the
OPEN_PACK phase and the PICK/SKIP step (engine.py).

SCOPE (wiki-verified, balatrowiki.org/w/Booster_Packs): ARCANA (Tarot), CELESTIAL (Planet),
BUFFOON (Joker) packs only. STANDARD (playing-card modifiers) and SPECTRAL (spectrals not
built) are reserved in PackKind but never generated — deferred to a follow-up.

Sizes -> (shown, pick): Arcana/Celestial Normal (3,1) / Jumbo (5,1) / Mega (5,2);
Buffoon Normal (2,1) / Jumbo (4,1) / Mega (4,2). Costs: Normal $4 / Jumbo $6 / Mega $8.

ENGINE-FIRST / agent BLIND: the agent never sees pack BUY (legal_actions withholds
Verb.OPEN), so packs are reachable only via a direct engine.step until the E5 widening.
"""
from __future__ import annotations

import dataclasses
from enum import IntEnum

from .consumables import IMPLEMENTED_TAROTS, PlanetType, planet, tarot
from .jokers import library as _library  # noqa: F401  (ensures REGISTRY is populated)
from .jokers.base import JokerState
from .shop import _pool, _roll_rarity


class PackKind(IntEnum):
    ARCANA = 1       # Tarot cards
    CELESTIAL = 2    # Planet cards
    BUFFOON = 3      # Jokers
    STANDARD = 4     # RESERVED (playing-card modifiers) — deferred, never generated
    SPECTRAL = 5     # RESERVED (spectrals not built) — deferred, never generated


class PackSize(IntEnum):
    NORMAL = 1
    JUMBO = 2
    MEGA = 3


class PackItemKind(IntEnum):
    """What a revealed pack item carries: a Consumable (Arcana/Celestial) or a JokerState
    (Buffoon). Mirrors how the engine routes a pick (consumable slot vs joker slot)."""
    CONSUMABLE = 1
    JOKER = 2


# (kind, size) -> (shown, pick), wiki-verified.
PACK_LAYOUT: dict[tuple[int, int], tuple[int, int]] = {
    # Arcana + Celestial share the same layout.
    **{(k, PackSize.NORMAL): (3, 1) for k in (PackKind.ARCANA, PackKind.CELESTIAL)},
    **{(k, PackSize.JUMBO): (5, 1) for k in (PackKind.ARCANA, PackKind.CELESTIAL)},
    **{(k, PackSize.MEGA): (5, 2) for k in (PackKind.ARCANA, PackKind.CELESTIAL)},
    (PackKind.BUFFOON, PackSize.NORMAL): (2, 1),
    (PackKind.BUFFOON, PackSize.JUMBO): (4, 1),
    (PackKind.BUFFOON, PackSize.MEGA): (4, 2),
}

# Cost by size (wiki: Normal $4 / Jumbo $6 / Mega $8).
_PACK_COST: dict[int, int] = {PackSize.NORMAL: 4, PackSize.JUMBO: 6, PackSize.MEGA: 8}

# Pack KIND weights. Arcana 4 / Celestial 4 / Buffoon 1.2 (approximate — the real shop pack
# pool is more elaborate; these are reasonable relative odds for the implemented kinds).
_PACK_KIND_WEIGHTS: tuple[tuple[int, int], ...] = (
    (PackKind.ARCANA, 40), (PackKind.CELESTIAL, 40), (PackKind.BUFFOON, 12),
)
# Pack SIZE weights: Normal common, Jumbo/Mega rarer (8 / 3 / 1).
_PACK_SIZE_WEIGHTS: tuple[tuple[int, int], ...] = (
    (PackSize.NORMAL, 8), (PackSize.JUMBO, 3), (PackSize.MEGA, 1),
)


def pack_cost(size: int) -> int:
    return _PACK_COST[size]


@dataclasses.dataclass(frozen=True, slots=True)
class Pack:
    """A booster pack offered in the shop. `kind` (PackKind) + `size` (PackSize) determine
    the (shown, pick) layout via PACK_LAYOUT; `cost` is the price in dollars (by size)."""
    kind: int
    size: int
    cost: int


@dataclasses.dataclass(frozen=True, slots=True)
class PackItem:
    """A single revealed item inside an opened pack. `kind` (PackItemKind) selects the
    payload type: CONSUMABLE -> a Consumable (Tarot/Planet); JOKER -> a JokerState. POD so
    it rides GameState.pack_open cleanly during the OPEN_PACK sub-phase."""
    kind: int
    payload: object   # Consumable | JokerState


def _weighted(weights: tuple[tuple[int, int], ...], rng):
    """Pick a value by integer weight, threading rng."""
    total = sum(w for _, w in weights)
    r, rng = rng.randint(0, total - 1)
    acc = 0
    for val, w in weights:
        acc += w
        if r < acc:
            return val, rng
    return weights[-1][0], rng   # unreachable; defensive


def roll_pack(rng) -> tuple[Pack, object]:
    """Roll a random pack: weighted KIND then SIZE, cost by size. Returns (Pack, rng)."""
    kind, rng = _weighted(_PACK_KIND_WEIGHTS, rng)
    size, rng = _weighted(_PACK_SIZE_WEIGHTS, rng)
    return Pack(kind=int(kind), size=int(size), cost=pack_cost(size)), rng


def _roll_tarot_item(rng) -> tuple[PackItem, object]:
    idx, rng = rng.randint(0, len(IMPLEMENTED_TAROTS) - 1)
    return PackItem(kind=int(PackItemKind.CONSUMABLE),
                    payload=tarot(IMPLEMENTED_TAROTS[idx])), rng


def _roll_planet_item(rng) -> tuple[PackItem, object]:
    planets = list(PlanetType)
    idx, rng = rng.randint(0, len(planets) - 1)
    return PackItem(kind=int(PackItemKind.CONSUMABLE),
                    payload=planet(planets[idx])), rng


def _roll_joker_item(rng) -> tuple[PackItem, object]:
    """A rarity-weighted JokerState (reuses shop._roll_rarity / shop._pool)."""
    from .jokers.base import Rarity
    rarity, rng = _roll_rarity(rng)
    pool = _pool(rarity) or _pool(Rarity.COMMON)
    idx, rng = rng.randint(0, len(pool) - 1)
    return PackItem(kind=int(PackItemKind.JOKER), payload=JokerState(type=pool[idx])), rng


def open_pack(pack: Pack, rng) -> tuple[tuple[PackItem, ...], int, object]:
    """Reveal a pack's items + its pick count. Returns (items, picks, rng).

    ARCANA -> random IMPLEMENTED Tarot Consumables; CELESTIAL -> random Planet Consumables;
    BUFFOON -> rarity-weighted JokerStates. `shown`/`pick` come from PACK_LAYOUT.
    """
    shown, pick = PACK_LAYOUT[(pack.kind, pack.size)]
    if pack.kind == PackKind.ARCANA:
        roll = _roll_tarot_item
    elif pack.kind == PackKind.CELESTIAL:
        roll = _roll_planet_item
    elif pack.kind == PackKind.BUFFOON:
        roll = _roll_joker_item
    else:
        raise NotImplementedError(f"pack kind {pack.kind} is deferred / not generated")
    items: list[PackItem] = []
    for _ in range(shown):
        item, rng = roll(rng)
        items.append(item)
    return tuple(items), pick, rng
