"""Shop: offer generation, pricing, reroll, sell. Verified values in
docs/reference/economy-shop.md.

Phase E1 generalizes shop offers beyond jokers: each card slot rolls a KIND by the
wiki composition (balatrowiki.org/w/Shop) then the specific item. The wiki base weights
are Joker 20 / Tarot 4 / Planet 4 (Spectral only via the Ghost deck). E1 implements
JOKER + PLANET only (Tarot's effects arrive in Phase E2), so the slot weighting here is
Joker 20 / Planet 4. Consumables cost $3 (wiki: Tarot/Planet base price). Per-joker prices
are the deterministic base costs from the registry (edition surcharges deferred).

The agent stays BLIND to consumable offers in E1: legal_actions only offers joker buys and
the obs encoder zeroes non-joker offer types. Consumable buys are reachable only via direct
engine.step((Verb.BUY, i)) (engine-first), until the Phase E5 obs/action widening.
"""
from __future__ import annotations

import dataclasses
from enum import IntEnum

from .consumables import ConsumableKind, PlanetType
from .jokers import library as _library  # noqa: F401  (ensures REGISTRY is populated)
from .jokers.base import REGISTRY, JokerState, JokerType, Rarity

CARD_SLOTS = 2
REROLL_BASE = 5
CONSUMABLE_COST = 3           # wiki: Tarot/Planet base shop price ($3); Spectral is $4

# Slot KIND weights (balatrowiki.org/w/Shop). E1 scope = JOKER + PLANET only (Tarot
# effects land in E2, Spectral is Ghost-deck-only and deferred), so we drop Tarot/Spectral
# from the roll for now. Wiki base: Joker 20 / Tarot 4 / Planet 4.
_KIND_WEIGHTS: tuple[tuple[int, int], ...] = ()   # set after ShopKind is defined (below)


class ShopKind(IntEnum):
    JOKER = 1
    TAROT = 2
    PLANET = 3
    SPECTRAL = 4


_KIND_WEIGHTS = ((ShopKind.JOKER, 20), (ShopKind.PLANET, 4))

# A bought consumable must be stored with its ConsumableKind (what USE/obs/replay read),
# NOT the ShopKind — the two enums number their members differently (e.g. ShopKind.PLANET=3
# vs ConsumableKind.PLANET=1). This maps the shop's consumable kinds onto consumable kinds.
SHOP_TO_CONSUMABLE_KIND: dict[int, int] = {
    int(ShopKind.PLANET): int(ConsumableKind.PLANET),
    int(ShopKind.TAROT): int(ConsumableKind.TAROT),
    int(ShopKind.SPECTRAL): int(ConsumableKind.SPECTRAL),
}


@dataclasses.dataclass(frozen=True, slots=True)
class ShopItem:
    """A typed shop offer. `kind` (ShopKind) selects the family; `type_id` is the specific
    item within it (a JokerType value for JOKER, a PlanetType value for PLANET); `cost` is
    the price in dollars. POD so it rides GameState cleanly."""
    kind: int
    type_id: int
    cost: int


def joker_cost(jtype: JokerType) -> int:
    return REGISTRY[jtype].cost


def item_cost(item: ShopItem) -> int:
    """Price of a shop offer: jokers read the registry, consumables use the flat base cost."""
    if item.kind == ShopKind.JOKER:
        return joker_cost(JokerType(item.type_id))
    return CONSUMABLE_COST


def reroll_cost(rerolls_done: int, base: int = REROLL_BASE) -> int:
    return max(0, base + rerolls_done)


def sell_value(jtype: JokerType, sell_bonus: int = 0) -> int:
    return max(1, REGISTRY[jtype].cost // 2) + sell_bonus


def _roll_rarity(rng):
    r, rng = rng.random()
    if r < 0.05:
        return Rarity.RARE, rng
    if r < 0.30:
        return Rarity.UNCOMMON, rng
    return Rarity.COMMON, rng


def _pool(rarity: Rarity) -> list[JokerType]:
    # Deterministic order (registry insertion order); never Legendary in shop.
    return [t for t in REGISTRY if REGISTRY[t].rarity == rarity and rarity != Rarity.LEGENDARY]


def _roll_kind(rng):
    """Roll a slot KIND by weight (Joker 20 / Planet 4), threading rng."""
    total = sum(w for _, w in _KIND_WEIGHTS)
    r, rng = rng.randint(0, total - 1)
    acc = 0
    for kind, w in _KIND_WEIGHTS:
        acc += w
        if r < acc:
            return kind, rng
    return _KIND_WEIGHTS[-1][0], rng   # unreachable; defensive


def _roll_joker(rng) -> tuple[ShopItem, object]:
    rarity, rng = _roll_rarity(rng)
    pool = _pool(rarity) or _pool(Rarity.COMMON)
    idx, rng = rng.randint(0, len(pool) - 1)
    jtype = pool[idx]
    return ShopItem(kind=int(ShopKind.JOKER), type_id=int(jtype),
                    cost=joker_cost(jtype)), rng


def _roll_planet(rng) -> tuple[ShopItem, object]:
    planets = list(PlanetType)
    idx, rng = rng.randint(0, len(planets) - 1)
    return ShopItem(kind=int(ShopKind.PLANET), type_id=int(planets[idx]),
                    cost=CONSUMABLE_COST), rng


def generate_offers(rng, n: int = CARD_SLOTS):
    """Generate n typed shop offers. Each slot rolls a KIND (Joker 20 / Planet 4) then the
    specific item: JOKER -> rarity-weighted registry pick (cost from registry); PLANET ->
    a uniform PlanetType (cost $3). Returns (tuple[ShopItem], rng)."""
    offers = []
    for _ in range(n):
        kind, rng = _roll_kind(rng)
        if kind == ShopKind.PLANET:
            item, rng = _roll_planet(rng)
        else:
            item, rng = _roll_joker(rng)
        offers.append(item)
    return tuple(offers), rng
