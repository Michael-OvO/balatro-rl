"""Shop: offer generation, pricing, reroll, sell. Verified values in
docs/reference/economy-shop.md. Tier-3 scope offers only Jokers in card slots
(Tarot/Planet/Spectral and packs/vouchers are later plans); per-joker prices are
the deterministic base costs from the registry (edition surcharges deferred).
"""
from __future__ import annotations

from .jokers import library as _library  # noqa: F401  (ensures REGISTRY is populated)
from .jokers.base import REGISTRY, JokerState, JokerType, Rarity

CARD_SLOTS = 2
REROLL_BASE = 5

# Joker-rarity distribution once a Joker rolls (Common 70 / Uncommon 25 / Rare 5).
_RARITY_THRESHOLDS = ((Rarity.RARE, 0.05), (Rarity.UNCOMMON, 0.30))  # else COMMON


def joker_cost(jtype: JokerType) -> int:
    return REGISTRY[jtype].cost


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


def generate_offers(rng, n: int = CARD_SLOTS):
    """Generate n shop joker offers, rarity-weighted, from the registry. Returns
    (tuple[JokerState], rng). Falls back to Common if a rarity pool is empty."""
    offers = []
    for _ in range(n):
        rarity, rng = _roll_rarity(rng)
        pool = _pool(rarity) or _pool(Rarity.COMMON)
        idx, rng = rng.randint(0, len(pool) - 1)
        offers.append(JokerState(type=pool[idx]))
    return tuple(offers), rng
