"""Playing-card encoding. Plain data so it crosses a future FFI boundary cleanly.

Modifier fields (enhancement/edition/seal) exist now but default to 0 ("none");
they stay unused until the Tier-2 card-modification plan.
"""
from __future__ import annotations

import dataclasses

RANK_MIN, RANK_MAX = 2, 14  # J=11, Q=12, K=13, A=14
_RANK_NAMES = {11: "J", 12: "Q", 13: "K", 14: "A"}
_SUIT_GLYPH = {0: "♠", 1: "♥", 2: "♣", 3: "♦"}  # ♠♥♣♦


@dataclasses.dataclass(frozen=True, slots=True)
class Card:
    rank: int            # 2..14
    suit: int            # 0..3
    enhancement: int = 0  # 0 = none (Tier-2+)
    edition: int = 0      # 0 = none (Tier-2+)
    seal: int = 0         # 0 = none (Tier-2+)


def rank_chip_value(rank: int) -> int:
    if rank == 14:        # Ace
        return 11
    if rank >= 11:        # J, Q, K
        return 10
    return rank           # 2..10


def standard_deck() -> list[Card]:
    return [Card(rank=r, suit=s)
            for s in range(4)
            for r in range(RANK_MIN, RANK_MAX + 1)]


def card_str(c: Card) -> str:
    r = _RANK_NAMES.get(c.rank, str(c.rank))
    return f"{r}{_SUIT_GLYPH[c.suit]}"
