"""Playing-card encoding. Plain data so it crosses a future FFI boundary cleanly.

Modifier fields (enhancement/edition/seal) carry an IntEnum code each (default 0
= "none"). Phase B scores them (see scoring.py); the obs/agent stays blind to them
until the final coordinated retrain (Phase D), so nothing here touches the network.
"""
from __future__ import annotations

import dataclasses
from enum import IntEnum

RANK_MIN, RANK_MAX = 2, 14  # J=11, Q=12, K=13, A=14
_RANK_NAMES = {11: "J", 12: "Q", 13: "K", 14: "A"}
_SUIT_GLYPH = {0: "♠", 1: "♥", 2: "♣", 3: "♦"}  # ♠♥♣♦


class Enhancement(IntEnum):
    """Card enhancements (wiki: https://balatrowiki.org/w/Enhancement).

    NONE keeps the byte-identical base game (no enhancement code runs). Values
    match the wiki: Bonus +30c, Mult +4m, Wild (any suit), Glass X2m + 1-in-4
    shatter, Steel X1.5m held, Gold +$3 held at round end, Lucky 1-in-5 +20m /
    1-in-15 +$20, Stone +50c / no rank or suit / always scores.
    """
    NONE = 0
    BONUS = 1
    MULT = 2
    WILD = 3
    GLASS = 4
    STEEL = 5
    GOLD = 6
    LUCKY = 7
    STONE = 8


class Edition(IntEnum):
    """Card editions (wiki: https://balatrowiki.org/w/Editions).

    Foil +50c, Holographic +10m, Polychrome X1.5m, all when scored. NEGATIVE is
    deliberately excluded: it only applies to Jokers/Consumables (+1 slot), never
    to playing-card scoring.
    """
    NONE = 0
    FOIL = 1
    HOLO = 2
    POLY = 3


class Seal(IntEnum):
    """Card seals (wiki: https://balatrowiki.org/w/Seals).

    GOLD +$3 when the card is played and scores; RED retriggers the card once;
    BLUE (planet on round end if held) and PURPLE (tarot on discard) require the
    consumables subsystem and are DEFERRED to Phase D (no-op here).
    """
    NONE = 0
    GOLD = 1
    RED = 2
    BLUE = 3
    PURPLE = 4


@dataclasses.dataclass(frozen=True, slots=True)
class Card:
    rank: int            # 2..14
    suit: int            # 0..3
    enhancement: int = 0  # Enhancement code (0 = none)
    edition: int = 0      # Edition code (0 = none)
    seal: int = 0         # Seal code (0 = none)


def rank_chip_value(rank: int) -> int:
    if rank == 14:        # Ace
        return 11
    if rank >= 11:        # J, Q, K
        return 10
    return rank           # 2..10


def is_stone(card: Card) -> bool:
    """A Stone card has no rank or suit and always scores (wiki: Stone Card)."""
    return card.enhancement == Enhancement.STONE


def scores_as_suit(card: Card, suit: int) -> bool:
    """Whether `card` counts as `suit` for scoring purposes.

    A Wild card is every suit simultaneously (matches any). A Stone card has no
    suit (matches none). Every other card matches only its own suit. Unmodified
    cards therefore behave exactly as `card.suit == suit`.
    """
    if is_stone(card):
        return False
    if card.enhancement == Enhancement.WILD:
        return True
    return card.suit == suit


def standard_deck() -> list[Card]:
    return [Card(rank=r, suit=s)
            for s in range(4)
            for r in range(RANK_MIN, RANK_MAX + 1)]


def card_str(c: Card) -> str:
    r = _RANK_NAMES.get(c.rank, str(c.rank))
    return f"{r}{_SUIT_GLYPH[c.suit]}"
