"""Poker-hand identification for a played subset of <=5 cards.

Returns the best hand type and the indices of the cards that *score* (which
differs by hand: a Four-of-a-Kind's kicker does not score; a Flush scores all).
Secret hands (Five of a Kind, Flush House, Flush Five) are encoded and reachable
only once duplicate ranks/wilds exist (Tier-2+); the base 52-card deck can't form
them, but the detection is here so scoring never needs to change later.
"""
from __future__ import annotations

from collections import Counter
from enum import IntEnum

from .cards import Card


class HandType(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    FIVE_OF_A_KIND = 9
    FLUSH_HOUSE = 10
    FLUSH_FIVE = 11


# (base_chips, base_mult) at level 1.
HAND_BASE: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD: (5, 1),
    HandType.PAIR: (10, 2),
    HandType.TWO_PAIR: (20, 2),
    HandType.THREE_OF_A_KIND: (30, 3),
    HandType.STRAIGHT: (30, 4),
    HandType.FLUSH: (35, 4),
    HandType.FULL_HOUSE: (40, 4),
    HandType.FOUR_OF_A_KIND: (60, 7),
    HandType.STRAIGHT_FLUSH: (100, 8),
    HandType.FIVE_OF_A_KIND: (120, 12),
    HandType.FLUSH_HOUSE: (140, 14),
    HandType.FLUSH_FIVE: (160, 16),
}


def _is_straight(ranks: list[int]) -> bool:
    u = sorted(set(ranks))
    if len(u) != 5:
        return False
    if u[-1] - u[0] == 4:
        return True
    return u == [2, 3, 4, 5, 14]  # Ace-low: A-2-3-4-5


def evaluate(cards: list[Card]) -> tuple[HandType, tuple[int, ...]]:
    """Best (HandType, scoring-card indices) for 1..5 played cards."""
    n = len(cards)
    if n == 0:
        raise ValueError("evaluate() requires at least one card")
    ranks = [c.rank for c in cards]
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    is_flush = n == 5 and len(set(suits)) == 1
    is_straight = n == 5 and _is_straight(ranks)
    all_idx = tuple(range(n))

    def idx_with_count(k: int) -> list[int]:
        targets = {r for r, c in rank_counts.items() if c == k}
        return [i for i, r in enumerate(ranks) if r in targets]

    if is_flush and counts == [5]:
        return HandType.FLUSH_FIVE, all_idx
    if is_flush and counts == [3, 2]:
        return HandType.FLUSH_HOUSE, all_idx
    if counts == [5]:
        return HandType.FIVE_OF_A_KIND, all_idx
    if is_flush and is_straight:
        return HandType.STRAIGHT_FLUSH, all_idx
    if counts and counts[0] == 4:
        return HandType.FOUR_OF_A_KIND, tuple(idx_with_count(4))
    if counts == [3, 2]:
        return HandType.FULL_HOUSE, all_idx
    if is_flush:
        return HandType.FLUSH, all_idx
    if is_straight:
        return HandType.STRAIGHT, all_idx
    if counts and counts[0] == 3:
        return HandType.THREE_OF_A_KIND, tuple(idx_with_count(3))
    if counts[:2] == [2, 2]:
        return HandType.TWO_PAIR, tuple(idx_with_count(2))
    if counts and counts[0] == 2:
        return HandType.PAIR, tuple(idx_with_count(2))
    hi = max(range(n), key=lambda i: ranks[i])
    return HandType.HIGH_CARD, (hi,)
