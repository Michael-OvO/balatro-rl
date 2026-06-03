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

from .cards import Card, Enhancement, is_stone
from .jokers.base import RuleFlags, NO_RULES


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

# Chips & Mult added PER LEVEL above 1 (Planet card upgrades). Verified against
# balatrowiki.org/w/Planet_Cards. A hand at level L scores HAND_BASE + INC*(L-1).
HAND_LEVEL_INC: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD: (10, 1),
    HandType.PAIR: (15, 1),
    HandType.TWO_PAIR: (20, 1),
    HandType.THREE_OF_A_KIND: (20, 2),
    HandType.STRAIGHT: (30, 3),
    HandType.FLUSH: (15, 2),
    HandType.FULL_HOUSE: (25, 2),
    HandType.FOUR_OF_A_KIND: (30, 3),
    HandType.STRAIGHT_FLUSH: (40, 4),
    HandType.FIVE_OF_A_KIND: (35, 3),
    HandType.FLUSH_HOUSE: (40, 4),
    HandType.FLUSH_FIVE: (50, 3),
}


def leveled_base(hand_type: HandType, levels: tuple = ()) -> tuple[int, int]:
    """Base (chips, mult) for a hand at its current level. `levels` is the 12-tuple of
    per-HandType levels (HandType order); missing/empty reads as level 1, so callers that
    don't track levels get the unchanged HAND_BASE (byte-identical)."""
    base_c, base_m = HAND_BASE[hand_type]
    ht = int(hand_type)
    lvl = levels[ht] if ht < len(levels) else 1
    if lvl <= 1:
        return base_c, base_m
    inc_c, inc_m = HAND_LEVEL_INC[hand_type]
    return base_c + inc_c * (lvl - 1), base_m + inc_m * (lvl - 1)


def _is_straight(ranks: list[int]) -> bool:
    u = sorted(set(ranks))
    if len(u) != 5:
        return False
    if u[-1] - u[0] == 4:
        return True
    return u == [2, 3, 4, 5, 14]  # Ace-low: A-2-3-4-5


def _is_flush(cards: list[Card]) -> bool:
    """5 cards that all share a suit, honoring mods.

    A Wild card matches any suit; a Stone card has no suit (so any Stone card
    blocks the flush). With unmodified cards this reduces to the old
    `len(set(suits)) == 1`. We test each of the four concrete suits against the
    non-Wild cards: the hand is a flush iff some suit is shared by every non-Wild
    card (and no card is Stone).
    """
    if len(cards) != 5:
        return False
    if any(is_stone(c) for c in cards):
        return False
    concrete = [c.suit for c in cards if c.enhancement != Enhancement.WILD]
    if not concrete:           # all Wild -> trivially one suit
        return True
    return all(s == concrete[0] for s in concrete)


def _rank_cards(cards: list[Card]) -> list[Card]:
    """Cards that carry a rank for poker evaluation (Stone has none)."""
    return [c for c in cards if not is_stone(c)]


def is_face(card: Card, rules: RuleFlags = NO_RULES) -> bool:
    """King/Queen/Jack, or any card when Pareidolia (all_face) is active.

    A Stone card has no rank and is never a face card, even under Pareidolia.
    """
    if is_stone(card):
        return False
    return rules.all_face or card.rank in (11, 12, 13)


def contains(cards: list[Card]) -> frozenset[HandType]:
    """Sub-hands the played cards *contain* (Balatro "contains" semantics).

    A hand contains a sub-hand if the cards include it, which differs from "is":
    a Four-of-a-Kind contains a Pair and a Three-of-a-Kind but NOT a Two Pair
    (it is one rank, not two distinct paired ranks); a Full House contains all of
    Pair/Two Pair/Three of a Kind/Full House. STRAIGHT/FLUSH need 5 cards.
    HIGH_CARD is never reported.
    """
    # Stone cards have no rank, so they are excluded from rank-based sub-hands
    # (pairs/trips/quads/straights) and from flushes (no suit). Unmodified hands
    # carry no Stone card, so `ranks`/`rank_counts` are exactly the old values.
    ranks = [c.rank for c in _rank_cards(cards)]
    rank_counts = Counter(ranks)
    pairs = [r for r, c in rank_counts.items() if c >= 2]
    has_trip = any(c >= 3 for c in rank_counts.values())
    has_quad = any(c >= 4 for c in rank_counts.values())
    # Full House: a rank with count>=3 AND a DIFFERENT rank with count>=2.
    has_full = any(c >= 3 for c in rank_counts.values()) and any(
        r2 != r1 and c2 >= 2
        for r1, c1 in rank_counts.items() if c1 >= 3
        for r2, c2 in rank_counts.items()
    )
    is_flush = _is_flush(cards)
    is_straight = len(cards) == 5 and _is_straight(ranks)

    out: set[HandType] = set()
    if len(pairs) >= 1:
        out.add(HandType.PAIR)
    if len(pairs) >= 2:
        out.add(HandType.TWO_PAIR)
    if has_trip:
        out.add(HandType.THREE_OF_A_KIND)
    if has_quad:
        out.add(HandType.FOUR_OF_A_KIND)
    if has_full:
        out.add(HandType.FULL_HOUSE)
    if is_straight:
        out.add(HandType.STRAIGHT)
    if is_flush:
        out.add(HandType.FLUSH)
    if is_straight and is_flush:
        out.add(HandType.STRAIGHT_FLUSH)
    return frozenset(out)


def evaluate(cards: list[Card], rules: RuleFlags = NO_RULES) -> tuple[HandType, tuple[int, ...]]:
    """Best (HandType, scoring-card indices) for 1..5 played cards.

    With rules.splash, every played card scores (indices = all), though the hand
    type is still the best poker hand.
    """
    n = len(cards)
    if n == 0:
        raise ValueError("evaluate() requires at least one card")
    # Stone cards carry no rank/suit: they are excluded from poker classification
    # here and force-added to the scoring set by score_play (like Splash). With an
    # unmodified hand `rank_idx` is range(n), so `ranks`/`counts` are unchanged.
    rank_idx = [i for i, c in enumerate(cards) if not is_stone(c)]
    ranks = [cards[i].rank for i in rank_idx]
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    is_flush = _is_flush(cards)
    is_straight = n == 5 and _is_straight(ranks)
    all_idx = tuple(range(n))

    def idx_with_count(k: int) -> tuple[int, ...]:
        targets = {r for r, c in rank_counts.items() if c == k}
        return tuple(i for i in rank_idx if cards[i].rank in targets)

    if is_flush and counts == [5]:
        hand_type, idx = HandType.FLUSH_FIVE, all_idx
    elif is_flush and counts == [3, 2]:
        hand_type, idx = HandType.FLUSH_HOUSE, all_idx
    elif counts == [5]:
        hand_type, idx = HandType.FIVE_OF_A_KIND, all_idx
    elif is_flush and is_straight:
        hand_type, idx = HandType.STRAIGHT_FLUSH, all_idx
    elif counts and counts[0] == 4:
        hand_type, idx = HandType.FOUR_OF_A_KIND, idx_with_count(4)
    elif counts == [3, 2]:
        hand_type, idx = HandType.FULL_HOUSE, all_idx
    elif is_flush:
        hand_type, idx = HandType.FLUSH, all_idx
    elif is_straight:
        hand_type, idx = HandType.STRAIGHT, all_idx
    elif counts and counts[0] == 3:
        hand_type, idx = HandType.THREE_OF_A_KIND, idx_with_count(3)
    elif counts[:2] == [2, 2]:
        hand_type, idx = HandType.TWO_PAIR, idx_with_count(2)
    elif counts and counts[0] == 2:
        hand_type, idx = HandType.PAIR, idx_with_count(2)
    elif rank_idx:
        hi = max(rank_idx, key=lambda i: cards[i].rank)
        hand_type, idx = HandType.HIGH_CARD, (hi,)
    else:
        # All played cards are Stone (no rank): no rank-based scorer. score_play
        # force-adds every Stone index, so the empty idx here is intentional.
        hand_type, idx = HandType.HIGH_CARD, ()

    if rules.splash:
        idx = all_idx
    return hand_type, idx
