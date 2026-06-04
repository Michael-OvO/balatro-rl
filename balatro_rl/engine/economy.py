"""Run economy: blind rewards and interest. Verified values in
docs/reference/economy-shop.md (balatrowiki.org). Stake/deck modifiers and
voucher cap-raises are later plans (default cap = 5).
"""
from __future__ import annotations

BLIND_REWARD: tuple[int, int, int] = (3, 4, 5)  # Small, Big, Boss
SHOWDOWN_REWARD = 8   # finisher (showdown) boss blind pays $8, not the regular boss $5
INTEREST_PER = 5      # +$1 per $5 held
INTEREST_CAP = 5      # default cap ($5, reached at $25)
MONEY_PER_UNUSED_HAND = 1  # standard decks


def blind_reward(blind_index: int, is_finisher: bool = False) -> int:
    """Cash-out reward for clearing a blind. A finisher/showdown boss (ante 8/16/...) pays
    $8 vs the regular boss $5 (wiki: /w/Economy). Unreachable in the ante 1-8 win (the engine
    wins instead of cashing the finisher), so only matters once Endless is enabled."""
    if blind_index == 2 and is_finisher:
        return SHOWDOWN_REWARD
    return BLIND_REWARD[blind_index]


def interest(money: int, cap: int = INTEREST_CAP) -> int:
    if money <= 0:
        return 0
    return min(money // INTEREST_PER, cap)
