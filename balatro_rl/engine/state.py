"""GameState: frozen, plain-data, carries its own RNG so step() is pure.

POD by design (tuples + scalars + a frozen RNG, no object graphs) so it crosses
a future FFI boundary cleanly and a replay is fully reconstructable from a seed.
"""
from __future__ import annotations

import dataclasses
from enum import IntEnum

from .cards import Card
from .rng import RNG


class Phase(IntEnum):
    PLAYING = 0
    WON = 1
    LOST = 2
    SHOP = 3


@dataclasses.dataclass(frozen=True, slots=True)
class GameState:
    deck: tuple[Card, ...]      # remaining draw pile (front = next to draw)
    hand: tuple[Card, ...]      # current hand
    ante: int
    blind_index: int            # 0 = small, 1 = big, 2 = boss
    round_score: int            # chips scored so far this blind
    required: int               # score needed to clear this blind
    hands_left: int
    discards_left: int
    hand_size: int
    levels: tuple[int, ...]     # 12 hand-type levels (HandType order)
    money: int
    rng: RNG
    phase: Phase
    done: bool
    won: bool
    jokers: tuple = ()   # tuple[JokerState, ...]; acquired via the shop
    shop_offers: tuple = ()   # tuple[JokerState, ...] offered in the shop
    rerolls_done: int = 0      # rerolls used in the current shop (for reroll cost)
    shop_steps: int = 0        # actions taken this shop visit; bounds shop dithering
