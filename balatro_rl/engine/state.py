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
    # Times each HandType has been played, in HandType order (mirrors `levels`).
    # _run never resets; _round resets to zeros at each blind boundary. Both are
    # incremented AFTER a played hand is scored (so a joker scoring THAT hand sees
    # the PRE-increment count via ScoreContext.hand_plays_run / _round).
    hand_plays_run: tuple[int, ...]
    hand_plays_round: tuple[int, ...]
    money: int
    rng: RNG
    phase: Phase
    done: bool
    won: bool
    jokers: tuple = ()   # tuple[JokerState, ...]; acquired via the shop
    shop_offers: tuple = ()   # tuple[JokerState, ...] offered in the shop
    rerolls_done: int = 0      # rerolls used in the current shop (for reroll cost)
    shop_steps: int = 0        # actions taken this shop visit; bounds shop dithering
    req_scale: float = 1.0     # curriculum: scales the blind score target (1.0 = real game)
    # The persistent set of owned cards (the 52-card master deck) WITH their mod
    # fields. Each blind reshuffles the working `deck` FROM this, so any card mod
    # (enhancement/edition/seal) rides forward across blinds. Defaults to () for
    # back-compat with directly-constructed states; reset() seeds it from
    # standard_deck(). Card destruction (Glass) drops entries from here.
    master_deck: tuple[Card, ...] = ()
    # Active boss on the current blind (BossEffect int; 0 = NONE / no boss). Set by
    # _advance_blind when entering the boss blind with bosses enabled; 0 on small/big
    # blinds and whenever bosses are disabled. `bosses_enabled` gates boss selection so
    # the default game is byte-identical (the agent stays boss-blind until the retrain).
    boss: int = 0
    bosses_enabled: bool = False
