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
    # E3: OPEN_PACK is the booster-pack sub-phase (pick K-of-M revealed items). The obs
    # encoder guards `phase < N_PHASES=4`, so OPEN_PACK gets NO phase one-hot and g[16] is
    # untouched — the agent stays blind to packs (it's only ever reached via a direct
    # engine.step, since legal_actions never offers Verb.OPEN until the E5 widening).
    OPEN_PACK = 4


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
    # Owned consumables (Tarot/Planet/Spectral) and the slot cap. Applied via the USE
    # action (engine.step). Empty by default -> no consumables, byte-identical. The agent
    # can't see or USE them until the Phase D obs/action widening.
    consumables: tuple = ()      # tuple[Consumable, ...]
    consumable_slots: int = 2
    # E3 booster packs. `pack_offers` are the shop's pack slots (tuple[Pack, ...]), generated
    # alongside the card offers. When a pack is BOUGHT (Verb.OPEN), the engine enters
    # Phase.OPEN_PACK: `pack_open` holds the revealed items (tuple[PackItem, ...]) and
    # `pack_picks` the remaining picks; PICK/SKIP_PACK drain them, then phase returns to SHOP.
    # All default empty/0 -> byte-identical for directly-constructed states; the agent is
    # blind (legal_actions never offers Verb.OPEN) until the E5 obs/action widening.
    pack_offers: tuple = ()      # tuple[Pack, ...] offered in the shop
    pack_open: tuple = ()        # tuple[PackItem, ...] revealed during OPEN_PACK
    pack_picks: int = 0          # remaining picks during OPEN_PACK
    # E4 vouchers. `vouchers` is the owned set (a tuple of VoucherType ids) and the SINGLE
    # SOURCE OF TRUTH for every persistent per-run modifier (extra hands/discards/slots,
    # interest cap, reroll discount, shop weights) — those are DERIVED from it where used,
    # never stored separately. `voucher_offer` is the shop's single voucher slot (0 = none
    # offered; otherwise a VoucherType id). Both default empty/0 -> byte-identical for
    # directly-constructed states; the agent is blind (legal_actions never offers
    # Verb.BUY_VOUCHER) until the E5 obs/action widening.
    vouchers: tuple = ()         # tuple[int, ...] owned VoucherType ids
    voucher_offer: int = 0       # the shop's voucher slot (0 = none; else a VoucherType id)
    # E5 targeting two-step. A card-targeting Tarot's USE needs hand indices the flat action
    # space can't pre-enumerate, so the agent USEs it in two steps: `(USE, ci)` arms it (sets
    # pending_consumable = ci, applies nothing), then `(USE_TARGET, subset)` applies it to those
    # hand cards and clears pending. -1 = nothing armed (the default, byte-identical: legal_actions
    # only arms targeting Tarots, which need consumables the default game never has). Only ever
    # armed in PLAYING with a non-empty hand, so a valid target always exists.
    pending_consumable: int = -1
