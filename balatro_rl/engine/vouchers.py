"""Vouchers (Phase E4): persistent per-run modifiers bought from the shop.

A voucher is a one-time purchase ($10) that applies a PERSISTENT run modifier — more shop
card slots, an extra hand/discard per round, a bigger hand size, a higher interest cap, etc.
Each voucher is redeemable once; some are gated behind a prerequisite voucher (the upgraded
tier requires its base, e.g. Money Tree requires Seed Money).

SINGLE SOURCE OF TRUTH: the engine stores only the owned-voucher tuple (GameState.vouchers);
EVERY modeled modifier is DERIVED on demand by the helpers here. That keeps the GameState a
plain bag of voucher ids and makes a modifier trivially reconstructable from a replay.

SCOPE (wiki-verified, balatrowiki.org/w/Vouchers): the ~20 vouchers whose effects touch
MODELED mechanics are implemented (slots/hands/discards/hand-size/joker-slot/interest-cap/
reroll-cost/shop-weights). Five vouchers (HONE/GLOW_UP — edition rates; MAGIC_TRICK — playing
cards in shop; HIEROGLYPH — ante reduction; DIRECTORS_CUT — boss reroll) need unbuilt systems,
so they are RESERVED in the enum but NEVER offered and move no modifier (deferred).

ENGINE-FIRST / agent BLIND: legal_actions never emits Verb.BUY_VOUCHER, so a voucher buy is
reachable only via a direct engine.step until the E5 obs/action widening.
"""
from __future__ import annotations

from enum import IntEnum

from .economy import INTEREST_CAP

VOUCHER_COST = 10        # every voucher costs $10 (wiki: base $10; the "+$10 additional" for
#                          an upgraded tier is just its own $10, not on top of the base's).


class VoucherType(IntEnum):
    """The ~24 vouchers (wiki: Major-then-tier order). The last five are DEFERRED — reserved
    in the enum so ids are stable, but never offered (their effects need unbuilt systems)."""
    OVERSTOCK = 1          # +1 shop card slot (-> 3)
    OVERSTOCK_PLUS = 2     # (Overstock) +1 more shop card slot (-> 4)
    CRYSTAL_BALL = 3       # +1 consumable slot
    GRABBER = 4           # +1 hand per round
    NACHO_TONG = 5         # (Grabber) +1 more hand per round
    WASTEFUL = 6          # +1 discard per round
    RECYCLOMANCY = 7       # (Wasteful) +1 more discard per round
    PAINT_BRUSH = 8        # +1 hand size
    PALETTE = 9            # (Paint Brush) +1 more hand size
    ANTIMATTER = 10        # +1 joker slot  (spec: treated as a BASE voucher; real prereq skipped)
    SEED_MONEY = 11        # interest cap -> $10
    MONEY_TREE = 12        # (Seed Money) interest cap -> $20
    REROLL_SURPLUS = 13    # rerolls cost $2 less
    REROLL_GLUT = 14       # (Reroll Surplus) rerolls cost $2 more less (-$4 total)
    TAROT_MERCHANT = 15    # Tarots appear 2x as often in the shop
    TAROT_TYCOON = 16      # (Tarot Merchant) Tarots 4x as often
    PLANET_MERCHANT = 17   # Planets appear 2x as often in the shop
    PLANET_TYCOON = 18     # (Planet Merchant) Planets 4x as often
    BLANK = 19            # does nothing (a no-op voucher)
    # ---- DEFERRED (reserved; never offered, no modeled effect) ----
    HONE = 20             # DEFERRED: edition (Foil/Holo/Poly) rates 2x  (no edition system)
    GLOW_UP = 21          # DEFERRED: (Hone) edition rates 4x
    MAGIC_TRICK = 22       # DEFERRED: playing cards purchasable in the shop
    HIEROGLYPH = 23        # DEFERRED: -1 Ante, -1 hand each round
    DIRECTORS_CUT = 24     # DEFERRED: reroll the Boss Blind once per ante ($10)


# DEFERRED vouchers: never offered (eligible_vouchers excludes them) and move no modifier.
DEFERRED_VOUCHERS: frozenset = frozenset({
    VoucherType.HONE, VoucherType.GLOW_UP, VoucherType.MAGIC_TRICK,
    VoucherType.HIEROGLYPH, VoucherType.DIRECTORS_CUT,
})

# Prerequisite: the voucher that must already be OWNED to unlock this one (None = base).
# Spec: ANTIMATTER is treated as a base voucher (its real "Blank x10" prereq is skipped).
VOUCHER_PREREQ: dict[VoucherType, VoucherType | None] = {
    VoucherType.OVERSTOCK: None,
    VoucherType.OVERSTOCK_PLUS: VoucherType.OVERSTOCK,
    VoucherType.CRYSTAL_BALL: None,
    VoucherType.GRABBER: None,
    VoucherType.NACHO_TONG: VoucherType.GRABBER,
    VoucherType.WASTEFUL: None,
    VoucherType.RECYCLOMANCY: VoucherType.WASTEFUL,
    VoucherType.PAINT_BRUSH: None,
    VoucherType.PALETTE: VoucherType.PAINT_BRUSH,
    VoucherType.ANTIMATTER: None,
    VoucherType.SEED_MONEY: None,
    VoucherType.MONEY_TREE: VoucherType.SEED_MONEY,
    VoucherType.REROLL_SURPLUS: None,
    VoucherType.REROLL_GLUT: VoucherType.REROLL_SURPLUS,
    VoucherType.TAROT_MERCHANT: None,
    VoucherType.TAROT_TYCOON: VoucherType.TAROT_MERCHANT,
    VoucherType.PLANET_MERCHANT: None,
    VoucherType.PLANET_TYCOON: VoucherType.PLANET_MERCHANT,
    VoucherType.BLANK: None,
    VoucherType.HONE: None,
    VoucherType.GLOW_UP: VoucherType.HONE,
    VoucherType.MAGIC_TRICK: None,
    VoucherType.HIEROGLYPH: None,
    VoucherType.DIRECTORS_CUT: None,
}


def _owns(vs, v: VoucherType) -> bool:
    """Membership test that tolerates a tuple of raw ints or VoucherType members."""
    return int(v) in {int(x) for x in vs}


def _count(vs, *members) -> int:
    """How many of `members` are present in the owned-vouchers iterable."""
    owned = {int(x) for x in vs}
    return sum(1 for m in members if int(m) in owned)


# ---- modifier derivation (the single source of truth) ------------------------

def extra_card_slots(vs) -> int:
    """Extra SHOP card slots: Overstock +1, Overstock Plus +1 more."""
    return _count(vs, VoucherType.OVERSTOCK, VoucherType.OVERSTOCK_PLUS)


def extra_consumable_slots(vs) -> int:
    """Extra consumable slots: Crystal Ball +1."""
    return _count(vs, VoucherType.CRYSTAL_BALL)


def extra_hands(vs) -> int:
    """Extra hands per round: Grabber +1, Nacho Tong +1 more."""
    return _count(vs, VoucherType.GRABBER, VoucherType.NACHO_TONG)


def extra_discards(vs) -> int:
    """Extra discards per round: Wasteful +1, Recyclomancy +1 more."""
    return _count(vs, VoucherType.WASTEFUL, VoucherType.RECYCLOMANCY)


def extra_hand_size(vs) -> int:
    """Extra hand size: Paint Brush +1, Palette +1 more."""
    return _count(vs, VoucherType.PAINT_BRUSH, VoucherType.PALETTE)


def extra_joker_slots(vs) -> int:
    """Extra joker slots: Antimatter +1."""
    return _count(vs, VoucherType.ANTIMATTER)


def interest_cap(vs) -> int:
    """The interest cap: Money Tree -> $20, else Seed Money -> $10, else default ($5)."""
    if _owns(vs, VoucherType.MONEY_TREE):
        return 20
    if _owns(vs, VoucherType.SEED_MONEY):
        return 10
    return INTEREST_CAP


def reroll_discount(vs) -> int:
    """Reroll-cost reduction: Reroll Surplus -$2, Reroll Glut -$2 more."""
    return 2 * _count(vs, VoucherType.REROLL_SURPLUS, VoucherType.REROLL_GLUT)


def tarot_weight_mult(vs) -> int:
    """Shop Tarot weight multiplier: Tycoon 4x, else Merchant 2x, else 1x."""
    if _owns(vs, VoucherType.TAROT_TYCOON):
        return 4
    if _owns(vs, VoucherType.TAROT_MERCHANT):
        return 2
    return 1


def planet_weight_mult(vs) -> int:
    """Shop Planet weight multiplier: Tycoon 4x, else Merchant 2x, else 1x."""
    if _owns(vs, VoucherType.PLANET_TYCOON):
        return 4
    if _owns(vs, VoucherType.PLANET_MERCHANT):
        return 2
    return 1


# ---- offer eligibility + roll ------------------------------------------------

def prereq_met(owned, v: VoucherType) -> bool:
    """True iff `v`'s prerequisite voucher (if any) is already owned."""
    req = VOUCHER_PREREQ[VoucherType(v)]
    return req is None or _owns(owned, req)


def eligible_vouchers(owned) -> list[VoucherType]:
    """Vouchers offerable now: not already owned, not deferred, and prereq satisfied.
    Deterministic enum order (so a seeded roll is reproducible)."""
    return [v for v in VoucherType
            if v not in DEFERRED_VOUCHERS
            and not _owns(owned, v)
            and prereq_met(owned, v)]


def roll_voucher(rng, owned):
    """Pick a uniform eligible voucher to offer in the shop (or None if none). Threads rng.
    Returns (VoucherType | None, rng)."""
    pool = eligible_vouchers(owned)
    if not pool:
        return None, rng
    idx, rng = rng.randint(0, len(pool) - 1)
    return pool[idx], rng
