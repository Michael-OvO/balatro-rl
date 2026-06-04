"""Phase E4 — vouchers module (pure modifier derivation + offer eligibility/roll).

Vouchers are the SINGLE SOURCE OF TRUTH: every per-run modifier (extra hands/discards/
slots, interest cap, reroll discount, shop weight mults) is DERIVED from the owned-voucher
tuple by a helper here. These tests pin the wiki-verified values and the prereq gating.
"""
from balatro_rl.engine.vouchers import (
    VOUCHER_COST, VOUCHER_PREREQ, VoucherType,
    eligible_vouchers, extra_card_slots, extra_consumable_slots, extra_discards,
    extra_hand_size, extra_hands, extra_joker_slots, interest_cap, planet_weight_mult,
    reroll_discount, roll_voucher, tarot_weight_mult,
)
from balatro_rl.engine.economy import INTEREST_CAP
from balatro_rl.engine.rng import RNG


# ---- enum + constants --------------------------------------------------------

def test_voucher_cost_is_ten():
    assert VOUCHER_COST == 10


def test_all_expected_vouchers_present():
    names = {v.name for v in VoucherType}
    for n in ("OVERSTOCK", "OVERSTOCK_PLUS", "CRYSTAL_BALL", "GRABBER", "NACHO_TONG",
              "WASTEFUL", "RECYCLOMANCY", "PAINT_BRUSH", "PALETTE", "ANTIMATTER",
              "SEED_MONEY", "MONEY_TREE", "REROLL_SURPLUS", "REROLL_GLUT",
              "TAROT_MERCHANT", "TAROT_TYCOON", "PLANET_MERCHANT", "PLANET_TYCOON",
              "BLANK", "HONE", "GLOW_UP", "MAGIC_TRICK", "HIEROGLYPH", "DIRECTORS_CUT"):
        assert n in names, f"missing VoucherType.{n}"


def test_prereq_chains():
    assert VOUCHER_PREREQ[VoucherType.OVERSTOCK] is None
    assert VOUCHER_PREREQ[VoucherType.OVERSTOCK_PLUS] == VoucherType.OVERSTOCK
    assert VOUCHER_PREREQ[VoucherType.NACHO_TONG] == VoucherType.GRABBER
    assert VOUCHER_PREREQ[VoucherType.RECYCLOMANCY] == VoucherType.WASTEFUL
    assert VOUCHER_PREREQ[VoucherType.PALETTE] == VoucherType.PAINT_BRUSH
    assert VOUCHER_PREREQ[VoucherType.MONEY_TREE] == VoucherType.SEED_MONEY
    assert VOUCHER_PREREQ[VoucherType.REROLL_GLUT] == VoucherType.REROLL_SURPLUS
    assert VOUCHER_PREREQ[VoucherType.TAROT_TYCOON] == VoucherType.TAROT_MERCHANT
    assert VOUCHER_PREREQ[VoucherType.PLANET_TYCOON] == VoucherType.PLANET_MERCHANT
    # Spec: Antimatter is treated as a BASE voucher (skip the real Blank x10 prereq).
    assert VOUCHER_PREREQ[VoucherType.ANTIMATTER] is None


# ---- modifier derivation (single source of truth) ----------------------------

def test_extra_card_slots():
    assert extra_card_slots(()) == 0
    assert extra_card_slots((VoucherType.OVERSTOCK,)) == 1
    assert extra_card_slots((VoucherType.OVERSTOCK, VoucherType.OVERSTOCK_PLUS)) == 2
    # Plus without base never happens in-game, but the derivation only counts what's owned.
    assert extra_card_slots((VoucherType.OVERSTOCK_PLUS,)) == 1


def test_extra_consumable_slots():
    assert extra_consumable_slots(()) == 0
    assert extra_consumable_slots((VoucherType.CRYSTAL_BALL,)) == 1


def test_extra_hands():
    assert extra_hands(()) == 0
    assert extra_hands((VoucherType.GRABBER,)) == 1
    assert extra_hands((VoucherType.GRABBER, VoucherType.NACHO_TONG)) == 2


def test_extra_discards():
    assert extra_discards(()) == 0
    assert extra_discards((VoucherType.WASTEFUL,)) == 1
    assert extra_discards((VoucherType.WASTEFUL, VoucherType.RECYCLOMANCY)) == 2


def test_extra_hand_size():
    assert extra_hand_size(()) == 0
    assert extra_hand_size((VoucherType.PAINT_BRUSH,)) == 1
    assert extra_hand_size((VoucherType.PAINT_BRUSH, VoucherType.PALETTE)) == 2


def test_extra_joker_slots():
    assert extra_joker_slots(()) == 0
    assert extra_joker_slots((VoucherType.ANTIMATTER,)) == 1


def test_interest_cap():
    assert interest_cap(()) == INTEREST_CAP            # default 5
    assert interest_cap((VoucherType.SEED_MONEY,)) == 10
    assert interest_cap((VoucherType.SEED_MONEY, VoucherType.MONEY_TREE)) == 20
    # Money Tree wins even if listed alone (only count what's owned).
    assert interest_cap((VoucherType.MONEY_TREE,)) == 20


def test_reroll_discount():
    assert reroll_discount(()) == 0
    assert reroll_discount((VoucherType.REROLL_SURPLUS,)) == 2
    assert reroll_discount((VoucherType.REROLL_SURPLUS, VoucherType.REROLL_GLUT)) == 4


def test_tarot_weight_mult():
    assert tarot_weight_mult(()) == 1
    assert tarot_weight_mult((VoucherType.TAROT_MERCHANT,)) == 2
    assert tarot_weight_mult((VoucherType.TAROT_MERCHANT, VoucherType.TAROT_TYCOON)) == 4
    assert tarot_weight_mult((VoucherType.TAROT_TYCOON,)) == 4


def test_planet_weight_mult():
    assert planet_weight_mult(()) == 1
    assert planet_weight_mult((VoucherType.PLANET_MERCHANT,)) == 2
    assert planet_weight_mult((VoucherType.PLANET_MERCHANT, VoucherType.PLANET_TYCOON)) == 4


def test_deferred_vouchers_have_no_modeled_effect():
    """The 4 deferred vouchers (Hone/Glow Up/Magic Trick/Hieroglyph/Director's Cut) touch
    only unbuilt systems, so they move no modeled modifier."""
    for v in (VoucherType.HONE, VoucherType.GLOW_UP, VoucherType.MAGIC_TRICK,
              VoucherType.HIEROGLYPH, VoucherType.DIRECTORS_CUT, VoucherType.BLANK):
        owned = (v,)
        assert extra_card_slots(owned) == 0
        assert extra_consumable_slots(owned) == 0
        assert extra_hands(owned) == 0
        assert extra_discards(owned) == 0
        assert extra_hand_size(owned) == 0
        assert extra_joker_slots(owned) == 0
        assert interest_cap(owned) == INTEREST_CAP
        assert reroll_discount(owned) == 0
        assert tarot_weight_mult(owned) == 1
        assert planet_weight_mult(owned) == 1


# ---- eligibility + roll ------------------------------------------------------

def test_eligible_vouchers_excludes_owned_and_gates_prereq():
    elig = set(eligible_vouchers(()))
    # An upgraded voucher is NOT eligible before its base is owned.
    assert VoucherType.MONEY_TREE not in elig
    assert VoucherType.OVERSTOCK_PLUS not in elig
    # Base vouchers are eligible from the start.
    assert VoucherType.SEED_MONEY in elig
    assert VoucherType.OVERSTOCK in elig
    # Deferred vouchers are NEVER offered.
    for v in (VoucherType.HONE, VoucherType.GLOW_UP, VoucherType.MAGIC_TRICK,
              VoucherType.HIEROGLYPH, VoucherType.DIRECTORS_CUT):
        assert v not in elig


def test_eligible_unlocks_after_prereq_owned():
    elig = set(eligible_vouchers((VoucherType.SEED_MONEY,)))
    assert VoucherType.MONEY_TREE in elig      # prereq satisfied -> now offerable
    assert VoucherType.SEED_MONEY not in elig  # already owned -> not re-offered


def test_roll_voucher_returns_eligible_or_none():
    rng = RNG.from_seed(7)
    seen = set()
    for _ in range(50):
        v, rng = roll_voucher(rng, owned=())
        if v is not None:
            assert isinstance(v, VoucherType)
            assert VOUCHER_PREREQ[v] is None      # only base vouchers eligible when owned=()
            seen.add(v)
    assert seen  # at least some rolls produced a voucher


def test_roll_voucher_none_when_nothing_eligible():
    """Owning every offerable voucher -> nothing left to roll -> None."""
    all_offerable = tuple(eligible_vouchers(()))
    # Build the full owned set by repeatedly unlocking until eligibility is empty.
    owned = set()
    changed = True
    while changed:
        changed = False
        for v in eligible_vouchers(tuple(owned)):
            owned.add(v)
            changed = True
    v, _ = roll_voucher(RNG.from_seed(3), owned=tuple(owned))
    assert v is None
    assert all_offerable  # sanity: there were offerable vouchers to begin with
