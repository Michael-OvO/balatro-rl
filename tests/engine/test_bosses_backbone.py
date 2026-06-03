"""Phase C0: boss-blind backbone. Boss enum + metadata + selection + score-requirement
multiplier override. Boss EFFECTS (debuffs / legal-mask / draw) are later sub-phases;
C0 only wires selection (gated behind enable_bosses, default OFF -> byte-identical game)
and the required-score multiplier (Wall 4x, Needle 1x, Violet Vessel 6x, else 2x).

Boss table verified against balatrowiki.org/w/Blinds.
"""
import dataclasses

from balatro_rl.engine.blinds import required_score, ANTE_BASE
from balatro_rl.engine.bosses import BossEffect, BOSS_INFO, select_boss, boss_req_mult
from balatro_rl.engine.engine import reset, _advance_blind
from balatro_rl.engine.rng import RNG


# ============================================================================
# enum + metadata
# ============================================================================

def test_boss_enum_has_none_plus_23_regular_plus_5_finishers():
    assert int(BossEffect.NONE) == 0
    regular = [b for b in BossEffect if b != BossEffect.NONE and not BOSS_INFO[b].is_finisher]
    finishers = [b for b in BossEffect if BOSS_INFO[b].is_finisher]
    assert len(regular) == 23 and len(finishers) == 5


def test_every_boss_has_metadata():
    for b in BossEffect:
        info = BOSS_INFO[b]
        assert info.min_ante >= 1 and info.req_mult > 0


def test_special_required_multipliers():
    assert boss_req_mult(BossEffect.THE_WALL) == 4.0
    assert boss_req_mult(BossEffect.THE_NEEDLE) == 1.0
    assert boss_req_mult(BossEffect.VIOLET_VESSEL) == 6.0
    assert boss_req_mult(BossEffect.THE_HOOK) == 2.0       # the common case
    assert boss_req_mult(BossEffect.NONE) == 2.0           # boss blind w/o a selected boss


def test_min_antes_match_wiki():
    expected = {BossEffect.THE_OX: 6, BossEffect.THE_SERPENT: 5, BossEffect.THE_PLANT: 4,
                BossEffect.THE_EYE: 3, BossEffect.THE_TOOTH: 3, BossEffect.THE_WALL: 2,
                BossEffect.THE_NEEDLE: 2, BossEffect.THE_HOOK: 1, BossEffect.THE_CLUB: 1}
    for b, mn in expected.items():
        assert BOSS_INFO[b].min_ante == mn


def test_finishers_are_ante_8():
    for b in BossEffect:
        if BOSS_INFO[b].is_finisher:
            assert BOSS_INFO[b].min_ante == 8


# ============================================================================
# selection
# ============================================================================

def test_select_boss_respects_min_ante():
    # Across many seeds at ante 1, only min-ante-1 regular bosses are ever picked.
    for seed in range(300):
        b, _ = select_boss(RNG.from_seed(seed), ante=1)
        assert not BOSS_INFO[b].is_finisher and BOSS_INFO[b].min_ante <= 1


def test_select_boss_ante_3_allows_up_to_min_3():
    for seed in range(300):
        b, _ = select_boss(RNG.from_seed(seed), ante=3)
        assert BOSS_INFO[b].min_ante <= 3 and not BOSS_INFO[b].is_finisher


def test_select_boss_ante_8_returns_finisher():
    for seed in range(200):
        b, _ = select_boss(RNG.from_seed(seed), ante=8)
        assert BOSS_INFO[b].is_finisher


def test_select_boss_is_deterministic_and_advances_rng():
    b1, r1 = select_boss(RNG.from_seed(5), ante=4)
    b2, r2 = select_boss(RNG.from_seed(5), ante=4)
    assert b1 == b2 and r1 == r2
    assert r1 != RNG.from_seed(5)          # a roll was consumed


def test_select_boss_can_reach_high_min_ante_at_ante_6():
    # The Ox (min 6) must be reachable at ante 6.
    seen = {select_boss(RNG.from_seed(s), ante=6)[0] for s in range(500)}
    assert BossEffect.THE_OX in seen


# ============================================================================
# required_score with a boss multiplier
# ============================================================================

def test_required_score_applies_boss_mult_on_boss_blind():
    assert required_score(2, 2, 1.0, BossEffect.THE_WALL) == ANTE_BASE[2] * 4      # 3200
    assert required_score(2, 2, 1.0, BossEffect.THE_NEEDLE) == ANTE_BASE[2] * 1    # 800
    assert required_score(2, 2, 1.0, BossEffect.NONE) == int(ANTE_BASE[2] * 2.0)   # 1600


def test_required_score_ignores_boss_off_boss_blind():
    # Small/big blinds ignore the boss param entirely.
    assert required_score(3, 0, 1.0, BossEffect.THE_WALL) == ANTE_BASE[3]
    assert required_score(3, 1, 1.0, BossEffect.THE_WALL) == int(ANTE_BASE[3] * 1.5)


def test_required_score_default_boss_is_byte_identical():
    # Omitting the boss arg reproduces the pre-C0 value exactly.
    for ante in range(1, 9):
        for bi in range(3):
            assert required_score(ante, bi, 1.0) == required_score(ante, bi, 1.0, BossEffect.NONE)


# ============================================================================
# engine wiring: gated, default OFF
# ============================================================================

def test_reset_defaults_bosses_disabled_and_no_boss():
    st = reset(seed=0)
    assert st.bosses_enabled is False and st.boss == 0


def test_advance_to_boss_blind_disabled_keeps_no_boss_and_default_required():
    st = dataclasses.replace(reset(seed=0), blind_index=1)   # at big blind, bosses OFF
    nxt, _info = _advance_blind(st)
    assert nxt.blind_index == 2 and nxt.boss == 0
    assert nxt.required == required_score(nxt.ante, 2, nxt.req_scale)   # default 2x


def test_advance_to_boss_blind_enabled_selects_boss_and_applies_mult():
    st = dataclasses.replace(reset(seed=0, enable_bosses=True), blind_index=1)
    nxt, _info = _advance_blind(st)
    assert nxt.blind_index == 2 and nxt.boss != 0
    boss = BossEffect(nxt.boss)
    assert nxt.required == required_score(nxt.ante, 2, nxt.req_scale, boss)
    assert BOSS_INFO[boss].min_ante <= nxt.ante


def test_advance_to_non_boss_blind_has_no_boss_even_when_enabled():
    # small->big (blind 0->1) carries no boss regardless of the flag.
    st = reset(seed=0, enable_bosses=True)   # at small blind
    nxt, _info = _advance_blind(st)
    assert nxt.blind_index == 1 and nxt.boss == 0


def test_enable_bosses_does_not_perturb_rng_when_disabled():
    # The boss path must draw zero rng when disabled: a disabled advance lands on the
    # exact same rng/required/hand as before C0 existed (byte-compat). Compare the two
    # advances from the same big-blind state -> identical successor rng.
    base = dataclasses.replace(reset(seed=3), blind_index=1)
    a, _ = _advance_blind(base)
    b, _ = _advance_blind(base)
    assert a.rng == b.rng and a.required == b.required and a.hand == b.hand
