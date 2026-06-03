"""Blind score requirements for antes 1-8 (White-stake base values).

required = ANTE_BASE[ante] * BLIND_MULT[blind_index], floored.
Boss-blind *effects* (debuffs) and higher-stake scaling arrive in later plans;
Tier-0 uses only the score requirement and the score-multiplier bosses implicitly
(via the 2.0x boss requirement).
"""
from __future__ import annotations

from .bosses import BossEffect, boss_req_mult

ANTE_BASE: dict[int, int] = {
    1: 300, 2: 800, 3: 2000, 4: 5000,
    5: 11000, 6: 20000, 7: 35000, 8: 50000,
}
BLIND_MULT: tuple[float, float, float] = (1.0, 1.5, 2.0)  # small, big, boss


def required_score(ante: int, blind_index: int, scale: float = 1.0,
                   boss: BossEffect = BossEffect.NONE) -> int:
    """Score to clear a blind. `scale` (curriculum) shrinks the target; default 1.0 is
    the real game. On the boss blind (index 2) the multiplier comes from the boss
    (Wall 4x / Needle 1x / Violet Vessel 6x; default 2x), so NONE reproduces the pre-boss
    2x exactly. Floored at 1 so a low scale never yields a 0-chip (auto-clear) blind."""
    mult = boss_req_mult(boss) if blind_index == 2 else BLIND_MULT[blind_index]
    return max(1, int(round(ANTE_BASE[ante] * mult * scale)))
