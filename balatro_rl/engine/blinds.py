"""Blind score requirements for antes 1-8 (White-stake base values).

required = ANTE_BASE[ante] * BLIND_MULT[blind_index], floored.
Boss-blind *effects* (debuffs) and higher-stake scaling arrive in later plans;
Tier-0 uses only the score requirement and the score-multiplier bosses implicitly
(via the 2.0x boss requirement).
"""
from __future__ import annotations

ANTE_BASE: dict[int, int] = {
    1: 300, 2: 800, 3: 2000, 4: 5000,
    5: 11000, 6: 20000, 7: 35000, 8: 50000,
}
BLIND_MULT: tuple[float, float, float] = (1.0, 1.5, 2.0)  # small, big, boss


def required_score(ante: int, blind_index: int, scale: float = 1.0) -> int:
    """Score to clear a blind. `scale` (curriculum) shrinks the target; default 1.0 is
    the real game. Floored at 1 so a low scale never yields a 0-chip (auto-clear) blind."""
    return max(1, int(round(ANTE_BASE[ante] * BLIND_MULT[blind_index] * scale)))
