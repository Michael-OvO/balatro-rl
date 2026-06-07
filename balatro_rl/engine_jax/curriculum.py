"""Curriculum helpers for the JAX engine.

``build_required_table(scale)`` constructs the ``int32[9, 3]`` required-score
table consumed by ``reset_jax`` and ``step`` when advancing blind boundaries.
Keeping it in a non-test module lets production code (``JaxVectorEnv``,
``set_req_scale``) call it without importing from the test tree.
"""
from __future__ import annotations

import numpy as np


def build_required_table(scale: float) -> np.ndarray:
    """Return the int32[9, 3] required-score table for the JAX engine.

    Entry ``[ante, blind_index]`` is ``engine.required_score(ante, blind, scale,
    BossEffect.NONE)`` for ante 1..8 and blind 0..2.  Row 0 is unused (antes
    are 1-based) and left zero, matching the placeholder shape in ``CoreState``.

    Bosses are disabled (``BossEffect.NONE``), so the boss blind (index 2) uses
    the default 2× multiplier — the same path taken with ``enable_bosses=False``.

    Args:
        scale: Curriculum scale factor (e.g. 0.2 for easy warmup, 1.0 for the
               real game).  Passed directly to ``required_score``.

    Returns:
        int32 numpy array of shape (9, 3).
    """
    from balatro_rl.engine.blinds import required_score
    from balatro_rl.engine.bosses import BossEffect

    NONE = BossEffect.NONE
    table = np.zeros((9, 3), dtype=np.int32)
    for ante in range(1, 9):
        for blind in range(3):
            table[ante, blind] = required_score(ante, blind, scale, NONE)
    return table
