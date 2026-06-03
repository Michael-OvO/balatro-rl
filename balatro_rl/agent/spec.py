"""Obs spec helpers shared by the agent: dtypes per obs key + a dummy batch for
network init/tests. Single source of truth for shapes is envs.obs.OBS_SHAPES.
"""
from __future__ import annotations

import numpy as np

from ..envs.obs import OBS_SHAPES

# Integer-id keys are embedded (nn.Embed); everything else is float32.
_INT_KEYS = ("joker_types", "shop_types", "consum_types")
OBS_DTYPES = {k: (np.int32 if k in _INT_KEYS else np.float32) for k in OBS_SHAPES}


def dummy_obs(batch: int = 1) -> dict:
    """A zero-filled batched obs dict with the exact shapes/dtypes the env emits."""
    return {k: np.zeros((batch,) + shape, dtype=OBS_DTYPES[k]) for k, shape in OBS_SHAPES.items()}
