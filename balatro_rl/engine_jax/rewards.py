"""JAX-native shaped reward — mirrors ``balatro_rl.envs.rewards.Shaped`` exactly.

Oracle (rewards.py):
    def _symlog(x):  math.copysign(math.log1p(abs(x)), x)

    def _shaped_potential(s):
        ratio = min(s.round_score / max(s.required, 1), 1.0)
        return ratio + 0.05 * _symlog(s.money) + 0.5 * s.ante

    class Shaped:
        def __call__(self, prev, action_id, nxt, info):
            shaped = self.gamma * _shaped_potential(nxt) - _shaped_potential(prev)
            if info.get("cleared"):
                shaped += 1.0
            if nxt.done and nxt.won:
                shaped += 10.0
            return shaped

``shaped_core`` is ``jit``- and ``vmap``-compatible: no Python branching on traced
values.
"""
from __future__ import annotations

import jax.numpy as jnp

from balatro_rl.engine_jax.state import CoreState
from balatro_rl.engine_jax.obs import _symlog


def _phi(s: CoreState) -> jnp.ndarray:
    """Potential Φ(s) — exactly mirrors ``_shaped_potential`` in rewards.py.

    Φ(s) = min(round_score / max(required, 1), 1.0)
           + 0.05 * symlog(money)
           + 0.5  * ante
    """
    rs  = s.round_score.astype(jnp.float32)
    req = jnp.maximum(s.required.astype(jnp.float32), 1.0)
    ratio = jnp.minimum(rs / req, 1.0)
    money_f = s.money.astype(jnp.float32)
    ante_f  = s.ante.astype(jnp.float32)
    return ratio + jnp.float32(0.05) * _symlog(money_f) + jnp.float32(0.5) * ante_f


def shaped_core(
    prev: CoreState,
    nxt: CoreState,
    cleared,
    won,
    gamma: float = 0.999,
) -> jnp.ndarray:
    """Potential-based shaped reward, mirroring ``Shaped.__call__`` in rewards.py.

    r = gamma * Φ(nxt) - Φ(prev) + 1.0 * cleared + 10.0 * won

    Args:
        prev:    CoreState BEFORE the step.
        nxt:     CoreState AFTER the step.
        cleared: bool/float scalar — True iff the blind was cleared this step.
                 Corresponds to ``info.get("cleared")`` in the Python oracle.
        won:     bool/float scalar — True iff the episode was won this step.
                 Corresponds to ``nxt.done and nxt.won`` in the Python oracle.
        gamma:   discount factor for the potential difference (default 0.999).

    Returns:
        float32 scalar reward.
    """
    phi_prev = _phi(prev)
    phi_nxt  = _phi(nxt)
    r = jnp.float32(gamma) * phi_nxt - phi_prev
    r = r + jnp.float32(1.0) * cleared.astype(jnp.float32)
    r = r + jnp.float32(10.0) * won.astype(jnp.float32)
    return r.astype(jnp.float32)
