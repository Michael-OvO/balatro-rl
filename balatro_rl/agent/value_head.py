"""Symlog two-hot distributional value head (DreamerV3, arXiv 2301.04104).
Absorbs Balatro's 10^2..10^12+ returns: predict a categorical over symlog bins,
decode via symexp. Exact formulas verified in docs/reference/jax-agent.md.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

NBINS = 255
LO, HI = -30.0, 30.0           # symlog(1e12) ~ 27.6; widened so deep-endless returns don't clip
BINS = jnp.linspace(LO, HI, NBINS)


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def two_hot(scalar):
    """Two-hot encode a scalar (or array) into bin weights over symlog space.
    The nearer bin gets MORE weight; the encoding's bin-mean equals symlog(scalar)."""
    t = symlog(jnp.asarray(scalar, dtype=jnp.float32))[..., None]   # (..., 1)
    below = (BINS <= t).sum(-1) - 1
    above = NBINS - (BINS > t).sum(-1)
    below = jnp.clip(below, 0, NBINS - 1)
    above = jnp.clip(above, 0, NBINS - 1)
    equal = below == above
    d_below = jnp.where(equal, 1.0, jnp.abs(BINS[below] - t[..., 0]))
    d_above = jnp.where(equal, 1.0, jnp.abs(BINS[above] - t[..., 0]))
    total = d_below + d_above
    w_below = d_above / total      # cross-assignment: closer bin -> larger weight
    w_above = d_below / total
    return (jax.nn.one_hot(below, NBINS) * w_below[..., None]
            + jax.nn.one_hot(above, NBINS) * w_above[..., None])


def value_loss(value_logits, scalar):
    """Categorical cross-entropy against the (stop-grad) two-hot target."""
    tgt = jax.lax.stop_gradient(two_hot(scalar))
    log_pred = value_logits - jax.nn.logsumexp(value_logits, -1, keepdims=True)
    return -(tgt * log_pred).sum(-1)


def value_decode(value_logits):
    """Predicted distribution -> scalar value (symexp of the bin-weighted mean)."""
    probs = jax.nn.softmax(value_logits, -1)
    return symexp((probs * BINS).sum(-1))
