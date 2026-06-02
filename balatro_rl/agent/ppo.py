"""Maskable-PPO core: hand-rolled masked categorical (Gumbel-max sample,
log-softmax log-prob, masked entropy), GAE (reverse scan), and the clipped
surrogate + value-CE + masked-entropy loss. The mask is RE-APPLIED in the loss
via the stored masked logits (the #1 maskable-PPO correctness bug if skipped).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .value_head import value_loss


def sample_action(masked_logits, key):
    u = jax.random.uniform(key, masked_logits.shape)
    return jnp.argmax(masked_logits - jnp.log(-jnp.log(u)), axis=-1)


def log_prob(masked_logits, action):
    logp = jax.nn.log_softmax(masked_logits, -1)
    return jnp.take_along_axis(logp, action[..., None], axis=-1)[..., 0]


def masked_entropy(masked_logits, mask):
    logp = jax.nn.log_softmax(masked_logits, -1)
    p = jax.nn.softmax(masked_logits, -1)
    return -jnp.where(mask, p * logp, 0.0).sum(-1)


def gae(rewards, values, dones, last_value, gamma=0.999, lam=0.95):
    """rewards/values/dones: [T,N]; last_value: [N]. Returns (advantages, targets)."""
    def step(carry, x):
        adv, next_v = carry
        r, v, d = x
        delta = r + gamma * next_v * (1.0 - d) - v
        adv = delta + gamma * lam * (1.0 - d) * adv
        return (adv, v), adv
    _, advantages = jax.lax.scan(step, (jnp.zeros_like(last_value), last_value),
                                 (rewards, values, dones), reverse=True)
    return advantages, advantages + values


def ppo_loss(params, apply_fn, mb, clip=0.2, vf_coef=0.5, ent_coef=0.01):
    logits, value_logits = apply_fn(params, mb["obs"], mb["masks"])   # re-applies the mask
    logp = log_prob(logits, mb["actions"])
    ratio = jnp.exp(logp - mb["old_logp"])
    adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv).mean()
    vl = value_loss(value_logits, mb["targets"]).mean()
    ent = masked_entropy(logits, mb["masks"]).mean()
    total = pg + vf_coef * vl - ent_coef * ent
    return total, (pg, vl, ent)
