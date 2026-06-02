"""Greedy-policy evaluation: run the argmax policy to completion over fixed seeds
and report how WELL it plays (ante reached, win-rate, run chips). Deterministic.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..envs.balatro_env import BalatroEnv

_MAX_STEPS = 3000


def _batch(obs: dict):
    return {k: jnp.asarray(v)[None] for k, v in obs.items()}


def evaluate(net, params, seeds, reward_name: str = "shaped") -> dict:
    apply = jax.jit(net.apply)   # compile the forward once; reused across all steps/seeds
    antes, wins, chips, lengths = [], [], [], []
    for seed in seeds:
        env = BalatroEnv(reward_name)
        obs, mask = env.reset(int(seed))
        run_chips, steps, done = 0, 0, False
        while not done and steps < _MAX_STEPS:
            logits, _ = apply(params, _batch(obs), jnp.asarray(mask)[None])
            a = int(jnp.argmax(logits[0]))
            obs, _reward, done, info, mask = env.step(a)
            if info.get("verb") == "play":
                run_chips += int(info.get("score", 0))
            steps += 1
        antes.append(env.state.ante)
        wins.append(1.0 if env.state.won else 0.0)
        chips.append(run_chips)
        lengths.append(steps)
    return {
        "eval/mean_ante": float(np.mean(antes)),
        "eval/max_ante": float(np.max(antes)),
        "eval/win_rate": float(np.mean(wins)),
        "eval/mean_run_chips": float(np.mean(chips)),
        "eval/mean_ep_len": float(np.mean(lengths)),
    }
