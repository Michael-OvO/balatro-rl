"""Greedy-policy evaluation: run the argmax policy to completion over fixed seeds
and report how WELL it plays (blinds cleared, ante reached, win-rate). Deterministic.

`blinds_cleared = (ante-1)*3 + blind_index` is the headline depth metric: it is
loop-immune (dithering in the shop never advances a blind) and finer than `max_ante`
(it sees a cleared *small* blind, where ante stays 1). The shop-action cap in the
engine now prevents the infinite-reorder loop, so `_MAX_STEPS` is only a backstop.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..envs.balatro_env import BalatroEnv

_MAX_STEPS = 1000   # backstop only; the engine's shop-action cap prevents infinite loops


def _batch(obs: dict):
    return {k: jnp.asarray(v)[None] for k, v in obs.items()}


def evaluate(net, params, seeds, reward_name: str = "shaped") -> dict:
    apply = jax.jit(net.apply)   # compile the forward once; reused across all steps/seeds
    antes, wins, chips, lengths, depths = [], [], [], [], []
    for seed in seeds:
        env = BalatroEnv(reward_name)
        obs, mask = env.reset(int(seed))
        run_chips, steps, done, max_depth = 0, 0, False, 0
        while not done and steps < _MAX_STEPS:
            logits, _ = apply(params, _batch(obs), jnp.asarray(mask)[None])
            a = int(jnp.argmax(logits[0]))
            obs, _reward, done, info, mask = env.step(a)
            s = env.state
            max_depth = max(max_depth, (s.ante - 1) * 3 + s.blind_index)   # blinds cleared
            if info.get("verb") == "play":
                run_chips += int(info.get("score", 0))
            steps += 1
        antes.append(env.state.ante)
        wins.append(1.0 if env.state.won else 0.0)
        chips.append(run_chips)
        lengths.append(steps)
        depths.append(max_depth)
    return {
        "eval/mean_ante": float(np.mean(antes)),
        "eval/max_ante": float(np.max(antes)),
        "eval/win_rate": float(np.mean(wins)),
        "eval/mean_run_chips": float(np.mean(chips)),
        "eval/mean_ep_len": float(np.mean(lengths)),
        "eval/mean_blinds_cleared": float(np.mean(depths)),
        "eval/max_blinds_cleared": float(np.max(depths)),
        "eval/blind1_clear_rate": float(np.mean([d >= 1 for d in depths])),
    }
