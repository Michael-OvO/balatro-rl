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

from ..engine.state import Phase
from ..envs.balatro_env import BalatroEnv

_MAX_STEPS = 1000   # backstop only; the engine's shop-action cap prevents infinite loops


def _batch(obs: dict):
    return {k: jnp.asarray(v)[None] for k, v in obs.items()}


def _depth(s) -> int:
    """Blinds cleared so far. blind_index only advances on LEAVE_SHOP, so credit the
    just-cleared blind while we're in the cash-out SHOP (or WON), otherwise a cleared
    blind goes uncounted until the agent leaves the shop."""
    bonus = 1 if s.phase in (Phase.SHOP, Phase.WON) else 0
    return (s.ante - 1) * 3 + s.blind_index + bonus


def evaluate(net, params, seeds, reward_name: str = "shaped", *,
             enable_bosses: bool = False, enhance_rate: float = 0.0,
             grant_planets: int = 0, req_scale: float = 1.0) -> dict:
    """Greedy eval. By default the plain real game; pass `enable_bosses`/exposure/`req_scale`
    to evaluate on the training distribution (so train-time metrics aren't measuring a
    distribution the agent never saw)."""
    apply = jax.jit(net.apply)   # compile the forward once; reused across all steps/seeds
    antes, wins, chips, lengths, depths = [], [], [], [], []
    for seed in seeds:
        env = BalatroEnv(reward_name, req_scale=req_scale, enable_bosses=enable_bosses,
                         enhance_rate=enhance_rate, grant_planets=grant_planets)
        obs, mask = env.reset(int(seed))
        run_chips, steps, done, max_depth = 0, 0, False, 0
        while not done and steps < _MAX_STEPS:
            logits, _ = apply(params, _batch(obs), jnp.asarray(mask)[None])
            a = int(jnp.argmax(logits[0]))
            obs, _reward, done, info, mask = env.step(a)
            s = env.state
            max_depth = max(max_depth, _depth(s))   # blinds cleared (credits cash-out shop)
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
