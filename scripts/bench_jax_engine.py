#!/usr/bin/env python3
"""Throughput micro-benchmark for the JAX core engine (Task 1.10).

Times a fixed-length on-device rollout (a ``lax.scan`` over ``batched_step``) at a
range of batch sizes and reports **env-steps/sec** = num_envs x num_steps / wall_time.
On a CUDA box it also prints ``nvidia-smi`` GPU utilization. This is the Phase-1
throughput metric: it shows the env+policy running entirely on-device with the env
vectorized to thousands of parallel games.

Usage
-----
    python scripts/bench_jax_engine.py                 # default sizes (CPU-safe)
    BENCH_SIZES=1000,10000,50000 python scripts/bench_jax_engine.py
    BENCH_STEPS=200 python scripts/bench_jax_engine.py

Defaults are CPU-memory-safe (1k, 10k). 50k is heavy on CPU; pass it explicitly via
BENCH_SIZES (it is the intended GPU batch size). On a GPU box, run with
e.g. ``BENCH_SIZES=1000,10000,50000``.

The policy is a cheap on-device "first legal action" (argmax of the legal mask), so the
benchmark measures engine+obs+mask throughput, not policy quality.
"""
from __future__ import annotations

import os
import subprocess
import time

import jax
import jax.numpy as jnp

from balatro_rl.engine_jax.curriculum import build_required_table
from balatro_rl.engine_jax.obs import legal_mask_core
from balatro_rl.engine_jax.step import batched_reset, batched_step


def _sizes() -> list[int]:
    raw = os.environ.get("BENCH_SIZES")
    if raw:
        return [int(x) for x in raw.split(",") if x.strip()]
    return [1000, 10000]  # CPU-safe default; add 50000 via BENCH_SIZES on a GPU box


def _num_steps() -> int:
    return int(os.environ.get("BENCH_STEPS", "200"))


def _first_legal_action(state) -> jnp.ndarray:
    """A cheap on-device policy: the lowest-index legal action per env (argmax of the
    legal mask). Keeps the benchmark engine-bound, not policy-bound."""
    masks = jax.vmap(legal_mask_core)(state)          # [N, 708] bool
    return jnp.argmax(masks.astype(jnp.int32), axis=1)  # [N] int32


def _make_rollout(num_steps: int):
    """Build a jitted ``lax.scan`` rollout of length ``num_steps`` over batched_step."""
    def rollout(state):
        def body(carry, _):
            st = carry
            actions = _first_legal_action(st)
            st2, reward, done, _sig = batched_step(st, actions)
            return st2, reward
        final_state, rewards = jax.lax.scan(body, state, None, length=num_steps)
        return final_state, rewards
    return jax.jit(rollout)


def _gpu_util() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def main() -> None:
    sizes = _sizes()
    num_steps = _num_steps()
    req_table = build_required_table(0.5)  # mid curriculum; scale doesn't affect throughput

    print(f"JAX devices: {jax.devices()}")
    print(f"steps/rollout = {num_steps}; batch sizes = {sizes}\n")
    print(f"{'num_envs':>10} {'wall_s':>9} {'env-steps/s':>16}")
    print("-" * 40)

    for n in sizes:
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        state = batched_reset(keys, req_table)
        jax.block_until_ready(state)
        rollout = _make_rollout(num_steps)

        # Warmup (compile) — excluded from timing.
        fs, _ = rollout(state)
        jax.block_until_ready(fs)

        t0 = time.perf_counter()
        fs, rewards = rollout(state)
        jax.block_until_ready((fs, rewards))
        dt = time.perf_counter() - t0

        eps = (n * num_steps) / dt
        print(f"{n:>10} {dt:>9.3f} {eps:>16,.0f}")

    util = _gpu_util()
    if util is not None:
        print(f"\nnvidia-smi (util%, mem.used, mem.total): {util}")
    else:
        print("\n(no CUDA device detected — CPU run; GPU utilization N/A)")


if __name__ == "__main__":
    main()
