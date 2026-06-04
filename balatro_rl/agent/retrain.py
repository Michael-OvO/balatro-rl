"""E5 retrain: train the card-aware agent on the FULL acquisition game — the agent now sees and
uses every shop consumable, booster pack, voucher, and card-targeting Tarot — with boss blinds
under the E5 boss-rate curriculum (bosses fade in with the score bar).

    python -m balatro_rl.agent.retrain            # full run (GPU-sized; ~2000 updates)
    python -m balatro_rl.agent.retrain --smoke    # tiny end-to-end check (a few updates, small net)

Config is env-overridable so the same entrypoint sizes up on a RunPod GPU and down for a local
smoke (see docs/RUNPOD.md):
    BALATRO_DMODEL   network width      (default 256, smoke 64)
    BALATRO_NUM_ENVS parallel envs      (default 64,  smoke 8)
    BALATRO_UPDATES  PPO updates        (default 2000, smoke 5)
    BALATRO_EPISODE_DIR  output dir     (default /tmp/sweep_out)

89% of the wall-clock is the PPO backward pass (GPU-helpable); the Python env stepping is ~7%
and stays on CPU. On a single modern GPU expect roughly 5-8x over this Mac's CPU.
"""
from __future__ import annotations

import os
import sys

# Cap host threads BEFORE jax/XLA import -> avoids the oversubscription that thrashed the machine.
os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")

import time

import jax

from .train import train, TrainConfig
from .metrics_logger import ConsoleLogger, MultiLogger
from .networks import ActorCritic
from ..envs.actions import NUM_ACTIONS
from ..viz.replay_data import record_agent_episode, save_episode

SMOKE = "--smoke" in sys.argv
OUT = os.environ.get("BALATRO_EPISODE_DIR", "/tmp/sweep_out")
D_MODEL = int(os.environ.get("BALATRO_DMODEL", 64 if SMOKE else 256))
NUM_ENVS = int(os.environ.get("BALATRO_NUM_ENVS", 8 if SMOKE else 64))
NUM_UPDATES = int(os.environ.get("BALATRO_UPDATES", 5 if SMOKE else 2000))
NUM_MB = 2 if SMOKE else 8
# Exposure OFF: the enhanced-card "crutch" made an earlier retrain rely on mods it doesn't have
# at deploy. The agent must learn the real, plain-deck game; the curriculum bootstraps it instead.
ENHANCE_RATE = 0.0
GRANT_PLANETS = 0


def _ent(u: int) -> float:
    """Entropy coefficient: decay 0.04 -> 0.01 over the run (encourages early exploration)."""
    return max(0.01, 0.04 * (1.0 - u / (NUM_UPDATES * 0.7)))


def build_config() -> TrainConfig:
    return TrainConfig(
        num_updates=NUM_UPDATES, num_envs=NUM_ENVS, num_steps=64, d_model=D_MODEL,
        num_minibatches=NUM_MB, update_epochs=4, lr=3e-4,
        reward_name="shaped", seed=0, ent_coef=_ent,
        curr_floor=0.2, ramp_clear_rate=0.7, ramp_step=0.05, ramp_window=20,
        enable_bosses=True, boss_curriculum=True,          # E5: bosses fade in with the score bar
        enhance_rate=ENHANCE_RATE, grant_planets=GRANT_PLANETS,
        eval_interval=(2 if SMOKE else 50), eval_seeds=(0, 1, 2, 3),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = build_config()
    devices = jax.devices()
    on_gpu = any(d.platform == "gpu" for d in devices)
    print(f"[retrain] JAX devices: {devices} ({'GPU' if on_gpu else 'CPU'})", flush=True)
    if not on_gpu and not SMOKE:
        print("[retrain] WARNING: no GPU detected — a full run will be slow on CPU. "
              "Use --smoke for a local check, or run on a CUDA host (docs/RUNPOD.md).", flush=True)

    trk = None
    try:
        from .metrics_logger import TrackioLogger
        trk = TrackioLogger(project="balatro-retrain-e5", name="e5-smoke" if SMOKE else "e5-full",
                            config={"d_model": D_MODEL, "updates": cfg.num_updates,
                                    "num_envs": NUM_ENVS, "enable_bosses": True,
                                    "boss_curriculum": True, "curr_floor": cfg.curr_floor,
                                    "device": "gpu" if on_gpu else "cpu"})
    except Exception as e:   # trackio optional; console logging still streams progress
        print(f"[retrain] trackio unavailable ({e}); console-only", flush=True)
    logger = MultiLogger(ConsoleLogger(every=5), trk)

    t0 = time.time()
    print(f"[retrain] start: {cfg.num_updates} updates, d_model={D_MODEL}, num_envs={NUM_ENVS}, "
          f"full acquisition game + boss curriculum (floor {cfg.curr_floor})", flush=True)
    result = train(cfg, logger=logger)
    print(f"[retrain] training done in {(time.time() - t0) / 60:.1f} min", flush=True)

    import flax.serialization
    ppath = os.path.join(OUT, "retrain_e5_params.msgpack")
    with open(ppath, "wb") as f:
        f.write(flax.serialization.to_bytes(result.params))
    print(f"[retrain] saved params -> {ppath}", flush=True)

    # Record replays on the REAL deploy game (full bosses via the env's default boss_rate=1.0).
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=D_MODEL)
    for s in (4, 7):
        steps = record_agent_episode(net, result.params, seed=s, reward_name="shaped",
                                     enable_bosses=True, enhance_rate=ENHANCE_RATE,
                                     grant_planets=GRANT_PLANETS)
        path = os.path.join(OUT, f"retrain_e5_seed{s}.episode.json")
        save_episode(steps, path)
        print(f"[retrain] recorded {len(steps)} steps -> {path}", flush=True)
    print("[retrain] DONE", flush=True)


if __name__ == "__main__":
    main()
