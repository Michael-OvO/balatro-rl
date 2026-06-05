"""E5 retrain: train the card-aware agent on the FULL acquisition game — the agent now sees and
uses every shop consumable, booster pack, voucher, and card-targeting Tarot — with boss blinds
under the E5 boss-rate curriculum (bosses fade in with the score bar).

    python -m balatro_rl.agent.retrain            # full run (GPU-sized; ~2000 updates)
    python -m balatro_rl.agent.retrain --smoke    # tiny end-to-end check (a few updates, small net)

Config is env-overridable so the same entrypoint sizes up on a RunPod GPU and down for a local
smoke (see docs/RUNPOD.md):
    BALATRO_DMODEL          network width   (default 256, smoke 64)
    BALATRO_NUM_ENVS        parallel envs   (default 64,  smoke 8)
    BALATRO_UPDATES         PPO updates     (default 2000, smoke 5)
    BALATRO_EPISODE_DIR     output dir      (default /tmp/sweep_out)
    BALATRO_CHECKPOINT_EVERY save params every N updates (default 50; 0 = off) — so a multi-hour
                            run survives a late crash. Latest -> <OUT>/retrain_e5_ckpt.msgpack.
    BALATRO_RESUME          path to a .msgpack to WARM-START from (resume a crashed run)

The bottleneck is the CPU env-stepping (Python), not the GPU — a 4.7M-param backward finishes in
ms on any modern GPU, so the GPU tier barely matters and throughput scales with num_envs + vCPUs.
To fill a longer time budget, raise BALATRO_UPDATES (more training), not the GPU size.
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
CHECKPOINT_EVERY = int(os.environ.get("BALATRO_CHECKPOINT_EVERY", 2 if SMOKE else 50))
RESUME = os.environ.get("BALATRO_RESUME")    # path to a .msgpack to warm-start from (resume)


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

    # Live dashboard. Set BALATRO_TRACKIO_SPACE=<user/space> to host a PUBLIC (or private)
    # HuggingFace-Spaces dashboard you can open from any browser while the pod trains — the
    # remote-training answer (needs an HF token on the box: `huggingface-cli login` or HF_TOKEN).
    # Unset -> a local-only Trackio db (view via `trackio show` / SSH-forward). GPU traces auto-on
    # when a GPU is present, so the dashboard shows whether the GPU is saturated.
    space = os.environ.get("BALATRO_TRACKIO_SPACE")
    private = os.environ.get("BALATRO_TRACKIO_PRIVATE", "0") == "1"
    trk = None
    try:
        from .metrics_logger import TrackioLogger
        trk = TrackioLogger(project="balatro-retrain-e5", name="e5-smoke" if SMOKE else "e5-full",
                            config={"d_model": D_MODEL, "updates": cfg.num_updates,
                                    "num_envs": NUM_ENVS, "enable_bosses": True,
                                    "boss_curriculum": True, "curr_floor": cfg.curr_floor,
                                    "device": "gpu" if on_gpu else "cpu"},
                            space_id=space, private=private, auto_log_gpu=on_gpu)
        if space:
            print(f"[retrain] live dashboard -> HF Space '{space}' "
                  f"({'private' if private else 'public'}); opens in a browser shortly", flush=True)
    except Exception as e:   # trackio optional; console logging still streams progress
        print(f"[retrain] trackio unavailable ({e}); console-only", flush=True)
    logger = MultiLogger(ConsoleLogger(every=5), trk)

    import flax.serialization
    ckpt_path = os.path.join(OUT, "retrain_e5_ckpt.msgpack")

    def _checkpoint(update_idx: int, params):
        """Periodic checkpoint so a long run survives a late crash; always overwrites the latest."""
        if CHECKPOINT_EVERY and update_idx > 0 and update_idx % CHECKPOINT_EVERY == 0:
            with open(ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(params))
            print(f"[retrain] checkpoint @ update {update_idx} -> {ckpt_path}", flush=True)

    init_params = None
    if RESUME:                                 # resume a crashed run: load into a shape-matched target
        net0 = ActorCritic(action_dim=NUM_ACTIONS, d_model=D_MODEL)
        import jax.numpy as jnp
        from .spec import dummy_obs
        target = net0.init(jax.random.PRNGKey(0), {k: jnp.asarray(v) for k, v in dummy_obs(1).items()},
                           jnp.ones((1, NUM_ACTIONS), bool))
        with open(RESUME, "rb") as f:
            init_params = flax.serialization.from_bytes(target, f.read())
        print(f"[retrain] resuming (warm-start) from {RESUME}", flush=True)

    t0 = time.time()
    print(f"[retrain] start: {cfg.num_updates} updates, d_model={D_MODEL}, num_envs={NUM_ENVS}, "
          f"full acquisition game + boss curriculum (floor {cfg.curr_floor}), "
          f"checkpoint every {CHECKPOINT_EVERY or 'off'}", flush=True)
    result = train(cfg, logger=logger, init_params=init_params, on_update=_checkpoint)
    print(f"[retrain] training done in {(time.time() - t0) / 60:.1f} min", flush=True)

    ppath = os.path.join(OUT, "retrain_e5_params.msgpack")
    with open(ppath, "wb") as f:
        f.write(flax.serialization.to_bytes(result.params))
    print(f"[retrain] saved params -> {ppath}", flush=True)

    # Record replays on the REAL deploy game (full bosses via the env's default boss_rate=1.0).
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=D_MODEL)
    for s in (4, 7):
        steps = record_agent_episode(net, result.params, seed=s, reward_name="shaped",
                                     enable_bosses=True)
        path = os.path.join(OUT, f"retrain_e5_seed{s}.episode.json")
        save_episode(steps, path)
        print(f"[retrain] recorded {len(steps)} steps -> {path}", flush=True)
    print("[retrain] DONE", flush=True)


if __name__ == "__main__":
    main()
