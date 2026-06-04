"""Phase D retrain: re-train the card-aware agent on the WIDER game -- boss blinds + the
enhancement/consumable exposure -- with the closed-loop curriculum. Saves the trained params
and records replays (under the trained-on conditions) for the viewer.

    python -m balatro_rl.agent.retrain

Long CPU run (this Mac has no CUDA). Progress streams via ConsoleLogger (every 5 updates +
every eval) and, if available, a Trackio dashboard. Recommended config from the plan's
Phase D: d_model 128, ~800 updates, curriculum floor 0.2 -> 1.0, bosses on, enhance_rate 0.2,
1 starting Planet, shaped reward, decaying entropy 0.04 -> 0.01.
"""
from __future__ import annotations

import os

# Cap threads BEFORE jax/XLA import -> avoids the host-thread oversubscription that thrashed
# the machine during the earlier ante-8 recording.
os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")

import time

from .train import train, TrainConfig
from .metrics_logger import ConsoleLogger, MultiLogger
from .networks import ActorCritic
from ..envs.actions import NUM_ACTIONS
from ..viz.replay_data import record_agent_episode, save_episode

OUT = os.environ.get("BALATRO_EPISODE_DIR", "/tmp/sweep_out")
D_MODEL = 128
ENHANCE_RATE = 0.2
GRANT_PLANETS = 1


def _ent(u: int) -> float:
    """Entropy coefficient: decay 0.04 -> 0.01 over the run."""
    return max(0.01, 0.04 * (1.0 - u / 700.0))


def build_config() -> TrainConfig:
    return TrainConfig(
        num_updates=800, num_envs=32, num_steps=64, d_model=D_MODEL,
        num_minibatches=4, update_epochs=4, lr=3e-4,
        reward_name="shaped", seed=0, ent_coef=_ent,
        curr_floor=0.2, ramp_clear_rate=0.7, ramp_step=0.05, ramp_window=20,
        enable_bosses=True, enhance_rate=ENHANCE_RATE, grant_planets=GRANT_PLANETS,
        eval_interval=25, eval_seeds=(0, 1, 2, 3),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = build_config()
    trk = None
    try:
        from .metrics_logger import TrackioLogger
        trk = TrackioLogger(project="balatro-retrain-d", name="phaseD",
                            config={"d_model": D_MODEL, "updates": cfg.num_updates,
                                    "enable_bosses": True, "enhance_rate": ENHANCE_RATE,
                                    "grant_planets": GRANT_PLANETS, "curr_floor": cfg.curr_floor})
    except Exception as e:   # trackio optional; console logging still streams progress
        print(f"[retrain] trackio unavailable ({e}); console-only")
    logger = MultiLogger(ConsoleLogger(every=5), trk)

    t0 = time.time()
    print(f"[retrain] start: {cfg.num_updates} updates, d_model={D_MODEL}, "
          f"bosses on, enhance_rate={ENHANCE_RATE}, planets={GRANT_PLANETS}", flush=True)
    result = train(cfg, logger=logger)
    print(f"[retrain] training done in {(time.time() - t0) / 60:.1f} min", flush=True)

    import flax.serialization
    ppath = os.path.join(OUT, "retrain_d_params.msgpack")
    with open(ppath, "wb") as f:
        f.write(flax.serialization.to_bytes(result.params))
    print(f"[retrain] saved params -> {ppath}", flush=True)

    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=D_MODEL)
    for s in (4, 7):
        steps = record_agent_episode(net, result.params, seed=s, reward_name="shaped",
                                     enable_bosses=True, enhance_rate=ENHANCE_RATE,
                                     grant_planets=GRANT_PLANETS)
        path = os.path.join(OUT, f"retrain_d_seed{s}.episode.json")
        save_episode(steps, path)
        print(f"[retrain] recorded {len(steps)} steps -> {path}", flush=True)
    print("[retrain] DONE", flush=True)


if __name__ == "__main__":
    main()
