"""CLI: train a short PPO run in-process, then record the greedy agent's episode
to JSON for the replay viewer. (No checkpoint format yet — train and record in
one process.)  Usage: python -m balatro_rl.viz.record [out.json]
"""
from __future__ import annotations

from ..agent.networks import ActorCritic
from ..agent.train import TrainConfig, train
from ..envs.actions import NUM_ACTIONS
from .replay_data import record_agent_episode, save_episode


def record_demo(out_path: str = "episode.json", train_updates: int = 20, seed: int = 0,
                num_envs: int = 16, num_steps: int = 64, d_model: int = 128,
                reward_name: str = "shaped") -> list[dict]:
    cfg = TrainConfig(num_updates=train_updates, num_envs=num_envs, num_steps=num_steps,
                      d_model=d_model, reward_name=reward_name, seed=seed)
    result = train(cfg)
    net = ActorCritic(action_dim=NUM_ACTIONS, d_model=d_model)
    steps = record_agent_episode(net, result.params, seed=seed, reward_name=reward_name)
    save_episode(steps, out_path)
    return steps


def main():
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "episode.json"
    steps = record_demo(out_path=out)
    print(f"recorded {len(steps)} steps -> {out}")
    print(f"view it:  python -m balatro_rl.viz.viewer   (then upload {out})")


if __name__ == "__main__":
    main()
