"""Training entry point. `run_training(cfg, logger)` trains with a metrics logger;
the CLI uses a live Trackio dashboard. View it with: `trackio show --project <project>`.
"""
from __future__ import annotations

import dataclasses

from .metrics_logger import NullLogger, TrackioLogger
from .train import TrainConfig, TrainResult, train


def run_training(cfg: TrainConfig, logger=None) -> TrainResult:
    if logger is None:
        logger = NullLogger()
    return train(cfg, logger=logger)


def main():
    cfg = TrainConfig(num_updates=50, num_envs=64, num_steps=128, eval_interval=5)
    logger = TrackioLogger(project="balatro-rl", name="ppo", config=dataclasses.asdict(cfg))
    result = run_training(cfg, logger=logger)
    print(f"done: {len(result.losses)} updates; "
          f"last eval = {result.eval_history[-1] if result.eval_history else 'n/a'}")
    print("view the dashboard with:  trackio show --project balatro-rl")


if __name__ == "__main__":
    main()
